/* Policy-gradient family for discrete actions: REINFORCE, REINFORCE + baseline, A2C (with GAE), PPO-clip. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { softmax, sampleCat, entropy, argmax } = RL.util;

  const PG_ALGOS = {
    reinforce: 'REINFORCE (vanilla Monte Carlo PG)',
    reinforce_b: 'REINFORCE + learned baseline',
    a2c: 'Advantage Actor-Critic (A2C, n-step + GAE)',
    ppo: 'PPO-Clip (actor-critic, GAE)',
  };

  class PGAgent {
    constructor(cfg = {}) {
      this.cfg = Object.assign({
        algo: 'ppo', obsDim: 4, nActions: 2, hidden: [64, 64],
        lr: 3e-4, lrV: 1e-3, gamma: 0.99, lam: 0.95, entCoef: 0.0,
        nSteps: 32, rollout: 1024, epochs: 4, minibatch: 64, clip: 0.2, normAdv: true, seed: undefined,
      }, cfg);
      this.rng = new RL.Rng(this.cfg.seed);
      this.reset();
    }
    get usesCritic() { return this.cfg.algo !== 'reinforce'; }
    get horizon() { return this.cfg.algo === 'ppo' ? this.cfg.rollout : this.cfg.nSteps; }

    reset() {
      const c = this.cfg;
      this.pi = new RL.MLP([c.obsDim, ...c.hidden, c.nActions], { act: 'tanh', rng: this.rng, lastScale: 0.01 });
      this.V = new RL.MLP([c.obsDim, ...c.hidden, 1], { act: 'tanh', rng: this.rng });
      this.store = [];
      this.steps = 0; this.updates = 0;
      this.diag = { entropy: Math.log(c.nActions), kl: 0, clipFrac: 0, vLoss: 0 };
      this.lastProbs = new Float64Array(c.nActions).fill(1 / c.nActions);
    }

    probs(s) { return softmax(this.pi.forward(s)); }
    value(s) { return this.V.forward(s)[0]; }

    act(s, greedy = false) {
      const p = this.probs(s);
      this.lastProbs = p;
      const a = greedy ? argmax(p) : sampleCat(p, this.rng);
      this._logp = Math.log(Math.max(p[a], 1e-12));
      this._v = this.usesCritic ? this.value(s) : 0;
      return a;
    }

    /** Record a transition; may trigger a learning update. Returns diagnostics when an update happened. */
    observe(s, a, r, s2, done, truncated) {
      const c = this.cfg;
      this.steps++;
      let rr = r;
      const boundary = done || truncated;
      // Time-limit truncation is not a real terminal state: bootstrap through it (Pardo et al., 2018).
      if (truncated && this.usesCritic && c.algo !== 'reinforce_b') rr += c.gamma * this.value(s2);
      this.store.push({ s: Float64Array.from(s), a, r: rr, boundary, logp: this._logp, v: this._v });
      this.lastS2 = s2;
      const isMC = c.algo === 'reinforce' || c.algo === 'reinforce_b';
      if ((isMC && boundary) || (!isMC && this.store.length >= this.horizon)) return this.update();
      return null;
    }

    _advantages() {
      const c = this.cfg, st = this.store, T = st.length;
      const adv = new Float64Array(T), ret = new Float64Array(T);
      if (c.algo === 'reinforce' || c.algo === 'reinforce_b') {
        let G = 0;
        for (let t = T - 1; t >= 0; t--) {
          if (st[t].boundary) G = 0;
          G = st[t].r + c.gamma * G;
          ret[t] = G;
          adv[t] = c.algo === 'reinforce' ? G : G - st[t].v;
        }
        return { adv, ret };
      }
      // Generalized Advantage Estimation (Schulman et al., 2016)
      const lastV = st[T - 1].boundary ? 0 : this.value(this.lastS2);
      let gae = 0;
      for (let t = T - 1; t >= 0; t--) {
        const nonterm = st[t].boundary ? 0 : 1;
        const nextV = t === T - 1 ? lastV : st[t + 1].v;
        const delta = st[t].r + c.gamma * nextV * nonterm - st[t].v;
        gae = delta + c.gamma * c.lam * nonterm * gae;
        adv[t] = gae; ret[t] = gae + st[t].v;
      }
      return { adv, ret };
    }

    update() {
      const c = this.cfg, st = this.store, T = st.length;
      const { adv, ret } = this._advantages();
      if (c.algo === 'ppo' && c.normAdv && T > 1) {
        let m = 0; for (const x of adv) m += x; m /= T;
        let v = 0; for (const x of adv) v += (x - m) * (x - m); const sd = Math.sqrt(v / T) + 1e-8;
        for (let i = 0; i < T; i++) adv[i] = (adv[i] - m) / sd;
      }
      const all = Array.from({ length: T }, (_, i) => i);
      let ent = 0, kl = 0, clipped = 0, vLoss = 0, count = 0, vCount = 0;

      const policyStep = (idx, ppo) => {
        for (const i of idx) {
          const tr = st[i];
          const p = softmax(this.pi.forward(tr.s));
          const H = entropy(p);
          let coef = adv[i];
          if (ppo) {
            const logp = Math.log(Math.max(p[tr.a], 1e-12));
            const lr = logp - tr.logp, ratio = Math.exp(lr);
            const active = adv[i] >= 0 ? ratio <= 1 + c.clip : ratio >= 1 - c.clip;
            coef = active ? ratio * adv[i] : 0;
            if (!active) clipped++;
            kl += ratio - 1 - lr; // unbiased, non-negative KL(old||new) estimator (Schulman's k3)
          }
          const d = new Float64Array(c.nActions);
          for (let j = 0; j < c.nActions; j++) {
            d[j] = -coef * ((j === tr.a ? 1 : 0) - p[j]);
            if (c.entCoef) d[j] += c.entCoef * p[j] * (Math.log(Math.max(p[j], 1e-12)) + H);
          }
          this.pi.backward(d);
          ent += H; count++;
        }
        this.pi.step(c.lr, { scale: 1 / idx.length, clipNorm: ppo || c.algo === 'a2c' ? 0.5 : 0 });
      };
      const valueStep = (idx) => {
        for (const i of idx) {
          const v = this.V.forward(st[i].s)[0];
          const e = v - ret[i];
          vLoss += e * e; vCount++;
          this.V.backward([e]);
        }
        this.V.step(c.lrV, { scale: 1 / idx.length, clipNorm: 0 });
      };

      if (c.algo === 'ppo') {
        for (let ep = 0; ep < c.epochs; ep++) {
          // shuffle
          for (let i = T - 1; i > 0; i--) { const j = this.rng.int(i + 1); const t = all[i]; all[i] = all[j]; all[j] = t; }
          for (let b = 0; b < T; b += c.minibatch) {
            const idx = all.slice(b, b + c.minibatch);
            policyStep(idx, true);
            valueStep(idx);
          }
        }
      } else {
        policyStep(all, false);
        if (this.usesCritic) {
          // A few critic steps per batch keep the baseline reasonably fresh.
          const vIters = c.algo === 'reinforce_b' ? 4 : 1;
          for (let k = 0; k < vIters; k++) valueStep(all);
        }
      }
      this.updates++;
      this.diag = {
        entropy: ent / Math.max(1, count),
        kl: c.algo === 'ppo' ? kl / Math.max(1, count) : 0,
        clipFrac: c.algo === 'ppo' ? clipped / Math.max(1, count) : 0,
        vLoss: vCount ? vLoss / vCount : 0,
        batch: T,
      };
      this.store = [];
      return this.diag;
    }
  }

  RL.PGAgent = PGAgent;
  RL.PGAgent.ALGOS = PG_ALGOS;
})(window.RL);
