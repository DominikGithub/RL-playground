/* Tabular model-free control: Monte Carlo, SARSA, Expected SARSA, Q-learning, Double Q-learning, Dyna-Q. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { argmaxRandom } = RL.util;

  const ALGOS = {
    mc: 'Monte Carlo control (every-visit, constant α)',
    sarsa: 'SARSA (on-policy TD)',
    expsarsa: 'Expected SARSA',
    qlearning: 'Q-learning (off-policy TD)',
    doubleq: 'Double Q-learning',
    dynaq: 'Dyna-Q (Q-learning + planning)',
  };

  class TabularAgent {
    /**
     * cfg: {algo, alpha, gamma, epsilon, epsDecay, epsMin, planning, maxSteps, seed}
     */
    constructor(env, cfg) {
      this.env = env;
      this.cfg = Object.assign({ algo: 'qlearning', alpha: 0.5, gamma: 0.95, epsilon: 0.1, epsDecay: 1, epsMin: 0.01, planning: 10, maxSteps: 500, qInit: 0 }, cfg);
      this.rng = new RL.Rng(cfg.seed);
      this.reset();
    }

    reset() {
      const n = this.env.nS * 4;
      this.Q = new Float64Array(n).fill(this.cfg.qInit);
      this.Q2 = new Float64Array(n).fill(this.cfg.qInit); // second table for Double Q-learning
      this.visits = new Uint32Array(this.env.nS);
      this.model = new Map();
      this.modelKeys = [];
      this.episode = 0; this.totalSteps = 0;
      this.eps = this.cfg.epsilon;
      this.returns = []; this.lengths = [];
      this.beginEpisode();
    }

    /** Q-values used for acting/display (average of both tables for Double Q). */
    q(s, a) { return this.cfg.algo === 'doubleq' ? 0.5 * (this.Q[s * 4 + a] + this.Q2[s * 4 + a]) : this.Q[s * 4 + a]; }
    qRow(s) { return [0, 1, 2, 3].map((a) => this.q(s, a)); }

    greedy(s) { return argmaxRandom(this.qRow(s), this.rng); }
    choose(s) { return this.rng.next() < this.eps ? this.rng.int(4) : this.greedy(s); }

    /** ε-greedy action probabilities (greedy mass split among ties). */
    probs(s) {
      const row = this.qRow(s);
      const m = Math.max(...row);
      const ties = row.map((v) => (Math.abs(v - m) < 1e-12 ? 1 : 0));
      const nt = ties.reduce((a, b) => a + b, 0);
      return row.map((_, a) => this.eps / 4 + (ties[a] ? (1 - this.eps) / nt : 0));
    }

    beginEpisode() {
      this.s = this.env.start; this.a = this.choose(this.s);
      this.t = 0; this.G = 0; this.traj = [];
      this.trail = [this.s];
    }

    _qlUpdate(Q, s, a, r, s2, done) {
      const target = r + (done ? 0 : this.cfg.gamma * Math.max(Q[s2 * 4], Q[s2 * 4 + 1], Q[s2 * 4 + 2], Q[s2 * 4 + 3]));
      Q[s * 4 + a] += this.cfg.alpha * (target - Q[s * 4 + a]);
      return target;
    }

    /** One environment step (plus learning). Returns transition info; info.episodeEnd marks a finished episode. */
    step() {
      const { env, cfg } = this;
      const s = this.s;
      const a = cfg.algo === 'sarsa' ? this.a : this.choose(s);
      const res = env.step(s, a, this.rng);
      const { s2, r } = res;
      let done = res.done;
      this.t++; this.totalSteps++; this.G += r; this.visits[s]++;
      const truncated = !done && this.t >= cfg.maxSteps;
      let tdError = 0;

      switch (cfg.algo) {
        case 'qlearning': {
          const old = this.Q[s * 4 + a];
          const tgt = this._qlUpdate(this.Q, s, a, r, s2, done);
          tdError = tgt - old;
          break;
        }
        case 'doubleq': {
          // Randomly pick which table to update; select with one, evaluate with the other.
          const [A, B] = this.rng.next() < 0.5 ? [this.Q, this.Q2] : [this.Q2, this.Q];
          let tgt = r;
          if (!done) { const aStar = argmaxRandom(A, this.rng, s2 * 4, 4); tgt += cfg.gamma * B[s2 * 4 + aStar]; }
          tdError = tgt - A[s * 4 + a];
          A[s * 4 + a] += cfg.alpha * tdError;
          break;
        }
        case 'sarsa': {
          const a2 = this.choose(s2);
          const tgt = r + (done ? 0 : cfg.gamma * this.Q[s2 * 4 + a2]);
          tdError = tgt - this.Q[s * 4 + a];
          this.Q[s * 4 + a] += cfg.alpha * tdError;
          this.a = a2;
          break;
        }
        case 'expsarsa': {
          let ev = 0;
          if (!done) { const p = this.probs(s2); for (let b = 0; b < 4; b++) ev += p[b] * this.Q[s2 * 4 + b]; }
          const tgt = r + cfg.gamma * ev;
          tdError = tgt - this.Q[s * 4 + a];
          this.Q[s * 4 + a] += cfg.alpha * tdError;
          break;
        }
        case 'dynaq': {
          const old = this.Q[s * 4 + a];
          tdError = this._qlUpdate(this.Q, s, a, r, s2, done) - old;
          const key = s * 4 + a;
          if (!this.model.has(key)) this.modelKeys.push(key);
          this.model.set(key, { r, s2, done });
          for (let i = 0; i < cfg.planning; i++) {
            const k = this.modelKeys[this.rng.int(this.modelKeys.length)];
            const m = this.model.get(k);
            this._qlUpdate(this.Q, (k / 4) | 0, k % 4, m.r, m.s2, m.done);
          }
          break;
        }
        case 'mc': {
          this.traj.push([s, a, r]);
          break;
        }
      }

      const info = { s, a, r, s2, done, truncated, tdError, fell: !!res.fell, cliffCell: res.cliffCell };
      this.s = s2;
      this.trail.push(s2);
      if (this.trail.length > 400) this.trail.shift();

      if (done || truncated) {
        if (cfg.algo === 'mc') {
          // Every-visit constant-α Monte Carlo: move Q(s,a) toward the observed return G_t.
          let G = 0;
          for (let i = this.traj.length - 1; i >= 0; i--) {
            const [ss, aa, rr] = this.traj[i];
            G = rr + cfg.gamma * G;
            this.Q[ss * 4 + aa] += cfg.alpha * (G - this.Q[ss * 4 + aa]);
          }
        }
        info.episodeEnd = true; info.episodeReturn = this.G; info.episodeLength = this.t;
        this.returns.push(this.G); this.lengths.push(this.t);
        this.episode++;
        this.eps = Math.max(cfg.epsMin, this.eps * cfg.epsDecay);
        this.beginEpisode();
      }
      return info;
    }

    /** Run whole episodes headlessly. */
    runEpisodes(n) {
      const target = this.episode + n;
      let guard = 0;
      while (this.episode < target && guard++ < 5e6) this.step();
    }
  }

  /** Average learning curves over several independent runs. */
  function compareAlgos(envFactory, algos, cfg, episodes, runs) {
    const out = {};
    for (const algo of algos) {
      const sum = new Float64Array(episodes);
      for (let r = 0; r < runs; r++) {
        const ag = new TabularAgent(envFactory(), Object.assign({}, cfg, { algo, seed: 1000 + r * 7919 }));
        ag.runEpisodes(episodes);
        for (let i = 0; i < episodes; i++) sum[i] += ag.returns[i];
      }
      out[algo] = Array.from(sum, (x) => x / runs);
    }
    return out;
  }

  RL.TabularAgent = TabularAgent;
  RL.TabularAgent.ALGOS = ALGOS;
  RL.compareTabular = compareAlgos;
})(window.RL);
