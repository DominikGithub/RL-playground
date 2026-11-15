/* Deep Q-Network (Mnih et al. 2015) with toggles for replay, target network and Double DQN. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { argmax } = RL.util;

  class ReplayBuffer {
    constructor(cap, dim) {
      this.cap = cap; this.dim = dim; this.size = 0; this.ptr = 0;
      this.s = new Float64Array(cap * dim); this.s2 = new Float64Array(cap * dim);
      this.a = new Int32Array(cap); this.r = new Float64Array(cap); this.d = new Uint8Array(cap);
    }
    push(s, a, r, s2, done) {
      const i = this.ptr;
      this.s.set(s, i * this.dim); this.s2.set(s2, i * this.dim);
      this.a[i] = a; this.r[i] = r; this.d[i] = done ? 1 : 0;
      this.ptr = (this.ptr + 1) % this.cap; this.size = Math.min(this.size + 1, this.cap);
    }
    get(i) {
      const d = this.dim;
      return { s: this.s.subarray(i * d, i * d + d), a: this.a[i], r: this.r[i], s2: this.s2.subarray(i * d, i * d + d), done: this.d[i] === 1 };
    }
    latest() { return this.get((this.ptr - 1 + this.cap) % this.cap); }
  }

  class DQNAgent {
    constructor(cfg = {}) {
      this.cfg = Object.assign({
        obsDim: 4, nActions: 2, hidden: [64], lr: 1e-3, gamma: 0.99, batch: 32,
        buffer: 50000, warmup: 1000, targetEvery: 500, epsStart: 1, epsEnd: 0.05, epsSteps: 10000,
        useReplay: true, useTarget: true, double: true, trainEvery: 1, loss: 'mse', tau: 0.005, seed: undefined,
      }, cfg);
      this.rng = new RL.Rng(this.cfg.seed);
      this.reset();
    }
    reset() {
      const c = this.cfg;
      this.q = new RL.MLP([c.obsDim, ...c.hidden, c.nActions], { act: 'relu', rng: this.rng });
      this.target = this.q.clone();
      this.buf = new ReplayBuffer(Math.round(c.buffer), c.obsDim);
      this.steps = 0; this.updates = 0; this.lastLoss = 0; this.lastQ = 0;
    }
    epsilon() {
      const c = this.cfg;
      return Math.max(c.epsEnd, c.epsStart + (c.epsEnd - c.epsStart) * Math.min(1, this.steps / c.epsSteps));
    }
    qValues(s) { return this.q.forward(s); }
    act(s, greedy = false) {
      const qs = this.q.forward(s);
      this.lastQ = Math.max(...qs);
      if (!greedy && this.rng.next() < this.epsilon()) return this.rng.int(this.cfg.nActions);
      return argmax(qs);
    }
    /** Store a transition and learn. `terminal` must be true only for real terminations (not time-limit truncation). */
    observe(s, a, r, s2, terminal) {
      const c = this.cfg;
      this.buf.push(s, a, r, s2, terminal);
      this.steps++;
      const warm = c.useReplay ? c.warmup : 0;
      if (this.steps >= warm && this.steps % c.trainEvery === 0) this.train();
      if (c.useTarget) {
        if (c.tau > 0) this.softUpdate(c.tau);
        else if (this.steps % c.targetEvery === 0) this.target.copyFrom(this.q);
      }
    }
    /** Polyak averaging: θ⁻ ← τ θ + (1 − τ) θ⁻ */
    softUpdate(tau) {
      for (let l = 0; l < this.q.L; l++) {
        const W = this.q.W[l], Wt = this.target.W[l], b = this.q.b[l], bt = this.target.b[l];
        for (let i = 0; i < W.length; i++) Wt[i] += tau * (W[i] - Wt[i]);
        for (let i = 0; i < b.length; i++) bt[i] += tau * (b[i] - bt[i]);
      }
    }
    train() {
      const c = this.cfg;
      const B = c.useReplay ? Math.min(c.batch, this.buf.size) : 1;
      const tnet = c.useTarget ? this.target : this.q;
      let loss = 0;
      const batch = [];
      for (let i = 0; i < B; i++) batch.push(c.useReplay ? this.buf.get(this.rng.int(this.buf.size)) : this.buf.latest());
      // Compute all targets first (forward() caches activations, so do it before the trainable pass).
      const ys = batch.map((tr) => {
        if (tr.done) return tr.r;
        if (c.double) {
          const aStar = argmax(this.q.forward(tr.s2));
          return tr.r + c.gamma * tnet.forward(tr.s2)[aStar];
        }
        return tr.r + c.gamma * Math.max(...tnet.forward(tr.s2));
      });
      for (let i = 0; i < B; i++) {
        const tr = batch[i];
        const qs = this.q.forward(tr.s);
        const td = qs[tr.a] - ys[i];
        const d = new Float64Array(c.nActions);
        if (c.loss === 'mse') { loss += 0.5 * td * td; d[tr.a] = td; }
        else {
          // Huber loss: quadratic near 0, linear beyond |td| > 1 (gradient is clipped to ±1)
          loss += Math.abs(td) <= 1 ? 0.5 * td * td : Math.abs(td) - 0.5;
          d[tr.a] = Math.max(-1, Math.min(1, td));
        }
        this.q.backward(d);
      }
      this.q.step(c.lr, { scale: 1 / B, clipNorm: 10 });
      this.updates++;
      this.lastLoss = loss / B;
    }
  }

  RL.ReplayBuffer = ReplayBuffer;
  RL.DQNAgent = DQNAgent;
})(window.RL);
