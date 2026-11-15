/* Dynamic programming on a known model: policy evaluation, value iteration, policy iteration. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';

  const dp = {};

  /** Q(s,·) from a value function by one-step lookahead using the model. */
  dp.qFromV = (env, V, s, gamma) => {
    const q = new Float64Array(4);
    for (let a = 0; a < 4; a++) {
      let v = 0;
      for (const o of env.outcomes(s, a)) v += o.p * (o.r + (o.done ? 0 : gamma * V[o.s2]));
      q[a] = v;
    }
    return q;
  };

  /** One synchronous (Jacobi) sweep of the Bellman optimality backup. */
  dp.valueIterationSweep = (env, V, gamma) => {
    const nV = new Float64Array(V.length);
    let delta = 0;
    for (let s = 0; s < env.nS; s++) {
      if (!env.isActive(s)) continue;
      const q = dp.qFromV(env, V, s, gamma);
      nV[s] = Math.max(q[0], q[1], q[2], q[3]);
      delta = Math.max(delta, Math.abs(nV[s] - V[s]));
    }
    return { V: nV, delta };
  };

  /** One synchronous sweep of the Bellman expectation backup for a stochastic policy pi[s] = probs(4). */
  dp.policyEvalSweep = (env, V, pi, gamma) => {
    const nV = new Float64Array(V.length);
    let delta = 0;
    for (let s = 0; s < env.nS; s++) {
      if (!env.isActive(s)) continue;
      const q = dp.qFromV(env, V, s, gamma);
      let v = 0; for (let a = 0; a < 4; a++) v += pi[s][a] * q[a];
      nV[s] = v;
      delta = Math.max(delta, Math.abs(nV[s] - V[s]));
    }
    return { V: nV, delta };
  };

  /** Greedy policy w.r.t. V. Returns best-action sets and a stochastic policy uniform over ties. */
  dp.greedy = (env, V, gamma, tol = 1e-9) => {
    const best = [], pi = [];
    for (let s = 0; s < env.nS; s++) {
      if (!env.isActive(s)) { best.push([]); pi.push(new Float64Array(4).fill(0.25)); continue; }
      const q = dp.qFromV(env, V, s, gamma);
      const m = Math.max(q[0], q[1], q[2], q[3]);
      const b = []; for (let a = 0; a < 4; a++) if (q[a] >= m - tol) b.push(a);
      const p = new Float64Array(4); for (const a of b) p[a] = 1 / b.length;
      best.push(b); pi.push(p);
    }
    return { best, pi };
  };

  dp.uniformPolicy = (env) => Array.from({ length: env.nS }, () => new Float64Array(4).fill(0.25));

  /**
   * Stateful runner so the UI can animate one sweep at a time.
   * mode: 'eval' (random policy), 'vi' (value iteration), 'pi' (policy iteration).
   */
  class DPRunner {
    constructor(env, mode, gamma, theta = 1e-4) {
      this.env = env; this.mode = mode; this.gamma = gamma; this.theta = theta;
      this.reset();
    }
    reset() {
      this.V = new Float64Array(this.env.nS);
      this.sweeps = 0; this.deltas = []; this.done = false;
      this.pi = dp.uniformPolicy(this.env);
      this.phase = this.mode === 'pi' ? 'evaluate' : this.mode;
      this.improvements = 0; this.policyStable = false;
      this.lastEvent = '';
    }
    /** Perform one unit of work; returns a short description of what happened. */
    step() {
      if (this.done) return 'converged';
      const { env, gamma } = this;
      if (this.mode === 'vi') {
        const r = dp.valueIterationSweep(env, this.V, gamma);
        this.V = r.V; this.sweeps++; this.deltas.push(r.delta);
        if (r.delta < this.theta || this.sweeps >= 2000) this.done = true;
        return (this.lastEvent = `sweep ${this.sweeps}: max |ΔV| = ${RL.util.fmt(r.delta, 4)}`);
      }
      if (this.mode === 'eval') {
        const r = dp.policyEvalSweep(env, this.V, this.pi, gamma);
        this.V = r.V; this.sweeps++; this.deltas.push(r.delta);
        if (r.delta < this.theta || this.sweeps >= 2000) this.done = true;
        return (this.lastEvent = `sweep ${this.sweeps}: max |ΔV| = ${RL.util.fmt(r.delta, 4)}`);
      }
      // policy iteration
      if (this.phase === 'evaluate') {
        const r = dp.policyEvalSweep(env, this.V, this.pi, gamma);
        this.V = r.V; this.sweeps++; this.deltas.push(r.delta);
        if (r.delta < this.theta || this.sweeps >= 5000) this.phase = 'improve';
        return (this.lastEvent = `evaluate π${this.improvements}: sweep ${this.sweeps}, max |ΔV| = ${RL.util.fmt(r.delta, 4)}`);
      }
      // improvement
      const g = dp.greedy(env, this.V, gamma);
      let changed = 0;
      for (let s = 0; s < env.nS; s++) {
        if (!env.isActive(s)) continue;
        // Stable if the old policy's support is within the new greedy set (avoid tie flip-flop).
        const oldBest = []; for (let a = 0; a < 4; a++) if (this.pi[s][a] > 0) oldBest.push(a);
        const same = oldBest.every((a) => g.best[s].includes(a)) && oldBest.length > 0 && !(oldBest.length === 4 && g.best[s].length < 4);
        if (!same) changed++;
      }
      this.pi = g.pi; this.improvements++;
      if (changed === 0) { this.done = true; this.policyStable = true; return (this.lastEvent = `policy stable after ${this.improvements - 1} improvements → optimal`); }
      this.phase = 'evaluate';
      return (this.lastEvent = `improve: greedy policy π${this.improvements} changed in ${changed} states`);
    }
  }

  dp.DPRunner = DPRunner;
  RL.dp = dp;
})(window.RL);
