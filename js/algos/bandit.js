/* k-armed Gaussian bandits and classic exploration strategies. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { argmaxRandom, softmax, sampleCat } = RL.util;

  class Bandit {
    constructor(k, rng, { spread = 1, noise = 1 } = {}) {
      this.k = k; this.rng = rng; this.noise = noise;
      this.means = Array.from({ length: k }, () => rng.normal() * spread);
      this.best = this.means.indexOf(Math.max(...this.means));
      this.bestMean = this.means[this.best];
    }
    pull(a) { return this.means[a] + this.noise * this.rng.normal(); }
  }

  const STRATEGIES = {
    greedy: { label: 'Greedy', color: '--c5' },
    optimistic: { label: 'Optimistic greedy (Q₀ = +5)', color: '--c6' },
    egreedy: { label: 'ε-greedy', color: '--c1' },
    ucb: { label: 'UCB1', color: '--c2' },
    thompson: { label: 'Thompson sampling', color: '--c3' },
    gradient: { label: 'Gradient bandit (softmax)', color: '--c4' },
  };

  function makeAgent(type, k, p, rng) {
    const Q = new Float64Array(k), N = new Float64Array(k);
    switch (type) {
      case 'greedy':
      case 'optimistic':
      case 'egreedy': {
        const eps = type === 'egreedy' ? p.epsilon : 0;
        if (type === 'optimistic') Q.fill(5);
        return {
          Q, N,
          select() { return rng.next() < eps ? rng.int(k) : argmaxRandom(Q, rng); },
          // optimistic uses a constant step so the optimism wears off gradually (Sutton & Barto §2.6)
          update(a, r) { N[a]++; const step = type === 'optimistic' ? 0.1 : 1 / N[a]; Q[a] += step * (r - Q[a]); },
        };
      }
      case 'ucb': {
        let t = 0;
        return {
          Q, N,
          select() {
            t++;
            for (let a = 0; a < k; a++) if (N[a] === 0) return a;
            const u = new Float64Array(k);
            for (let a = 0; a < k; a++) u[a] = Q[a] + p.c * Math.sqrt(Math.log(t) / N[a]);
            return argmaxRandom(u, rng);
          },
          update(a, r) { N[a]++; Q[a] += (r - Q[a]) / N[a]; },
        };
      }
      case 'thompson': {
        // Gaussian prior N(0, 1) on each mean, known reward noise σ² = 1 → conjugate Gaussian posterior.
        const S = new Float64Array(k);
        return {
          Q, N,
          select() {
            const th = new Float64Array(k);
            for (let a = 0; a < k; a++) { const prec = 1 + N[a]; th[a] = S[a] / prec + rng.normal() / Math.sqrt(prec); }
            return argmaxRandom(th, rng);
          },
          update(a, r) { N[a]++; S[a] += r; Q[a] = S[a] / (1 + N[a]); },
        };
      }
      case 'gradient': {
        const H = new Float64Array(k); let avg = 0, t = 0;
        return {
          Q: H, N,
          select() { return sampleCat(softmax(H), rng); },
          update(a, r) {
            t++; N[a]++;
            const base = p.baseline ? avg : 0;
            const pi = softmax(H);
            for (let b = 0; b < k; b++) H[b] += p.alpha * (r - base) * ((b === a ? 1 : 0) - pi[b]);
            avg += (r - avg) / t;
          },
          probs() { return softmax(H); },
        };
      }
    }
    throw new Error('unknown strategy ' + type);
  }

  /** Run strategies on `runs` independent bandit problems for T steps. Same problems for every strategy. */
  function runExperiment(types, { k = 10, T = 1000, runs = 200, seed = 1, params = {}, spread = 1, noise = 1 }) {
    const res = {};
    for (const type of types) res[type] = { reward: new Float64Array(T), optimal: new Float64Array(T), regret: new Float64Array(T) };
    for (let run = 0; run < runs; run++) {
      for (const type of types) {
        const probRng = new RL.Rng(seed * 100003 + run * 7919);
        const bandit = new Bandit(k, probRng, { spread, noise });
        const agent = makeAgent(type, k, params, new RL.Rng(seed * 31 + run * 104729 + type.length * 13));
        let cum = 0;
        const R = res[type];
        for (let t = 0; t < T; t++) {
          const a = agent.select();
          const r = bandit.pull(a);
          agent.update(a, r);
          R.reward[t] += r;
          if (a === bandit.best) R.optimal[t] += 1;
          cum += bandit.bestMean - bandit.means[a];
          R.regret[t] += cum;
        }
      }
    }
    for (const type of types) for (const key of ['reward', 'optimal', 'regret']) {
      const arr = res[type][key]; for (let t = 0; t < T; t++) arr[t] /= runs;
      if (key === 'optimal') for (let t = 0; t < T; t++) arr[t] *= 100;
    }
    return res;
  }

  RL.Bandit = Bandit;
  RL.bandit = { STRATEGIES, makeAgent, runExperiment };
})(window.RL);
