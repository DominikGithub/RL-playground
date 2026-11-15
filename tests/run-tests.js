// Headless tests for the algorithm code.
//   node tests/run-tests.js quick        -> fast correctness tests only
//   node tests/run-tests.js              -> also trains agents on CartPole (takes a few minutes)
//   node tests/run-tests.js dqn ppo      -> only selected CartPole trainings (dqn, ppo, a2c, rb, r)
globalThis.window = globalThis;
const path = require('path');
const load = (f) => require(path.join(__dirname, '..', 'js', f));
['core/util.js', 'core/nn.js', 'envs/gridworld.js', 'envs/cartpole.js', 'algos/dp.js', 'algos/tabular.js',
  'algos/bandit.js', 'algos/dqn.js', 'algos/pg.js', 'algos/mcts.js'].forEach(load);
const RL = window.RL;
let fails = 0;
const check = (name, cond, extra = '') => { console.log((cond ? 'PASS ' : 'FAIL ') + name + (extra ? '  ' + extra : '')); if (!cond) fails++; };
const args = process.argv.slice(2);
const quick = args.includes('quick');
const which = args.filter((a) => a !== 'quick');
const runQuick = quick || which.length === 0;

if (runQuick) {
  // 1. Value iteration on Russell & Norvig 4x3 reproduces textbook utilities (gamma = 1)
  {
    const env = RL.GridWorld.preset('classic');
    const run = new RL.dp.DPRunner(env, 'vi', 1, 1e-10);
    while (!run.done) run.step();
    const V = (c, r) => run.V[env.idx(c - 1, 3 - r)]; // R&N coords: (col,row) from bottom-left
    const exp = [[1, 1, 0.705], [1, 2, 0.762], [1, 3, 0.812], [2, 3, 0.868], [3, 3, 0.918], [3, 2, 0.660], [3, 1, 0.611], [2, 1, 0.655], [4, 1, 0.388]];
    const errs = exp.map(([c, r, v]) => [c, r, V(c, r).toFixed(3), v]);
    const err = Math.max(...exp.map(([c, r, v]) => Math.abs(V(c, r) - v)));
    check('VI matches R&N 4x3 utilities', err < 0.002, 'max err ' + err.toFixed(4) + ' ' + JSON.stringify(errs));
    const pi = new RL.dp.DPRunner(env, 'pi', 1, 1e-10);
    let g = 0; while (!pi.done && g++ < 100000) pi.step();
    const err2 = Math.max(...Array.from(run.V, (v, i) => Math.abs(v - pi.V[i])));
    check('Policy iteration agrees with VI', err2 < 1e-4 && pi.policyStable, 'err ' + err2.toExponential(2) + ', improvements ' + pi.improvements);
  }
  // 2. Presets: goal reachable from start (BFS)
  for (const key of Object.keys(RL.GridWorld.PRESETS)) {
    const env = RL.GridWorld.preset(key);
    const seen = new Set([env.start]); const q = [env.start]; let ok = false;
    while (q.length) {
      const s = q.shift();
      if (env.cells[s] === 2) { ok = true; break; }
      if (!env.isActive(s)) continue;
      for (let a = 0; a < 4; a++) { const n = env.moveDet(s, a); if (!seen.has(n)) { seen.add(n); q.push(n); } }
    }
    check('preset ' + key + ' goal reachable', ok);
  }
  // 3. Cliff walking
  {
    const mk = () => RL.GridWorld.preset('cliff');
    const res = RL.compareTabular(mk, ['qlearning', 'sarsa', 'expsarsa', 'mc', 'dynaq', 'doubleq'], { alpha: 0.5, gamma: 1, epsilon: 0.1, maxSteps: 500, planning: 5 }, 500, 10);
    const tail = (a) => RL.util.mean(a.slice(-100));
    console.log('   cliff online returns (last 100 eps):', Object.fromEntries(Object.entries(res).map(([k, v]) => [k, tail(v).toFixed(1)])));
    check('SARSA online return > Q-learning on cliff', tail(res.sarsa) > tail(res.qlearning));
    const ag = new RL.TabularAgent(mk(), { algo: 'qlearning', alpha: 0.5, gamma: 1, epsilon: 0.1, seed: 3 });
    ag.runEpisodes(500); ag.eps = 0; ag.beginEpisode();
    let G = 0, t = 0;
    for (;;) { const i = ag.step(); G += i.r; t++; if (i.episodeEnd || t > 100) break; }
    check('Q-learning greedy return on cliff = -13', G === -13, 'G=' + G);
  }
  // 4. Bandits
  {
    const r = RL.bandit.runExperiment(['greedy', 'egreedy', 'ucb', 'thompson', 'gradient', 'optimistic'], { T: 1000, runs: 200, params: { epsilon: 0.1, c: 2, alpha: 0.1, baseline: true } });
    console.log('   bandit final regret:', Object.fromEntries(Object.entries(r).map(([k, v]) => [k, v.regret[999].toFixed(1)])));
    console.log('   bandit final %optimal:', Object.fromEntries(Object.entries(r).map(([k, v]) => [k, v.optimal[999].toFixed(0)])));
    check('Thompson & UCB regret < greedy & eps-greedy', r.thompson.regret[999] < r.egreedy.regret[999] && r.ucb.regret[999] < r.greedy.regret[999]);
  }
  // 5. MCTS
  {
    const st = RL.TTT.empty(); [0, 3, 1, 4].forEach((m) => RL.TTT.play(st, m));
    const r = RL.mcts(st, { sims: 500, rng: new RL.Rng(1) });
    check('MCTS finds immediate win', r.best === 2, 'best=' + r.best);
    const st2 = RL.TTT.empty(); [0, 3, 1].forEach((m) => RL.TTT.play(st2, m));
    const r2 = RL.mcts(st2, { sims: 1000, rng: new RL.Rng(2) });
    check('MCTS blocks opponent win', r2.best === 2, 'best=' + r2.best);
    check('minimax: empty board is a draw', RL.ttt.minimax(RL.TTT.empty()) === 0);
  }
  // 6. NN gradient check
  {
    for (const act of ['tanh', 'relu']) {
      const net = new RL.MLP([3, 5, 4, 2], { act, rng: new RL.Rng(5) });
      const x = [0.3, -0.2, 0.9];
      const f = () => { const y = net.forward(x); return 0.5 * y[0] * y[0] + 2 * y[1]; };
      net.zeroGrad(); const y = net.forward(x); net.backward([y[0], 2]);
      let maxRel = 0;
      for (const l of [0, 1, 2]) for (const i of [0, 3, 7]) {
        const old = net.W[l][i];
        net.W[l][i] = old + 1e-6; const fp = f(); net.W[l][i] = old - 1e-6; const fm = f(); net.W[l][i] = old;
        const num = (fp - fm) / 2e-6, an = net.gW[l][i];
        maxRel = Math.max(maxRel, Math.abs(num - an) / (Math.abs(num) + Math.abs(an) + 1e-9));
      }
      check('MLP backprop (' + act + ') matches finite differences', maxRel < 1e-4, 'rel ' + maxRel.toExponential(2));
    }
  }
}

// 7. CartPole learning (slow)
function trainCartPole(agent, steps, kind) {
  const env = new RL.CartPole(new RL.Rng(agent.cfg.seed + 1));
  let s = env.reset(); let ep = 0; const rets = [];
  const t0 = Date.now();
  for (let t = 0; t < steps; t++) {
    const a = agent.act(s); const res = env.step(a);
    if (kind === 'dqn') agent.observe(s, a, 1, res.obs, res.done);
    else agent.observe(s, a, 1, res.obs, res.done, res.truncated);
    ep++; s = res.obs;
    if (res.done || res.truncated) { rets.push(ep); ep = 0; s = env.reset(); }
  }
  return { rets, ms: Date.now() - t0 };
}
if (!quick) {
  const report = (name, { rets, ms }) => {
    const last = RL.util.mean(rets.slice(-20));
    console.log(`   ${name}: episodes=${rets.length}, mean last 20=${last.toFixed(1)}, best=${Math.max(...rets)}, ${ms}ms`);
    return last;
  };
  const want = (k) => which.length === 0 || which.includes(k);
  if (want('dqn')) for (const seed of [1, 2]) check('DQN learns CartPole (seed ' + seed + ')', report('DQN', trainCartPole(new RL.DQNAgent({ seed }), 40000, 'dqn')) > 150);
  if (want('ppo')) for (const seed of [1, 2]) check('PPO learns CartPole (seed ' + seed + ')', report('PPO', trainCartPole(new RL.PGAgent({ algo: 'ppo', seed }), 60000)) > 300);
  if (want('a2c')) for (const seed of [1, 2]) check('A2C learns CartPole (seed ' + seed + ')', report('A2C', trainCartPole(new RL.PGAgent({ algo: 'a2c', lr: 1e-3, seed }), 60000)) > 150);
  if (want('rb')) for (const seed of [1, 2]) check('REINFORCE+baseline learns (seed ' + seed + ')', report('REINFORCE+b', trainCartPole(new RL.PGAgent({ algo: 'reinforce_b', lr: 1e-3, seed }), 60000)) > 150);
  if (want('r')) for (const seed of [1, 2]) check('REINFORCE improves (seed ' + seed + ')', report('REINFORCE', trainCartPole(new RL.PGAgent({ algo: 'reinforce', lr: 1e-3, seed }), 60000)) > 60);
}
console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
