/* Reusable CartPole training lab for DQN and policy-gradient agents, with live function landscapes. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const sw = RL.ui.sw, lg = RL.ui.lg;
  const { h, fmt, setupCanvas, cssVar, divColor, mean } = RL.util;

  const SPEEDS = { realtime: 1, '10×': 10, '100×': 100, max: Infinity };

  /** Sequential colour between --mid and --pos. */
  function seqColor(t) {
    const p = RL.util.palette();
    t = Math.max(0, Math.min(1, t));
    const c = [0, 1, 2].map((i) => Math.round(p.mid[i] + (p.pos[i] - p.mid[i]) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  /** Heatmap of f(θ, θ̇) at x = ẋ = 0. */
  class Landscape {
    constructor(parent, title, sub, info) {
      this.canvas = h('canvas');
      const t = h('div.chart-title', h('span', title));
      if (info) RL.ui.info(t, info, title);
      this.el = h('div.landscape', t, h('div.small.muted', { style: { fontSize: '11px', marginBottom: '3px' } }, sub), this.canvas);
      parent.appendChild(this.el);
      this.n = 32;
    }
    draw(values, { diverging = false, marker = null, range = null, emptyText = '' } = {}) {
      const W = Math.max(140, Math.min(260, this.el.clientWidth || 220)), H = Math.round(W * 0.78);
      const ctx = setupCanvas(this.canvas, W, H);
      const padL = 26, padB = 18, pw = W - padL - 2, ph = H - padB - 2;
      const n = this.n;
      let lo = Infinity, hi = -Infinity;
      for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
      if (range) { lo = range[0]; hi = range[1]; }
      const cw = pw / n, chh = ph / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = values[j * n + i];
        ctx.fillStyle = diverging ? divColor(v - 0.5, 0.5) : seqColor((v - lo) / (hi - lo || 1));
        ctx.fillRect(padL + i * cw, 2 + (n - 1 - j) * chh, Math.ceil(cw), Math.ceil(chh));
      }
      ctx.fillStyle = cssVar('--muted'); ctx.font = '10px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('pole angle θ  (−12° … +12°)', padL + pw / 2, ph + 5);
      ctx.save(); ctx.translate(9, 2 + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText('ang. velocity θ̇', 0, 0); ctx.restore();
      ctx.strokeStyle = cssVar('--text'); ctx.globalAlpha = 0.25; ctx.beginPath();
      ctx.moveTo(padL + pw / 2, 2); ctx.lineTo(padL + pw / 2, 2 + ph); ctx.moveTo(padL, 2 + ph / 2); ctx.lineTo(padL + pw, 2 + ph / 2); ctx.stroke(); ctx.globalAlpha = 1;
      if (marker) {
        const [tx, ty] = marker;
        const mx = padL + ((tx + 1) / 2) * pw, my = 2 + (1 - (ty + 1) / 2) * ph;
        if (mx >= padL && mx <= padL + pw && my >= 2 && my <= 2 + ph) {
          ctx.strokeStyle = cssVar('--text'); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(mx, my, 4, 0, Math.PI * 2); ctx.stroke();
        }
      }
      if (emptyText) { ctx.fillStyle = cssVar('--panel'); ctx.fillRect(padL, 2, pw, ph); ctx.fillStyle = cssVar('--muted'); ctx.font = '12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(emptyText, padL + pw / 2, 2 + ph / 2); return; }
      if (!diverging && Number.isFinite(lo)) { ctx.fillStyle = cssVar('--muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(`${fmt(lo, 0)} … ${fmt(hi, 0)}`, W - 4, 4); }
    }
  }

  /**
   * Build a CartPole lab.
   * opts.kind: 'dqn' | 'pg';  opts.algos: list of PG algo keys;  opts.defaults: cfg overrides.
   */
  RL.cartpoleLab = function (root, opts) {
    const kind = opts.kind;
    const isPG = kind === 'pg';
    const cfg = Object.assign(isPG
      ? { algo: opts.algos[0], lr: 3e-4, lrV: 1e-3, gamma: 0.99, lam: 0.95, entCoef: 0, nSteps: 32, rollout: 1024, epochs: 4, minibatch: 64, clip: 0.2, normAdv: true, width: 64 }
      : { lr: 1e-3, gamma: 0.99, batch: 32, buffer: 50000, warmup: 1000, tau: 0.005, epsStart: 1, epsEnd: 0.05, epsSteps: 10000, useReplay: true, useTarget: true, double: true, loss: 'mse', width: 64 },
    opts.defaults || {});

    let agent, env, obs, epRet, training = true, speed = '10×', watchRet = 0;
    let returns = [], frame = 0, diagAcc = { loss: [], q: [] };

    const lab = RL.page.lab(root, { title: opts.title || 'CartPole lab' });
    const C = lab.controls;
    C.appendChild(h('div.lab-title', 'Training'));
    const tr = RL.ui.transport(C, {
      onPlay: (p) => { if (p) loop.start(); else { loop.stop(); drawAll(true); } },
      onReset: () => { reset(); },
      playLabel: 'Train',
    });
    const trainT = RL.ui.toggle(C, {
      label: 'Learning on (off = watch greedy policy)', value: true,
      onChange: (v) => { training = v; if (isPG) agent.store = []; obs = env.reset(); epRet = 0; watchRet = 0; },
    });
    const speedSeg = RL.ui.segmented(C, { label: 'Simulation speed', value: speed, options: Object.keys(SPEEDS).map((k) => ({ value: k, label: k })), onChange: (v) => { speed = v; } });
    C.appendChild(h('div', { style: { height: '10px' } }));

    const ctl = {};
    const live = (key) => (v) => { cfg[key] = v; agent.cfg[key] = v; };
    const structural = (key) => (v) => { cfg[key] = v; reset(); };
    const lr = (label, key, help) => RL.ui.slider(g, { label, min: 1e-5, max: 1e-2, log: true, value: cfg[key], fmt: (v) => v.toExponential(1), help, onInput: live(key) });
    let g;
    if (isPG) {
      g = RL.ui.group(C, 'Algorithm');
      ctl.algo = RL.ui.select(g, { label: 'Method', value: cfg.algo, options: opts.algos.map((k) => ({ value: k, label: RL.PGAgent.ALGOS[k] })), onChange: (v) => { cfg.algo = v; showHide(); reset(); } });
      ctl.lr = lr('Actor learning rate', 'lr', 'Adam step size of the policy network. Too large → policy collapse (watch the return crash and entropy drop).');
      ctl.lrV = lr('Critic learning rate', 'lrV', 'Adam step size of the value network V(s) used as baseline / critic.');
      ctl.gamma = RL.ui.slider(g, { label: 'Discount γ', min: 0.9, max: 0.999, step: 0.001, value: cfg.gamma, fmt: (v) => v.toFixed(3), onInput: live('gamma') });
      ctl.lam = RL.ui.slider(g, { label: 'GAE λ', min: 0, max: 1, step: 0.01, value: cfg.lam, help: 'λ = 0: one-step TD advantage (low variance, biased). λ = 1: Monte Carlo advantage (unbiased, high variance).', onInput: live('lam') });
      ctl.entCoef = RL.ui.slider(g, { label: 'Entropy bonus', min: 0, max: 0.05, step: 0.001, value: cfg.entCoef, fmt: (v) => v.toFixed(3), help: 'Adds c·H(π) to the objective to keep the policy stochastic (exploration).', onInput: live('entCoef') });
      ctl.nSteps = RL.ui.slider(g, { label: 'Rollout length n (A2C)', min: 4, max: 256, step: 4, value: cfg.nSteps, help: 'Steps collected before each A2C update.', onInput: live('nSteps') });
      ctl.rollout = RL.ui.slider(g, { label: 'Rollout length T (PPO)', min: 128, max: 4096, step: 128, value: cfg.rollout, onInput: live('rollout') });
      ctl.epochs = RL.ui.slider(g, { label: 'Epochs K per batch', min: 1, max: 30, step: 1, value: cfg.epochs, help: 'How many passes PPO makes over each batch. More = more sample-efficient, but the policy drifts further from the one that collected the data.', onInput: live('epochs') });
      ctl.minibatch = RL.ui.slider(g, { label: 'Minibatch size', min: 16, max: 1024, step: 16, value: cfg.minibatch, onInput: live('minibatch') });
      ctl.clip = RL.ui.slider(g, { label: 'Clip ε', min: 0.01, max: 1, step: 0.01, value: cfg.clip, help: 'PPO only lets the probability ratio π_new/π_old move inside [1−ε, 1+ε] before the gradient is cut off.', onInput: live('clip') });
      ctl.normAdv = RL.ui.toggle(g, { label: 'Normalize advantages (PPO)', value: cfg.normAdv, onChange: live('normAdv') });
    } else {
      g = RL.ui.group(C, 'Q-learning');
      ctl.lr = lr('Learning rate', 'lr', 'Adam step size of the Q-network.');
      ctl.gamma = RL.ui.slider(g, { label: 'Discount γ', min: 0.9, max: 0.999, step: 0.001, value: cfg.gamma, fmt: (v) => v.toFixed(3), onInput: live('gamma') });
      ctl.loss = RL.ui.select(g, { label: 'Loss', value: cfg.loss, options: [{ value: 'mse', label: 'Squared TD error (MSE)' }, { value: 'huber', label: 'Huber (clipped gradient, Nature DQN)' }], onChange: live('loss') });
      ctl.double = RL.ui.toggle(g, { label: 'Double DQN target', value: cfg.double, help: 'Select the next action with the online net, evaluate it with the target net — removes maximization bias.', onChange: live('double') });
      g = RL.ui.group(C, 'Stabilizers');
      ctl.useReplay = RL.ui.toggle(g, { label: 'Experience replay', value: cfg.useReplay, help: 'Off: train on the latest transition only (online, highly correlated data).', onChange: (v) => { live('useReplay')(v); } });
      ctl.buffer = RL.ui.slider(g, { label: 'Replay capacity', min: 500, max: 100000, log: true, value: cfg.buffer, fmt: RL.util.compact, help: 'Changing this restarts training.', onInput: structural('buffer') });
      ctl.batch = RL.ui.slider(g, { label: 'Batch size', min: 1, max: 128, step: 1, value: cfg.batch, onInput: live('batch') });
      ctl.useTarget = RL.ui.toggle(g, { label: 'Target network', value: cfg.useTarget, help: 'Off: the bootstrap target r + γ max Q(s′) uses the network being trained — a moving target.', onChange: (v) => { live('useTarget')(v); if (v) agent.target.copyFrom(agent.q); } });
      ctl.tau = RL.ui.slider(g, { label: 'Target update rate τ', min: 0.0005, max: 1, log: true, value: cfg.tau, fmt: (v) => v.toFixed(4), help: 'Polyak averaging θ⁻ ← τθ + (1−τ)θ⁻ every step. τ = 1 means no lag at all.', onInput: live('tau') });
      g = RL.ui.group(C, 'Exploration');
      ctl.epsSteps = RL.ui.slider(g, { label: 'ε decay steps', min: 1000, max: 50000, step: 1000, value: cfg.epsSteps, fmt: RL.util.compact, help: 'ε decays linearly from 1 to its final value over this many environment steps.', onInput: live('epsSteps') });
      ctl.epsEnd = RL.ui.slider(g, { label: 'Final ε', min: 0, max: 0.5, step: 0.01, value: cfg.epsEnd, onInput: live('epsEnd') });
    }
    g = RL.ui.group(C, 'Network');
    ctl.width = RL.ui.slider(g, { label: isPG ? 'Hidden units (2 layers)' : 'Hidden units (1 layer)', min: 8, max: 128, step: 8, value: cfg.width, help: 'Changing this restarts training.', onInput: structural('width') });

    function showHide() {
      if (!isPG) return;
      const a = cfg.algo;
      ctl.lrV.show(a !== 'reinforce');
      ctl.lam.show(a === 'a2c' || a === 'ppo');
      ctl.nSteps.show(a === 'a2c');
      for (const k of ['rollout', 'epochs', 'minibatch', 'clip', 'normAdv']) ctl[k].show(a === 'ppo');
    }

    // viz
    const cpCanvas = h('canvas'); const cpHolder = h('div', cpCanvas);
    RL.ui.info(lab.viz.appendChild(h('div.lab-title', h('span', 'Environment'), h('span.muted', { style: { textTransform: 'none', letterSpacing: 0, fontWeight: 400 } }, 'reward +1 per step · fail at |θ| > 12° or |x| > 2.4 · max 500'))), lg([
      [sw.box('var(--text)'), '<b>Cart</b> on its track. The small arrow shows the last push (always the same force, left or right).'],
      [sw.line('var(--accent)'), '<b>Pole</b>: teal while safe, orange when it is close to the 12° failure angle.'],
      [sw.dashed('var(--neg)'), 'Orange dashed lines: track limits x = ±2.4. Leaving them also ends the episode.'],
      [sw.box('var(--pos)'), 'Bars at the bottom left (policy-gradient methods only): action probabilities π(left | s) and π(right | s) in the current state.'],
      [sw.text('t'), 't = steps survived in this episode; each step earns reward +1, the episode is cut at 500.'],
      [sw.seq(), 'The two maps below are slices of the learned network over all pole angles and angular velocities; their ⓘ explains the colours.'],
    ]), 'CartPole');
    lab.viz.appendChild(cpHolder);
    const status = h('div.lab-status'); lab.viz.appendChild(status);
    const lsRow = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '14px' } });
    lab.viz.appendChild(lsRow);
    const AXES = 'Each pixel is one state with the cart centred and at rest: pole angle θ on the x-axis (−12° … +12°), angular velocity θ̇ on the y-axis. Thin cross = θ = 0 and θ̇ = 0. Black ring = the current state of the cart-pole.';
    const ls1 = new Landscape(lsRow, isPG ? 'Policy: P(push right | s)' : 'Value: max_a Q(s, a)', isPG ? 'orange = left, blue = right (cart at x = 0, at rest)' : 'cart at x = 0 and at rest',
      lg(isPG ? [[sw.grad(), '<b>Probability of pushing right</b>: orange = left is more likely, blue = right is more likely, pale ≈ 50/50 (the policy is still unsure).'], [sw.ring('var(--text)'), AXES]]
        : [[sw.seq(), '<b>max<sub>a</sub> Q(s,a)</b>, the predicted return from this state: pale = low, deep blue = high (range at the top right).'], [sw.ring('var(--text)'), AXES]]));
    const ls2 = new Landscape(lsRow, isPG ? 'Critic: V(s)' : 'Greedy action argmax_a Q(s, a)', isPG ? 'learned state value' : 'orange = push left, blue = push right',
      lg(isPG ? [[sw.seq(), '<b>Critic V(s)</b>, the learned state value used as baseline: pale = low, deep blue = high (range at the top right). Vanilla REINFORCE has no critic.'], [sw.ring('var(--text)'), AXES]]
        : [[sw.grad(), '<b>Greedy action</b>: orange = the network prefers push left, blue = push right. A good controller shows a diagonal boundary: push in the direction the pole is falling.'], [sw.ring('var(--text)'), AXES]]));
    lab.viz.appendChild(h('div.small.muted', { html: opts.vizNote || '' }));

    // side
    const stats = RL.ui.stats(lab.side, [['ep', 'Episodes'], ['steps', 'Env steps'], ['last', 'Last return'], ['avg', 'Avg return (20)'], ['best', 'Best'], ['extra', isPG ? 'Updates' : 'ε']]);
    const chRet = new RL.LineChart(lab.side, { title: 'Return per episode (training)', info: 'Return of each training episode = number of steps the pole stayed up. Training uses exploration, so it is noisier than the greedy policy (switch learning off to watch that).', height: 165, smooth: 0.85, xLabel: 'episode', legend: false, hlines: [{ y: 500, label: 'max 500', color: '--muted' }] });
    chRet.addSeries('ret', '--accent');
    let chD1, chD2;
    if (isPG) {
      chD1 = new RL.LineChart(lab.side, { title: 'Policy entropy H(π) (ln 2 = uniform)', info: 'Average entropy of π(·|s) per update: ln 2 ≈ 0.69 = 50/50 (fully random), 0 = deterministic. A sudden drop to ≈ 0 together with low return signals policy collapse.', height: 120, xLabel: 'update', legend: false, hlines: [{ y: Math.log(2), color: '--muted' }] });
      chD1.addSeries('ent', '--c4');
      chD2 = new RL.LineChart(lab.side, { title: 'PPO diagnostics', info: 'Orange: approximate KL divergence between the policy before and after each update (how far it moved). Green: fraction of samples whose probability ratio left [1−ε, 1+ε] and were clipped.', height: 130, xLabel: 'update' });
      chD2.addSeries('approx KL', '--c2'); chD2.addSeries('clip fraction', '--c3');
    } else {
      chD1 = new RL.LineChart(lab.side, { title: 'TD loss (avg per 100 steps)', info: 'Average TD error of the training batches (log scale). Unlike in supervised learning it need not go to zero: the targets move as the network learns.', height: 120, xLabel: '×100 steps', legend: false, logY: true });
      chD1.addSeries('loss', '--c2');
      chD2 = new RL.LineChart(lab.side, { title: 'max Q of visited states (avg per 100 steps)', info: 'Average max<sub>a</sub> Q of the states visited. A pole balanced forever is worth Σγᵏ = 1/(1−γ) = 100 at γ = 0.99; values far above that indicate overestimation.', height: 120, xLabel: '×100 steps', legend: false });
      chD2.addSeries('q', '--c1');
    }

    function makeAgent() {
      const seed = Math.floor(Math.random() * 1e9);
      if (isPG) return new RL.PGAgent(Object.assign({}, cfg, { hidden: [cfg.width, cfg.width], seed }));
      return new RL.DQNAgent(Object.assign({}, cfg, { hidden: [cfg.width], seed }));
    }
    function reset() {
      tr.setPlaying(false, true); loop.stop();
      agent = makeAgent();
      env = new RL.CartPole(new RL.Rng());
      obs = env.reset(); epRet = 0; watchRet = 0; returns = []; diagAcc = { loss: [], q: [] };
      chRet.clear(); chD1.clear(); chD2.clear();
      if (isPG) chD2.el.style.display = cfg.algo === 'ppo' ? '' : 'none';
      drawAll(true);
    }

    function envStep() {
      const s = obs;
      if (training) {
        const a = agent.act(s);
        const res = env.step(a);
        if (isPG) {
          const d = agent.observe(s, a, 1, res.obs, res.done, res.truncated);
          if (d) { chD1.get('ent').data.push(d.entropy); if (cfg.algo === 'ppo') { chD2.get('approx KL').data.push(d.kl); chD2.get('clip fraction').data.push(d.clipFrac); } }
        } else {
          agent.observe(s, a, 1, res.obs, res.done);
          diagAcc.loss.push(agent.lastLoss); diagAcc.q.push(agent.lastQ);
          if (diagAcc.loss.length >= 100) { chD1.get('loss').data.push(mean(diagAcc.loss) || 1e-6); chD2.get('q').data.push(mean(diagAcc.q)); diagAcc = { loss: [], q: [] }; }
        }
        epRet++; obs = res.obs;
        if (res.done || res.truncated) { returns.push(epRet); chRet.get('ret').data.push(epRet); epRet = 0; obs = env.reset(); }
      } else {
        const a = agent.act(s, true);
        const res = env.step(a); watchRet++; obs = res.obs;
        if (res.done || res.truncated) { status.textContent = `Greedy episode finished with return ${watchRet}${res.truncated ? ' (time limit — solved!)' : ''}.`; watchRet = 0; obs = env.reset(); }
      }
    }

    const N = 32;
    function landscapes() {
      const v1 = new Float64Array(N * N), v2 = new Float64Array(N * N);
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const th = (-1 + (2 * (i + 0.5)) / N) * RL.CartPole.TH_LIM, thd = (-1 + (2 * (j + 0.5)) / N) * 2;
        const o = RL.CartPole.normalize(0, 0, th, thd);
        if (isPG) { v1[j * N + i] = agent.probs(o)[1]; v2[j * N + i] = agent.usesCritic ? agent.value(o) : 0; }
        else { const q = agent.q.forward(o); v1[j * N + i] = Math.max(q[0], q[1]); v2[j * N + i] = q[1] > q[0] ? 1 : 0; }
      }
      const marker = [env.th / RL.CartPole.TH_LIM, env.thd / 2];
      if (isPG) { ls1.draw(v1, { diverging: true, marker }); ls2.draw(v2, { marker, emptyText: agent.usesCritic ? '' : 'no critic in vanilla REINFORCE' }); }
      else { ls1.draw(v1, { marker }); ls2.draw(v2, { diverging: true, marker }); }
    }
    function drawAll(full = false) {
      const W = Math.max(280, cpHolder.clientWidth || 520);
      const ctx = setupCanvas(cpCanvas, W, 210);
      const probs = isPG ? agent.lastProbs : null;
      env.render(ctx, W, 210, { probs, label: training ? 'training' : 'greedy policy' });
      if (full || frame % 20 === 0) {
        landscapes();
        stats.ep(returns.length); stats.steps(isPG ? agent.steps : agent.steps);
        stats.last(returns.length ? returns[returns.length - 1] : '–');
        stats.avg(returns.length ? mean(returns.slice(-20)) : '–');
        stats.best(returns.length ? Math.max(...returns) : '–');
        stats.extra(isPG ? agent.updates : fmt(agent.epsilon(), 2));
        chRet.requestDraw(); chD1.requestDraw(); chD2.requestDraw();
        if (training) {
          const avg = returns.length ? mean(returns.slice(-20)) : 0;
          status.textContent = returns.length === 0 ? 'Collecting experience…'
            : avg >= 475 ? 'Solved: average return ≥ 475 over the last 20 episodes. Switch learning off to watch the greedy policy.'
              : `Training… average of last 20 episodes: ${fmt(avg, 1)}`;
        }
      }
    }
    const loop = new RL.Loop(() => {
      frame++;
      const n = SPEEDS[speed];
      if (n === Infinity) { const t0 = performance.now(); let k = 0; while (performance.now() - t0 < 14) { for (let i = 0; i < 20; i++) envStep(); k++; } }
      else for (let i = 0; i < n; i++) envStep();
      drawAll();
    });
    RL.onTheme(() => drawAll(true));
    let ro = null;
    if (window.ResizeObserver) { ro = new ResizeObserver(() => drawAll(true)); ro.observe(cpHolder); }

    showHide();
    reset();

    const api = {
      lab, cfg, ctl,
      /** Apply a configuration (updates controls) and restart. */
      configure(o, { play = true, speedKey } = {}) {
        trainT.set(true, true); training = true;
        for (const k in o) {
          cfg[k] = o[k];
          if (ctl[k]) ctl[k].set(o[k], true);
        }
        showHide();
        if (speedKey) { speed = speedKey; speedSeg.set(speedKey, true); }
        reset();
        if (play) tr.setPlaying(true);
      },
      /** Change a live setting without restarting. */
      tweak(o) { for (const k in o) { cfg[k] = o[k]; agent.cfg[k] = o[k]; if (ctl[k]) ctl[k].set(o[k], true); } },
      destroy() { loop.stop(); chRet.destroy(); chD1.destroy(); chD2.destroy(); if (ro) ro.disconnect(); },
    };
    return api;
  };
})(window.RL);
