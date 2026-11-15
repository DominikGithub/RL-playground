/* Chapter 3: model-free prediction & control — Monte Carlo, TD, SARSA, Q-learning, Double Q, Dyna-Q. */
(function (RL) {
  'use strict';
  const sw = RL.ui.sw, lg = RL.ui.lg;
  const { h, fmt, argmaxRandom } = RL.util;
  const { ARROWS, PRESETS } = RL.GridWorld;
  const ALGOS = RL.TabularAgent.ALGOS;
  const COLORS = { qlearning: '--c1', sarsa: '--c2', expsarsa: '--c3', mc: '--c5', dynaq: '--c4', doubleq: '--c6' };

  const theory = String.raw`
<h2>Learning without a model</h2>
<p>Dynamic programming averaged over all next states using $p(s',r\mid s,a)$. Model-free methods replace that expectation with <b>samples</b> — transitions the agent actually experiences. The question is what to use as the learning target.</p>
<p><b>Monte Carlo</b> waits until the episode ends and uses the actual return:</p>
$$V(s_t) \leftarrow V(s_t) + \alpha\big[\,G_t - V(s_t)\,\big].$$
<p><b>Temporal-difference (TD) learning</b> updates after one step, using its own estimate of the rest — <em>bootstrapping</em>:</p>
$$V(s_t) \leftarrow V(s_t) + \alpha\big[\,\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD target}} - V(s_t)\,\big], \qquad \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t).$$
<p>The TD error $\delta_t$ is arguably the most important quantity in RL: it drives Q-learning, every critic in actor-critic methods, GAE in PPO, and — as dopamine neurons famously appear to signal something very similar — it is a leading model of reward learning in the brain.</p>

<details class="deep math"><summary><span class="lbl">Math</span> Bias and variance of MC vs. TD targets</summary><div class="body">
<p><b>MC target</b> $G_t$: unbiased for $V^\pi(s_t)$, but its variance accumulates the randomness of <em>every</em> future action, transition and reward — roughly growing with the horizon. <b>TD target</b> $r_{t+1}+\gamma V(s_{t+1})$: involves only one random step, so much lower variance, but biased whenever $V(s_{t+1}) \ne V^\pi(s_{t+1})$. The bias vanishes as the estimates converge.</p>
<p>In the batch setting the two converge to different answers (Sutton & Barto §6.3): batch MC finds the values that minimize squared error on the observed returns; batch TD(0) finds the values of the <em>maximum-likelihood Markov model</em> of the data (the "certainty-equivalence" estimate). When the Markov property holds, TD's answer generalizes better; when it does not (partial observability), MC can be more robust.</p>
<p><b>n-step returns</b> interpolate: $G_t^{(n)} = r_{t+1} + \dots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})$. The <b>λ-return</b> averages all of them geometrically, $G^\lambda_t = (1-\lambda)\sum_{n\ge1} \lambda^{n-1} G_t^{(n)}$, and can be computed incrementally with <em>eligibility traces</em> (TD(λ)). $\lambda=0$ is TD(0), $\lambda=1$ is Monte Carlo. Chapter 5's <b>GAE</b> is precisely the λ-return idea applied to advantages.</p>
</div></details>

<h2>From prediction to control: SARSA vs. Q-learning</h2>
<p>For control we learn action values $Q(s,a)$ (so that we can act greedily without a model) and keep exploring with an ε-greedy policy. The two classic algorithms differ only in their target:</p>
$$\textbf{SARSA:}\quad Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma\, Q(s', a') - Q(s,a)\big], \quad a' \sim \pi_\varepsilon(\cdot\mid s')$$
$$\textbf{Q-learning:}\quad Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\big]$$
<ul>
<li><b>SARSA is on-policy</b>: it uses the action the agent will actually take next, so it learns $Q^{\pi_\varepsilon}$ — the value of the exploring policy, including its mistakes.</li>
<li><b>Q-learning is off-policy</b>: the $\max$ makes the target independent of how the data was collected. It learns $Q^*$ — it is sample-based value iteration — regardless of the behavior policy, as long as everything keeps being tried.</li>
<li><b>Expected SARSA</b> uses $\sum_{a'}\pi(a'\mid s')Q(s',a')$, removing the variance from sampling $a'$. With a greedy target policy it becomes Q-learning.</li>
</ul>
<div class="callout"><strong>The cliff.</strong> On cliff walking, Q-learning learns the optimal path right along the edge — but while exploring with ε = 0.1 it regularly falls off (−100), so its <em>online</em> return is worse. SARSA accounts for its own exploration and learns the safer path one row up. Which is "right" depends on whether you care about performance during learning (safety!) or about the final greedy policy.</div>

<details class="deep math"><summary><span class="lbl">Math</span> Convergence conditions and the maximization bias</summary><div class="body">
<p>Tabular Q-learning converges to $Q^*$ with probability 1 (Watkins & Dayan, 1992) if every state–action pair is visited infinitely often and step sizes satisfy the Robbins–Monro conditions $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$. SARSA converges to $Q^*$ if, additionally, the policy becomes greedy in the limit (GLIE, e.g. $\varepsilon_t = 1/t$).</p>
<p>The $\max$ in Q-learning introduces a systematic <b>overestimation</b>: with noisy estimates $\hat Q = Q + \epsilon$, Jensen's inequality gives $\mathbb E[\max_a \hat Q(s,a)] \ge \max_a \mathbb E[\hat Q(s,a)]$. The algorithm selects actions <em>because</em> their noise is positive. <b>Double Q-learning</b> (van Hasselt, 2010) keeps two tables and uses one to select the argmax and the other to evaluate it:</p>
$$Q_A(s,a) \leftarrow Q_A(s,a) + \alpha\big[r + \gamma\, Q_B\big(s', \arg\max_{a'} Q_A(s',a')\big) - Q_A(s,a)\big].$$
<p>Since the selection noise and evaluation noise are independent, the bias disappears (it becomes a slight underestimate). This exact idea, transplanted to neural networks, is Double DQN (Chapter 4) and the "clipped double-Q" trick of TD3 and SAC.</p>
</div></details>

<details class="deep math"><summary><span class="lbl">Math</span> Off-policy learning and importance sampling</summary><div class="body">
<p>To evaluate a target policy $\pi$ from data generated by a behavior policy $b$, Monte Carlo must reweight returns by the likelihood ratio of the trajectory:</p>
$$\rho_{t:T} = \prod_{k=t}^{T-1}\frac{\pi(a_k\mid s_k)}{b(a_k\mid s_k)}, \qquad V^\pi(s) = \mathbb E_b\big[\rho_{t:T} G_t \mid s_t = s\big].$$
<p>Products of ratios have variance that can grow exponentially with the horizon. One-step Q-learning sidesteps this: the action $a$ in $Q(s,a)$ is given, and the target's $\max$ needs no ratio. This is why off-policy value learning (DQN, SAC) is practical while off-policy Monte Carlo is not. Ratios return in PPO (Chapter 6), but only for a single step and clipped to stay near 1; multi-step corrections such as Retrace and V-trace truncate the ratios for the same reason.</p>
</div></details>

<h2>A first look at model-based RL: Dyna-Q</h2>
<p>Dyna-Q (Sutton, 1990) stores every observed transition in a model and, after each real step, performs $n$ extra Q-learning updates on transitions <em>replayed from the model</em>. Real experience is expensive, computation is cheap: with $n = 20$ planning steps, the maze is solved in a fraction of the real episodes. Two descendants matter today: the <b>replay buffer</b> of DQN is essentially a non-parametric Dyna model, and <b>learned world models</b> (Dreamer, MuZero; Chapter 7) are parametric ones.</p>

<details class="deep practice"><summary><span class="lbl">Practice</span> Hyperparameters in tabular control</summary><div class="body">
<ul>
<li><b>Step size α.</b> Deterministic worlds tolerate α close to 1 (just overwrite). With stochastic transitions (slip &gt; 0) large α makes Q-values jump around; small α averages but learns slowly.</li>
<li><b>Exploration.</b> Constant ε keeps learning forever but the policy is never greedy; decaying ε converges but can stop exploring too early. Optimistic initialization ($Q_0$ above any achievable return) drives systematic early exploration.</li>
<li><b>Episode caps.</b> Monte Carlo needs episodes to end; early random policies can wander for a long time. A step limit is a practical necessity — but be aware that it makes the problem technically non-Markov unless time is part of the state.</li>
</ul>
</div></details>
<details class="deep history"><summary><span class="lbl">History</span> Where tabular RL stands today</summary><div class="body">
<p>Samuel's checkers player (1959) already used a TD-like update; Sutton (1988) formalized TD(λ); Watkins (1989) introduced Q-learning; Tesauro's TD-Gammon (1992–95) reached world-class backgammon with TD(λ) and a small neural network — the first big deep-RL success, two decades early. Tabular methods themselves are now used mainly for teaching and theory, and eligibility traces are rare in deep RL (replaced by n-step returns and GAE). But the update rules on this page are used <em>verbatim</em>: the DQN loss is the squared Q-learning TD error, and every critic is trained on TD targets.</p>
</div></details>
`;

  const SPEEDS = { slow: { perSec: 3 }, normal: { perSec: 20 }, fast: { perFrame: 1 }, turbo: { perFrame: 30 }, max: { perFrame: 1500 } };

  RL.chapters.push({
    id: 'td', num: 3, part: 'Learning from Experience', title: 'Monte Carlo & TD Learning',
    mount(root) {
      RL.page.head(root, {
        num: 3, title: 'Model-free control: Monte Carlo, TD, SARSA & Q-learning',
        lede: 'No model, only experience. Watch an agent wander, fall off cliffs and gradually fill its Q-table — and see how the choice of learning target (MC, SARSA, Q-learning) changes both what it learns and how it behaves while learning.',
        tags: [['model-free'], ['tabular'], ['on- vs off-policy'], ['TD error = core of all critics', 'now']],
      });

      let presetKey = 'cliff';
      let env = RL.GridWorld.preset(presetKey);
      const cfg = { algo: 'qlearning', alpha: 0.5, gamma: PRESETS.cliff.gamma, epsilon: 0.1, epsDecay: 1, epsMin: 0.01, planning: 10, maxSteps: 500 };
      let agent = new RL.TabularAgent(env, cfg);
      let speed = 'normal', display = 'Q', showArrows = true, showTrail = true;
      let hoverS = -1, acc = 0, lastT = 0, lastInfo = null, greedyTrail = null;

      const lab = RL.page.lab(root, { title: 'Tabular control lab' });
      const C = lab.controls;
      C.appendChild(h('div.lab-title', 'Agent'));
      const tr = RL.ui.transport(C, {
        onPlay: (p) => { if (p) { greedyTrail = null; lastT = performance.now(); loop.start(); } else loop.stop(); },
        onStep: () => { tr.setPlaying(false); greedyTrail = null; lastInfo = agent.step(); record(lastInfo); refresh(); },
        onReset: () => { tr.setPlaying(false); newAgent(); },
      });
      const g0 = RL.ui.group(C, 'Environment');
      const presetSel = RL.ui.select(g0, { label: 'World', value: presetKey, options: Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.name })), onChange: (k) => loadPreset(k) });
      const slipS = RL.ui.slider(g0, { label: 'Slip probability', min: 0, max: 0.6, step: 0.05, value: env.slip, onInput: (v) => { env.slip = v; } });
      const g1 = RL.ui.group(C, 'Algorithm');
      const algoSel = RL.ui.select(g1, { label: 'Method', value: cfg.algo, options: Object.entries(ALGOS).map(([k, v]) => ({ value: k, label: v })), onChange: (v) => { cfg.algo = v; planS.show(v === 'dynaq'); newAgent(); } });
      const alphaS = RL.ui.slider(g1, { label: 'Step size α', min: 0.01, max: 1, step: 0.01, value: cfg.alpha, help: 'How far each update moves Q(s,a) toward the target. 1 = overwrite with the newest target.', onInput: (v) => { cfg.alpha = v; agent.cfg.alpha = v; } });
      const gammaS = RL.ui.slider(g1, { label: 'Discount γ', min: 0.5, max: 1, step: 0.01, value: cfg.gamma, onInput: (v) => { cfg.gamma = v; agent.cfg.gamma = v; } });
      const epsS = RL.ui.slider(g1, { label: 'Exploration ε', min: 0, max: 1, step: 0.01, value: cfg.epsilon, help: 'Probability of a uniformly random action (ε-greedy behavior policy).', onInput: (v) => { cfg.epsilon = v; agent.eps = v; } });
      const decT = RL.ui.toggle(g1, { label: 'Decay ε ×0.99 per episode', value: false, onChange: (v) => { cfg.epsDecay = v ? 0.99 : 1; agent.cfg.epsDecay = cfg.epsDecay; } });
      const planS = RL.ui.slider(g1, { label: 'Planning steps n (Dyna)', min: 0, max: 100, step: 1, value: cfg.planning, help: 'Simulated Q-learning updates from the learned model after every real step.', onInput: (v) => { cfg.planning = v; agent.cfg.planning = v; } });
      planS.show(false);
      const g2 = RL.ui.group(C, 'Run');
      const speedSeg = RL.ui.segmented(g2, { label: 'Speed', value: speed, options: Object.keys(SPEEDS).map((k) => ({ value: k, label: k })), onChange: (v) => { speed = v; } });
      const br = h('div.btn-row'); g2.appendChild(br);
      RL.ui.button(br, '+100 episodes', () => { tr.setPlaying(false); fastForward(100); }, 'small');
      RL.ui.button(br, 'Greedy rollout', () => { tr.setPlaying(false); greedyRollout(); }, 'small');
      const g3 = RL.ui.group(C, 'Display');
      RL.ui.segmented(g3, { value: display, options: [{ value: 'Q', label: 'Q(s,a)' }, { value: 'V', label: 'max Q' }, { value: 'visits', label: 'visits' }], onChange: (v) => { display = v; refresh(); } });
      g3.appendChild(h('div', { style: { height: '8px' } }));
      RL.ui.toggle(g3, { label: 'Greedy arrows', value: true, onChange: (v) => { showArrows = v; refresh(); } });
      RL.ui.toggle(g3, { label: 'Episode trail', value: true, onChange: (v) => { showTrail = v; refresh(); } });

      RL.ui.info(lab.viz.appendChild(h('div.lab-title', h('span', 'Q-table'), h('span.muted', { style: { textTransform: 'none', letterSpacing: 0, fontWeight: 400 } }, 'each cell: 4 triangles = Q(s,↑→↓←)'))), lg([
        [sw.grad(), 'Colour = value: orange negative, blue positive, pale ≈ 0. The bar below the grid gives the scale.'],
        [sw.text('◩'), '<b>Q(s,a) view</b>: each cell is split into 4 triangles, top = up, right = right, bottom = down, left = left, each coloured by Q(s,a) (numbers shown when cells are large).'],
        [sw.seq(), '<b>max Q</b> view: one colour per cell, V(s) = max<sub>a</sub> Q(s,a). <b>visits</b> view: blue intensity = how often the agent has been in the cell (log scale).'],
        [sw.text('→'), '<b>Arrows</b>: greedy action(s) under the current Q-table (visited cells only).'],
        [sw.dot('var(--accent)'), '<b>Teal dot</b>: the agent. <b>Teal line</b>: its path in the current episode; a diagonal jump back to S means it fell off the cliff.'],
        [sw.box('var(--pit)'), '<b>Red</b>: cliff (−100, back to start) or pit (terminal, −1). <b>Green</b>: goal. <b>Dark</b>: wall. <b>S</b>: start.'],
      ]), 'Q-table');
      const view = new RL.GridView(lab.viz, { maxW: 680, maxH: 380, onHover: (s) => { hoverS = s; renderInfo(); } });
      view.setEnv(env);
      const legend = RL.GridView.legend(lab.viz, () => view.scale, 'Q value');
      const status = h('div.lab-status'); lab.viz.appendChild(status);
      const info = h('div.info-box'); lab.viz.appendChild(info);
      const compareBox = h('div'); lab.viz.appendChild(compareBox);

      const stats = RL.ui.stats(lab.side, [['ep', 'Episode'], ['steps', 'Total steps'], ['ret', 'Last return'], ['avg', 'Avg return (20)'], ['len', 'Last length'], ['eps', 'Current ε']]);
      const chRet = new RL.LineChart(lab.side, { title: 'Return per episode (online, with exploration)', info: 'Sum of rewards of each episode while the agent still explores with ε-greedy, i.e. performance during learning. The greedy policy alone usually does better (try “Greedy rollout”).', height: 165, smooth: 0.9, xLabel: 'episode', legend: false });
      chRet.addSeries('ret', '--accent');
      const chLen = new RL.LineChart(lab.side, { title: 'Episode length', info: 'Number of steps per episode; it drops as the agent finds shorter routes (and episodes are capped at 500 steps).', height: 130, smooth: 0.9, xLabel: 'episode', legend: false });
      chLen.addSeries('len', '--c4');
      const cmpBtn = RL.ui.button(lab.side, 'Compare all algorithms (10 runs each)', () => compare(), 'small');
      const cmpStatus = h('div.small.muted', { style: { marginTop: '6px' } }); lab.side.appendChild(cmpStatus);
      const chCmp = new RL.LineChart(compareBox, { title: 'Algorithm comparison: average online return per episode', info: 'Each line averages 10 independent runs of one algorithm with the current α, γ and ε. Colours = algorithms, see the legend above the chart. Online returns include exploration mistakes.', height: 210, smooth: 0.8, xLabel: 'episode', emptyText: 'press “Compare all algorithms” in the right panel' });

      function newAgent() {
        agent = new RL.TabularAgent(env, Object.assign({}, cfg));
        lastInfo = null; greedyTrail = null; chRet.clear(); chLen.clear(); refresh();
      }
      function loadPreset(k) {
        tr.setPlaying(false); presetKey = k;
        env = RL.GridWorld.preset(k);
        cfg.gamma = PRESETS[k].gamma; gammaS.set(cfg.gamma, true); slipS.set(env.slip, true);
        view.setEnv(env); chCmp.removeAll(); newAgent();
      }
      function record(i) {
        if (i && i.episodeEnd) { chRet.push('ret', i.episodeReturn); chLen.push('len', i.episodeLength); }
      }
      function fastForward(n) { const before = agent.returns.length; agent.runEpisodes(n); for (let i = before; i < agent.returns.length; i++) { chRet.get('ret').data.push(agent.returns[i]); chLen.get('len').data.push(agent.lengths[i]); } chRet.requestDraw(); chLen.requestDraw(); lastInfo = null; refresh(); }
      function greedyRollout() {
        const rng = new RL.Rng(); let s = env.start, G = 0, t = 0; const trail = [s];
        while (t < 200) {
          const a = argmaxRandom(agent.qRow(s), rng);
          const r = env.step(s, a, rng); G += r.r; t++; s = r.s2; trail.push(s);
          if (r.done) break;
        }
        greedyTrail = trail;
        status.textContent = `Greedy rollout (ε = 0, no learning): return ${fmt(G, 2)} in ${t} steps${t >= 200 ? ' (stopped at 200 — policy loops)' : ''}.`;
        refresh(true);
      }
      function compare() {
        cmpBtn.disabled = true; cmpStatus.textContent = 'Running 6 algorithms × 10 runs…';
        setTimeout(() => {
          const algos = ['qlearning', 'sarsa', 'expsarsa', 'doubleq', 'dynaq', 'mc'];
          const episodes = presetKey === 'maze' || presetKey === 'traps' ? 200 : 300;
          const mk = () => { const e = env.clone(); return e; };
          const t0 = performance.now();
          const res = RL.compareTabular(mk, algos, Object.assign({}, cfg, { epsDecay: cfg.epsDecay }), episodes, 10);
          chCmp.removeAll();
          chCmp.opts.yMin = presetKey === 'cliff' ? -150 : undefined;
          for (const a of algos) chCmp.addSeries(ALGOS[a].split(' (')[0], COLORS[a], { raw: false }).data = res[a];
          chCmp.requestDraw();
          const tail = (arr) => RL.util.mean(arr.slice(-50));
          cmpStatus.textContent = 'Mean of last 50 episodes — ' + algos.map((a) => `${ALGOS[a].split(' (')[0]}: ${fmt(tail(res[a]), 1)}`).join(' · ') + ` (${Math.round(performance.now() - t0)} ms, current α, γ, ε)`;
          cmpBtn.disabled = false;
          compareBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 30);
      }

      function renderInfo() {
        const s = hoverS;
        if (s < 0 || !env.isActive(s)) {
          info.innerHTML = '<span class="k">Hover a cell to inspect its Q-values and ε-greedy action probabilities.</span>' + lastUpdateHtml();
          return;
        }
        const row = agent.qRow(s), p = agent.probs(s), m = Math.max(...row);
        info.innerHTML = `<div><span class="k">cell (${env.xy(s).join(', ')}) · visits</span> <b>${agent.visits[s]}</b></div><table>` +
          [0, 1, 2, 3].map((a) => `<tr class="${row[a] === m ? 'best' : ''}"><td>Q(s,${ARROWS[a]}) = ${fmt(row[a], 3)}</td><td class="k">π<sub>ε</sub>(${ARROWS[a]}|s) = ${fmt(p[a], 3)}</td></tr>`).join('') + '</table>' + lastUpdateHtml();
      }
      function lastUpdateHtml() {
        const i = lastInfo; if (!i || cfg.algo === 'mc') return cfg.algo === 'mc' ? '<div class="k" style="margin-top:6px">Monte Carlo: no update until the episode ends; then every visited (s,a) moves toward its observed return G<sub>t</sub>.</div>' : '';
        const tgt = { qlearning: 'r + γ·max<sub>a′</sub>Q(s′,a′)', doubleq: 'r + γ·Q<sub>B</sub>(s′, argmax Q<sub>A</sub>)', sarsa: 'r + γ·Q(s′,a′)', expsarsa: 'r + γ·Σπ(a′|s′)Q(s′,a′)', dynaq: 'r + γ·max<sub>a′</sub>Q(s′,a′)  (+ n planning updates)' }[cfg.algo];
        return `<div style="margin-top:6px"><span class="k">last update:</span> s=(${env.xy(i.s).join(',')}), a=${ARROWS[i.a]}${i.a !== undefined && lastInfo.fell ? ' — fell off the cliff!' : ''}, r=${fmt(i.r, 2)} → target ${tgt}; TD error δ = <b>${fmt(i.tdError, 3)}</b>, Q moved by α·δ = ${fmt(cfg.alpha * i.tdError, 3)}</div>`;
      }

      function refresh(keepStatus = false) {
        const arrows = showArrows ? Array.from({ length: env.nS }, (_, s) => {
          if (!env.isActive(s) || agent.visits[s] === 0) return [];
          const row = agent.qRow(s); const m = Math.max(...row);
          return [0, 1, 2, 3].filter((a) => row[a] >= m - 1e-9);
        }) : null;
        const V = display === 'V' ? Float64Array.from({ length: env.nS }, (_, s) => Math.max(...agent.qRow(s))) : null;
        const Q = display === 'Q' ? Float64Array.from({ length: env.nS * 4 }, (_, i) => agent.q((i / 4) | 0, i % 4)) : null;
        view.draw({ mode: display, V, Q, visits: agent.visits, arrows, agent: greedyTrail ? greedyTrail[greedyTrail.length - 1] : agent.s, trail: greedyTrail || (showTrail ? agent.trail : null) });
        legend.update();
        stats.ep(agent.episode); stats.steps(agent.totalSteps);
        const R = agent.returns;
        stats.ret(R.length ? R[R.length - 1] : '–');
        stats.avg(R.length ? RL.util.mean(R.slice(-20)) : '–');
        stats.len(agent.lengths.length ? agent.lengths[agent.lengths.length - 1] : '–');
        stats.eps(agent.eps);
        if (!keepStatus) status.textContent = `${ALGOS[cfg.algo]} · episode ${agent.episode + 1}, step ${agent.t}`;
        renderInfo();
      }

      const loop = new RL.Loop(() => {
        const sp = SPEEDS[speed];
        let n;
        if (sp.perSec) { const now = performance.now(); acc += ((now - lastT) / 1000) * sp.perSec; lastT = now; n = Math.floor(acc); acc -= n; }
        else n = sp.perFrame;
        if (n <= 0) return;
        for (let i = 0; i < n; i++) { lastInfo = agent.step(); record(lastInfo); }
        refresh();
      });
      RL.onTheme(() => legend.update());
      refresh();

      const setup = (o) => () => {
        tr.setPlaying(false);
        if (o.preset && o.preset !== presetKey) { presetSel.set(o.preset, true); loadPreset(o.preset); }
        if (o.algo) { cfg.algo = o.algo; algoSel.set(o.algo, true); planS.show(o.algo === 'dynaq'); }
        if (o.alpha !== undefined) { cfg.alpha = o.alpha; alphaS.set(o.alpha, true); }
        if (o.epsilon !== undefined) { cfg.epsilon = o.epsilon; epsS.set(o.epsilon, true); }
        if (o.decay !== undefined) { cfg.epsDecay = o.decay ? 0.99 : 1; decT.set(o.decay, true); }
        if (o.slip !== undefined) { env.slip = o.slip; slipS.set(o.slip, true); }
        if (o.planning !== undefined) { cfg.planning = o.planning; planS.set(o.planning, true); }
        if (o.speed) { speed = o.speed; speedSeg.set(o.speed, true); }
        newAgent();
        if (o.compare) compare();
        if (o.play) tr.setPlaying(true);
      };
      RL.ui.experiments(root, [
        { title: 'SARSA takes the safe road', text: 'Cliff walking with SARSA, ε = 0.1, fast. After a few hundred episodes the greedy arrows lead along the <em>top</em> rows, away from the cliff. Press “Greedy rollout”.', apply: setup({ preset: 'cliff', algo: 'sarsa', alpha: 0.5, epsilon: 0.1, decay: false, speed: 'turbo', play: true }), labEl: lab.el },
        { title: 'Q-learning walks the edge', text: 'Same with Q-learning. Its greedy path hugs the cliff (optimal: return −13), yet its online returns are lower because ε-exploration occasionally steps off. Then run the comparison.', apply: setup({ preset: 'cliff', algo: 'qlearning', alpha: 0.5, epsilon: 0.1, decay: false, speed: 'turbo', play: true }), labEl: lab.el },
        { title: 'The classic cliff plot', text: 'Compare all algorithms on the cliff (ε = 0.1). Expected SARSA ≥ SARSA &gt; Q-learning online; Monte Carlo is far below the chart — falling off costs −100 and full returns are extremely noisy.', apply: setup({ preset: 'cliff', alpha: 0.5, epsilon: 0.1, decay: false, compare: true }), labEl: lab.el },
        { title: 'Monte Carlo waits, TD bootstraps', text: 'Corridor, slow-ish. With Q-learning the value of the goal creeps back <em>one cell per episode</em>. Switch to Monte Carlo: all visited cells update at once at episode end — but each update is noisier.', apply: setup({ preset: 'corridor', algo: 'qlearning', alpha: 0.5, epsilon: 0.1, speed: 'fast', play: true }), labEl: lab.el },
        { title: 'Planning multiplies experience', text: 'Maze with Dyna-Q, $n = 20$ planning steps. The whole Q-table lights up after one or two episodes. Set $n = 0$ (plain Q-learning) and compare.', apply: setup({ preset: 'maze', algo: 'dynaq', planning: 20, alpha: 0.5, epsilon: 0.1, speed: 'fast', play: true }), labEl: lab.el },
        { title: 'Noise needs small steps', text: 'Trap field with slip 0.3 and α = 1: Q-values flicker, because each update overwrites with one noisy sample. Lower α to 0.1 and watch the values settle.', apply: setup({ preset: 'traps', algo: 'qlearning', alpha: 1, epsilon: 0.1, slip: 0.3, speed: 'turbo', play: true }), labEl: lab.el },
      ]);
      RL.page.prose(root, theory);
      return () => { loop.stop(); [chRet, chLen, chCmp].forEach((c) => c.destroy()); view.destroy(); };
    },
  });
})(window.RL);
