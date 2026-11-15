/* Chapter 1: MDPs, Bellman equations, dynamic programming (policy evaluation, value & policy iteration). */
(function (RL) {
  'use strict';
  const sw = RL.ui.sw, lg = RL.ui.lg;
  const { h, fmt } = RL.util;
  const { ARROWS, CELL, PRESETS } = RL.GridWorld;

  const theory = String.raw`
<h2>Markov decision processes</h2>
<p>A (finite) MDP is a tuple $(\mathcal S, \mathcal A, p, \gamma)$ with dynamics $p(s', r \mid s, a)$ — the probability of landing in $s'$ with reward $r$ after taking $a$ in $s$. The defining assumption is the <b>Markov property</b>: the future depends on the past only through the present state, $\Pr(s_{t+1}, r_{t+1}\mid s_t,a_t,s_{t-1},\dots) = p(s_{t+1},r_{t+1}\mid s_t,a_t)$. In the lab, the grid cell is the state, the four moves are actions, the <em>slip</em> slider makes transitions stochastic, and the <em>step reward</em> is paid on every move (a "living cost" when negative).</p>

<h2>Value functions and the Bellman expectation equation</h2>
<p>The state-value and action-value functions of a policy $\pi$ are expected returns:</p>
$$V^\pi(s) = \mathbb E_\pi\left[G_t \mid s_t = s\right], \qquad Q^\pi(s,a) = \mathbb E_\pi\left[G_t \mid s_t = s, a_t = a\right].$$
<p>Because $G_t = r_{t+1} + \gamma G_{t+1}$, both satisfy a one-step recursion — the <b>Bellman expectation equation</b>:</p>
$$V^\pi(s) = \sum_a \pi(a\mid s) \sum_{s',r} p(s',r\mid s,a)\,\big[r + \gamma V^\pi(s')\big], \qquad V^\pi(s) = \sum_a \pi(a\mid s)\, Q^\pi(s,a).$$
<p>Hover any cell in the lab: the box below the grid shows exactly this sum with the current numbers — one term per possible outcome of each action.</p>

<details class="deep math"><summary><span class="lbl">Math</span> Policy evaluation is just a linear system</summary><div class="body">
<p>Write $r_\pi(s) = \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)\,r$ and $P_\pi(s,s') = \sum_a \pi(a|s)\, p(s'|s,a)$. Then the Bellman expectation equation in vector form is</p>
$$V^\pi = r_\pi + \gamma P_\pi V^\pi \quad\Longrightarrow\quad V^\pi = (I - \gamma P_\pi)^{-1} r_\pi .$$
<p>For $\gamma<1$ the inverse exists because $P_\pi$ is stochastic (spectral radius 1). Direct inversion costs $O(|\mathcal S|^3)$; <em>iterative policy evaluation</em> applies $V \leftarrow r_\pi + \gamma P_\pi V$ repeatedly ($O(|\mathcal S|^2)$ per sweep, less for sparse dynamics), which is what the lab animates. This view also explains the effective horizon: $(I-\gamma P)^{-1} = \sum_k \gamma^k P^k$, a geometric weighting of the future with mean length $1/(1-\gamma)$.</p>
</div></details>

<h2>Optimality: the Bellman optimality equation</h2>
<p>The optimal value functions $V^*(s) = \max_\pi V^\pi(s)$, $Q^*(s,a)=\max_\pi Q^\pi(s,a)$ satisfy a nonlinear version where the average over the policy is replaced by a max:</p>
$$V^*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma V^*(s')\big], \qquad Q^*(s,a) = \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma \max_{a'} Q^*(s',a')\big].$$
<p>Once you have $Q^*$, acting is trivial: $\pi^*(s) = \arg\max_a Q^*(s,a)$. Every finite MDP has a <b>deterministic</b> optimal policy that is optimal from every state simultaneously. (Keep this in mind for Chapter 5: with function approximation or partial observability this stops being true, and stochastic policies can be strictly better.)</p>

<details class="deep math"><summary><span class="lbl">Math</span> Why value iteration converges: the contraction argument</summary><div class="body">
<p>Define the Bellman optimality operator $(\mathcal T V)(s) = \max_a \sum_{s',r} p(s',r|s,a)[r+\gamma V(s')]$. For any two value functions $U, V$:</p>
$$\|\mathcal T U - \mathcal T V\|_\infty \;\le\; \max_{s,a} \gamma \sum_{s'} p(s'|s,a)\,|U(s')-V(s')| \;\le\; \gamma \|U - V\|_\infty,$$
<p>using $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a)-g(a)|$. So $\mathcal T$ is a $\gamma$-contraction in the sup-norm. By Banach's fixed-point theorem it has a unique fixed point ($V^*$) and iterating converges geometrically from <em>any</em> start:</p>
$$\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty .$$
<p>This is why the convergence chart in the lab is (asymptotically) a straight line on a log scale, with slope at most $\log_{10}\gamma$ per sweep — try $\gamma = 0.5$ vs $0.99$. A practical stopping rule follows from the same inequality: if $\|V_{k+1}-V_k\|_\infty < \varepsilon$ then $\|V_{k+1}-V^*\|_\infty < \varepsilon\gamma/(1-\gamma)$.</p>
<p>With $\gamma = 1$ (allowed in episodic tasks such as the 4×3 world) the contraction argument fails, but convergence still holds if every policy eventually terminates ("proper" policies, Bertsekas & Tsitsiklis). If a region cannot reach a terminal state with negative step rewards, values diverge — try walling off a pocket with $\gamma=1$.</p>
</div></details>

<h2>Two ways to solve it: value iteration and policy iteration</h2>
<p><b>Value iteration</b> turns the optimality equation into an update rule, $V_{k+1} \leftarrow \mathcal T V_k$. <b>Policy iteration</b> alternates two steps until the policy stops changing:</p>
<ol>
<li><b>Evaluation</b>: compute $V^{\pi_k}$ (solve the linear system, or iterate the expectation backup to convergence).</li>
<li><b>Improvement</b>: $\pi_{k+1}(s) = \arg\max_a Q^{\pi_k}(s,a)$ — act greedily with respect to the current values.</li>
</ol>
<details class="deep math"><summary><span class="lbl">Math</span> The policy improvement theorem</summary><div class="body">
<p>If $\pi'$ is greedy w.r.t. $Q^\pi$, then $Q^\pi(s,\pi'(s)) \ge V^\pi(s)$ for all $s$. Unrolling this inequality:</p>
$$V^\pi(s) \le Q^\pi(s,\pi'(s)) = \mathbb E\left[r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t=s, a_t=\pi'(s)\right] \le \mathbb E_{\pi'}\left[r_{t+1} + \gamma r_{t+2} + \gamma^2 V^\pi(s_{t+2})\right] \le \dots \le V^{\pi'}(s).$$
<p>So greedy improvement never hurts, and if it leaves the policy unchanged then $V^\pi$ satisfies the Bellman optimality equation, i.e. $\pi$ is optimal. Since a finite MDP has finitely many deterministic policies, policy iteration terminates in finitely many iterations — in practice often fewer than ten, even when value iteration needs hundreds of sweeps.</p>
<p>The theorem has a modern echo: <b>policy gradient</b> and <b>PPO</b> (Chapters 5–6) are "soft" policy improvement — instead of jumping to the greedy policy, they move a parametrized policy a small step in the direction of higher advantage $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. The performance-difference lemma, $J(\pi') - J(\pi) = \frac{1}{1-\gamma}\mathbb E_{s\sim d^{\pi'},a\sim\pi'}[A^\pi(s,a)]$, is the quantitative version.</p>
</div></details>

<div class="callout"><strong>Generalized policy iteration (GPI)</strong> is the single most useful mental model in RL: <em>some</em> process makes the value estimate consistent with the current policy, <em>another</em> makes the policy greedy(-ish) with respect to the values, and the two chase each other to the optimum. Value iteration (one truncated evaluation sweep per improvement), SARSA, Q-learning, actor-critic, PPO and even AlphaZero are all instances — they differ in how evaluation and improvement are approximated.</div>

<div class="table-wrap"><table class="compare">
<tr><th></th><th>Value iteration</th><th>Policy iteration</th></tr>
<tr><td>Update</td><td>$V \leftarrow \max_a$ backup (nonlinear)</td><td>full evaluation (linear) + greedy step</td></tr>
<tr><td>Cost per iteration</td><td>one sweep, $O(|\mathcal S||\mathcal A| \cdot b)$ with branching $b$</td><td>many sweeps or a linear solve</td></tr>
<tr><td>Iterations</td><td>many — $O\!\left(\frac{\log(1/\varepsilon)}{1-\gamma}\right)$</td><td>few; finite and exact</td></tr>
<tr><td>Intermediate $V$</td><td>not the value of any particular policy</td><td>exact $V^{\pi_k}$ of a real policy</td></tr>
<tr><td>Descendants</td><td>Q-learning, DQN (sample-based VI)</td><td>actor-critic, PPO (approximate PI)</td></tr>
</table></div>

<h2>Discounting and reward design</h2>
<p>The discount does two jobs. Formally it defines the objective (how much do we care about the future?). In deep RL it is also used as a <em>variance-reduction knob</em> — a smaller $\gamma$ means shorter effective horizons and easier credit assignment, at the price of optimizing a slightly different (myopic) objective. Typical values: $0.99$ for control, $0.997$–$0.999$ for long-horizon games, and often $\gamma = 1$ for LLM fine-tuning where episodes are short and the reward arrives at the end.</p>
<details class="deep practice"><summary><span class="lbl">Practice</span> Reward shaping without changing the optimal policy</summary><div class="body">
<p>Sparse rewards (only at the goal) make learning slow. Adding hand-made bonuses can speed it up — or teach the agent to farm the bonus. Ng, Harada & Russell (1999) showed that <b>potential-based</b> shaping $F(s,s') = \gamma\Phi(s') - \Phi(s)$ for any function $\Phi$ provably leaves the optimal policy unchanged (the shaping terms telescope along any trajectory, shifting all returns from $s$ by the constant $-\Phi(s)$). A good potential is an estimate of the value itself — which is exactly what a learned critic provides.</p>
<p>The lab's 4×3 world shows how sensitive behavior is to the reward: with step reward $-0.04$ the agent takes the long safe route; around $-0.4$ it accepts risk near the pit; below $\approx -1.63$ it prefers to end the episode in the pit rather than keep paying the living cost. With a positive step reward it avoids both exits forever.</p>
</div></details>

<h2>Why dynamic programming is not the end of the story</h2>
<p>DP needs two things real problems rarely give you: a <b>known model</b> $p(s',r\mid s,a)$, and the ability to <b>sweep over every state</b>. Bellman himself named the second problem the <em>curse of dimensionality</em>. The rest of RL can be read as two relaxations of DP:</p>
<ul>
<li><b>Sample instead of sum</b> — replace the expectation over $s'$ by experienced transitions (Monte Carlo, TD, Q-learning: Chapter 3).</li>
<li><b>Approximate instead of tabulate</b> — represent $V$, $Q$ or $\pi$ with a function approximator such as a neural net (Chapter 4 onwards).</li>
</ul>
<details class="deep history"><summary><span class="lbl">History</span> Classic DP variants</summary><div class="body">
<p>Bellman (1957) introduced the optimality principle and value iteration; Howard (1960) policy iteration. Variants you may meet: <em>modified policy iteration</em> (truncate evaluation to $m$ sweeps — VI is $m=1$, PI is $m=\infty$), <em>asynchronous / Gauss–Seidel DP</em> (update states in place, in any order), <em>prioritized sweeping</em> (update states with the largest Bellman error first — the ancestor of prioritized replay), and the <em>linear-programming</em> formulation (minimize $\sum_s V(s)$ subject to $V \ge \mathcal T V$), whose dual over state–action occupancy measures underlies much of modern offline-RL and constrained-RL theory.</p>
</div></details>
`;

  RL.chapters.push({
    id: 'mdp', num: 1, part: 'Foundations', title: 'MDPs & Bellman Equations',
    mount(root) {
      RL.page.head(root, {
        num: 1, title: 'MDPs, Bellman equations & dynamic programming',
        lede: 'Before learning from experience, solve the problem with a perfect model. Watch value information flow backwards from rewards, and see how discounting, noise and living costs reshape the optimal policy.',
        tags: [['model known'], ['tabular'], ['foundation', 'now']],
      });

      let env = RL.GridWorld.preset('classic');
      let gamma = PRESETS.classic.gamma;
      let mode = 'vi';
      let runner = new RL.dp.DPRunner(env, mode, gamma);
      let showArrows = true, showNumbers = true;
      let speed = 4, acc = 0, lastT = 0;
      let hoverS = -1, selS = -1, focusA = -1; // box shows the hovered cell while over the grid, else the selected one

      const lab = RL.page.lab(root, { title: 'Dynamic programming lab' });
      const C = lab.controls;
      C.appendChild(h('div.lab-title', 'Setup'));
      const transport = RL.ui.transport(C, {
        onPlay: (p) => { if (p) { if (runner.done) runner.done = false; loop.start(); lastT = performance.now(); } else loop.stop(); },
        onStep: () => { transport.setPlaying(false); const before = snapshot(); runner.done = false; runner.step(); refresh(); explainSweep(before); },
        onReset: () => { transport.setPlaying(false); runner.reset(); chart.clear(); stableSweeps = 0; tutor.hide(); refresh(); },
        stepLabel: 'Sweep', playLabel: 'Run',
      });
      const g1 = RL.ui.group(C, 'Environment');
      const presetSel = RL.ui.select(g1, {
        label: 'World', value: 'classic',
        options: Object.entries(PRESETS).filter(([k]) => k !== 'cliff').map(([k, p]) => ({ value: k, label: p.name })),
        onChange: (k) => loadPreset(k),
      });
      const slipS = RL.ui.slider(g1, { label: 'Slip probability', min: 0, max: 0.8, step: 0.05, value: env.slip,
        help: 'With this probability the move goes 90° left or right of the intended direction (half each). Makes transitions stochastic: p(s′|s,a) has 3 outcomes.',
        onInput: (v) => { env.slip = v; changed(); } });
      const stepS = RL.ui.slider(g1, { label: 'Step reward (living cost)', min: -2, max: 0.2, step: 0.01, value: env.stepReward,
        help: 'Reward received on every move. Negative values push the agent to finish quickly; positive values make it avoid terminal states.',
        onInput: (v) => { env.stepReward = v; changed(); } });
      const g2 = RL.ui.group(C, 'Algorithm');
      const modeSel = RL.ui.select(g2, {
        label: 'Method', value: mode,
        options: [{ value: 'eval', label: 'Policy evaluation (uniform random π)' }, { value: 'vi', label: 'Value iteration' }, { value: 'pi', label: 'Policy iteration' }],
        onChange: (m) => { mode = m; runner = new RL.dp.DPRunner(env, mode, gamma); chart.clear(); refresh(); },
      });
      const gammaS = RL.ui.slider(g2, { label: 'Discount γ', min: 0, max: 1, step: 0.01, value: gamma,
        help: 'Weight of future rewards. Effective horizon ≈ 1/(1−γ). γ = 1 is fine for episodic tasks in which every policy terminates.',
        onInput: (v) => { gamma = v; runner.gamma = v; changed(); } });
      const speedS = RL.ui.slider(g2, { label: 'Sweeps per second', min: 1, max: 60, step: 1, value: speed, onInput: (v) => { speed = v; } });
      const g3 = RL.ui.group(C, 'Display');
      RL.ui.toggle(g3, { label: 'Greedy policy arrows', value: true, onChange: (v) => { showArrows = v; refresh(); } });
      RL.ui.toggle(g3, { label: 'Show numbers', value: true, onChange: (v) => { showNumbers = v; refresh(); } });
      const tutor = RL.ui.coach();
      RL.ui.tutorToggle(g3, tutor);
      let stableSweeps = 0;

      /* ---- world editor: drag a tile onto a cell, or select a cell and click a tile ---- */
      const g4 = RL.ui.group(C, 'Edit the world');
      const TYPES = [
        { type: 'empty', label: 'Empty', key: 'E', help: 'Free cell the agent can walk on.' },
        { type: 'wall', label: 'Wall', key: 'W', help: 'Blocks movement; bumping into it leaves the agent in place.' },
        { type: 'pit', label: 'Pit', key: 'P', help: 'Terminal cell with the pit reward (−1).' },
        { type: 'goal', label: 'Goal', key: 'G', help: 'Terminal cell with the goal reward (+1).' },
        { type: 'start', label: 'Start', key: 'S', help: 'Where episodes begin (only one).' },
      ];
      const selLabel = h('div.edit-selected', 'No cell selected');
      const tileGrid = h('div.tool-grid', { role: 'toolbar', 'aria-label': 'Cell types' });
      const editHint = h('div.edit-hint', { html: '<b>Drag</b> a tile onto the grid, or <b>click a cell</b> and then a tile. Keys: arrows move, <span class="kbd">W</span> <span class="kbd">P</span> <span class="kbd">G</span> <span class="kbd">S</span> <span class="kbd">E</span> set the type.' });
      g4.append(selLabel, tileGrid, editHint);
      const tiles = {};
      for (const t of TYPES) {
        const el = h('button.tool-tile', { type: 'button', title: `${t.label} (${t.key}) — ${t.help}`, 'aria-pressed': 'false' },
          h('span.tool-swatch.sw-' + t.type, t.type === 'start' ? 'S' : t.type === 'goal' ? '+1' : t.type === 'pit' ? '−1' : ''),
          h('span.tool-name', t.label));
        tiles[t.type] = el;
        tileGrid.appendChild(el);
        attachDrag(el, t);
      }

      // viz
      RL.ui.info(lab.viz.appendChild(h('div.lab-title', h('span', 'Values V(s) and greedy policy'), h('span.muted', { style: { textTransform: 'none', letterSpacing: 0, fontWeight: 400 } }, 'hover or click a cell for its Bellman backup'))), lg([
        [sw.grad(), 'Cell colour = current value <b>V(s)</b>: orange negative, blue positive, pale ≈ 0. The bar under the grid gives the scale; the number in a cell is V(s).'],
        [sw.text('→'), '<b>Arrows</b>: greedy action(s), i.e. the move(s) with the highest one-step lookahead Q(s,a). Several arrows = a tie. Policy iteration shows its current policy π<sub>k</sub> instead.'],
        [sw.box('var(--goal)'), '<b>Goal</b>: terminal, reward +1 on entering.'],
        [sw.box('var(--pit)'), '<b>Pit</b>: terminal, reward −1 on entering.'],
        [sw.box('var(--wall)'), '<b>Wall</b>: not a state; bumping into it leaves the agent where it is.'],
        [sw.text('S'), '<b>Start</b> cell of every episode.'],
        [sw.outline('var(--accent)'), '<b>Selected</b> cell (click it). The box under the grid shows its Bellman backup.'],
        [sw.outline('var(--text)'), 'Cell under the mouse. While you hover, the box previews this cell; leaving the grid returns to the selected one.'],
        [sw.box('var(--accent-soft)'), 'Box under the grid: one tile per action with Q(s,a); ★ = best. The line below expands one action’s sum Σ p·(r + γ·V(s′)), one term per slip outcome.'],
      ]), 'Grid world');
      const view = new RL.GridView(lab.viz, {
        maxW: 640, maxH: 430,
        onHover: (s) => { if (s !== hoverS) { hoverS = s; focusA = -1; renderInfo(); } },
        onSelect: (s) => select(s),
        onKey: (e) => onGridKey(e),
      });
      view.setEnv(env);
      const legend = RL.GridView.legend(lab.viz, () => view.scale, 'V(s)');
      const status = h('div.lab-status');
      lab.viz.appendChild(status);
      const info = h('div.info-box.bellman-box', { 'aria-live': 'polite' });
      lab.viz.appendChild(info);
      info.addEventListener('mouseover', (e) => {
        const t = e.target.closest('.q-tile');
        if (t && +t.dataset.a !== focusA) { focusA = +t.dataset.a; renderInfo(); }
      });

      // side
      lab.side.appendChild(h('div.lab-title', 'Convergence'));
      const stats = RL.ui.stats(lab.side, [['sweeps', 'Sweeps'], ['delta', 'max |ΔV|', 'Largest change of any state value in the last sweep. Converged when below 1e-4.'], ['vstart', 'V(start)'], ['phase', 'Phase']]);
      const chart = new RL.LineChart(lab.side, { title: 'max |ΔV| per sweep (log scale)', info: 'Largest change of any state value in each sweep. Value iteration is a γ-contraction, so this error shrinks at least by a factor γ per sweep; the run stops once it is below 10⁻⁴.', logY: true, height: 170, xLabel: 'sweep', legend: false });
      chart.addSeries('delta', '--accent');
      const explain = h('div.small.muted');
      lab.side.appendChild(explain);

      function loadPreset(k) {
        transport.setPlaying(false);
        env = RL.GridWorld.preset(k);
        gamma = PRESETS[k].gamma;
        gammaS.set(gamma, true); slipS.set(env.slip, true); stepS.set(env.stepReward, true);
        runner = new RL.dp.DPRunner(env, mode, gamma);
        selS = -1; hoverS = -1; view.setEnv(env); chart.clear(); updateEditor(); refresh();
      }
      function changed() {
        runner.env = env; runner.done = false;
        if (mode === 'pi') runner.phase = 'evaluate';
        for (let s = 0; s < env.nS; s++) if (!env.isActive(s)) runner.V[s] = 0;
        refresh();
      }
      /* ---------- world editing ---------- */
      const cellType = (s) => (s === env.start ? 'start' : { [CELL.WALL]: 'wall', [CELL.PIT]: 'pit', [CELL.GOAL]: 'goal' }[env.cells[s]] || 'empty');
      const typeLabel = (t) => TYPES.find((x) => x.type === t).label;
      const say = (msg, warn = false) => { editHint.innerHTML = msg; editHint.classList.toggle('warn', warn); };
      function select(s) {
        selS = s; view.selected = s; focusA = -1;
        updateEditor(); renderInfo(); view.redraw();
      }
      function updateEditor() {
        const cur = selS >= 0 ? cellType(selS) : null;
        for (const t of TYPES) {
          tiles[t.type].classList.toggle('active', t.type === cur);
          tiles[t.type].setAttribute('aria-pressed', t.type === cur ? 'true' : 'false');
        }
        selLabel.innerHTML = selS >= 0
          ? `Selected: cell (${env.xy(selS).join(', ')}) · <b>${typeLabel(cur)}</b>`
          : 'No cell selected';
      }
      /** Set cell s to a type. Returns false (with a message) when the change is not allowed. */
      function apply(type, s) {
        const cur = cellType(s);
        if (type === cur) return true;
        if (type === 'start') { env.cells[s] = CELL.EMPTY; env.start = s; }
        else {
          if (cur === 'start') { say('This is the start cell — it must stay free. Move the <b>Start</b> tile to another cell first.', true); return false; }
          env.cells[s] = { empty: CELL.EMPTY, wall: CELL.WALL, pit: CELL.PIT, goal: CELL.GOAL }[type];
        }
        if (mode === 'pi' || mode === 'eval') runner.pi[s] = new Float64Array(4).fill(0.25);
        say(`Cell (${env.xy(s).join(', ')}) is now <b>${typeLabel(type)}</b>. Values update on the next sweep — press Run.`);
        changed(); updateEditor();
        return true;
      }
      function applyToSelected(type) {
        if (selS < 0) { say('First click a cell on the grid (or drag this tile onto a cell).', true); return; }
        apply(type, selS);
      }
      /** Pointer-based drag & drop (works with mouse, pen and touch); a click without movement applies to the selection. */
      function attachDrag(el, t) {
        let start = null, dragging = false, ghost = null;
        el.addEventListener('pointerdown', (e) => {
          if (e.button !== 0) return;
          start = { x: e.clientX, y: e.clientY };
          el.setPointerCapture(e.pointerId);
        });
        el.addEventListener('pointermove', (e) => {
          if (!start) return;
          if (!dragging && Math.hypot(e.clientX - start.x, e.clientY - start.y) > 5) {
            dragging = true;
            ghost = h('div.tool-ghost', h('span.tool-swatch.sw-' + t.type, t.type === 'start' ? 'S' : ''), t.label);
            document.body.appendChild(ghost);
            document.body.classList.add('dragging-tool');
          }
          if (dragging) {
            ghost.style.left = e.clientX + 'px'; ghost.style.top = e.clientY + 'px';
            const s = view.cellAt(e.clientX, e.clientY);
            if (s !== view.dropTarget) { view.dropTarget = s; view.redraw(); }
          }
        });
        const end = (cancel) => {
          if (!start) return;
          const wasDrag = dragging, s = view.dropTarget;
          start = null; dragging = false;
          if (ghost) { ghost.remove(); ghost = null; }
          document.body.classList.remove('dragging-tool');
          view.dropTarget = -1;
          if (cancel) { view.redraw(); return; }
          if (wasDrag) { if (s >= 0) { select(s); apply(t.type, s); } else { view.redraw(); say('Drop the tile onto a cell of the grid.', true); } }
          else applyToSelected(t.type);
          // keep keyboard shortcuts (arrows, W/P/G/S/E) working after a mouse/touch interaction
          if (selS >= 0) view.canvas.focus({ preventScroll: true });
        };
        el.addEventListener('pointerup', () => end(false));
        el.addEventListener('pointercancel', () => end(true));
        el.addEventListener('click', (e) => { if (e.detail === 0) applyToSelected(t.type); }); // keyboard activation
      }
      function onGridKey(e) {
        const k = e.key;
        const DIR = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
        if (DIR[k]) {
          e.preventDefault();
          if (selS < 0) { select(env.start); return; }
          const [x, y] = env.xy(selS);
          const nx = Math.max(0, Math.min(env.w - 1, x + DIR[k][0])), ny = Math.max(0, Math.min(env.h - 1, y + DIR[k][1]));
          select(env.idx(nx, ny));
          return;
        }
        if (k === 'Escape') { select(-1); return; }
        const t = TYPES.find((x) => x.key.toLowerCase() === k.toLowerCase()) || ((k === 'Delete' || k === 'Backspace') && TYPES[0]);
        if (t && selS >= 0) { e.preventDefault(); apply(t.type, selS); }
      }

      /* ---------- Bellman backup box ---------- */
      function renderInfo() {
        const s = hoverS >= 0 ? hoverS : selS;
        if (s < 0) {
          info.innerHTML = '<span class="k">Hover a cell to preview its one-step lookahead, or <b>click</b> it to pin it here: for each action, the expected reward plus discounted value of the next state, averaged over the possible slip outcomes.</span>';
          return;
        }
        const [x, y] = env.xy(s);
        // Make it explicit which cell the box describes, so it never looks out of sync with the selection frame.
        const which = s === selS ? '<span class="bb-tag sel">selected</span>' : '<span class="bb-tag">hover</span>';
        const sel = selS >= 0 ? `(${env.xy(selS).join(', ')})` : '';
        const tip = s === selS ? 'Hover an action tile to see its sum.'
          : selS >= 0 ? `Previewing the hovered cell — move off the grid to return to the selected cell ${sel}.`
            : 'Previewing — click the cell to pin it here, then hover an action tile to see its sum.';
        if (!env.isActive(s)) {
          info.innerHTML = `<div class="bb-head">${which}<b>Cell (${x}, ${y}) · ${typeLabel(cellType(s))}</b></div><span class="k">${env.isWall(s) ? 'Walls are not states — the agent can never be here.' : 'Terminal state: the episode ends here, so V = 0 by definition; its reward is paid on the transition into it.'}</span>` +
            (s === selS ? '' : `<div class="k bb-tip">${tip}</div>`);
          return;
        }
        const V = runner.V;
        const q = RL.dp.qFromV(env, V, s, gamma);
        const m = Math.max(...q);
        const best = q.indexOf(m);
        const a0 = focusA >= 0 ? focusA : best;
        const pi = runner.pi[s];
        const vNew = mode === 'vi' ? m : [0, 1, 2, 3].reduce((acc2, a) => acc2 + pi[a] * q[a], 0);
        const rule = mode === 'vi' ? 'max<sub>a</sub> Q(s,a)' : 'Σ<sub>a</sub> π(a|s)·Q(s,a)';
        const tilesHtml = [0, 1, 2, 3].map((a) => {
          const isBest = Math.abs(q[a] - m) < 1e-9;
          return `<div class="q-tile${isBest ? ' best' : ''}${a === a0 ? ' focus' : ''}" data-a="${a}">` +
            `<span class="qa">${ARROWS[a]} ${RL.GridWorld.ACTION_NAMES[a]}${isBest ? ' ★' : ''}</span><span class="qv">${fmt(q[a], 3)}</span>` +
            (mode !== 'vi' ? `<span class="qp">π = ${fmt(pi[a], 2)}</span>` : '') + '</div>';
        }).join('');
        const terms = env.outcomes(s, a0).map((o) =>
          `<span class="bb-term"><span class="k">${o.actual === a0 ? '' : 'slip ' + ARROWS[o.actual] + ' '}</span>${fmt(o.p, 2)}·(${fmt(o.r, 2)} + ${fmt(gamma, 2)}·${o.done ? '0' : fmt(V[o.s2], 3)})</span>`).join(' + ');
        info.innerHTML =
          `<div class="bb-head">${which}<b>Cell (${x}, ${y})</b> · V(s) = <b>${fmt(V[s], 3)}</b> → next sweep: ${rule} = <b>${fmt(vNew, 3)}</b></div>` +
          `<div class="q-tiles">${tilesHtml}</div>` +
          `<div class="bb-break"><span class="k">Q(s,${ARROWS[a0]}) = Σ p·(r + γ·V(s′)) =</span> ${terms} = <b>${fmt(q[a0], 3)}</b></div>` +
          `<div class="k bb-tip">${tip}</div>`;
      }

      /* ---------- Tutor: rule-based explanation after a manual "Sweep" ---------- */
      const policyKey = () => (mode === 'pi'
        ? runner.pi.map((p, s) => (env.isActive(s) ? [0, 1, 2, 3].filter((a) => p[a] > 0).join('') : ''))
        : RL.dp.greedy(env, runner.V, gamma).best.map((b, s) => (env.isActive(s) ? b.join('') : '')));
      const snapshot = () => ({ policy: policyKey(), phase: runner.phase, improvements: runner.improvements, nDeltas: runner.deltas.length, done: runner.done });
      const f4 = (x) => fmt(x, x < 0.01 ? 5 : 3);
      const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
      function explainSweep(b) {
        const D = runner.deltas, d = D[D.length - 1], dPrev = D[D.length - 2];
        const after = policyKey();
        const changed = after.reduce((n, k, s) => n + (k !== b.policy[s] ? 1 : 0), 0);
        const g = fmt(gamma, 2);
        if (mode === 'pi') {
          if (D.length > b.nDeltas) { // this click was an evaluation sweep
            if (runner.phase === 'improve') return tutor.say({ tone: 'info', title: `Evaluation of π${runner.improvements} finished`,
              text: `The largest change was ${f4(d)}, below the 10⁻⁴ threshold, so V now equals V<sup>π</sup> of the current policy. The next Sweep is an <b>improvement step</b>: every state switches to its greedy action.`,
              basis: `max |ΔV| of sweep ${runner.sweeps}` });
            return tutor.say({ tone: dPrev === undefined || d < dPrev ? 'good' : 'unclear', title: `Evaluating π${runner.improvements}`,
              text: (dPrev === undefined ? `First evaluation sweep: largest change ${f4(d)}. ` : `Largest change ${f4(dPrev)} → ${f4(d)}. `) +
                'Policy iteration first computes the value of the <em>current</em> policy; the arrows do not change during these sweeps. When the change drops below 10⁻⁴ the policy gets improved.',
              basis: 'max |ΔV| of the last two evaluation sweeps' });
          }
          if (runner.policyStable) return tutor.say({ tone: 'done', title: 'Optimal policy found',
            text: `The greedy step changed nothing, so the policy satisfies the Bellman optimality equation: it is <b>optimal</b>. It took ${plural(runner.improvements - 1, 'improvement step')} but ${plural(runner.sweeps, 'evaluation sweep')} — few improvements, many cheap sweeps.`,
            basis: 'comparison of the old and new greedy policy in every state' });
          return tutor.say({ tone: 'good', title: `Improvement step → π${runner.improvements}`,
            text: `The policy changed in ${plural(changed, 'state')}. By the <b>policy improvement theorem</b> the new policy is at least as good as the old one in every state. Next, it is evaluated again.`,
            basis: 'states whose greedy action differs from the previous policy' });
        }
        const name = mode === 'vi' ? 'value iteration' : 'policy evaluation';
        if (runner.done) return tutor.say({ tone: 'done', title: 'Converged',
          text: mode === 'vi'
            ? `The largest change was ${f4(d)}, below the 10⁻⁴ threshold. The values (almost) satisfy the Bellman optimality equation, so the arrows show an <b>optimal</b> policy.`
            : `The values of the <em>random</em> policy have converged (largest change ${f4(d)}). The arrows show one greedy improvement over it — usually good, but not guaranteed optimal; that needs policy or value iteration.`,
          basis: `max |ΔV| of sweep ${runner.sweeps}` });
        if (D.length === 1) return tutor.say({ tone: 'info', title: 'First sweep',
          text: `Starting from V = 0, every state looked one step ahead. Only cells next to a goal or pit, or paying the step reward, changed (largest change ${f4(d)}). Press Sweep again: information about rewards travels <b>one cell further per sweep</b>.`,
          basis: 'max |ΔV| of sweep 1' });
        const ratio = d / dPrev;
        let tone, text;
        if (gamma < 1 && ratio <= gamma + 1e-9) {
          tone = 'good';
          text = `The largest change shrank from ${f4(dPrev)} to ${f4(d)} (×${fmt(ratio, 2)}). ${name[0].toUpperCase() + name.slice(1)} is a γ-contraction, so this shrinks by at least the factor γ = ${g} per sweep.`;
        } else if (gamma < 1) {
          tone = 'unclear';
          text = `The largest change went from ${f4(dPrev)} to ${f4(d)}, more than the factor γ = ${g} that the contraction guarantees. That guarantee only holds while the problem stays the same — did you just edit the world or move a slider? From here on it applies again.`;
        } else if (ratio < 0.999) {
          tone = 'good';
          text = `The largest change shrank from ${f4(dPrev)} to ${f4(d)}. With γ = 1 there is no contraction factor, but convergence still holds as long as every policy eventually reaches a terminal state.`;
        } else {
          tone = 'unclear';
          text = `The largest change did not shrink (${f4(dPrev)} → ${f4(d)}). With γ = 1 nothing is discounted, so reward information keeps spreading at full strength until it has reached every state — expected early on, but if it never shrinks, some states may be unable to reach a terminal state.`;
        }
        stableSweeps = changed === 0 ? stableSweeps + 1 : 0;
        text += changed > 0
          ? ` The greedy arrows changed in ${plural(changed, 'state')}.`
          : stableSweeps >= 3
            ? ` The arrows have not changed for ${stableSweeps} sweeps: the policy often settles <b>before</b> the values converge — the remaining sweeps mostly polish the numbers.`
            : ' The arrows did not change in this sweep.';
        tutor.say({ tone, title: `Sweep ${runner.sweeps}`, text, basis: 'max |ΔV| of the last two sweeps and the greedy arrows before vs. after' });
      }

      function refresh() {
        let arrows = null;
        if (showArrows) {
          if (mode === 'pi') arrows = runner.pi.map((p, s) => (env.isActive(s) && !(p[0] === 0.25 && p[1] === 0.25 && p[2] === 0.25) ? [0, 1, 2, 3].filter((a) => p[a] > 0) : []));
          else arrows = RL.dp.greedy(env, runner.V, gamma).best;
        }
        view.draw({ mode: 'V', V: runner.V, arrows, numbers: showNumbers });
        legend.update();
        const d = runner.deltas[runner.deltas.length - 1];
        stats.sweeps(runner.sweeps);
        stats.delta(d === undefined ? '–' : fmt(d, 5));
        stats.vstart(runner.V[env.start]);
        stats.phase(runner.done ? (mode === 'pi' ? 'optimal ✓' : 'converged ✓') : mode === 'pi' ? runner.phase : mode === 'vi' ? 'iterating' : 'evaluating');
        chart.set('delta', runner.deltas.slice());
        status.textContent = runner.lastEvent || 'Press Run (or Sweep) to start from V = 0.';
        explain.innerHTML = mode === 'vi'
          ? 'Each sweep applies V ← max<sub>a</sub> Σ p·[r + γV(s′)] to every state. Rewards propagate one step per sweep; the error shrinks at least by a factor γ per sweep.'
          : mode === 'pi'
            ? 'Policy iteration evaluates the current policy to convergence (many sweeps), then improves it greedily. Arrows show the current policy π<sub>k</sub>; count how few improvements are needed.'
            : 'Evaluates the uniformly random policy: V<sup>π</sup> is the expected return of a random walk. Arrows show the greedy improvement over it — one step of policy iteration.';
        renderInfo();
      }

      const loop = new RL.Loop(() => {
        const now = performance.now();
        acc += ((now - lastT) / 1000) * speed; lastT = now;
        let n = Math.floor(acc); acc -= n;
        if (n > 0) {
          n = Math.min(n, 50);
          for (let i = 0; i < n && !runner.done; i++) runner.step();
          if (runner.done) transport.setPlaying(false);
          refresh();
        }
      });
      RL.onTheme(() => legend.update());
      refresh();

      const setup = (o) => () => {
        transport.setPlaying(false);
        if (o.preset) { presetSel.set(o.preset, true); loadPreset(o.preset); }
        if (o.mode) { modeSel.set(o.mode, true); mode = o.mode; runner = new RL.dp.DPRunner(env, mode, gamma); chart.clear(); }
        if (o.gamma !== undefined) { gammaS.set(o.gamma, true); gamma = o.gamma; runner.gamma = o.gamma; }
        if (o.slip !== undefined) { slipS.set(o.slip, true); env.slip = o.slip; }
        if (o.step !== undefined) { stepS.set(o.step, true); env.stepReward = o.step; }
        if (o.speed) { speedS.set(o.speed, true); speed = o.speed; }
        runner.reset(); chart.clear(); refresh();
        if (o.play) transport.setPlaying(true);
      };
      RL.ui.experiments(root, [
        { title: 'Watch value flow backwards', text: 'Long corridor, value iteration, 2 sweeps/s. Information about the goal travels exactly one cell per sweep — DP is "backward induction". At convergence $V(\\text{start}) = \\gamma^{13}$.', apply: setup({ preset: 'corridor', mode: 'vi', gamma: 0.9, speed: 2, play: true }), labEl: lab.el },
        { title: 'Discounting = myopia', text: 'Same corridor with $\\gamma = 0.5$. Values near the start become almost zero ($0.5^{13}\\approx 10^{-4}$) — yet the arrows are still correct. Then watch the convergence chart: steeper slope = faster contraction.', apply: setup({ preset: 'corridor', mode: 'vi', gamma: 0.5, speed: 8, play: true }), labEl: lab.el },
        { title: 'Risk changes the route', text: 'Cliff edge world with slip 0: the policy walks straight along the pits. Now raise <b>slip</b> to 0.2 and re-run: arrows near the edge turn away from the pits — the optimal path detours.', apply: setup({ preset: 'edge', mode: 'vi', slip: 0, gamma: 0.95, speed: 20, play: true }), labEl: lab.el },
        { title: 'Living cost so high that the pit looks attractive', text: 'Classic 4×3 with step reward −2. Ending the episode in the −1 pit is better than paying −2 per step, so cells near the pit point into it. Try −0.04 (textbook), −0.4 and +0.1 (then the agent never wants to leave; set γ = 0.99).', apply: setup({ preset: 'classic', mode: 'vi', step: -2, gamma: 1, speed: 20, play: true }), labEl: lab.el },
        { title: 'Policy iteration needs few improvements', text: 'Maze with policy iteration. Many evaluation sweeps per phase, but only a handful of improvement steps until the policy is provably optimal. Compare the total with value iteration.', apply: setup({ preset: 'maze', mode: 'pi', gamma: 0.95, speed: 40, play: true }), labEl: lab.el },
        { title: 'The value of acting randomly', text: 'Evaluate the uniformly random policy in the trap field. $V^\\pi$ is very different from $V^*$, yet the greedy arrows w.r.t. $V^\\pi$ are already a decent policy: one improvement step does most of the work.', apply: setup({ preset: 'traps', mode: 'eval', gamma: 0.95, speed: 30, play: true }), labEl: lab.el },
      ]);
      RL.page.prose(root, theory);
      return () => { loop.stop(); chart.destroy(); view.destroy(); tutor.hide(); };
    },
  });
})(window.RL);
