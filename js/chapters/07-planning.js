/* Chapter 7: model-based RL and planning — MCTS, AlphaZero/MuZero, world models. */
(function (RL) {
  'use strict';
  const sw = RL.ui.sw, lg = RL.ui.lg;
  const { h, fmt } = RL.util;
  const TTT = RL.TTT;

  const intro = String.raw`
<h2>Two ways to use a model</h2>
<p>Everything since Chapter 3 was model-free: the agent never asked "what would happen if…?". A model $\hat p(s', r\mid s, a)$ — given (game rules, a simulator) or learned — can be used in two fundamentally different ways:</p>
<ul>
<li><b>Background planning</b>: use the model to generate extra experience or gradients for learning a policy/value function. Dyna-Q (Chapter 3), MBPO, and world-model agents such as Dreamer, which train an actor-critic entirely "in imagination".</li>
<li><b>Decision-time planning</b>: at each step, look ahead from the <em>current</em> state with the model and pick an action from the search result. Model-predictive control (MPC) and Monte Carlo Tree Search (MCTS).</li>
</ul>
<p>The lab below is pure decision-time planning: Monte Carlo Tree Search with random rollouts, given the rules of tic-tac-toe as its model, and no learning at all. Its strength comes only from computation — which you control with the simulation budget.</p>`;

  const theory = String.raw`
<h2>Monte Carlo Tree Search</h2>
<p>MCTS grows an asymmetric search tree, spending simulations on promising lines. Each simulation has four phases:</p>
<ol>
<li><b>Selection</b> — from the root, repeatedly choose the child maximizing an upper-confidence score until reaching a node with untried moves.</li>
<li><b>Expansion</b> — add one new child for an untried move.</li>
<li><b>Evaluation</b> — estimate the value of the new node: originally by a random <em>rollout</em> to the end of the game (as here), in AlphaZero by a value network.</li>
<li><b>Backup</b> — propagate the result up the path, updating visit counts $N$ and mean values $Q$.</li>
</ol>
<p>The selection rule is a bandit algorithm at every node — <b>UCT</b>, "UCB applied to trees" (Kocsis & Szepesvári, 2006):</p>
$$a^* = \arg\max_a \Big[\, Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}\,\Big].$$
<p>The move finally played is the most visited child (more robust than the highest $Q$). As the number of simulations grows, UCT's value estimates converge to the minimax values. Set the budget to 10 and the agent blunders; at a few thousand it never loses.</p>

<h2>AlphaZero: search as a policy-improvement operator</h2>
<p>AlphaGo Zero and AlphaZero (Silver et al., 2017–2018) replace the rollout with a network $f_\theta(s) = (\mathbf p, v)$ that outputs move priors and a value, and use the <b>PUCT</b> selection rule</p>
$$a^* = \arg\max_a \Big[\, Q(s,a) + c_\text{puct}\, P(s,a)\,\frac{\sqrt{N(s)}}{1 + N(s,a)}\,\Big].$$
<p>The key insight is that the visit-count distribution $\boldsymbol\pi_\text{MCTS}(a\mid s) \propto N(s,a)^{1/T}$ is a <em>better policy</em> than the raw network prior $\mathbf p$. Training then alternates (generalized policy iteration once more!):</p>
$$\mathcal L(\theta) = \underbrace{(z - v)^2}_{\text{value} \to \text{game outcome}} \;-\; \underbrace{\boldsymbol\pi_\text{MCTS}^\top \log \mathbf p}_{\text{policy} \to \text{search result}} \;+\; c\|\theta\|^2 .$$
<ul>
<li><b>Policy improvement</b>: MCTS with the current network produces stronger move distributions.</li>
<li><b>Policy evaluation / distillation</b>: the network is trained to predict those distributions and the final self-play outcomes $z$.</li>
</ul>
<details class="deep math"><summary><span class="lbl">Math</span> MuZero and planning with a learned model</summary><div class="body">
<p>MuZero (Schrittwieser et al., 2020) removes the need for known rules. It learns three functions: a <em>representation</em> $h_\theta$: observation → latent state, <em>dynamics</em> $g_\theta(s^k, a^{k+1}) \to (r^{k+1}, s^{k+1})$, and <em>prediction</em> $f_\theta(s^k) \to (\mathbf p^k, v^k)$. The model is trained end-to-end only to predict quantities needed for planning — rewards, values and policies $K$ steps ahead — not to reconstruct observations:</p>
$$\mathcal L = \sum_{k=0}^{K}\Big[\ell^r(u_{t+k}, r^k) + \ell^v(z_{t+k}, v^k) + \ell^p(\boldsymbol\pi_{t+k}, \mathbf p^k)\Big].$$
<p>This "value-equivalent" model may be useless for predicting pixels but is exactly what MCTS needs. Successors: EfficientZero (data-efficient Atari), Sampled MuZero (continuous actions), Stochastic MuZero (chance nodes), Gumbel MuZero (policy improvement guarantees with few simulations).</p>
</div></details>

<h2>World models and imagination</h2>
<p>For continuous control and visual domains, the dominant model-based family learns a latent dynamics model and trains a policy inside it. <b>Dreamer</b> (Hafner et al., 2019–2023) uses a recurrent state-space model; DreamerV3 works across 150+ tasks with one set of hyperparameters and was the first to collect diamonds in Minecraft from scratch. <b>TD-MPC2</b> combines a learned latent model with short-horizon MPC and a terminal value function. <b>MBPO</b> (Janner et al., 2019) shows the key practical rule: model errors compound with rollout length, so use <em>short</em> model rollouts branched from real states and let a model-free learner (SAC) handle the long horizon.</p>

<div class="table-wrap"><table class="compare">
<tr><th></th><th>Model-free</th><th>Model-based</th></tr>
<tr><td>Sample efficiency</td><td>low (millions of steps)</td><td>high (often 10–100× fewer real steps)</td></tr>
<tr><td>Asymptotic performance</td><td>high; no model bias</td><td>limited by model errors unless corrected</td></tr>
<tr><td>Compute per real step</td><td>low</td><td>high (model training, search, imagination)</td></tr>
<tr><td>Adapting to new goals</td><td>retrain</td><td>re-plan with the same model</td></tr>
<tr><td>Typical use</td><td>cheap simulators, LLM post-training</td><td>board games, real robots, expensive data</td></tr>
</table></div>

<h2>Search, test-time compute and language models</h2>
<p>The AlphaZero lesson — <em>spend computation at decision time and distill the result back into the policy</em> — reappears in reasoning models. Sampling many solutions and keeping verified ones (best-of-$n$, majority voting), tree search over reasoning steps, and RL that teaches a model to produce long chains of thought are all ways to trade inference compute for quality. In practice, RL on verifiable rewards (Chapter 8) has so far been more scalable for LLMs than explicit MCTS over tokens, whose branching factor is huge and whose intermediate states are hard to evaluate; but explicit search remains central where a precise model exists, e.g. AlphaProof's RL in the Lean proof assistant.</p>

<details class="deep practice"><summary><span class="lbl">Practice</span> MCTS details</summary><div class="body">
<ul>
<li><b>Perspective.</b> In two-player games every node stores values from the viewpoint of the player who made the move into it; the backup flips perspective each ply. Getting this wrong is the classic MCTS bug.</li>
<li><b>Exploration constant.</b> $c\approx\sqrt2$ for rewards in $[0,1]$; smaller values make search greedy and deep, larger ones broad and shallow. Try $c=0$ in the lab: the search commits to the first promising line and misses threats.</li>
<li><b>Final move choice.</b> Use the most-visited child, not the highest mean; low-visit children have unreliable means.</li>
<li><b>Dirichlet noise</b> at the root and temperature-based move sampling provide exploration in AlphaZero self-play.</li>
<li><b>Tree reuse</b>: keep the subtree of the chosen move for the next decision.</li>
</ul>
</div></details>
<details class="deep history"><summary><span class="lbl">History</span> Older model-based ideas</summary><div class="body">
<p>Classical planning and heuristic search (A*, minimax with alpha–beta, as in Deep Blue) assume perfect models. Dyna (1990) and prioritized sweeping are the tabular ancestors of learned-model RL. <b>PILCO</b> (2011) used Gaussian-process dynamics for remarkable data efficiency on small systems but does not scale; <b>PETS</b> (2018) used probabilistic ensembles with MPC. Theory-driven exploration algorithms such as <b>E³</b> and <b>R-max</b> ("optimism under uncertainty" with a learned model) gave the first polynomial sample-complexity bounds but are not practical at scale.</p>
</div></details>
`;

  RL.chapters.push({
    id: 'planning', num: 7, part: 'Beyond the Basics', title: 'Model-Based RL & Planning',
    mount(root) {
      RL.page.head(root, {
        num: 7, title: 'Model-based RL & planning with tree search',
        lede: 'When you can simulate the consequences of an action, you can think before you act. Play against Monte Carlo Tree Search, see its statistics, and follow the path to AlphaZero, MuZero and world models.',
        tags: [['model-based'], ['MCTS · UCT'], ['AlphaZero · MuZero · Dreamer'], ['test-time compute', 'now']],
      });
      RL.page.prose(root, intro);

      let state = TTT.empty(), human = 1, sims = 400, c = 1.4, analysis = true, perfect = false, busy = false;
      const score = { you: 0, ai: 0, draw: 0 };
      let lastAI = null;

      const lab = RL.page.lab(root, { title: 'MCTS lab' });
      const C = lab.controls;
      C.appendChild(h('div.lab-title', 'Game'));
      const br = h('div.btn-row'); C.appendChild(br);
      RL.ui.button(br, 'New game', () => newGame(), 'primary');
      RL.ui.button(br, 'AI plays my move', () => { if (!busy && TTT.winner(state) === null) aiMove(); }, 'small');
      RL.ui.segmented(C, { label: 'You play', value: '1', options: [{ value: '1', label: 'X (first)' }, { value: '-1', label: 'O (second)' }], onChange: (v) => { human = +v; newGame(); } });
      C.appendChild(h('div', { style: { height: '8px' } }));
      const g = RL.ui.group(C, 'Search');
      RL.ui.slider(g, { label: 'Simulations per move', min: 5, max: 20000, log: true, value: sims, fmt: RL.util.compact, help: 'Number of select–expand–rollout–backup iterations before choosing a move. This is the AI\'s "thinking time".', onInput: (v) => { sims = Math.round(v); refreshAnalysis(); } });
      RL.ui.slider(g, { label: 'Exploration c', min: 0, max: 3, step: 0.1, value: c, help: 'Weight of the UCB exploration bonus in the tree policy.', onInput: (v) => { c = v; refreshAnalysis(); } });
      const g2 = RL.ui.group(C, 'Display');
      RL.ui.toggle(g2, { label: 'Show MCTS analysis on my turn', value: analysis, onChange: (v) => { analysis = v; refreshAnalysis(); } });
      RL.ui.toggle(g2, { label: 'Show perfect-play values', value: perfect, help: 'Exact minimax values (W = win, D = draw, L = loss for the player to move) computed by exhaustive search.', onChange: (v) => { perfect = v; draw(); } });
      const g3 = RL.ui.group(C, 'Benchmark');
      const tourBtn = RL.ui.button(g3, 'MCTS vs. random & perfect (20 games each)', () => tournament(), 'small');
      const tourOut = h('div.small.muted', { style: { marginTop: '6px' } }); g3.appendChild(tourOut);

      const boardEl = h('div.ttt', { role: 'grid', 'aria-label': 'Tic-tac-toe board' });
      const cells = [];
      for (let i = 0; i < 9; i++) {
        const cell = h('div.ttt-cell', { role: 'gridcell', tabindex: 0, 'aria-label': 'cell ' + (i + 1) });
        cell.addEventListener('click', () => humanMove(i));
        cell.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); humanMove(i); } });
        cells.push(cell); boardEl.appendChild(cell);
      }
      RL.ui.info(lab.viz.appendChild(h('div.lab-title', 'Board')), lg([
        [sw.text('×', 'var(--pos)'), 'Blue × = X (moves first). Orange ○ = O.'],
        [sw.text('N'), 'On your turn (analysis on), each empty cell shows <b>N</b> = how many MCTS simulations started with that move.'],
        [sw.text('Q'), '<b>Q</b> = average result of those simulations for the player to move: 1 = win, ½ = draw, 0 = loss.'],
        [sw.box('var(--pos)'), 'Blue tint: share of all visits. The most visited move is MCTS’s recommendation.'],
        [sw.text('W/D/L'), 'With “perfect-play values” on: the exact game-theoretic outcome of each move (win, draw, loss).'],
        [sw.box('var(--accent-soft)'), 'Highlighted cells: the winning line.'],
      ]), 'Board');
      lab.viz.appendChild(h('div.viz-center', boardEl));
      const status = h('div.lab-status', { style: { textAlign: 'center' } }); lab.viz.appendChild(status);
      lab.viz.appendChild(h('div.small.muted', { html: 'Overlay on empty cells: <b>N</b> = visits from the root, <b>Q</b> = estimated win rate (draw = ½) for the player to move, heat = share of visits. The move with most visits is the recommendation.' }));

      RL.ui.info(lab.side.appendChild(h('div.lab-title', 'Last AI decision')), lg([
        [sw.box('var(--pos)'), 'One row per legal move of the AI’s last decision. <b>Bar length</b> = share of simulations spent on that move; the AI plays the longest bar.'],
        [sw.text('Q'), 'Estimated win rate of the move from the AI’s point of view (draw = ½). (W/D/L) = perfect-play value, if enabled.'],
      ]), 'AI decision');
      const stats = RL.ui.stats(lab.side, [['sims', 'Simulations'], ['nodes', 'Tree nodes'], ['depth', 'Max depth'], ['score', 'You · AI · draws']]);
      const decision = h('div.bars'); lab.side.appendChild(decision);
      const decNote = h('div.small.muted', { style: { marginTop: '8px' } }); lab.side.appendChild(decNote);

      let analysisRes = null;
      function newGame() { state = TTT.empty(); lastAI = null; analysisRes = null; busy = false; if (human === -1) setTimeout(aiMove, 300); refreshAnalysis(); }
      function humanMove(i) {
        if (busy || state.b[i] !== 0 || TTT.winner(state) !== null || state.toMove !== human) return;
        TTT.play(state, i); analysisRes = null; draw();
        if (TTT.winner(state) === null) { busy = true; setTimeout(aiMove, 250); } else endGame();
      }
      function aiMove() {
        const res = RL.mcts(state, { sims, c, rng: new RL.Rng() });
        lastAI = { res, before: TTT.clone(state) };
        TTT.play(state, res.best); busy = false;
        showDecision(res);
        if (TTT.winner(state) !== null) endGame();
        refreshAnalysis();
      }
      function endGame() {
        const w = TTT.winner(state);
        if (w === 0) score.draw++; else if (w === human) score.you++; else score.ai++;
        draw();
      }
      function refreshAnalysis() {
        analysisRes = null;
        if (analysis && TTT.winner(state) === null && state.toMove === human) analysisRes = RL.mcts(state, { sims, c, rng: new RL.Rng() });
        draw();
      }
      function showDecision(res) {
        decision.innerHTML = '';
        const total = res.stats.reduce((s, x) => s + x.N, 0);
        const pv = RL.ttt.moveValues(lastAI.before);
        for (const st of res.stats) {
          const r = Math.floor(st.move / 3) + 1, col = (st.move % 3) + 1;
          const lab2 = `row ${r}, col ${col}` + (perfect ? ` (${pv[st.move] > 0 ? 'W' : pv[st.move] < 0 ? 'L' : 'D'})` : '');
          decision.appendChild(h('div.bar-row', h('span', lab2), h('div.bar-track', h('div.bar-fill', { style: { width: (100 * st.N / total) + '%' } })), h('span.bar-num', 'Q ' + fmt(st.Q, 2))));
        }
        stats.sims(sims); stats.nodes(res.nodes); stats.depth(res.maxDepth);
        const best = res.stats[0];
        const optimal = pv[best.move] === Math.max(...Object.values(pv));
        decNote.innerHTML = `Chose the most visited move (${fmt(100 * best.N / total, 0)}% of visits, Q = ${fmt(best.Q, 2)}). ` +
          (optimal ? 'This move is game-theoretically optimal.' : '<b style="color:var(--bad)">Not optimal under perfect play</b> — try more simulations.');
      }
      function draw() {
        const w = TTT.winner(state), line = TTT.winLine(state);
        const pv = perfect && w === null ? RL.ttt.moveValues(state) : null;
        let totalN = 0; if (analysisRes) totalN = analysisRes.stats.reduce((s, x) => s + x.N, 0);
        cells.forEach((cell, i) => {
          cell.className = 'ttt-cell' + (state.b[i] === 1 ? ' x' : state.b[i] === -1 ? ' o' : '') + (line && line.includes(i) ? ' win' : '');
          cell.innerHTML = '';
          if (state.b[i]) { cell.textContent = state.b[i] === 1 ? '×' : '○'; return; }
          const st = analysisRes && analysisRes.stats.find((x) => x.move === i);
          if (st || pv) {
            const ov = h('div.ov');
            if (st) ov.appendChild(h('div.heat', { style: { opacity: (0.08 + 0.6 * st.N / Math.max(1, totalN)).toFixed(3) } }));
            if (st) { ov.appendChild(h('span', 'N ' + st.N)); ov.appendChild(h('span', 'Q ' + fmt(st.Q, 2))); }
            if (pv) ov.appendChild(h('span', { style: { fontWeight: 700 } }, pv[i] > 0 ? 'W' : pv[i] < 0 ? 'L' : 'D'));
            cell.appendChild(ov);
          }
        });
        stats.score(`${score.you} · ${score.ai} · ${score.draw}`);
        status.textContent = w === null
          ? (state.toMove === human ? 'Your move (' + (human === 1 ? '×' : '○') + ').' : 'AI is thinking…')
          : w === 0 ? 'Draw. (With enough simulations, MCTS never loses.)' : w === human ? 'You win! Try a larger simulation budget.' : 'AI wins.';
      }
      function tournament() {
        tourBtn.disabled = true; tourOut.textContent = 'Playing…';
        setTimeout(() => {
          const rng = new RL.Rng();
          const play = (opp) => {
            const res = { w: 0, d: 0, l: 0 };
            for (let gi = 0; gi < 20; gi++) {
              const st = TTT.empty(); const me = gi % 2 === 0 ? 1 : -1;
              while (TTT.winner(st) === null) {
                let m;
                if (st.toMove === me) m = RL.mcts(st, { sims, c, rng }).best;
                else if (opp === 'random') { const ms = TTT.moves(st); m = ms[rng.int(ms.length)]; }
                else { const v = RL.ttt.moveValues(st); const best = Math.max(...Object.values(v)); const ms = Object.keys(v).filter((k) => v[k] === best).map(Number); m = ms[rng.int(ms.length)]; }
                TTT.play(st, m);
              }
              const w = TTT.winner(st); if (w === 0) res.d++; else if (w === me) res.w++; else res.l++;
            }
            return res;
          };
          const r1 = play('random'), r2 = play('perfect');
          tourOut.innerHTML = `With ${RL.util.compact(sims)} simulations — vs random: <b>${r1.w}W ${r1.d}D ${r1.l}L</b>; vs perfect player: <b>${r2.w}W ${r2.d}D ${r2.l}L</b> (best possible: all draws).`;
          tourBtn.disabled = false;
        }, 30);
      }
      newGame();
      stats.score('0 · 0 · 0');
      RL.page.prose(root, theory);
      return () => { };
    },
  });
})(window.RL);
