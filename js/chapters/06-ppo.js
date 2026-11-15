/* Chapter 6: trust regions, TRPO, PPO-clip. */
(function (RL) {
  'use strict';
  const sw = RL.ui.sw, lg = RL.ui.lg;
  const { h, fmt, setupCanvas, cssVar, softmax } = RL.util;

  const theory1 = String.raw`
<h2>The step-size problem</h2>
<p>In supervised learning a bad gradient step costs a bit of loss and the next step repairs it. In policy optimization a bad step changes <em>the data you will collect next</em>. If the policy suddenly becomes almost deterministic in some state, it stops trying the alternatives and the gradient estimates there become useless — the collapse you could provoke in Chapter 5. Worse, the size of a step in parameter space says little about the size of the change in behavior: the same $\Delta\theta$ may barely move one policy and completely flip another.</p>
<p>The fix is to measure and limit the step in <b>policy space</b>, typically with the KL divergence between the old and new action distributions.</p>

<h2>The surrogate objective</h2>
<p>Let $\pi_\text{old}$ be the policy that collected the batch. By the performance-difference lemma, $J(\pi) - J(\pi_\text{old}) = \frac{1}{1-\gamma}\mathbb E_{s\sim d^{\pi},\, a\sim\pi}[A^{\pi_\text{old}}(s,a)]$. We cannot sample from $d^\pi$ for a policy we have not run yet, so we substitute the old state distribution and correct the action distribution with an importance ratio:</p>
$$L_{\pi_\text{old}}(\theta) = \mathbb E_{s,a\sim\pi_\text{old}}\Big[\, r_t(\theta)\, \hat A_t \Big], \qquad r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_\text{old}(a_t\mid s_t)} .$$
<p>At $\theta = \theta_\text{old}$ this surrogate has exactly the policy gradient as its gradient ($\nabla r = \nabla\log\pi$ when $r=1$). Away from it, the state-distribution mismatch grows; the surrogate is only trustworthy <em>near</em> $\pi_\text{old}$.</p>

<details class="deep math"><summary><span class="lbl">Math</span> The TRPO bound and the natural gradient</summary><div class="body">
<p>Kakade & Langford (2002) and Schulman et al. (2015) bound the error of the surrogate:</p>
$$J(\pi) \;\ge\; J(\pi_\text{old}) + L'_{\pi_\text{old}}(\pi) - C \cdot \max_s \mathrm{KL}\big(\pi_\text{old}(\cdot|s)\,\|\,\pi(\cdot|s)\big), \qquad C = \frac{4\epsilon\gamma}{(1-\gamma)^2},$$
<p>with $L'$ the surrogate improvement and $\epsilon = \max_{s,a}|A^{\pi_\text{old}}(s,a)|$. Maximizing the right-hand side guarantees monotonic improvement (a minorize–maximize scheme). The penalty coefficient is far too conservative in practice, so <b>TRPO</b> uses a constraint instead:</p>
$$\max_\theta\ L_{\pi_\text{old}}(\theta) \quad \text{s.t.}\quad \bar{\mathrm{KL}}(\theta_\text{old}, \theta) = \mathbb E_s\big[\mathrm{KL}(\pi_\text{old}(\cdot|s)\|\pi_\theta(\cdot|s))\big] \le \delta .$$
<p>Linearizing the objective ($g = \nabla L$) and taking the second-order expansion of the KL, $\bar{\mathrm{KL}} \approx \tfrac12\Delta\theta^\top F\Delta\theta$ with the Fisher information $F = \mathbb E[\nabla\log\pi\,\nabla\log\pi^\top]$, gives the <b>natural gradient</b> step</p>
$$\Delta\theta = \sqrt{\frac{2\delta}{g^\top F^{-1}g}}\;F^{-1}g ,$$
<p>computed with conjugate gradient (Fisher-vector products only) and a backtracking line search that checks the actual KL and improvement. The natural gradient is invariant to how the policy is parametrized — it measures steps in distribution space. It works well, but the second-order machinery is awkward with large networks, shared actor-critic trunks, dropout, or recurrent policies.</p>
</div></details>

<h2>PPO: clip the incentive instead of constraining the step</h2>
<p>Proximal Policy Optimization (Schulman et al., 2017) keeps the spirit of TRPO with only first-order SGD. It maximizes a <em>clipped</em> surrogate:</p>
$$L^{\text{CLIP}}(\theta) = \mathbb E_t\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big].$$
<p>Read it case by case (the plot below makes it visual):</p>
<ul>
<li>$\hat A_t &gt; 0$ (good action): increasing its probability helps, but once $r_t &gt; 1+\epsilon$ the objective is flat — <b>no further incentive</b> to push it up within this batch.</li>
<li>$\hat A_t &lt; 0$ (bad action): decreasing its probability helps until $r_t &lt; 1-\epsilon$; then the objective is flat.</li>
<li>The $\min$ makes the objective a <b>pessimistic lower bound</b> of the unclipped surrogate: if an update has accidentally moved a ratio the <em>wrong</em> way (e.g. $r_t &gt; 1+\epsilon$ for a negative advantage), the unclipped term is kept and the gradient pulls it back.</li>
</ul>
`;

  const theory2 = String.raw`
<p>Because the clipped objective removes the incentive to move far, PPO can safely take <b>several epochs</b> of minibatch SGD on each batch — far better sample use than A2C's single step. The full loss adds a value-function term and an entropy bonus: $\;\mathcal L = -L^{\text{CLIP}} + c_1 (V_\phi(s) - \hat R)^2 - c_2\,\mathcal H(\pi_\theta)$.</p>
<div class="callout note"><strong>Clipping is not a hard trust region.</strong> It removes the gradient of individual samples once they leave $[1-\epsilon,1+\epsilon]$, but the shared network still moves through other samples' gradients, so the KL is not strictly bounded (Engstrom et al., 2020; Wang et al., 2020). In practice one monitors the approximate KL and stops the epoch loop early if it exceeds a target such as 0.02. The lab shows both the approximate KL and the fraction of clipped samples.</div>

<details class="deep practice"><summary><span class="lbl">Practice</span> The implementation details that make PPO work</summary><div class="body">
<p>Much of PPO's performance comes from details beyond the clipped objective (Engstrom et al., 2020, “Implementation Matters”; Huang et al., 2022, “The 37 implementation details of PPO”):</p>
<ul>
<li><b>Advantage normalization</b> per minibatch/batch (zero mean, unit std).</li>
<li><b>GAE</b> with $\lambda\approx0.95$ and a learned value baseline; value targets $\hat R_t = \hat A_t + V_\text{old}(s_t)$.</li>
<li><b>Gradient-norm clipping</b> (≈0.5), Adam with learning-rate annealing, orthogonal initialization with a small final policy layer (near-uniform initial policy).</li>
<li><b>Observation normalization</b> (running mean/std) and reward scaling for unnormalized environments.</li>
<li><b>Many parallel environments</b>, rollouts of 128–2048 steps each; 3–10 epochs; 4–32 minibatches.</li>
<li>Healthy diagnostics: approx. KL ≈ 0.005–0.03 per update, clip fraction ≈ 0.05–0.3, entropy decaying slowly. Explained variance of the critic rising toward 1.</li>
</ul>
<p>The KL estimator in the lab is $\hat{\mathrm{KL}} = \mathbb E[(r-1) - \log r]$, which is unbiased for $\mathrm{KL}(\pi_\text{old}\|\pi)$ and always non-negative — the same "k3" estimator used in GRPO's KL penalty (Chapter 8).</p>
</div></details>

<h2>Why PPO became the default</h2>
<p>PPO is first-order, works with any architecture, tolerates a wide range of hyperparameters, and scales trivially by adding parallel environments. It trained OpenAI Five (Dota 2), most sim-to-real locomotion policies (thousands of simulated robots in parallel on GPUs), and the original RLHF pipeline of InstructGPT/ChatGPT. Its main weaknesses: on-policy data hunger (fine in simulation, painful in the real world — where SAC and model-based methods are preferred), and for LLMs the memory cost of a critic as large as the policy — which motivated critic-free variants such as GRPO and RLOO.</p>
<details class="deep history"><summary><span class="lbl">History</span> Relatives of PPO</summary><div class="body">
<p><b>TRPO</b> (2015) — the principled ancestor, still cited for its theory; rarely used in new code. <b>PPO-penalty</b> replaces the clip with an adaptive KL penalty $\beta\,\mathrm{KL}$ (β grows when KL is too large) — less common, but the KL-penalty form returns in RLHF. <b>ACKTR</b> used a Kronecker-factored approximation of the natural gradient. <b>MPO / V-MPO</b> (DeepMind) solve a KL-regularized policy improvement step via an EM-like procedure and are strong but less widespread. <b>IMPALA</b> with V-trace is the standard for massively distributed off-policy-corrected actor-critic training.</p>
</div></details>
`;

  /* ---------- Widget: the clipped objective ---------- */
  function clipWidget(root) {
    const w = RL.page.widget(root, 'The clipped surrogate, one sample at a time',
      'Objective contribution of one sample as a function of the probability ratio $r = \\pi_\\theta(a|s)/\\pi_\\text{old}(a|s)$. Dashed: unclipped $rA$. Solid: PPO\'s $\\min(rA, \\operatorname{clip}(r)A)$. Shaded: region where this sample contributes <b>zero gradient</b>.');
    RL.ui.info(w.querySelector('.widget-title'), lg([
      [sw.dashed('var(--muted)'), '<b>Dashed</b>: unclipped surrogate r·A. It keeps rewarding larger moves forever.'],
      [sw.line('var(--accent)'), '<b>Teal</b>: PPO’s objective min(r·A, clip(r, 1−ε, 1+ε)·A).'],
      [sw.band('var(--neg)'), '<b>Shaded</b>: ratios where this sample’s gradient is exactly zero: no further incentive to move the policy.'],
      [sw.vline('var(--muted)'), 'Dotted verticals: the clip range 1 − ε and 1 + ε.'],
      [sw.text('r'), 'x-axis: probability ratio r = π<sub>θ</sub>(a|s) / π<sub>old</sub>(a|s) (1 = unchanged). y-axis: objective contribution of the sample.'],
    ]), 'Clipped objective');
    const ctlRow = h('div', { style: { maxWidth: '320px' } }); w.appendChild(ctlRow);
    let eps = 0.2;
    RL.ui.slider(ctlRow, { label: 'Clip range ε', min: 0.02, max: 0.8, step: 0.01, value: eps, onInput: (v) => { eps = v; draw(); } });
    const row = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' } }); w.appendChild(row);
    const panels = [1, -1].map((A) => { const c = h('canvas'); const d = h('div', h('div.chart-title', A > 0 ? 'Advantage A > 0 (good action)' : 'Advantage A < 0 (bad action)'), c); row.appendChild(d); return { A, c, d }; });
    function draw() {
      for (const { A, c, d } of panels) {
        const W = Math.max(220, d.clientWidth || 300), H = 190;
        const ctx = setupCanvas(c, W, H);
        const padL = 30, padB = 22, pw = W - padL - 8, ph = H - padB - 8;
        const X = (r) => padL + (r / 2) * pw, Y = (y) => 8 + (1 - (y + 2) / 4) * ph;
        ctx.clearRect(0, 0, W, H);
        // zero-gradient region
        ctx.fillStyle = cssVar('--neg'); ctx.globalAlpha = 0.12;
        if (A > 0) ctx.fillRect(X(1 + eps), 8, X(2) - X(1 + eps), ph); else ctx.fillRect(X(0), 8, X(1 - eps) - X(0), ph);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = cssVar('--grid'); ctx.lineWidth = 1;
        for (let y = -2; y <= 2; y++) { ctx.beginPath(); ctx.moveTo(padL, Y(y)); ctx.lineTo(padL + pw, Y(y)); ctx.stroke(); }
        ctx.fillStyle = cssVar('--muted'); ctx.font = '11px system-ui'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (let y = -2; y <= 2; y++) ctx.fillText(String(y).replace('-', '−'), padL - 5, Y(y));
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        for (const r of [0, 0.5, 1, 1.5, 2]) ctx.fillText(String(r), X(r), 8 + ph + 4);
        ctx.setLineDash([3, 3]); ctx.strokeStyle = cssVar('--muted');
        for (const r of [1 - eps, 1 + eps]) { ctx.beginPath(); ctx.moveTo(X(r), 8); ctx.lineTo(X(r), 8 + ph); ctx.stroke(); }
        ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(2), Y(2 * A)); ctx.lineWidth = 1.5; ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 3; ctx.beginPath();
        for (let i = 0; i <= 200; i++) {
          const r = (2 * i) / 200; const cl = Math.max(1 - eps, Math.min(1 + eps, r));
          const y = Math.min(r * A, cl * A);
          if (i === 0) ctx.moveTo(X(r), Y(y)); else ctx.lineTo(X(r), Y(y));
        }
        ctx.stroke();
        ctx.fillStyle = cssVar('--text'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(A > 0 ? 'flat: no incentive beyond 1+ε' : 'flat: no incentive below 1−ε', A > 0 ? X(1 + eps) + 4 : X(0) + 4, 12);
      }
    }
    draw();
    RL.onTheme(draw);
    let ro; if (window.ResizeObserver) { ro = new ResizeObserver(draw); ro.observe(row); }
    RL.util.renderMath(w);
    return () => { if (ro) ro.disconnect(); };
  }

  /* ---------- Widget: many epochs on one batch ---------- */
  function epochsWidget(root) {
    const w = RL.page.widget(root, 'Why clipping allows many epochs on one batch',
      'One state, three actions, a fixed batch collected by $\\pi_\\text{old}$ (uniform) with advantages $A = (+1, 0, -1)$. We run many gradient steps on the <em>same</em> batch, maximizing either the plain surrogate $\\mathbb E[rA]$ or PPO\'s clipped one.');
    const grid = h('div.widget-grid'); w.appendChild(grid);
    const left = h('div'), right = h('div'); grid.appendChild(left); grid.appendChild(right);
    RL.ui.info(w.querySelector('.widget-title'), lg([
      [sw.line('var(--neg)'), '<b>Orange</b>: maximizing the plain surrogate E[r·A] on the same batch again and again.'],
      [sw.line('var(--accent)'), '<b>Teal</b>: maximizing PPO’s clipped surrogate on the same batch.'],
      [sw.text('π'), 'Top chart: probability of the best action (advantage +1). Bottom chart: KL divergence from the policy that collected the batch, i.e. how far the policy has drifted.'],
    ]), 'Many epochs on one batch');
    const p = { eps: 0.2, lr: 0.3, steps: 60 };
    RL.ui.slider(left, { label: 'Clip range ε', min: 0.05, max: 0.8, step: 0.01, value: p.eps, onInput: (v) => { p.eps = v; run(); } });
    RL.ui.slider(left, { label: 'Step size', min: 0.05, max: 2, step: 0.05, value: p.lr, onInput: (v) => { p.lr = v; run(); } });
    RL.ui.slider(left, { label: 'Gradient steps on the batch', min: 5, max: 200, step: 5, value: p.steps, onInput: (v) => { p.steps = v; run(); } });
    const out = h('div.small', { style: { marginTop: '8px' } }); left.appendChild(out);
    const ch1 = new RL.LineChart(right, { title: 'π(best action) — starts at 1/3', info: 'Probability of the action with advantage +1 after each gradient step on the same fixed batch.', height: 150, yMin: 0, yMax: 1, xLabel: 'gradient step on the same batch' });
    ch1.addSeries('plain surrogate', '--neg'); ch1.addSeries('PPO clipped', '--accent');
    const ch2 = new RL.LineChart(right, { title: 'KL(π_old ‖ π)', info: 'How far the updated policy has moved from the policy that collected the batch.', height: 130, xLabel: 'gradient step', yMin: 0 });
    ch2.addSeries('plain surrogate', '--neg'); ch2.addSeries('PPO clipped', '--accent');
    const A = [1, 0, -1], old = [1 / 3, 1 / 3, 1 / 3];
    function simulate(clipped) {
      const z = [0, 0, 0]; const best = [], kls = [];
      for (let t = 0; t <= p.steps; t++) {
        const pi = softmax(z);
        best.push(pi[0]);
        kls.push(old.reduce((s, q, i) => s + q * Math.log(q / pi[i]), 0));
        const g = [0, 0, 0];
        for (let a = 0; a < 3; a++) {
          const r = pi[a] / old[a];
          const active = !clipped || (A[a] >= 0 ? r <= 1 + p.eps : r >= 1 - p.eps);
          if (!active || A[a] === 0) continue;
          for (let j = 0; j < 3; j++) g[j] += old[a] * A[a] * r * ((a === j ? 1 : 0) - pi[j]);
        }
        for (let j = 0; j < 3; j++) z[j] += p.lr * g[j] * 3;
      }
      return { best, kls };
    }
    function run() {
      const u = simulate(false), c = simulate(true);
      ch1.set('plain surrogate', u.best); ch1.set('PPO clipped', c.best);
      ch2.set('plain surrogate', u.kls); ch2.set('PPO clipped', c.kls);
      out.innerHTML = `After ${p.steps} steps — plain: π(best) = <b>${fmt(u.best[p.steps], 3)}</b>, KL = ${fmt(u.kls[p.steps], 3)}; ` +
        `clipped: π(best) = <b>${fmt(c.best[p.steps], 3)}</b>, KL = ${fmt(c.kls[p.steps], 3)}. ` +
        `The plain surrogate keeps extrapolating from a batch that says nothing about the new policy; the clipped one stops pushing each action once its ratio leaves $[1-\\varepsilon, 1+\\varepsilon]$ in the direction its advantage favours, so the policy settles close to $\\pi_\\text{old}$ no matter how many steps you take.`;
      RL.util.renderMath(out);
    }
    run();
    RL.util.renderMath(w);
    return () => { ch1.destroy(); ch2.destroy(); };
  }

  RL.chapters.push({
    id: 'ppo', num: 6, part: 'Policy Optimization', title: 'Trust Regions & PPO',
    mount(root) {
      RL.page.head(root, {
        num: 6, title: 'Trust regions & Proximal Policy Optimization',
        lede: 'How far should one update move a policy? From the TRPO bound and natural gradients to PPO’s clipped objective — the most widely used deep RL algorithm, from robot locomotion to RLHF.',
        tags: [['actor-critic'], ['on-policy'], ['TRPO · PPO'], ['industry default', 'now']],
      });
      const cleanups = [];
      RL.page.prose(root, theory1);
      cleanups.push(clipWidget(root));
      cleanups.push(epochsWidget(root));
      RL.page.prose(root, String.raw`<h2>Lab: PPO on CartPole</h2><p>The full algorithm: collect $T$ steps, compute GAE advantages, normalize them, then run $K$ epochs of minibatch SGD on the clipped objective (plus a value loss). Compare with A2C, which takes a single gradient step per batch.</p>`);
      const lab = RL.cartpoleLab(root, {
        kind: 'pg', algos: ['ppo', 'a2c', 'reinforce_b', 'reinforce'], title: 'PPO lab',
        defaults: { algo: 'ppo', lr: 3e-4, lrV: 1e-3, rollout: 1024, epochs: 4, minibatch: 64, clip: 0.2, normAdv: true, lam: 0.95 },
        vizNote: 'Watch the PPO diagnostics: approx. KL stays small (≈ 0.01) and a fraction of samples hits the clip each update. Increase ε or the epochs and see both grow.',
      });
      cleanups.push(() => lab.destroy());
      const base = { algo: 'ppo', lr: 3e-4, lrV: 1e-3, gamma: 0.99, lam: 0.95, entCoef: 0, rollout: 1024, epochs: 4, minibatch: 64, clip: 0.2, normAdv: true, width: 64 };
      const c = (o) => Object.assign({}, base, o);
      RL.ui.experiments(root, [
        { title: 'Default PPO', text: 'ε = 0.2, 4 epochs, T = 1024, GAE(0.95). Expect the 500 ceiling after roughly 10–20k steps, reliably across seeds — compare with the fragility of DQN and REINFORCE.', apply: () => lab.configure(c({}), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Remove the clip, keep the epochs', text: 'ε = 1 (effectively no clipping for positive advantages) with 20 epochs per batch. The policy over-fits each batch: KL spikes, entropy collapses, and training becomes erratic.', apply: () => lab.configure(c({ clip: 1, epochs: 20 }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Clip protects a large learning rate', text: 'Learning rate $3\\cdot10^{-3}$ (10×) with ε = 0.2. Clipping limits the damage; now try the same learning rate with ε = 1.', apply: () => lab.configure(c({ lr: 3e-3 }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'One epoch = expensive A2C', text: '1 epoch per batch. Each sample is used once, so learning per environment step is much slower. More epochs are where PPO’s sample efficiency comes from.', apply: () => lab.configure(c({ epochs: 1 }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Unnormalized advantages', text: 'Turn advantage normalization off. With returns of ~100 and an untrained critic the effective step size changes over training. Often still works on CartPole — but notice the different speed and KL profile.', apply: () => lab.configure(c({ normAdv: false }), { speedKey: 'max' }), labEl: lab.lab.el },
      ]);
      RL.page.prose(root, theory2);
      return () => cleanups.forEach((f) => f());
    },
  });
})(window.RL);
