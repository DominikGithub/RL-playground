/* Chapter 5: policy gradients, baselines, actor-critic, GAE, and value- vs policy-based methods. */
(function (RL) {
  'use strict';
  const sw = RL.ui.sw, lg = RL.ui.lg;
  const { h, fmt, setupCanvas, cssVar } = RL.util;

  const why = String.raw`
<h2>Why optimize the policy directly?</h2>
<p>Value-based methods learn $Q$ and derive the policy implicitly via $\arg\max$. Policy-based methods parametrize $\pi_\theta(a\mid s)$ — a softmax over logits for discrete actions, a Gaussian for continuous ones, a transformer over tokens for language — and do gradient ascent on the objective we actually care about, $J(\theta) = \mathbb E_{\pi_\theta}[G_0]$. Reasons to do so:</p>
<ul>
<li><b>Any action space.</b> Continuous torques, combinatorial choices, sequences of tokens: we only need to sample from $\pi_\theta$ and evaluate $\log\pi_\theta(a\mid s)$, never to maximize over actions.</li>
<li><b>Smooth improvement.</b> A small change of $\theta$ changes behavior a little. A small change of $Q$ can flip an $\arg\max$ and change behavior abruptly.</li>
<li><b>Stochastic policies can be optimal</b> when the agent cannot distinguish states (partial observability, function approximation). The widget below is Sutton & Barto's Example 13.1.</li>
<li><b>Priors.</b> If you already have a good policy — e.g. a pretrained language model — you can start from it and improve it. There is no comparable way to "initialize" a Q-function from a generative model.</li>
</ul>`;

  const theory = String.raw`
<h2>The policy gradient theorem</h2>
<p>Let $\tau = (s_0,a_0,r_1,s_1,\dots)$ be a trajectory and $R(\tau)$ its return. Its probability under $\pi_\theta$ is $p_\theta(\tau) = p(s_0)\prod_t \pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)$. Using the <b>log-derivative trick</b> $\nabla p = p\,\nabla \log p$:</p>
$$\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau)\,d\tau = \mathbb E_{\tau\sim p_\theta}\big[R(\tau)\,\nabla_\theta \log p_\theta(\tau)\big] = \mathbb E_{\tau}\Big[R(\tau) \sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big].$$
<p>The environment dynamics $p(s_{t+1}\mid s_t,a_t)$ do not depend on $\theta$ and drop out of the gradient of the log: <b>we can estimate the gradient from samples without any model</b>. This "score-function" or likelihood-ratio estimator is the foundation of REINFORCE (Williams, 1992), PPO and LLM fine-tuning alike.</p>

<details class="deep math"><summary><span class="lbl">Math</span> From the trajectory form to the policy gradient theorem</summary><div class="body">
<p><b>Causality.</b> The action at time $t$ cannot influence rewards received before $t$. Since $\mathbb E[r_{k}\,\nabla\log\pi(a_t\mid s_t)] = 0$ for $k \le t$ (by the same argument as for baselines below), we may replace $R(\tau)$ by the <em>reward-to-go</em> $G_t$:</p>
$$\nabla_\theta J = \mathbb E\Big[\sum_t \gamma^t\, G_t\, \nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big], \qquad \mathbb E[G_t\mid s_t,a_t] = Q^{\pi}(s_t,a_t).$$
<p><b>Policy gradient theorem</b> (Sutton et al., 2000). Writing $d^\pi(s) = (1-\gamma)\sum_t \gamma^t \Pr(s_t = s)$ for the discounted state-visitation distribution:</p>
$$\nabla_\theta J(\theta) = \frac{1}{1-\gamma}\,\mathbb E_{s\sim d^{\pi}, a\sim \pi_\theta}\big[Q^{\pi}(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)\big].$$
<p>Remarkably, the gradient does not involve $\nabla_\theta d^\pi(s)$ — how the state distribution changes with the policy — even though $J$ depends on it. A direct proof unrolls $\nabla V^\pi(s) = \sum_a \big[\nabla\pi(a|s) Q^\pi(s,a) + \pi(a|s)\,\gamma\sum_{s'}p(s'|s,a)\nabla V^\pi(s')\big]$ recursively.</p>
<p><b>Practice note:</b> almost all implementations drop the $\gamma^t$ factor in front of each term. The resulting update is not exactly the gradient of the discounted objective (Thomas, 2014; Nota & Thomas, 2020), but it works better in practice, essentially because $\gamma$ is used as a variance-reduction device rather than as the true objective.</p>
<p><b>Score functions</b> of common policies:</p>
<ul>
<li>Softmax over logits $z$: $\nabla_z \log\pi(a) = e_a - \pi$ (one-hot minus probabilities) — exactly the gradient-bandit update of Chapter 2.</li>
<li>Gaussian $\mathcal N(\mu_\theta(s), \sigma^2)$: $\nabla_\mu\log\pi(a\mid s) = (a-\mu)/\sigma^2$.</li>
<li>Autoregressive token policy: $\log\pi(y\mid x) = \sum_t \log\pi(y_t\mid x, y_{\lt t})$, so the score is the sum of per-token scores.</li>
</ul>
</div></details>

<h2>Variance reduction: baselines and advantages</h2>
<p>The vanilla estimator is unbiased but extremely noisy: in CartPole every return is positive, so every action taken gets reinforced — good ones just slightly more. Subtracting a <b>baseline</b> $b(s)$ fixes this without introducing bias:</p>
$$\mathbb E_{a\sim\pi}\big[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\big] = b(s)\sum_a \pi_\theta(a\mid s)\frac{\nabla\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)} = b(s)\,\nabla_\theta \underbrace{\sum_a\pi_\theta(a\mid s)}_{=1} = 0.$$
<p>The natural choice is $b(s) = V^\pi(s)$, which turns $Q^\pi$ into the <b>advantage</b> $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$: "how much better than usual was this action?" Positive advantages make an action more likely, negative ones less likely. Experiment with the estimator yourself:</p>`;

  const theory2 = String.raw`
<h2>Actor-critic and generalized advantage estimation</h2>
<p>An <b>actor-critic</b> method learns a critic $V_\phi(s)$ by TD or regression and uses it to estimate advantages for the actor. The choice of estimator is a bias–variance dial:</p>
$$\hat A_t^{\text{MC}} = G_t - V_\phi(s_t), \qquad \hat A_t^{\text{TD}} = \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t), \qquad \hat A_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k+1} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t).$$
<p><b>Generalized Advantage Estimation</b> (Schulman et al., 2016) takes an exponentially weighted average of all $n$-step estimators, which collapses to a discounted sum of TD errors:</p>
$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{t+l}, \qquad \hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}\ \ \text{(computed backwards over a rollout)}.$$
<details class="deep math"><summary><span class="lbl">Math</span> Why GAE telescopes, and what λ trades off</summary><div class="body">
<p>Each $n$-step advantage is a telescoping sum of TD errors: $\hat A^{(n)}_t = \sum_{l=0}^{n-1}\gamma^l\delta_{t+l}$ (the intermediate $V$ terms cancel). Weighting them by $(1-\lambda)\lambda^{n-1}$ and exchanging the sums gives $\sum_l (\gamma\lambda)^l\delta_{t+l}$. With $\lambda = 0$ the estimate is $\delta_t$: low variance, but biased whenever $V_\phi \ne V^\pi$. With $\lambda=1$ it is $G_t - V_\phi(s_t)$: unbiased for any $V_\phi$, but with Monte Carlo variance. The default $\lambda\approx 0.95$, $\gamma = 0.99$ is the most-copied pair of hyperparameters in deep RL. GAE is TD(λ) applied to advantages.</p>
</div></details>
<p>The <b>A2C</b> agent in the lab collects $n$ steps, computes GAE advantages, takes one gradient step on the actor loss $-\hat A_t\log\pi_\theta(a_t|s_t)$ and one on the critic loss $(V_\phi(s_t) - \hat R_t)^2$, and repeats. An <b>entropy bonus</b> $c\,\mathcal H(\pi_\theta(\cdot|s))$ discourages premature collapse to a deterministic policy.</p>
<details class="deep practice"><summary><span class="lbl">Practice</span> Things that matter when training policy-gradient agents</summary><div class="body">
<ul>
<li><b>Learning rate is the most dangerous hyperparameter.</b> A step that is too big makes the policy nearly deterministic in the wrong place; then it stops exploring, the data becomes uninformative, and it cannot recover. Watch entropy: a sudden drop to ~0 with low return is policy collapse. This is the motivation for trust regions (Chapter 6).</li>
<li><b>Many environments in parallel</b> (A2C's "synchronous" part) decorrelate the batch; the lab uses a single environment for clarity, which makes A2C noisier than in practice.</li>
<li><b>Advantage normalization</b> per batch (zero mean, unit variance) is a cheap and very effective stabilizer — it also makes the learning rate less sensitive to reward scale.</li>
<li><b>Shared vs. separate networks.</b> Sharing a trunk between actor and critic saves compute but couples their gradients; the lab uses separate networks.</li>
</ul>
</div></details>

<h2>Value-based vs. policy-based: the full comparison</h2>
<p>Both families solve the same problem, but they optimize different things. A value method minimizes a <em>Bellman error</em> and hopes that a good $Q$ implies a good greedy policy; a policy method ascends $J$ itself and only uses values (if at all) to reduce variance.</p>
<div class="table-wrap"><table class="compare">
<tr><th></th><th>Value-based (Q-learning, DQN)</th><th>Policy-based (REINFORCE, A2C, PPO)</th></tr>
<tr><td>Learns</td><td>$Q_\theta(s,a) \approx Q^*$; policy implicit: $\arg\max_a Q$</td><td>$\pi_\theta(a\mid s)$ explicitly; optionally a critic $V_\phi$</td></tr>
<tr><td>Objective</td><td>Squared TD error $\big(r + \gamma\max Q(s',\cdot) - Q(s,a)\big)^2$ — a surrogate</td><td>$J(\theta)$ itself, via unbiased (or low-bias) gradient estimates</td></tr>
<tr><td>Improvement step</td><td>"hard": greedy $\arg\max$ (policy iteration-like)</td><td>"soft": small step toward higher advantage</td></tr>
<tr><td>Data</td><td>off-policy: replay buffers, reuse old data, offline data</td><td>on-policy: fresh samples per update (PPO reuses a few epochs)</td></tr>
<tr><td>Sample efficiency</td><td>higher</td><td>lower — but trivially parallel and cheap per sample in simulation</td></tr>
<tr><td>Bias / variance</td><td>low variance, biased bootstrap targets</td><td>high variance; bias only through the critic</td></tr>
<tr><td>Stability with neural nets</td><td>deadly triad; no convergence guarantee</td><td>converges to a local optimum under standard SGD conditions; can collapse with large steps</td></tr>
<tr><td>Action spaces</td><td>discrete and small (needs $\max_a$)</td><td>anything you can sample and score: continuous, structured, token sequences</td></tr>
<tr><td>Stochastic policies</td><td>only via heuristics (ε, Boltzmann)</td><td>native; can represent stochastic optima</td></tr>
<tr><td>Typical use today</td><td>discrete control, offline RL, critics inside actor-critic</td><td>robotics in simulation, games, RLHF and reasoning-model training</td></tr>
</table></div>

<details class="deep math"><summary><span class="lbl">Math</span> How small value errors become large policy errors — and how the two views meet</summary><div class="body">
<p><b>From value error to policy loss.</b> If $\pi$ is greedy with respect to an approximation $\hat Q$, then (Singh & Yee, 1994)</p>
$$\|V^* - V^{\pi}\|_\infty \le \frac{2\gamma}{1-\gamma}\,\|\hat Q - Q^*\|_\infty .$$
<p>The bound is in the sup-norm: a single badly estimated state–action pair can make the greedy policy bad, while regression minimizes an <em>average</em> error weighted by the data distribution. This mismatch between what is optimized (mean squared TD error) and what matters (policy return) is a core reason why value methods can be brittle with function approximation.</p>
<p><b>Policy methods optimize the right thing, locally.</b> Gradient ascent on $J$ with an unbiased gradient estimate and Robbins–Monro step sizes converges to a stationary point of $J$ — with any differentiable parametrization (Sutton et al., 2000, even give "compatible" critic features for which using $\hat Q_w$ introduces no bias). The price: only local optima, and variance that makes each step expensive in samples.</p>
<p><b>They are closer than they look.</b> Add an entropy term to the objective, $J_\text{soft}(\pi) = \mathbb E\big[\sum_t \gamma^t (r_t + \alpha\,\mathcal H(\pi(\cdot|s_t)))\big]$. Its optimal policy is a Boltzmann distribution of the soft-optimal Q-function,</p>
$$\pi^*(a\mid s) = \exp\big((Q^*_\text{soft}(s,a) - V^*_\text{soft}(s))/\alpha\big),\qquad V^*_\text{soft}(s) = \alpha \log\sum_a \exp(Q^*_\text{soft}(s,a)/\alpha),$$
<p>and soft Q-learning and entropy-regularized policy gradient perform equivalent updates (Schulman, Chen & Abbeel, 2017). As $\alpha\to 0$ we recover the hard max of Q-learning. Soft Actor-Critic (SAC) lives exactly at this junction — and the same closed form, with a KL to a reference policy instead of entropy, is the starting point of DPO for language models (Chapter 8).</p>
</div></details>
<details class="deep history"><summary><span class="lbl">History</span> A3C, natural gradients and other milestones</summary><div class="body">
<p><b>REINFORCE</b> (Williams, 1992) introduced the likelihood-ratio gradient to RL; <b>actor-critic</b> architectures go back to Barto, Sutton & Anderson (1983) — the same paper that introduced cart-pole. <b>Natural policy gradients</b> (Kakade, 2001) precondition the gradient with the inverse Fisher information so that steps are measured in distribution space rather than parameter space — the theoretical basis of TRPO (Chapter 6). <b>A3C</b> (Mnih et al., 2016) ran asynchronous actor-learners on CPU threads; the synchronous, batched variant A2C turned out to be just as good and simpler, and both were largely superseded by PPO. Vanilla REINFORCE without baseline is today mostly a teaching device — though variants with better baselines (RLOO, GRPO) made a comeback in LLM training.</p>
</div></details>
`;

  /* ---------- Widget: aliased short corridor ---------- */
  function corridorWidget(root) {
    const w = RL.page.widget(root, 'When the optimal policy must be stochastic',
      'Three cells look identical to the agent (same features), so it must use the same action probabilities everywhere. In the middle cell the actions are <b>reversed</b>. Reward −1 per step until the goal. Choose $p = \\Pr(\\text{right})$.');
    RL.ui.info(w.querySelector('.widget-title'), lg([
      [sw.box('var(--panel2)'), 'Grey cells: three states the agent <b>cannot tell apart</b>; it must use the same p everywhere. In “state 2” left and right are swapped.'],
      [sw.line('var(--accent)'), '<b>Curve</b>: expected return J(p) for every p (reward −1 per step, so higher = fewer steps).'],
      [sw.dot('var(--neg)'), '<b>Orange dots</b>: ε-greedy (ε = 0.1) around “left” and “right”, and the best stochastic policy p ≈ 0.59.'],
      [sw.vline('var(--text)'), 'Dashed vertical line: your current p (slider).'],
    ]), 'Aliased corridor');
    const svg = h('div', {
      html: String.raw`<svg viewBox="0 0 520 70" style="width:100%;max-width:520px;display:block;margin:6px 0 10px" role="img" aria-label="Short corridor with switched actions">
      ${[0, 1, 2].map((i) => `<rect x="${10 + i * 125}" y="10" width="115" height="48" rx="8" style="fill:var(--panel2);stroke:var(--border)"/>
      <text x="${67 + i * 125}" y="32" text-anchor="middle" style="fill:var(--text);font:600 13px system-ui">${i === 0 ? 'Start' : 'state ' + (i + 1)}</text>
      <text x="${67 + i * 125}" y="49" text-anchor="middle" style="fill:var(--muted);font:12px system-ui">${i === 1 ? 'right ↔ left swapped!' : 'normal'}</text>`).join('')}
      <rect x="385" y="10" width="115" height="48" rx="8" style="fill:var(--goal)"/>
      <text x="442" y="39" text-anchor="middle" style="fill:#fff;font:700 14px system-ui">Goal</text></svg>`,
    });
    w.appendChild(svg);
    const grid = h('div.widget-grid'); w.appendChild(grid);
    const left = h('div'), right = h('div'); grid.appendChild(left); grid.appendChild(right);
    const J = (p) => (-2 * (2 - p)) / (p * (1 - p));
    const out = h('div.small', { style: { marginTop: '8px' } });
    const chart = new RL.LineChart(right, { title: 'Expected return J(p) = −2(2−p) / (p(1−p))', info: 'J(p) for every probability p of choosing “right”. Dots: ε-greedy policies and the optimum.', height: 190, yMin: -100, yMax: 0, xMin: 0, xMax: 1, xLabel: 'p = probability of choosing “right”' });
    const xs = [], ys = [];
    for (let i = 1; i < 200; i++) { const p = i / 200; xs.push(p); ys.push(Math.max(-100, J(p))); }
    chart.addSeries('J(p)', '--accent', { raw: false }); chart.set('J(p)', ys, xs);
    const pts = chart.addSeries('ε-greedy (ε=0.1) & optimum', '--neg', { type: 'points', radius: 4 });
    pts.xs = [0.05, 0.95, 0.5918]; pts.data = [Math.max(-100, J(0.05)), J(0.95), J(0.5918)]; pts.labels = ['ε-greedy “left”', 'ε-greedy “right”', 'optimum'];
    const update = (p) => {
      chart.markerX = p; chart.requestDraw();
      out.innerHTML = `J(${fmt(p, 2)}) = <b>${fmt(J(p), 1)}</b> expected steps-cost. Best stochastic policy: p ≈ 0.59 → J ≈ −11.6. ` +
        `ε-greedy around the best deterministic action reaches only ≈ −44 (and −82 for “left”). Deterministic policies (p = 0 or 1) never reach the goal.`;
    };
    RL.ui.slider(left, { label: 'p = Pr(right)', min: 0.02, max: 0.98, step: 0.01, value: 0.5, onInput: update });
    left.appendChild(out);
    left.appendChild(h('div.small.muted', { html: 'A value-based agent with these features can only produce (near-)deterministic greedy policies plus ε-noise; a policy-gradient agent simply learns p ≈ 0.59. With perfect state information a deterministic policy would be optimal — the stochasticity compensates for missing information.' }));
    update(0.5);
    RL.util.renderMath(w);
    return () => chart.destroy();
  }

  /* ---------- Widget: score-function gradient variance ---------- */
  function varianceWidget(root) {
    const w = RL.page.widget(root, 'The score-function estimator and its variance',
      'Two actions, softmax policy with a single logit $\\theta$: $\\pi(\\text{R}) = \\sigma(\\theta)$. Each sample draws an action, a noisy reward, and forms $\\hat g = (R - b)\\,\\partial_\\theta\\log\\pi(a)$. The baseline $b$ never moves the mean — only the spread.');
    const grid = h('div.widget-grid'); w.appendChild(grid);
    const left = h('div'), right = h('div'); grid.appendChild(left); grid.appendChild(right);
    RL.ui.info(w.querySelector('.widget-title'), lg([
      [sw.band('var(--pos)'), '<b>Histogram</b> of 4000 single-sample gradient estimates ĝ = (R − b)·∂<sub>θ</sub> log π(a): one sampled action and one noisy reward each. Values outside −8 … 8 are not drawn.'],
      [sw.vline('var(--neg)'), '<b>Orange line</b>: the true gradient ∂J/∂θ, the mean of the histogram.'],
      [sw.text('b'), 'Changing the baseline b moves the two clusters (one per action) without moving their mean. A <b>narrower</b> histogram = lower variance = fewer samples needed per reliable step.'],
    ]), 'Gradient estimates');
    const p = { theta: 0.5, rL: 1, rR: 2, sigma: 1, b: 0 };
    const canvas = h('canvas'); const holder = h('div', canvas); right.appendChild(holder);
    const out = h('div.small', { style: { marginTop: '6px' } }); right.appendChild(out);
    let bS;
    const sl = (label, key, min, max, step, help) => RL.ui.slider(left, { label, min, max, step, value: p[key], help, onInput: (v) => { p[key] = v; draw(); } });
    sl('Logit θ', 'theta', -3, 3, 0.1, 'π(right) = 1/(1+e^{−θ}).');
    sl('Mean reward of “left”', 'rL', -5, 10, 0.1);
    sl('Mean reward of “right”', 'rR', -5, 10, 0.1);
    sl('Reward noise σ', 'sigma', 0, 5, 0.1);
    bS = sl('Baseline b', 'b', -5, 10, 0.1, 'Any constant (or state-dependent) value subtracted from the reward.');
    const br = h('div.btn-row'); left.appendChild(br);
    const pi = () => 1 / (1 + Math.exp(-p.theta));
    RL.ui.button(br, 'b = 0', () => { bS.set(0); }, 'small');
    RL.ui.button(br, 'b = V (mean reward)', () => { const q = pi(); bS.set(+(q * p.rR + (1 - q) * p.rL).toFixed(2)); }, 'small');
    RL.ui.button(br, 'b = optimal', () => { const q = pi(); bS.set(+((1 - q) * p.rR + q * p.rL).toFixed(2)); }, 'small');
    left.appendChild(h('div.small.muted', { html: 'Optimal constant baseline: $b^* = \\mathbb E[(\\partial\\log\\pi)^2 R] / \\mathbb E[(\\partial\\log\\pi)^2]$, which here equals $(1-\\pi_R)\\,r_R + \\pi_R\\, r_L$ — close to, but not exactly, the mean reward.' }));
    const rng = new RL.Rng(5);
    function draw() {
      const q = pi(), n = 4000, samples = new Float64Array(n);
      let m = 0;
      for (let i = 0; i < n; i++) {
        const right = rng.next() < q;
        const R = (right ? p.rR : p.rL) + p.sigma * rng.normal();
        samples[i] = (R - p.b) * (right ? 1 - q : -q);
        m += samples[i];
      }
      m /= n;
      const trueG = q * (1 - q) * (p.rR - p.rL);
      const e2 = q * (1 - q) ** 2 * ((p.rR - p.b) ** 2 + p.sigma ** 2) + (1 - q) * q * q * ((p.rL - p.b) ** 2 + p.sigma ** 2);
      const sd = Math.sqrt(Math.max(0, e2 - trueG * trueG));
      const W = Math.max(260, holder.clientWidth || 420), H = 180;
      const ctx = setupCanvas(canvas, W, H);
      const lo = -8, hi = 8, bins = 64, counts = new Array(bins).fill(0);
      for (const s of samples) { const k = Math.floor(((s - lo) / (hi - lo)) * bins); if (k >= 0 && k < bins) counts[k]++; }
      const mx = Math.max(...counts, 1);
      const X = (v) => 10 + ((v - lo) / (hi - lo)) * (W - 20);
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = cssVar('--pos'); ctx.globalAlpha = 0.7;
      const bw = (W - 20) / bins;
      counts.forEach((c, k) => { const bh = (c / mx) * (H - 40); ctx.fillRect(10 + k * bw, H - 22 - bh, bw - 1, bh); });
      ctx.globalAlpha = 1;
      ctx.strokeStyle = cssVar('--border'); ctx.beginPath(); ctx.moveTo(10, H - 22); ctx.lineTo(W - 10, H - 22); ctx.stroke();
      ctx.fillStyle = cssVar('--muted'); ctx.font = '11px system-ui'; ctx.textAlign = 'center';
      for (let v = -8; v <= 8; v += 2) ctx.fillText(String(v).replace('-', '−'), X(v), H - 8);
      const vline = (v, color, label, yy) => { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(v), 8); ctx.lineTo(X(v), H - 22); ctx.stroke(); ctx.fillStyle = color; ctx.textAlign = 'left'; ctx.fillText(label, X(v) + 4, yy); };
      vline(trueG, cssVar('--neg'), 'true gradient', 14);
      out.innerHTML = `π(right) = <b>${fmt(q, 3)}</b> · true gradient ∂J/∂θ = <b>${fmt(trueG, 3)}</b> · sample mean of 4000 estimates = ${fmt(m, 3)} · <b>std of one estimate = ${fmt(sd, 3)}</b> (signal-to-noise ${fmt(Math.abs(trueG) / (sd || 1e-9), 3)})`;
    }
    draw();
    RL.onTheme(draw);
    let ro; if (window.ResizeObserver) { ro = new ResizeObserver(draw); ro.observe(holder); }
    RL.util.renderMath(w);
    return () => { if (ro) ro.disconnect(); };
  }

  RL.chapters.push({
    id: 'pg', num: 5, part: 'Policy Optimization', title: 'Policy Gradients & Actor-Critic',
    mount(root) {
      RL.page.head(root, {
        num: 5, title: 'Policy gradients & actor-critic',
        lede: 'Stop estimating values to derive a policy — differentiate the expected return with respect to the policy parameters directly. The log-derivative trick, baselines, critics and GAE, and a careful comparison with value-based methods.',
        tags: [['policy-based'], ['on-policy'], ['REINFORCE · A2C · GAE'], ['basis of PPO & LLM RL', 'now']],
      });
      const cleanups = [];
      RL.page.prose(root, why);
      cleanups.push(corridorWidget(root));
      RL.page.prose(root, theory);
      cleanups.push(varianceWidget(root));
      RL.page.prose(root, String.raw`<h2>Lab: REINFORCE → baseline → actor-critic</h2><p>The same CartPole as in Chapter 4, now with a softmax policy network. The left map shows the policy itself, $\pi_\theta(\text{right}\mid s)$; the right map shows the critic $V_\phi(s)$ (blank for vanilla REINFORCE, which has none).</p>`);
      const lab = RL.cartpoleLab(root, {
        kind: 'pg', algos: ['reinforce', 'reinforce_b', 'a2c', 'ppo'], title: 'Policy gradient lab',
        defaults: { algo: 'reinforce', lr: 1e-3, lrV: 1e-3, nSteps: 32, entCoef: 0 },
        vizNote: 'A learned policy shows a diagonal boundary: push right when the pole leans or rotates right. Unlike DQN’s greedy map, these are probabilities — the policy stays stochastic near the boundary.',
      });
      cleanups.push(() => lab.destroy());
      const base = { lr: 1e-3, lrV: 1e-3, gamma: 0.99, lam: 0.95, entCoef: 0, nSteps: 32, width: 64 };
      const c = (o) => Object.assign({}, base, o);
      RL.ui.experiments(root, [
        { title: 'Vanilla REINFORCE', text: 'Monte Carlo returns, no baseline, one update per episode. It does learn — every positive return pushes up all its actions — but progress is noisy and it often regresses.', apply: () => lab.configure(c({ algo: 'reinforce' }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Add a learned baseline', text: 'REINFORCE with $V_\\phi(s)$ subtracted. Same expected gradient, lower variance: steadier improvement. Compare the critic map with DQN’s max-Q map.', apply: () => lab.configure(c({ algo: 'reinforce_b' }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Actor-critic with GAE', text: 'A2C: update every 32 steps with GAE(λ = 0.95) advantages and a small entropy bonus. No need to wait for episodes to end.', apply: () => lab.configure(c({ algo: 'a2c', lr: 7e-4, entCoef: 0.01 }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'λ = 0: pure TD advantages', text: 'A2C with $\\lambda = 0$ uses only $\\delta_t$. Early on the critic is wrong, so the advantages are biased and learning is slower. Try $\\lambda = 1$ for the other extreme.', apply: () => lab.configure(c({ algo: 'a2c', lr: 7e-4, entCoef: 0.01, lam: 0 }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Step too large: policy collapse', text: 'REINFORCE + baseline with learning rate $10^{-2}$. Watch the entropy chart: it can crash toward 0 as the policy becomes deterministic, often in the wrong place, and learning stalls. This is the problem PPO solves.', apply: () => lab.configure(c({ algo: 'reinforce_b', lr: 1e-2 }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Entropy keeps it exploring', text: 'Same high learning rate with an entropy bonus of 0.02. Entropy stays higher for longer; collapse is delayed or avoided, at the price of a less sharp final policy.', apply: () => lab.configure(c({ algo: 'reinforce_b', lr: 1e-2, entCoef: 0.02 }), { speedKey: 'max' }), labEl: lab.lab.el },
      ]);
      RL.page.prose(root, theory2);
      return () => cleanups.forEach((f) => f());
    },
  });
})(window.RL);
