/* Chapter 8: RL for language models — RLHF, KL regularization, DPO, GRPO, RLVR, reward hacking. */
(function (RL) {
  'use strict';
  const sw = RL.ui.sw, lg = RL.ui.lg;
  const { h, fmt, softmax, sampleCat } = RL.util;

  const part1 = String.raw`
<h2>A language model is a policy</h2>
<p>Map the LLM onto the RL vocabulary. The <b>state</b> is the prompt plus all tokens generated so far, the <b>action</b> is the next token, the <b>policy</b> is the model $\pi_\theta(y_t\mid x, y_{\lt t})$, and the transition is deterministic (append the token). The <b>reward</b> usually arrives once, at the end of the response: a reward-model score, a human rating, or a programmatic check ("is the final answer 42?", "do the unit tests pass?").</p>
<p>Because the reward is a function of the whole response, the problem is effectively a <b>contextual bandit</b> with an astronomically large, structured action space: context $x$, action $y = (y_1,\dots,y_T)$, reward $r(x,y)$. Everything from Chapter 2 (exploration, baselines) and Chapter 5 (score-function gradients) applies directly, with</p>
$$\nabla_\theta\, \mathbb E_{y\sim\pi_\theta(\cdot|x)}[r(x,y)] = \mathbb E_{y}\Big[\big(r(x,y) - b(x)\big)\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(y_t\mid x, y_{\lt t})\Big].$$
<p>The token-level view (each token an action, $\gamma = 1$, intermediate rewards zero except per-token KL penalties) is what PPO-based RLHF implements; the sequence-level view is what GRPO, RLOO and DPO exploit.</p>

<h2>The post-training pipeline</h2>
<ol>
<li><b>Pretraining</b> — next-token prediction on a large corpus. Produces a broad but unaligned policy.</li>
<li><b>Supervised fine-tuning (SFT)</b> — imitation learning (behavior cloning) on demonstrations of desired behavior. This model becomes the <b>reference policy</b> $\pi_\text{ref}$.</li>
<li><b>Preference / RL stage</b> — optimize a reward while staying close to $\pi_\text{ref}$: RLHF with PPO, DPO-style preference optimization, or RL from verifiable rewards with GRPO-like algorithms. In practice several of these are combined and iterated.</li>
</ol>

<h2>Reward models from human preferences</h2>
<p>Humans are better at comparing two answers than at assigning absolute scores. Given pairs where $y_w$ was preferred over $y_l$, a reward model $r_\phi$ is trained with the <b>Bradley–Terry</b> model:</p>
$$\Pr(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big), \qquad \mathcal L_\text{RM}(\phi) = -\mathbb E\big[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\big].$$
<p>Only reward <em>differences</em> are identified — any per-prompt offset cancels. The reward model is a learned proxy and it is wrong in places; the RL stage will find those places.</p>

<h2>The KL-regularized objective</h2>
<p>RLHF maximizes reward while penalizing divergence from the reference model:</p>
$$\max_{\pi}\ \mathbb E_{x\sim\mathcal D,\ y\sim\pi(\cdot|x)}\big[r(x,y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x)\big).$$
<p>The KL term keeps outputs fluent and diverse, keeps the policy where the reward model was trained (and therefore somewhat accurate), and acts as the "trust region" of the whole fine-tuning run. It has a beautiful closed-form solution:</p>
$$\pi^*_\beta(y\mid x) = \frac{1}{Z_\beta(x)}\,\pi_\text{ref}(y\mid x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big), \qquad Z_\beta(x) = \sum_y \pi_\text{ref}(y|x)\, e^{r(x,y)/\beta}.$$
<details class="deep math"><summary><span class="lbl">Math</span> Derivation of the optimal KL-regularized policy</summary><div class="body">
<p>For a fixed $x$, write the objective with the candidate $\pi^*_\beta$ defined above:</p>
$$\mathbb E_\pi[r] - \beta\,\mathbb E_\pi\Big[\log\frac{\pi}{\pi_\text{ref}}\Big] = -\beta\,\mathbb E_\pi\Big[\log\frac{\pi(y)}{\pi_\text{ref}(y)e^{r(y)/\beta}}\Big] = -\beta\,\mathrm{KL}\big(\pi\,\|\,\pi^*_\beta\big) + \beta\log Z_\beta .$$
<p>$\log Z_\beta$ does not depend on $\pi$ and the KL is minimized (zero) exactly at $\pi = \pi^*_\beta$. This is the Gibbs variational principle; the same exponential-tilting form is the soft-optimal policy of Chapter 5 with entropy replaced by a KL to a prior. As $\beta\to\infty$ we keep $\pi_\text{ref}$; as $\beta\to 0$ all mass goes to the argmax of $r$ — including any response where $r$ is wrong.</p>
</div></details>
<p>The widget computes $\pi^*_\beta$ exactly for six candidate responses. The proxy reward (what the reward model says) differs from the true quality on two of them — a verbose answer the reward model over-rates, and a sycophantic answer that exploits it.</p>`;

  const part2 = String.raw`
<div class="callout warn"><strong>Reward over-optimization (Goodhart's law).</strong> As the policy moves away from $\pi_\text{ref}$, the proxy reward keeps rising while the true reward first rises, then falls. Gao, Schulman & Hilton (2023) measured exactly this curve for real reward models, with the gold reward following $d(\alpha - \beta\log d)$ for RL, $d = \sqrt{\mathrm{KL}}$. Typical real-world exploits: length (longer answers score higher), sycophancy (agreeing with the user), confident formatting, and for code: tests that are special-cased rather than solved.</div>

<h2>RLHF with PPO</h2>
<p>The InstructGPT recipe (Ouyang et al., 2022) runs PPO on the token MDP. The per-token reward is the KL penalty, with the reward-model score added at the last token:</p>
$$r_t = -\beta\log\frac{\pi_\theta(y_t\mid x,y_{\lt t})}{\pi_\text{ref}(y_t\mid x,y_{\lt t})} + \mathbb 1[t = T]\; r_\phi(x, y).$$
<p>GAE over tokens with a learned value head, the clipped objective, several epochs per batch — exactly Chapter 6. The cost: four large models in memory (policy, critic, reference, reward model), a critic that must learn values of partial sentences, and many sensitive hyperparameters. Two simplifications now dominate: remove the reward model and RL loop (DPO), or remove the critic (GRPO, RLOO).</p>

<h2>Direct Preference Optimization</h2>
<p>DPO (Rafailov et al., 2023) observes that the closed-form solution can be inverted: any reward function corresponds to an optimal policy and vice versa,</p>
$$r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)} + \beta\log Z(x).$$
<p>Plug this into the Bradley–Terry likelihood: the intractable $\beta\log Z(x)$ appears in both terms of the difference and <b>cancels</b>. What remains is a supervised loss on the policy itself:</p>
$$\mathcal L_\text{DPO}(\theta) = -\mathbb E_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\Big)\Big].$$
<details class="deep math"><summary><span class="lbl">Math</span> What the DPO gradient does</summary><div class="body">
<p>With the implicit reward $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$:</p>
$$\nabla_\theta\mathcal L_\text{DPO} = -\beta\,\mathbb E\Big[\, w_\theta \,\big(\nabla_\theta\log\pi_\theta(y_w|x) - \nabla_\theta\log\pi_\theta(y_l|x)\big)\Big], \qquad w_\theta = \sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big).$$
<p>The weight $w_\theta$ measures how wrong the implicit reward model currently is about this pair.</p>
<p>It raises the likelihood of preferred responses and lowers that of rejected ones, weighting each pair by how badly the implicit reward model currently ranks it. Note that it only constrains the <em>difference</em>: both log-likelihoods can (and often do) decrease together, moving mass to responses that appear in neither.</p>
</div></details>
<p><b>Trade-offs.</b> DPO is simple, stable and cheap — no sampling during training, no reward model, no critic. But it is <em>offline</em>: it never explores beyond the preference dataset and cannot discover new behavior the way online RL can. Iterative/online DPO (re-sample, re-label, repeat) recovers part of the gap. Many variants change the loss: IPO (bounded, avoids over-fitting deterministic preferences), KTO (unpaired good/bad labels), SimPO (reference-free, length-normalized), ORPO (merged with SFT).</p>

<h2>RL from verifiable rewards and GRPO</h2>
<p>For math, code and other checkable tasks, the reward can be computed exactly: $r = 1$ if the final answer matches or the tests pass, else 0. No reward model to exploit (though weak checkers can still be gamed). Large-scale RL with such rewards is what turned LLMs into <b>reasoning models</b> that produce long chains of thought, check their work and backtrack; DeepSeek-R1 (2025) documented how this behavior emerges from RL with rule-based rewards alone.</p>
<p><b>GRPO</b> (Group Relative Policy Optimization; Shao et al., 2024) is PPO without a critic. For each prompt it samples a <em>group</em> of $G$ responses and uses the group statistics as the baseline:</p>
$$\hat A_i = \frac{r_i - \operatorname{mean}(\mathbf r)}{\operatorname{std}(\mathbf r)}, \qquad \mathbf r = (r_1,\dots,r_G),$$
<p>and the same $\hat A_i$ is used for every token of response $i$.</p>
$$\mathcal J_\text{GRPO}(\theta) = \mathbb E\Big[\frac1G\sum_{i=1}^G\frac{1}{|y_i|}\sum_{t}\min\big(\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t},1\pm\epsilon)\hat A_i\big) - \beta\,\mathbb D_\text{KL}\big[\pi_\theta\|\pi_\text{ref}\big]\Big],$$
<p>where $\rho_{i,t}$ is the per-token probability ratio and the KL is estimated per token with $\frac{\pi_\text{ref}}{\pi_\theta} - \log\frac{\pi_\text{ref}}{\pi_\theta} - 1$. The group mean is exactly the baseline $\bar R$ of the gradient bandit in Chapter 2.</p>
<details class="deep math"><summary><span class="lbl">Math</span> Baselines in group methods: bias, variance and known pitfalls</summary><div class="body">
<ul>
<li><b>RLOO</b> (leave-one-out; Kool et al. 2019, Ahmadian et al. 2024) uses $b_i = \frac{1}{G-1}\sum_{j\ne i} r_j$. Because $b_i$ is independent of response $i$, the estimator is exactly unbiased. Using the full group mean including $r_i$ shrinks every advantage by the factor $\frac{G-1}{G}$ — a scale change, harmless with Adam.</li>
<li><b>Dividing by the std</b> is not a baseline: it rescales the gradient per prompt, up-weighting prompts where almost all answers agree (very easy or very hard ones). Dr. GRPO (Liu et al., 2025) argues this, and the per-response $1/|y_i|$ length normalization (which favours long wrong answers), bias the optimization, and removes both.</li>
<li><b>Zero-signal groups.</b> If all $G$ answers get the same reward — all wrong on a hard prompt, all right on an easy one — every advantage is 0 and the prompt contributes nothing. DAPO (Yu et al., 2025) therefore filters such groups ("dynamic sampling"), uses an asymmetric "clip-higher" range to keep exploration alive, and a token-level loss.</li>
<li><b>Exploration</b> happens only through sampling diversity (temperature, group size). RL from a strong base model mostly sharpens behaviors the base model can already produce with some probability; how much genuinely new capability RL creates is an active research question.</li>
</ul>
</div></details>
<p>The widget trains a toy "reasoning policy" that chooses among six solution strategies for a math problem with a 0/1 verifier. Watch the group advantages: only groups with <em>mixed</em> outcomes produce a learning signal.</p>`;

  const part3 = String.raw`
<h2>Process vs. outcome rewards</h2>
<p>An outcome reward says only whether the final answer was right — a sparse, delayed signal over thousands of tokens (credit assignment again). <b>Process reward models</b> (PRMs) score individual reasoning steps (Lightman et al., 2023, "Let's verify step by step"), giving denser feedback and better search guidance, but they are expensive to label and themselves hackable. Current large-scale reasoning RL relies mostly on outcome rewards with large groups and long training; PRMs are used more for search and reranking.</p>

<h2>Comparing the methods</h2>
<div class="table-wrap"><table class="compare">
<tr><th>Method</th><th>Signal</th><th>Online sampling?</th><th>Critic</th><th>Notes</th></tr>
<tr><td>Best-of-$n$ / rejection sampling</td><td>RM or verifier</td><td>at inference</td><td>–</td><td>no training; strong baseline; cost grows with $n$. Fine-tuning on the best samples = expert iteration / STaR / RFT</td></tr>
<tr><td>PPO-RLHF</td><td>learned RM + KL</td><td>yes</td><td>yes</td><td>the original recipe; flexible, expensive, hyperparameter-sensitive</td></tr>
<tr><td>DPO and variants</td><td>preference pairs</td><td>no (offline)</td><td>–</td><td>simple and cheap; no exploration; iterate to go online</td></tr>
<tr><td>GRPO / RLOO / REINFORCE++</td><td>RM or verifier</td><td>yes (groups)</td><td>no</td><td>default for reasoning RL with verifiable rewards</td></tr>
<tr><td>RLAIF / Constitutional AI</td><td>AI feedback from principles</td><td>either</td><td>either</td><td>replaces human labels with model judgments guided by a written constitution</td></tr>
</table></div>

<h2>Where this is going</h2>
<ul>
<li><b>Agentic, multi-turn RL.</b> Tool use, browsing, coding agents and long tasks turn the bandit back into a genuine multi-step MDP with real environments, delayed rewards and expensive rollouts. Value functions, credit assignment and off-policy learning — the classic topics of Chapters 1–4 — become important again.</li>
<li><b>Reward specification</b> is the bottleneck: verifiers for open-ended tasks, rubric-based and model-based judges, and robustness to reward hacking.</li>
<li><b>Scaling RL compute</b>: stable large-batch training, asynchronous off-policy generation (with importance corrections — Chapter 3's ratios), and the trade-off between training-time RL and test-time search (Chapter 7).</li>
</ul>
<details class="deep history"><summary><span class="lbl">History</span> How we got here</summary><div class="body">
<p>Christiano et al. (2017) learned reward models from human preferences for Atari and MuJoCo; Ziegler et al. (2019) and Stiennon et al. (2020) applied it to language (summarization); InstructGPT (2022) made RLHF the standard for assistants; Constitutional AI (Bai et al., 2022) introduced AI feedback guided by principles. DPO (2023) made preference optimization a supervised problem. DeepSeekMath (2024) introduced GRPO; reasoning models trained with large-scale RL on verifiable rewards (OpenAI o1 in 2024, DeepSeek-R1 in 2025) made RL central to frontier LLM training.</p>
</div></details>
`;

  /* ---------- Widget A: KL-regularized optimum and Goodhart ---------- */
  function klWidget(root) {
    const w = RL.page.widget(root, 'The KL-regularized optimal policy and reward hacking',
      'Six possible answers to one prompt. Bars: $\\pi^*_\\beta(y) \\propto \\pi_\\text{ref}(y)\\,e^{\\hat r(y)/\\beta}$ (filled) vs. $\\pi_\\text{ref}$ (black tick). $\\hat r$ = reward-model score, $r^\\star$ = true quality. Decrease β to optimize harder.');
    const R = [
      { name: 'Correct and concise', ref: 0.20, proxy: 1.0, truth: 1.0 },
      { name: 'Correct, padded & verbose', ref: 0.12, proxy: 1.3, truth: 0.7 },
      { name: 'Partially correct', ref: 0.30, proxy: 0.4, truth: 0.4 },
      { name: 'Confidently wrong', ref: 0.25, proxy: 0.2, truth: -0.5 },
      { name: 'Polite refusal', ref: 0.10, proxy: -0.3, truth: 0.0 },
      { name: 'Sycophantic, games the RM', ref: 0.03, proxy: 2.0, truth: -1.0 },
    ];
    RL.ui.info(w.querySelector('.widget-title'), lg([
      [sw.box('var(--pos)'), '<b>Filled bar</b>: probability of the answer under the optimal regularized policy π*<sub>β</sub>. Blue = answers with non-negative true quality.'],
      [sw.box('var(--neg)'), 'Orange fill: answers whose <b>true</b> quality r⋆ is negative (wrong, sycophantic).'],
      [sw.tick('var(--text)'), 'Black tick: probability under the reference policy π<sub>ref</sub> (the starting point).'],
      [sw.line('var(--c2)'), 'Chart, orange: expected <b>proxy</b> reward E[r̂] (what the reward model says) as β is swept.'],
      [sw.line('var(--c3)'), 'Chart, green: expected <b>true</b> reward E[r⋆]. x-axis: KL distance from π<sub>ref</sub>.'],
      [sw.vline('var(--text)'), 'Dashed vertical line: your current β.'],
    ]), 'KL-regularized optimum');
    let beta = 1;
    const grid = h('div.widget-grid'); w.appendChild(grid);
    const left = h('div'), right = h('div'); grid.appendChild(left); grid.appendChild(right);
    RL.ui.slider(left, { label: 'KL coefficient β', min: 0.03, max: 10, log: true, value: beta, fmt: (v) => v.toFixed(2), help: 'Large β: stay at the reference policy. Small β: maximize the proxy reward at any cost.', onInput: (v) => { beta = v; update(); } });
    RL.ui.slider(left, { label: 'RM score of the sycophantic answer', min: -1, max: 3, step: 0.05, value: R[5].proxy, help: 'How badly the reward model is fooled by the exploit.', onInput: (v) => { R[5].proxy = v; update(); } });
    const st = RL.ui.stats(left, [['kl', 'KL(π‖π_ref)'], ['proxy', 'E[proxy r̂]'], ['truth', 'E[true r⋆]'], ['hack', 'P(exploit)']]);
    const bars = h('div.bars'); right.appendChild(bars);
    const rows = R.map((r) => {
      const fill = h('div.bar-fill'), tick = h('div.bar-fill.ref'), num = h('span.bar-num');
      bars.appendChild(h('div.bar-row', h('span', { title: `RM score ${r.proxy}, true ${r.truth}` }, r.name, h('span.muted', ` (r̂ ${r.proxy >= 0 ? '+' : ''}${r.proxy}, r⋆ ${r.truth >= 0 ? '+' : ''}${r.truth})`)), h('div.bar-track', fill, tick), num));
      return { fill, tick, num, r };
    });
    const chart = new RL.LineChart(right, { title: 'Sweeping β: rewards vs. distance from the reference policy', info: 'Each point is the optimal policy for one β; x = its KL distance from the reference policy.', height: 190, xLabel: 'KL(π*_β ‖ π_ref)  (nats)' });
    chart.addSeries('proxy reward (RM)', '--c2', { raw: false }); chart.addSeries('true reward', '--c3', { raw: false });
    const solve = (b) => {
      const z = R.map((r) => Math.log(r.ref) + r.proxy / b);
      const p = softmax(z);
      let kl = 0, ep = 0, et = 0;
      p.forEach((q, i) => { if (q > 0) kl += q * Math.log(q / R[i].ref); ep += q * R[i].proxy; et += q * R[i].truth; });
      return { p, kl, ep, et };
    };
    function update() {
      const s = solve(beta);
      rows.forEach((row, i) => { row.fill.style.width = (100 * s.p[i]) + '%'; row.tick.style.width = (100 * row.r.ref) + '%'; row.num.textContent = s.p[i] < 0.0005 ? '<0.001' : fmt(s.p[i], 3); row.fill.style.background = row.r.truth < 0 ? 'var(--neg)' : 'var(--pos)'; });
      st.kl(s.kl); st.proxy(s.ep); st.truth(s.et); st.hack(s.p[5]);
      const xs = [], yp = [], yt = [];
      for (let k = 0; k <= 120; k++) { const b = Math.exp(Math.log(20) + (k / 120) * (Math.log(0.02) - Math.log(20))); const r = solve(b); xs.push(r.kl); yp.push(r.ep); yt.push(r.et); }
      chart.set('proxy reward (RM)', yp, xs); chart.set('true reward', yt, xs);
      chart.markerX = s.kl; chart.requestDraw();
    }
    left.appendChild(h('div.small.muted', { html: 'Orange bars are answers whose true quality is negative. Find the β that maximizes the <em>true</em> reward — it is neither the smallest nor the largest. In real RLHF you only observe the proxy curve.' }));
    update();
    RL.util.renderMath(w);
    return () => chart.destroy();
  }

  /* ---------- Widget B: GRPO toy ---------- */
  function grpoWidget(root) {
    const S = [
      { name: 'Answer immediately', short: 'guess', p: 0.10, ref: 0.40 },
      { name: 'Short reasoning', short: 'short', p: 0.35, ref: 0.25 },
      { name: 'Step-by-step', short: 'CoT', p: 0.65, ref: 0.15 },
      { name: 'Step-by-step + self-check', short: 'CoT+check', p: 0.85, ref: 0.05 },
      { name: 'Write & run code', short: 'code', p: 0.75, ref: 0.05 },
      { name: 'Restate question, no answer', short: 'restate', p: 0.0, ref: 0.10 },
    ];
    const lab = RL.page.lab(root, { title: 'GRPO lab' });
    const C = lab.controls;
    C.appendChild(h('div.lab-title', 'Group RL on a verifier'));
    const cfg = { algo: 'grpo', G: 8, lr: 0.5, beta: 0.02, difficulty: 1, speed: 6 };
    let z, it, hist, kls, zeroGroups, lastGroup, rng, acc = 0, lastT = 0;
    const tr = RL.ui.transport(C, {
      onPlay: (p) => { if (p) { lastT = performance.now(); loop.start(); } else loop.stop(); },
      onStep: () => { tr.setPlaying(false); step(); draw(); },
      onReset: () => { tr.setPlaying(false); reset(); },
      stepLabel: 'One group',
    });
    const algoSel = RL.ui.select(C, {
      label: 'Advantage estimator', value: cfg.algo,
      options: [
        { value: 'grpo', label: 'GRPO: (r − mean) / std' },
        { value: 'drgrpo', label: 'Dr. GRPO: r − mean' },
        { value: 'rloo', label: 'RLOO: r − mean of others' },
        { value: 'reinforce', label: 'REINFORCE: r (no baseline)' },
      ],
      onChange: (v) => { cfg.algo = v; },
    });
    const gS = RL.ui.slider(C, { label: 'Group size G', min: 2, max: 32, step: 1, value: cfg.G, help: 'Responses sampled per prompt. Larger groups give better baselines and fewer all-equal groups.', onInput: (v) => { cfg.G = v; } });
    const lrS = RL.ui.slider(C, { label: 'Learning rate', min: 0.02, max: 3, log: true, value: cfg.lr, fmt: (v) => v.toFixed(2), onInput: (v) => { cfg.lr = v; } });
    const bS = RL.ui.slider(C, { label: 'KL penalty β', min: 0, max: 0.5, step: 0.01, value: cfg.beta, help: 'Pulls the policy back toward the reference (SFT) policy.', onInput: (v) => { cfg.beta = v; } });
    const dS = RL.ui.slider(C, { label: 'Prompt difficulty', min: 1, max: 10, step: 0.5, value: cfg.difficulty, fmt: (v) => '×' + (1 / v).toFixed(2) + ' success', help: 'Divides every strategy\'s success probability. Hard prompts produce many all-wrong groups with zero advantage.', onInput: (v) => { cfg.difficulty = v; } });
    RL.ui.slider(C, { label: 'Groups per second', min: 1, max: 60, step: 1, value: cfg.speed, onInput: (v) => { cfg.speed = v; } });

    RL.ui.info(lab.viz.appendChild(h('div.lab-title', 'Latest group (sampled from the current policy)')), lg([
        [sw.text('CoT'), 'One chip per sampled answer: the solution strategy it used.'],
        [sw.text('✓', 'var(--good)'), 'Verifier reward: ✓ r = 1 (correct) or ✗ r = 0.'],
        [sw.text('A'), '<b>A</b> = advantage computed from the group (depends on the estimator chosen on the left).'],
        [sw.outline('var(--pos)'), 'Blue border: A > 0, this strategy is made <b>more</b> likely.'],
        [sw.outline('var(--neg)'), 'Orange border: A < 0, made <b>less</b> likely.'],
        [sw.outline('var(--muted)'), 'Grey border: A = 0, no learning signal (e.g. every answer in the group got the same reward).'],
      ]), 'Sampled group');
    const groupEl = h('div.group-samples'); lab.viz.appendChild(groupEl);
    const groupNote = h('div.lab-status'); lab.viz.appendChild(groupNote);
    RL.ui.info(lab.viz.appendChild(h('div.lab-title', { style: { marginTop: '8px' } }, 'Policy over strategies (tick = reference policy)')), lg([
        [sw.box('var(--pos)'), 'Filled bar: current probability of choosing the strategy.'],
        [sw.tick('var(--text)'), 'Tick: probability under the reference (SFT) policy.'],
        [sw.text('p✓'), 'Success probability of the strategy, known to the simulation but not to the learner.'],
      ]), 'Policy');
    const bars = h('div.bars'); lab.viz.appendChild(bars);
    const rows = S.map((s) => {
      const fill = h('div.bar-fill'), tick = h('div.bar-fill.ref'), num = h('span.bar-num');
      bars.appendChild(h('div.bar-row', h('span', s.name, h('span.muted', ` · p✓ ${s.p}`)), h('div.bar-track', fill, tick), num));
      return { fill, tick, num };
    });
    const stats = RL.ui.stats(lab.side, [['it', 'Groups (steps)'], ['succ', 'Expected success'], ['kl', 'KL(π‖π_ref)'], ['zero', 'Zero-signal groups']]);
    const ch = new RL.LineChart(lab.side, { title: 'Expected success rate of the policy', info: 'Σ π(strategy)·p✓(strategy): the probability that a freshly sampled answer is correct.', height: 160, yMin: 0, yMax: 1, xLabel: 'step', legend: false });
    ch.addSeries('succ', '--accent', { raw: false });
    const chK = new RL.LineChart(lab.side, { title: 'KL to reference', info: 'KL(π‖π_ref): how far the policy has moved away from the reference (SFT) policy.', height: 120, xLabel: 'step', legend: false, yMin: 0 });
    chK.addSeries('kl', '--c2', { raw: false });

    const refP = S.map((s) => s.ref);
    const succ = () => S.map((s) => s.p / cfg.difficulty);
    function reset() { z = refP.map(Math.log); it = 0; hist = []; kls = []; zeroGroups = 0; lastGroup = null; rng = new RL.Rng(); ch.clear(); chK.clear(); draw(); }
    function step() {
      const pi = softmax(z), ps = succ();
      const G = cfg.G, grp = [];
      for (let i = 0; i < G; i++) { const a = sampleCat(pi, rng); grp.push({ a, r: rng.next() < ps[a] ? 1 : 0 }); }
      const rs = grp.map((g) => g.r), m = RL.util.mean(rs), sd = RL.util.std(rs);
      grp.forEach((g, i) => {
        if (cfg.algo === 'grpo') g.A = (g.r - m) / (sd + 1e-4);
        else if (cfg.algo === 'drgrpo') g.A = g.r - m;
        else if (cfg.algo === 'rloo') g.A = g.r - (rs.reduce((s, x) => s + x, 0) - g.r) / (G - 1);
        else g.A = g.r;
      });
      if (sd === 0) zeroGroups++;
      const grad = new Float64Array(S.length);
      for (const g of grp) for (let j = 0; j < S.length; j++) grad[j] += (g.A * ((j === g.a ? 1 : 0) - pi[j])) / G;
      let kl = 0; for (let j = 0; j < S.length; j++) kl += pi[j] * Math.log(pi[j] / refP[j]);
      for (let j = 0; j < S.length; j++) grad[j] -= cfg.beta * pi[j] * (Math.log(pi[j] / refP[j]) - kl);
      for (let j = 0; j < S.length; j++) z[j] += cfg.lr * grad[j];
      it++; lastGroup = { grp, m, sd };
      const p2 = softmax(z);
      hist.push(p2.reduce((s, q, j) => s + q * ps[j], 0));
      kls.push(p2.reduce((s, q, j) => s + q * Math.log(q / refP[j]), 0));
      ch.get('succ').data.push(hist[hist.length - 1]); chK.get('kl').data.push(kls[kls.length - 1]);
    }
    function draw() {
      const pi = softmax(z), ps = succ();
      rows.forEach((r, i) => { r.fill.style.width = (100 * pi[i]) + '%'; r.tick.style.width = (100 * refP[i]) + '%'; r.num.textContent = fmt(pi[i], 3); });
      groupEl.innerHTML = '';
      if (lastGroup) {
        for (const g of lastGroup.grp) {
          const col = g.A > 1e-9 ? 'var(--pos)' : g.A < -1e-9 ? 'var(--neg)' : 'var(--muted)';
          groupEl.appendChild(h('div.sample', { style: { borderColor: col } }, h('div', S[g.a].short), h('div', h('span.r' + (g.r ? '.ok' : '.no'), g.r ? '✓ r=1' : '✗ r=0'), ' ', h('span.adv', { style: { color: col } }, 'A=' + fmt(g.A, 2)))));
        }
        groupNote.textContent = lastGroup.sd === 0 && cfg.algo !== 'reinforce'
          ? `All ${cfg.G} rewards equal (${lastGroup.m}) → every advantage is 0: this prompt gives no learning signal.`
          : `mean reward ${fmt(lastGroup.m, 2)}, std ${fmt(lastGroup.sd, 2)} — above-average responses get A > 0 (pushed up), below-average A < 0 (pushed down).`;
      } else groupNote.textContent = 'Press “One group” to sample G responses and take one policy-gradient step.';
      stats.it(it); stats.succ(pi.reduce((s, q, j) => s + q * ps[j], 0));
      stats.kl(pi.reduce((s, q, j) => s + q * Math.log(q / refP[j]), 0));
      stats.zero(it ? fmt((100 * zeroGroups) / it, 0) + '%' : '–');
      ch.requestDraw(); chK.requestDraw();
    }
    const loop = new RL.Loop(() => {
      const now = performance.now(); acc += ((now - lastT) / 1000) * cfg.speed; lastT = now;
      let n = Math.min(20, Math.floor(acc)); acc -= n;
      if (n > 0) { while (n--) step(); draw(); }
    });
    reset();
    const setup = (o) => () => {
      tr.setPlaying(false);
      if (o.algo) { cfg.algo = o.algo; algoSel.set(o.algo, true); }
      if (o.G) { cfg.G = o.G; gS.set(o.G, true); }
      if (o.lr) { cfg.lr = o.lr; lrS.set(o.lr, true); }
      if (o.beta !== undefined) { cfg.beta = o.beta; bS.set(o.beta, true); }
      if (o.difficulty) { cfg.difficulty = o.difficulty; dS.set(o.difficulty, true); }
      reset(); tr.setPlaying(true);
    };
    RL.ui.experiments(root, [
      { title: 'GRPO discovers careful reasoning', text: 'Default GRPO, G = 8. The reference policy mostly guesses (10% success). Within a few dozen groups mass moves to “step-by-step + self-check”, success rises toward 85%.', apply: setup({ algo: 'grpo', G: 8, lr: 0.5, beta: 0.02, difficulty: 1 }), labEl: lab.el },
      { title: 'Hard prompts: no signal', text: 'Difficulty ×0.1. Most groups are all-wrong, so advantages are exactly zero (see “zero-signal groups”). Learning crawls. Increase G to 32: more groups contain a lucky success.', apply: setup({ algo: 'grpo', G: 8, lr: 0.5, beta: 0.02, difficulty: 10 }), labEl: lab.el },
      { title: 'No baseline', text: 'REINFORCE with raw 0/1 rewards: only successes push, failures do nothing. Still unbiased, but noisier and slower — and every success also pushes up mediocre strategies.', apply: setup({ algo: 'reinforce', G: 8, lr: 0.5, beta: 0.02, difficulty: 1 }), labEl: lab.el },
      { title: 'KL anchors the policy', text: 'β = 0.3. The pull toward the reference policy stops the policy from fully committing to the best strategy: a trade-off between reward and staying close to π_ref.', apply: setup({ algo: 'grpo', G: 8, lr: 0.5, beta: 0.3, difficulty: 1 }), labEl: lab.el },
      { title: 'std normalization on hard prompts', text: 'Compare GRPO with Dr. GRPO at difficulty ×0.25: dividing by a small std inflates the rare non-zero signals (larger, noisier steps). Dr. GRPO keeps the raw scale.', apply: setup({ algo: 'drgrpo', G: 8, lr: 0.5, beta: 0.02, difficulty: 4 }), labEl: lab.el },
    ]);
    return () => { loop.stop(); ch.destroy(); chK.destroy(); };
  }

  RL.chapters.push({
    id: 'llm', num: 8, part: 'Beyond the Basics', title: 'RL for Language Models',
    mount(root) {
      RL.page.head(root, {
        num: 8, title: 'RL for language models: RLHF, DPO, GRPO',
        lede: 'Post-training turns a pretrained text predictor into an assistant or a reasoning model — and its core is the policy-gradient and trust-region machinery of Chapters 5 and 6, applied to a policy with billions of parameters and a reward that is often learned and imperfect.',
        tags: [['contextual bandit view'], ['KL regularization'], ['preferences & verifiers'], ['frontier', 'now']],
      });
      const cleanups = [];
      RL.page.prose(root, part1);
      cleanups.push(klWidget(root));
      RL.page.prose(root, part2);
      cleanups.push(grpoWidget(root));
      RL.page.prose(root, part3);
      return () => cleanups.forEach((f) => f());
    },
  });
})(window.RL);
