/* Chapter 10: cheat sheet — key equations, algorithm chooser, default hyperparameters, glossary, reading. */
(function (RL) {
  'use strict';

  const html = String.raw`
<h2>Key equations</h2>
<div class="table-wrap"><table class="compare">
<tr><th>Concept</th><th>Equation</th><th>Ch.</th></tr>
<tr><td>Return</td><td>$G_t = \sum_{k\ge0}\gamma^k r_{t+k+1}$, &nbsp; $J(\pi) = \mathbb E_\pi[G_0]$</td><td>0</td></tr>
<tr><td>Bellman expectation</td><td>$V^\pi(s) = \sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V^\pi(s')]$</td><td>1</td></tr>
<tr><td>Bellman optimality</td><td>$Q^*(s,a) = \mathbb E[r + \gamma\max_{a'}Q^*(s',a')]$</td><td>1</td></tr>
<tr><td>Contraction</td><td>$\|\mathcal TU - \mathcal TV\|_\infty \le \gamma\|U-V\|_\infty$</td><td>1</td></tr>
<tr><td>Regret</td><td>$\mathcal R_T = \sum_a\Delta_a\,\mathbb E[N_a(T)]$; optimal rate $\Theta(\log T)$</td><td>2</td></tr>
<tr><td>UCB1</td><td>$a_t = \arg\max_a Q_a + c\sqrt{\ln t / N_a}$</td><td>2</td></tr>
<tr><td>TD error</td><td>$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$</td><td>3</td></tr>
<tr><td>SARSA / Q-learning</td><td>$Q \mathrel{+}= \alpha[r+\gamma Q(s',a') - Q]$ / $Q \mathrel{+}= \alpha[r + \gamma\max_{a'}Q(s',a') - Q]$</td><td>3</td></tr>
<tr><td>DQN loss</td><td>$\big(r + \gamma(1-d)\,Q_{\theta^-}(s', \arg\max_{a'}Q_\theta(s',a')) - Q_\theta(s,a)\big)^2$ (Double DQN)</td><td>4</td></tr>
<tr><td>Policy gradient</td><td>$\nabla J = \mathbb E[\sum_t \hat A_t\nabla\log\pi_\theta(a_t|s_t)]$</td><td>5</td></tr>
<tr><td>Baseline is unbiased</td><td>$\mathbb E_{a\sim\pi}[b(s)\nabla\log\pi(a|s)] = 0$</td><td>5</td></tr>
<tr><td>GAE</td><td>$\hat A_t = \sum_l(\gamma\lambda)^l\delta_{t+l} = \delta_t + \gamma\lambda\hat A_{t+1}$</td><td>5</td></tr>
<tr><td>Performance difference</td><td>$J(\pi') - J(\pi) = \frac1{1-\gamma}\mathbb E_{s\sim d^{\pi'},a\sim\pi'}[A^\pi(s,a)]$</td><td>1, 6</td></tr>
<tr><td>PPO-clip</td><td>$\mathbb E[\min(r_t\hat A_t, \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$, $r_t = \pi_\theta/\pi_\text{old}$</td><td>6</td></tr>
<tr><td>Natural gradient</td><td>$\Delta\theta \propto F^{-1}\nabla L$, $F = \mathbb E[\nabla\log\pi\nabla\log\pi^\top]$</td><td>6</td></tr>
<tr><td>UCT / PUCT</td><td>$Q + c\sqrt{\ln N(s)/N(s,a)}$ / $Q + c\,P(s,a)\sqrt{N(s)}/(1+N(s,a))$</td><td>7</td></tr>
<tr><td>Bradley–Terry</td><td>$\Pr(y_w\succ y_l) = \sigma(r(y_w) - r(y_l))$</td><td>8</td></tr>
<tr><td>KL-regularized optimum</td><td>$\pi^*(y|x)\propto\pi_\text{ref}(y|x)\exp(r(x,y)/\beta)$</td><td>8</td></tr>
<tr><td>DPO</td><td>$-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}\big)$</td><td>8</td></tr>
<tr><td>GRPO advantage</td><td>$\hat A_i = (r_i - \operatorname{mean}(\mathbf r))/\operatorname{std}(\mathbf r)$ over a group of $G$ samples</td><td>8</td></tr>
<tr><td>SAC objective</td><td>$\mathbb E[\sum_t\gamma^t(r_t + \alpha\mathcal H(\pi(\cdot|s_t)))]$</td><td>9</td></tr>
</table></div>

<h2>Which algorithm should I try first?</h2>
<div class="table-wrap"><table class="compare">
<tr><th>Situation</th><th>Start with</th><th>Why</th></tr>
<tr><td>Fast simulator, any action space</td><td><b>PPO</b> (many parallel envs)</td><td>robust, simple, scales with compute</td></tr>
<tr><td>Continuous control, expensive samples (real robot)</td><td><b>SAC</b> (or TD3); model-based (TD-MPC2, Dreamer) if very scarce</td><td>off-policy reuse; entropy-driven exploration</td></tr>
<tr><td>Discrete actions, limited samples</td><td><b>DQN</b> with Double + Dueling + n-step (+ PER)</td><td>off-policy, mature</td></tr>
<tr><td>Fixed dataset, no interaction</td><td><b>IQL</b>, CQL, TD3+BC; BC as baseline</td><td>avoid out-of-distribution actions</td></tr>
<tr><td>Expert demonstrations available</td><td><b>Behavior cloning</b> first, then RL fine-tuning</td><td>cheapest strong baseline</td></tr>
<tr><td>Known model / game rules</td><td><b>MCTS</b>; AlphaZero-style if you can self-play</td><td>planning beats learning when the model is exact</td></tr>
<tr><td>One-step decisions with context (recsys, ads)</td><td><b>Contextual bandits</b>: Thompson sampling, LinUCB</td><td>no credit assignment needed</td></tr>
<tr><td>LLM with preference data</td><td><b>DPO</b> (or iterative/online DPO)</td><td>simple, stable, no RM</td></tr>
<tr><td>LLM with verifiable rewards</td><td><b>GRPO / RLOO</b>-style group policy gradients</td><td>no critic; strong for reasoning</td></tr>
<tr><td>LLM with a learned reward, online</td><td><b>PPO-RLHF</b> or group methods + KL penalty</td><td>exploration beyond the dataset</td></tr>
</table></div>

<h2>Typical hyperparameters (starting points)</h2>
<div class="table-wrap"><table class="compare">
<tr><th>PPO</th><th>DQN</th><th>SAC</th></tr>
<tr><td>lr $3\cdot10^{-4}$ (Adam), annealed<br>γ 0.99, GAE λ 0.95<br>clip ε 0.2, 3–10 epochs<br>rollout 128–2048 × N envs<br>minibatches 4–32, grad-clip 0.5<br>entropy 0–0.01, adv. normalization</td><td>lr $10^{-4}$–$10^{-3}$<br>γ 0.99, batch 32–256<br>replay $10^5$–$10^6$, warm-up $10^3$–$10^4$<br>target: hard every 1–10k steps or τ ≈ 0.005<br>ε: 1 → 0.01–0.05 over 10% of training<br>Huber loss, Double DQN</td><td>lr $3\cdot10^{-4}$ (actor & critics)<br>γ 0.99, τ 0.005<br>batch 256, replay $10^6$<br>2 critics, clipped double-Q<br>auto-α with target entropy $-\dim(\mathcal A)$<br>1 gradient step per env step</td></tr>
</table></div>

<h2>Glossary</h2>
<div class="table-wrap"><table class="compare">
<tr><th>Term</th><th>Definition</th></tr>
<tr><td>Advantage $A(s,a)$</td><td>$Q(s,a) - V(s)$: how much better an action is than the policy's average.</td></tr>
<tr><td>Bootstrapping</td><td>Building a target from current estimates ($r + \gamma V(s')$) instead of complete returns.</td></tr>
<tr><td>Behavior / target policy</td><td>The policy generating the data vs. the policy being learned about. Equal ⇒ on-policy.</td></tr>
<tr><td>Credit assignment</td><td>Determining which earlier actions caused a later reward.</td></tr>
<tr><td>Critic</td><td>A learned value function used to evaluate/improve the actor (policy).</td></tr>
<tr><td>Deadly triad</td><td>Function approximation + bootstrapping + off-policy data: risk of divergence.</td></tr>
<tr><td>Discount γ</td><td>Geometric weight on future rewards; effective horizon ≈ $1/(1-\gamma)$.</td></tr>
<tr><td>Entropy bonus</td><td>Term $c\,\mathcal H(\pi)$ added to the objective to keep policies stochastic.</td></tr>
<tr><td>Episode / truncation</td><td>A trajectory ending in a terminal state; truncation (time limit) is <em>not</em> a terminal state.</td></tr>
<tr><td>GPI</td><td>Generalized policy iteration: interleaved evaluation and improvement — the template of almost all RL.</td></tr>
<tr><td>Importance ratio</td><td>$\pi(a|s)/b(a|s)$, reweights samples from $b$ to estimate expectations under $\pi$.</td></tr>
<tr><td>Markov property</td><td>The current state contains all information relevant for the future.</td></tr>
<tr><td>Model</td><td>An estimate of the dynamics $p(s',r|s,a)$; given or learned.</td></tr>
<tr><td>Off-/on-/offline</td><td>Learning from other policies' data / own current data / a fixed dataset only.</td></tr>
<tr><td>Over-optimization</td><td>Exploiting errors of a proxy reward so that true performance drops (Goodhart).</td></tr>
<tr><td>Policy</td><td>$\pi(a|s)$: the agent's (possibly stochastic) mapping from states to actions.</td></tr>
<tr><td>Regret</td><td>Cumulative reward lost relative to the optimal policy.</td></tr>
<tr><td>Replay buffer</td><td>Memory of past transitions sampled for off-policy updates.</td></tr>
<tr><td>Reward model</td><td>A learned proxy for human judgment, usually from pairwise preferences.</td></tr>
<tr><td>Target network</td><td>A lagged copy of the value network used to compute stable bootstrap targets.</td></tr>
<tr><td>Trust region</td><td>A limit on how far one update may move the policy, typically in KL.</td></tr>
<tr><td>Value function</td><td>Expected return from a state ($V$) or state–action pair ($Q$) under a policy.</td></tr>
</table></div>

<h2>Further reading</h2>
<ul>
<li>R. Sutton & A. Barto, <a href="http://incompleteideas.net/book/the-book-2nd.html" target="_blank" rel="noopener"><em>Reinforcement Learning: An Introduction</em></a>, 2nd ed. (2018) — the textbook; Chapters 1–6 of this playground follow its notation.</li>
<li>OpenAI, <a href="https://spinningup.openai.com" target="_blank" rel="noopener">Spinning Up in Deep RL</a> — concise derivations and reference implementations of PG, PPO, DDPG, TD3, SAC.</li>
<li>S. Levine, <a href="https://rail.eecs.berkeley.edu/deeprlcourse/" target="_blank" rel="noopener">CS 285: Deep Reinforcement Learning</a> (Berkeley) — lectures covering model-based, offline and inverse RL in depth.</li>
<li>S. Huang et al., <a href="https://github.com/vwxyzjn/cleanrl" target="_blank" rel="noopener">CleanRL</a> — single-file, readable implementations; and “The 37 implementation details of PPO”.</li>
<li>N. Lambert, <a href="https://rlhfbook.com" target="_blank" rel="noopener"><em>Reinforcement Learning from Human Feedback</em></a> — the LLM post-training pipeline in detail.</li>
<li>D. Bertsekas, <em>Reinforcement Learning and Optimal Control</em>; Cs. Szepesvári, <em>Algorithms for Reinforcement Learning</em> — for the theory.</li>
</ul>
`;

  RL.chapters.push({
    id: 'cheatsheet', num: 10, part: 'Reference', title: 'Cheat Sheet & Glossary',
    mount(root) {
      RL.page.head(root, {
        num: 10, title: 'Cheat sheet, algorithm chooser & glossary',
        lede: 'Everything on one page: the equations worth memorizing, a decision table for picking an algorithm, default hyperparameters, and definitions.',
        tags: [['reference']],
      });
      RL.page.prose(root, html, 'concepts.wide');
      return null;
    },
  });
})(window.RL);
