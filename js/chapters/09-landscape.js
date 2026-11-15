/* Chapter 9: the wider landscape — continuous control, offline RL, imitation, exploration, multi-agent, and what is outdated. */
(function (RL) {
  'use strict';

  const html = String.raw`
<h2>Continuous control: DDPG → TD3 → SAC</h2>
<p>With continuous actions, Q-learning's $\max_{a'} Q(s',a')$ is an optimization problem at every step. The off-policy actor-critic family trains a policy network to approximately solve it, keeping the sample efficiency of replay buffers.</p>
<ul>
<li><b>DDPG</b> (Lillicrap et al., 2016) — a deterministic actor $\mu_\theta(s)$ trained with the <em>deterministic policy gradient</em> $\nabla_\theta J = \mathbb E_s\big[\nabla_a Q(s,a)\big|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)\big]$ (Silver et al., 2014): "move the action uphill on the critic". Brittle; largely superseded.</li>
<li><b>TD3</b> (Fujimoto et al., 2018) fixes DDPG's overestimation with three tricks: <em>clipped double-Q</em> targets $y = r + \gamma\min_{i=1,2} Q_{\phi_i^-}(s', \tilde a')$ (Chapter 3's double learning, made pessimistic), <em>target-policy smoothing</em> (noise on $\tilde a'$), and <em>delayed</em> actor updates.</li>
<li><b>SAC</b> (Haarnoja et al., 2018) optimizes the maximum-entropy objective $\mathbb E\big[\sum_t \gamma^t (r_t + \alpha\mathcal H(\pi(\cdot|s_t)))\big]$ with a stochastic Gaussian actor (tanh-squashed), the soft Bellman backup $y = r + \gamma\big(\min_i Q_{\phi_i^-}(s',a') - \alpha\log\pi(a'|s')\big)$, a reparameterized actor loss $\mathbb E_{s,\epsilon}\big[\alpha\log\pi_\theta(\tilde a_\theta(s,\epsilon)|s) - Q(s,\tilde a_\theta(s,\epsilon))\big]$, and automatic tuning of $\alpha$ toward a target entropy. Robust, sample-efficient, and the default for off-policy continuous control and real-robot learning.</li>
</ul>
<details class="deep math"><summary><span class="lbl">Math</span> Score-function vs. reparameterization gradients</summary><div class="body">
<p>Two ways to differentiate $\mathbb E_{a\sim\pi_\theta}[f(a)]$:</p>
<p><b>Score function</b> (REINFORCE, PPO):</p>
$$\nabla_\theta\,\mathbb E_{a\sim\pi_\theta}[f(a)] = \mathbb E_{a\sim\pi_\theta}\big[f(a)\,\nabla_\theta\log\pi_\theta(a)\big]$$
<p><b>Pathwise / reparameterization</b> (SAC, DDPG), with $a = \mu_\theta + \sigma_\theta\epsilon$:</p>
$$\nabla_\theta\,\mathbb E_{a\sim\pi_\theta}[f(a)] = \mathbb E_{\epsilon\sim\mathcal N(0,I)}\big[\nabla_a f(a)\,\nabla_\theta(\mu_\theta + \sigma_\theta\epsilon)\big]$$
<p>The score-function estimator needs only samples of $f$ (any reward, non-differentiable, discrete actions) but has high variance. The pathwise estimator needs a differentiable $f$ — here the learned critic $Q$ — and has much lower variance. This is the deepest technical difference between on-policy PG methods and off-policy actor-critics: the latter push gradients <em>through the critic</em>, so their quality is bounded by the critic's.</p>
</div></details>

<h2>Offline RL: learning from a fixed dataset</h2>
<p>In robotics, healthcare or recommendation, exploration is expensive or dangerous, but logs are plentiful. Naively running Q-learning on a fixed dataset fails: the $\max_{a'}$ queries actions the data never contains, the critic's errors there are never corrected, and the policy exploits them (<b>extrapolation error / distribution shift</b>). Solutions constrain the policy or the values to the data's support:</p>
<ul>
<li><b>Policy constraints</b>: BCQ, BEAR, TD3+BC (add a behavior-cloning term $-\lambda\|\pi(s) - a\|^2$ to the actor loss — strikingly simple and strong).</li>
<li><b>Pessimistic values</b>: CQL adds $\alpha\big(\mathbb E_{a\sim\pi}Q(s,a) - \mathbb E_{a\sim\mathcal D}Q(s,a)\big)$ to push down values of out-of-distribution actions.</li>
<li><b>In-sample learning</b>: IQL never evaluates unseen actions; it fits an upper expectile of $Q$ over dataset actions and extracts the policy by advantage-weighted regression.</li>
<li><b>Sequence modeling</b>: Decision Transformer conditions a transformer on desired return-to-go and imitates — RL as conditional generative modeling.</li>
</ul>
<p>DPO (Chapter 8) is an offline method too, and shares the same weakness: it cannot correct errors outside the data.</p>

<h2>Imitation learning and robot foundation models</h2>
<p><b>Behavior cloning</b> (supervised learning on expert state–action pairs) is the strongest baseline whenever demonstrations exist — it is how LLM SFT works. Its flaw is <b>compounding error</b>: small mistakes lead to states the expert never visited, so errors grow as $O(T^2\epsilon)$ over horizon $T$. <b>DAgger</b> (Ross et al., 2011) queries the expert on the learner's own states and reduces this to $O(T\epsilon)$. <b>Inverse RL</b> and <b>GAIL</b> recover a reward from demonstrations and then run RL. Modern robot learning is dominated by large imitation-trained policies — diffusion policies and vision-language-action (VLA) models — with RL increasingly used as a fine-tuning stage, mirroring the LLM pipeline (pretrain → imitate → RL).</p>

<h2>More areas, in brief</h2>
<div class="cards">
  <div class="card"><h4>Exploration bonuses <span class="status niche">research</span></h4><p>Intrinsic rewards for novelty: pseudo-counts, curiosity as prediction error (ICM), random network distillation (RND), Go-Explore. Essential for sparse-reward games like Montezuma's Revenge; less used where rewards are dense or demonstrations exist.</p></div>
  <div class="card"><h4>Multi-agent RL <span class="status core">core in games</span></h4><p>Self-play (AlphaZero), league training against past selves (AlphaStar), centralized training with decentralized execution (MADDPG, QMIX, MAPPO). Imperfect-information games use counterfactual regret minimization (Libratus, Pluribus for poker). Non-stationarity: other learners are part of your environment.</p></div>
  <div class="card"><h4>Sim-to-real <span class="status core">core in robotics</span></h4><p>Train PPO in massively parallel GPU simulation with domain randomization (masses, friction, latencies), often a privileged "teacher" distilled into a sensor-only "student". The standard route to legged locomotion and dexterous manipulation.</p></div>
  <div class="card"><h4>Distributional RL <span class="status core">common</span></h4><p>Learn the distribution of returns $Z(s,a)$ (C51, QR-DQN, IQN) via a distributional Bellman operator. Better representations and risk-sensitive control; used in game-playing agents such as GT Sophy (quantile-regression SAC).</p></div>
  <div class="card"><h4>Safe & constrained RL <span class="status rising">growing</span></h4><p>Constrained MDPs: maximize reward subject to $\mathbb E[\text{cost}] \le d$, usually via Lagrangian methods (PPO-Lagrangian) or CPO. Shielding, safe exploration, and the safety of RL-trained LLMs (reward hacking, specification gaming).</p></div>
  <div class="card"><h4>Partial observability <span class="status core">core</span></h4><p>When observations are not Markov, policies need memory: frame stacking, recurrent (R2D2) or transformer policies, belief states. LLM agents are POMDP agents whose memory is the context window.</p></div>
  <div class="card"><h4>Meta-RL & in-context RL <span class="status niche">research</span></h4><p>Learn to learn: RL² (an RNN whose hidden state performs adaptation), MAML (learn an initialization), and "algorithm distillation". Large sequence models that adapt in context have revived interest.</p></div>
  <div class="card"><h4>Hierarchical RL <span class="status niche">niche</span></h4><p>Options framework (temporally extended actions with initiation set, policy and termination), feudal networks, goal-conditioned low-level policies (HIRO, HAC). Elegant, but rarely beats flat agents in practice; temporal abstraction in LLM agents (tools, sub-tasks) is a modern analogue.</p></div>
  <div class="card"><h4>Evolution strategies <span class="status niche">niche</span></h4><p>Black-box optimization of policy parameters (OpenAI-ES, CMA-ES): perturb, evaluate, move toward better perturbations. Embarrassingly parallel and indifferent to reward sparsity or horizon, but sample-hungry for large networks.</p></div>
  <div class="card"><h4>RL theory <span class="status niche">theory</span></h4><p>Sample-complexity and regret bounds for MDPs (UCRL, R-max, Q-learning with UCB bonuses), function approximation theory (linear MDPs, Bellman rank/eluder dimension), and convergence of policy gradients (e.g. global convergence of softmax PG, natural PG rates).</p></div>
</div>

<h2>Largely superseded — know the names, use the successors</h2>
<div class="table-wrap"><table class="compare">
<tr><th>Method</th><th>What it did</th><th>Replaced by</th></tr>
<tr><td>Tabular methods for large problems</td><td>one value per state</td><td>function approximation (Chapter 4+)</td></tr>
<tr><td>Tile coding, RBFs, LSTD/LSPI</td><td>linear features, closed-form TD solutions</td><td>neural networks; still useful for theory and tiny problems</td></tr>
<tr><td>Eligibility traces (TD(λ), Q(λ), SARSA(λ))</td><td>backward-view multi-step credit</td><td>n-step returns, GAE, Retrace/V-trace</td></tr>
<tr><td>Vanilla DQN</td><td>first deep value-based agent</td><td>Double/Dueling/PER/n-step (Rainbow-style) DQN</td></tr>
<tr><td>REINFORCE without baseline</td><td>Monte Carlo policy gradient</td><td>actor-critic, PPO; RLOO/GRPO baselines for LLMs</td></tr>
<tr><td>Natural actor-critic, TRPO</td><td>second-order trust regions</td><td>PPO</td></tr>
<tr><td>A3C</td><td>asynchronous parallel actor-critic</td><td>A2C, PPO, IMPALA</td></tr>
<tr><td>DDPG, NAF</td><td>continuous-action Q-learning</td><td>TD3, SAC</td></tr>
<tr><td>PILCO, guided policy search</td><td>model-based policy search</td><td>Dreamer, TD-MPC, MBPO</td></tr>
<tr><td>Gittins indices, Boltzmann exploration</td><td>classic bandit exploration</td><td>UCB, Thompson sampling</td></tr>
</table></div>
`;

  RL.chapters.push({
    id: 'landscape', num: 9, part: 'Beyond the Basics', title: 'The Wider Landscape',
    mount(root) {
      RL.page.head(root, {
        num: 9, title: 'The wider landscape',
        lede: 'Continuous control, learning from logs, learning from demonstrations, many agents — the main branches beyond the core path, each with its key idea and where it stands today. And a list of methods you will meet in papers but should not reach for first.',
        tags: [['SAC · TD3'], ['offline RL'], ['imitation'], ['status map']],
      });
      RL.page.prose(root, html);
      return null;
    },
  });
})(window.RL);
