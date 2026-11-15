/* Chapter 4: function approximation and Deep Q-Networks. */
(function (RL) {
  'use strict';

  const intro = String.raw`
<p>CartPole has a continuous 4-dimensional state $(x, \dot x, \theta, \dot\theta)$. A table cannot store a value for every state, and even if you discretized it, the agent would have to visit every cell to learn about it. The fix is <b>generalization</b>: represent $Q(s,a;\theta)$ with a parametric function — here a small neural network with one hidden layer — so that learning about one state also changes the predictions for similar states. The two heatmaps below the cart are slices of that function: they show what the network currently believes about <em>all</em> pole angles and angular velocities at once.</p>`;

  const theory = String.raw`
<h2>From Q-learning to a regression problem</h2>
<p>DQN turns the Q-learning update into supervised regression on a moving target. For transitions $(s,a,r,s')$ sampled from a replay buffer $\mathcal D$:</p>
$$\mathcal L(\theta) = \mathbb E_{(s,a,r,s')\sim\mathcal D}\Big[\big(\underbrace{r + \gamma (1-d)\max_{a'} Q(s',a';\theta^-)}_{y,\ \text{held fixed}} - Q(s,a;\theta)\big)^2\Big]$$
<p>where $d$ marks a true terminal state and $\theta^-$ are the parameters of a slowly updated <b>target network</b>. The gradient only flows through $Q(s,a;\theta)$, not through the target $y$ — a <em>semi-gradient</em> method.</p>

<details class="deep math"><summary><span class="lbl">Math</span> Why semi-gradient, and the deadly triad</summary><div class="body">
<p>The "true" gradient of the squared Bellman error would also differentiate the target, $\nabla_\theta\big(y(\theta) - Q(s,a;\theta)\big)^2$. These <em>residual-gradient</em> methods (Baird, 1995) are genuine gradient descent and therefore stable, but they converge slowly and, with stochastic transitions, to the wrong answer unless one can sample two independent successors of the same state (the "double sampling" problem). Semi-gradient TD instead treats the target as a label. It is not the gradient of any fixed objective, and nothing guarantees convergence.</p>
<p>Indeed, Sutton & Barto call the combination of</p>
<ol><li><b>function approximation</b> (generalization across states),</li><li><b>bootstrapping</b> (targets built from current estimates), and</li><li><b>off-policy</b> training (data distribution ≠ target policy's distribution)</li></ol>
<p>the <b>deadly triad</b>: with all three present, value estimates can diverge to infinity even in tiny linear examples (Baird's counterexample; Tsitsiklis & Van Roy, 1997). Remove any one leg and things are much safer — Monte Carlo targets (no bootstrapping), tabular values (no generalization) or on-policy linear TD (which provably converges). DQN keeps all three and instead adds engineering that keeps the iteration close to stable fitted value iteration.</p>
</div></details>

<h2>The two stabilizers</h2>
<ul>
<li><b>Experience replay.</b> Consecutive CartPole states are almost identical. Training on them in order is like SGD on a sorted dataset: the network overfits to the current part of state space and forgets the rest. A replay buffer of the last $10^4$–$10^6$ transitions, sampled uniformly, decorrelates minibatches, reuses each sample many times (data efficiency) and averages over many past policies (smoother distribution shift). Replay is only possible because Q-learning is off-policy.</li>
<li><b>Target network.</b> If the target $y$ is computed with the same weights being updated, each gradient step moves the target too, often in the same direction — a feedback loop that makes values chase themselves. Keeping a lagged copy $\theta^-$ (copied every $C$ steps, or Polyak-averaged $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$) turns each phase into an ordinary regression problem: an approximate step of value iteration.</li>
</ul>
<p>In the lab, switch each stabilizer off and watch what happens. Without replay, learning usually fails completely; without a target network it often learns, then collapses and oscillates.</p>

<h2>Overestimation and Double DQN</h2>
<p>The max in the target picks out actions whose values are overestimated by noise (Chapter 3's maximization bias), and with function approximation the errors are correlated and compound through bootstrapping. <b>Double DQN</b> (van Hasselt et al., 2016) uses the online network to <em>select</em> and the target network to <em>evaluate</em>:</p>
$$y = r + \gamma\, Q\big(s', \arg\max_{a'} Q(s',a';\theta);\ \theta^-\big).$$
<p>It costs nothing and is almost always used. The diagnostics chart "max Q" lets you compare the learned values with the true scale: a pole balanced forever is worth $\sum_k \gamma^k = 1/(1-\gamma) = 100$ at $\gamma = 0.99$.</p>

<details class="deep practice"><summary><span class="lbl">Practice</span> Details that make or break a DQN implementation</summary><div class="body">
<ul>
<li><b>Termination vs. truncation.</b> CartPole stops at 500 steps. That is a time limit, not a failure: the state after it still has value. The lab only zeroes the bootstrap ($d = 1$) when the pole actually falls. Treating time-outs as terminal teaches the agent that "being at step 499" is bad — an easy bug that costs a lot of performance (Pardo et al., 2018).</li>
<li><b>Input and reward scale.</b> Observations are normalized to roughly $[-1,1]$. Rewards in Atari were clipped to $[-1,1]$; with large rewards, the Huber loss (quadratic near zero, linear beyond) limits gradient magnitudes.</li>
<li><b>Exploration schedule.</b> ε decays from 1 to ~0.05; the training curve includes exploratory actions and is therefore noisier and lower than a greedy evaluation. Switch “learning” off to see the greedy policy.</li>
<li><b>Seeds matter.</b> Run the same configuration three times: deep RL results vary a lot between seeds (Henderson et al., 2018, “Deep RL that matters”). Always report several seeds.</li>
<li><b>Catastrophic drops</b> after solving the task are common for DQN on CartPole: once the buffer is full of successful episodes, failure states are rare in the data and get forgotten. Larger buffers, lower learning rates and early stopping help.</li>
</ul>
</div></details>

<h2>The DQN family today</h2>
<div class="table-wrap"><table class="compare">
<tr><th>Extension</th><th>Idea</th><th>Status</th></tr>
<tr><td>Double DQN</td><td>decouple argmax selection from evaluation</td><td><span class="status core">standard</span></td></tr>
<tr><td>Dueling networks</td><td>$Q(s,a) = V(s) + A(s,a) - \frac1{|\mathcal A|}\sum_{a'} A(s,a')$: learn state value separately</td><td><span class="status core">standard</span></td></tr>
<tr><td>Prioritized replay (PER)</td><td>sample transitions with large $|\delta|$ more often; correct with importance weights</td><td><span class="status core">common</span></td></tr>
<tr><td>n-step returns</td><td>targets $\sum_{k\lt n}\gamma^k r_{t+k+1} + \gamma^n \max Q$: faster credit propagation</td><td><span class="status core">standard</span></td></tr>
<tr><td>Distributional RL (C51, QR-DQN, IQN)</td><td>learn the full return distribution $Z(s,a)$, not just its mean</td><td><span class="status core">common</span></td></tr>
<tr><td>Noisy nets</td><td>learned parametric noise in weights instead of ε-greedy</td><td><span class="status niche">niche</span></td></tr>
<tr><td>Rainbow (2018)</td><td>all of the above combined</td><td><span class="status niche">reference baseline</span></td></tr>
<tr><td>R2D2, Agent57, BBF</td><td>recurrent replay at scale; exploration meta-controllers; data-efficient scaling</td><td><span class="status niche">research</span></td></tr>
</table></div>
<p>Value-based deep RL remains the method of choice for discrete actions when data is expensive (off-policy reuse) and is the backbone of offline RL. Its weakness is the $\max_{a'}$: with continuous actions it becomes an optimization problem per step. Continuous-control methods (DDPG → TD3, SAC; Chapter 9) solve this by training a policy network to output the argmax — which leads us to policy-based methods.</p>

<details class="deep history"><summary><span class="lbl">History</span> Before deep Q-learning</summary><div class="body">
<p>Linear function approximation with hand-designed features — <b>tile coding</b> (coarse overlapping grids), radial basis functions, Fourier bases — dominated until ~2013, together with least-squares methods (LSTD, LSPI) that solve for the TD fixed point in closed form. <b>Neural fitted Q-iteration</b> (Riedmiller, 2005) already trained an MLP on a replay set with fixed targets. DQN (Mnih et al., 2013/2015) scaled this to raw Atari pixels with a convolutional network, one architecture and hyperparameter set for 49 games — the moment deep RL began.</p>
</div></details>
`;

  RL.chapters.push({
    id: 'dqn', num: 4, part: 'Learning from Experience', title: 'Deep Q-Networks',
    mount(root) {
      RL.page.head(root, {
        num: 4, title: 'Function approximation & Deep Q-Networks',
        lede: 'Replace the Q-table with a neural network and Q-learning becomes regression on a moving target. Toggle replay and the target network to see why DQN needs both.',
        tags: [['value-based'], ['off-policy'], ['neural network'], ['deadly triad']],
      });
      RL.page.prose(root, intro);
      const lab = RL.cartpoleLab(root, {
        kind: 'dqn', title: 'DQN lab',
        vizNote: 'Both maps are computed by feeding a grid of (θ, θ̇) states through the Q-network. Early on they are noise; a trained network shows high value near the centre and a clean diagonal decision boundary: push in the direction the pole is falling.',
      });
      const base = { lr: 1e-3, gamma: 0.99, batch: 32, tau: 0.005, epsEnd: 0.05, epsSteps: 10000, useReplay: true, useTarget: true, double: true, loss: 'mse', width: 64 };
      const cfgWith = (o) => Object.assign({}, base, o);
      RL.ui.experiments(root, [
        { title: 'Default DQN solves CartPole', text: 'Replay + Polyak target + Double DQN, max speed. Expect the first 500-step episodes after ~10–20k environment steps (well under a minute). Then switch learning off to watch the greedy policy.', apply: () => lab.configure(cfgWith({}), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'No experience replay', text: 'Train only on the latest transition. The network sees a stream of nearly identical, correlated states and keeps overwriting what it learned. Returns stay near random (≈ 10–30).', apply: () => lab.configure(cfgWith({ useReplay: false }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'No target network', text: 'Bootstrap from the network being trained. It often still learns, but watch the max-Q chart overshoot and the returns oscillate or collapse.', apply: () => lab.configure(cfgWith({ useTarget: false }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Learning rate too high', text: 'Learning rate $10^{-2}$. Value estimates swing wildly (watch max Q and the loss) — bootstrapping amplifies every error.', apply: () => lab.configure(cfgWith({ lr: 1e-2 }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Shorter horizon', text: '$\\gamma = 0.9$: values saturate at $1/(1-\\gamma) = 10$ instead of 100. Credit assignment is easier, but the agent only “sees” ~10 steps ahead — usually enough here. Compare the max-Q scale.', apply: () => lab.configure(cfgWith({ gamma: 0.9 }), { speedKey: 'max' }), labEl: lab.lab.el },
        { title: 'Explore less', text: 'ε decays to 0 within 1000 steps. Sometimes it still works (CartPole is forgiving), often it gets stuck. Exploration is a design decision, not a detail.', apply: () => lab.configure(cfgWith({ epsSteps: 1000, epsEnd: 0 }), { speedKey: 'max' }), labEl: lab.lab.el },
      ]);
      RL.page.prose(root, theory);
      return () => lab.destroy();
    },
  });
})(window.RL);
