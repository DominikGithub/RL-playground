/* Chapter 0: the RL problem, the map of methods, how to use the playground. */
(function (RL) {
  'use strict';

  const loopSVG = String.raw`
<svg class="loop-diagram" viewBox="0 0 720 230" role="img" aria-label="Agent-environment interaction loop">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/>
    </marker>
  </defs>
  <rect x="40" y="60" width="200" height="110" rx="14" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
  <text x="140" y="98" text-anchor="middle" style="fill:var(--accent);font:700 18px system-ui">Agent</text>
  <text x="140" y="124" text-anchor="middle" style="fill:var(--text);font:14px system-ui">policy π(a | s)</text>
  <text x="140" y="146" text-anchor="middle" style="fill:var(--muted);font:12px system-ui">(+ value fn, model, …)</text>
  <rect x="480" y="60" width="200" height="110" rx="14" style="fill:var(--panel2);stroke:var(--border);stroke-width:1.5"/>
  <text x="580" y="98" text-anchor="middle" style="fill:var(--text);font:700 18px system-ui">Environment</text>
  <text x="580" y="124" text-anchor="middle" style="fill:var(--text);font:14px system-ui">p(s′, r | s, a)</text>
  <text x="580" y="146" text-anchor="middle" style="fill:var(--muted);font:12px system-ui">unknown to the agent</text>
  <path d="M240 85 C 330 30, 390 30, 478 85" style="fill:none;stroke:var(--muted);stroke-width:2" marker-end="url(#arr)"/>
  <text x="360" y="38" text-anchor="middle" style="fill:var(--text);font:600 15px system-ui">action aₜ</text>
  <path d="M480 150 C 390 205, 330 205, 242 150" style="fill:none;stroke:var(--muted);stroke-width:2" marker-end="url(#arr)"/>
  <text x="360" y="214" text-anchor="middle" style="fill:var(--text);font:600 15px system-ui">next state sₜ₊₁ , reward rₜ₊₁</text>
  <circle r="6" style="fill:var(--accent)">
    <animateMotion dur="3.2s" repeatCount="indefinite" path="M240 85 C 330 30, 390 30, 478 85 L480 150 C 390 205, 330 205, 242 150 Z"/>
  </circle>
</svg>`;

  const html = String.raw`
<h2>The interaction loop</h2>
${loopSVG}
<p>At every time step the agent observes a state $s_t$, picks an action $a_t \sim \pi(\cdot\mid s_t)$, and the environment answers with a reward $r_{t+1}$ and a next state $s_{t+1}\sim p(\cdot \mid s_t,a_t)$. The agent's goal is to maximize the expected <b>return</b> — the (discounted) sum of future rewards:</p>
$$G_t \;=\; r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots \;=\; \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}, \qquad J(\pi) = \mathbb{E}_{\pi}\left[G_0\right].$$
<p>That is the entire problem statement. Everything in this playground — Bellman equations, Q-learning, policy gradients, PPO, MCTS, RLHF — is a different answer to the question <em>"how do I find a policy with high $J(\pi)$ when I can only learn about the environment by interacting with it?"</em></p>

<details class="deep basics"><summary><span class="lbl">Basics</span> Vocabulary refresher</summary><div class="body">
<table class="compare">
<tr><th>Term</th><th>Meaning</th></tr>
<tr><td>State $s$ / observation $o$</td><td>What the agent knows about the world. If the observation contains everything relevant for the future, it is <em>Markov</em>; otherwise the problem is partially observable (POMDP) and agents need memory (RNN/transformer, frame stacking).</td></tr>
<tr><td>Action $a$</td><td>Discrete (left/right, a token from a vocabulary) or continuous (joint torques).</td></tr>
<tr><td>Policy $\pi(a\mid s)$</td><td>The behavior: a mapping from states to (distributions over) actions. Deterministic policies are written $a=\mu(s)$.</td></tr>
<tr><td>Reward $r$</td><td>Scalar feedback. The <em>reward hypothesis</em>: all goals can be expressed as maximizing expected cumulative reward. Designing rewards is hard (see reward hacking in Chapter 8).</td></tr>
<tr><td>Return $G_t$</td><td>Sum of (discounted) future rewards from time $t$.</td></tr>
<tr><td>Discount $\gamma\in[0,1]$</td><td>Trades off immediate vs. future reward; also keeps infinite sums finite. Effective horizon ≈ $1/(1-\gamma)$ steps.</td></tr>
<tr><td>Episode</td><td>A trajectory from a start state to a terminal state (or a time limit).</td></tr>
<tr><td>Value $V^\pi(s)$, $Q^\pi(s,a)$</td><td>Expected return from $s$ (and first taking $a$), then following $\pi$.</td></tr>
<tr><td>Model</td><td>The agent's own estimate of $p(s',r\mid s,a)$. Agents that learn/use one are <em>model-based</em>.</td></tr>
</table>
</div></details>

<h2>Why RL is harder than supervised learning</h2>
<ol>
<li><b>Evaluative, not instructive feedback.</b> A label says <em>what</em> the right answer was. A reward only says <em>how good</em> your answer was — you never see what the other actions would have earned. This forces <b>exploration</b> (Chapter 2).</li>
<li><b>The data distribution depends on the learner.</b> Changing the policy changes which states you visit. Samples are neither independent nor identically distributed; improving on yesterday's data can make tomorrow's data worse. This is behind most instability in deep RL (Chapters 4 and 6).</li>
<li><b>Delayed consequences.</b> A move made at step 3 may only pay off at step 300. Assigning credit across time is what value functions and bootstrapping are for (Chapters 1 and 3).</li>
<li><b>Non-stationary targets.</b> In value-based methods, the regression targets are themselves computed from the current estimates (“bootstrapping”).</li>
</ol>

<h2>A map of the territory</h2>
<p>Almost every algorithm can be located by answering three questions. Keep them in mind; each chapter flags which answers it chooses.</p>
<div class="taxonomy">
  <div class="tax-col"><h4>1 · What is learned?</h4><div class="q">What object represents the solution?</div>
  <ul><li><b>Value-based</b>: learn $Q^*(s,a)$, act greedily (Q-learning, DQN).</li><li><b>Policy-based</b>: learn $\pi_\theta$ directly by gradient ascent on $J$ (REINFORCE).</li><li><b>Actor-critic</b>: both — a policy updated using a learned value baseline (A2C, PPO, SAC).</li><li><b>Model-based</b>: learn/use $p(s',r\mid s,a)$ and plan (Dyna, MCTS, MuZero, Dreamer).</li></ul></div>
  <div class="tax-col"><h4>2 · Where does the data come from?</h4><div class="q">Whose behavior generated the experience?</div>
  <ul><li><b>On-policy</b>: data from the current policy only, discarded after the update (SARSA, REINFORCE, PPO).</li><li><b>Off-policy</b>: learn about one policy from data of another, reuse old data (Q-learning, DQN, SAC).</li><li><b>Offline</b>: a fixed dataset, no further interaction (CQL, IQL, DPO is offline too).</li></ul></div>
  <div class="tax-col"><h4>3 · How is the return estimated?</h4><div class="q">What target does the learner regress on?</div>
  <ul><li><b>Monte Carlo</b>: wait for the actual return $G_t$ — unbiased, high variance.</li><li><b>Temporal difference</b>: bootstrap $r + \gamma V(s')$ — biased, low variance.</li><li><b>In between</b>: $n$-step returns, TD($\lambda$), GAE.</li><li><b>Planning</b>: simulate with a model (search trees, imagined rollouts).</li></ul></div>
</div>

<h2>Which ideas matter today?</h2>
<p>RL has accumulated decades of algorithms. Many are now mostly of historical or pedagogical value. This playground spends its time on the ideas that carry modern systems, and briefly names the rest so that you recognize them in papers.</p>
<div class="table-wrap"><table class="compare">
<tr><th>Idea / algorithm</th><th>Status</th><th>Where it lives today</th><th>Chapter</th></tr>
<tr><td>Bellman equations, value functions, discounting</td><td><span class="status core">foundation</span></td><td>Every algorithm, including LLM critics and reward shaping</td><td>1</td></tr>
<tr><td>Exploration: UCB, Thompson sampling</td><td><span class="status core">core</span></td><td>Recommenders, A/B testing, ads; exploration bonuses in deep RL; best-of-n sampling ideas</td><td>2</td></tr>
<tr><td>TD learning, Q-learning</td><td><span class="status core">foundation</span></td><td>Inside DQN, SAC, offline RL; critics of all actor-critics</td><td>3</td></tr>
<tr><td>DQN family (Double, Dueling, PER, Rainbow, distributional)</td><td><span class="status niche">mature</span></td><td>Discrete-action off-policy problems, Atari-style benchmarks</td><td>4</td></tr>
<tr><td>Policy gradient, actor-critic, GAE</td><td><span class="status core">core</span></td><td>The workhorse family: robotics in simulation, games, LLMs</td><td>5</td></tr>
<tr><td>PPO (and trust-region thinking)</td><td><span class="status core">core</span></td><td>Default on-policy algorithm; classic RLHF; sim-to-real locomotion</td><td>6</td></tr>
<tr><td>MCTS + learned models (AlphaZero, MuZero), world models (Dreamer)</td><td><span class="status core">core</span></td><td>Games, planning, test-time search, sample-efficient control</td><td>7</td></tr>
<tr><td>RLHF, DPO, GRPO, RL from verifiable rewards</td><td><span class="status rising">frontier</span></td><td>Post-training of language and reasoning models</td><td>8</td></tr>
<tr><td>SAC / TD3 (continuous control), offline RL, imitation learning</td><td><span class="status core">core</span></td><td>Robotics, real-world control from logged data</td><td>9</td></tr>
<tr><td>Tabular methods in large problems, tile coding, TRPO, A3C, DDPG, vanilla REINFORCE</td><td><span class="status hist">historical</span></td><td>Teaching, theory, and as ancestors of the above</td><td>3–9</td></tr>
</table></div>

<h2>How to use this playground</h2>
<ul>
<li><b>Labs</b> (the bordered panels) run real algorithms live in your browser — no server, no precomputed results. Change hyperparameters while they train.</li>
<li><b>Guided experiments</b> below each lab configure it for a specific lesson. Make a prediction before you press <em>Set up</em>; RL intuition is built by being surprised.</li>
<li><b>Expandable boxes</b>: <span style="color:var(--good)">Basics</span> for refreshers, <span style="color:var(--accent)">Math</span> for derivations, <span style="color:var(--neg)">Practice</span> for engineering details that matter when implementing, <b>History</b> for older methods.</li>
<li>Hover the small <span class="help" style="display:inline-grid">?</span> icons next to controls for an explanation of each hyperparameter.</li>
</ul>
<div class="callout"><strong>Suggested paths.</strong> <em>New to RL:</em> read chapters 1 → 6 in order and open the Basics boxes. <em>Refreshing a degree-level background:</em> skim Chapter 1, then focus on the Math boxes of Chapters 3, 5 and 6 — the value-vs-policy comparison at the end of Chapter 5 ties everything together — and finish with Chapters 7 and 8. <em>Here for LLMs:</em> Chapter 5 (policy gradient theorem, baselines) and Chapter 6 (PPO) are the prerequisites for Chapter 8.</div>
`;

  RL.chapters.push({
    id: 'intro', part: 'Foundations', title: 'The RL Problem',
    mount(root) {
      RL.page.head(root, {
        title: 'Reinforcement Learning, interactively',
        lede: 'A hands-on tour of the ideas behind modern RL — from Bellman equations to PPO, MCTS and the post-training of language models. Every lab runs real algorithms in your browser.',
        tags: [['11 chapters'], ['interactive labs'], ['math on demand'], ['focus on what matters now', 'now']],
      });
      RL.page.prose(root, html);
      return null;
    },
  });
})(window.RL);
