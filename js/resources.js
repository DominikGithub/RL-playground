/* Study material: lectures, slides, books and papers linked next to the sections they deepen.
   Every entry carries a name, author, year and short description so it can be found again if a URL changes.
   All links were checked (HTTP 200, correct title) in November 2026. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { h } = RL.util;

  const yt = (id) => 'https://www.youtube.com/watch?v=' + id;
  const DB = 'https://web.mit.edu/dimitrib/www/';
  const OCW = 'https://ocw.mit.edu/courses/6-231-dynamic-programming-and-stochastic-control-fall-2015/';
  const SILVER = 'https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/';
  const arxiv = (id) => 'https://arxiv.org/abs/' + id;

  /* ---------- sources ---------- */
  const B = 'Dimitri Bertsekas';
  const asu = (year, n, vid, topic, desc) => ({
    kind: 'video', who: B, year,
    short: `Bertsekas L${n} (${year})`,
    title: `ASU “Topics in Reinforcement Learning”, ${year}, Lecture ${n}: ${topic}`,
    desc,
    links: [{ label: 'Video', url: yt(vid) }, { label: 'Slides (PDF)', url: `${DB}RLTopics_${year}_Lect${n}.pdf` }],
  });
  const silver = (n, vid, slide, topic, desc) => ({
    kind: 'video', who: 'David Silver (DeepMind × UCL)', year: 2015,
    short: `Silver L${n}`,
    title: `UCL Course on RL, Lecture ${n}: ${topic}`,
    desc,
    links: [{ label: 'Video', url: yt(vid) }, { label: 'Slides (PDF)', url: SILVER + slide }],
  });
  const ocw = (n, topic) => ({
    kind: 'slides', who: B + ' (MIT OpenCourseWare)', year: 2015,
    short: `MIT 6.231 L${n} slides`,
    title: `MIT 6.231 Dynamic Programming and Stochastic Control, Lecture ${n} slides: ${topic}`,
    desc: 'Lecture slides from Bertsekas’ MIT graduate course (Fall 2015), the classical DP view of the same ideas.',
    links: [{ label: 'Slides', url: `${OCW}resources/mit6_231f15_lec${n}/` }, { label: 'All 6.231 slides', url: `${OCW}pages/lecture-notes/` }],
  });
  /** `read`: [first page, last page, what is there, total PDF pages] of the linked arXiv version. */
  const paper = (short, title, who, year, id, desc, read) => ({ kind: 'paper', who, year, short, title, desc, read, links: [{ label: 'arXiv', url: arxiv(id) }] });

  const S = {
    /* Bertsekas, ASU course (YouTube channel “Dimitri Bertsekas”); slides on his MIT page */
    b25_1: asu(2025, 1, 'AdxhPj0PDHM', 'Course overview: RL and DP, AlphaZero, deterministic DP, examples, applications',
      'The big picture: how RL relates to dynamic programming, and AlphaZero’s off-line training + on-line play as the template for modern RL.'),
    b25_2: asu(2025, 2, 'JiTWTFajYNg', 'Stochastic finite and infinite horizon DP, approximation in value and policy space',
      'Bellman’s equation for stochastic problems, finite vs. infinite horizon, and the two great families: approximation in value space vs. in policy space.'),
    b25_3: asu(2025, 3, 'oQNN292Q4jI', 'LQ problems, approximation in value space, VI and PI, Newton’s method',
      'Value and policy iteration in depth; Bertsekas’ key insight that one-step lookahead is a Newton step for solving Bellman’s equation.'),
    b25_4: asu(2025, 4, 'ooaG6k0phsU', 'POMDP, systems with changing parameters, adaptive control, model predictive control',
      'Partial observability, adaptive control and MPC — RL viewed from the control-theory side.'),
    b25_5: asu(2025, 5, 'y4j-hFdmcKk', 'Deterministic rollout and animations',
      'Rollout: improve any base heuristic by simulating it — the policy-improvement idea behind MCTS, with animated examples.'),
    b25_6: asu(2025, 6, 'QpnT2DWIbls', 'Multistep approximation in value space, constrained rollout, multiagent rollout',
      'Multistep lookahead and rollout variants for constrained and multiagent problems.'),
    b25_8: asu(2025, 8, 'KdX6o9Qi1vc', 'GPT, HMM, and Markov chains: rollout variants for most likely sequence generation',
      'Treats a GPT-style language model as a policy and applies rollout/lookahead to sequence generation — a DP view of LLM decoding.'),
    b25_11: asu(2025, 11, 'ZH4-02iGK40', 'Adversarial problems, minimax rollout, use of MPC methods, computer chess',
      'Two-player/minimax problems, rollout with MPC, and computer chess as a case study of search + learned evaluation.'),
    b25_12: asu(2025, 12, '43CXjDOWBFI', 'Training of cost functions, approximation in policy space, policy gradient methods',
      'Off-line training of value functions and policies, including policy gradient methods, from the DP perspective.'),
    b24_9: asu(2024, 9, 'vRVP8xCGCEM', 'Bayesian optimization and adaptive control with a POMDP approach; Wordle case study',
      'Exploration as a belief-state (POMDP) problem: Bayesian optimization and adaptive control, illustrated with Wordle.'),
    b24_11: asu(2024, 11, 'pdEItwxgQ2Q', 'On-line training, neural networks, and other approximation architectures',
      'Function approximation for value functions: features, neural networks and how they are trained.'),
    b24_12: asu(2024, 12, 'Om45O-JsNHY', 'Off-line training with neural nets for approximate VI and PI; aggregation',
      'Fitted (approximate) value and policy iteration with neural networks — the DP foundation of DQN-style methods.'),
    b24_13: asu(2024, 13, 'T5BR5Peb9Qg', 'Approximate LP; approximation in policy space, policy gradient methods; epilogue',
      'Approximate linear programming, policy-space approximation and policy gradients, plus a course-wide epilogue.'),
    b23_7: asu(2023, 7, '9vcUyk1ihvs', 'Rollout for stochastic problems, Monte Carlo Tree Search, infinite control spaces',
      'Stochastic rollout and its relation to Monte Carlo Tree Search.'),
    bAZ: {
      kind: 'video', who: B, year: 2022, short: 'Bertsekas: Lessons from AlphaZero',
      title: 'Lessons from AlphaZero for Optimal, Model Predictive, and Adaptive Control (MIT lecture + free book)',
      desc: 'Why AlphaZero’s on-line search on top of off-line training works so well, explained via Newton’s method; with the accompanying research monograph.',
      links: [{ label: 'Video', url: yt('A7OGgpuRnuo') }, { label: 'Slides (PDF)', url: DB + 'Slides_Lessons_MIT_Lecture.pdf' }, { label: 'Book (free PDF)', url: DB + 'LessonsfromAlphazero.pdf' }],
    },
    bNewton: {
      kind: 'video', who: B, year: 2025, short: 'Bertsekas: Newton step & MPC',
      title: 'Reinforcement Learning, Model Predictive Control, and the Newton Step for Solving Bellman’s Equation',
      desc: 'Lookahead/rollout as a Newton step on Bellman’s equation; unifies RL, MPC and computer chess.',
      links: [{ label: 'Video', url: yt('ZBRouvMat2Q') }, { label: 'Slides (PDF)', url: DB + 'MPC.CHESS.MIT.Harvard.pdf' }],
    },
    bTD: {
      kind: 'video', who: B, year: 2017, short: 'Bertsekas: Proximal & TD methods',
      title: 'Proximal Algorithms and Temporal Difference Methods',
      desc: 'Research talk connecting TD(λ) with proximal algorithms for solving large linear (Bellman) equations.',
      links: [{ label: 'Video', url: yt('TEEjzd4l7k0') }, { label: 'Slides (PDF)', url: DB + 'Proximal_TD_Slides.pdf' }],
    },
    bMulti: {
      kind: 'video', who: B, year: 2020, short: 'Bertsekas: Multiagent RL',
      title: 'Multiagent Reinforcement Learning: Rollout and Policy Iteration',
      desc: 'One-hour overview of multiagent rollout and policy iteration (ASU, 2020).',
      links: [{ label: 'Video', url: yt('eqbb6vVlN38') }, { label: 'Slides (PDF)', url: DB + 'Multiagent_RL.pdf' }],
    },
    bBook: {
      kind: 'book', who: B, year: 2025, short: 'Bertsekas: A Course in RL (book)',
      title: 'A Course in Reinforcement Learning, 2nd edition (Athena Scientific) — free PDF',
      desc: 'The textbook of his ASU course: DP-based treatment of approximation in value and policy space, rollout, MPC and AlphaZero-style methods.',
      links: [{ label: 'Book (free PDF)', url: DB + 'RLCOURSECOMPLETE%202ndEDITION.pdf' }],
    },
    bNDP: {
      kind: 'book', who: 'Dimitri Bertsekas & John Tsitsiklis', year: 1996, short: 'Neuro-Dynamic Programming (book)',
      title: 'Neuro-Dynamic Programming — free PDF',
      desc: 'The classic rigorous treatment of TD learning, Q-learning and function approximation (convergence and divergence results).',
      links: [{ label: 'Book (free PDF)', url: DB + 'NDP.pdf' }],
    },
    bKeyIdeas: {
      kind: 'slides', who: B, year: 2019, short: 'Bertsekas: Ten key ideas (slides)',
      title: 'Ten Key Ideas for Reinforcement Learning and Optimal Control',
      desc: 'Extended lecture slides summarizing the core ideas of RL from the DP/optimal-control perspective.',
      links: [{ label: 'Slides (PDF)', url: DB + 'Slides_Extended_RL_Lecture.pdf' }],
    },
    bHub: {
      kind: 'course', who: B, year: '2019–2025', short: 'Bertsekas: all lectures & slides',
      title: 'Bertsekas’ RL course page (MIT site): all ASU lectures 2019–2025, slides, books, talks',
      desc: 'The hub for every year of his ASU RL course (videos + slides), free books and research talks. Use it if any single link above stops working.',
      links: [{ label: 'Course page', url: DB + 'RLbook.html' }, { label: 'Other DP/RL video series', url: 'https://www.mit.edu/~dimitrib/Dynamic_Prog_Videos.html' }],
    },
    ocwAll: {
      kind: 'course', who: B + ' (MIT OpenCourseWare)', year: 2015, short: 'MIT 6.231 (OCW)',
      title: 'MIT 6.231 Dynamic Programming and Stochastic Control (Fall 2015): slides + Approximate DP video lectures',
      desc: 'Complete lecture slides of the MIT course and six recorded lectures on approximate dynamic programming (2014).',
      links: [{ label: 'Lecture slides', url: OCW + 'pages/lecture-notes/' }, { label: 'Video lectures', url: OCW + 'pages/related-video-lectures/' }, { label: 'Complete slides (PDF)', url: OCW + 'resources/mit6_231f15_notes/' }],
    },
    ocw10: ocw(10, 'infinite horizon, stochastic shortest path, Bellman’s equation, value iteration'),
    ocw16: ocw(16, 'value, policy and optimistic policy iteration; asynchronous algorithms'),
    ocw19: ocw(19, 'introduction to approximate DP: approximation in value and policy space, rollout'),
    ocw20: ocw(20, 'approximate fitted VI/PI, projected equations, LSTD, LSPE, TD methods'),
    ocw22: ocw(22, 'aggregation methods and Q-learning'),
    ocw23: ocw(23, 'advanced topics incl. policy-space gradient methods'),

    /* David Silver, UCL 2015 */
    s1: silver(1, '2pWv7GOvuf0', 'intro_rl.pdf', 'Introduction to Reinforcement Learning', 'Agent, environment, reward hypothesis, and the taxonomy of RL agents.'),
    s2: silver(2, 'lfHX2hHRMVQ', 'lecture-2-mdp.pdf', 'Markov Decision Processes', 'Markov processes → MRPs → MDPs, Bellman expectation and optimality equations.'),
    s3: silver(3, 'Nd1-UUMVfz4', 'lecture-3-planning-by-dynamic-programming-.pdf', 'Planning by Dynamic Programming', 'Policy evaluation, policy iteration, value iteration and the contraction argument.'),
    s4: silver(4, 'PnHCvfgC_ZA', 'lecture-4-model-free-prediction-.pdf', 'Model-Free Prediction', 'Monte Carlo vs. TD learning, n-step returns and TD(λ) with eligibility traces.'),
    s5: silver(5, '0g4j2k_Ggc4', 'lecture-5-model-free-control-.pdf', 'Model-Free Control', 'On-policy (SARSA) and off-policy (Q-learning) control, GLIE, importance sampling.'),
    s6: silver(6, 'UoPei5o4fps', 'lecture-6-value-function-approximation-.pdf', 'Value Function Approximation', 'Linear and nonlinear value approximation, convergence issues, experience replay and DQN.'),
    s7: silver(7, 'KHZVXao4qXs', 'lecture-7-policy-gradient-methods.pdf', 'Policy Gradient Methods', 'Score-function gradients, the policy gradient theorem, actor-critic and compatible function approximation.'),
    s8: silver(8, 'ItMutbeOHtc', 'lecture-8-integrating-learning-and-planning.pdf', 'Integrating Learning and Planning', 'Model-based RL, Dyna, simulation-based search and MCTS.'),
    s9: silver(9, 'sGuiWX07sKw', 'lecture-9-exploration-and-exploitation.pdf', 'Exploration and Exploitation', 'Bandits, regret, UCB, Thompson sampling, information state search.'),
    s10: silver(10, 'kZ_AUmFcZtk', 'lecture-10-case-study-rl-in-classic-games.pdf', 'Case Study: RL in Classic Games', 'Game theory, minimax search, self-play and TD learning in games.'),
    silverAll: {
      kind: 'course', who: 'David Silver (DeepMind × UCL)', year: 2015, short: 'Silver: full course',
      title: 'Introduction to Reinforcement Learning (UCL, 2015) — YouTube playlist and slides',
      desc: 'The most widely watched RL lecture series; ten lectures closely following Sutton & Barto.',
      links: [{ label: 'Playlist', url: 'https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ' }, { label: 'Slides page', url: 'https://davidstarsilver.wordpress.com/teaching/' }],
    },

    /* books & courses */
    sb: {
      kind: 'book', who: 'Richard Sutton & Andrew Barto', year: 2018, short: 'Sutton & Barto (book)',
      title: 'Reinforcement Learning: An Introduction, 2nd edition — free PDF',
      desc: 'The standard textbook. Ch. 2 bandits, 3–4 MDPs & DP, 5–7 MC/TD/n-step, 8 planning, 9–11 function approximation, 13 policy gradients.',
      links: [{ label: 'Book page', url: 'http://incompleteideas.net/book/the-book-2nd.html' }, { label: 'PDF', url: 'http://incompleteideas.net/book/RLbook2020.pdf' }],
    },
    lat: {
      kind: 'book', who: 'Tor Lattimore & Csaba Szepesvári', year: 2020, short: 'Bandit Algorithms (book)',
      title: 'Bandit Algorithms — free PDF',
      desc: 'The comprehensive modern reference on bandits: regret lower bounds, UCB, Thompson sampling, contextual and adversarial bandits.',
      links: [{ label: 'Book (PDF)', url: 'https://tor-lattimore.com/downloads/book/book.pdf' }],
    },
    cs285: {
      kind: 'course', who: 'Sergey Levine (UC Berkeley)', year: 'ongoing', short: 'Berkeley CS 285',
      title: 'CS 285: Deep Reinforcement Learning (UC Berkeley)',
      desc: 'Graduate deep-RL course with recorded lectures: policy gradients, actor-critic, Q-learning, model-based, offline RL, inverse RL.',
      links: [{ label: 'Course page', url: 'https://rail.eecs.berkeley.edu/deeprlcourse/' }],
    },
    su_pg: {
      kind: 'course', who: 'OpenAI (Josh Achiam)', year: 2018, short: 'Spinning Up: policy optimization',
      title: 'Spinning Up in Deep RL — Part 3: Intro to Policy Optimization',
      desc: 'A compact, careful derivation of the policy gradient, reward-to-go and baselines, with code.',
      links: [{ label: 'Article', url: 'https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html' }],
    },
    su_ppo: {
      kind: 'course', who: 'OpenAI (Josh Achiam)', year: 2018, short: 'Spinning Up: PPO',
      title: 'Spinning Up in Deep RL — Proximal Policy Optimization',
      desc: 'PPO-clip explained with pseudocode and a reference implementation.',
      links: [{ label: 'Article', url: 'https://spinningup.openai.com/en/latest/algorithms/ppo.html' }],
    },
    su_sac: {
      kind: 'course', who: 'OpenAI (Josh Achiam)', year: 2018, short: 'Spinning Up: SAC',
      title: 'Spinning Up in Deep RL — Soft Actor-Critic', desc: 'SAC’s entropy-regularized objective, reparameterization trick and pseudocode.',
      links: [{ label: 'Article', url: 'https://spinningup.openai.com/en/latest/algorithms/sac.html' }],
    },
    su_td3: {
      kind: 'course', who: 'OpenAI (Josh Achiam)', year: 2018, short: 'Spinning Up: TD3',
      title: 'Spinning Up in Deep RL — Twin Delayed DDPG (TD3)', desc: 'The three tricks of TD3 on top of DDPG, with pseudocode.',
      links: [{ label: 'Article', url: 'https://spinningup.openai.com/en/latest/algorithms/td3.html' }],
    },
    ppo37: {
      kind: 'course', who: 'Shengyi Huang et al. (ICLR Blog Track)', year: 2022, short: '37 PPO implementation details',
      title: 'The 37 Implementation Details of Proximal Policy Optimization',
      desc: 'Every detail that makes PPO actually work, matched line by line to the original implementation, with videos.',
      links: [{ label: 'Blog post', url: 'https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/' }],
    },
    rlhf: {
      kind: 'book', who: 'Nathan Lambert', year: 'ongoing', short: 'RLHF book',
      title: 'Reinforcement Learning from Human Feedback (online book)',
      desc: 'The full post-training pipeline: preference data, reward models, PPO/REINFORCE variants, DPO, RLVR and evaluation.',
      links: [{ label: 'Book', url: 'https://rlhfbook.com' }],
    },

    sb_67: {
      kind: 'book', who: 'Richard Sutton & Andrew Barto', year: 2018, short: 'Sutton & Barto §6.7',
      title: 'Reinforcement Learning: An Introduction, §6.7 “Maximization Bias and Double Learning”',
      desc: 'Three pages with a tiny example that shows why taking the max over noisy estimates overestimates, and how Double Q-learning fixes it — the tabular idea behind Double DQN.',
      links: [{ label: 'Section (PDF, p. 134)', url: 'http://incompleteideas.net/book/RLbook2020.pdf#page=156' }],
    },
    su_trpo: {
      kind: 'course', who: 'OpenAI (Josh Achiam)', year: 2018, short: 'Spinning Up: TRPO',
      title: 'Spinning Up in Deep RL — Trust Region Policy Optimization',
      desc: 'The surrogate objective, the KL constraint and the conjugate-gradient step explained compactly, with pseudocode.',
      links: [{ label: 'Article', url: 'https://spinningup.openai.com/en/latest/algorithms/trpo.html' }],
    },
    worldmodels: {
      kind: 'course', who: 'David Ha & Jürgen Schmidhuber', year: 2018, short: 'World Models (interactive article)',
      title: 'World Models — interactive article',
      desc: 'A visual, interactive walk-through of an agent that learns a compressed model of its environment and trains its controller inside its own “dream”. The gentlest entry point to Dreamer-style agents.',
      links: [{ label: 'Article', url: 'https://worldmodels.github.io/' }],
    },
    hf_rlhf: {
      kind: 'course', who: 'Hugging Face (Lambert, Castricato, von Werra, Havrilla)', year: 2022, short: 'Illustrating RLHF',
      title: 'Illustrating Reinforcement Learning from Human Feedback (RLHF)',
      desc: 'A short, well-illustrated blog post on the three RLHF stages: pretrained model, reward model from human comparisons, and PPO fine-tuning with a KL penalty.',
      links: [{ label: 'Blog post', url: 'https://huggingface.co/blog/rlhf' }],
    },
    rlhf_dpo: {
      kind: 'book', who: 'Nathan Lambert', year: 'ongoing', short: 'RLHF book: direct alignment',
      title: 'RLHF book — chapter “Direct-Alignment Algorithms”',
      desc: 'DPO and its relatives derived step by step from the KL-regularized objective, with their practical trade-offs against RL-based training.',
      links: [{ label: 'Chapter', url: 'https://rlhfbook.com/c/08-direct-alignment' }],
    },
    rlhf_rl: {
      kind: 'book', who: 'Nathan Lambert', year: 'ongoing', short: 'RLHF book: RL for LLMs',
      title: 'RLHF book — chapter “Reinforcement Learning” (policy gradients for language models)',
      desc: 'REINFORCE, RLOO, PPO and GRPO side by side, with the loss functions written out and the implementation details that matter for language models.',
      links: [{ label: 'Chapter', url: 'https://rlhfbook.com/c/06-policy-gradients' }],
    },
    rlhf_reason: {
      kind: 'book', who: 'Nathan Lambert', year: 'ongoing', short: 'RLHF book: reasoning',
      title: 'RLHF book — chapter “Reasoning and Inference-Time Scaling”',
      desc: 'How RL with verifiable rewards produces reasoning models, and what is known about why it works.',
      links: [{ label: 'Chapter', url: 'https://rlhfbook.com/c/07-reasoning' }],
    },
    hf_r1: {
      kind: 'course', who: 'Hugging Face LLM Course', year: 2025, short: 'HF course: GRPO & R1',
      title: 'Hugging Face LLM Course, Chapter 12 “Open R1 for Students”',
      desc: 'A hands-on course unit: RL for LLMs from scratch, a guided reading of the DeepSeek-R1 paper, and GRPO implemented and run in TRL.',
      links: [{ label: 'Introduction', url: 'https://huggingface.co/learn/llm-course/chapter12/1' }, { label: 'Understanding the R1 paper', url: 'https://huggingface.co/learn/llm-course/chapter12/3' }, { label: 'GRPO in TRL', url: 'https://huggingface.co/learn/llm-course/chapter12/4' }],
    },
    cs285_offline: {
      kind: 'slides', who: 'Sergey Levine (UC Berkeley)', year: 'ongoing', short: 'CS 285: offline RL',
      title: 'CS 285 Deep Reinforcement Learning, Lectures 17–18: Offline RL',
      desc: 'Why offline RL is hard (distribution shift, overestimated values of unseen actions) and how conservative and in-sample methods such as CQL and IQL address it.',
      links: [{ label: 'Part 1 slides', url: 'https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-17.pdf' }, { label: 'Part 2 slides', url: 'https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-18.pdf' }],
    },

    /* papers */
    dqn: paper('DQN paper', 'Playing Atari with Deep Reinforcement Learning', 'Mnih et al.', 2013, '1312.5602', 'The original DQN paper (replay buffer, CNN on raw pixels); the Nature 2015 version added the target network.', [1, 5, 'introduction, background and the deep Q-learning algorithm with experience replay', 9]),
    ddqn: paper('Double DQN paper', 'Deep Reinforcement Learning with Double Q-learning', 'van Hasselt, Guez & Silver', 2015, '1509.06461', 'Shows DQN’s overestimation and fixes it by decoupling action selection and evaluation.', [1, 4, 'why Q-learning overestimates and the Double DQN update', 13]),
    gae: paper('GAE paper', 'High-Dimensional Continuous Control Using Generalized Advantage Estimation', 'Schulman et al.', 2015, '1506.02438', 'Introduces GAE(γ, λ), the advantage estimator used by almost every actor-critic method since.', [1, 5, 'advantage estimation and the role of γ and λ', 14]),
    trpo: paper('TRPO paper', 'Trust Region Policy Optimization', 'Schulman et al.', 2015, '1502.05477', 'Monotonic-improvement bound, KL trust region and the natural-gradient implementation.', [1, 5, 'the improvement bound, the KL trust region and the practical algorithm; proofs in the appendix', 16]),
    ppo: paper('PPO paper', 'Proximal Policy Optimization Algorithms', 'Schulman et al.', 2017, '1707.06347', 'The clipped surrogate objective and the KL-penalty variant.', [1, 5, 'background, the clipped surrogate objective and the algorithm', 12]),
    instructgpt: paper('InstructGPT paper', 'Training language models to follow instructions with human feedback', 'Ouyang et al.', 2022, '2203.02155', 'The SFT → reward model → PPO pipeline that made RLHF standard.', [6, 9, 'methods: data, SFT, reward model and PPO training (the introduction on pages 1–3 is a good overview)', 68]),
    dpo: paper('DPO paper', 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model', 'Rafailov et al.', 2023, '2305.18290', 'Derives the DPO loss from the closed-form KL-regularized optimum.', [3, 7, 'preliminaries, the DPO derivation and its theoretical analysis', 27]),
    grpo: paper('GRPO (DeepSeekMath) paper', 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models', 'Shao et al.', 2024, '2402.03300', 'Introduces GRPO: PPO with group-relative advantages instead of a critic.', [11, 14, 'section 4.1: from PPO to GRPO, outcome and process supervision', 30]),
    r1: paper('DeepSeek-R1 paper', 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning', 'DeepSeek-AI', 2025, '2501.12948', 'Large-scale RL with rule-based rewards producing long chain-of-thought reasoning.', [1, 6, 'introduction and DeepSeek-R1-Zero: GRPO, rule-based rewards and the “aha moment”', 86]),
    rloo: paper('RLOO paper', 'Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs', 'Ahmadian et al.', 2024, '2402.14740', 'Argues for REINFORCE with leave-one-out baselines (RLOO) instead of PPO in RLHF.', [4, 9, 'PPO, REINFORCE and RLOO compared, and why PPO’s machinery is rarely needed for LLMs', 28]),
    gao: paper('Reward over-optimization paper', 'Scaling Laws for Reward Model Overoptimization', 'Gao, Schulman & Hilton', 2022, '2210.10760', 'Measures how the true reward first rises then falls as the policy optimizes a proxy reward model.', [1, 10, 'setup, the scaling results and the Goodhart discussion', 28]),
    alphazero: paper('AlphaZero paper', 'Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm', 'Silver et al.', 2017, '1712.01815', 'MCTS guided by a policy/value network, trained purely by self-play.', [1, 4, 'the algorithm and the main results', 19]),
    muzero: paper('MuZero paper', 'Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model', 'Schrittwieser et al.', 2019, '1911.08265', 'Planning with a learned value-equivalent model instead of known rules.', [1, 4, 'introduction and the MuZero algorithm (search details: page 12)', 21]),
    dreamer: paper('DreamerV3 paper', 'Mastering Diverse Domains through World Models', 'Hafner et al.', 2023, '2301.04104', 'A world-model agent that works across 150+ tasks with fixed hyperparameters.', [2, 7, 'world model, critic and actor learning, and the robustness tricks', 40]),
    sac: paper('SAC paper', 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor', 'Haarnoja et al.', 2018, '1801.01290', 'Maximum-entropy off-policy actor-critic.', [3, 6, 'maximum-entropy RL, soft policy iteration and the SAC algorithm', 14]),
    td3: paper('TD3 paper', 'Addressing Function Approximation Error in Actor-Critic Methods', 'Fujimoto, van Hoof & Meger', 2018, '1802.09477', 'Clipped double-Q, target smoothing and delayed policy updates.', [3, 6, 'overestimation in actor-critic and the three fixes of TD3', 15]),
    cql: paper('CQL paper', 'Conservative Q-Learning for Offline Reinforcement Learning', 'Kumar et al.', 2020, '2006.04779', 'Penalizes Q-values of out-of-distribution actions.', [1, 7, 'the conservative Q-learning objective and the practical algorithm', 31]),
    iql: paper('IQL paper', 'Offline Reinforcement Learning with Implicit Q-Learning', 'Kostrikov, Nair & Levine', 2021, '2110.06169', 'In-sample learning with expectile regression; never queries unseen actions.', [3, 6, 'expectile regression and the IQL algorithm', 13]),
  };

  /* ---------- chapter mapping: section heading (prefix) → sources; `extra` only in the end-of-chapter list ---------- */
  const MAP = {
    intro: {
      sections: { 'The interaction loop': ['b25_1', 's1'], 'A map of the territory': ['bKeyIdeas', 'b25_2'] },
      extra: ['bBook', 'sb', 'silverAll', 'bHub'],
    },
    mdp: {
      sections: {
        'Markov decision processes': ['s2', 'b25_2'],
        'Value functions and the Bellman expectation': ['s2', 'ocw10'],
        'Optimality: the Bellman optimality': ['b25_2', 'ocw10'],
        'Two ways to solve it': ['b25_3', 's3', 'ocw16'],
        'Why dynamic programming is not the end': ['b25_2', 'ocw19'],
      },
      extra: ['bNewton', 'ocwAll', 'bBook', 'sb'],
    },
    bandits: {
      sections: { 'The simplest RL problem': ['s9', 'lat'], Strategies: ['s9', 'lat'], 'From bandits to full RL': ['b24_9'] },
      extra: ['sb'],
    },
    td: {
      sections: {
        'Learning without a model': ['s4', 'bTD'],
        'From prediction to control': ['s5', 'ocw22'],
        'A first look at model-based RL': ['s8'],
      },
      extra: ['ocw20', 'bNDP', 'sb'],
    },
    dqn: {
      sections: {
        'From Q-learning to a regression problem': ['s6', 'b24_12'],
        'The two stabilizers': ['dqn', 's6'],
        'Overestimation and Double DQN': ['sb_67', 'ddqn'],
      },
      extra: ['b24_11', 'ocw20', 'bNDP', 'cs285'],
    },
    pg: {
      sections: {
        'Why optimize the policy directly?': ['b25_12', 's7'],
        'The policy gradient theorem': ['s7', 'su_pg'],
        'Actor-critic and generalized advantage': ['gae', 's7'],
        'Value-based vs. policy-based': ['b25_2', 'b24_13'],
      },
      extra: ['ocw23', 'cs285', 'sb'],
    },
    ppo: {
      sections: {
        'The surrogate objective': ['su_trpo', 'trpo'],
        'PPO: clip the incentive': ['ppo', 'su_ppo'],
        'Why PPO became the default': ['ppo37'],
      },
      extra: ['b25_12', 'cs285'],
    },
    planning: {
      sections: {
        'Two ways to use a model': ['s8', 'b25_4'],
        'Monte Carlo Tree Search': ['b23_7', 'b25_5', 's10'],
        'AlphaZero: search as a policy-improvement': ['bAZ', 'alphazero', 'muzero'],
        'World models and imagination': ['worldmodels', 'dreamer'],
        'Search, test-time compute': ['bNewton', 'b25_11'],
      },
      extra: ['b25_6'],
    },
    llm: {
      sections: {
        'A language model is a policy': ['b25_8'],
        'The KL-regularized objective': ['rlhf', 'gao'],
        'RLHF with PPO': ['hf_rlhf', 'instructgpt'],
        'Direct Preference Optimization': ['rlhf_dpo', 'dpo'],
        'RL from verifiable rewards and GRPO': ['hf_r1', 'rlhf_rl', 'grpo', 'r1', 'rloo'],
      },
      extra: ['rlhf', 'rlhf_reason'],
    },
    landscape: {
      sections: {
        'Continuous control': ['su_sac', 'su_td3', 'sac', 'td3'],
        'Offline RL': ['cs285_offline', 'cql', 'iql'],
        'More areas, in brief': ['bMulti', 'b25_4'],
      },
      extra: ['cs285'],
    },
    cheatsheet: {
      sections: { 'Further reading': ['bBook', 'bHub', 'ocwAll', 'silverAll'] },
      extra: ['bNDP', 'bAZ', 'sb', 'cs285', 'rlhf'],
    },
  };

  const KIND_LABEL = { video: 'Video', slides: 'Slides', book: 'Book', course: 'Course', paper: 'Paper' };

  function chip(src) {
    const a = h('a.study-chip.k-' + src.kind, {
      href: src.links[0].url, target: '_blank', rel: 'noopener noreferrer',
      title: `${src.title} — ${src.who}, ${src.year}. ${src.desc}`,
    }, h('span.study-kind', KIND_LABEL[src.kind]), src.short);
    return a;
  }

  function card(src) {
    return h('li.study-card',
      h('div.study-card-head', h('span.study-kind.k-' + src.kind, KIND_LABEL[src.kind]), h('span.study-title', src.title)),
      h('div.study-meta', `${src.who} · ${src.year}`),
      h('div.study-desc', src.desc),
      src.read ? h('div.study-read', 'Read ', h('b', `pages ${src.read[0]}–${src.read[1]}`), `: ${src.read[2]} · ${src.read[3]} pages in total`) : null,
      h('div.study-links', src.links.map((l) => h('a', { href: l.url, target: '_blank', rel: 'noopener noreferrer' }, l.label + ' ↗'))));
  }

  /** Insert "Go deeper" chips after matching section headings and a full list at the end of the chapter. */
  function decorate(root, chId) {
    const m = MAP[chId];
    if (!m) return;
    const used = [];
    const heads = [...root.querySelectorAll('.concepts h2')];
    for (const [prefix, ids] of Object.entries(m.sections)) {
      const h2 = heads.find((x) => x.textContent.trim().startsWith(prefix));
      if (!h2) { console.warn('resources: no section', chId, prefix); continue; }
      const row = h('div.study-row', h('span.study-row-label', 'Go deeper:'));
      for (const id of ids) { if (!S[id]) continue; row.appendChild(chip(S[id])); if (!used.includes(id)) used.push(id); }
      h2.insertAdjacentElement('afterend', row);
    }
    for (const id of m.extra || []) if (S[id] && !used.includes(id)) used.push(id);
    if (!used.length) return;
    const order = (id) => (S[id].who.startsWith('Dimitri Bertsekas') ? 0 : S[id].kind === 'video' || S[id].kind === 'course' ? 1 : S[id].kind === 'book' ? 2 : 3);
    used.sort((a, b) => order(a) - order(b));
    const list = h('ul.study-list');
    used.forEach((id) => list.appendChild(card(S[id])));
    const sec = h('div.concepts.study-section',
      h('h2', 'Go deeper: lectures, slides & reading'),
      h('p.small.muted', 'Hand-picked lectures, slides and reading for this chapter, Dimitri Bertsekas’ course first. His course page (Cheat Sheet chapter) lists every year of his lectures.'),
      list);
    root.appendChild(sec);
  }

  RL.resources = { sources: S, map: MAP, decorate };
})(window.RL);
