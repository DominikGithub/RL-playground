/* Chapter 2: exploration vs exploitation with multi-armed bandits. */
(function (RL) {
  'use strict';
  const sw = RL.ui.sw, lg = RL.ui.lg;
  const { h, fmt, setupCanvas, cssVar } = RL.util;
  const S = RL.bandit.STRATEGIES;

  const theory1 = String.raw`
<h2>The simplest RL problem that still needs exploration</h2>
<p>A $k$-armed bandit is an MDP with a single state: each action $a$ pays a random reward with unknown mean $\mu_a$. There is no credit-assignment problem, no dynamics — only the dilemma of <b>exploration vs. exploitation</b>: pulling the arm that currently looks best (exploit) versus gathering information about arms that might be better (explore). Performance is measured by the <b>regret</b> — how much reward was lost relative to always pulling the best arm $a^*$:</p>
$$\mathcal R_T = T\mu^* - \mathbb E\Big[\sum_{t=1}^T r_t\Big] = \sum_{a} \Delta_a\, \mathbb E[N_a(T)], \qquad \Delta_a = \mu^* - \mu_a,$$
<p>where $N_a(T)$ counts the pulls of arm $a$. Good algorithms pull bad arms rarely, and the worse an arm, the more rarely.</p>
`;

  const theory2 = String.raw`
<h2>Strategies</h2>
<h3>Greedy and ε-greedy</h3>
<p>Estimate each mean by the sample average $Q_a = \frac1{N_a}\sum r$ and pick $\arg\max_a Q_a$. Pure greedy locks onto whichever arm got lucky early — linear regret. ε-greedy picks a uniformly random arm with probability $\varepsilon$: simple and robust (still the exploration scheme of DQN), but with constant $\varepsilon$ it keeps paying $\approx \varepsilon\,\bar\Delta$ per step forever, i.e. regret still grows linearly. Decaying $\varepsilon_t \propto 1/t$ fixes the rate in theory but needs tuning. <em>Optimistic initialization</em> ($Q_0 = +5$) makes greedy explore early because every untried arm looks great until disproven.</p>

<h3>Upper confidence bounds: optimism in the face of uncertainty</h3>
<p>Instead of exploring at random, explore where you are <em>uncertain</em>. UCB1 (Auer et al., 2002) picks</p>
$$a_t = \arg\max_a \Big[\, Q_a + c\sqrt{\frac{\ln t}{N_a}} \,\Big].$$
<details class="deep math"><summary><span class="lbl">Math</span> Where the bonus comes from, and why log T is optimal</summary><div class="body">
<p>For rewards in $[0,1]$, Hoeffding's inequality gives $\Pr\big(\mu_a > Q_a + u\big) \le e^{-2N_a u^2}$. Asking this failure probability to be $t^{-4}$ (shrinking over time) yields $u = \sqrt{2\ln t / N_a}$ — the UCB1 bonus with $c=\sqrt2$. With high probability the index is an upper bound on the true mean, so a suboptimal arm is only chosen while its bound still overlaps $\mu^*$, i.e. while $N_a \lesssim 8\ln t/\Delta_a^2$. Hence</p>
$$\mathcal R_T = O\Big(\sum_{a:\Delta_a>0} \frac{\ln T}{\Delta_a}\Big).$$
<p>Lai & Robbins (1985) proved that <em>no</em> consistent algorithm can do better asymptotically: $\liminf_{T\to\infty} \mathbb E[N_a(T)]/\ln T \ge 1/\mathrm{KL}(\nu_a\,\|\,\nu^*)$. Logarithmic regret is the best possible; the constants are what separate algorithms (KL-UCB and Thompson sampling attain the optimal constant).</p>
</div></details>

<h3>Thompson sampling: probability matching</h3>
<p>Keep a Bayesian posterior over each arm's mean. Each step, <b>draw one sample from every posterior and act greedily on the samples</b>. An arm is chosen with exactly the posterior probability that it is the best one. With Gaussian rewards (noise variance $\sigma^2=1$) and prior $\mathcal N(0,1)$ the posterior after $n$ pulls with sum $S$ is conjugate:</p>
$$\mu_a \mid \text{data} \sim \mathcal N\!\left(\frac{S_a}{1 + n_a},\; \frac{1}{1+n_a}\right).$$
<p>Thompson sampling (1933!) was ignored for decades and rediscovered around 2010; it is asymptotically optimal, typically the best in practice, handles delayed feedback and batching gracefully, and extends naturally to contextual and structured problems. It is the workhorse of industrial bandits (ads, recommendation, adaptive experiments).</p>

<h3>Gradient bandits: the first policy gradient</h3>
<p>Instead of estimating values, keep <em>preferences</em> $H_a$ and a softmax policy $\pi(a) = e^{H_a}/\sum_b e^{H_b}$. After pulling $A$ and receiving $R$:</p>
$$H_b \leftarrow H_b + \alpha\,(R - \bar R)\,\big(\mathbb 1[b=A] - \pi(b)\big).$$
<details class="deep math"><summary><span class="lbl">Math</span> This is exact stochastic gradient ascent — and the baseline is free</summary><div class="body">
<p>The objective is $J(H) = \mathbb E_{A\sim\pi}[\mu_A] = \sum_a \pi(a)\mu_a$. Using $\partial\pi(a)/\partial H_b = \pi(a)(\mathbb 1[a=b] - \pi(b))$:</p>
$$\frac{\partial J}{\partial H_b} = \sum_a \mu_a\, \pi(a)\big(\mathbb 1[a=b]-\pi(b)\big) = \mathbb E_{A\sim\pi}\Big[\mu_A\big(\mathbb 1[A=b] - \pi(b)\big)\Big].$$
<p>Replacing $\mu_A$ by the sampled reward $R$ gives an unbiased estimate. Subtracting any baseline $B$ that does not depend on $A$ changes nothing in expectation, because $\sum_a \partial\pi(a)/\partial H_b = \partial(1)/\partial H_b = 0$. But it changes the variance enormously: without a baseline, a positive reward increases the chosen arm's preference even when that arm is below average. Switch the baseline off in the comparison lab and watch the gradient bandit degrade.</p>
<p>This is <b>REINFORCE with a baseline</b> for a one-step problem. Chapter 5 generalizes it to sequential decisions; Chapter 8 shows that RL fine-tuning of language models with a single scalar reward per answer is essentially this, at the scale of billions of parameters (GRPO's "group mean" is literally the baseline $\bar R$).</p>
</div></details>

<h2>From bandits to full RL</h2>
<ul>
<li><b>Contextual bandits</b> observe a context $x$ (user, query) before choosing — a one-step MDP with many states. LinUCB, linear/neural Thompson sampling. This is what most production "RL" in recommendation and advertising is.</li>
<li><b>Exploration in deep RL</b> reuses the same ideas: ε-greedy (DQN), entropy bonuses (policy gradients, SAC), and optimism via <em>exploration bonuses</em> — count-based, curiosity/prediction-error (ICM), random network distillation (RND) — which act like a UCB bonus in large state spaces. Posterior-sampling ideas appear as bootstrapped DQN and ensembles.</li>
<li><b>LLMs</b>: sampling several answers and keeping the best (best-of-$n$), temperature, and the group sampling of GRPO are all exploration decisions.</li>
</ul>
<details class="deep practice"><summary><span class="lbl">Practice</span> Non-stationary rewards</summary><div class="body">
<p>Sample averages weight all history equally. If arm means drift, use a constant step size, $Q \leftarrow Q + \alpha(R-Q)$, which is an exponential recency-weighted average with effective memory $\approx 1/\alpha$. Thompson sampling handles drift by discounting the posterior counts. Optimistic initialization only helps at the beginning and is useless against drift.</p>
</div></details>
<details class="deep history"><summary><span class="lbl">History</span> Gittins indices and other classics</summary><div class="body">
<p>For discounted Bayesian bandits the <b>Gittins index</b> (1979) gives the exact Bayes-optimal policy — a beautiful result, but expensive to compute and brittle outside its assumptions, so it is rarely used. <b>Boltzmann/softmax exploration</b> on value estimates (temperature-controlled) and <b>pursuit</b> methods appear in older literature; they lack the regret guarantees of UCB/Thompson. Today's practical toolkit is essentially ε-greedy (for simplicity), UCB variants (for guarantees) and Thompson sampling (for performance).</p>
</div></details>
`;

  RL.chapters.push({
    id: 'bandits', num: 2, part: 'Foundations', title: 'Exploration: Bandits',
    mount(root) {
      RL.page.head(root, {
        num: 2, title: 'Exploration vs. exploitation: multi-armed bandits',
        lede: 'One state, k actions, noisy rewards. The cleanest setting to understand why an agent must sometimes take actions it believes are worse — and how to do it efficiently.',
        tags: [['single state'], ['regret'], ['UCB · Thompson'], ['bridge to policy gradients', 'now']],
      });
      RL.page.prose(root, theory1);
      const cleanups = [];

      /* ---------- A. play it yourself ---------- */
      {
        const w = RL.page.widget(root, 'You are the agent', '10 slot machines with unknown average payouts (rewards are noisy, σ = 1). You have <b>40 pulls</b>. Maximize your total reward — then see how you compare with UCB and Thompson sampling on the very same machines.');
        RL.ui.info(w.querySelector('.widget-title'), lg([
          [sw.text('#3'), 'One card per slot machine (arm). Click a card to pull it once.'],
          [sw.text('0.84', 'var(--accent)'), 'Big number: <b>your average reward</b> from this arm so far (“?” = never pulled).'],
          [sw.text('4×'), 'How often you pulled it, and the most recent reward (“last”). Rewards are noisy: one pull says little.'],
          [sw.text('μ', 'var(--neg)'), 'After 40 pulls: the arm’s <b>true mean</b> μ is revealed.'],
          [sw.ring('var(--good)'), 'Green ring: the truly best arm.'],
        ]), 'Slot machines');
        const K = 10, BUDGET = 40;
        let rng, bandit, N, sums, total, pulls;
        const armsEl = h('div.arms');
        const statusEl = h('div.small', { style: { margin: '10px 0' } });
        const btnRow = h('div.btn-row');
        w.appendChild(armsEl); w.appendChild(statusEl); w.appendChild(btnRow);
        const resultEl = h('div.small.muted');
        w.appendChild(resultEl);
        const armEls = [];
        const reset = () => {
          rng = new RL.Rng(); bandit = new RL.Bandit(K, rng);
          N = new Array(K).fill(0); sums = new Array(K).fill(0); total = 0; pulls = 0;
          resultEl.innerHTML = '';
          draw();
        };
        for (let a = 0; a < K; a++) {
          const el = h('button.arm', { type: 'button', 'aria-label': 'Pull arm ' + (a + 1) });
          el.addEventListener('click', () => {
            if (pulls >= BUDGET) return;
            const r = bandit.pull(a); N[a]++; sums[a] += r; total += r; pulls++;
            el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 180);
            el.dataset.last = fmt(r, 2);
            draw();
            if (pulls === BUDGET) finish();
          });
          armEls.push(el); armsEl.appendChild(el);
        }
        function draw() {
          const done = pulls >= BUDGET;
          armEls.forEach((el, a) => {
            el.innerHTML = '';
            el.appendChild(h('span.arm-n', '#' + (a + 1)));
            el.appendChild(h('span.arm-est', N[a] ? fmt(sums[a] / N[a], 2) : '?'));
            el.appendChild(h('span.arm-cnt', N[a] + '×' + (el.dataset.last && N[a] ? ' · last ' + el.dataset.last : '')));
            if (done) el.appendChild(h('span.arm-true', 'μ=' + fmt(bandit.means[a], 2)));
            el.classList.toggle('best', done && a === bandit.best);
          });
          statusEl.innerHTML = `Pulls left: <b>${BUDGET - pulls}</b> · total reward: <b>${fmt(total, 2)}</b>`;
        }
        function finish() {
          // Simulate the strategies on *these* machines for a fair comparison (expected reward of choices).
          const sim = (type) => {
            let tot = 0;
            for (let r = 0; r < 300; r++) {
              const ag = RL.bandit.makeAgent(type, K, { c: 2, epsilon: 0.1 }, new RL.Rng(r + 17));
              for (let t = 0; t < BUDGET; t++) { const a = ag.select(); const rr = bandit.means[a] + rng.normal(); ag.update(a, rr); tot += bandit.means[a]; }
            }
            return tot / 300;
          };
          const yourExp = N.reduce((s, n, a) => s + n * bandit.means[a], 0);
          const best = BUDGET * bandit.bestMean;
          resultEl.innerHTML = `Best arm was <b>#${bandit.best + 1}</b> (μ = ${fmt(bandit.bestMean, 2)}). ` +
            `Your expected reward from your choices: <b>${fmt(yourExp, 1)}</b> (regret ${fmt(best - yourExp, 1)}). ` +
            `Average over 300 runs on these machines — UCB: <b>${fmt(sim('ucb'), 1)}</b>, Thompson: <b>${fmt(sim('thompson'), 1)}</b>, ε-greedy (0.1): <b>${fmt(sim('egreedy'), 1)}</b>; always-best: ${fmt(best, 1)}. ` +
            `Did you explore enough? Most people commit too early.`;
        }
        RL.ui.button(btnRow, 'New machines', reset);
        reset();
      }

      /* ---------- B. inside the algorithm ---------- */
      {
        RL.page.prose(root, String.raw`<h2>Inside the algorithms</h2><p>Step through a single run and see <em>why</em> each strategy picks the arm it picks. Orange ticks are the true (hidden) means; dots are the agent's estimates; bars show the uncertainty the strategy uses.</p>`);
        const lab = RL.page.lab(root, { title: 'Bandit internals' });
        const C = lab.controls;
        C.appendChild(h('div.lab-title', 'Single run'));
        let strat = 'ucb', k = 10, c = 2, eps = 0.1, speed = 8;
        let rng, bandit, agent, t, regret, lastA, lastR = 0, lastInfo, acc = 0, lastT = 0;
        const tr = RL.ui.transport(C, {
          onPlay: (p) => { if (p) { lastT = performance.now(); loop.start(); } else loop.stop(); },
          onStep: () => { tr.setPlaying(false); const pre = preState(); step(); draw(); explainPull(pre); },
          onReset: () => { tr.setPlaying(false); reset(); },
        });
        RL.ui.select(C, { label: 'Strategy', value: strat, options: ['egreedy', 'ucb', 'thompson'].map((v) => ({ value: v, label: S[v].label })), onChange: (v) => { strat = v; reset(); } });
        const cS = RL.ui.slider(C, { label: 'UCB c', min: 0, max: 4, step: 0.1, value: c, help: 'Exploration weight of the confidence bonus c·√(ln t / N).', onInput: (v) => { c = v; reset(); } });
        const eS = RL.ui.slider(C, { label: 'ε', min: 0, max: 0.5, step: 0.01, value: eps, onInput: (v) => { eps = v; reset(); } });
        RL.ui.slider(C, { label: 'Steps per second', min: 1, max: 120, step: 1, value: speed, onInput: (v) => { speed = v; } });
        const tutor = RL.ui.coach();
        RL.ui.tutorToggle(C, tutor);
        let pulls = [];
        RL.ui.info(lab.viz.appendChild(h('div.lab-title', h('span', 'Arms: true means, estimates and uncertainty'))), lg([
          [sw.tick('var(--neg)'), '<b>Orange tick</b>: the arm’s true mean reward μ<sub>a</sub>. Hidden from the agent, shown for you.'],
          [sw.dot('var(--text)'), '<b>Dark dot</b>: the agent’s estimate Q<sub>a</sub> (average observed reward; for Thompson sampling the posterior mean).'],
          [sw.band('var(--pos)'), '<b>Blue bar (UCB1)</b>: runs from Q<sub>a</sub> up to the optimistic index Q<sub>a</sub> + c·√(ln t / N<sub>a</sub>), marked by the solid line on top. The arm with the <b>highest top</b> is pulled. Untried arms get a full-height bar (infinite bonus); bars shrink as an arm is pulled more often.'],
          [sw.band('var(--pos)'), '<b>Blue band (Thompson sampling)</b>: posterior mean ± 2 standard deviations (≈ 95% credible interval). Each step one value is sampled per arm and the largest sample wins, so wide bands still get tried.'],
          [sw.text('ε'), '<b>ε-greedy</b> has no bars: it pulls the highest dot, except with probability ε a random arm.'],
          [sw.box('var(--accent-soft)'), 'Shaded column: the arm pulled in the last step.'],
          [sw.text('7×'), 'Under each column: arm number and pull count N<sub>a</sub>. Vertical axis: reward.'],
        ], 'One column per arm.'), 'Bandit internals');
        const canvas = h('canvas'); const holder = h('div', canvas); lab.viz.appendChild(holder);
        const statusEl = h('div.lab-status'); lab.viz.appendChild(statusEl);
        const stats = RL.ui.stats(lab.side, [['t', 'Step t'], ['regret', 'Regret so far'], ['best', 'Pulls of best arm'], ['est', 'Estimated best']]);
        const chart = new RL.LineChart(lab.side, { title: 'Cumulative regret (this run)', info: 'Expected reward lost so far in this single run: Σₜ (μ* − μ of the pulled arm). It flattens once the agent mostly pulls the best arm.', height: 170, xLabel: 'step', legend: false });
        chart.addSeries('regret', '--accent');
        function reset() {
          cS.show(strat === 'ucb'); eS.show(strat === 'egreedy');
          rng = new RL.Rng(42); bandit = new RL.Bandit(k, new RL.Rng(7));
          agent = RL.bandit.makeAgent(strat, k, { c, epsilon: eps }, rng);
          t = 0; regret = 0; lastA = -1; lastInfo = ''; pulls = []; if (tutor) tutor.hide(); chart.clear(); draw();
        }
        /* ---------- Tutor: why this arm, and is the agent improving? ---------- */
        const preState = () => ({ Q: Array.from(agent.Q), N: Array.from(agent.N), t });
        const argmaxArr = (arr, mask) => { let b = -1; arr.forEach((v, i) => { if ((!mask || mask[i]) && (b < 0 || v > arr[b])) b = i; }); return b; };
        function explainPull(pre) {
          const a = lastA, best = bandit.best, n = pre.N[a];
          const tried = pre.N.map((x) => x > 0);
          const g = argmaxArr(pre.Q, strat === 'thompson' ? null : tried);
          const arm = (i) => '#' + (i + 1);
          let why;
          if (strat === 'ucb') {
            if (n === 0) why = `UCB pulled ${arm(a)} because it had <b>never been tried</b>: its uncertainty bonus is infinite, so UCB samples every arm once before comparing them.`;
            else {
              const bonus = c * Math.sqrt(Math.log(pre.t + 1) / n), idx = pre.Q[a] + bonus;
              why = g === a
                ? `UCB pulled ${arm(a)}: highest upper bound (Q ${fmt(pre.Q[a], 2)} + bonus ${fmt(bonus, 2)} = ${fmt(idx, 2)}) and also the highest estimate — this pull is <b>exploitation</b>.`
                : `UCB pulled ${arm(a)} although ${arm(g)} has the higher estimate (${fmt(pre.Q[g], 2)} vs ${fmt(pre.Q[a], 2)}). ${arm(a)} has only ${n} pull${n === 1 ? '' : 's'}, so its bonus ${fmt(bonus, 2)} lifts its upper bound to ${fmt(idx, 2)}, the highest — <b>optimism in the face of uncertainty</b>.`;
            }
          } else if (strat === 'thompson') {
            why = g === a
              ? `Thompson sampling pulled ${arm(a)}: its random posterior sample was the largest, and it also has the highest posterior mean (${fmt(pre.Q[a], 2)}) — mostly <b>exploitation</b>.`
              : `Thompson sampling pulled ${arm(a)} although ${arm(g)} has the higher posterior mean (${fmt(pre.Q[g], 2)} vs ${fmt(pre.Q[a], 2)}). With only ${n} pull${n === 1 ? '' : 's'} its posterior is wide, so its sample came out on top — <b>exploration</b> in proportion to the chance that it is the best arm.`;
          } else {
            const greedy = n > 0 && pre.Q[a] >= Math.max(...pre.Q.filter((_, i) => tried[i])) - 1e-12;
            why = greedy || pre.t === 0
              ? `ε-greedy pulled ${arm(a)}, the arm with the highest estimate — an <b>exploit</b> step (probability ≈ 1 − ε).`
              : `ε-greedy pulled ${arm(a)} although ${arm(g)} has a higher estimate: a <b>random exploration</b> step, which happens with probability ε = ${fmt(eps, 2)}.`;
          }
          const r = lastR;
          let text = why + ` It paid ${fmt(r, 2)}; its estimate is now ${fmt(agent.Q[a], 2)}.`;
          text += a === best ? ' No regret this step: that is the truly best arm (hidden from the agent).'
            : ` Regret +${fmt(bandit.bestMean - bandit.means[a], 2)}: the best arm ${arm(best)} pays that much more on average.`;
          let tone = 'info', basis = 'the estimates and pull counts just before this pull';
          if (pulls.length >= 20) {
            const last = pulls.slice(-10).filter((x) => x === best).length, prev = pulls.slice(-20, -10).filter((x) => x === best).length;
            tone = last >= prev + 2 ? 'good' : last <= prev - 2 ? 'bad' : 'unclear';
            text += ` <br><b>Trend:</b> ${last}/10 of the last pulls went to the best arm, vs ${prev}/10 in the 10 before — ` +
              (tone === 'good' ? 'the agent is concentrating on the best arm.' : tone === 'bad' ? 'it is exploring more again (normal now and then, e.g. when a good-looking arm disappoints).' : 'no clear change yet; ten pulls are a small sample.');
            basis += '; trend = share of pulls on the true best arm, last 10 vs previous 10';
          } else {
            text += ` <br><b>Trend:</b> too early to judge (${pulls.length} pulls); I compare windows of 10 pulls from pull 20 on.`;
          }
          tutor.say({ tone, title: `Step ${t}: pulled ${arm(a)}`, text, basis });
        }

        function step() {
          const a = agent.select();
          const r = bandit.pull(a); agent.update(a, r);
          t++; regret += bandit.bestMean - bandit.means[a]; lastA = a; lastR = r; pulls.push(a);
          lastInfo = `t=${t}: pulled #${a + 1}, reward ${fmt(r, 2)}`;
          chart.push('regret', regret);
        }
        function draw() {
          const W = Math.max(300, holder.clientWidth || 600), H = 300;
          const ctx = setupCanvas(canvas, W, H);
          const css = (n) => cssVar(n);
          const padL = 34, padB = 34, padT = 10;
          const lo = -3.5, hi = 3.5;
          const Y = (v) => padT + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - padT - padB);
          const colW = (W - padL - 10) / k;
          ctx.clearRect(0, 0, W, H);
          ctx.font = '11px system-ui'; ctx.fillStyle = css('--muted'); ctx.strokeStyle = css('--grid');
          ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
          for (let v = -3; v <= 3; v++) { ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(W - 10, Y(v)); ctx.stroke(); ctx.fillText(String(v).replace('-', '−'), padL - 6, Y(v)); }
          const Q = agent.Q, N = agent.N;
          // compute index / sample for display
          for (let a = 0; a < k; a++) {
            const cx = padL + colW * (a + 0.5);
            if (a === lastA) { ctx.fillStyle = css('--accent-soft'); ctx.fillRect(cx - colW / 2 + 2, padT, colW - 4, H - padT - padB); }
            ctx.strokeStyle = css('--neg'); ctx.lineWidth = 3;
            ctx.beginPath(); ctx.moveTo(cx - colW * 0.3, Y(bandit.means[a])); ctx.lineTo(cx + colW * 0.3, Y(bandit.means[a])); ctx.stroke();
            let est = Q[a], top = null, bot = null;
            if (strat === 'ucb' && t > 0) {
              if (N[a] > 0) { top = Q[a] + c * Math.sqrt(Math.log(t) / N[a]); bot = Q[a]; } else { top = hi; bot = lo; est = NaN; }
            } else if (strat === 'thompson') {
              const sd = 1 / Math.sqrt(1 + N[a]); top = Q[a] + 2 * sd; bot = Q[a] - 2 * sd;
            }
            if (top !== null) {
              ctx.fillStyle = css('--pos'); ctx.globalAlpha = 0.28;
              ctx.fillRect(cx - colW * 0.16, Y(top), colW * 0.32, Y(bot) - Y(top)); ctx.globalAlpha = 1;
              if (strat === 'ucb') { ctx.strokeStyle = css('--pos'); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx - colW * 0.22, Y(top)); ctx.lineTo(cx + colW * 0.22, Y(top)); ctx.stroke(); }
            }
            if (Number.isFinite(est) && (N[a] > 0 || strat === 'thompson')) {
              ctx.fillStyle = css('--text'); ctx.beginPath(); ctx.arc(cx, Y(est), 4.5, 0, Math.PI * 2); ctx.fill();
            }
            ctx.fillStyle = css('--muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText('#' + (a + 1), cx, H - padB + 4);
            ctx.fillText(N[a] + '×', cx, H - padB + 17);
          }
          const legend = strat === 'ucb' ? 'bar top = UCB index Q + c√(ln t/N); the highest top is pulled'
            : strat === 'thompson' ? 'band = posterior mean ± 2σ; each step one sample per arm is drawn and the largest wins'
              : 'ε-greedy: highest dot is pulled, except with probability ε a random arm';
          statusEl.textContent = (lastInfo ? lastInfo + '  ·  ' : '') + legend;
          stats.t(t); stats.regret(regret);
          stats.best(N[bandit.best] + ' / ' + t);
          let bi = 0; for (let a = 1; a < k; a++) if (Q[a] > Q[bi]) bi = a;
          stats.est(t === 0 ? '–' : '#' + (bi + 1) + (bi === bandit.best ? ' ✓' : ' ✗'));
        }
        const loop = new RL.Loop(() => {
          const now = performance.now(); acc += ((now - lastT) / 1000) * speed; lastT = now;
          let n = Math.min(50, Math.floor(acc)); acc -= n;
          if (n > 0) { while (n--) step(); draw(); }
        });
        RL.onTheme(draw);
        if (window.ResizeObserver) { const ro = new ResizeObserver(() => draw()); ro.observe(holder); cleanups.push(() => ro.disconnect()); }
        reset();
        cleanups.push(() => loop.stop(), () => chart.destroy(), () => tutor.hide());
      }

      /* ---------- C. comparison ---------- */
      let runCompare;
      {
        RL.page.prose(root, String.raw`<h2>Comparing strategies</h2><p>A single run is noise. The standard testbed (Sutton & Barto §2.3) averages many independent bandit problems with means $\mu_a \sim \mathcal N(0,1)$ and rewards $r\sim\mathcal N(\mu_a, \sigma^2)$.</p>`);
        const lab = RL.page.lab(root, { title: 'Bandit testbed' });
        const C = lab.controls;
        C.appendChild(h('div.lab-title', 'Testbed'));
        const p = { k: 10, T: 1000, runs: 200, epsilon: 0.1, c: 2, alpha: 0.1, baseline: true, noise: 1 };
        const enabled = { greedy: true, optimistic: false, egreedy: true, ucb: true, thompson: true, gradient: false };
        const btn = RL.ui.button(C, '▶ Run experiment', () => run(), 'primary');
        btn.style.marginBottom = '12px';
        const g1 = RL.ui.group(C, 'Problem');
        RL.ui.slider(g1, { label: 'Arms k', min: 2, max: 30, step: 1, value: p.k, onInput: (v) => { p.k = v; } });
        RL.ui.slider(g1, { label: 'Steps T', min: 100, max: 5000, step: 100, value: p.T, onInput: (v) => { p.T = v; } });
        RL.ui.slider(g1, { label: 'Runs averaged', min: 20, max: 1000, step: 10, value: p.runs, onInput: (v) => { p.runs = v; } });
        RL.ui.slider(g1, { label: 'Reward noise σ', min: 0.1, max: 4, step: 0.1, value: p.noise, help: 'Std. dev. of reward around each arm mean. More noise = harder to tell arms apart = more exploration needed.', onInput: (v) => { p.noise = v; } });
        const g2 = RL.ui.group(C, 'Strategies');
        const tg = {};
        for (const key of Object.keys(S)) tg[key] = RL.ui.toggle(g2, { label: S[key].label, value: enabled[key], onChange: (v) => { enabled[key] = v; } });
        const g3 = RL.ui.group(C, 'Hyperparameters');
        const epsS = RL.ui.slider(g3, { label: 'ε (ε-greedy)', min: 0, max: 0.5, step: 0.01, value: p.epsilon, onInput: (v) => { p.epsilon = v; } });
        const cS = RL.ui.slider(g3, { label: 'c (UCB)', min: 0, max: 4, step: 0.1, value: p.c, onInput: (v) => { p.c = v; } });
        const aS = RL.ui.slider(g3, { label: 'α (gradient bandit)', min: 0.01, max: 1, step: 0.01, value: p.alpha, onInput: (v) => { p.alpha = v; } });
        const bT = RL.ui.toggle(g3, { label: 'Gradient bandit uses baseline R̄', value: true, onChange: (v) => { p.baseline = v; } });
        const status = h('div.lab-status', 'Press “Run experiment”.');
        lab.viz.appendChild(status);
        const chReg = new RL.LineChart(lab.viz, { title: 'Cumulative regret (lower is better)', info: 'Average over all runs of Σₜ (μ* − μ of the pulled arm). A straight line = a constant rate of mistakes (e.g. ε-greedy never stops exploring); a curve that bends over = the strategy learns to avoid bad arms (logarithmic regret is optimal). Colours = strategies, see the legend above the chart.', height: 230, xLabel: 'step' });
        const chOpt = new RL.LineChart(lab.side, { title: '% optimal action', info: 'Share of runs that pulled the truly best arm at each step (lightly smoothed). Colours = strategies.', height: 170, yMin: 0, yMax: 100, xLabel: 'step' });
        const chRew = new RL.LineChart(lab.side, { title: 'Average reward', info: 'Mean reward received at each step across runs (lightly smoothed). The best possible value is the average of the best arm’s mean, ≈ 1.5 for 10 arms.', height: 170, xLabel: 'step' });
        function run() {
          const types = Object.keys(enabled).filter((k) => enabled[k]);
          if (!types.length) { status.textContent = 'Enable at least one strategy.'; return; }
          status.textContent = 'Running ' + types.length + ' strategies × ' + p.runs + ' runs × ' + p.T + ' steps…';
          btn.disabled = true;
          setTimeout(() => {
            const t0 = performance.now();
            const res = RL.bandit.runExperiment(types, { k: p.k, T: p.T, runs: p.runs, noise: p.noise, seed: 3, params: p });
            for (const ch of [chReg, chOpt, chRew]) ch.removeAll();
            for (const ty of types) {
              chReg.addSeries(S[ty].label, S[ty].color).data = Array.from(res[ty].regret);
              chOpt.addSeries(S[ty].label, S[ty].color, { smooth: 0.9, raw: false }).data = Array.from(res[ty].optimal);
              chRew.addSeries(S[ty].label, S[ty].color, { smooth: 0.9, raw: false }).data = Array.from(res[ty].reward);
            }
            [chReg, chOpt, chRew].forEach((c) => c.requestDraw());
            const fin = types.map((ty) => `${S[ty].label}: ${fmt(res[ty].regret[p.T - 1], 0)}`).join(' · ');
            status.textContent = `Final regret — ${fin}  (${Math.round(performance.now() - t0)} ms)`;
            btn.disabled = false;
          }, 30);
        }
        runCompare = (o) => {
          for (const key of Object.keys(S)) { const v = o.enable.includes(key); enabled[key] = v; tg[key].set(v, true); }
          if (o.epsilon !== undefined) { p.epsilon = o.epsilon; epsS.set(o.epsilon, true); }
          if (o.c !== undefined) { p.c = o.c; cS.set(o.c, true); }
          if (o.alpha !== undefined) { p.alpha = o.alpha; aS.set(o.alpha, true); }
          if (o.baseline !== undefined) { p.baseline = o.baseline; bT.set(o.baseline, true); }
          run();
        };
        cleanups.push(() => [chReg, chOpt, chRew].forEach((c) => c.destroy()));
        RL.ui.experiments(root, [
          { title: 'Greedy gets stuck', text: 'Greedy vs. ε-greedy. Greedy finds the best arm in only about 40% of problems and then never looks again; its regret grows linearly.', apply: () => runCompare({ enable: ['greedy', 'egreedy'], epsilon: 0.1 }), labEl: lab.el },
          { title: 'Directed beats random exploration', text: 'ε-greedy vs. UCB vs. Thompson. Notice the curvature: ε-greedy stays linear, the others bend over (logarithmic regret).', apply: () => runCompare({ enable: ['egreedy', 'ucb', 'thompson'], epsilon: 0.1, c: 2 }), labEl: lab.el },
          { title: 'Remove the baseline', text: 'Gradient bandit with and without the baseline $\\bar R$ (run twice: toggle it off and press Run). Same expected gradient, very different variance — the key lesson for Chapter 5.', apply: () => runCompare({ enable: ['gradient', 'egreedy'], alpha: 0.1, baseline: false }), labEl: lab.el },
          { title: 'Optimism as exploration', text: 'Optimistic greedy ($Q_0=+5$) explores systematically at the start, giving a characteristic early spike in % optimal action.', apply: () => runCompare({ enable: ['optimistic', 'egreedy', 'greedy'] }), labEl: lab.el },
        ]);
      }
      RL.page.prose(root, theory2);
      return () => cleanups.forEach((f) => f());
    },
  });
})(window.RL);
