// Regenerates the README images in docs/screenshots/.
//   python -m http.server 8765   (separate terminal), then: node tests/readme-screenshots.js
const { spawn } = require('child_process');
const fs = require('fs'), path = require('path'), os = require('os');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const OUT = path.join(__dirname, '..', 'docs', 'screenshots');
const BASE = process.env.BASE || 'http://127.0.0.1:8765/index.html';
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const proc = spawn(process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    ['--headless=new', '--remote-debugging-port=9338', '--no-first-run', '--hide-scrollbars', '--user-data-dir=' + fs.mkdtempSync(path.join(os.tmpdir(), 'rl-')), 'about:blank'], { stdio: 'ignore' });
  let ws;
  for (let i = 0; i < 40; i++) {
    try { const l = await (await fetch('http://127.0.0.1:9338/json')).json(); const p = l.find((t) => t.type === 'page'); if (p) { ws = new WebSocket(p.webSocketDebuggerUrl); break; } } catch (e) { /* not up */ }
    await sleep(250);
  }
  await new Promise((r) => ws.addEventListener('open', r));
  let id = 0; const pend = new Map();
  ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result.result.value;
  const mouse = async (type, x, y) => send('Input.dispatchMouseEvent', { type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : 1, clickCount: 1 });
  const click = async (x, y) => { await mouse('mouseMoved', x, y); await mouse('mousePressed', x, y); await mouse('mouseReleased', x, y); };
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1.5, mobile: false });

  async function open(hash, theme = 'light') {
    await send('Page.navigate', { url: BASE + '#/' + hash });
    await sleep(800);
    await ev(`localStorage.setItem('rl-theme', '${theme}')`);
    await send('Page.reload'); await sleep(1800);
  }
  /** Screenshot of one element (with a margin), scrolled into view first. */
  async function shoot(name, selectorExpr, pad = 14) {
    await ev(`(${selectorExpr}).scrollIntoView({ block: 'start' }); window.scrollBy(0, -${pad + 6})`);
    await sleep(500);
    const r = await ev(`(() => { const b = (${selectorExpr}).getBoundingClientRect(); return { x: b.left + scrollX, y: b.top + scrollY, w: b.width, h: b.height }; })()`);
    const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: r.x - pad, y: r.y - pad, width: r.w + 2 * pad, height: r.h + 2 * pad, scale: 1 } });
    fs.writeFileSync(path.join(OUT, name + '.png'), Buffer.from(shot.result.data, 'base64'));
    console.log('saved', name, Math.round(r.w + 2 * pad) + '×' + Math.round(r.h + 2 * pad));
  }
  const expButton = (n) => `document.querySelectorAll('.exp-card .btn')[${n}]`;
  const setSlider = (label, value) => ev(`(() => { const row = [...document.querySelectorAll('.ctl-slider')].find(r => r.querySelector('.ctl-label').textContent.startsWith(${JSON.stringify(label)})); const i = row.querySelector('input'); i.value = ${value}; i.dispatchEvent(new Event('input')); })()`);

  // 1. Dynamic programming lab: cliff-edge world with slip, converged, one cell selected
  await open('mdp');
  await ev(`${expButton(2)}.click()`); await sleep(400);
  await setSlider('Slip probability', 0.2);
  await ev(`document.querySelector('.transport .btn.primary').click()`); await sleep(2500);
  const cv = await ev(`(() => { document.querySelector('.grid-canvas').scrollIntoView({ block: 'center' }); const b = document.querySelector('.grid-canvas').getBoundingClientRect(); return { x: b.left, y: b.top, w: b.width, h: b.height }; })()`);
  await sleep(300);
  await click(cv.x + (2.5 / 9) * cv.w, cv.y + (2.5 / 4) * cv.h);
  await mouse('mouseMoved', 5, 5); await sleep(300);
  await shoot('dp-lab', `document.querySelector('.lab')`);

  // 2. Tabular TD: cliff walking, Q-learning, Q-triangles after training
  await open('td');
  await ev(`${expButton(1)}.click()`); await sleep(6000);
  await ev(`document.querySelector('.transport .btn.primary').click()`); await sleep(300);
  await shoot('td-cliff', `document.querySelector('.lab')`);

  // 3. Bandit internals (UCB) after some steps
  await open('bandits');
  await ev(`(() => { const lab = document.querySelectorAll('.lab')[0]; lab.querySelector('.transport .btn.primary').click(); })()`);
  await sleep(5000);
  await ev(`document.querySelectorAll('.lab')[0].querySelector('.transport .btn.primary').click()`);
  await shoot('bandit-ucb', `document.querySelectorAll('.lab')[0]`);

  // 3b. Tutor bubble after manual steps in the bandit lab
  await ev(`localStorage.removeItem('rl-tutor-muted')`);
  await ev(`(() => { const lab = document.querySelectorAll('.lab')[0]; lab.querySelector('.transport .btn.icon').click(); lab.scrollIntoView({ block: 'start' }); window.scrollBy(0, -20); })()`);
  await sleep(300);
  for (let k = 0; k < 24; k++) await ev(`[...document.querySelectorAll('.lab')[0].querySelectorAll('.transport .btn')].find(b => b.textContent.trim() === 'Step').click()`);
  await ev(`document.querySelectorAll('.lab')[0].scrollIntoView({ block: 'end' }); window.scrollBy(0, 24)`);
  await sleep(900);
  {
    const b = await ev(`(() => { const r = document.querySelector('.coach').getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height }; })()`);
    const x = Math.max(0, b.x - 560), y = Math.max(0, b.y - 140);
    const shot = await send('Page.captureScreenshot', { format: 'png', clip: { x, y, width: b.x + b.w + 16 - x, height: b.y + b.h + 16 - y, scale: 1 } });
    fs.writeFileSync(path.join(OUT, 'tutor.png'), Buffer.from(shot.result.data, 'base64'));
    console.log('saved tutor');
  }
  await ev(`RL.ui.closeCoach()`);

  // 4. PPO on CartPole (dark theme)
  await open('ppo');
  await ev(`${expButton(0)}.click()`); await sleep(9000);
  await shoot('ppo-cartpole', `document.querySelector('.lab')`);

  // 5. GRPO toy (dark theme)
  await open('llm');
  await ev(`${expButton(0)}.click()`); await sleep(7000);
  await ev(`document.querySelectorAll('.lab')[0].querySelector('.transport .btn.primary').click()`); await sleep(200);
  await shoot('grpo', `document.querySelectorAll('.lab')[0]`);

  // 6. MCTS: two moves played, analysis overlay visible
  await open('planning');
  const cell = async (i) => { const b = await ev(`(() => { const r = document.querySelectorAll('.ttt-cell')[${i}].getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; })()`); await click(b.x, b.y); };
  await ev(`document.querySelector('.ttt').scrollIntoView({ block: 'center' })`); await sleep(300);
  await cell(4); await sleep(1200);
  await mouse('mouseMoved', 5, 5); await sleep(300);
  await shoot('mcts', `document.querySelector('.lab')`);

  // 7. KL-regularized optimum / reward hacking widget
  await open('llm');
  await setSlider('KL coefficient', 480); // log slider position (0–1000) ≈ β 0.5
  await sleep(300);
  await shoot('reward-hacking', `document.querySelector('.widget')`);

  // 8. Search palette
  await open('pg');
  await ev(`RL.search.open('policy gradient')`); await sleep(600);
  await shoot('search', `document.querySelector('.search-dialog')`, 8);

  ws.close(); proc.kill();
  process.exit(0);
})();
