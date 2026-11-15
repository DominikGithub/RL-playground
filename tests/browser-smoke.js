// Browser smoke test via the Chrome DevTools Protocol (no dependencies; Node >= 22 for global WebSocket).
// Usage: start a static server on :8765 (python -m http.server 8765), then
//   node tests/browser-smoke.js [outDir]
// Visits every chapter, clicks the first guided experiment, collects JS errors and saves screenshots.
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const CHROME = process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const BASE = process.env.BASE || 'http://127.0.0.1:8765/index.html';
const OUT = process.argv[2] || path.join(__dirname, 'screenshots');
const CHAPTERS = ['intro', 'mdp', 'bandits', 'td', 'dqn', 'pg', 'ppo', 'planning', 'llm', 'landscape', 'cheatsheet'];
fs.mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const userDir = fs.mkdtempSync(path.join(require('os').tmpdir(), 'rlpg-'));
  const proc = spawn(CHROME, ['--headless=new', '--remote-debugging-port=9333', '--window-size=1440,1000', '--no-first-run', '--user-data-dir=' + userDir, 'about:blank'], { stdio: 'ignore' });
  let ws;
  for (let i = 0; i < 40; i++) {
    try {
      const list = await (await fetch('http://127.0.0.1:9333/json')).json();
      const page = list.find((t) => t.type === 'page');
      if (page) { ws = new WebSocket(page.webSocketDebuggerUrl); break; }
    } catch (e) { /* not up yet */ }
    await sleep(250);
  }
  await new Promise((r) => ws.addEventListener('open', r));
  let id = 0; const pending = new Map(); const errors = [];
  ws.addEventListener('message', (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
    if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
    if (msg.method === 'Runtime.consoleAPICalled' && (msg.params.type === 'error' || msg.params.type === 'warning')) errors.push('console.' + msg.params.type + ': ' + msg.params.args.map((a) => a.value || a.description).join(' '));
    if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') errors.push('log: ' + msg.params.entry.text + ' ' + (msg.params.entry.url || ''));
  });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
  await send('Runtime.enable'); await send('Log.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  if (process.env.DARK) await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'dark' }] });

  const shot = async (name) => {
    const r = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(path.join(OUT, name + '.png'), Buffer.from(r.result.data, 'base64'));
  };
  await send('Page.navigate', { url: BASE + '#/intro' });
  await sleep(1500);
  if (errors.length) { console.log('LOAD ERRORS:'); errors.forEach((e) => console.log('   ' + String(e).split('\n')[0])); }
  const loadErrors = errors.length;
  const report = { load: { errors: errors.slice() } };
  for (const ch of CHAPTERS) {
    errors.length = 0;
    await evaluate(`location.hash = '#/${ch}'`);
    await sleep(900);
    await shot(ch + '-top');
    const katexErrors = await evaluate(`document.querySelectorAll('.katex-error').length`);
    const study = await evaluate(`({ rows: document.querySelectorAll('.study-row').length, chips: document.querySelectorAll('.study-chip').length, cards: document.querySelectorAll('.study-card').length })`);
    if (ch !== 'intro' || study.cards) console.log(`   study material: ${study.rows} section rows, ${study.chips} chips, ${study.cards} cards`);
    const studyIssues = await evaluate(`(() => {
      const out = [];
      document.querySelectorAll('.study-row').forEach((r) => {
        const kinds = [...r.querySelectorAll('.study-chip')].map((c) => c.className);
        if (kinds.some((k) => k.includes('k-paper')) && kinds.every((k) => k.includes('k-paper'))) out.push('paper-only row: ' + r.previousElementSibling.textContent);
      });
      document.querySelectorAll('.study-card').forEach((c) => {
        if (c.querySelector('.study-links a[href*="arxiv.org"]') && !c.querySelector('.study-read')) out.push('paper card without reading hint: ' + c.querySelector('.study-title').textContent);
      });
      return out;
    })()`);
    if (studyIssues.length) { errors.push(ch + ': ' + studyIssues.join('; ')); console.log('   ✗ ' + studyIssues.join('\n   ✗ ')); }
    const infos = await evaluate(`({ total: document.querySelectorAll('.info-btn').length, charts: document.querySelectorAll('.chart-title').length, chartsWithInfo: document.querySelectorAll('.chart-title .info-btn').length })`);
    console.log(`   legends: ${infos.total} ⓘ icons (${infos.chartsWithInfo}/${infos.charts} charts)`);
    const rawDollars = await evaluate(`(document.querySelector('.concepts')?.innerText.match(/\\$[^$\\n]{1,40}\\$/g) || []).slice(0,3).join(' | ')`);
    // run the first experiment, if any
    const hasExp = await evaluate(`(() => { const b = document.querySelector('.exp-card .btn'); if (b) { b.click(); return true; } return false; })()`);
    if (hasExp) { await sleep(ch === 'dqn' || ch === 'pg' || ch === 'ppo' ? 6000 : 2500); await evaluate(`window.scrollTo(0, 0)`); await sleep(200); await shot(ch + '-experiment'); }
    if (process.env.PARTS) {
      const n = await evaluate(`document.querySelectorAll('.widget, .lab').length`);
      for (let i = 0; i < n; i++) {
        await evaluate(`document.querySelectorAll('.widget, .lab')[${i}].scrollIntoView({block: 'start'})`);
        await sleep(400);
        await shot(`${ch}-part${i}`);
      }
    }
    const labStatus = await evaluate(`[...document.querySelectorAll('.lab-status')].map(e => e.textContent).join(' || ').slice(0, 300)`);
    report[ch] = { errors: errors.slice(), katexErrors, rawDollars, labStatus };
    console.log(`[${ch}] errors=${errors.length} katexErrors=${katexErrors} rawMath="${rawDollars}"\n   status: ${labStatus}`);
    errors.forEach((e) => console.log('   ERR ' + String(e).split('\n').slice(0, 3).join(' / ')));
  }
  // ⓘ legend popover: hover the bandit-internals icon
  errors.length = 0;
  await evaluate(`location.hash = '#/bandits'`); await sleep(900);
  await evaluate(`[...document.querySelectorAll('.lab-title')].find(t => /true means/.test(t.textContent)).scrollIntoView({ block: 'center' })`); await sleep(300);
  const ib = await evaluate(`(() => { const b = [...document.querySelectorAll('.lab-title')].find(t => /true means/.test(t.textContent)).querySelector('.info-btn').getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; })()`);
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: ib.x, y: ib.y });
  await sleep(300);
  const pop = await evaluate(`(() => { const p = document.querySelector('.info-pop'); if (!p) return null; const r = p.getBoundingClientRect(); return { items: p.querySelectorAll('.lg-list li').length, inView: r.left >= 0 && r.right <= innerWidth && r.top >= 0 && r.bottom <= innerHeight, text: p.textContent.slice(0, 80) }; })()`);
  console.log('   legend popover on hover: ' + JSON.stringify(pop));
  if (!pop || pop.items < 5 || !pop.inView) errors.push('legend popover missing or off-screen');
  await shot('legend-bandits');
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 5, y: 5 }); await sleep(400);
  if (await evaluate(`!!document.querySelector('.info-pop')`)) errors.push('legend popover did not close');
  errors.forEach((e) => console.log('   ERR ' + e));
  report.legend = { errors: errors.slice() };

  // tutor bubble: only on manual steps, rule-based verdicts
  errors.length = 0;
  const coach = () => evaluate(`(() => { const c = document.querySelector('.coach'); if (!c) return null; const ms = c.querySelectorAll('.coach-msg'), m = ms[ms.length - 1]; if (!m) return { tone: '(typing)', text: '' }; const tone = m.querySelector('.coach-tone'); return { tone: tone ? tone.textContent : '', text: m.textContent.replace(/\\s+/g, ' ').slice(0, 700) }; })()`);
  const clickBtn = (labIdx, label) => evaluate(`[...document.querySelectorAll('.lab')[${labIdx}].querySelectorAll('.transport .btn')].find(b => b.textContent.trim() === ${JSON.stringify(label)}).click()`);
  await evaluate(`localStorage.removeItem('rl-tutor-muted'); location.hash = '#/mdp'`); await sleep(900);
  await clickBtn(0, 'Sweep'); await sleep(600);
  let c1 = await coach();
  console.log('   tutor after sweep 1: ' + JSON.stringify(c1));
  if (!c1 || !/first sweep/i.test(c1.text)) errors.push('tutor: no first-sweep message');
  await clickBtn(0, 'Sweep'); await sleep(600);
  c1 = await coach();
  console.log('   tutor after sweep 2: ' + JSON.stringify(c1));
  if (!c1 || c1.tone !== 'improving' || !/contraction/.test(c1.text)) errors.push('tutor: sweep 2 should report contraction');
  await shot('tutor-dp');
  // policy iteration until optimal → "Optimal policy found"
  await evaluate(`(() => { const s = [...document.querySelectorAll('.lab')[0].querySelectorAll('select')].find(x => [...x.options].some(o => o.value === 'pi')); s.value = 'pi'; s.dispatchEvent(new Event('change')); })()`); await sleep(200);
  for (let k = 0; k < 400; k++) {
    await evaluate(`[...document.querySelectorAll('.lab')[0].querySelectorAll('.transport .btn')].find(b => b.textContent.trim() === 'Sweep').click()`);
    if (await evaluate(`/optimal/.test(document.querySelector('.lab-status').textContent)`)) break;
  }
  await sleep(600);
  c1 = await coach();
  console.log('   tutor at end of policy iteration: ' + JSON.stringify(c1));
  if (!c1 || !/Optimal policy found/.test(c1.text)) errors.push('tutor: policy iteration did not report optimal');
  // Run must not trigger the tutor
  // stays open (no timer), keeps a history, can be closed and reopened
  const hist = await evaluate(`document.querySelectorAll('.coach .coach-msg').length`);
  await evaluate(`document.querySelector('.coach .coach-x').click()`); await sleep(400);
  const launcher = await evaluate(`!!document.querySelector('.coach-launcher') && !document.querySelector('.coach')`);
  await evaluate(`document.querySelector('.coach-launcher').click()`); await sleep(300);
  const reopened = await evaluate(`document.querySelectorAll('.coach .coach-msg').length`);
  console.log(`   tutor history: ${hist} messages; closed → reopen button: ${launcher}; reopened with ${reopened} messages`);
  if (hist < 2 || !launcher || reopened !== hist) errors.push('tutor: history / reopen does not work');
  await shot('tutor-history');
  await evaluate(`document.querySelector('.coach .coach-x').click()`); await sleep(400);
  await evaluate(`document.querySelectorAll('.lab')[0].querySelector('.transport .btn.primary').click()`); await sleep(1500);
  await evaluate(`document.querySelectorAll('.lab')[0].querySelector('.transport .btn.primary').click()`);
  if (await evaluate(`!!document.querySelector('.coach')`)) errors.push('tutor appeared during Run');
  // bandit internals: 22 manual steps → trend verdict
  await evaluate(`location.hash = '#/bandits'`); await sleep(900);
  for (let k = 0; k < 22; k++) await clickBtn(0, 'Step');
  await sleep(700);
  const c2 = await coach();
  console.log('   tutor after 22 bandit steps: ' + JSON.stringify(c2));
  if (!c2 || !/Trend:/.test(c2.text) || !/pulled #/.test(c2.text)) errors.push('tutor: bandit message missing');
  await shot('tutor-bandit');
  await evaluate(`RL.ui.closeCoach()`);
  errors.forEach((e) => console.log('   ERR ' + e));
  report.tutor = { errors: errors.slice() };

  // grid-world editor (Chapter 1): drag & drop, select-then-click, start-cell guard, keyboard, info box overflow
  errors.length = 0;
  await evaluate(`location.hash = '#/mdp'`); await sleep(900);
  await evaluate(`document.querySelector('.grid-canvas').scrollIntoView({ block: 'center' })`); await sleep(300);
  const rectOf = (expr) => evaluate(`(() => { const r = (${expr}).getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; })()`);
  const mouse = (type, x, y) => send('Input.dispatchMouseEvent', { type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : 1, clickCount: 1, pointerType: 'mouse' });
  const tileSel = (name) => `[...document.querySelectorAll('.tool-tile')].find(t => t.querySelector('.tool-name').textContent === '${name}')`;
  const cv = await rectOf(`document.querySelector('.grid-canvas')`);
  const cs = cv.w / 4; // classic 4×3 world
  const cell = (x, y) => [cv.x + (x + 0.5) * cs, cv.y + (y + 0.5) * cs];
  const center = (r) => [r.x + r.w / 2, r.y + r.h / 2];
  const editorState = () => evaluate(`({ sel: document.querySelector('.edit-selected').textContent, hint: document.querySelector('.edit-hint').textContent, warn: document.querySelector('.edit-hint').classList.contains('warn'), active: [...document.querySelectorAll('.tool-tile.active')].map(t => t.querySelector('.tool-name').textContent).join(',') })`);
  const expect = (name, st, cond) => { console.log(`   ${cond ? 'OK ' : 'BAD'} ${name}: ${JSON.stringify(st)}`); if (!cond) errors.push('editor: ' + name); };
  // 1. drag the Wall tile onto cell (1,0)
  let [sx, sy] = center(await rectOf(tileSel('Wall')));
  let [tx, ty] = cell(1, 0);
  await mouse('mousePressed', sx, sy);
  for (let k = 1; k <= 12; k++) await mouse('mouseMoved', sx + ((tx - sx) * k) / 12, sy + ((ty - sy) * k) / 12);
  await shot('editor-dragging');
  await mouse('mouseReleased', tx, ty); await sleep(200);
  let st = await editorState();
  expect('drag Wall onto (1,0)', st, /\(1, 0\).*Wall/.test(st.sel) && st.active === 'Wall');
  // 2. select cell (2,0) then click the Pit tile
  [tx, ty] = cell(2, 0);
  await mouse('mousePressed', tx, ty); await mouse('mouseReleased', tx, ty); await sleep(100);
  st = await editorState();
  expect('select empty cell highlights Empty', st, st.active === 'Empty' && /\(2, 0\)/.test(st.sel));
  [sx, sy] = center(await rectOf(tileSel('Pit')));
  await mouse('mousePressed', sx, sy); await mouse('mouseReleased', sx, sy); await sleep(150);
  st = await editorState();
  expect('click Pit applies to selection', st, /Pit/.test(st.sel) && st.active === 'Pit');
  // 3. start cell is protected
  [tx, ty] = cell(0, 2);
  await mouse('mousePressed', tx, ty); await mouse('mouseReleased', tx, ty); await sleep(100);
  [sx, sy] = center(await rectOf(tileSel('Goal')));
  await mouse('mousePressed', sx, sy); await mouse('mouseReleased', sx, sy); await sleep(150);
  st = await editorState();
  expect('start cell cannot become Goal', st, st.warn && st.active === 'Start');
  // 4. keyboard: move right, press G
  await send('Input.dispatchKeyEvent', { type: 'rawKeyDown', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 });
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'g', code: 'KeyG', text: 'g', windowsVirtualKeyCode: 71 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'g', code: 'KeyG', windowsVirtualKeyCode: 71 });
  await sleep(150);
  st = await editorState();
  expect('keyboard → then G', st, /\(1, 2\).*Goal/.test(st.sel) && st.active === 'Goal');
  // 4b. Bellman box stays in sync with the selection: hover shows the hovered cell, leaving the grid returns to the selection
  const boxHead = () => evaluate(`document.querySelector('.bellman-box .bb-head') ? document.querySelector('.bellman-box .bb-head').textContent : document.querySelector('.bellman-box').textContent.slice(0, 60)`);
  [tx, ty] = cell(2, 1);
  await mouse('mouseMoved', tx, ty); await mouse('mousePressed', tx, ty); await mouse('mouseReleased', tx, ty); await sleep(100);
  let head = await boxHead();
  expect('click (2,1) → box shows selected (2,1)', head, /selected/i.test(head) && /\(2, 1\)/.test(head));
  [tx, ty] = cell(0, 0);
  await mouse('mouseMoved', tx, ty); await sleep(100);
  head = await boxHead();
  expect('hover (0,0) → box shows hovered (0,0)', head, /hover/i.test(head) && /\(0, 0\)/.test(head));
  await mouse('mouseMoved', cv.x + cv.w + 120, cv.y - 40); await sleep(100); // leave the grid
  head = await boxHead();
  expect('leave grid → box back to selected (2,1)', head, /selected/i.test(head) && /\(2, 1\)/.test(head));

  // 5. Bellman box on a free cell: no horizontal scrolling
  [tx, ty] = cell(2, 1);
  await mouse('mouseMoved', tx, ty); await mouse('mousePressed', tx, ty); await mouse('mouseReleased', tx, ty); await sleep(150);
  await evaluate(`document.querySelector('.bellman-box').scrollIntoView({ block: 'center' })`); await sleep(200);
  const ov = await evaluate(`(() => { const b = document.querySelector('.bellman-box'); return { scrollW: b.scrollWidth, clientW: b.clientWidth, tiles: b.querySelectorAll('.q-tile').length }; })()`);
  expect('Bellman box fits (no h-scroll)', ov, ov.tiles === 4 && ov.scrollW <= ov.clientW + 1);
  await shot('editor-bellman');
  await evaluate(`document.querySelector('.lab').scrollIntoView({ block: 'start' })`); await sleep(200);
  await shot('editor-lab');
  report.editor = { errors: errors.slice() };
  errors.forEach((e) => console.log('   ERR ' + String(e).split('\n')[0]));

  // search: local index, jump to section, open collapsed boxes, Wikipedia, LLM settings
  errors.length = 0;
  const typeQuery = (q) => evaluate(`(() => { RL.search.open(); const i = document.querySelector('.search-input'); i.value = ${JSON.stringify(q)}; i.dispatchEvent(new Event('input')); })()`);
  const t0 = Date.now();
  await evaluate(`RL.search.buildIndex().length`);
  console.log(`[search] index entries: ${await evaluate(`RL.search.buildIndex().length`)} (built in ${Date.now() - t0} ms)`);
  for (const [q, expectCh] of [['clipped surrogate', 'ppo'], ['contraction', 'mdp'], ['deadly triad', 'dqn'], ['thompson', 'bandits'], ['GRPO', 'llm']]) {
    await typeQuery(q); await sleep(300);
    const top = await evaluate(`[...document.querySelectorAll('.search-item')].slice(0, 3).map(li => li.querySelector('.search-item-title').textContent + ' [' + li.querySelector('.search-item-ch').textContent + ']').join(' | ')`);
    await evaluate(`document.querySelector('.search-input').dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))`);
    await sleep(1200);
    const where = await evaluate(`location.hash + ' hit=' + !!document.querySelector('.search-hit') + ' openDetails=' + document.querySelectorAll('#main details[open]').length`);
    const ok = where.startsWith('#/' + expectCh) && where.includes('hit=true');
    console.log(`   ${ok ? 'OK ' : 'BAD'} "${q}" → ${where}\n       top: ${top}`);
    if (!ok) errors.push('search "' + q + '" landed at ' + where);
  }
  await shot('search-landing');
  await typeQuery('temporal difference'); await sleep(300);
  await shot('search-overlay');
  await evaluate(`[...document.querySelectorAll('.search-ext-buttons button')].find(b => b.textContent === 'Wikipedia').click()`);
  await sleep(4000);
  const wiki = await evaluate(`document.querySelector('.search-ext-result').innerText.slice(0, 260).replace(/\\s+/g, ' ')`);
  console.log('   wikipedia: ' + wiki);
  await shot('search-wikipedia');
  await evaluate(`[...document.querySelectorAll('.search-ext-buttons button')].find(b => b.textContent === 'Ask an LLM').click()`);
  await sleep(300);
  console.log('   llm settings form shown: ' + (await evaluate(`!!document.querySelector('.llm-form')`)));
  await shot('search-llm-settings');
  // LLM answer path against a local mock of an OpenAI-compatible endpoint (CORS enabled)
  let received = null;
  const mock = require('http').createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    if (req.method === 'OPTIONS') { res.end(); return; }
    let body = ''; req.on('data', (c) => { body += c; });
    req.on('end', () => {
      received = JSON.parse(body);
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ model: 'mock-model', choices: [{ message: { content: 'TD learning bootstraps: $\\delta_t = r + \\gamma V(s\') - V(s)$. <img src=x onerror=alert(1)>' } }] }));
    });
  }).listen(8799);
  await evaluate(`localStorage.setItem('rl-llm-settings', JSON.stringify({ preset: 'custom', url: 'http://127.0.0.1:8799/v1/chat/completions', model: 'mock-model', key: '' }))`);
  await evaluate(`[...document.querySelectorAll('.search-ext-buttons button')].find(b => b.textContent === 'Ask an LLM').click()`);
  await sleep(1500);
  const llm = await evaluate(`({ text: document.querySelector('.search-ext-result').innerText.replace(/\\s+/g, ' ').slice(0, 300), katex: document.querySelectorAll('.llm-answer .katex').length, injectedImg: document.querySelectorAll('.llm-answer img').length })`);
  console.log('   llm answer: ' + JSON.stringify(llm));
  console.log('   llm request: model=' + (received && received.model) + ', messages=' + (received && received.messages.length) + ', context chars=' + (received && received.messages[1].content.length));
  if (!llm || !/mock-model/.test(llm.text) || llm.injectedImg) errors.push('LLM answer path failed or injected HTML');
  await shot('search-llm-answer');
  await evaluate(`localStorage.removeItem('rl-llm-settings')`);
  mock.close();
  await evaluate(`RL.search.close()`);
  report.search = { errors: errors.slice() };
  errors.forEach((e) => console.log('   ERR ' + String(e).split('\n')[0]));

  // mobile layout check
  await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
  await evaluate(`location.hash = '#/td'`); await sleep(1200);
  const overflow = await evaluate(`document.documentElement.scrollWidth - document.documentElement.clientWidth`);
  console.log('mobile horizontal overflow px:', overflow);
  await shot('mobile-td');
  ws.close(); proc.kill();
  const total = Object.values(report).reduce((s, r) => s + r.errors.length + (r.katexErrors || 0), 0);
  console.log(total ? `\n${total} problems found` : '\nno errors');
})();
