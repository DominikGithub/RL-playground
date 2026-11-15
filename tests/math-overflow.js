// Reports display formulas that need horizontal scrolling, per chapter, at several viewport widths.
//   python -m http.server 8765  (separate terminal), then: node tests/math-overflow.js [width ...]
const { spawn } = require('child_process'); const fs = require('fs'), path = require('path'), os = require('os');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const WIDTHS = process.argv.slice(2).map(Number).filter(Boolean);
(async () => {
  const proc = spawn(process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe', ['--headless=new', '--remote-debugging-port=9336', '--no-first-run', '--user-data-dir=' + fs.mkdtempSync(path.join(os.tmpdir(), 'rl-')), 'about:blank'], { stdio: 'ignore' });
  let ws; for (let i = 0; i < 40; i++) { try { const l = await (await fetch('http://127.0.0.1:9336/json')).json(); const p = l.find((t) => t.type === 'page'); if (p) { ws = new WebSocket(p.webSocketDebuggerUrl); break; } } catch (e) {} await sleep(250); }
  await new Promise((r) => ws.addEventListener('open', r)); let id = 0; const pend = new Map();
  ws.addEventListener('message', (ev) => { const m = JSON.parse(ev.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); } });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (e) => (await send('Runtime.evaluate', { expression: e, awaitPromise: true, returnByValue: true })).result.result.value;
  await send('Page.enable');
  let total = 0;
  for (const W of WIDTHS.length ? WIDTHS : [1440, 1920, 1100, 390]) {
    await send('Emulation.setDeviceMetricsOverride', { width: W, height: 1000, deviceScaleFactor: 1, mobile: W < 600 });
    await send('Page.navigate', { url: 'http://127.0.0.1:8765/index.html#/intro' }); await sleep(1500);
    const out = [];
    for (const ch of ['intro', 'mdp', 'bandits', 'td', 'dqn', 'pg', 'ppo', 'planning', 'llm', 'landscape', 'cheatsheet']) {
      await ev(`location.hash = '#/${ch}'`); await sleep(500);
      await ev(`document.querySelectorAll('details').forEach(d => d.open = true)`); await sleep(350);
      const r = await ev(`[...document.querySelectorAll('.katex-display')].filter(d => d.scrollWidth > d.clientWidth + 1).map(d => ({ over: d.scrollWidth - d.clientWidth, w: d.clientWidth, fs: getComputedStyle(d).fontSize, tex: (d.querySelector('annotation') || {}).textContent.slice(0, 60) }))`);
      const page = await ev(`document.documentElement.scrollWidth - document.documentElement.clientWidth`);
      r.forEach((x) => out.push(`  ${ch}: +${x.over}px (box ${x.w}px, font ${x.fs}) ${x.tex.replace(/\s+/g, ' ')}`));
      if (page > 0) out.push(`  ${ch}: PAGE overflows horizontally by ${page}px`);
    }
    total += out.length;
    console.log(`viewport ${W}px: ${out.length} scrolling formulas` + (out.length ? '\n' + out.join('\n') : ''));
  }
  ws.close(); proc.kill();
  process.exit(0);
})();
