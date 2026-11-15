/* Search palette: local full-text index of all chapters (jump to the exact section), plus external sources —
   Wikipedia (inline, via its public API), DuckDuckGo (link-out) and an optional, user-configured LLM endpoint.
   Every external result is labelled with its source. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { h } = RL.util;

  /* ---------------- text helpers ---------------- */
  const TEX = {
    alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', theta: 'θ', lambda: 'λ', mu: 'μ',
    pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', phi: 'φ', nabla: '∇', sum: 'Σ', prod: 'Π', infty: '∞', le: '≤', ge: '≥',
    leq: '≤', geq: '≥', approx: '≈', sim: '~', in: '∈', cdot: '·', mid: '|', to: '→', leftarrow: '←', pm: '±', lt: '<', gt: '>',
    max: 'max', min: 'min', log: 'log', exp: 'exp', arg: 'arg', ln: 'ln', E: 'E', Pr: 'Pr', KL: 'KL', mathcal: '',
  };
  /** Turn inline LaTeX into readable plain text (for snippets and indexing). */
  const texToText = (s) => s.replace(/\$\$?([^$]+)\$\$?/g, (_, t) => ' ' + t
    .replace(/\\(?:text|mathrm|operatorname|mathbb|mathbf|hat|bar|tilde|boldsymbol)\s*\{([^}]*)\}/g, '$1')
    .replace(/\\([a-zA-Z]+)/g, (m, w) => (w in TEX ? TEX[w] : ''))
    .replace(/[{}^_&]/g, '')
    .replace(/\\[,;!:]/g, ' ') + ' ');
  const clean = (s) => texToText(s || '').replace(/\s+/g, ' ').trim();
  const norm = (s) => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[-_/]/g, ' ');
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /** Abbreviations → expansions, so "ppo" also finds "proximal policy optimization" and vice versa. */
  const SYN = {
    ppo: 'proximal policy optimization', trpo: 'trust region', dqn: 'deep q', mcts: 'monte carlo tree search',
    rlhf: 'human feedback', gae: 'generalized advantage', td: 'temporal difference', mdp: 'markov decision',
    sac: 'soft actor critic', dpo: 'direct preference', grpo: 'group relative', kl: 'kullback', pg: 'policy gradient',
    ucb: 'upper confidence', mc: 'monte carlo', rlvr: 'verifiable reward', bc: 'behavior cloning', irl: 'inverse rl',
    a2c: 'advantage actor critic', uct: 'upper confidence', per: 'prioritized replay', llm: 'language model',
  };

  /* ---------------- index ---------------- */
  let INDEX = null;
  const BLOCK = 'p,li,td,th,div,h1,h2,h3,h4,summary,tr,ul,ol,table,label,option,button,.tag,.lbl,.kicker';
  /** textContent that inserts a space whenever we cross into a different block element. */
  function textOf(el) {
    if (!el) return '';
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let out = '', lastBlock = null, n;
    while ((n = w.nextNode())) {
      const b = n.parentElement && n.parentElement.closest(BLOCK);
      out += (b !== lastBlock ? ' ' : '') + n.nodeValue;
      lastBlock = b;
    }
    return out;
  }

  /** Extract searchable entries from a mounted chapter. Each entry remembers (selector, index) to jump back to it. */
  function extract(ch, root, out) {
    const add = (sel, i, kind, title, text) => {
      const t = clean(title), x = clean(text);
      if (!t) return;
      out.push({ ch: ch.id, chTitle: ch.title, num: ch.num, sel, i, kind, title: t, text: x, nt: norm(t), nx: norm(x) });
    };
    const head = root.querySelector('.chapter-head');
    if (head) add('.chapter-head', 0, 'chapter', ch.title, textOf(head.querySelector('h1')) + ' — ' + textOf(head.querySelector('.lede')) + ' ' + textOf(head.querySelector('.tags')));
    const textUntil = (el, stops) => {
      let t = '', n = el.nextElementSibling;
      while (n && !stops.includes(n.tagName)) { if (!n.matches('details.deep, .study-row')) t += ' ' + textOf(n); n = n.nextElementSibling; }
      return t;
    };
    root.querySelectorAll('.concepts h2').forEach((el, i) => add('.concepts h2', i, 'section', el.textContent, textUntil(el, ['H2'])));
    root.querySelectorAll('.concepts h3').forEach((el, i) => add('.concepts h3', i, 'section', el.textContent, textUntil(el, ['H2', 'H3'])));
    root.querySelectorAll('details.deep').forEach((el, i) => {
      const sum = el.querySelector('summary');
      const lbl = sum.querySelector('.lbl');
      const kind = lbl ? lbl.textContent.trim().toLowerCase() : 'details';
      add('details.deep', i, kind, sum.textContent.replace(lbl ? lbl.textContent : '', ''), textOf(el.querySelector('.body')));
    });
    root.querySelectorAll('.widget').forEach((el, i) => {
      const tips = [...el.querySelectorAll('[data-tip]')].map((x) => x.dataset.tip).join(' ');
      add('.widget', i, 'interactive', el.querySelector('.widget-title').textContent, textOf(el.querySelector('.widget-sub')) + ' ' + textOf(el) + ' ' + tips);
    });
    root.querySelectorAll('.lab').forEach((el, i) => {
      const labels = [...el.querySelectorAll('.ctl-label, .ctl-group-title, option, .chart-title')].map((x) => x.textContent).join(' · ');
      const tips = [...el.querySelectorAll('[data-tip]')].map((x) => x.dataset.tip).join(' ');
      add('.lab', i, 'lab', (el.getAttribute('aria-label') || 'Lab') + ' — ' + ch.title, labels + ' ' + tips);
    });
    root.querySelectorAll('.exp-card').forEach((el, i) => add('.exp-card', i, 'experiment', el.querySelector('.exp-card-title').textContent, el.querySelector('.exp-text').textContent));
    root.querySelectorAll('table.compare tr').forEach((el, i) => {
      const cells = el.querySelectorAll('td');
      if (cells.length < 2) return;
      const first = cells[0].textContent;
      if (first.length > 90) return;
      add('table.compare tr', i, 'table', first, [...cells].map((c) => textOf(c)).join(' · '));
    });
    root.querySelectorAll('.card').forEach((el, i) => add('.card', i, 'card', (el.querySelector('h4') || el).firstChild.textContent, textOf(el.querySelector('p'))));
  }

  /** Mount every chapter once into a hidden container (no math rendering), harvest its text, then destroy it. */
  function buildIndex() {
    if (INDEX) return INDEX;
    const host = h('div', { style: { display: 'none' }, 'aria-hidden': 'true' });
    document.body.appendChild(host);
    const savedTheme = new Set(RL._theme);
    RL._noMath = true;
    const entries = [];
    for (const ch of RL.chapters) {
      const root = h('article.chapter');
      host.appendChild(root);
      let destroy = null;
      try { destroy = ch.mount(root); if (RL.resources) RL.resources.decorate(root, ch.id); extract(ch, root, entries); } catch (e) { console.warn('search index: ' + ch.id, e); }
      try { if (destroy) destroy(); } catch (e) { /* ignore */ }
      root.remove();
    }
    RL._noMath = false;
    RL._theme.clear(); savedTheme.forEach((f) => RL._theme.add(f));
    host.remove();
    INDEX = entries;
    return INDEX;
  }

  const KIND_W = { chapter: 1.25, section: 1.15, math: 1.05, basics: 1.05, practice: 1, history: 0.95, interactive: 1, lab: 0.85, experiment: 0.85, table: 0.9, card: 1 };

  function search(q, limit = 8) {
    const idx = buildIndex();
    const nq = norm(q).trim();
    if (!nq) return [];
    const terms = nq.split(/\s+/).filter(Boolean);
    const results = [];
    for (const e of idx) {
      let score = 0, matched = 0;
      for (const t of terms) {
        const alts = [t]; if (SYN[t]) alts.push(SYN[t]);
        for (const k in SYN) if (SYN[k] === t || (t.length > 3 && SYN[k].startsWith(t))) alts.push(k);
        let s = 0;
        for (const a of alts) {
          const wordRe = new RegExp('(^|\\s|\\()' + a.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
          if (e.nt.includes(a)) s = Math.max(s, wordRe.test(e.nt) ? 12 : 7);
          else if (e.nx.includes(a)) {
            const c = e.nx.split(a).length - 1;
            s = Math.max(s, (wordRe.test(e.nx) ? 3 : 1.5) + Math.min(4, c) * 0.5);
          }
        }
        if (s > 0) matched++;
        score += s;
      }
      if (!matched) continue;
      if (matched < terms.length) score *= 0.35 * (matched / terms.length);
      if (terms.length > 1) { if (e.nt.includes(nq)) score += 15; else if (e.nx.includes(nq)) score += 5; }
      score *= KIND_W[e.kind] || 1;
      results.push({ e, score, complete: matched === terms.length });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, limit);
  }

  function snippet(text, q, len = 170) {
    const terms = norm(q).split(/\s+/).filter((t) => t.length > 1);
    const nt = norm(text);
    let pos = -1;
    for (const t of terms) { pos = nt.indexOf(t); if (pos >= 0) break; }
    let start = Math.max(0, pos - 60);
    if (pos < 0) start = 0;
    let s = text.slice(start, start + len);
    s = (start > 0 ? '…' : '') + s + (start + len < text.length ? '…' : '');
    let html = esc(s);
    for (const t of terms) {
      const re = new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
      html = html.replace(re, '<mark>$1</mark>');
    }
    return html;
  }

  /* ---------------- external sources ---------------- */
  const CONTEXT = ' reinforcement learning';
  const withContext = (q) => (/reinforcement|machine learning/i.test(q) ? q : q + CONTEXT);

  async function wikipedia(q, signal) {
    const api = (s) => 'https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&origin=*&srlimit=6&srsearch=' + encodeURIComponent(s);
    let used = withContext(q);
    let j = await (await fetch(api(used), { signal })).json();
    if (!j.query || !j.query.search.length) { used = q; j = await (await fetch(api(used), { signal })).json(); }
    const hits = (j.query && j.query.search) || [];
    if (!hits.length) return { used, hits: [] };
    const top = hits[0].title;
    const sum = await (await fetch('https://en.wikipedia.org/api/rest_v1/page/summary/' + encodeURIComponent(top.replace(/ /g, '_')), { signal })).json();
    const strip = (html) => new DOMParser().parseFromString(html || '', 'text/html').body.textContent;
    return {
      used,
      top: { title: sum.title || top, extract: sum.extract || '', url: (sum.content_urls && sum.content_urls.desktop && sum.content_urls.desktop.page) || 'https://en.wikipedia.org/wiki/' + encodeURIComponent(top.replace(/ /g, '_')) },
      hits: hits.slice(1).map((x) => ({ title: x.title, snippet: strip(x.snippet), url: 'https://en.wikipedia.org/wiki/' + encodeURIComponent(x.title.replace(/ /g, '_')) })),
    };
  }

  const LLM_PRESETS = {
    ollama: { label: 'Ollama (local, free)', url: 'http://localhost:11434/v1/chat/completions', model: 'llama3.2', needsKey: false,
      help: 'Runs on your machine. Start Ollama with OLLAMA_ORIGINS set to this site\'s origin (e.g. OLLAMA_ORIGINS="*") so the browser may call it.' },
    openrouter: { label: 'OpenRouter', url: 'https://openrouter.ai/api/v1/chat/completions', model: '', needsKey: true,
      help: 'Needs an OpenRouter API key. Models whose id ends in ":free" cost nothing (subject to rate limits).' },
    openai: { label: 'OpenAI', url: 'https://api.openai.com/v1/chat/completions', model: '', needsKey: true, help: 'Needs an OpenAI API key.' },
    custom: { label: 'Custom (OpenAI-compatible)', url: '', model: '', needsKey: false, help: 'Any endpoint implementing POST /v1/chat/completions with CORS enabled.' },
  };
  const LLM_KEY = 'rl-llm-settings';
  const loadLLM = () => { try { return JSON.parse(localStorage.getItem(LLM_KEY)) || null; } catch (e) { return null; } };
  const saveLLM = (s) => { try { if (s) localStorage.setItem(LLM_KEY, JSON.stringify(s)); else localStorage.removeItem(LLM_KEY); } catch (e) { /* storage unavailable */ } };

  async function askLLM(q, cfg, contextResults, signal) {
    const ctx = contextResults.slice(0, 4).map((r, i) =>
      `[${i + 1}] ${r.e.chTitle}${r.e.num !== undefined ? ' (Chapter ' + r.e.num + ')' : ''} — ${r.e.title}: ${r.e.text.slice(0, 600)}`).join('\n\n');
    const messages = [
      { role: 'system', content: 'You are a concise, precise tutor inside "RL Playground", an interactive reinforcement-learning course for ML engineers. ' +
        'Answer in at most ~200 words. Use LaTeX between $...$ for math. Prefer the provided course excerpts; refer to them by chapter name when relevant. ' +
        'If you are not sure, say so instead of guessing.' },
      { role: 'user', content: (ctx ? 'Course excerpts:\n' + ctx + '\n\n' : '') + 'Question: ' + q },
    ];
    const headers = { 'Content-Type': 'application/json' };
    if (cfg.key) headers.Authorization = 'Bearer ' + cfg.key;
    const res = await fetch(cfg.url, { method: 'POST', headers, signal, body: JSON.stringify({ model: cfg.model, messages, temperature: 0.3, stream: false }) });
    if (!res.ok) { let t = ''; try { t = (await res.text()).slice(0, 300); } catch (e) { /* ignore */ } throw new Error('HTTP ' + res.status + (t ? ': ' + t : '')); }
    const j = await res.json();
    const text = j.choices && j.choices[0] && (j.choices[0].message ? j.choices[0].message.content : j.choices[0].text);
    if (!text) throw new Error('Unexpected response format (no choices[0].message.content).');
    return { text, model: j.model || cfg.model };
  }

  /* ---------------- UI ---------------- */
  let ui = null, results = [], sel = 0, ctrl = null, lastFocus = null;

  function buildUI() {
    const input = h('input.search-input', { type: 'search', placeholder: 'Search concepts, algorithms, formulas…  e.g. “clipped objective”, “deadly triad”', 'aria-label': 'Search', autocomplete: 'off', spellcheck: 'false' });
    const list = h('ul.search-results', { role: 'listbox', 'aria-label': 'Results in this playground' });
    const ext = h('div.search-ext-result', { 'aria-live': 'polite' });
    const btnWiki = h('button.btn.small', { type: 'button' }, 'Wikipedia');
    const btnWeb = h('a.btn.small', { target: '_blank', rel: 'noopener noreferrer', href: '#' }, 'Web search (DuckDuckGo) ↗');
    const btnLLM = h('button.btn.small', { type: 'button' }, 'Ask an LLM');
    const btnCfg = h('button.btn.small.icon', { type: 'button', title: 'LLM settings', 'aria-label': 'LLM settings' }, '⚙');
    const closeBtn = h('button.search-close', { type: 'button', 'aria-label': 'Close search' }, 'Esc');
    const dialog = h('div.search-dialog', { role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Search' },
      h('div.search-bar', h('span.search-icon', { 'aria-hidden': 'true' }, '⌕'), input, closeBtn),
      h('div.search-body',
        ext,
        h('div.search-section-title', 'In this playground'),
        list),
      h('div.search-footer',
        h('span.search-footer-label', 'Other sources', h('span.muted', ' — sends your query to the chosen service')),
        h('div.search-ext-buttons', btnWiki, btnWeb, btnLLM, btnCfg)));
    const overlay = h('div.search-overlay', { hidden: true }, dialog);
    document.body.appendChild(overlay);

    overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });
    closeBtn.addEventListener('click', close);
    let t = 0;
    input.addEventListener('input', () => { clearTimeout(t); t = setTimeout(runLocal, 60); if (ctrl) ctrl.abort(); ext.innerHTML = ''; });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
      else if (e.key === 'Enter') { e.preventDefault(); if (results[sel]) go(results[sel]); else if (input.value.trim()) runWiki(); }
    });
    dialog.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.preventDefault(); close(); }
      if (e.key === 'Tab') { // simple focus trap
        const f = [...dialog.querySelectorAll('input, button, a[href], select, [tabindex="0"]')].filter((x) => !x.disabled && x.offsetParent !== null);
        if (!f.length) return;
        if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
        else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
      }
    });
    btnWiki.addEventListener('click', runWiki);
    btnLLM.addEventListener('click', () => runLLM(false));
    btnCfg.addEventListener('click', () => runLLM(true));
    btnWeb.addEventListener('click', (e) => { if (!input.value.trim()) { e.preventDefault(); input.focus(); } });
    ui = { overlay, input, list, ext, btnWeb, body: dialog.querySelector('.search-body') };
  }

  function suggestions() {
    ui.list.innerHTML = '';
    const li = h('li.search-hint');
    li.appendChild(h('span.muted', 'Try: '));
    for (const s of ['Bellman equation', 'deadly triad', 'GAE', 'clipped objective', 'Thompson sampling', 'reward hacking', 'DPO', 'MCTS']) {
      const b = h('button.chip', { type: 'button' }, s);
      b.addEventListener('click', () => { ui.input.value = s; runLocal(); ui.input.focus(); });
      li.appendChild(b);
    }
    ui.list.appendChild(li);
  }

  function runLocal() {
    const q = ui.input.value.trim();
    ui.btnWeb.href = q ? 'https://duckduckgo.com/?q=' + encodeURIComponent(withContext(q)) : '#';
    ui.btnWeb.title = q ? 'Opens DuckDuckGo for “' + withContext(q) + '” in a new tab' : '';
    if (!q) { results = []; suggestions(); return; }
    results = search(q);
    sel = 0;
    ui.list.innerHTML = '';
    if (!results.length) {
      ui.list.appendChild(h('li.search-empty', 'Nothing in the playground matches “' + q + '”. Try Wikipedia, a web search or the LLM below.'));
      return;
    }
    results.forEach((r, i) => {
      const e = r.e;
      const li = h('li.search-item', { role: 'option', id: 'sr-' + i, 'aria-selected': i === sel ? 'true' : 'false' },
        h('div.search-item-head',
          h('span.search-kind.k-' + e.kind.replace(/\W/g, ''), e.kind),
          h('span.search-item-title', e.title),
          h('span.search-item-ch', (e.num !== undefined ? 'Ch. ' + e.num + ' · ' : '') + e.chTitle)),
        h('div.search-item-snippet', { html: snippet(e.text || e.title, q) }));
      li.addEventListener('click', () => go(r));
      li.addEventListener('mousemove', () => { if (sel !== i) { sel = i; highlightSel(); } });
      ui.list.appendChild(li);
    });
    highlightSel();
  }
  function highlightSel() {
    [...ui.list.querySelectorAll('.search-item')].forEach((li, i) => {
      li.classList.toggle('selected', i === sel);
      li.setAttribute('aria-selected', i === sel ? 'true' : 'false');
      if (i === sel) { li.scrollIntoView({ block: 'nearest' }); ui.input.setAttribute('aria-activedescendant', li.id); }
    });
  }
  function move(d) { if (!results.length) return; sel = (sel + d + results.length) % results.length; highlightSel(); }
  function go(r) { close(); RL.navigate(r.e.ch, { sel: r.e.sel, i: r.e.i }); }

  function sourceNote(html) { return h('div.search-source', { html }); }
  function showExt() { if (ui && ui.body) ui.body.scrollTop = 0; }
  function abortPending() { if (ctrl) ctrl.abort(); ctrl = new AbortController(); return ctrl.signal; }

  async function runWiki() {
    const q = ui.input.value.trim();
    if (!q) { ui.input.focus(); return; }
    const signal = abortPending();
    ui.ext.innerHTML = '';
    ui.ext.appendChild(h('div.muted.small', 'Searching Wikipedia…')); showExt();
    try {
      const r = await wikipedia(q, signal);
      ui.ext.innerHTML = '';
      if (!r.top) { ui.ext.appendChild(h('div.muted.small', 'No Wikipedia article found for “' + r.used + '”.')); return; }
      const card = h('div.ext-card',
        h('div.ext-title', h('a', { href: r.top.url, target: '_blank', rel: 'noopener noreferrer' }, r.top.title + ' ↗')),
        h('p.ext-text', r.top.extract));
      if (r.hits.length) {
        const ul = h('ul.ext-more');
        for (const x of r.hits) ul.appendChild(h('li', h('a', { href: x.url, target: '_blank', rel: 'noopener noreferrer' }, x.title), h('span.muted', ' — ' + x.snippet.slice(0, 110) + '…')));
        card.appendChild(h('div.small.muted', 'Other matches:'));
        card.appendChild(ul);
      }
      card.appendChild(sourceNote(`Source: <b>Wikipedia</b> (en.wikipedia.org), searched for “${esc(r.used)}”. Text licensed CC BY-SA 4.0. Wikipedia summaries are general — the playground sections above are tailored to this course.`));
      ui.ext.appendChild(card); showExt();
    } catch (e) {
      if (e.name === 'AbortError') return;
      ui.ext.innerHTML = '';
      ui.ext.appendChild(h('div.callout.warn.small', 'Could not reach Wikipedia (' + e.message + '). Are you offline?'));
    }
  }

  function settingsForm(onSaved) {
    const cur = loadLLM() || { preset: 'ollama', ...LLM_PRESETS.ollama, key: '' };
    const f = h('form.llm-form');
    const preset = h('select', { 'aria-label': 'Provider' });
    for (const k in LLM_PRESETS) preset.appendChild(h('option', { value: k }, LLM_PRESETS[k].label));
    preset.value = cur.preset || 'custom';
    const url = h('input', { type: 'url', value: cur.url || '', placeholder: 'https://…/v1/chat/completions', required: true });
    const model = h('input', { type: 'text', value: cur.model || '', placeholder: 'model id, e.g. llama3.2', required: true });
    const key = h('input', { type: 'password', value: cur.key || '', placeholder: 'API key (optional for local models)', autocomplete: 'off' });
    const help = h('div.small.muted');
    const setHelp = () => { help.textContent = LLM_PRESETS[preset.value].help; };
    preset.addEventListener('change', () => { const p = LLM_PRESETS[preset.value]; url.value = p.url; model.value = p.model; setHelp(); });
    setHelp();
    const row = (label, el) => h('label.llm-row', h('span', label), el);
    f.append(
      h('div.ext-title', 'LLM endpoint'),
      row('Provider', preset), row('Endpoint URL', url), row('Model', model), row('API key', key), help,
      h('div.small.muted', 'Settings, including the key, are stored only in this browser (localStorage) and sent only to the endpoint above. Don’t enter a key on a shared computer.'));
    const btns = h('div.btn-row');
    const save = h('button.btn.small.primary', { type: 'submit' }, 'Save & ask');
    const clear = h('button.btn.small', { type: 'button' }, 'Forget settings');
    btns.append(save, clear); f.appendChild(btns);
    clear.addEventListener('click', () => { saveLLM(null); key.value = ''; help.textContent = 'Settings removed from this browser.'; });
    f.addEventListener('submit', (e) => {
      e.preventDefault();
      saveLLM({ preset: preset.value, url: url.value.trim(), model: model.value.trim(), key: key.value.trim() });
      onSaved();
    });
    return f;
  }

  async function runLLM(forceSettings) {
    const cfg = loadLLM();
    if (forceSettings || !cfg || !cfg.url || !cfg.model) {
      abortPending();
      ui.ext.innerHTML = '';
      ui.ext.appendChild(settingsForm(() => runLLM(false))); showExt();
      return;
    }
    const q = ui.input.value.trim();
    if (!q) { ui.input.focus(); return; }
    const signal = abortPending();
    const ctxResults = results.length ? results : search(q);
    let host = cfg.url; try { host = new URL(cfg.url).host; } catch (e) { /* keep raw */ }
    ui.ext.innerHTML = '';
    ui.ext.appendChild(h('div.muted.small', `Asking ${cfg.model} via ${host}…`)); showExt();
    try {
      const r = await askLLM(q, cfg, ctxResults, signal);
      ui.ext.innerHTML = '';
      const body = h('div.ext-text.llm-answer'); body.textContent = r.text; // plain text: never inject model output as HTML
      RL.util.renderMath(body);
      const card = h('div.ext-card', h('div.ext-title', 'LLM answer'), body);
      const used = ctxResults.slice(0, 4);
      const note = sourceNote(`Source: <b>AI-generated</b> by <b>${esc(r.model)}</b> via <b>${esc(host)}</b> — may be inaccurate; verify against the linked sections.` + (used.length ? ' Context given to the model:' : ''));
      if (used.length) {
        const ul = h('ul.ext-more');
        for (const u of used) { const a = h('a', { href: '#/' + u.e.ch }, u.e.chTitle + ' — ' + u.e.title); a.addEventListener('click', (ev) => { ev.preventDefault(); go(u); }); ul.appendChild(h('li', a)); }
        note.appendChild(ul);
      }
      card.appendChild(note);
      ui.ext.appendChild(card); showExt();
    } catch (e) {
      if (e.name === 'AbortError') return;
      ui.ext.innerHTML = '';
      const hint = /Failed to fetch|NetworkError|Load failed/i.test(e.message)
        ? ' The endpoint is unreachable or blocks cross-origin requests (CORS). For Ollama, set OLLAMA_ORIGINS; hosted APIs need a valid key.' : '';
      ui.ext.appendChild(h('div.callout.warn.small', `LLM request to ${host} failed: ${e.message}.${hint}`));
      const b = h('button.btn.small', { type: 'button' }, 'Change LLM settings'); b.addEventListener('click', () => runLLM(true));
      ui.ext.appendChild(b);
    }
  }

  function open(prefill) {
    if (!ui) buildUI();
    lastFocus = document.activeElement;
    ui.overlay.hidden = false;
    document.body.classList.add('search-open');
    if (typeof prefill === 'string') ui.input.value = prefill;
    ui.input.focus(); ui.input.select();
    if (!INDEX) { ui.list.innerHTML = ''; ui.list.appendChild(h('li.search-empty', 'Indexing chapters…')); setTimeout(runLocal, 20); }
    else runLocal();
  }
  function close() {
    if (!ui || ui.overlay.hidden) return;
    if (ctrl) ctrl.abort();
    ui.overlay.hidden = true;
    document.body.classList.remove('search-open');
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    const typing = /INPUT|TEXTAREA|SELECT/.test(tag) || (e.target && e.target.isContentEditable);
    if (((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') || (e.key === '/' && !typing)) { e.preventDefault(); open(); }
  });

  RL.search = { open, close, search, buildIndex, texToText };
})(window.RL);
