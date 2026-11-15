/* App shell: sidebar navigation, hash router, chapter lifecycle, theme handling, page helpers. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { h, renderMath } = RL.util;

  RL.chapters = RL.chapters || [];
  RL._theme = new Set();
  RL.onTheme = (fn) => RL._theme.add(fn);

  /* ---------- page helpers used by chapters ---------- */
  RL.page = {
    head(root, { num, title, lede, tags = [] }) {
      const el = h('header.chapter-head',
        h('div.kicker', num === undefined ? 'Overview' : 'Chapter ' + num),
        h('h1', title),
        lede ? h('p.lede', { html: lede }) : null,
        tags.length ? h('div.tags', tags.map((t) => {
          const [txt, cls] = Array.isArray(t) ? t : [t, ''];
          return h('span.tag' + (cls ? '.' + cls : ''), txt);
        })) : null);
      root.appendChild(el);
      renderMath(el);
      return el;
    },
    prose(root, html, cls = 'concepts') {
      const el = h('div.' + cls, { html });
      root.appendChild(el);
      renderMath(el);
      return el;
    },
    lab(root, { two = false, title = 'Lab' } = {}) {
      const controls = h('div.lab-controls'), viz = h('div.lab-viz'), side = h('div.lab-side');
      const el = h('section.lab' + (two ? '.two' : ''), { 'aria-label': title }, controls, viz, two ? null : side);
      root.appendChild(el);
      return { el, controls, viz, side };
    },
    widget(root, title, sub) {
      const el = h('div.widget', h('div.widget-title', title), sub ? h('div.widget-sub', { html: sub }) : null);
      root.appendChild(el);
      renderMath(el);
      return el;
    },
  };

  /* ---------- theme ---------- */
  const THEMES = ['auto', 'light', 'dark'];
  let theme = 'auto';
  try { theme = localStorage.getItem('rl-theme') || 'auto'; } catch (e) { /* storage unavailable */ }
  const applyTheme = () => {
    const root = document.documentElement;
    if (theme === 'auto') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', theme);
    const dark = theme === 'dark' || (theme === 'auto' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
    root.setAttribute('data-theme-resolved', dark ? 'dark' : 'light');
    RL.util.invalidatePalette();
    RL._theme.forEach((fn) => { try { fn(); } catch (e) { console.warn(e); } });
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = 'Theme: ' + theme;
  };
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

  /* ---------- router ---------- */
  let current = null, pendingAnchor = null;
  const main = () => document.getElementById('main');

  /** Scroll to (and briefly highlight) element #i matching `sel` in the current chapter; opens collapsed boxes. */
  function applyAnchor() {
    const a = pendingAnchor; pendingAnchor = null;
    if (!a) return;
    const root = document.querySelector('#main article.chapter');
    const el = root && root.querySelectorAll(a.sel)[a.i];
    if (!el) return;
    if (el.matches('details')) el.open = true;
    for (let p = el.parentElement; p; p = p.parentElement) if (p.tagName === 'DETAILS') p.open = true;
    el.scrollIntoView({ block: 'start', behavior: 'smooth' });
    el.classList.remove('search-hit'); void el.offsetWidth; el.classList.add('search-hit');
    setTimeout(() => el.classList.remove('search-hit'), 2600);
  }
  /** Go to a chapter, optionally to a specific element in it: RL.navigate('ppo', {sel: '.concepts h2', i: 2}). */
  RL.navigate = (id, anchor) => {
    pendingAnchor = anchor || null;
    if (current && current.id === id) requestAnimationFrame(applyAnchor);
    else location.hash = '#/' + id;
  };

  function render() {
    const id = (location.hash.replace(/^#\/?/, '') || RL.chapters[0].id).split('?')[0];
    const ch = RL.chapters.find((c) => c.id === id) || RL.chapters[0];
    if (current && current.destroy) { try { current.destroy(); } catch (e) { console.warn(e); } }
    RL._theme.clear();
    if (RL.ui && RL.ui.closeAllInfo) RL.ui.closeAllInfo();
    if (RL.ui && RL.ui.closeCoach) RL.ui.closeCoach();
    const m = main(); m.innerHTML = '';
    const root = h('article.chapter');
    m.appendChild(root);
    let destroy = null;
    try { destroy = ch.mount(root); } catch (e) {
      console.error(e);
      root.appendChild(h('div.callout.warn', h('strong', 'Error: '), 'this chapter failed to load: ' + e.message));
    }
    if (RL.resources) { try { RL.resources.decorate(root, ch.id); } catch (e) { console.warn(e); } }
    current = { id: ch.id, destroy };
    // prev / next
    const i = RL.chapters.indexOf(ch);
    const nav = h('nav.chapter-nav', { 'aria-label': 'Chapter navigation' });
    const prev = RL.chapters[i - 1], next = RL.chapters[i + 1];
    nav.appendChild(prev ? h('a', { href: '#/' + prev.id }, h('span.dir', '← Previous'), prev.title) : h('span'));
    nav.appendChild(next ? h('a', { href: '#/' + next.id, style: { textAlign: 'right' } }, h('span.dir', 'Next →'), next.title) : h('span'));
    m.appendChild(nav);
    document.querySelectorAll('.nav a').forEach((a) => a.classList.toggle('active', a.dataset.id === ch.id));
    document.title = ch.title + ' · RL Playground';
    document.getElementById('sidebar').classList.remove('open');
    window.scrollTo(0, 0);
    if (RL.fitMath) requestAnimationFrame(() => RL.fitMath(root));
    if (pendingAnchor) requestAnimationFrame(applyAnchor);
  }

  function buildSidebar() {
    const sb = document.getElementById('sidebar');
    sb.innerHTML = '';
    sb.appendChild(h('a.brand', { href: '#/' + RL.chapters[0].id, html: `
      <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">
        <rect x="1" y="1" width="32" height="32" rx="8" style="fill:var(--accent)"/>
        <circle cx="11" cy="17" r="4.2" fill="#fff"/>
        <path d="M16 12.5 C21 8, 26 10, 26 15" stroke="#fff" stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <path d="M24 13.5 l2 2 l2 -2.6" stroke="#fff" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M18 21.5 C13 26, 8 24.5, 8 21.5" stroke="#fff" stroke-width="2.2" fill="none" stroke-linecap="round" opacity=".75"/>
      </svg>
      <span><span class="brand-name">RL Playground</span><br><span class="brand-sub">Reinforcement learning, interactively</span></span>` }));
    const sbtn = h('button.search-trigger', { type: 'button', 'aria-label': 'Search (shortcut: / or Ctrl+K)' },
      h('span', { 'aria-hidden': 'true' }, '⌕'), h('span.st-label', 'Search…'), h('kbd', '/'));
    sbtn.addEventListener('click', () => RL.search && RL.search.open());
    sb.appendChild(sbtn);
    const ul = h('ul.nav');
    let part = null;
    for (const ch of RL.chapters) {
      if (ch.part !== part) { part = ch.part; ul.appendChild(h('li.nav-part', part)); }
      ul.appendChild(h('li', h('a', { href: '#/' + ch.id, 'data-id': ch.id }, h('span.num', ch.num === undefined ? '•' : String(ch.num)), h('span', ch.title))));
    }
    sb.appendChild(h('nav', { 'aria-label': 'Chapters' }, ul));
    const foot = h('div.side-foot');
    const tb = h('button.btn.small#theme-btn', { type: 'button' }, 'Theme: ' + theme);
    tb.addEventListener('click', () => {
      theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
      try { localStorage.setItem('rl-theme', theme); } catch (e) { /* ignore */ }
      applyTheme();
    });
    const gh = h('a.btn.small.repo-link', {
      href: 'https://github.com/DominikGithub/RL-playground', target: '_blank', rel: 'noopener noreferrer',
      title: 'Source code on GitHub',
      html: '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg><span>GitHub</span>',
    });
    foot.appendChild(h('div.row', tb, gh));
    foot.appendChild(h('div', { html: 'Expandable boxes: <b style="color:var(--good)">Basics</b> refreshers, <b style="color:var(--accent)">Math</b> derivations, <b style="color:var(--neg)">Practice</b> tips, <b>History</b> notes.' }));
    sb.appendChild(foot);
  }

  function boot() {
    buildSidebar();
    const mb = document.querySelector('.mobile-bar button');
    if (mb) mb.addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));
    const ms = document.querySelector('.mobile-bar .mobile-search');
    if (ms) ms.addEventListener('click', () => RL.search && RL.search.open());
    applyTheme();
    window.addEventListener('hashchange', render);
    render();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})(window.RL);
