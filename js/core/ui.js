/* UI control builders: sliders (linear/log), selects, toggles, buttons, stats, experiment cards. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { h, fmt } = RL.util;

  const helpIcon = (text) => (text ? h('span.help', { 'data-tip': text, tabindex: 0, 'aria-label': text }, '?') : null);

  const ui = {};

  ui.group = (parent, title, cls = '') => {
    const g = h('div.ctl-group' + (cls ? '.' + cls : ''));
    if (title) g.appendChild(h('div.ctl-group-title', title));
    parent.appendChild(g);
    return g;
  };

  /** Slider. opts: {label, min, max, step, value, log, fmt, help, onInput} */
  ui.slider = (parent, o) => {
    const f = o.fmt || ((v) => fmt(v, o.step && o.step >= 1 ? 0 : 2));
    const toPos = (v) => (o.log ? ((Math.log(v) - Math.log(o.min)) / (Math.log(o.max) - Math.log(o.min))) * 1000 : v);
    const fromPos = (p) => {
      if (!o.log) return +p;
      const v = Math.exp(Math.log(o.min) + (p / 1000) * (Math.log(o.max) - Math.log(o.min)));
      const mag = Math.pow(10, Math.floor(Math.log10(v)) - 1);
      return Math.round(v / mag) * mag; // two significant digits
    };
    const input = h('input', {
      type: 'range', min: o.log ? 0 : o.min, max: o.log ? 1000 : o.max, step: o.log ? 1 : o.step || 0.01,
      value: toPos(o.value), 'aria-label': o.label,
    });
    const val = h('span.ctl-val', f(o.value));
    const row = h('label.ctl.ctl-slider', h('span.ctl-label', o.label, helpIcon(o.help)), val, input);
    parent.appendChild(row);
    const api = {
      el: row, value: o.value,
      set(v, silent = false) { api.value = v; input.value = toPos(v); val.textContent = f(v); if (!silent && o.onInput) o.onInput(v); },
      show(b) { row.style.display = b ? '' : 'none'; },
    };
    input.addEventListener('input', () => { api.value = fromPos(input.value); val.textContent = f(api.value); if (o.onInput) o.onInput(api.value); });
    return api;
  };

  /** Select. opts: {label, options:[{value,label}], value, help, onChange} */
  ui.select = (parent, o) => {
    const sel = h('select', { 'aria-label': o.label });
    for (const op of o.options) {
      if (op.group) { const g = h('optgroup', { label: op.group }); for (const x of op.options) g.appendChild(h('option', { value: x.value }, x.label)); sel.appendChild(g); }
      else sel.appendChild(h('option', { value: op.value }, op.label));
    }
    sel.value = o.value;
    const row = h('label.ctl.ctl-select', h('span.ctl-label', o.label, helpIcon(o.help)), sel);
    parent.appendChild(row);
    const api = {
      el: row, value: o.value, input: sel,
      set(v, silent = false) { sel.value = v; api.value = v; if (!silent && o.onChange) o.onChange(v); },
      show(b) { row.style.display = b ? '' : 'none'; },
    };
    sel.addEventListener('change', () => { api.value = sel.value; if (o.onChange) o.onChange(sel.value); });
    return api;
  };

  /** Toggle switch. opts: {label, value, help, onChange} */
  ui.toggle = (parent, o) => {
    const cb = h('input', { type: 'checkbox', 'aria-label': o.label });
    cb.checked = !!o.value;
    const row = h('label.ctl.ctl-toggle', cb, h('span.switch'), h('span.ctl-label', o.label, helpIcon(o.help)));
    parent.appendChild(row);
    const api = {
      el: row, value: !!o.value,
      set(v, silent = false) { cb.checked = !!v; api.value = !!v; if (!silent && o.onChange) o.onChange(!!v); },
      show(b) { row.style.display = b ? '' : 'none'; },
    };
    cb.addEventListener('change', () => { api.value = cb.checked; if (o.onChange) o.onChange(cb.checked); });
    return api;
  };

  /** Segmented radio buttons. opts: {label, options:[{value,label,title}], value, onChange} */
  ui.segmented = (parent, o) => {
    const wrap = h('div.segmented', { role: 'radiogroup', 'aria-label': o.label || '' });
    const btns = {};
    const api = {
      value: o.value,
      set(v, silent = false) {
        api.value = v;
        for (const k in btns) btns[k].classList.toggle('active', k === String(v));
        if (!silent && o.onChange) o.onChange(v);
      },
    };
    for (const op of o.options) {
      const b = h('button', { type: 'button', title: op.title || '', role: 'radio' }, op.label);
      b.addEventListener('click', () => api.set(op.value));
      btns[op.value] = b; wrap.appendChild(b);
    }
    const row = o.label ? h('div.ctl', h('span.ctl-label', o.label, helpIcon(o.help)), wrap) : wrap;
    parent.appendChild(row);
    api.el = row;
    api.set(o.value, true);
    return api;
  };

  ui.button = (parent, label, onClick, cls = '') => {
    const b = h('button.btn' + (cls ? '.' + cls.split(' ').join('.') : ''), { type: 'button' }, label);
    b.addEventListener('click', onClick);
    parent.appendChild(b);
    return b;
  };

  /** Transport bar: play/pause, step, reset. */
  ui.transport = (parent, { onPlay, onStep, onReset, stepLabel = 'Step', playLabel = 'Run' }) => {
    const bar = h('div.transport');
    const reset = ui.button(bar, '↺', onReset, 'icon');
    reset.title = 'Reset'; reset.setAttribute('aria-label', 'Reset');
    const play = ui.button(bar, '▶ ' + playLabel, () => api.setPlaying(!api.playing), 'primary');
    const step = onStep ? ui.button(bar, stepLabel, onStep) : null;
    parent.appendChild(bar);
    const api = {
      el: bar, playing: false, playBtn: play, stepBtn: step,
      setPlaying(p, silent = false) {
        api.playing = p;
        play.textContent = p ? '❚❚ Pause' : '▶ ' + playLabel;
        play.classList.toggle('playing', p);
        if (!silent && onPlay) onPlay(p);
      },
    };
    return api;
  };

  ui.stats = (parent, items) => {
    const box = h('div.stats');
    const api = {};
    for (const [key, label, help] of items) {
      const v = h('span.stat-val', '–');
      box.appendChild(h('div.stat', h('span.stat-label', label, helpIcon(help)), v));
      api[key] = (x) => { v.textContent = typeof x === 'number' ? (Number.isInteger(x) ? (Math.abs(x) >= 1e5 ? RL.util.compact(x) : String(x).replace('-', '−')) : fmt(x, Math.abs(x) >= 100 ? 0 : 2)) : x; };
    }
    parent.appendChild(box);
    return api;
  };

  /** Experiment cards ("Try this"). items: [{title, text, apply}] */
  ui.experiments = (parent, items, title = 'Guided experiments') => {
    const wrap = h('div.experiments', h('div.exp-head', h('span.exp-title', title), h('span.exp-sub', 'Each card configures the lab above. Predict the outcome before you press it.')));
    const list = h('div.exp-list');
    items.forEach((it, i) => {
      const card = h('div.exp-card', h('div.exp-num', String(i + 1)), h('div.exp-body', h('div.exp-card-title', it.title), h('div.exp-text', { html: it.text })));
      const btn = ui.button(card, 'Set up', () => {
        list.querySelectorAll('.exp-card').forEach((c) => c.classList.remove('active'));
        card.classList.add('active');
        it.apply();
        if (it.scrollTo !== false && it.labEl) it.labEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 'small');
      btn.setAttribute('aria-label', 'Set up experiment: ' + it.title);
      list.appendChild(card);
    });
    wrap.appendChild(list);
    parent.appendChild(wrap);
    RL.util.renderMath(wrap);
    return wrap;
  };

  /* ---------- ⓘ legend popover ---------- */
  /** Swatch snippets (HTML strings) for legends. Colours may be CSS variables, e.g. 'var(--pos)'. */
  ui.sw = {
    box: (c) => `<span class="lgs box" style="background:${c}"></span>`,
    outline: (c) => `<span class="lgs outline" style="border-color:${c}"></span>`,
    line: (c) => `<span class="lgs line" style="border-color:${c}"></span>`,
    dashed: (c) => `<span class="lgs line dashed" style="border-color:${c}"></span>`,
    faint: (c) => `<span class="lgs line faint" style="border-color:${c}"></span>`,
    vline: (c) => `<span class="lgs vline" style="border-color:${c}"></span>`,
    dot: (c) => `<span class="lgs dot" style="background:${c}"></span>`,
    ring: (c) => `<span class="lgs ring" style="border-color:${c}"></span>`,
    tick: (c) => `<span class="lgs tick" style="background:${c}"></span>`,
    band: (c) => `<span class="lgs band" style="background:${c}"></span>`,
    grad: () => '<span class="lgs grad"></span>',
    seq: () => '<span class="lgs seq"></span>',
    text: (t, c = 'var(--text)') => `<span class="lgs txt" style="color:${c}">${t}</span>`,
  };
  /** Legend list from [swatchHtml, explanationHtml] rows, optionally with an intro paragraph. */
  ui.lg = (rows, intro = '') => (intro ? `<p class="lg-intro">${intro}</p>` : '') +
    '<ul class="lg-list">' + rows.map(([s, t]) => `<li><span class="lg-sw">${s}</span><span>${t}</span></li>`).join('') + '</ul>';

  const openPops = new Set();
  ui.closeAllInfo = () => openPops.forEach((f) => f(true));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') ui.closeAllInfo(); });

  /** Circled-i button that shows `html` on hover/focus; click (or tap) pins it open. */
  ui.info = (parent, html, label = 'Legend') => {
    const btn = h('button.info-btn', { type: 'button', 'aria-label': 'Legend: ' + label, 'aria-expanded': 'false', title: '' }, 'i');
    parent.appendChild(btn);
    let pop = null, pinned = false, hideT = 0;
    const position = () => {
      if (!pop) return;
      const r = btn.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
      const left = Math.max(8, Math.min(window.innerWidth - pw - 8, r.left - 16));
      let top = r.bottom + 8;
      if (top + ph > window.innerHeight - 8 && r.top - ph - 8 > 8) top = r.top - ph - 8;
      pop.style.left = left + 'px'; pop.style.top = Math.max(8, top) + 'px';
    };
    const onScroll = () => { if (pinned) position(); else close(true); };
    const outside = (e) => { if (pop && !pop.contains(e.target) && e.target !== btn) close(true); };
    function close(force) {
      if (pinned && !force) return;
      pinned = false; clearTimeout(hideT);
      if (pop) { pop.remove(); pop = null; }
      btn.setAttribute('aria-expanded', 'false'); btn.classList.remove('pinned');
      window.removeEventListener('scroll', onScroll, true);
      document.removeEventListener('pointerdown', outside, true);
      openPops.delete(close);
    }
    const open = () => {
      clearTimeout(hideT);
      if (!pop) {
        openPops.forEach((f) => { if (f !== close) f(false); });
        pop = h('div.info-pop', { role: 'dialog', 'aria-label': label }, h('div.info-pop-title', label), h('div.info-pop-body', { html }));
        document.body.appendChild(pop);
        RL.util.renderMath(pop);
        pop.addEventListener('mouseenter', () => clearTimeout(hideT));
        pop.addEventListener('mouseleave', () => { if (!pinned) hideT = setTimeout(() => close(false), 200); });
        window.addEventListener('scroll', onScroll, true);
        openPops.add(close);
      }
      btn.setAttribute('aria-expanded', 'true');
      position();
    };
    btn.addEventListener('mouseenter', open);
    btn.addEventListener('mouseleave', () => { if (!pinned) hideT = setTimeout(() => close(false), 200); });
    btn.addEventListener('focus', open);
    btn.addEventListener('blur', () => { if (!pinned) close(false); });
    btn.addEventListener('click', (e) => {
      e.preventDefault(); e.stopPropagation();
      if (pinned) { close(true); return; }
      pinned = true; btn.classList.add('pinned'); open();
      document.addEventListener('pointerdown', outside, true);
    });
    return btn;
  };

  /* ---------- Tutor: rule-based "what just happened?" chat bubble ---------- */
  const TUTOR_KEY = 'rl-tutor-muted';
  const TONES = { good: 'improving', done: 'converged', bad: 'getting worse', unclear: 'unclear', info: 'note' };
  ui.tutorEnabled = () => { try { return localStorage.getItem(TUTOR_KEY) !== '1'; } catch (e) { return true; } };
  ui.setTutorEnabled = (on) => { try { localStorage.setItem(TUTOR_KEY, on ? '0' : '1'); } catch (e) { /* storage unavailable */ } };
  let activeCoach = null;
  ui.closeCoach = () => { if (activeCoach) activeCoach.hide(); };

  /**
   * Tutor chat window (bottom-right). Messages come from hand-written rules in each lab — nothing is generated.
   * say({tone, title, text, basis}); tone ∈ good | done | bad | unclear | info.
   * The window stays open until the user closes it; closing leaves a small button to reopen it with the
   * full message history. hide() removes window, button and history (used on reset / leaving the page).
   */
  ui.coach = () => {
    const MAX = 8;
    let win = null, thread = null, launcher = null, typingT = 0, unread = 0;
    const history = []; // rendered message elements, oldest first
    const AVATAR = `<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="16" style="fill:var(--accent)"/>
      <path d="M5 13 L16 8 L27 13 L16 18 Z" fill="#fff"/><path d="M10 15.6 V20 C10 22.6 22 22.6 22 20 V15.6 L16 18.4 Z" fill="#fff" opacity=".85"/>
      <path d="M25.5 13.4 V19.5" stroke="#fff" stroke-width="1.4" stroke-linecap="round"/></svg>`;

    function openWindow() {
      if (win) return;
      removeLauncher();
      const close = h('button.coach-x', { type: 'button', 'aria-label': 'Close tutor (you can reopen it)', title: 'Close — reopen with the tutor button' }, '×');
      thread = h('div.coach-body');
      win = h('div.coach.coach-in', { role: 'log', 'aria-live': 'polite', 'aria-label': 'Tutor' },
        h('div.coach-head', h('span.coach-avatar', { html: AVATAR }), h('span.coach-who', h('b', 'Tutor')), close),
        thread);
      close.addEventListener('click', () => minimize());
      history.forEach((m) => thread.appendChild(m));
      document.body.appendChild(win);
      markOld();
      thread.scrollTop = thread.scrollHeight;
    }
    function markOld() { history.forEach((m, i) => m.classList.toggle('coach-old', i < history.length - 1)); }
    function removeLauncher() { if (launcher) { launcher.remove(); launcher = null; } }
    function showLauncher() {
      removeLauncher();
      if (!history.length) return;
      launcher = h('button.coach-launcher', { type: 'button', 'aria-label': 'Open tutor', title: 'Open tutor' },
        h('span.coach-avatar', { html: AVATAR }), h('span', 'Tutor'), unread ? h('span.coach-unread', String(unread)) : null);
      launcher.addEventListener('click', () => { unread = 0; openWindow(); });
      document.body.appendChild(launcher);
    }
    function minimize() {
      clearTimeout(typingT);
      const typing = win && win.querySelector('.coach-typing');
      if (typing) typing.remove();
      if (win) { const old = win; win = null; thread = null; old.classList.add('coach-out'); setTimeout(() => old.remove(), 220); }
      showLauncher();
    }

    const api = {
      say({ tone = 'info', title = '', text = '', basis = '' }) {
        if (!ui.tutorEnabled()) return;
        if (activeCoach && activeCoach !== api) activeCoach.hide();
        activeCoach = api;
        openWindow();
        clearTimeout(typingT);
        const old = thread.querySelector('.coach-typing'); if (old) old.remove();
        const typing = h('div.coach-typing', { 'aria-label': 'typing' }, h('i'), h('i'), h('i'));
        thread.appendChild(typing);
        thread.scrollTop = thread.scrollHeight;
        typingT = setTimeout(() => {
          typing.remove();
          const msg = h('div.coach-msg',
            h('div.coach-msg-head', h('span.coach-tone.t-' + tone, TONES[tone] || tone), title ? h('b', title) : null),
            h('div.coach-text', { html: text }),
            basis ? h('div.coach-basis', { html: 'Based on: ' + basis }) : null);
          history.push(msg);
          while (history.length > MAX) history.shift().remove();
          if (thread) {
            thread.appendChild(msg);
            RL.util.renderMath(msg);
            markOld();
            msg.scrollIntoView({ block: 'nearest' });
            thread.scrollTop = thread.scrollHeight;
          } else { unread++; showLauncher(); }
        }, 380);
      },
      /** Remove window, reopen button and history. */
      hide() {
        clearTimeout(typingT);
        if (win) { win.remove(); win = null; thread = null; }
        removeLauncher();
        history.length = 0; unread = 0;
        if (activeCoach === api) activeCoach = null;
      },
    };
    return api;
  };

  /** Toggle in a lab's controls that switches the tutor on/off (shared setting). */
  ui.tutorToggle = (parent, coach) => ui.toggle(parent, {
    label: 'Tutor hints on manual steps', value: ui.tutorEnabled(),
    help: 'After each manual step a short, rule-based explanation pops up: what changed, whether it is an improvement, and which numbers that judgement is based on. Not shown while running continuously.',
    onChange: (v) => { ui.setTutorEnabled(v); if (!v) coach.hide(); },
  });

  ui.helpIcon = helpIcon;
  RL.ui = ui;
})(window.RL);
