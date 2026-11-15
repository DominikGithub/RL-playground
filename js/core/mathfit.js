/* Make display formulas fit without horizontal scrolling.
   Strategy per formula that is wider than its column, in order of preference:
   1. widen it (or the box it sits in) into free space beside the text column — keeps the formula on one line;
   2. re-render it as line-breakable math (KaTeX breaks after =, ≤, +, − at the top level), centred;
   3. as a last resort, scale the font down slightly (≥ 85%); only if that still fails does it scroll. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';

  const BOX_SEL = 'details.deep, .callout, .widget';
  const MIN_SCALE = 0.85;

  const texOf = (el) => {
    if (el.dataset.tex) return el.dataset.tex;
    const a = el.querySelector('annotation[encoding="application/x-tex"]');
    return a ? a.textContent : null;
  };
  const over = (el) => el.scrollWidth - el.clientWidth;

  function reset(root) {
    root.querySelectorAll('[data-fit-box]').forEach((b) => { b.style.width = ''; b.style.marginLeft = ''; b.removeAttribute('data-fit-box'); });
    root.querySelectorAll('.katex-display').forEach((el) => {
      el.style.marginLeft = ''; el.style.marginRight = ''; el.style.fontSize = '';
      if (el.classList.contains('katex-wrapped') && window.katex) {
        window.katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false });
        el.classList.remove('katex-wrapped');
      }
    });
  }

  function limits() {
    const main = document.getElementById('main');
    if (!main) return null;
    const r = main.getBoundingClientRect(), cs = getComputedStyle(main);
    return { left: r.left + parseFloat(cs.paddingLeft), right: r.right - parseFloat(cs.paddingRight) };
  }

  /** Widen `el` by `need` px using free space to the right first, then to the left. Returns true if it fits. */
  function widen(el, need, lim, isBox) {
    const r = el.getBoundingClientRect();
    const right = Math.max(0, Math.floor(lim.right - r.right)), left = Math.max(0, Math.floor(r.left - lim.left));
    if (need > right + left) return false;
    const useR = Math.min(need, right), useL = need - useR;
    if (isBox) {
      el.style.width = (r.width + need) + 'px';
      if (useL) el.style.marginLeft = -useL + 'px';
      el.setAttribute('data-fit-box', '');
    } else {
      if (useR) el.style.marginRight = -useR + 'px';
      if (useL) el.style.marginLeft = -useL + 'px';
    }
    return true;
  }

  function fit(root) {
    if (!root || !window.katex) return;
    reset(root);
    const lim = limits();
    if (!lim) return;
    const displays = [...root.querySelectorAll('.katex-display')].filter((el) => el.offsetParent !== null);
    // 1. widen boxes (by their widest formula) or bare formulas into free space
    const boxNeed = new Map();
    for (const el of displays) {
      const need = over(el);
      if (need <= 1) continue;
      const tex = texOf(el); if (tex) el.dataset.tex = tex;
      const box = el.closest(BOX_SEL);
      if (box && root.contains(box)) boxNeed.set(box, Math.max(boxNeed.get(box) || 0, need));
      else widen(el, need + 2, lim, false);
    }
    for (const [box, need] of boxNeed) widen(box, need + 2, lim, true);
    // 2. wrap, 3. scale
    for (const el of displays) {
      if (over(el) <= 1 || !el.dataset.tex) continue;
      window.katex.render('\\displaystyle ' + el.dataset.tex, el, { displayMode: false, throwOnError: false });
      el.classList.add('katex-wrapped');
      const o = over(el);
      if (o > 1) el.style.fontSize = Math.max(MIN_SCALE, el.clientWidth / el.scrollWidth).toFixed(3) + 'em';
    }
  }

  let timer = 0;
  const current = () => document.querySelector('#main article.chapter');
  const refit = () => { clearTimeout(timer); timer = setTimeout(() => fit(current()), 120); };
  window.addEventListener('resize', refit);
  // formulas inside a <details> have no size until it is opened ('toggle' does not bubble → capture)
  document.addEventListener('toggle', (e) => { if (e.target.matches && e.target.matches('details') && e.target.open) fit(current()); }, true);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(refit);

  RL.fitMath = fit;
  RL.refitMath = refit;
})(window.RL);
