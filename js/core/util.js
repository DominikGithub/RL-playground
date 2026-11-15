/* Core utilities: RNG, math helpers, DOM helpers, colors. No DOM access at load time. */
window.RL = window.RL || {};
window.RL.chapters = window.RL.chapters || [];
(function (RL) {
  'use strict';

  /** Seeded PRNG (mulberry32) with a few convenience samplers. */
  class Rng {
    constructor(seed) {
      this.s = (seed === undefined ? Math.floor(Math.random() * 4294967296) : seed) >>> 0;
      this._spare = null;
    }
    next() {
      let t = (this.s = (this.s + 0x6d2b79f5) | 0);
      t = Math.imul(t ^ (t >>> 15), 1 | t);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    int(n) { return Math.floor(this.next() * n); }
    normal() {
      if (this._spare !== null) { const v = this._spare; this._spare = null; return v; }
      let u = 0, v = 0;
      while (u === 0) u = this.next();
      v = this.next();
      const r = Math.sqrt(-2 * Math.log(u));
      this._spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    }
    choice(arr) { return arr[this.int(arr.length)]; }
  }

  const U = {};
  U.clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  U.lerp = (a, b, t) => a + (b - a) * t;
  U.mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
  U.std = (a) => {
    if (a.length < 2) return 0;
    const m = U.mean(a);
    return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / a.length);
  };

  /** argmax over arr[offset .. offset+len), ties broken uniformly at random. Returns index relative to offset. */
  U.argmaxRandom = (arr, rng, offset = 0, len = arr.length - offset) => {
    let best = -Infinity, count = 0, idx = 0;
    for (let i = 0; i < len; i++) {
      const v = arr[offset + i];
      if (v > best + 1e-12) { best = v; idx = i; count = 1; }
      else if (Math.abs(v - best) <= 1e-12) { count++; if (rng.next() * count < 1) idx = i; }
    }
    return idx;
  };
  U.argmax = (arr) => { let b = 0; for (let i = 1; i < arr.length; i++) if (arr[i] > arr[b]) b = i; return b; };
  U.max = (arr, offset = 0, len = arr.length - offset) => {
    let m = -Infinity; for (let i = 0; i < len; i++) if (arr[offset + i] > m) m = arr[offset + i]; return m;
  };

  U.softmax = (z, temp = 1) => {
    let m = -Infinity; for (const v of z) if (v > m) m = v;
    const e = new Float64Array(z.length); let s = 0;
    for (let i = 0; i < z.length; i++) { e[i] = Math.exp((z[i] - m) / temp); s += e[i]; }
    for (let i = 0; i < z.length; i++) e[i] /= s;
    return e;
  };
  U.sampleCat = (p, rng) => {
    let u = rng.next(), c = 0;
    for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; }
    return p.length - 1;
  };
  U.entropy = (p) => { let h = 0; for (const x of p) if (x > 0) h -= x * Math.log(x); return h; };

  U.compact = (n) => (n >= 1e6 ? +(n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? +(n / 1e3).toFixed(1) + 'k' : String(Math.round(n)));
  U.fmt = (x, d = 2) => {
    if (x === undefined || x === null || Number.isNaN(x)) return '–';
    if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
    const s = Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(1) : x.toFixed(d);
    return s.replace('-', '−');
  };

  /** Tiny hyperscript DOM builder: h('div.cls#id', {attrs}, children...) */
  U.h = (sel, attrs, ...kids) => {
    const m = sel.match(/^([a-z0-9]+)?((?:[.#][\w-]+)*)$/i);
    const el = document.createElement((m && m[1]) || 'div');
    if (m && m[2]) for (const part of m[2].match(/[.#][\w-]+/g)) {
      if (part[0] === '.') el.classList.add(part.slice(1)); else el.id = part.slice(1);
    }
    if (attrs && (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs))) { kids.unshift(attrs); attrs = null; }
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (k === 'html') el.innerHTML = v;
      else if (k === 'text') el.textContent = v;
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (v !== false && v !== null && v !== undefined) el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      el.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  };

  U.cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /** Diverging color map in the spirit of TF Playground: orange (negative) – neutral – blue (positive). */
  const hexToRgb = (hex) => {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  };
  let palCache = null, palKey = '';
  U.palette = () => {
    const key = document.documentElement.getAttribute('data-theme-resolved') || '';
    if (palCache && palKey === key) return palCache;
    palKey = key;
    palCache = {
      neg: hexToRgb(U.cssVar('--neg') || '#e8710a'),
      pos: hexToRgb(U.cssVar('--pos') || '#1a73e8'),
      mid: hexToRgb(U.cssVar('--mid') || '#f1f3f4'),
    };
    return palCache;
  };
  U.invalidatePalette = () => { palCache = null; };
  U.divColor = (v, scale, alpha = 1) => {
    const p = U.palette();
    const t = U.clamp(scale > 0 ? v / scale : 0, -1, 1);
    const tgt = t < 0 ? p.neg : p.pos;
    const a = Math.pow(Math.abs(t), 0.8);
    const c = [0, 1, 2].map((i) => Math.round(p.mid[i] + (tgt[i] - p.mid[i]) * a));
    return alpha === 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
  };

  /** Set up a canvas for crisp HiDPI rendering; returns 2D context scaled to CSS pixels. */
  U.setupCanvas = (canvas, cssW, cssH) => {
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  };

  /** Render KaTeX inside an element if KaTeX is available. */
  U.renderMath = (el) => {
    if (window.RL._noMath) return;
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(el, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
          ],
          throwOnError: false,
        });
      } catch (e) { console.warn(e); }
    }
  };
  U.tex = (src, display = false) => {
    if (window.katex) {
      try { return window.katex.renderToString(src, { throwOnError: false, displayMode: display }); } catch (e) { /* fallthrough */ }
    }
    return src;
  };

  /** A requestAnimationFrame loop that can be started/stopped and auto-destroyed. */
  class Loop {
    constructor(fn) { this.fn = fn; this.running = false; this._id = 0; }
    start() {
      if (this.running) return; this.running = true;
      const tick = () => { if (!this.running) return; this.fn(); this._id = requestAnimationFrame(tick); };
      this._id = requestAnimationFrame(tick);
    }
    stop() { this.running = false; cancelAnimationFrame(this._id); }
  }

  RL.Rng = Rng;
  RL.Loop = Loop;
  RL.util = U;
})(window.RL);
