/* Lightweight canvas line chart with smoothing, log axis, legend and markers. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { h, cssVar, setupCanvas, fmt } = RL.util;

  function niceTicks(min, max, n = 5) {
    const span = max - min || Math.abs(max) || 1;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return ticks;
  }
  function tickLabel(v) {
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e4 || a < 1e-3) return v.toExponential(0).replace('-', '−');
    if (a >= 100) return Math.round(v).toString().replace('-', '−');
    return (+v.toPrecision(3)).toString().replace('-', '−');
  }
  const resolve = (c) => (c && c.startsWith('--') ? cssVar(c) : c);

  class LineChart {
    constructor(parent, opts = {}) {
      this.opts = Object.assign({
        height: 170, title: '', xLabel: '', yLabel: '', logY: false, smooth: 0,
        yMin: undefined, yMax: undefined, xMin: undefined, xMax: undefined, xStep: 1, x0: 1,
        hlines: [], legend: true,
      }, opts);
      this.series = [];
      this.markerX = null;
      this.el = h('div.chart');
      if (this.opts.title) {
        this.titleEl = h('div.chart-title', h('span', this.opts.title));
        this.el.appendChild(this.titleEl);
        // ⓘ legend: chart-specific text plus automatic notes on smoothing, log axis and reference lines
        if (RL.ui && RL.ui.info) {
          const sw = RL.ui.sw, rows = [];
          if (this.opts.smooth) rows.push([sw.line('var(--text)'), '<b>Bold line</b>: smoothed trend (exponential moving average).'], [sw.faint('var(--text)'), '<b>Faint line</b>: raw values (one point per episode/step).']);
          for (const hl of this.opts.hlines) rows.push([sw.dashed('var(--muted)'), hl.info || (hl.label ? `Dashed line: ${hl.label}.` : `Dashed line at ${hl.y.toPrecision(3)} (reference).`)]);
          if (this.opts.logY) rows.push([sw.text('10ˣ'), 'Logarithmic y-axis: equal distances = equal <em>factors</em>. A straight downward line means geometric (exponential) decrease.']);
          if (this.opts.marker) rows.push([sw.vline('var(--text)'), this.opts.marker]);
          if (this.opts.info || rows.length) {
            RL.ui.info(this.titleEl, (this.opts.info ? `<p class="lg-intro">${this.opts.info}</p>` : '') + (rows.length ? RL.ui.lg(rows) : ''), this.opts.title);
          }
        }
      }
      this.legendEl = h('div.chart-legend');
      if (this.opts.legend) this.el.appendChild(this.legendEl);
      this.canvas = h('canvas');
      this.el.appendChild(this.canvas);
      parent.appendChild(this.el);
      this._pending = false;
      if (window.ResizeObserver) {
        this._ro = new ResizeObserver(() => this.requestDraw());
        this._ro.observe(this.el);
      }
      if (RL.onTheme) RL.onTheme(() => this.requestDraw());
      this.requestDraw();
    }

    addSeries(name, color, o = {}) {
      const s = Object.assign({ name, color, data: [], xs: null, type: 'line', width: 2, dashed: false, raw: true, smooth: undefined }, o);
      this.series.push(s);
      this._legend();
      this.requestDraw();
      return s;
    }
    get(name) { return this.series.find((s) => s.name === name); }
    push(name, y) { const s = this.get(name); if (s) { s.data.push(y); this.requestDraw(); } }
    set(name, data, xs = null) { const s = this.get(name); if (s) { s.data = data; s.xs = xs; this.requestDraw(); } }
    clear() { for (const s of this.series) { s.data = []; s.xs = null; } this.requestDraw(); }
    removeAll() { this.series = []; this._legend(); this.requestDraw(); }

    _legend() {
      if (!this.opts.legend) return;
      this.legendEl.innerHTML = '';
      for (const s of this.series) {
        if (s.hideLegend) continue;
        this.legendEl.appendChild(h('span.lg-item', h('span.lg-swatch', { style: { background: resolve(s.color) } }), s.name));
      }
    }

    requestDraw() {
      if (this._pending) return;
      this._pending = true;
      requestAnimationFrame(() => { this._pending = false; this.draw(); });
    }

    _smooth(data, k) {
      if (!k) return data;
      const out = new Array(data.length); let m = 0, w = 0;
      for (let i = 0; i < data.length; i++) {
        m = k * m + (1 - k) * data[i]; w = k * w + (1 - k);
        out[i] = m / w;
      }
      return out;
    }

    draw() {
      const W = this.canvas.parentElement ? this.canvas.parentElement.clientWidth : 0;
      if (!W) return;
      const H = this.opts.height;
      const ctx = setupCanvas(this.canvas, W, H);
      const o = this.opts;
      const padL = 46, padR = 10, padT = 8, padB = o.xLabel ? 32 : 20;
      const pw = W - padL - padR, ph = H - padT - padB;
      const text = cssVar('--muted'), grid = cssVar('--grid'), axis = cssVar('--border');
      ctx.clearRect(0, 0, W, H);
      ctx.font = '11px system-ui, sans-serif';

      // ranges
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      const prepared = this.series.map((s) => {
        const k = s.smooth !== undefined ? s.smooth : o.smooth;
        const ys = s.type === 'line' ? this._smooth(s.data, k) : s.data;
        const xs = s.xs || s.data.map((_, i) => o.x0 + i * o.xStep);
        for (let i = 0; i < ys.length; i++) {
          let y = ys[i]; if (!Number.isFinite(y)) continue;
          if (o.logY) { if (y <= 0) continue; y = Math.log10(y); }
          if (y < ymin) ymin = y; if (y > ymax) ymax = y;
          if (xs[i] < xmin) xmin = xs[i]; if (xs[i] > xmax) xmax = xs[i];
        }
        if (k && s.raw && s.type === 'line') {
          for (let i = 0; i < s.data.length; i++) {
            let y = s.data[i]; if (!Number.isFinite(y)) continue;
            if (o.logY) { if (y <= 0) continue; y = Math.log10(y); }
            if (y < ymin) ymin = y; if (y > ymax) ymax = y;
          }
        }
        return { s, ys, xs, k };
      });
      for (const hl of o.hlines) { const y = o.logY ? Math.log10(hl.y) : hl.y; if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
      if (o.yMin !== undefined) ymin = o.logY ? Math.log10(o.yMin) : o.yMin;
      if (o.yMax !== undefined) ymax = o.logY ? Math.log10(o.yMax) : o.yMax;
      if (o.xMin !== undefined) xmin = o.xMin;
      if (o.xMax !== undefined) xmax = o.xMax;
      const empty = !Number.isFinite(ymin) || !Number.isFinite(xmin);
      if (empty) { ymin = 0; ymax = 1; xmin = 0; xmax = 1; }
      if (ymax - ymin < 1e-9) { ymin -= 0.5; ymax += 0.5; }
      if (xmax - xmin < 1e-9) { xmax = xmin + 1; }
      if (o.yMin === undefined && !o.logY) { const pad = (ymax - ymin) * 0.06; ymin -= pad; ymax += pad; }
      const X = (x) => padL + ((x - xmin) / (xmax - xmin)) * pw;
      const Y = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * ph;

      // grid + ticks
      ctx.strokeStyle = grid; ctx.fillStyle = text; ctx.lineWidth = 1;
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      let yt;
      if (o.logY) { yt = []; for (let e = Math.floor(ymin); e <= Math.ceil(ymax); e++) if (e >= ymin - 1e-9 && e <= ymax + 1e-9) yt.push(e); if (yt.length < 2) yt = niceTicks(ymin, ymax, 4); }
      else yt = niceTicks(ymin, ymax, 4);
      for (const t of yt) {
        const y = Y(t); if (y < padT - 1 || y > padT + ph + 1) continue;
        ctx.beginPath(); ctx.moveTo(padL, y + 0.5); ctx.lineTo(padL + pw, y + 0.5); ctx.stroke();
        ctx.fillText(o.logY ? (Number.isInteger(Math.round(t * 1e6) / 1e6) ? '1e' + Math.round(t) : tickLabel(Math.pow(10, t))) : tickLabel(t), padL - 5, y);
      }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const t of niceTicks(xmin, xmax, Math.max(2, Math.floor(pw / 70)))) {
        const x = X(t); if (x < padL - 1 || x > padL + pw + 1) continue;
        ctx.beginPath(); ctx.moveTo(x + 0.5, padT); ctx.lineTo(x + 0.5, padT + ph); ctx.stroke();
        ctx.textAlign = x > W - 22 ? 'right' : 'center';
        ctx.fillText(tickLabel(t), x > W - 22 ? W - 1 : x, padT + ph + 4);
      }
      ctx.strokeStyle = axis;
      ctx.beginPath(); ctx.moveTo(padL + 0.5, padT); ctx.lineTo(padL + 0.5, padT + ph); ctx.lineTo(padL + pw, padT + ph + 0.5); ctx.stroke();
      if (o.xLabel) { ctx.textAlign = 'center'; ctx.fillText(o.xLabel, padL + pw / 2, padT + ph + 18); }
      if (o.yLabel) {
        ctx.save(); ctx.translate(11, padT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(o.yLabel, 0, 0); ctx.restore();
      }
      for (const hl of o.hlines) {
        const y = Y(o.logY ? Math.log10(hl.y) : hl.y);
        ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = resolve(hl.color || '--muted');
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke(); ctx.restore();
        if (hl.label) { ctx.fillStyle = resolve(hl.color || '--muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(hl.label, padL + 4, y - 2); }
      }

      ctx.save();
      ctx.beginPath(); ctx.rect(padL, padT - 2, pw, ph + 4); ctx.clip();
      const tr = (y) => (o.logY ? (y > 0 ? Math.log10(y) : NaN) : y);
      const drawLine = (xs, ys, color, width, alpha, dashed) => {
        const n = ys.length; if (!n) return;
        const stride = Math.max(1, Math.floor(n / (pw * 1.5)));
        ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width;
        ctx.setLineDash(dashed ? [5, 4] : []);
        ctx.beginPath(); let started = false;
        for (let i = 0; i < n; i += stride) {
          const y = tr(ys[i]); if (!Number.isFinite(y)) { started = false; continue; }
          const px = X(xs[i]), py = Y(y);
          if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
        }
        if ((n - 1) % stride !== 0) { const y = tr(ys[n - 1]); if (Number.isFinite(y)) ctx.lineTo(X(xs[n - 1]), Y(y)); }
        ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([]);
      };
      for (const { s, ys, xs, k } of prepared) {
        const color = resolve(s.color);
        if (s.type === 'points') {
          ctx.fillStyle = color;
          for (let i = 0; i < ys.length; i++) {
            const y = tr(ys[i]); if (!Number.isFinite(y)) continue;
            ctx.beginPath(); ctx.arc(X(xs[i]), Y(y), s.radius || 4, 0, Math.PI * 2); ctx.fill();
            if (s.labels && s.labels[i]) { const px = X(xs[i]), right = px > padL + pw * 0.7; ctx.textAlign = right ? 'right' : 'left'; ctx.textBaseline = 'bottom'; ctx.font = '11px system-ui, sans-serif'; ctx.fillText(s.labels[i], px + (right ? -6 : 6), Y(y) - 2); }
          }
          continue;
        }
        if (k && s.raw) drawLine(xs, s.data, color, 1, 0.25, false);
        drawLine(xs, ys, color, s.width, 1, s.dashed);
      }
      if (this.markerX !== null && this.markerX !== undefined) {
        const x = X(this.markerX);
        ctx.strokeStyle = cssVar('--text'); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.restore();
      if (empty && this.opts.emptyText !== '') {
        ctx.fillStyle = text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(this.opts.emptyText || 'no data yet', padL + pw / 2, padT + ph / 2);
      }
    }

    destroy() { if (this._ro) this._ro.disconnect(); }
  }

  RL.LineChart = LineChart;
  RL.niceTicks = niceTicks;
  RL.fmtTick = tickLabel;
  RL._fmt = fmt;
})(window.RL);
