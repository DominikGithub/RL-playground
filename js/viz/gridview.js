/* Canvas renderer for GridWorld: value heatmaps, Q-triangles, policy arrows, agent and trail, editing. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';
  const { h, cssVar, setupCanvas, divColor, fmt } = RL.util;
  const CELL = RL.GridWorld.CELL;

  class GridView {
    constructor(parent, o = {}) {
      this.o = Object.assign({ maxW: 620, maxH: 440, maxCell: 76 }, o);
      this.canvas = h('canvas.grid-canvas', { role: 'img', 'aria-label': 'Grid world' });
      this.wrap = h('div.viz-center', this.canvas);
      parent.appendChild(this.wrap);
      this.hover = -1; this.painting = false; this.last = null;
      this.selected = -1; this.dropTarget = -1;
      const cellAt = (e) => this.cellAt(e.clientX, e.clientY);
      if (this.o.onSelect) {
        this.canvas.classList.add('selectable');
        this.canvas.setAttribute('tabindex', '0');
        this.canvas.setAttribute('aria-label', 'Grid world editor. Click a cell to select it; arrow keys move the selection.');
        this.canvas.addEventListener('pointerdown', (e) => { const s = cellAt(e); if (s >= 0) { this.o.onSelect(s); this.canvas.focus({ preventScroll: true }); } });
        if (this.o.onKey) this.canvas.addEventListener('keydown', (e) => this.o.onKey(e));
      }
      this.canvas.addEventListener('pointermove', (e) => {
        const s = cellAt(e);
        if (s !== this.hover) { this.hover = s; if (this.o.onHover) this.o.onHover(s); this.redraw(); }
        if (this.painting && s >= 0 && this.o.onPaint) this.o.onPaint(s, false);
      });
      this.canvas.addEventListener('pointerleave', () => { this.hover = -1; this.painting = false; if (this.o.onHover) this.o.onHover(-1); this.redraw(); });
      this.canvas.addEventListener('pointerdown', (e) => {
        const s = cellAt(e); if (s < 0 || !this.o.onPaint) return;
        this.painting = true; this.canvas.setPointerCapture(e.pointerId); this.o.onPaint(s, true);
      });
      this.canvas.addEventListener('pointerup', () => { this.painting = false; });
      if (RL.onTheme) RL.onTheme(() => this.redraw());
      if (window.ResizeObserver) { this._ro = new ResizeObserver(() => this.redraw()); this._ro.observe(parent); }
    }

    /** Cell index under a viewport point, or -1. */
    cellAt(clientX, clientY) {
      if (!this.env || !this.cs) return -1;
      const r = this.canvas.getBoundingClientRect();
      const x = Math.floor((clientX - r.left) / this.cs), y = Math.floor((clientY - r.top) / this.cs);
      if (x < 0 || y < 0 || x >= this.env.w || y >= this.env.h) return -1;
      return this.env.idx(x, y);
    }

    setEnv(env) { this.env = env; this.selected = -1; this.dropTarget = -1; this.redraw(); }
    redraw() { if (this.last) this.draw(this.last); }

    /**
     * d: {mode: 'V'|'Q'|'visits'|'plain', V, Q, visits, arrows: number[][], agent, trail, scale, numbers}
     */
    draw(d) {
      this.last = d;
      const env = this.env; if (!env) return;
      const avail = Math.max(160, Math.min(this.o.maxW, (this.wrap.clientWidth || this.o.maxW) - 2));
      const cs = Math.max(14, Math.floor(Math.min(avail / env.w, this.o.maxH / env.h, this.o.maxCell)));
      this.cs = cs;
      const W = cs * env.w, H = cs * env.h;
      const ctx = setupCanvas(this.canvas, W, H);
      const C = { wall: cssVar('--wall'), goal: cssVar('--goal'), pit: cssVar('--pit'), text: cssVar('--text'), panel: cssVar('--panel'), accent: cssVar('--accent'), muted: cssVar('--muted'), mid: cssVar('--mid'), pos: cssVar('--pos') };
      ctx.clearRect(0, 0, W, H);
      const mode = d.mode || 'V';
      let scale = d.scale;
      if (!scale) {
        scale = 1e-9;
        for (let s = 0; s < env.nS; s++) {
          if (!env.isActive(s)) continue;
          if (mode === 'V' && d.V) scale = Math.max(scale, Math.abs(d.V[s]));
          if (mode === 'Q' && d.Q) for (let a = 0; a < 4; a++) scale = Math.max(scale, Math.abs(d.Q[s * 4 + a]));
        }
        if (scale < 1e-6) scale = 1; // all values ~0 (e.g. before the first sweep): neutral colours, readable legend
      }
      let maxVisit = 1;
      if (mode === 'visits' && d.visits) for (const v of d.visits) maxVisit = Math.max(maxVisit, v);
      const font = (px, w = 500) => `${w} ${px}px system-ui, sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';

      for (let s = 0; s < env.nS; s++) {
        const [x, y] = env.xy(s); const x0 = x * cs, y0 = y * cs;
        const c = env.cells[s];
        if (c === CELL.WALL) { ctx.fillStyle = C.wall; ctx.fillRect(x0, y0, cs, cs); continue; }
        if (c === CELL.GOAL || c === CELL.PIT || c === CELL.CLIFF) {
          ctx.fillStyle = c === CELL.GOAL ? C.goal : C.pit;
          ctx.globalAlpha = c === CELL.CLIFF ? 0.75 : 1;
          ctx.fillRect(x0, y0, cs, cs); ctx.globalAlpha = 1;
          ctx.fillStyle = '#fff'; ctx.font = font(Math.max(9, Math.min(14, cs * 0.24)), 700);
          const lbl = c === CELL.GOAL ? env.goalReward === 0 ? (cs >= 40 ? 'goal' : 'G') : (env.goalReward >= 0 ? '+' : '') + fmt(env.goalReward, env.goalReward % 1 ? 1 : 0) : c === CELL.PIT ? fmt(env.pitReward, env.pitReward % 1 ? 1 : 0) : (cs >= 40 ? 'cliff' : '');
          ctx.fillText(c === CELL.GOAL && cs < 30 ? 'G' : lbl, x0 + cs / 2, y0 + cs / 2);
          continue;
        }
        // empty cell
        if (mode === 'Q' && d.Q) {
          const cx = x0 + cs / 2, cy = y0 + cs / 2;
          const corners = [[x0, y0], [x0 + cs, y0], [x0 + cs, y0 + cs], [x0, y0 + cs]];
          // triangle a: up uses corners 0,1; right 1,2; down 2,3; left 3,0
          for (let a = 0; a < 4; a++) {
            const p1 = corners[a], p2 = corners[(a + 1) % 4];
            ctx.fillStyle = divColor(d.Q[s * 4 + a], scale);
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.closePath(); ctx.fill();
          }
          ctx.strokeStyle = C.panel; ctx.lineWidth = 0.6; ctx.globalAlpha = 0.7;
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + cs, y0 + cs); ctx.moveTo(x0 + cs, y0); ctx.lineTo(x0, y0 + cs); ctx.stroke(); ctx.globalAlpha = 1;
          if (d.numbers !== false && cs >= 62) {
            ctx.fillStyle = C.text; ctx.font = font(Math.min(11, cs * 0.15));
            const off = cs * 0.3;
            [[0, -off], [off, 0], [0, off], [-off, 0]].forEach(([dx, dy], a) => ctx.fillText(fmt(d.Q[s * 4 + a], Math.abs(d.Q[s * 4 + a]) >= 10 ? 0 : 1), cx + dx * (a % 2 ? 0.92 : 1), cy + dy));
          }
        } else if (mode === 'visits' && d.visits) {
          const t = Math.log(1 + d.visits[s]) / Math.log(1 + maxVisit);
          ctx.fillStyle = C.mid; ctx.fillRect(x0, y0, cs, cs);
          ctx.fillStyle = C.pos; ctx.globalAlpha = t * 0.9; ctx.fillRect(x0, y0, cs, cs); ctx.globalAlpha = 1;
          if (cs >= 40) { ctx.fillStyle = C.text; ctx.font = font(Math.min(11, cs * 0.2)); ctx.fillText(String(d.visits[s]), x0 + cs / 2, y0 + cs * 0.78); }
        } else if (mode === 'V' && d.V) {
          ctx.fillStyle = divColor(d.V[s], scale); ctx.fillRect(x0, y0, cs, cs);
          if (d.numbers !== false && cs >= 40) {
            ctx.fillStyle = C.text; ctx.font = font(Math.min(12, cs * 0.2));
            ctx.fillText(fmt(d.V[s], Math.abs(d.V[s]) >= 10 ? 1 : 2), x0 + cs / 2, y0 + cs * 0.8);
          }
        } else { ctx.fillStyle = C.mid; ctx.fillRect(x0, y0, cs, cs); }
        if (s === env.start) {
          ctx.fillStyle = C.text; ctx.globalAlpha = 0.75; ctx.font = font(Math.max(9, Math.min(12, cs * 0.2)), 700);
          ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('S', x0 + 3, y0 + 2);
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.globalAlpha = 1;
        }
      }
      // grid lines
      ctx.strokeStyle = C.panel; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x <= env.w; x++) { ctx.moveTo(x * cs + 0.5, 0); ctx.lineTo(x * cs + 0.5, H); }
      for (let y = 0; y <= env.h; y++) { ctx.moveTo(0, y * cs + 0.5); ctx.lineTo(W, y * cs + 0.5); }
      ctx.stroke();

      // arrows
      if (d.arrows) {
        ctx.strokeStyle = C.text; ctx.fillStyle = C.text; ctx.globalAlpha = 0.8;
        for (let s = 0; s < env.nS; s++) {
          const acts = d.arrows[s]; if (!acts || !acts.length || acts.length === 4 || !env.isActive(s)) continue;
          const [x, y] = env.xy(s); const cx = x * cs + cs / 2, cy = y * cs + (mode === 'V' && cs >= 40 && d.numbers !== false ? cs * 0.4 : cs / 2);
          const len = cs * (acts.length > 1 ? 0.24 : 0.3), hw = Math.max(2.5, cs * 0.07);
          ctx.lineWidth = Math.max(1.2, cs * 0.035);
          for (const a of acts) {
            const dx = [0, 1, 0, -1][a], dy = [-1, 0, 1, 0][a];
            const ex = cx + dx * len, ey = cy + dy * len;
            ctx.beginPath(); ctx.moveTo(cx - dx * len * 0.35, cy - dy * len * 0.35); ctx.lineTo(ex, ey); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(ex + dx * hw, ey + dy * hw); ctx.lineTo(ex - dy * hw, ey + dx * hw); ctx.lineTo(ex + dy * hw, ey - dx * hw); ctx.closePath(); ctx.fill();
          }
        }
        ctx.globalAlpha = 1;
      }
      // trail
      if (d.trail && d.trail.length > 1) {
        ctx.strokeStyle = C.accent; ctx.lineWidth = Math.max(2, cs * 0.06); ctx.globalAlpha = 0.55; ctx.lineJoin = 'round';
        ctx.beginPath();
        d.trail.forEach((s, i) => { const [x, y] = env.xy(s); const px = x * cs + cs / 2, py = y * cs + cs / 2; if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
        ctx.stroke(); ctx.globalAlpha = 1;
      }
      // agent
      if (d.agent !== undefined && d.agent >= 0) {
        const [x, y] = env.xy(d.agent);
        ctx.fillStyle = C.accent; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x * cs + cs / 2, y * cs + cs / 2, Math.max(4, cs * 0.2), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      // hover
      if (this.hover >= 0 && this.hover !== this.selected) {
        const [x, y] = env.xy(this.hover);
        ctx.strokeStyle = C.text; ctx.lineWidth = 2; ctx.strokeRect(x * cs + 1, y * cs + 1, cs - 2, cs - 2);
      }
      // selected cell: thick accent frame with a light inner line for contrast on any colour
      if (this.selected >= 0 && this.selected < env.nS) {
        const [x, y] = env.xy(this.selected);
        ctx.strokeStyle = C.accent; ctx.lineWidth = 4; ctx.strokeRect(x * cs + 2, y * cs + 2, cs - 4, cs - 4);
        ctx.strokeStyle = C.panel; ctx.lineWidth = 1; ctx.strokeRect(x * cs + 4.5, y * cs + 4.5, cs - 9, cs - 9);
      }
      // drop target while dragging a tile
      if (this.dropTarget >= 0) {
        const [x, y] = env.xy(this.dropTarget);
        ctx.fillStyle = C.accent; ctx.globalAlpha = 0.35; ctx.fillRect(x * cs, y * cs, cs, cs); ctx.globalAlpha = 1;
        ctx.setLineDash([5, 4]); ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.strokeRect(x * cs + 1, y * cs + 1, cs - 2, cs - 2); ctx.setLineDash([]);
      }
      this.scale = scale;
    }

    destroy() { if (this._ro) this._ro.disconnect(); }
  }

  /** Small horizontal legend for the diverging color map. */
  GridView.legend = (parent, getScale, label = 'value') => {
    const bar = h('span.bar');
    const lo = h('span'), hi = h('span');
    const el = h('div.color-legend', lo, bar, hi, h('span', label));
    parent.appendChild(el);
    const update = () => {
      const s = getScale() || 1;
      bar.style.background = `linear-gradient(90deg, ${divColor(-s, s)}, ${divColor(0, s)}, ${divColor(s, s)})`;
      lo.textContent = fmt(-s, s >= 10 ? 0 : 2); hi.textContent = fmt(s, s >= 10 ? 0 : 2);
    };
    update();
    return { el, update };
  };

  RL.GridView = GridView;
})(window.RL);
