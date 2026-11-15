/* CartPole with the classic Barto-Sutton-Anderson dynamics (same constants as Gym CartPole-v1). */
window.RL = window.RL || {};
(function (RL) {
  'use strict';

  const G = 9.8, MC = 1.0, MP = 0.1, TOTAL = MC + MP, LEN = 0.5, PML = MP * LEN, FORCE = 10, TAU = 0.02;
  const X_LIM = 2.4, TH_LIM = (12 * 2 * Math.PI) / 360;

  class CartPole {
    constructor(rng, maxSteps = 500) {
      this.rng = rng || new RL.Rng();
      this.maxSteps = maxSteps;
      this.reset();
    }
    reset() {
      const u = () => (this.rng.next() - 0.5) * 0.1;
      this.x = u(); this.xd = u(); this.th = u(); this.thd = u();
      this.t = 0; this.lastAction = -1;
      return this.obs();
    }
    /** Normalized observation fed to networks (roughly in [-1, 1]). */
    obs() { return CartPole.normalize(this.x, this.xd, this.th, this.thd); }
    static normalize(x, xd, th, thd) { return Float64Array.of(x / X_LIM, xd / 3, th / TH_LIM, thd / 3.5); }

    step(a) {
      const force = a === 1 ? FORCE : -FORCE;
      const c = Math.cos(this.th), s = Math.sin(this.th);
      const temp = (force + PML * this.thd * this.thd * s) / TOTAL;
      const thacc = (G * s - c * temp) / (LEN * (4 / 3 - (MP * c * c) / TOTAL));
      const xacc = temp - (PML * thacc * c) / TOTAL;
      this.x += TAU * this.xd; this.xd += TAU * xacc;
      this.th += TAU * this.thd; this.thd += TAU * thacc;
      this.t++; this.lastAction = a;
      const done = Math.abs(this.x) > X_LIM || Math.abs(this.th) > TH_LIM;
      const truncated = !done && this.t >= this.maxSteps;
      return { obs: this.obs(), r: 1, done, truncated };
    }

    /** Draw the cart-pole into a 2D context of size w×h (CSS px). */
    render(ctx, w, h, { probs = null, label = '' } = {}) {
      const css = RL.util.cssVar;
      ctx.clearRect(0, 0, w, h);
      const scale = (w - 40) / (2 * X_LIM + 0.6);
      const cx = w / 2 + this.x * scale, ground = h * 0.72;
      // track & limits
      ctx.strokeStyle = css('--border'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(10, ground); ctx.lineTo(w - 10, ground); ctx.stroke();
      ctx.strokeStyle = css('--neg'); ctx.lineWidth = 2; ctx.setLineDash([3, 3]);
      for (const lim of [-X_LIM, X_LIM]) { const lx = w / 2 + lim * scale; ctx.beginPath(); ctx.moveTo(lx, ground - 70); ctx.lineTo(lx, ground + 8); ctx.stroke(); }
      ctx.setLineDash([]);
      // cart
      const cw = 54, ch = 26;
      ctx.fillStyle = css('--text');
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(cx - cw / 2, ground - ch - 4, cw, ch, 5) : ctx.rect(cx - cw / 2, ground - ch - 4, cw, ch); ctx.fill();
      ctx.fillStyle = css('--muted');
      for (const dx of [-15, 15]) { ctx.beginPath(); ctx.arc(cx + dx, ground - 3, 5, 0, Math.PI * 2); ctx.fill(); }
      // pole (length drawn for 2*LEN = 1 m)
      const pl = Math.min(h * 0.6, 1.0 * scale * 1.3);
      const px = cx + Math.sin(this.th) * pl, py = ground - ch - 4 - Math.cos(this.th) * pl;
      const danger = Math.min(1, Math.abs(this.th) / TH_LIM);
      ctx.strokeStyle = danger > 0.75 ? css('--neg') : css('--accent');
      ctx.lineWidth = 7; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(cx, ground - ch - 4); ctx.lineTo(px, py); ctx.stroke();
      ctx.lineCap = 'butt';
      ctx.fillStyle = css('--panel'); ctx.beginPath(); ctx.arc(cx, ground - ch - 4, 4, 0, Math.PI * 2); ctx.fill();
      // force arrow
      if (this.lastAction >= 0) {
        const dir = this.lastAction === 1 ? 1 : -1;
        ctx.strokeStyle = css('--muted'); ctx.fillStyle = css('--muted'); ctx.lineWidth = 2;
        const ax = cx + dir * (cw / 2 + 4), ay = ground - ch / 2 - 4;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + dir * 22, ay); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ax + dir * 28, ay); ctx.lineTo(ax + dir * 20, ay - 5); ctx.lineTo(ax + dir * 20, ay + 5); ctx.fill();
      }
      // action probability bars
      if (probs) {
        const bw = 90, bx = 12, by = h - 26;
        ctx.font = '11px system-ui, sans-serif'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        ['← push left', '→ push right'].forEach((lbl, i) => {
          const y = by + i * 12 - 6;
          ctx.fillStyle = css('--grid'); ctx.fillRect(bx + 80, y - 4, bw, 8);
          ctx.fillStyle = css('--pos'); ctx.fillRect(bx + 80, y - 4, bw * probs[i], 8);
          ctx.fillStyle = css('--muted'); ctx.fillText(lbl, bx, y);
        });
      }
      ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillStyle = css('--muted');
      ctx.fillText(`t = ${this.t}${label ? '  ·  ' + label : ''}`, w - 10, 8);
    }
  }
  CartPole.X_LIM = X_LIM; CartPole.TH_LIM = TH_LIM;
  RL.CartPole = CartPole;
})(window.RL);
