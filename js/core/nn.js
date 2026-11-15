/* A tiny fully-connected network with manual backprop and Adam. Enough for CartPole-sized problems. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';

  class MLP {
    /**
     * @param {number[]} sizes e.g. [4, 64, 64, 2]
     * @param {{act?: 'tanh'|'relu', rng?: RL.Rng, lastScale?: number}} opts
     */
    constructor(sizes, opts = {}) {
      this.sizes = sizes.slice();
      this.act = opts.act || 'tanh';
      this.rng = opts.rng || new RL.Rng();
      this.L = sizes.length - 1;
      this.W = []; this.b = []; this.gW = []; this.gb = [];
      this.mW = []; this.vW = []; this.mb = []; this.vb = [];
      for (let l = 0; l < this.L; l++) {
        const nin = sizes[l], nout = sizes[l + 1];
        const W = new Float64Array(nout * nin);
        const last = l === this.L - 1;
        const scale = last && opts.lastScale !== undefined
          ? opts.lastScale
          : Math.sqrt((this.act === 'relu' ? 2 : 1) / nin);
        for (let i = 0; i < W.length; i++) W[i] = this.rng.normal() * scale;
        this.W.push(W); this.b.push(new Float64Array(nout));
        this.gW.push(new Float64Array(nout * nin)); this.gb.push(new Float64Array(nout));
        this.mW.push(new Float64Array(nout * nin)); this.vW.push(new Float64Array(nout * nin));
        this.mb.push(new Float64Array(nout)); this.vb.push(new Float64Array(nout));
      }
      this.t = 0;
      this.acts = null;
    }

    /** Forward pass; caches post-activations for a subsequent backward(). */
    forward(x) {
      let a = x instanceof Float64Array ? x : Float64Array.from(x);
      const acts = [a];
      for (let l = 0; l < this.L; l++) {
        const nin = this.sizes[l], nout = this.sizes[l + 1];
        const W = this.W[l], b = this.b[l];
        const z = new Float64Array(nout);
        for (let o = 0; o < nout; o++) {
          let s = b[o]; const row = o * nin;
          for (let i = 0; i < nin; i++) s += W[row + i] * a[i];
          z[o] = s;
        }
        if (l < this.L - 1) {
          if (this.act === 'relu') { for (let o = 0; o < nout; o++) if (z[o] < 0) z[o] = 0; }
          else { for (let o = 0; o < nout; o++) z[o] = Math.tanh(z[o]); }
        }
        acts.push(z); a = z;
      }
      this.acts = acts;
      return a;
    }

    /** Accumulate gradients of a scalar loss given dLoss/dOutput for the last forward() input. */
    backward(dOut) {
      const acts = this.acts;
      let d = dOut instanceof Float64Array ? dOut : Float64Array.from(dOut);
      for (let l = this.L - 1; l >= 0; l--) {
        const nin = this.sizes[l], nout = this.sizes[l + 1];
        const aIn = acts[l], W = this.W[l], gW = this.gW[l], gb = this.gb[l];
        for (let o = 0; o < nout; o++) {
          const dO = d[o]; if (dO === 0) continue;
          gb[o] += dO; const row = o * nin;
          for (let i = 0; i < nin; i++) gW[row + i] += dO * aIn[i];
        }
        if (l > 0) {
          const dPrev = new Float64Array(nin);
          for (let o = 0; o < nout; o++) {
            const dO = d[o]; if (dO === 0) continue; const row = o * nin;
            for (let i = 0; i < nin; i++) dPrev[i] += W[row + i] * dO;
          }
          if (this.act === 'relu') { for (let i = 0; i < nin; i++) if (aIn[i] <= 0) dPrev[i] = 0; }
          else { for (let i = 0; i < nin; i++) dPrev[i] *= 1 - aIn[i] * aIn[i]; }
          d = dPrev;
        }
      }
    }

    zeroGrad() {
      for (let l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); }
    }

    gradNorm(scale = 1) {
      let s = 0;
      for (let l = 0; l < this.L; l++) {
        for (const g of this.gW[l]) s += g * g;
        for (const g of this.gb[l]) s += g * g;
      }
      return Math.sqrt(s) * scale;
    }

    /** Adam step. `scale` multiplies raw gradients (e.g. 1/batch). Optional global-norm clipping. */
    step(lr, { scale = 1, clipNorm = 0, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
      let k = scale;
      if (clipNorm > 0) { const n = this.gradNorm(scale); if (n > clipNorm) k *= clipNorm / n; }
      this.t++;
      const bc1 = 1 - Math.pow(beta1, this.t), bc2 = 1 - Math.pow(beta2, this.t);
      const upd = (P, G, M, V) => {
        for (let i = 0; i < P.length; i++) {
          const g = G[i] * k;
          M[i] = beta1 * M[i] + (1 - beta1) * g;
          V[i] = beta2 * V[i] + (1 - beta2) * g * g;
          P[i] -= (lr * (M[i] / bc1)) / (Math.sqrt(V[i] / bc2) + eps);
        }
      };
      for (let l = 0; l < this.L; l++) {
        upd(this.W[l], this.gW[l], this.mW[l], this.vW[l]);
        upd(this.b[l], this.gb[l], this.mb[l], this.vb[l]);
      }
      this.zeroGrad();
    }

    copyFrom(o) {
      for (let l = 0; l < this.L; l++) { this.W[l].set(o.W[l]); this.b[l].set(o.b[l]); }
    }

    clone() {
      const c = new MLP(this.sizes, { act: this.act, rng: new RL.Rng(1) });
      c.copyFrom(this);
      return c;
    }

    paramCount() {
      let n = 0; for (let l = 0; l < this.L; l++) n += this.W[l].length + this.b[l].length; return n;
    }
  }

  RL.MLP = MLP;
})(window.RL);
