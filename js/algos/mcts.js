/* Tic-tac-toe and UCT Monte Carlo Tree Search with random rollouts. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';

  const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];

  const TTT = {
    empty: () => ({ b: new Int8Array(9), toMove: 1 }), // 1 = X, -1 = O
    clone: (st) => ({ b: Int8Array.from(st.b), toMove: st.toMove }),
    moves: (st) => { const m = []; for (let i = 0; i < 9; i++) if (st.b[i] === 0) m.push(i); return m; },
    play: (st, m) => { st.b[m] = st.toMove; st.toMove = -st.toMove; return st; },
    winner: (st) => {
      for (const [a, b, c] of LINES) if (st.b[a] && st.b[a] === st.b[b] && st.b[a] === st.b[c]) return st.b[a];
      for (let i = 0; i < 9; i++) if (st.b[i] === 0) return null;
      return 0; // draw
    },
    winLine: (st) => { for (const l of LINES) { const [a, b, c] = l; if (st.b[a] && st.b[a] === st.b[b] && st.b[a] === st.b[c]) return l; } return null; },
  };

  class Node {
    constructor(parent, move, player, untried) {
      this.parent = parent; this.move = move;
      this.player = player; // the player who made `move` (whose perspective W is counted from)
      this.children = []; this.untried = untried;
      this.N = 0; this.W = 0;
    }
  }

  /**
   * Run UCT from `state`. Returns root statistics per move.
   * Value convention: node.W accumulates results from the perspective of node.player (1 win, ½ draw, 0 loss).
   */
  function mcts(state, { sims = 500, c = 1.4, rng = new RL.Rng() } = {}) {
    const root = new Node(null, -1, -state.toMove, TTT.moves(state));
    let maxDepth = 0, nodes = 1;
    for (let i = 0; i < sims; i++) {
      let node = root; const st = TTT.clone(state); let depth = 0;
      // 1. Selection: descend with UCB1 while fully expanded
      while (node.untried.length === 0 && node.children.length > 0) {
        let best = null, bestU = -Infinity;
        const lnN = Math.log(node.N);
        for (const ch of node.children) {
          const u = ch.W / ch.N + c * Math.sqrt(lnN / ch.N);
          if (u > bestU) { bestU = u; best = ch; }
        }
        node = best; TTT.play(st, node.move); depth++;
      }
      // 2. Expansion
      if (node.untried.length > 0 && TTT.winner(st) === null) {
        const k = rng.int(node.untried.length);
        const m = node.untried.splice(k, 1)[0];
        const mover = st.toMove;
        TTT.play(st, m);
        const ch = new Node(node, m, mover, TTT.winner(st) === null ? TTT.moves(st) : []);
        node.children.push(ch); node = ch; depth++; nodes++;
      }
      maxDepth = Math.max(maxDepth, depth);
      // 3. Simulation (random rollout)
      let w = TTT.winner(st);
      while (w === null) { const ms = TTT.moves(st); TTT.play(st, ms[rng.int(ms.length)]); w = TTT.winner(st); }
      // 4. Backpropagation
      while (node) {
        node.N++;
        node.W += w === 0 ? 0.5 : w === node.player ? 1 : 0;
        node = node.parent;
      }
    }
    const stats = root.children.map((ch) => ({ move: ch.move, N: ch.N, Q: ch.W / ch.N }));
    stats.sort((a, b) => b.N - a.N);
    return { best: stats.length ? stats[0].move : -1, stats, maxDepth, nodes, root };
  }

  /** Perfect play via minimax (for "is this move optimal?" feedback). Value from X's perspective. */
  const memo = new Map();
  function minimax(st) {
    const key = st.b.join(',') + st.toMove;
    if (memo.has(key)) return memo.get(key);
    const w = TTT.winner(st);
    let v;
    if (w !== null) v = w; else {
      v = st.toMove === 1 ? -Infinity : Infinity;
      for (const m of TTT.moves(st)) {
        const c = minimax(TTT.play(TTT.clone(st), m));
        v = st.toMove === 1 ? Math.max(v, c) : Math.min(v, c);
      }
    }
    memo.set(key, v);
    return v;
  }
  /** Game-theoretic value of each legal move for the player to move (+1 win, 0 draw, −1 loss). */
  function moveValues(st) {
    const out = {};
    for (const m of TTT.moves(st)) out[m] = minimax(TTT.play(TTT.clone(st), m)) * st.toMove;
    return out;
  }

  RL.TTT = TTT;
  RL.mcts = mcts;
  RL.ttt = { minimax, moveValues };
})(window.RL);
