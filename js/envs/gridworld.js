/* GridWorld MDP with walls, goals, pits, cliffs, stochastic "slip" and a living (step) reward. */
window.RL = window.RL || {};
(function (RL) {
  'use strict';

  const CELL = { EMPTY: 0, WALL: 1, GOAL: 2, PIT: 3, CLIFF: 4 };
  const DX = [0, 1, 0, -1], DY = [-1, 0, 1, 0];
  const ARROWS = ['↑', '→', '↓', '←'];
  const ACTION_NAMES = ['up', 'right', 'down', 'left'];

  class GridWorld {
    constructor(w, h, o = {}) {
      this.w = w; this.h = h;
      this.cells = new Uint8Array(w * h);
      this.start = o.start || 0;
      this.slip = o.slip ?? 0;
      this.stepReward = o.stepReward ?? -0.04;
      this.goalReward = o.goalReward ?? 1;
      this.pitReward = o.pitReward ?? -1;
      this.cliffReward = o.cliffReward ?? -100;
    }

    /** Build from strings: '.' empty, '#' wall, 'G' goal, 'P' pit, 'C' cliff, 'S' start. */
    static fromStrings(rows, o = {}) {
      const g = new GridWorld(rows[0].length, rows.length, o);
      rows.forEach((row, y) => [...row].forEach((ch, x) => {
        const s = y * g.w + x;
        g.cells[s] = ch === '#' ? CELL.WALL : ch === 'G' ? CELL.GOAL : ch === 'P' ? CELL.PIT : ch === 'C' ? CELL.CLIFF : CELL.EMPTY;
        if (ch === 'S') g.start = s;
      }));
      return g;
    }

    get nS() { return this.w * this.h; }
    xy(s) { return [s % this.w, Math.floor(s / this.w)]; }
    idx(x, y) { return y * this.w + x; }
    isWall(s) { return this.cells[s] === CELL.WALL; }
    isTerminal(s) { const c = this.cells[s]; return c === CELL.GOAL || c === CELL.PIT; }
    /** States the agent can actually occupy and act in (not walls, not terminal, not cliff). */
    isActive(s) { const c = this.cells[s]; return c === CELL.EMPTY; }

    moveDet(s, a) {
      const [x, y] = this.xy(s);
      const nx = x + DX[a], ny = y + DY[a];
      if (nx < 0 || ny < 0 || nx >= this.w || ny >= this.h) return s;
      const n = this.idx(nx, ny);
      return this.cells[n] === CELL.WALL ? s : n;
    }

    /** Actual action distribution: intended with 1-slip, each perpendicular with slip/2. */
    actualActions(a) {
      if (this.slip <= 0) return [[a, 1]];
      return [[a, 1 - this.slip], [(a + 1) % 4, this.slip / 2], [(a + 3) % 4, this.slip / 2]];
    }

    _result(s, a2) {
      const n = this.moveDet(s, a2);
      const c = this.cells[n];
      if (c === CELL.GOAL) return { s2: n, r: this.stepReward + this.goalReward, done: true };
      if (c === CELL.PIT) return { s2: n, r: this.stepReward + this.pitReward, done: true };
      if (c === CELL.CLIFF) return { s2: this.start, r: this.cliffReward, done: false, fell: true, cliffCell: n };
      return { s2: n, r: this.stepReward, done: false };
    }

    /** Full model p(s', r | s, a) as a list of outcomes (used by dynamic programming). */
    outcomes(s, a) {
      return this.actualActions(a).map(([a2, p]) => Object.assign({ p, actual: a2 }, this._result(s, a2)));
    }

    /** Sample one transition. */
    step(s, a, rng) {
      let a2 = a;
      if (this.slip > 0) {
        const u = rng.next();
        if (u < this.slip / 2) a2 = (a + 1) % 4; else if (u < this.slip) a2 = (a + 3) % 4;
      }
      return Object.assign({ actual: a2 }, this._result(s, a2));
    }

    clone() {
      const g = new GridWorld(this.w, this.h, this);
      g.cells.set(this.cells); g.start = this.start;
      return g;
    }
  }

  const PRESETS = {
    classic: {
      name: 'Classic 4×3 (Russell & Norvig)',
      rows: ['...G', '.#.P', 'S...'],
      opts: { slip: 0.2, stepReward: -0.04, goalReward: 1, pitReward: -1 }, gamma: 1,
    },
    edge: {
      name: 'Cliff edge (stochastic)',
      rows: ['.........', '.........', 'S.......G', 'PPPPPPPPP'],
      opts: { slip: 0.2, stepReward: -0.02, goalReward: 1, pitReward: -1 }, gamma: 0.95,
    },
    maze: {
      name: 'Maze',
      rows: [
        'S..#......',
        '.#.#.####.',
        '.#...#....',
        '.####.#.##',
        '......#...',
        '.####.###.',
        '....#....G',
      ],
      opts: { slip: 0, stepReward: -0.01, goalReward: 1, pitReward: -1 }, gamma: 0.95,
    },
    traps: {
      name: 'Open field with traps',
      rows: ['S.......', '..P.....', '....#...', '.#..#.P.', '....#...', '..P....G'],
      opts: { slip: 0.1, stepReward: -0.02, goalReward: 1, pitReward: -1 }, gamma: 0.95,
    },
    corridor: {
      name: 'Long corridor (credit assignment)',
      rows: ['S.............G'],
      opts: { slip: 0, stepReward: 0, goalReward: 1, pitReward: -1 }, gamma: 0.9,
    },
    cliff: {
      name: 'Cliff walking (Sutton & Barto)',
      rows: ['............', '............', '............', 'SCCCCCCCCCCG'],
      opts: { slip: 0, stepReward: -1, goalReward: 0, cliffReward: -100 }, gamma: 1,
    },
  };

  RL.GridWorld = GridWorld;
  RL.GridWorld.CELL = CELL;
  RL.GridWorld.ARROWS = ARROWS;
  RL.GridWorld.ACTION_NAMES = ACTION_NAMES;
  RL.GridWorld.PRESETS = PRESETS;
  RL.GridWorld.preset = (key) => { const p = PRESETS[key]; return GridWorld.fromStrings(p.rows, p.opts); };
})(window.RL);
