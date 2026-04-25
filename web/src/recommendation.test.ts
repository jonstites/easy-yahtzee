import { describe, expect, test } from 'vitest';
import {
  buildChoices,
  type Choice,
  type Recommendation,
} from './recommendation';

function entry(idx: number, ev: number, turn_ev = 0): Recommendation['entries'][number] {
  return { entry: idx, ev, turn_ev };
}
function keeper(dice: number[], ev: number, turn_ev = 0): NonNullable<Recommendation['keepers']>[number] {
  return { dice, ev, turn_ev };
}

describe('buildChoices', () => {
  test('null/undefined rec → empty array', () => {
    expect(buildChoices(null)).toEqual([]);
    expect(buildChoices(undefined)).toEqual([]);
  });

  test('roll 3 (keepers null) returns entries as score choices in given order', () => {
    const rec: Recommendation = {
      value: 100,
      keepers: null,
      // Already EV-sorted by the wasm.
      entries: [entry(11, 50, 50), entry(6, 30, 18), entry(12, 25, 25)],
    };
    const out = buildChoices(rec);
    expect(out).toEqual<Choice[]>([
      { kind: 'score', entryIdx: 11, ev: 50, turn_ev: 50 },
      { kind: 'score', entryIdx: 6, ev: 30, turn_ev: 18 },
      { kind: 'score', entryIdx: 12, ev: 25, turn_ev: 25 },
    ]);
  });

  test('rolls 1/2 merge keepers + entries and sort by ev desc', () => {
    const rec: Recommendation = {
      value: 250,
      keepers: [
        keeper([2, 2, 2], 270, 8), // best
        keeper([2, 2], 260, 4),
      ],
      entries: [
        entry(6, 265, 11), // sandwiched between the two keepers
        entry(11, 240, 0),
      ],
    };
    const out = buildChoices(rec);
    expect(out.map(c => c.ev)).toEqual([270, 265, 260, 240]);
    expect(out[0]).toMatchObject({ kind: 'keep', dice: [2, 2, 2] });
    expect(out[1]).toMatchObject({ kind: 'score', entryIdx: 6 });
    expect(out[2]).toMatchObject({ kind: 'keep', dice: [2, 2] });
    expect(out[3]).toMatchObject({ kind: 'score', entryIdx: 11 });
  });

  test('empty-dice keeper becomes a "reroll" choice', () => {
    const rec: Recommendation = {
      value: 200,
      keepers: [keeper([], 200, 5)],
      entries: [],
    };
    const out = buildChoices(rec);
    expect(out).toEqual<Choice[]>([{ kind: 'reroll', ev: 200, turn_ev: 5 }]);
  });

  test('respects the limit (default 8)', () => {
    const rec: Recommendation = {
      value: 0,
      keepers: Array.from({ length: 12 }, (_, i) => keeper([i % 6 + 1], 100 - i)),
      entries: Array.from({ length: 13 }, (_, i) => entry(i, 50 - i)),
    };
    expect(buildChoices(rec)).toHaveLength(8);
    // Custom limit honoured.
    expect(buildChoices(rec, 3)).toHaveLength(3);
    // Top of the slice is the best-EV row overall (a keeper at 100).
    expect(buildChoices(rec)[0].ev).toBe(100);
  });

  test('score entry can win the top slot when its EV beats all keepers', () => {
    // Models "rolled a Yahtzee on roll 1": best move is to score it now.
    const rec: Recommendation = {
      value: 290,
      keepers: [
        keeper([5, 5, 5, 5], 270, 6),
        keeper([5, 5, 5], 250, 4),
        keeper([5, 5], 230, 2),
      ],
      entries: [
        entry(11, 295, 50), // score in Yahtzee for 50 right now
        entry(12, 240, 25),
      ],
    };
    const out = buildChoices(rec);
    expect(out[0]).toMatchObject({ kind: 'score', entryIdx: 11, ev: 295 });
  });

  test('empty keepers list still returns entries (degenerate but safe)', () => {
    const rec: Recommendation = {
      value: 100,
      keepers: [],
      entries: [entry(0, 100, 4)],
    };
    expect(buildChoices(rec)).toEqual<Choice[]>([
      { kind: 'score', entryIdx: 0, ev: 100, turn_ev: 4 },
    ]);
  });
});
