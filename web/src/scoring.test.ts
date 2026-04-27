import { describe, expect, test } from 'vitest';
import {
  cycleFace,
  formatAllowed,
  isJokerActive,
  isYahtzee,
  keepMaskFor,
  maxYahtzeeBonuses,
  parseScoreInput,
  randomDice,
  scoreContributionMask,
  type AllowedScores,
} from './scoring';
import { SMALL_STRAIGHT_IDX, YAHTZEE_IDX } from './constants';

describe('isJokerActive', () => {
  // Build a 13-slot scorecard with given fills. `fills` maps idx → value
  // (use 0 for "filled with forfeit", 50 for "scored Yahtzee", etc.).
  function scorecard(fills: Record<number, number>): (number | null)[] {
    const s: (number | null)[] = new Array(13).fill(null);
    for (const [k, v] of Object.entries(fills)) s[Number(k)] = v;
    return s;
  }

  test('non-Yahtzee dice → false regardless of scorecard', () => {
    expect(isJokerActive([1, 2, 3, 4, 5], scorecard({}))).toBe(false);
    // Even with everything filled, dice aren't a Yahtzee.
    expect(isJokerActive([1, 1, 1, 1, 2], scorecard({
      0: 5, [YAHTZEE_IDX]: 50,
    }))).toBe(false);
  });

  test('Yahtzee dice but Yahtzee box unfilled → false', () => {
    // Rolled Yahtzee of 3s, but Yahtzee box still empty: not yet a joker
    // (you'd score this *as* a Yahtzee for 50).
    expect(isJokerActive([3, 3, 3, 3, 3], scorecard({ 2: 9 }))).toBe(false);
  });

  test('Yahtzee dice + Yahtzee box filled but matching upper unfilled → false', () => {
    // Rolled Yahtzee of 4s, Yahtzee box scored, but Fours row still empty:
    // the right move is to score this as Fours (= 20), not joker.
    expect(isJokerActive([4, 4, 4, 4, 4], scorecard({
      [YAHTZEE_IDX]: 50,
    }))).toBe(false);
  });

  test('Yahtzee dice + both Yahtzee box and matching upper filled → true', () => {
    // Rolled Yahtzee of 6s; Sixes already scored 24, Yahtzee already 50.
    expect(isJokerActive([6, 6, 6, 6, 6], scorecard({
      5: 24, [YAHTZEE_IDX]: 50,
    }))).toBe(true);
  });

  test('Yahtzee box scored as 0 (forfeit) still counts as filled', () => {
    // Forfeit Yahtzee: the rule still enables joker on subsequent Yahtzees.
    expect(isJokerActive([2, 2, 2, 2, 2], scorecard({
      1: 8, [YAHTZEE_IDX]: 0,
    }))).toBe(true);
  });

  test('upper box scored as 0 still counts as filled', () => {
    // Imagine the player crossed off the Ones row earlier. Now they roll a
    // Yahtzee of 1s with Yahtzee already scored — joker fires.
    expect(isJokerActive([1, 1, 1, 1, 1], scorecard({
      0: 0, [YAHTZEE_IDX]: 50,
    }))).toBe(true);
  });
});

describe('isYahtzee', () => {
  test('all-same returns true', () => {
    expect(isYahtzee([3, 3, 3, 3, 3])).toBe(true);
    expect(isYahtzee([1, 1, 1, 1, 1])).toBe(true);
  });
  test('any difference returns false', () => {
    expect(isYahtzee([1, 1, 1, 1, 2])).toBe(false);
    expect(isYahtzee([1, 2, 3, 4, 5])).toBe(false);
    expect(isYahtzee([6, 6, 6, 6, 5])).toBe(false);
  });
});

describe('keepMaskFor', () => {
  test('exact subset is matched in original positions', () => {
    // Keep two 2s out of [2,2,3,4,5] → first two positions.
    expect(keepMaskFor([2, 2], [2, 2, 3, 4, 5])).toEqual(
      [true, true, false, false, false],
    );
    // Order of currentDice matters for which physical dice are kept.
    expect(keepMaskFor([2, 2], [3, 2, 4, 2, 5])).toEqual(
      [false, true, false, true, false],
    );
  });

  test('multiset semantics — only consume as many copies as requested', () => {
    // Three 1s in dice but we only want two.
    expect(keepMaskFor([1, 1], [1, 1, 1, 4, 5])).toEqual(
      [true, true, false, false, false],
    );
  });

  test('faces not present in dice are silently dropped', () => {
    // Asking to keep a 6 we don't have doesn't error and doesn't affect 1s.
    expect(keepMaskFor([1, 6], [1, 2, 3, 4, 5])).toEqual(
      [true, false, false, false, false],
    );
  });

  test('empty faces → all-false mask', () => {
    expect(keepMaskFor([], [1, 2, 3, 4, 5])).toEqual(
      [false, false, false, false, false],
    );
  });

  test('keeping all 5 → all-true mask', () => {
    expect(keepMaskFor([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])).toEqual(
      [true, true, true, true, true],
    );
  });
});

describe('scoreContributionMask', () => {
  test('upper rows highlight only the matching face', () => {
    // Threes (idx 2) on [3,3,1,3,5] → positions 0,1,3.
    expect(scoreContributionMask(2, [3, 3, 1, 3, 5])).toEqual(
      [true, true, false, true, false],
    );
    // Sixes (idx 5) with no sixes → all false.
    expect(scoreContributionMask(5, [1, 2, 3, 4, 5])).toEqual(
      [false, false, false, false, false],
    );
  });

  test('small straight highlights the first matching 4-in-a-row', () => {
    // [1,2,3,4,6] has run 1-4 → first four positions.
    expect(scoreContributionMask(SMALL_STRAIGHT_IDX, [1, 2, 3, 4, 6])).toEqual(
      [true, true, true, true, false],
    );
    // [2,3,4,5,5] has run 2-5 → first four positions (one of the 5s extra).
    expect(scoreContributionMask(SMALL_STRAIGHT_IDX, [2, 3, 4, 5, 5])).toEqual(
      [true, true, true, true, false],
    );
    // No run → nothing.
    expect(scoreContributionMask(SMALL_STRAIGHT_IDX, [1, 2, 3, 5, 6])).toEqual(
      [false, false, false, false, false],
    );
  });

  test('shaped lower-row categories use all 5 dice', () => {
    // Full house, large straight, three/four of a kind, Yahtzee, chance.
    for (const idx of [6, 7, 8, 10, YAHTZEE_IDX, 12]) {
      expect(scoreContributionMask(idx, [1, 2, 3, 4, 5])).toEqual(
        [true, true, true, true, true],
      );
    }
  });
});

describe('randomDice', () => {
  test('always returns 5 values in 1..6', () => {
    for (let i = 0; i < 50; i++) {
      const d = randomDice();
      expect(d).toHaveLength(5);
      for (const v of d) {
        expect(v).toBeGreaterThanOrEqual(1);
        expect(v).toBeLessThanOrEqual(6);
        expect(Number.isInteger(v)).toBe(true);
      }
    }
  });
});

describe('parseScoreInput', () => {
  // Stand-in for what App.svelte passes once wasm has loaded.
  // Models a row whose only legal scores are 0 plus the multiples of 5 in
  // 5..30 (close to e.g. Three of a kind's set).
  const allowed: AllowedScores = {
    values: new Set([0, 5, 10, 15, 20, 25, 30]),
    label: '0 or 5–30',
  };
  // Yahtzee row: only 0 and 50 are legal.
  const yahtzeeAllowed: AllowedScores = {
    values: new Set([0, 50]),
    label: '0, 50',
  };

  test('empty / whitespace-only input parses as "not filled"', () => {
    expect(parseScoreInput('', allowed)).toEqual({ value: null, error: null });
    expect(parseScoreInput('   ', allowed)).toEqual({ value: null, error: null });
    expect(parseScoreInput('\t\n', allowed)).toEqual({ value: null, error: null });
  });

  test('non-integer text rejected with "whole number"', () => {
    for (const bad of ['abc', '1.5', '3,5', '5e2', '+5', '5.0', '--5', 'one']) {
      expect(parseScoreInput(bad, allowed)).toEqual({
        value: null,
        error: 'whole number',
      });
    }
  });

  test('valid in-range integer returns value with no error', () => {
    expect(parseScoreInput('15', allowed)).toEqual({ value: 15, error: null });
    expect(parseScoreInput('0', allowed)).toEqual({ value: 0, error: null });
    expect(parseScoreInput('30', allowed)).toEqual({ value: 30, error: null });
  });

  test('valid integer that is not in the allowed set returns the label', () => {
    // 12 is an integer but not a legal Three-of-a-kind score.
    expect(parseScoreInput('12', allowed)).toEqual({
      value: null,
      error: '0 or 5–30',
    });
    // Above the upper bound.
    expect(parseScoreInput('100', allowed)).toEqual({
      value: null,
      error: '0 or 5–30',
    });
    // Negative parses through the regex but no allowed set contains it.
    expect(parseScoreInput('-5', allowed)).toEqual({
      value: null,
      error: '0 or 5–30',
    });
  });

  test('Yahtzee-shaped allow-set: 50 is the only non-zero legal value', () => {
    expect(parseScoreInput('50', yahtzeeAllowed)).toEqual({ value: 50, error: null });
    expect(parseScoreInput('0', yahtzeeAllowed)).toEqual({ value: 0, error: null });
    expect(parseScoreInput('25', yahtzeeAllowed)).toEqual({
      value: null,
      error: '0, 50',
    });
  });

  test('null allowed (wasm not ready) accepts any integer without error', () => {
    expect(parseScoreInput('999', null)).toEqual({ value: 999, error: null });
    expect(parseScoreInput('-5', null)).toEqual({ value: -5, error: null });
    // Non-integer is still rejected — that's a syntactic check, not a range one.
    expect(parseScoreInput('abc', null)).toEqual({
      value: null,
      error: 'whole number',
    });
    // Empty is still "not filled".
    expect(parseScoreInput('', null)).toEqual({ value: null, error: null });
  });

  test('surrounding whitespace is trimmed before parsing', () => {
    expect(parseScoreInput('  15  ', allowed)).toEqual({ value: 15, error: null });
    expect(parseScoreInput('\t30\n', allowed)).toEqual({ value: 30, error: null });
  });

  test('leading zeros are accepted (parsed as base 10)', () => {
    // '08' must not be parsed as octal — the regex passes and parseInt(..., 10)
    // gives 8. Whether 8 is in the allowed set is a separate question.
    expect(parseScoreInput('05', allowed)).toEqual({ value: 5, error: null });
    expect(parseScoreInput('08', allowed)).toEqual({
      value: null,
      error: '0 or 5–30',
    });
  });
});

describe('cycleFace', () => {
  test('+1 advances to the next face', () => {
    expect(cycleFace(1, 1)).toBe(2);
    expect(cycleFace(5, 1)).toBe(6);
  });

  test('+1 wraps from 6 back to 1', () => {
    expect(cycleFace(6, 1)).toBe(1);
  });

  test('-1 goes to the previous face', () => {
    expect(cycleFace(2, -1)).toBe(1);
    expect(cycleFace(6, -1)).toBe(5);
  });

  test('-1 wraps from 1 back to 6', () => {
    expect(cycleFace(1, -1)).toBe(6);
  });

  test('delta 0 is a no-op', () => {
    for (let f = 1; f <= 6; f++) expect(cycleFace(f, 0)).toBe(f);
  });

  test('larger deltas wrap correctly', () => {
    // +6 is one full revolution.
    for (let f = 1; f <= 6; f++) expect(cycleFace(f, 6)).toBe(f);
    // -6 is also one full revolution.
    for (let f = 1; f <= 6; f++) expect(cycleFace(f, -6)).toBe(f);
    // +2 from 5 wraps to 1.
    expect(cycleFace(5, 2)).toBe(1);
    // -3 from 2 wraps to 5.
    expect(cycleFace(2, -3)).toBe(5);
  });

  test('always returns a value in 1..6', () => {
    for (let f = 1; f <= 6; f++) {
      for (let d = -6; d <= 6; d++) {
        const out = cycleFace(f, d);
        expect(out).toBeGreaterThanOrEqual(1);
        expect(out).toBeLessThanOrEqual(6);
        expect(Number.isInteger(out)).toBe(true);
      }
    }
  });
});

describe('maxYahtzeeBonuses', () => {
  test('null Yahtzee score → 0', () => {
    expect(maxYahtzeeBonuses(null, 5)).toBe(0);
    expect(maxYahtzeeBonuses(null, 13)).toBe(0);
  });

  test('Yahtzee scored as 0 → 0 (you forfeited the bonus chain)', () => {
    expect(maxYahtzeeBonuses(0, 5)).toBe(0);
    expect(maxYahtzeeBonuses(0, 13)).toBe(0);
  });

  test('Yahtzee = 50, N turns played → N-1 max bonuses', () => {
    expect(maxYahtzeeBonuses(50, 1)).toBe(0); // only the Yahtzee turn so far
    expect(maxYahtzeeBonuses(50, 2)).toBe(1);
    expect(maxYahtzeeBonuses(50, 7)).toBe(6);
    expect(maxYahtzeeBonuses(50, 13)).toBe(12);
  });

  test('Yahtzee = 50 with 0 turns played is clamped to 0 (defensive)', () => {
    // Shouldn't happen in practice — a 50 in the box implies ≥1 turn — but
    // the implementation defends against it via Math.max.
    expect(maxYahtzeeBonuses(50, 0)).toBe(0);
  });
});

describe('formatAllowed', () => {
  test('small lists are joined with commas', () => {
    expect(formatAllowed([0, 25])).toBe('0, 25');
    expect(formatAllowed([0, 5, 10, 15, 20, 25])).toBe('0, 5, 10, 15, 20, 25');
  });

  test('long contiguous range collapses to en-dash', () => {
    const range = Array.from({ length: 10 }, (_, i) => i);
    expect(formatAllowed(range)).toBe('0–9');
  });

  test('long contiguous-after-zero formats as "0 or N–M"', () => {
    // 0, then 5..30 contiguous (e.g. three of a kind: 0 plus reachable totals)
    const vals = [0, ...Array.from({ length: 26 }, (_, i) => i + 5)];
    expect(formatAllowed(vals)).toBe('0 or 5–30');
  });

  test('long non-contiguous falls back to comma list', () => {
    const vals = [0, 5, 7, 10, 12, 15, 18];
    expect(formatAllowed(vals)).toBe('0, 5, 7, 10, 12, 15, 18');
  });

  test('exactly 6 elements uses comma list (boundary)', () => {
    // Six contiguous values still go through the comma branch.
    expect(formatAllowed([1, 2, 3, 4, 5, 6])).toBe('1, 2, 3, 4, 5, 6');
  });

  test('seven contiguous values uses range', () => {
    expect(formatAllowed([1, 2, 3, 4, 5, 6, 7])).toBe('1–7');
  });
});
