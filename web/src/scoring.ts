// Pure dice/category helpers used by the UI. Nothing here touches Svelte
// state or the wasm — they're plain functions you can unit-test in isolation.

import { SMALL_STRAIGHT_IDX, YAHTZEE_IDX } from './constants';

/** All 5 dice the same face. */
export function isYahtzee(dice: number[]): boolean {
  return dice.every(v => v === dice[0]);
}

/**
 * Does the Joker rule fire for the current dice + scorecard state?
 *
 * Conditions (per official Yahtzee rules):
 *   1. The dice are a Yahtzee (all 5 the same face).
 *   2. The Yahtzee box is already filled (any value — including 0, the
 *      forfeit case — counts).
 *   3. The matching upper box (Ones for a Yahtzee of 1s, etc.) is already
 *      filled.
 *
 * When active, the lower-row "shaped" categories (Full House / Small
 * Straight / Large Straight) accept fixed joker scores (25 / 30 / 40).
 *
 * `scores` is indexed by canonical entry index (`scores[YAHTZEE_IDX]`,
 * `scores[face - 1]` for upper rows); `null` means "not yet filled".
 */
export function isJokerActive(
  dice: number[],
  scores: readonly (number | null)[],
): boolean {
  if (!isYahtzee(dice)) return false;
  const face = dice[0]; // 1..6 by construction
  return scores[YAHTZEE_IDX] !== null && scores[face - 1] !== null;
}

/**
 * Build a boolean mask over `currentDice` indicating which physical dice to
 * "keep" so that the kept multiset equals `faces` (a multiset of 1..6 face
 * values). When `faces` doesn't appear as a sub-multiset of `currentDice` the
 * unmatched faces are silently dropped — the result is always a valid mask
 * that consumes as much of `faces` as possible.
 */
export function keepMaskFor(faces: number[], currentDice: number[]): boolean[] {
  const mask = [false, false, false, false, false];
  const remaining = [0, 0, 0, 0, 0, 0];
  for (const v of faces) remaining[v - 1]++;
  for (let i = 0; i < 5; i++) {
    const f = currentDice[i];
    if (remaining[f - 1] > 0) {
      mask[i] = true;
      remaining[f - 1]--;
    }
  }
  return mask;
}

/** Three 4-in-a-row windows that satisfy the small-straight category. */
const SMALL_STRAIGHT_RUNS: readonly number[][] = [
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6],
];

/**
 * Mask of dice that "matter" if you scored these dice in `entryIdx`. Used by
 * the UI to fade out non-contributing dice. Approximations:
 *   - Upper rows (0..=5): only dice with the matching face count.
 *   - Small straight: the first 4-in-a-row found; nothing if none present.
 *   - Everything else: all 5 dice (full house, large straight, three/four
 *     of a kind, chance, Yahtzee — all consume the whole roll).
 */
export function scoreContributionMask(entryIdx: number, dice: number[]): boolean[] {
  if (entryIdx >= 0 && entryIdx <= 5) {
    const face = entryIdx + 1;
    return dice.map(v => v === face);
  }
  if (entryIdx === SMALL_STRAIGHT_IDX) {
    const present = new Set(dice);
    const chosen = SMALL_STRAIGHT_RUNS.find(r => r.every(x => present.has(x)));
    return chosen ? keepMaskFor(chosen, dice) : [false, false, false, false, false];
  }
  return [true, true, true, true, true];
}

/** A fresh random roll of 5 dice. */
export function randomDice(): number[] {
  return Array.from({ length: 5 }, () => 1 + Math.floor(Math.random() * 6));
}

/**
 * Advance a single die's face by `delta` (typically ±1), wrapping around the
 * 1..6 cycle. Used by the click/scroll/right-click handlers on dice. Assumes
 * |delta| ≤ 6 (the UI only ever passes ±1).
 */
export function cycleFace(face: number, delta: number): number {
  return ((face - 1 + delta + 6) % 6) + 1;
}

/**
 * The maximum number of "extra Yahtzee" +100 bonuses the player could
 * legitimately have earned, given the current Yahtzee-box score and how many
 * turns have been played. Used to clamp the Extra-Yahtzees input.
 *
 * Rule: bonuses only accrue once the Yahtzee box holds 50 (i.e. the player
 * scored an actual Yahtzee earlier). Each subsequent turn could in principle
 * have rolled another Yahtzee, so the cap is `turnsPlayed - 1`.
 */
export function maxYahtzeeBonuses(
  yahtzeeScore: number | null,
  turnsPlayed: number,
): number {
  return yahtzeeScore === 50 ? Math.max(0, turnsPlayed - 1) : 0;
}

/**
 * Per-entry validation config. `values` is the set of legal scores for the
 * entry (e.g. {0, 5, 10, …, 30} for Three of a kind). `label` is the
 * human-readable hint shown when an out-of-range value is entered.
 */
export type AllowedScores = { values: Set<number>; label: string };

export type ParsedScoreInput = { value: number | null; error: string | null };

/**
 * Validate a raw scorecard input string against the allowed values for the
 * entry. Empty/whitespace-only input is treated as "not yet filled" (no
 * error). Any non-integer-looking string returns 'whole number'. When
 * `allowed` is null (wasm not yet loaded), we accept any integer so the user
 * can keep typing — the row will get re-validated once the solver is ready.
 */
export function parseScoreInput(
  raw: string,
  allowed: AllowedScores | null,
): ParsedScoreInput {
  const t = raw.trim();
  if (t === '') return { value: null, error: null };
  if (!/^-?\d+$/.test(t)) return { value: null, error: 'whole number' };
  const n = parseInt(t, 10);
  if (allowed === null) return { value: n, error: null };
  if (allowed.values.has(n)) return { value: n, error: null };
  return { value: null, error: allowed.label };
}

/**
 * Render an array of allowed numeric scores (sorted ascending, no duplicates)
 * as a short human hint shown next to a scorecard input. Three cases:
 *   - 6 or fewer values → comma-joined list ("0, 25").
 *   - More than 6 and contiguous → range ("0–30").
 *   - More than 6 and contiguous except for a leading 0 → "0 or 5–30"
 *     (covers categories like Three of a kind where the legal scores are 0
 *     plus a contiguous tail).
 *   - Otherwise → comma-joined list (the verbose fallback).
 */
export function formatAllowed(vals: number[]): string {
  if (vals.length <= 6) return vals.join(', ');
  const contiguous = (xs: number[]) =>
    xs.every((v, i) => i === 0 || v === xs[i - 1] + 1);
  if (contiguous(vals)) return `${vals[0]}–${vals[vals.length - 1]}`;
  if (vals[0] === 0) {
    const rest = vals.slice(1);
    if (contiguous(rest)) return `0 or ${rest[0]}–${rest[rest.length - 1]}`;
  }
  return vals.join(', ');
}
