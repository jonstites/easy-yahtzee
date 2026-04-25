// Wire types that mirror `crates/wasm/src/lib.rs` and the UI's view of them
// (`Choice`), plus the merge-and-sort that turns a `Recommendation` into the
// flat list the UI renders. Kept pure so it can be unit-tested in isolation.

// --- Wire types from yahtzee-wasm. Keep in sync with crates/wasm/src/lib.rs.
//
// EV semantics (all in expected points):
//   - Recommendation.value: EV from the current state under optimal play —
//     i.e., points you expect to score from now until the game ends.
//   - KeeperRec.ev / EntryRec.ev: same metric but conditioned on taking that
//     specific choice (then continuing optimally).
//   - turn_ev: expected points scored on this turn alone if you take this
//     choice (then continue optimally), incl. any +35 / +100 bonuses
//     triggered this turn. Roll-3 entry rows have deterministic turn_ev
//     (the immediate score of the entry).

export type KeeperRec = { dice: number[]; ev: number; turn_ev: number };
export type EntryRec = { entry: number; ev: number; turn_ev: number };
export type Recommendation = {
  value: number;
  keepers: KeeperRec[] | null;
  entries: EntryRec[];
};

// --- UI-side discriminated union. Kinds are mutually exclusive and carry
// only the data they need:
//   - 'reroll': re-roll all dice (keep nothing). Only on rolls 1 and 2.
//   - 'keep':   keep these specific faces, re-roll the rest. dice.length is
//               in 1..=4 (the all-5 keeper is excluded by the wasm).
//   - 'score':  write to scorecard row entryIdx and end the turn.
// ev / turn_ev have the same meaning as on the wire types above.

export type Choice =
  | { kind: 'reroll'; ev: number; turn_ev: number }
  | { kind: 'keep'; dice: number[]; ev: number; turn_ev: number }
  | { kind: 'score'; entryIdx: number; ev: number; turn_ev: number };

/**
 * Flatten a `Recommendation` into a sorted `Choice[]` for the UI.
 *
 * On rolls 1 and 2 (keepers present) we merge keep choices with score-now
 * choices, sort by overall EV descending, and slice to `limit`. Empty-dice
 * keepers render as `'reroll'`. On roll 3 (keepers null) we just pass through
 * the entries — they're already sorted by the wasm.
 *
 * `null`/`undefined` rec → `[]` so callers don't have to nullcheck.
 */
export function buildChoices(
  rec: Recommendation | null | undefined,
  limit = 8,
): Choice[] {
  if (!rec) return [];
  const entryChoices: Choice[] = rec.entries.map(e => ({
    kind: 'score' as const,
    entryIdx: e.entry,
    turn_ev: e.turn_ev,
    ev: e.ev,
  }));
  if (!rec.keepers) return entryChoices;
  const keeperChoices: Choice[] = rec.keepers.map(k =>
    k.dice.length === 0
      ? { kind: 'reroll' as const, ev: k.ev, turn_ev: k.turn_ev }
      : { kind: 'keep' as const, dice: k.dice, ev: k.ev, turn_ev: k.turn_ev },
  );
  const merged = [...keeperChoices, ...entryChoices];
  merged.sort((a, b) => b.ev - a.ev);
  return merged.slice(0, limit);
}
