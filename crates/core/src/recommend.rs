//! Recommendation API: the "given a state and dice, what should I do" view of
//! the solver, in a shape that's friendly to serialize across an API boundary.
//!
//! Both the wasm crate (for the web UI) and the CLI consume this module. The
//! types here are intentionally `Serialize`-friendly (`entry` is an index into
//! [`ENTRY_ACTIONS`] rather than a non-serializable `EntryAction` bitflag, and
//! [`StateInput`] uses `[bool; 13]` instead of bitflags) so the same struct
//! flows out as JSON to a browser or a terminal user without re-translation.
//!
//! The canonical entry point is [`Scores::recommend`]; [`recommend`] is the
//! free-function version that takes the friendlier [`StateInput`] / `&[u8]`
//! dice representation and is what frontends typically call.

use serde::{Deserialize, Serialize};

use crate::{DiceCounts, EntryAction, Scores, State, ENTRY_ACTIONS};

/// Frontend-friendly representation of a [`State`]: per-category booleans
/// instead of bitflags. Round-trips cleanly through JSON / `serde_wasm_bindgen`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StateInput {
    /// Aligned with [`ENTRY_ACTIONS`]: `entries[i] == true` means category `i`
    /// has already been filled in.
    pub entries: [bool; 13],
    pub yahtzee_bonus_eligible: bool,
    /// 0..=63. Points still needed in the upper section to clear the +35 bonus.
    pub upper_score_remaining: u8,
}

/// One entry in the keepers ranking (rolls 1 and 2).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct KeeperRec {
    /// Faces of the kept dice, ascending. Length 0..=5.
    pub dice: Vec<u8>,
    /// Overall expected score from this state if we keep these dice and play
    /// optimally from here to the end of the game.
    pub ev: f32,
    /// Expected points scored *on this turn only* if we keep these dice and
    /// play optimally. Includes any +35 / +100 bonuses triggered this turn.
    pub turn_ev: f32,
}

/// One entry in the entries ranking. On rolls 1 and 2 this represents
/// "stop rolling and score the current dice in this category".
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EntryRec {
    /// Index into [`ENTRY_ACTIONS`] (0 = ones, …, 12 = chance).
    pub entry: u8,
    pub ev: f32,
    pub turn_ev: f32,
}

/// What the solver thinks you should do from here. See field docs for the
/// keepers/entries semantics on each roll.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Recommendation {
    /// Overall expected final score from this state under optimal play. State
    /// only — independent of the current dice / roll.
    pub value: f32,
    /// Per-keeper rankings on rolls 1 and 2; `None` on roll 3 (no rolls left).
    /// The all-5 keeper is excluded: "keep all 5 then re-roll some later" is
    /// strictly dominated by "re-roll those now". The useful interpretation of
    /// "keep everything" is "score these dice now", which the `entries` field
    /// captures with the actual score-now EV.
    pub keepers: Option<Vec<KeeperRec>>,
    /// Per-entry rankings, populated on every roll. On roll 3 these are the
    /// only options. On rolls 1 and 2 they represent "skip the remaining rolls
    /// and score this dice in this category", with `ev` / `turn_ev` reflecting
    /// that decision (no further rolls), so a UI can merge them with the
    /// keeper list and sort by overall EV.
    pub entries: Vec<EntryRec>,
}

impl Scores {
    /// Canonical recommendation API. Takes already-validated `State` /
    /// `DiceCounts` (use [`recommend`] if you have raw inputs to validate).
    /// `roll` must be 1, 2, or 3.
    pub fn recommend(
        &self,
        state: State,
        dice: DiceCounts,
        roll: u8,
    ) -> Result<Recommendation, &'static str> {
        if !(1..=3).contains(&roll) {
            return Err("roll must be 1, 2, or 3");
        }
        let values = self.values(state);

        let entries: Vec<EntryRec> = values
            .entries_with_turn_ev(dice.clone())
            .into_iter()
            .map(|(a, ev, turn_ev)| EntryRec {
                entry: action_idx(a),
                ev,
                turn_ev,
            })
            .collect();

        let keepers = match roll {
            1 => Some(
                values
                    .first_keepers_with_turn_ev(dice)
                    .into_iter()
                    .filter(|(d, _, _)| keeper_size(d) < 5)
                    .map(|(d, ev, turn_ev)| KeeperRec {
                        dice: counts_to_faces(&d),
                        ev,
                        turn_ev,
                    })
                    .collect(),
            ),
            2 => Some(
                values
                    .second_keepers_with_turn_ev(dice)
                    .into_iter()
                    .filter(|(d, _, _)| keeper_size(d) < 5)
                    .map(|(d, ev, turn_ev)| KeeperRec {
                        dice: counts_to_faces(&d),
                        ev,
                        turn_ev,
                    })
                    .collect(),
            ),
            3 => None,
            _ => unreachable!(),
        };

        Ok(Recommendation {
            value: values.value,
            keepers,
            entries,
        })
    }
}

/// Frontend-facing convenience wrapper around [`Scores::recommend`] that takes
/// the friendlier [`StateInput`] / raw-dice representation and returns the
/// same [`Recommendation`].
pub fn recommend(
    scores: &Scores,
    input: &StateInput,
    dice: &[u8],
    roll: u8,
) -> Result<Recommendation, String> {
    let state = build_state(input)?;
    let dice_counts = dice_to_counts(dice)?;
    scores
        .recommend(state, dice_counts, roll)
        .map_err(|e| e.to_string())
}

/// Convert a [`StateInput`] to a [`State`], validating `upper_score_remaining`.
pub fn build_state(input: &StateInput) -> Result<State, String> {
    let mut entries = EntryAction::empty();
    for (i, &on) in input.entries.iter().enumerate() {
        if on {
            entries |= ENTRY_ACTIONS[i];
        }
    }
    if input.upper_score_remaining > 63 {
        return Err("upper_score_remaining must be 0..=63".into());
    }
    Ok(State {
        entries,
        yahtzee_bonus_eligible: input.yahtzee_bonus_eligible,
        upper_score_remaining: input.upper_score_remaining,
    })
}

/// Build a [`DiceCounts`] from five raw face values (1..=6, in any order).
pub fn dice_to_counts(dice: &[u8]) -> Result<DiceCounts, String> {
    if dice.len() != 5 {
        return Err(format!("dice must have length 5, got {}", dice.len()));
    }
    let mut counts = [0_u8; 6];
    for &d in dice {
        if !(1..=6).contains(&d) {
            return Err(format!("each die must be 1..=6, got {}", d));
        }
        counts[(d - 1) as usize] += 1;
    }
    Ok(DiceCounts(counts))
}

/// Inverse of [`dice_to_counts`]: expand counts back into a sorted face list.
pub fn counts_to_faces(dc: &DiceCounts) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, &c) in dc.0.iter().enumerate() {
        for _ in 0..c {
            out.push((i + 1) as u8);
        }
    }
    out
}

fn keeper_size(dice: &DiceCounts) -> u16 {
    dice.0.iter().map(|&c| c as u16).sum()
}

/// Index of an `EntryAction` in [`ENTRY_ACTIONS`] (0..13). Panics on a
/// non-canonical action; in practice the core only ever hands us actions from
/// `ENTRY_ACTIONS` so this is unreachable.
pub(crate) fn action_idx(a: EntryAction) -> u8 {
    ENTRY_ACTIONS
        .iter()
        .position(|&x| x == a)
        .expect("unknown EntryAction") as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_state_rejects_upper_score_above_63() {
        let input = StateInput {
            entries: [false; 13],
            yahtzee_bonus_eligible: false,
            upper_score_remaining: 64,
        };
        assert!(build_state(&input).is_err());
    }

    #[test]
    fn build_state_accepts_boundary_values() {
        for r in [0, 1, 62, 63] {
            let input = StateInput {
                entries: [false; 13],
                yahtzee_bonus_eligible: false,
                upper_score_remaining: r,
            };
            let s = build_state(&input).expect("0..=63 is valid");
            assert_eq!(s.upper_score_remaining, r);
        }
    }

    #[test]
    fn build_state_translates_entries_and_flags() {
        let mut entries = [false; 13];
        entries[0] = true; // ONES
        entries[11] = true; // YAHTZEE
        let input = StateInput {
            entries,
            yahtzee_bonus_eligible: true,
            upper_score_remaining: 60,
        };
        let s = build_state(&input).unwrap();
        assert!(s.entries.contains(EntryAction::ONE));
        assert!(s.entries.contains(EntryAction::YAHTZEE));
        assert!(!s.entries.contains(EntryAction::TWO));
        assert!(s.yahtzee_bonus_eligible);
        assert_eq!(s.upper_score_remaining, 60);
    }

    #[test]
    fn dice_to_counts_validates_length() {
        assert!(dice_to_counts(&[1, 2, 3, 4]).is_err());
        assert!(dice_to_counts(&[1, 2, 3, 4, 5, 6]).is_err());
        assert!(dice_to_counts(&[]).is_err());
    }

    #[test]
    fn dice_to_counts_validates_face() {
        assert!(dice_to_counts(&[0, 1, 2, 3, 4]).is_err());
        assert!(dice_to_counts(&[1, 2, 3, 4, 7]).is_err());
    }

    #[test]
    fn dice_to_counts_round_trips_through_counts_to_faces() {
        let raw = [1_u8, 1, 3, 5, 6];
        let dc = dice_to_counts(&raw).unwrap();
        let faces = counts_to_faces(&dc);
        // counts_to_faces emits ascending; raw happens to already be sorted.
        assert_eq!(faces, raw.to_vec());
    }
}
