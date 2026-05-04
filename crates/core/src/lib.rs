//! Yahtzee solver core: a backward-induction DP over scorecard states.
//!
//! Conceptually:
//!
//! - A [`State`] is `(filled-categories, yahtzee-bonus-eligible, upper-score-remaining)`,
//!   packed into a `usize` so we can index a flat `Array1<f32>` of state EVs.
//! - [`Scores::new`] fills that array level-by-level (from 13 categories filled
//!   down to 0). Each level only depends on strictly higher levels, so each
//!   level is computed in parallel via rayon.
//! - For any state, [`Scores::values`] walks the three-roll decision tree
//!   backward to produce an [`ExpectedValues`] view: per-keeper EVs on rolls 1
//!   and 2, per-entry EVs on roll 3, and the overall state value.
//!
//! Consumer-facing API (CLI, wasm, web) lives in [`recommend`] — it wraps the
//! raw types in serializable shapes and validates user input. Day-to-day, most
//! callers want [`recommend::recommend`] rather than the lower-level types
//! here.

// Anchor `blas-src` so the BLAS provider's symbols (e.g. `cblas_sdot`) survive
// `--as-needed` link-time DCE. Without this, ndarray's `.dot()` calls fail to
// link with `--features blas`. Has no runtime effect; pure link-time glue.
#[cfg(feature = "blas")]
extern crate blas_src;

use std::collections::HashMap;
use std::convert::From;
use std::fmt;
use std::fmt::Display;
use std::sync::LazyLock;

use bitflags::bitflags;
use ndarray::Zip;
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod recommend;
pub use recommend::{
    build_state, counts_to_faces, dice_to_counts, recommend, EntryRec, KeeperRec, Recommendation,
    StateInput,
};

#[cfg(test)]
mod proptests;

// 2^13 * 2 * 64
// 13 Entries, 1 bit for Yahtzee bonus eligibility, and 64 for upper score
const NUM_STATES: u32 = 1_048_576;

// Calculated empirically - many states can never be reached.
// Only referenced from `tests::test_valid_states`; kept here as the canonical
// source of truth for the reachable-state count.
#[cfg(test)]
const NUM_VALID_STATES: u32 = 536_448;

// d6
const NUM_DICE_FACES: u8 = 6;

// 5 dice are used in yahtzee
const NUM_DICE: u8 = 5;

// C(10, 5) + C(9, 4) + ... C(5, 0)
const NUM_KEEPERS: u16 = 462;

// C(10, 5)
const NUM_DICE_COMBINATIONS: u8 = 252;
const NUM_ENTRY_ACTIONS: u8 = 13;

pub const ENTRY_ACTIONS: [EntryAction; 13] = [
    EntryAction::ONE,
    EntryAction::TWO,
    EntryAction::THREE,
    EntryAction::FOUR,
    EntryAction::FIVE,
    EntryAction::SIX,
    EntryAction::THREE_OF_A_KIND,
    EntryAction::FOUR_OF_A_KIND,
    EntryAction::FULL_HOUSE,
    EntryAction::SMALL_STRAIGHT,
    EntryAction::LARGE_STRAIGHT,
    EntryAction::YAHTZEE,
    EntryAction::CHANCE,
];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiceCounts(pub [u8; NUM_DICE_FACES as usize]);

impl Display for DiceCounts {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut dice = Vec::new();
        for (idx, &count) in self.0.iter().enumerate() {
            for _i in 0..count {
                dice.push((1 + idx).to_string());
            }
        }
        while dice.len() < 5 {
            dice.push("-".to_string());
        }
        write!(f, "{}", dice.join(" "))
    }
}

// Global static variables. Initialize once, read-only from anywhere.
static YAHTZEE_DICE: LazyLock<Vec<Option<EntryAction>>> = LazyLock::new(math::yahtzee_dice);
static DICE_IDX_LOOKUP: LazyLock<HashMap<DiceCounts, usize>> = LazyLock::new(math::dice_idx_lookup);
static IDX_KEEPERS_LOOKUP: LazyLock<HashMap<usize, DiceCounts>> =
    LazyLock::new(math::idx_keepers_lookup);
static DICE_AND_ENTRY_SCORES: LazyLock<Array2<u8>> = LazyLock::new(math::dice_and_entry_scores);
static DICE_TO_ALLOWED_KEEPERS: LazyLock<Array2<f32>> = LazyLock::new(math::dice_to_keepers);
static KEEPERS_TO_DICE_PROBABILITIES: LazyLock<Array2<f32>> = LazyLock::new(math::keepers_to_dice);

mod math {
    use super::EntryAction;
    use super::*;
    /// Generates all dice combinations for a given number of dice
    pub fn dice_combinations(num_dice: u8) -> Vec<DiceCounts> {
        let mut dice = [0_u8; NUM_DICE_FACES as usize];
        dice[0] = num_dice;

        let mut dice_combinations = Vec::new();
        dice_combinations.push(DiceCounts(dice));

        // Continue until the last dice combination in lexicographic order is created
        while dice[NUM_DICE_FACES as usize - 1] != num_dice {
            // index of rightmost non-zero count
            let mut rightmost = 0;
            for (idx, count) in dice.iter().enumerate() {
                if *count > 0 {
                    rightmost = idx;
                }
            }

            // If possible, move one from to the right by one
            if rightmost + 1 < dice.len() {
                dice[rightmost] -= 1;
                dice[rightmost + 1] += 1;

            // Otherwise, go to the second rightmost count, move one of _it_ to the right by one.
            // Then, also take the rightmost count and dump all of them one past the second rightmost.
            } else {
                let mut second_rightmost = 0;
                for (idx, count) in dice.iter().enumerate() {
                    if *count > 0 && idx < rightmost {
                        second_rightmost = idx;
                    }
                }
                // Save the current count at rightmost, in case this in the current target
                // from second rightmost
                let target = second_rightmost + 1;
                let num_rightmost = dice[rightmost];

                // Move one from second rightmost
                dice[second_rightmost] -= 1;
                dice[target] += 1;

                // Move all from rightmost
                dice[target] += num_rightmost;
                dice[rightmost] -= num_rightmost;
            }
            dice_combinations.push(DiceCounts(dice));
        }
        dice_combinations
    }

    /// Generates a lookup from `DiceCounts` to index
    pub fn dice_idx_lookup() -> HashMap<DiceCounts, usize> {
        dice_combinations(NUM_DICE)
            .into_iter()
            .enumerate()
            .map(|(idx, dice)| (dice, idx))
            .collect()
    }

    pub fn idx_keepers_lookup() -> HashMap<usize, DiceCounts> {
        (0..=5).flat_map(dice_combinations).enumerate().collect()
    }

    /// Creates a vector specifying which `DiceCounts` are Yahtzees
    /// If it is a Yahtzee, specify the kind (ones, twos, etc)
    pub fn yahtzee_dice() -> Vec<Option<EntryAction>> {
        let mut yahtzees = vec![None; NUM_DICE_COMBINATIONS as usize];
        let dice_lookup = dice_idx_lookup();

        yahtzees[dice_lookup[&DiceCounts([5, 0, 0, 0, 0, 0])]] = Some(EntryAction::ONE);
        yahtzees[dice_lookup[&DiceCounts([0, 5, 0, 0, 0, 0])]] = Some(EntryAction::TWO);
        yahtzees[dice_lookup[&DiceCounts([0, 0, 5, 0, 0, 0])]] = Some(EntryAction::THREE);
        yahtzees[dice_lookup[&DiceCounts([0, 0, 0, 5, 0, 0])]] = Some(EntryAction::FOUR);
        yahtzees[dice_lookup[&DiceCounts([0, 0, 0, 0, 5, 0])]] = Some(EntryAction::FIVE);
        yahtzees[dice_lookup[&DiceCounts([0, 0, 0, 0, 0, 5])]] = Some(EntryAction::SIX);

        yahtzees
    }

    pub fn dice_and_entry_scores() -> Array2<u8> {
        let shape = (NUM_ENTRY_ACTIONS as usize, NUM_DICE_COMBINATIONS as usize);
        let mut scores = Array2::zeros(shape);

        let dice_combinations = dice_combinations(NUM_DICE);

        for (dice_idx, dice) in dice_combinations.into_iter().enumerate() {
            let dice = dice.0;
            let small_straight = dice[..4].iter().all(|&x| x > 0)
                || dice[1..5].iter().all(|&x| x > 0)
                || dice[2..6].iter().all(|&x| x > 0);
            for (action_idx, &action) in ENTRY_ACTIONS.iter().enumerate() {
                let score = match action {
                    EntryAction::ONE => dice[0],
                    EntryAction::TWO => 2 * dice[1],
                    EntryAction::THREE => 3 * dice[2],
                    EntryAction::FOUR => 4 * dice[3],
                    EntryAction::FIVE => 5 * dice[4],
                    EntryAction::SIX => 6 * dice[5],
                    EntryAction::THREE_OF_A_KIND if *dice.iter().max().unwrap() >= 3_u8 => dice
                        .iter()
                        .enumerate()
                        .map(|(idx, count)| count * (idx as u8 + 1))
                        .sum::<u8>(),
                    EntryAction::FOUR_OF_A_KIND if *dice.iter().max().unwrap() >= 4_u8 => dice
                        .iter()
                        .enumerate()
                        .map(|(idx, count)| count * (idx as u8 + 1))
                        .sum::<u8>(),
                    EntryAction::FULL_HOUSE
                        if *dice.iter().max().unwrap() == 3_u8
                            && *dice.iter().filter(|&&i| i != 3_u8).max().unwrap() == 2_u8 =>
                    {
                        25
                    }
                    EntryAction::SMALL_STRAIGHT if small_straight => 30,
                    EntryAction::LARGE_STRAIGHT
                        if dice == [1, 1, 1, 1, 1, 0] || dice == [0, 1, 1, 1, 1, 1] =>
                    {
                        40
                    }
                    EntryAction::YAHTZEE if *dice.iter().max().unwrap() == 5_u8 => 50,
                    EntryAction::CHANCE => dice
                        .iter()
                        .enumerate()
                        .map(|(idx, count)| count * (idx as u8 + 1))
                        .sum::<u8>(),
                    _ => 0,
                };
                scores[(action_idx, dice_idx)] = score;
            }
        }
        scores
    }
    // Matrix of 252x462 of allowed keepers from each dice roll
    pub fn dice_to_keepers() -> Array2<f32> {
        let shape = (NUM_DICE_COMBINATIONS as usize, NUM_KEEPERS as usize);
        let mut dice_to_keepers: Array2<f32> = Array2::ones(shape);

        let dice: Vec<DiceCounts> = dice_combinations(NUM_DICE);
        let keepers: Vec<DiceCounts> = (0..=5).flat_map(dice_combinations).collect();

        for (dice_idx, dice) in dice.iter().enumerate() {
            for (keeper_idx, keeper) in keepers.iter().enumerate() {
                for (die_count, keeper_die_count) in dice.0.iter().zip(keeper.0.iter()) {
                    // Invalid action - cannot legitimately have keeper
                    // if count is greater than the dice roll
                    if keeper_die_count > die_count {
                        dice_to_keepers[(dice_idx, keeper_idx)] = 0_f32;
                    }
                }
            }
        }
        dice_to_keepers
    }

    // Matrix of 462x252 of transition probabilities from Keepers to Dice
    pub fn keepers_to_dice() -> Array2<f32> {
        let dice_idx_lookup: HashMap<DiceCounts, usize> = dice_combinations(NUM_DICE)
            .into_iter()
            .enumerate()
            .map(|(idx, dice)| (dice, idx))
            .collect();

        let shape = (NUM_KEEPERS as usize, NUM_DICE_COMBINATIONS as usize);
        let mut keepers_to_dice = Array2::zeros(shape);

        // all possible dice combinations, from 0 dice to 5 dice
        let keepers = (0..=NUM_DICE).flat_map(dice_combinations);

        for (keeper_idx, keeper) in keepers.enumerate() {
            let num_keeper_dice = keeper.0.iter().sum::<u8>();
            let num_remaining_dice = 5_u8 - num_keeper_dice;

            for mut roll in dice_combinations(num_remaining_dice) {
                let roll_probability = dice_probability(&roll);
                // merge keeper with thrown roll
                for die_idx in 0..keeper.0.len() {
                    roll.0[die_idx] += keeper.0[die_idx];
                }
                let dice_idx = dice_idx_lookup[&roll];
                keepers_to_dice[(keeper_idx, dice_idx)] = roll_probability;
            }
        }

        keepers_to_dice
    }

    // Odds of rolling a particular dice combination
    // Can be computed from the values themselves - no need to consider permutations
    // Ignore clippy warnings because we know these values (explain better here)
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn dice_probability(dice: &DiceCounts) -> f32 {
        let total_dice: u8 = dice.0.iter().sum();
        let mut permutations_num = 1;
        let mut remaining_dice = total_dice;
        for &count in &dice.0 {
            permutations_num *= choose(remaining_dice as usize, count as usize);
            remaining_dice -= count;
        }

        let total_permutations = f32::from(NUM_DICE_FACES).powi(i32::from(total_dice));
        (permutations_num as f32) / total_permutations
    }

    // C(n, k) - does not need to be any more efficient than this
    fn choose(n: usize, k: usize) -> usize {
        let mut answer = 1;
        for num in (k + 1)..=n {
            answer *= num;
        }

        for num in 1..=(n - k) {
            answer /= num;
        }
        answer
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_dice_combinations() {
            let dice_combinations = dice_combinations(NUM_DICE);
            assert_eq!(dice_combinations.len(), NUM_DICE_COMBINATIONS as usize);
        }

        #[test]
        fn test_yahtzee_dice() {
            let yahtzee_dice = yahtzee_dice();

            assert_eq!(yahtzee_dice.len(), NUM_DICE_COMBINATIONS as usize);

            assert_eq!(
                yahtzee_dice.iter().filter(|d| d.is_some()).count(),
                NUM_DICE_FACES as usize
            );

            let expected = vec![
                ([5, 0, 0, 0, 0, 0], EntryAction::ONE),
                ([0, 5, 0, 0, 0, 0], EntryAction::TWO),
                ([0, 0, 5, 0, 0, 0], EntryAction::THREE),
                ([0, 0, 0, 5, 0, 0], EntryAction::FOUR),
                ([0, 0, 0, 0, 5, 0], EntryAction::FIVE),
                ([0, 0, 0, 0, 0, 5], EntryAction::SIX),
            ];

            for (dice, value) in expected.into_iter() {
                let idx = DICE_IDX_LOOKUP[&DiceCounts(dice)];
                assert_eq!(yahtzee_dice[idx], Some(value));
            }
        }
    }
}

// Packed bitset over NUM_STATES entries (16_384 u64 words = 128 KiB).
const VALID_STATES_WORDS: usize = (NUM_STATES as usize).div_ceil(64);

/// Per-keeper expected dice-value: for each keeper `k`, sum over re-roll
/// outcomes `d` of `P(k → d) * dice_values[d]`. This is exactly the GEMV
/// `KEEPERS_TO_DICE_PROBABILITIES (462x252) * dice_values (252)`, so we
/// dispatch through `ndarray::Array2::dot`. With the `blas` feature off,
/// that uses ndarray's default `matrixmultiply` backend (pure-Rust SIMD);
/// with `blas` on, it goes to `cblas_sgemv`. Called twice per state via
/// `Scores::values` / `Scores::state_value`, so it's the GEMV worth
/// optimizing.
fn keepers_from_dice(dice_values: &Array1<f32>) -> Array1<f32> {
    KEEPERS_TO_DICE_PROBABILITIES.dot(dice_values)
}

/// Per-dice "best keeper" value: for each dice combo `d`, max over keepers
/// valid for `d` of `keeper_values[k]`. (`DICE_TO_ALLOWED_KEEPERS` is an
/// indicator matrix: 1 where the keeper is achievable from `d`, 0 elsewhere,
/// so element-wise multiply masks the invalid keepers to 0 before the max.)
fn dice_from_keepers(keeper_values: &Array1<f32>) -> Array1<f32> {
    let mut dice: Array1<f32> = Array1::zeros(NUM_DICE_COMBINATIONS as usize);
    Zip::from(&mut dice)
        .and(DICE_TO_ALLOWED_KEEPERS.rows())
        .for_each(|val, mask_row| {
            *val = (&mask_row * keeper_values).fold(0_f32, |acc, elem| acc.max(*elem));
        });
    dice
}

/// Initial-roll distribution: row 0 of `KEEPERS_TO_DICE_PROBABILITIES` is the
/// "kept nothing → roll 5 fresh dice" row, which is the marginal we want for
/// turning per-dice values into the state EV.
fn first_roll_probabilities() -> ndarray::ArrayView1<'static, f32> {
    KEEPERS_TO_DICE_PROBABILITIES.index_axis(Axis(0), 0)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    state_scores: Array1<f32>,
    valid_states: Box<[u64]>,
}

impl Scores {
    // `Default::default()` traditionally implies a cheap construction;
    // `Scores::new()` runs the entire DP and takes ~22s in tests / minutes at
    // opt-level 0. Implementing `Default` would be misleading. The CLI / wasm
    // load a precomputed table from disk rather than calling `new()` at
    // startup; only the `build` subcommand and tests instantiate fresh.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Scores {
        let state_scores = Array1::zeros(NUM_STATES as usize);
        let valid_states = vec![0_u64; VALID_STATES_WORDS].into_boxed_slice();
        let mut scores = Scores {
            state_scores,
            valid_states,
        };
        scores.set_valid_states();
        scores.set_scores();
        scores
    }

    fn set_scores(&mut self) {
        // We go level-by-level, bottom to top, for correctness when multiprocessing
        for level in (0..NUM_ENTRY_ACTIONS).rev() {
            let mut new_scores = vec![0_f32; NUM_STATES as usize];

            new_scores
                .par_iter_mut()
                .enumerate()
                .for_each(|(state_idx, score)| {
                    let state: State = state_idx.into();
                    let state_level = state.level();
                    if self.valid_state(state) && state_level == level as usize {
                        *score = self.state_value(state);
                    }
                });

            // write this level's scores
            for (score_idx, score) in new_scores.into_iter().enumerate() {
                if score > 0_f32 {
                    self.state_scores[score_idx] = score;
                }
            }
        }
    }

    /// Overall expected final score from `state` under optimal play. This is
    /// the scalar the DP step (`set_scores`) needs and the same number that
    /// [`Scores::values`] returns as `.value` — but without materializing the
    /// per-roll intermediate arrays the recommendation API also needs. The
    /// underlying math is identical, so the perf delta is small (skips a
    /// struct construction); use this when you only need the EV.
    pub fn state_value(&self, state: State) -> f32 {
        let entry_actions = self.entry_actions_array(state);
        let third_dice = entry_actions.fold_axis(Axis(0), 0_f32, |acc, v| acc.max(*v));
        let second_keepers = keepers_from_dice(&third_dice);
        let second_dice = dice_from_keepers(&second_keepers);
        let first_keepers = keepers_from_dice(&second_dice);
        let first_dice = dice_from_keepers(&first_keepers);
        first_roll_probabilities().dot(&first_dice)
    }

    pub fn values(&self, state: State) -> ExpectedValues {
        // Overall EV of each (action, final dice) pair: `score + V(child)` for
        // valid actions, 0 for invalid. Read by the per-entry EV API.
        let entry_actions = self.entry_actions_array(state);

        // Value of each roll-3 dice combo: max action overall EV.
        let third_dice = entry_actions.fold_axis(Axis(0), 0_f32, |acc, v| acc.max(*v));

        // Value of each roll-2 keeper / roll-2 dice / roll-1 keeper / roll-1
        // dice — alternating "max over allowed keepers" and "expectation over
        // re-roll outcomes".
        let second_keepers = keepers_from_dice(&third_dice);
        let second_dice = dice_from_keepers(&second_keepers);
        let first_keepers = keepers_from_dice(&second_dice);
        let first_dice = dice_from_keepers(&first_keepers);

        // Marginal over the initial-roll distribution gives overall state EV.
        let value = first_roll_probabilities().dot(&first_dice);

        ExpectedValues {
            entry_actions,
            third_dice,
            second_keepers,
            second_dice,
            first_keepers,
            first_dice,
            value,
            state,
        }
    }

    fn entry_actions_array(&self, state: State) -> Array2<f32> {
        Array2::from_shape_fn((13, 252), |(action_idx, dice_idx)| {
            let action = EntryAction::from_bits(1 << action_idx).unwrap();
            if state.is_valid_action(action) {
                let (score, child) = state.score_and_child(action, dice_idx as u8);
                let child_idx: usize = child.into();
                score + self.state_scores[child_idx]
            } else {
                0_f32
            }
        })
    }

    fn set_valid_states(&mut self) {
        let mut words = vec![0_u64; VALID_STATES_WORDS];
        let default_idx: usize = State::default().into();
        words[default_idx / 64] |= 1_u64 << (default_idx % 64);

        for state_idx in 0..(NUM_STATES as usize) {
            if (words[state_idx / 64] >> (state_idx % 64)) & 1 == 1 {
                let elem: State = state_idx.into();
                for &action in &ENTRY_ACTIONS {
                    if elem.is_valid_action(action) {
                        for dice_idx in 0..NUM_DICE_COMBINATIONS {
                            let child = elem.child(action, dice_idx);
                            let idx: usize = child.into();
                            words[idx / 64] |= 1_u64 << (idx % 64);
                        }
                    }
                }
            }
        }

        self.valid_states = words.into_boxed_slice();
    }

    pub fn valid_state(&self, state: State) -> bool {
        let state_idx: usize = state.into();
        (self.valid_states[state_idx / 64] >> (state_idx % 64)) & 1 == 1
    }
}

bitflags! {
    #[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct EntryAction: u16 {
        const ONE             = 1;
        const TWO             = 1 << 1;
        const THREE           = 1 << 2;
        const FOUR            = 1 << 3;
        const FIVE            = 1 << 4;
        const SIX             = 1 << 5;
        const THREE_OF_A_KIND = 1 << 6;
        const FOUR_OF_A_KIND  = 1 << 7;
        const FULL_HOUSE      = 1 << 8;
        const SMALL_STRAIGHT  = 1 << 9;
        const LARGE_STRAIGHT  = 1 << 10;
        const YAHTZEE         = 1 << 11;
        const CHANCE          = 1 << 12;
    }
}

/// Look up the precomputed dice-combination index for `dice`.
///
/// Panics if `dice` is not a valid 5-dice multiset (sum of counts != 5).
/// In practice every caller goes through validated input —
/// [`crate::dice_to_counts`] for external callers, or solver-internal tables
/// keyed off `DICE_IDX_LOOKUP` itself — so this is unreachable in practice.
fn lookup_dice_idx(dice: &DiceCounts) -> usize {
    *DICE_IDX_LOOKUP
        .get(dice)
        .expect("DiceCounts is not a valid 5-dice multiset (caller must validate via dice_to_counts)")
}

/// Joker-rule fixed score for the lower-section categories.
///
/// Per the Yahtzee joker rule, when the Yahtzee box is already filled and the
/// player rolls a Yahtzee whose matching upper category is also filled, the
/// lower-section categories take guaranteed values: Full House = 25,
/// Small Straight = 30, Large Straight = 40. Returns `None` for every other
/// category — those score normally even under the joker rule.
fn joker_lower_score(action: EntryAction) -> Option<u8> {
    match action {
        EntryAction::FULL_HOUSE => Some(25),
        EntryAction::SMALL_STRAIGHT => Some(30),
        EntryAction::LARGE_STRAIGHT => Some(40),
        _ => None,
    }
}

// The set returned is state-independent: for upper categories, joker-enabled
// values (e.g. 25 in Fives from a Yahtzee joker) are only included if they're
// already reachable via normal scoring, so the set may not reflect every value
// achievable under the joker rule in a given state.
pub fn achievable_scores(entry_idx: usize) -> Vec<u8> {
    let action = ENTRY_ACTIONS[entry_idx];
    let row = DICE_AND_ENTRY_SCORES.row(action.as_idx());
    let mut values: std::collections::BTreeSet<u8> = row.iter().copied().collect();
    match action {
        EntryAction::FULL_HOUSE => {
            values.insert(25);
        }
        EntryAction::SMALL_STRAIGHT => {
            values.insert(30);
        }
        EntryAction::LARGE_STRAIGHT => {
            values.insert(40);
        }
        _ => {}
    }
    values.into_iter().collect()
}

impl EntryAction {
    /// Index into [`ENTRY_ACTIONS`] / row index into the precomputed score
    /// tables. Each canonical [`EntryAction`] is a single bit, so its index is
    /// the bit position. Calling this on a multi-bit set or `EntryAction::empty()`
    /// returns a meaningless number; callers always pass canonical values.
    pub fn as_idx(self) -> usize {
        self.bits().trailing_zeros() as usize
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct State {
    pub entries: EntryAction,
    pub yahtzee_bonus_eligible: bool,
    pub upper_score_remaining: u8,
}

impl Default for State {
    fn default() -> State {
        State {
            entries: EntryAction::empty(),
            yahtzee_bonus_eligible: false,
            upper_score_remaining: 63,
        }
    }
}

impl From<usize> for State {
    fn from(value: usize) -> Self {
        let value = value.min(NUM_STATES as usize - 1);
        State {
            entries: EntryAction::from_bits((value >> 7) as u16).unwrap(),
            yahtzee_bonus_eligible: (value >> 6) & 1 == 1,
            upper_score_remaining: (value & 0b111_111) as u8,
        }
    }
}

impl From<State> for usize {
    fn from(value: State) -> usize {
        let mut result = (value.entries.bits() as usize) << 7;
        if value.yahtzee_bonus_eligible {
            result |= 1 << 6;
        }
        result |= value.upper_score_remaining as usize;
        result
    }
}

impl State {
    pub fn level(self) -> usize {
        self.entries.bits().count_ones() as usize
    }

    pub fn child(self, action: EntryAction, dice_idx: u8) -> State {
        let mut child = self;

        // set action
        child.entries |= action;

        // set upper score
        let upper_actions = EntryAction::ONE
            | EntryAction::TWO
            | EntryAction::THREE
            | EntryAction::FOUR
            | EntryAction::FIVE
            | EntryAction::SIX;

        if upper_actions.contains(action) {
            let score = DICE_AND_ENTRY_SCORES[(action.as_idx(), dice_idx as usize)];
            child.upper_score_remaining = child.upper_score_remaining.saturating_sub(score);
        }

        // set yahtzee eligibility
        if action == EntryAction::YAHTZEE && YAHTZEE_DICE[dice_idx as usize].is_some() {
            child.yahtzee_bonus_eligible = true;
        }
        child
    }

    pub fn entry_score(self, action: EntryAction, dice: &DiceCounts) -> u8 {
        let dice_idx = lookup_dice_idx(dice);
        let mut score = DICE_AND_ENTRY_SCORES[(action.as_idx(), dice_idx)];
        // Joker rule: if the Yahtzee box is filled, the dice are a Yahtzee, and
        // the matching upper category is also filled, the lower-section
        // categories take their joker fixed scores.
        if self.entries.contains(EntryAction::YAHTZEE)
            && let Some(yahtzee_face) = YAHTZEE_DICE[dice_idx]
            && self.entries.contains(yahtzee_face)
            && let Some(joker) = joker_lower_score(action)
        {
            score = joker;
        }
        score
    }

    pub fn score_and_child(self, action_idx: EntryAction, dice_idx: u8) -> (f32, State) {
        let child = self.child(action_idx, dice_idx);

        let mut normal_score =
            f32::from(DICE_AND_ENTRY_SCORES[(action_idx.as_idx(), dice_idx as usize)]);
        let upper_bonus = if !self.upper_complete() && child.upper_complete() {
            35_f32
        } else {
            0_f32
        };

        let yahtzee_bonus =
            if YAHTZEE_DICE[dice_idx as usize].is_some() && self.yahtzee_bonus_eligible {
                100_f32
            } else {
                0_f32
            };

        // Joker rule: if the Yahtzee box is filled, the dice are a Yahtzee,
        // and the matching upper category is also filled, the lower-section
        // categories take their joker fixed scores.
        if self.entries.contains(EntryAction::YAHTZEE)
            && let Some(yahtzee_idx) = YAHTZEE_DICE[dice_idx as usize]
            && !self.is_valid_action(yahtzee_idx)
            && let Some(joker) = joker_lower_score(action_idx)
        {
            normal_score = f32::from(joker);
        }

        let score = normal_score + upper_bonus + yahtzee_bonus;
        (score, child)
    }

    fn is_valid_action(self, action_idx: EntryAction) -> bool {
        !self.entries.contains(action_idx)
    }

    fn upper_complete(self) -> bool {
        self.upper_score_remaining == 0
    }
}

/// Per-state precomputed expected-value tables. Build with [`Scores::values`].
///
/// Internals (the `*_dice` / `*_keepers` arrays and `entry_actions`) are
/// private; use the methods on `&self` (`first_keepers_score`,
/// `entries_with_turn_ev`, `this_turn_ev`, …) to read them.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedValues {
    entry_actions: Array2<f32>,
    third_dice: Array1<f32>,
    second_keepers: Array1<f32>,
    second_dice: Array1<f32>,
    first_keepers: Array1<f32>,
    first_dice: Array1<f32>,
    /// Overall expected final score from this state under optimal play. State
    /// only — independent of any current dice / roll.
    pub value: f32,
    state: State,
}

impl ExpectedValues {
    pub fn first_keepers_score(&self, dice: DiceCounts) -> Vec<(DiceCounts, f32)> {
        let mut result = Vec::new();
        let dice_idx = lookup_dice_idx(&dice);

        for keeper_idx in 0..(NUM_KEEPERS as usize) {
            if DICE_TO_ALLOWED_KEEPERS[(dice_idx, keeper_idx)] > 0.0 {
                result.push((
                    IDX_KEEPERS_LOOKUP.get(&keeper_idx).unwrap().clone(),
                    self.first_keepers[keeper_idx],
                ))
            }
        }

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    pub fn second_keepers_score(&self, dice: DiceCounts) -> Vec<(DiceCounts, f32)> {
        let mut result = Vec::new();
        let dice_idx = lookup_dice_idx(&dice);

        for keeper_idx in 0..(NUM_KEEPERS as usize) {
            if DICE_TO_ALLOWED_KEEPERS[(dice_idx, keeper_idx)] > 0.0 {
                let keeper = IDX_KEEPERS_LOOKUP.get(&keeper_idx).unwrap().clone();
                result.push((keeper, self.second_keepers[keeper_idx]));
            }
        }

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Expected points scored on this turn alone (including any +35 upper or
    /// +100 Yahtzee bonuses earned during it), given the current dice and roll
    /// number, assuming optimal play from here. Complements `value`, which is
    /// the EV of all remaining points (this turn + future turns).
    pub fn this_turn_ev(&self, dice: DiceCounts, roll: u8) -> f32 {
        let dice_idx = lookup_dice_idx(&dice);
        match roll {
            3 => self.this_turn_roll3_dice(dice_idx),
            2 => {
                let k = self.best_keeper(dice_idx, &self.second_keepers);
                self.this_turn_from_keeper_to_roll3(k)
            }
            1 => {
                let k = self.best_keeper(dice_idx, &self.first_keepers);
                self.this_turn_from_keeper_to_roll2(k)
            }
            _ => 0.0,
        }
    }

    fn this_turn_roll3_dice(&self, dice_idx: usize) -> f32 {
        let mut best_action_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for a in 0..(NUM_ENTRY_ACTIONS as usize) {
            let v = self.entry_actions[(a, dice_idx)];
            if v > best_val {
                best_val = v;
                best_action_idx = a;
            }
        }
        if best_val == f32::NEG_INFINITY {
            return 0.0;
        }
        self.state
            .score_and_child(ENTRY_ACTIONS[best_action_idx], dice_idx as u8)
            .0
    }

    fn best_keeper(&self, dice_idx: usize, keeper_values: &Array1<f32>) -> usize {
        let mut best_k = 0;
        let mut best_val = f32::NEG_INFINITY;
        for k in 0..(NUM_KEEPERS as usize) {
            if DICE_TO_ALLOWED_KEEPERS[(dice_idx, k)] > 0.0 && keeper_values[k] > best_val {
                best_val = keeper_values[k];
                best_k = k;
            }
        }
        best_k
    }

    fn this_turn_from_keeper_to_roll3(&self, keeper_idx: usize) -> f32 {
        let mut expected = 0.0_f32;
        for d in 0..(NUM_DICE_COMBINATIONS as usize) {
            let p = KEEPERS_TO_DICE_PROBABILITIES[(keeper_idx, d)];
            if p > 0.0 {
                expected += p * self.this_turn_roll3_dice(d);
            }
        }
        expected
    }

    fn this_turn_from_keeper_to_roll2(&self, keeper_idx: usize) -> f32 {
        let mut expected = 0.0_f32;
        for d in 0..(NUM_DICE_COMBINATIONS as usize) {
            let p = KEEPERS_TO_DICE_PROBABILITIES[(keeper_idx, d)];
            if p > 0.0 {
                let k2 = self.best_keeper(d, &self.second_keepers);
                expected += p * self.this_turn_from_keeper_to_roll3(k2);
            }
        }
        expected
    }

    /// Per-keeper `(overall_ev, this_turn_ev)` pairs for roll 1. `overall_ev`
    /// is the EV of playing the rest of the game optimally after keeping those
    /// dice; `this_turn_ev` is the portion scored on this turn only.
    pub fn first_keepers_with_turn_ev(
        &self,
        dice: DiceCounts,
    ) -> Vec<(DiceCounts, f32, f32)> {
        let dice_idx = lookup_dice_idx(&dice);
        let turn_ev_roll3 = self.turn_ev_by_roll3_dice();
        let turn_ev_roll2 = self.turn_ev_by_roll2_dice(&turn_ev_roll3);
        let mut result = Vec::new();
        for keeper_idx in 0..(NUM_KEEPERS as usize) {
            if DICE_TO_ALLOWED_KEEPERS[(dice_idx, keeper_idx)] > 0.0 {
                let keeper = IDX_KEEPERS_LOOKUP.get(&keeper_idx).unwrap().clone();
                let turn_ev: f32 = (0..NUM_DICE_COMBINATIONS as usize)
                    .map(|d| KEEPERS_TO_DICE_PROBABILITIES[(keeper_idx, d)] * turn_ev_roll2[d])
                    .sum();
                result.push((keeper, self.first_keepers[keeper_idx], turn_ev));
            }
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Per-keeper `(overall_ev, this_turn_ev)` pairs for roll 2.
    pub fn second_keepers_with_turn_ev(
        &self,
        dice: DiceCounts,
    ) -> Vec<(DiceCounts, f32, f32)> {
        let dice_idx = lookup_dice_idx(&dice);
        let turn_ev_roll3 = self.turn_ev_by_roll3_dice();
        let mut result = Vec::new();
        for keeper_idx in 0..(NUM_KEEPERS as usize) {
            if DICE_TO_ALLOWED_KEEPERS[(dice_idx, keeper_idx)] > 0.0 {
                let keeper = IDX_KEEPERS_LOOKUP.get(&keeper_idx).unwrap().clone();
                let turn_ev: f32 = (0..NUM_DICE_COMBINATIONS as usize)
                    .map(|d| KEEPERS_TO_DICE_PROBABILITIES[(keeper_idx, d)] * turn_ev_roll3[d])
                    .sum();
                result.push((keeper, self.second_keepers[keeper_idx], turn_ev));
            }
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Per-entry `(overall_ev, this_turn_ev)` pairs for roll 3. `this_turn_ev`
    /// is deterministic — the immediate score of the entry (including any +35
    /// upper or +100 Yahtzee bonus triggered by taking it this turn).
    pub fn entries_with_turn_ev(
        &self,
        dice: DiceCounts,
    ) -> Vec<(EntryAction, f32, f32)> {
        let mut result = Vec::new();
        let dice_idx = lookup_dice_idx(&dice);
        for (action_idx, &action) in ENTRY_ACTIONS.iter().enumerate() {
            if self.state.is_valid_action(action) {
                let overall = self.entry_actions[(action_idx, dice_idx)];
                let turn = self.state.score_and_child(action, dice_idx as u8).0;
                result.push((action, overall, turn));
            }
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    fn turn_ev_by_roll3_dice(&self) -> Vec<f32> {
        (0..NUM_DICE_COMBINATIONS as usize)
            .map(|d| self.this_turn_roll3_dice(d))
            .collect()
    }

    fn turn_ev_by_roll2_dice(&self, turn_ev_roll3: &[f32]) -> Vec<f32> {
        (0..NUM_DICE_COMBINATIONS as usize)
            .map(|d| {
                let k2 = self.best_keeper(d, &self.second_keepers);
                (0..NUM_DICE_COMBINATIONS as usize)
                    .map(|d3| KEEPERS_TO_DICE_PROBABILITIES[(k2, d3)] * turn_ev_roll3[d3])
                    .sum()
            })
            .collect()
    }

    pub fn entries_score(&self, dice: DiceCounts) -> Vec<(EntryAction, f32)> {
        let mut result = Vec::new();
        let dice_idx = lookup_dice_idx(&dice);
        for (action_idx, &action) in ENTRY_ACTIONS.iter().enumerate() {
            if self.state.is_valid_action(action) {
                result.push((action, self.entry_actions[(action_idx, dice_idx)]));
            }
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// `Scores::new()` is expensive even at `[profile.test] opt-level = 3`;
    /// share one instance across all tests in this binary.
    pub(crate) fn shared_scores() -> &'static Scores {
        static SCORES: OnceLock<Scores> = OnceLock::new();
        SCORES.get_or_init(Scores::new)
    }

    fn dc(counts: [u8; 6]) -> DiceCounts {
        DiceCounts(counts)
    }

    fn dc_idx(counts: [u8; 6]) -> u8 {
        lookup_dice_idx(&dc(counts)) as u8
    }

    #[test]
    fn test_valid_states() {
        let scores = shared_scores();
        let num_valid: usize = scores
            .valid_states
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();
        assert_eq!(num_valid, NUM_VALID_STATES as usize);
    }

    #[test]
    fn test_expected_value() {
        let default_idx: usize = State::default().into();
        let scores = shared_scores();
        let expected_value = scores.state_scores[default_idx];
        assert!((expected_value - 254.5896).abs() < 0.0001);
    }

    /// State for joker-rule tests: Yahtzee box filled and Threes filled, so a
    /// rolled three-yahtzee triggers the joker rule for any lower category.
    /// `yahtzee_bonus_eligible: false` keeps the +100 bonus out of the picture.
    fn joker_state_threes() -> State {
        State {
            entries: EntryAction::YAHTZEE | EntryAction::THREE,
            yahtzee_bonus_eligible: false,
            upper_score_remaining: 63 - 9,
        }
    }

    #[test]
    fn entry_score_joker_rule_fixes_lower_section() {
        let s = joker_state_threes();
        let three_yahtzee = dc([0, 0, 5, 0, 0, 0]);
        assert_eq!(s.entry_score(EntryAction::FULL_HOUSE, &three_yahtzee), 25);
        assert_eq!(s.entry_score(EntryAction::SMALL_STRAIGHT, &three_yahtzee), 30);
        assert_eq!(s.entry_score(EntryAction::LARGE_STRAIGHT, &three_yahtzee), 40);
        // Categories outside the joker override score normally.
        assert_eq!(s.entry_score(EntryAction::CHANCE, &three_yahtzee), 15);
        assert_eq!(s.entry_score(EntryAction::FOUR_OF_A_KIND, &three_yahtzee), 15);
    }

    #[test]
    fn entry_score_no_joker_when_dice_not_yahtzee() {
        let s = joker_state_threes();
        // Five distinct faces — large straight, not a yahtzee. Joker doesn't
        // apply, so Full House falls back to its normal 0.
        let large_straight = dc([1, 1, 1, 1, 1, 0]);
        assert_eq!(s.entry_score(EntryAction::FULL_HOUSE, &large_straight), 0);
        assert_eq!(s.entry_score(EntryAction::LARGE_STRAIGHT, &large_straight), 40);
    }

    #[test]
    fn entry_score_no_joker_when_yahtzee_box_unfilled() {
        // Yahtzee not filled → joker doesn't apply even for a yahtzee roll.
        let s = State {
            entries: EntryAction::THREE,
            yahtzee_bonus_eligible: false,
            upper_score_remaining: 63 - 9,
        };
        let three_yahtzee = dc([0, 0, 5, 0, 0, 0]);
        assert_eq!(s.entry_score(EntryAction::FULL_HOUSE, &three_yahtzee), 0);
        assert_eq!(s.entry_score(EntryAction::SMALL_STRAIGHT, &three_yahtzee), 0);
        assert_eq!(s.entry_score(EntryAction::LARGE_STRAIGHT, &three_yahtzee), 0);
    }

    #[test]
    fn entry_score_no_joker_when_matching_upper_unfilled() {
        // Yahtzee filled but Threes not filled — the joker rule requires the
        // matching upper face to also be filled.
        let s = State {
            entries: EntryAction::YAHTZEE,
            yahtzee_bonus_eligible: false,
            upper_score_remaining: 63,
        };
        let three_yahtzee = dc([0, 0, 5, 0, 0, 0]);
        assert_eq!(s.entry_score(EntryAction::FULL_HOUSE, &three_yahtzee), 0);
        assert_eq!(s.entry_score(EntryAction::SMALL_STRAIGHT, &three_yahtzee), 0);
    }

    #[test]
    fn score_and_child_joker_rule_fixes_lower_section() {
        let s = joker_state_threes();
        let idx = dc_idx([0, 0, 5, 0, 0, 0]);
        assert_eq!(s.score_and_child(EntryAction::FULL_HOUSE, idx).0, 25.0);
        assert_eq!(s.score_and_child(EntryAction::SMALL_STRAIGHT, idx).0, 30.0);
        assert_eq!(s.score_and_child(EntryAction::LARGE_STRAIGHT, idx).0, 40.0);
    }

    #[test]
    fn score_and_child_upper_bonus_fires_on_completion() {
        // 5 points away from the +35 bonus; scoring 30 sixes saturates
        // remaining to 0 and triggers the bonus.
        let s = State {
            entries: EntryAction::ONE
                | EntryAction::TWO
                | EntryAction::THREE
                | EntryAction::FOUR
                | EntryAction::FIVE,
            yahtzee_bonus_eligible: false,
            upper_score_remaining: 5,
        };
        let (total, child) = s.score_and_child(EntryAction::SIX, dc_idx([0, 0, 0, 0, 0, 5]));
        assert_eq!(total, 30.0 + 35.0);
        assert_eq!(child.upper_score_remaining, 0);
    }

    #[test]
    fn score_and_child_upper_bonus_does_not_double_fire() {
        // Upper section already complete (remaining = 0). Filling another
        // upper category must not re-award the bonus.
        let s = State {
            entries: EntryAction::ONE,
            yahtzee_bonus_eligible: false,
            upper_score_remaining: 0,
        };
        let (total, _) = s.score_and_child(EntryAction::SIX, dc_idx([0, 0, 0, 0, 0, 5]));
        assert_eq!(total, 30.0);
    }

    #[test]
    fn score_and_child_yahtzee_bonus_when_eligible() {
        // Eligible + dice are a yahtzee → +100, regardless of category. Use
        // CHANCE to avoid joker-rule interaction.
        let s = State {
            entries: EntryAction::YAHTZEE,
            yahtzee_bonus_eligible: true,
            upper_score_remaining: 63,
        };
        let (total, _) = s.score_and_child(EntryAction::CHANCE, dc_idx([0, 0, 0, 0, 5, 0]));
        assert_eq!(total, 25.0 + 100.0);
    }

    #[test]
    fn score_and_child_no_yahtzee_bonus_when_ineligible() {
        let s = State {
            entries: EntryAction::YAHTZEE,
            yahtzee_bonus_eligible: false,
            upper_score_remaining: 63,
        };
        let (total, _) = s.score_and_child(EntryAction::CHANCE, dc_idx([0, 0, 0, 0, 5, 0]));
        assert_eq!(total, 25.0);
    }

    #[test]
    fn score_and_child_no_yahtzee_bonus_on_non_yahtzee_dice() {
        let s = State {
            entries: EntryAction::YAHTZEE,
            yahtzee_bonus_eligible: true,
            upper_score_remaining: 63,
        };
        let (total, _) =
            s.score_and_child(EntryAction::LARGE_STRAIGHT, dc_idx([1, 1, 1, 1, 1, 0]));
        assert_eq!(total, 40.0);
    }

    #[test]
    fn achievable_scores_includes_joker_values() {
        // Indices match ENTRY_ACTIONS order.
        let fh = achievable_scores(8);
        assert!(fh.contains(&0));
        assert!(fh.contains(&25));
        let ss = achievable_scores(9);
        assert!(ss.contains(&0));
        assert!(ss.contains(&30));
        let ls = achievable_scores(10);
        assert!(ls.contains(&0));
        assert!(ls.contains(&40));
    }

    #[test]
    fn achievable_scores_upper_categories() {
        let ones = achievable_scores(0);
        for v in 0..=5 {
            assert!(ones.contains(&v), "ones missing {v}: {ones:?}");
        }
        let sixes = achievable_scores(5);
        for v in [0, 6, 12, 18, 24, 30] {
            assert!(sixes.contains(&v), "sixes missing {v}: {sixes:?}");
        }
    }

    #[test]
    fn entries_with_turn_ev_roll3_returns_immediate_score() {
        let scores = shared_scores();
        let values = scores.values(State::default());
        let dice = dice_to_counts(&[1, 2, 3, 4, 5]).unwrap();
        let entries = values.entries_with_turn_ev(dice);

        // LARGE_STRAIGHT on [1,2,3,4,5] from default state: immediate 40, no
        // bonuses (not yahtzee_bonus_eligible, upper section still incomplete).
        let (_, _ev, ls_turn) = entries
            .iter()
            .find(|(a, _, _)| *a == EntryAction::LARGE_STRAIGHT)
            .expect("LARGE_STRAIGHT must be a valid action from default state");
        assert_eq!(*ls_turn, 40.0);

        // Every entry's turn_ev is bounded above by overall ev (turn EV is a
        // component of total EV, and future EV is ≥ 0 for Yahtzee).
        for (action, ev, turn) in entries {
            assert!(turn <= ev + 1e-3, "{action:?}: turn_ev > ev: {turn} > {ev}");
            assert!(turn >= -1e-3, "{action:?}: turn_ev < 0: {turn}");
        }
    }

    #[test]
    fn keepers_with_turn_ev_invariants() {
        let scores = shared_scores();
        let values = scores.values(State::default());
        let dice = dice_to_counts(&[1, 2, 3, 4, 5]).unwrap();

        for (k, ev, turn) in values.first_keepers_with_turn_ev(dice.clone()) {
            assert!(turn <= ev + 1e-3, "roll 1 {k:?}: turn_ev > ev: {turn} > {ev}");
            assert!(turn >= -1e-3, "roll 1 {k:?}: turn_ev < 0: {turn}");
        }
        for (k, ev, turn) in values.second_keepers_with_turn_ev(dice) {
            assert!(turn <= ev + 1e-3, "roll 2 {k:?}: turn_ev > ev: {turn} > {ev}");
            assert!(turn >= -1e-3, "roll 2 {k:?}: turn_ev < 0: {turn}");
        }
    }

    #[test]
    fn recommend_rejects_invalid_roll() {
        let scores = shared_scores();
        let dice = dice_to_counts(&[1, 2, 3, 4, 5]).unwrap();
        assert!(scores.recommend(State::default(), dice.clone(), 0).is_err());
        assert!(scores.recommend(State::default(), dice, 4).is_err());
    }
}
