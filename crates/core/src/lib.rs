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

// 2^13 * 2 * 64
// 13 Entries, 1 bit for Yahtzee bonus eligibility, and 64 for upper score
const NUM_STATES: u32 = 1_048_576;

// Calculated empirically - many states can never be reached
#[allow(dead_code)]
const NUM_VALID_STATES: u32 = 536_448;
#[warn(dead_code)]

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
static KEEPER_IDX_LOOKUP: LazyLock<HashMap<DiceCounts, usize>> =
    LazyLock::new(math::keepers_idx_lookup);
static IDX_DICE_LOOKUP: LazyLock<HashMap<usize, DiceCounts>> =
    LazyLock::new(math::idx_dice_lookup);
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
        dice_combinations(NUM_DICE as u8)
            .into_iter()
            .enumerate()
            .map(|(idx, dice)| (dice, idx))
            .collect()
    }

    /// Generates a lookup from `DiceCounts` to index, for keepers
    pub fn keepers_idx_lookup() -> HashMap<DiceCounts, usize> {
        (0..=5)
            .flat_map(dice_combinations)
            .into_iter()
            .enumerate()
            .map(|(idx, dice)| (dice, idx))
            .collect()
    }
    pub fn idx_dice_lookup() -> HashMap<usize, DiceCounts> {
        dice_combinations(NUM_DICE as u8)
            .into_iter()
            .enumerate()
            .collect()
    }
    pub fn idx_keepers_lookup() -> HashMap<usize, DiceCounts> {
        (0..=5)
            .flat_map(dice_combinations)
            .into_iter()
            .enumerate()
            .collect()
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
const VALID_STATES_WORDS: usize = (NUM_STATES as usize + 63) / 64;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    state_scores: Array1<f32>,
    valid_states: Box<[u64]>,
}

impl Scores {
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
                        *score = self.values(state).value;
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

    pub fn values(&self, state: State) -> ExpectedValues {
        let mut expected_values = ExpectedValues::default();
        expected_values.state = state;

        // values of each entry for each final dice roll
        let entry_actions = Array2::from_shape_fn((13, 252), |(action_idx, dice_idx)| {
            let action = EntryAction::from_bits(1 << action_idx).unwrap();
            if state.is_valid_action(action) {
                let (score, child) = state.score_and_child(action, dice_idx as u8);
                let child_idx: usize = child.into();
                score + self.state_scores[child_idx]
            } else {
                0_f32
            }
        });

        expected_values.entry_actions = entry_actions.clone();

        // value of each final dice roll
        let third_dice = entry_actions.fold_axis(Axis(0), 0_f32, |acc, value| acc.max(*value));

        expected_values.third_dice = third_dice.clone();

        let mut second_keepers: Array1<f32> = Array1::zeros(462);

        Zip::from(&mut second_keepers)
            .and(KEEPERS_TO_DICE_PROBABILITIES.rows())
            .for_each(|avg, act| {
                *avg = (&act * &third_dice).sum();
            });
        expected_values.second_keepers = second_keepers.clone();

        let mut second_dice = third_dice;
        Zip::from(&mut second_dice)
            .and(DICE_TO_ALLOWED_KEEPERS.rows())
            .for_each(|val, dice_to_action| {
                *val = (&dice_to_action * &second_keepers).fold(0_f32, |acc, elem| acc.max(*elem));
            });

        expected_values.second_dice = second_dice.clone();

        let mut first_keepers = second_keepers;

        Zip::from(&mut first_keepers)
            .and(KEEPERS_TO_DICE_PROBABILITIES.rows())
            .for_each(|avg, act| {
                *avg = (&act * &second_dice).sum();
            });

        expected_values.first_keepers = first_keepers.clone();

        let mut first_dice = second_dice;
        Zip::from(&mut first_dice)
            .and(DICE_TO_ALLOWED_KEEPERS.rows())
            .for_each(|val, dice_to_action| {
                *val = (&dice_to_action * &first_keepers).fold(0_f32, |acc, elem| acc.max(*elem));
            });

        expected_values.first_dice = first_dice.clone();

        let first_roll_probabilities = KEEPERS_TO_DICE_PROBABILITIES.index_axis(Axis(0), 0);
        expected_values.value = first_roll_probabilities.dot(&first_dice);
        expected_values
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
    pub fn as_idx(self) -> usize {
        let mut idx = 0;
        let mut value = self.bits();
        while value > 1 {
            value >>= 1;
            idx += 1;
        }
        idx
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
        result as usize
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
            child.upper_score_remaining = child.upper_score_remaining.saturating_sub(score as u8);
        }

        // set yahtzee eligibility
        if action == EntryAction::YAHTZEE && YAHTZEE_DICE[dice_idx as usize].is_some() {
            child.yahtzee_bonus_eligible = true;
        }
        child
    }

    pub fn entry_score(self, action: EntryAction, dice: &DiceCounts) -> u8 {
        let dice_idx = *DICE_IDX_LOOKUP.get(dice).unwrap();
        let mut score = DICE_AND_ENTRY_SCORES[(action.as_idx(), dice_idx)];
        if self.entries.contains(EntryAction::YAHTZEE) {
            if let Some(yahtzee_face) = YAHTZEE_DICE[dice_idx] {
                if self.entries.contains(yahtzee_face) {
                    score = match action {
                        EntryAction::FULL_HOUSE => 25,
                        EntryAction::SMALL_STRAIGHT => 30,
                        EntryAction::LARGE_STRAIGHT => 40,
                        _ => score,
                    };
                }
            }
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

        // joker rule
        // yahtzee box filled
        if self.entries.contains(EntryAction::YAHTZEE) {
            // dice is yahtzee
            if let Some(yahtzee_idx) = YAHTZEE_DICE[dice_idx as usize] {
                // upper entry filled
                if !self.is_valid_action(yahtzee_idx) {
                    if action_idx == EntryAction::FULL_HOUSE {
                        normal_score = 25_f32;
                    } else if action_idx == EntryAction::SMALL_STRAIGHT {
                        normal_score = 30_f32;
                    } else if action_idx == EntryAction::LARGE_STRAIGHT {
                        normal_score = 40_f32;
                    }
                }
            }
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

#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedValues {
    entry_actions: Array2<f32>,
    third_dice: Array1<f32>,
    second_keepers: Array1<f32>,
    second_dice: Array1<f32>,
    first_keepers: Array1<f32>,
    first_dice: Array1<f32>,
    pub value: f32,
    state: State,
}

impl Default for ExpectedValues {
    fn default() -> ExpectedValues {
        ExpectedValues {
            entry_actions: Array2::zeros((0, 0)),
            third_dice: Array1::zeros(0),
            second_keepers: Array1::zeros(0),
            second_dice: Array1::zeros(0),
            first_keepers: Array1::zeros(0),
            first_dice: Array1::zeros(0),
            value: 0_f32,
            state: State::default(),
        }
    }
}

impl ExpectedValues {
    pub fn first_keepers_score(&self, dice: DiceCounts) -> Vec<(DiceCounts, f32)> {
        let mut result = Vec::new();
        let dice_idx = *DICE_IDX_LOOKUP.get(&dice).unwrap();

        for keeper_idx in 0..(NUM_KEEPERS as usize) {
            if DICE_TO_ALLOWED_KEEPERS[(dice_idx, keeper_idx)] > 0.0 {
                result.push((
                    IDX_KEEPERS_LOOKUP.get(&keeper_idx).unwrap().clone(),
                    self.first_keepers[keeper_idx],
                ))
            }
        }

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }

    pub fn second_keepers_score(&self, dice: DiceCounts) -> Vec<(DiceCounts, f32)> {
        let mut result = Vec::new();
        let dice_idx = *DICE_IDX_LOOKUP.get(&dice).unwrap();

        for keeper_idx in 0..(NUM_KEEPERS as usize) {
            if DICE_TO_ALLOWED_KEEPERS[(dice_idx, keeper_idx)] > 0.0 {
                let keeper = IDX_KEEPERS_LOOKUP.get(&keeper_idx).unwrap().clone();
                result.push((keeper, self.second_keepers[keeper_idx]));
            }
        }

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }

    /// Expected points scored on this turn alone (including any +35 upper or
    /// +100 Yahtzee bonuses earned during it), given the current dice and roll
    /// number, assuming optimal play from here. Complements `value`, which is
    /// the EV of all remaining points (this turn + future turns).
    pub fn this_turn_ev(&self, dice: DiceCounts, roll: u8) -> f32 {
        let dice_idx = *DICE_IDX_LOOKUP.get(&dice).unwrap();
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
        let dice_idx = *DICE_IDX_LOOKUP.get(&dice).unwrap();
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
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }

    /// Per-keeper `(overall_ev, this_turn_ev)` pairs for roll 2.
    pub fn second_keepers_with_turn_ev(
        &self,
        dice: DiceCounts,
    ) -> Vec<(DiceCounts, f32, f32)> {
        let dice_idx = *DICE_IDX_LOOKUP.get(&dice).unwrap();
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
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
        let dice_idx = *DICE_IDX_LOOKUP.get(&dice).unwrap();
        for (action_idx, &action) in ENTRY_ACTIONS.iter().enumerate() {
            if self.state.is_valid_action(action) {
                let overall = self.entry_actions[(action_idx, dice_idx)];
                let turn = self.state.score_and_child(action, dice_idx as u8).0;
                result.push((action, overall, turn));
            }
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
        let dice_idx = *DICE_IDX_LOOKUP.get(&dice).unwrap();
        for (action_idx, &action) in ENTRY_ACTIONS.iter().enumerate() {
            if self.state.is_valid_action(action) {
                result.push((action, self.entry_actions[(action_idx, dice_idx)]));
            }
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_valid_states() {
        let scores = Scores::new();
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
        let scores = Scores::new();
        let expected_value = scores.state_scores[default_idx];
        assert!((expected_value - 254.5896).abs() < 0.0001);
    }
}
