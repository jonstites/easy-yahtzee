use std::collections::HashMap;
use std::convert::From;

extern crate ndarray;
extern crate rayon;

use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;

#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate bitflags;

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
pub struct DiceCounts([u8; NUM_DICE_FACES as usize]);

lazy_static! {
    // Global static variables. Initialize once, read-only from anywhere.
    static ref YAHTZEE_DICE: Vec<Option<EntryAction>> = math::yahtzee_dice();
    static ref DICE_IDX_LOOKUP: HashMap<DiceCounts, usize> = math::dice_idx_lookup();
    static ref IDX_DICE_LOOKUP: HashMap<usize, DiceCounts> = math::idx_dice_lookup();
    static ref DICE_AND_ENTRY_SCORES: Array2<u8> = math::dice_and_entry_scores();
    static ref DICE_TO_ALLOWED_KEEPERS: Array2<f32> = math::dice_to_keepers();
    static ref KEEPERS_TO_DICE_PROBABILITIES: Array2<f32> = math::keepers_to_dice();
}

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

    pub fn idx_dice_lookup() -> HashMap<usize, DiceCounts> {
        dice_combinations(NUM_DICE as u8)
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

#[derive(Debug, Clone, PartialEq)]
pub struct Scores {
    pub state_scores: Array1<f32>,
}

impl Default for Scores {
    fn default() -> Scores {
        let state_scores = Array1::zeros(NUM_STATES as usize);

        Scores { state_scores }
    }
}

impl Scores {
    pub fn new() -> Scores {
        Scores::default()
    }

    pub fn build(&mut self) {
        let valid_states = self.valid_states();

        // We go level-by-level, bottom to top, for correctness when multiprocessing
        for level in (0..NUM_ENTRY_ACTIONS).rev() {
            let mut new_scores = vec![0_f32; NUM_STATES as usize];

            new_scores
                .par_iter_mut()
                .enumerate()
                .for_each(|(state_idx, score)| {
                    let state: State = state_idx.into();
                    let state_level = state.level();
                    if valid_states[state_idx] && state_level == level as usize {
                        *score = self.widget(state);
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

    // Todo. Required for basic library functionality.
    pub fn values(&self, _state: State) -> ExpectedValues {
        ExpectedValues {
            entry_actions: Vec::new(),
            third_dice: Vec::new(),
            second_keepers: Vec::new(),
            second_dice: Vec::new(),
            first_keepers: Vec::new(),
            first_dice: Vec::new(),
            value: 0_f32,
        }
    }

    pub fn widget(&self, state: State) -> f32 {
        // values of each entry for each final dice roll
        let entry_scores = Array2::from_shape_fn((13, 252), |(action_idx, dice_idx)| {
            let action = EntryAction::from_bits(1 << action_idx).unwrap();
            if state.is_valid_action(action) {
                let (score, child) = state.score_and_child(action, dice_idx as u8);
                let child_idx: usize = child.into();
                score + self.state_scores[child_idx]
            } else {
                0_f32
            }
        });
        // value of each final dice roll
        let max_action_values =
            entry_scores.fold_axis(Axis(0), 0_f32, |acc, value| acc.max(*value));

        let mut avg_action_values: Array1<f32> = Array1::zeros(462);

        Zip::from(&mut avg_action_values)
            .and(KEEPERS_TO_DICE_PROBABILITIES.genrows())
            .apply(|avg, act| {
                *avg = (&act * &max_action_values).sum();
            });

        let mut dice_values: Array1<f32> = Array1::zeros(252);
        Zip::from(&mut dice_values)
            .and(DICE_TO_ALLOWED_KEEPERS.genrows())
            .apply(|val, dice_to_action| {
                *val =
                    (&dice_to_action * &avg_action_values).fold(0_f32, |acc, elem| acc.max(*elem));
            });

        Zip::from(&mut avg_action_values)
            .and(KEEPERS_TO_DICE_PROBABILITIES.genrows())
            .apply(|avg, act| {
                *avg = (&act * &dice_values).sum();
            });

        Zip::from(&mut dice_values)
            .and(DICE_TO_ALLOWED_KEEPERS.genrows())
            .apply(|val, dice_to_action| {
                *val =
                    (&dice_to_action * &avg_action_values).fold(0_f32, |acc, elem| acc.max(*elem));
            });

        let first_roll = KEEPERS_TO_DICE_PROBABILITIES.index_axis(Axis(0), 0);
        first_roll.dot(&dice_values)
    }

    // this can be rewritten as merely iterating in order over states
    // no queue or stack required
    fn valid_states(&self) -> Vec<bool> {
        let mut valid_markers = vec![false; NUM_STATES as usize];
        let default_idx: usize = State::default().into();
        valid_markers[default_idx] = true;

        for state_idx in 0..(NUM_STATES as usize) {
            let elem: State = state_idx.into();
            if valid_markers[state_idx] {
                for &action in &ENTRY_ACTIONS {
                    if elem.is_valid_action(action) {
                        for dice_idx in 0..NUM_DICE_COMBINATIONS {
                            let child = elem.child(action, dice_idx);
                            let idx: usize = child.into();
                            valid_markers[idx] = true;
                        }
                    }
                }
            }
        }

        valid_markers
    }
}

bitflags! {
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
    entries: EntryAction,
    yahtzee_bonus_eligible: bool,
    upper_score_remaining: u8,
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
    entry_actions: Vec<Option<f32>>,
    third_dice: Vec<f32>,
    second_keepers: Vec<f32>,
    second_dice: Vec<f32>,
    first_keepers: Vec<f32>,
    first_dice: Vec<f32>,
    value: f32,
}

pub fn score() -> ExpectedValues {
    let mut score = Scores::new();
    score.build();
    score.values(State::default())
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_valid_states() {
        let scores = Scores::new();
        let states = scores.valid_states();
        let num_valid = states.iter().filter(|x| **x).count();
        assert_eq!(num_valid, NUM_VALID_STATES as usize);
    }

    #[test]
    fn test_expected_value() {
        let default_idx: usize = State::default().into();
        let mut scores = Scores::new();
        scores.build();
        let expected_value = scores.state_scores[default_idx];
        assert!((expected_value - 254.5896).abs() < 0.0001);
    }
}
