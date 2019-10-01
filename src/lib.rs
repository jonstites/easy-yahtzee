use std::convert::From;
use std::fmt;
use std::collections::{HashMap, VecDeque};

extern crate ndarray;
extern crate rayon;

use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;


#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate bitflags;


// 2^13 * 2 * 64
// 13 Entries, 1 bit for Yahtzee bonus eligibility, and 64 for upper score
const NUM_STATES: usize = 1048576;

// Calculated empirically - many states can never be reached
const NUM_VALID_STATES: usize = 536448;

// d6
const NUM_DICE_FACES: usize = 6;

// 5 dice are used in yahtzee
const NUM_DICE: usize = 5;

// C(10, 5) + C(9, 4) + ... C(5, 0)
const NUM_KEEPERS: usize = 462;

// C(10, 5)
const NUM_DICE_COMBINATIONS: usize = 252;
const NUM_ENTRY_ACTIONS: usize = 13;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DiceCounts([usize; NUM_DICE_FACES]);


lazy_static! {
    // Global static variables. Initialize once, read-only from anywhere.
    static ref YAHTZEE_DICE: Vec<Option<EntryAction>> = math::yahtzee_dice();
    static ref DICE_IDX_LOOKUP: HashMap<DiceCounts, usize> = math::dice_idx_lookup();
    static ref DICE_AND_ENTRY_SCORES: Array2<usize> = math::dice_and_entry_scores();
    static ref DICE_TO_ALLOWED_KEEPERS: Array2<f32> = math::dice_to_keepers();
    static ref KEEPERS_TO_DICE_PROBABILITIES: Array2<f32> = math::keepers_to_dice();
}

mod math {
    use super::*;

    /// Generates all dice combinations for a given number of dice
    pub fn dice_combinations(num_dice: usize) -> Vec<DiceCounts> {
        let mut dice = [0; NUM_DICE_FACES];
        dice[0] = num_dice;

        let mut dice_combinations = Vec::new();
        dice_combinations.push(DiceCounts(dice));

        // Continue until the last dice combination in lexicographic order is created
        while dice[NUM_DICE_FACES - 1] != num_dice {

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

    /// Generates a lookup from DiceCounts to index
    pub fn dice_idx_lookup() -> HashMap<DiceCounts, usize> {
        dice_combinations(NUM_DICE)
            .into_iter()
            .enumerate()
            .map(|(idx, dice)| (dice, idx))
            .collect()
    }        

    /// Creates a vector specifying which DiceCounts are Yahtzees
    /// If it is a Yahtzee, specify the kind (ones, twos, etc)
    pub fn yahtzee_dice() -> Vec<Option<EntryAction>> {
        let mut yahtzees = vec![None; NUM_DICE_COMBINATIONS];
        let dice_lookup = dice_idx_lookup();

        yahtzees[dice_lookup[&DiceCounts([5, 0, 0, 0, 0, 0])]] = Some(EntryAction::ONE);
        yahtzees[dice_lookup[&DiceCounts([0, 5, 0, 0, 0, 0])]] = Some(EntryAction::TWO);
        yahtzees[dice_lookup[&DiceCounts([0, 0, 5, 0, 0, 0])]] = Some(EntryAction::THREE);
        yahtzees[dice_lookup[&DiceCounts([0, 0, 0, 5, 0, 0])]] = Some(EntryAction::FOUR);
        yahtzees[dice_lookup[&DiceCounts([0, 0, 0, 0, 5, 0])]] = Some(EntryAction::FIVE);
        yahtzees[dice_lookup[&DiceCounts([0, 0, 0, 0, 0, 5])]] = Some(EntryAction::SIX);

        yahtzees
    }

    pub fn dice_and_entry_scores() -> Array2<usize> {
        let shape = (NUM_ENTRY_ACTIONS, NUM_DICE_COMBINATIONS);
        let mut scores = Array2::zeros(shape);
        
        let dice_combinations = dice_combinations(NUM_DICE);

        for (dice_idx, dice) in dice_combinations.into_iter().enumerate() {
            let dice = dice.0;
            let small_straight = dice[..4].iter().all(|&x| x > 0) || dice[1..5].iter().all(|&x| x > 0) || dice[2..6].iter().all(|&x| x > 0) ;
            for action in 0..13 {
                let score = match action {
                    idx if idx < 6 => (dice[action] * (action + 1)),
                    6 if *dice.iter().max().unwrap() >= 3_usize => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>(),
                    7 if *dice.iter().max().unwrap() >= 4_usize => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>(),
                    8 if *dice.iter().max().unwrap() == 3_usize && *dice.iter().filter(|&&i| i != 3_usize).max().unwrap() == 2_usize => 25,
                    9 if small_straight => 30,
                    10 if dice == [1, 1, 1, 1, 1, 0] || dice == [0, 1, 1, 1, 1, 1] => 40,
                    11 if *dice.iter().max().unwrap() == 5_usize => 50,
                    12 => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>(),
                    _ => 0,
                };
                scores[(action, dice_idx)] = score;
            }
        }
        scores
    }
    // Matrix of 252x462 of allowed keepers from each dice roll
    pub fn dice_to_keepers() -> Array2<f32> {
        let shape = (NUM_DICE_COMBINATIONS, NUM_KEEPERS);
        let mut dice_to_keepers: Array2<f32> = Array2::ones(shape);

        let dice: Vec<DiceCounts> = dice_combinations(NUM_DICE);
        let keepers: Vec<DiceCounts> = (0..=5).flat_map(|n| dice_combinations(n)).collect();    

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

        let shape = (NUM_KEEPERS, NUM_DICE_COMBINATIONS);
        let mut keepers_to_dice = Array2::zeros(shape);
        
        // all possible dice combinations, from 0 dice to 5 dice
        let keepers = (0..=NUM_DICE).flat_map(|n| dice_combinations(n));

        for (keeper_idx, keeper) in keepers.enumerate() {

            let num_keeper_dice = keeper.0.iter().sum::<usize>();
            let num_remaining_dice = 5_usize - num_keeper_dice;

            for mut roll in dice_combinations(num_remaining_dice) {
                let roll_probability = dice_probability(&roll.0);
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
    fn dice_probability(dice: &[usize; 6]) -> f32 {
        let total_dice: usize = dice.iter().sum();
        let mut permutations_num = 1;
        let mut remaining_dice = total_dice;
        for count in dice.iter() {
            permutations_num *= choose(remaining_dice, *count);
            remaining_dice -= count;
        }

        let total_permutations = (NUM_DICE_FACES as f32).powf(total_dice as f32);
        (permutations_num as f32) / total_permutations
    }

    // C(n, k) - does not need to be any more efficient than this
    fn choose(n:usize, k:usize) -> usize {
        let mut answer = 1;
        for num in (k+1)..=n {
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
            assert_eq!(dice_combinations.len(), NUM_DICE_COMBINATIONS);
        }

        #[test]
        fn test_yahtzee_dice() {
            let yahtzee_dice = yahtzee_dice();

            assert_eq!(yahtzee_dice.len(), NUM_DICE_COMBINATIONS);

            assert_eq!(yahtzee_dice.iter().filter(|d| d.is_some()).count(), NUM_DICE_FACES);

            let expected = vec![
                ([5, 0, 0, 0, 0, 0], 0),
                ([0, 5, 0, 0, 0, 0], 1),
                ([0, 0, 5, 0, 0, 0], 2),
                ([0, 0, 0, 5, 0, 0], 3),
                ([0, 0, 0, 0, 5, 0], 4),
                ([0, 0, 0, 0, 0, 5], 5),
            ];

            for (dice, value) in expected.into_iter() {
                let idx = DICE_IDX_LOOKUP[&DiceCounts(dice)];
                assert_eq!(yahtzee_dice[idx], Some(value));
            }
        }
    }
}




bitflags! {
    struct EntryAction: u16 {
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
        let value = value.min(NUM_STATES - 1);
        State(value)
    }
}

impl From<State> for usize {
    fn from(value: State) -> Self {
        value.0
    }
}

impl State {    

    pub fn child(&self, action_idx: EntryAction, dice_idx: usize) -> State {
        let mut child = *self;

        // set action
        child.entries |= action_idx;

        // set upper score
        let upper_actions = EntryAction::ONE 
            | EntryAction::TWO 
            | EntryAction::THREE 
            | EntryAction::FOUR
            | EntryAction::FIVE 
            | EntryAction::SIX;

        if upper_actions.contains(action_idx) {
            let score = DICE_AND_ENTRY_SCORES[(action_idx.bits() as usize, dice_idx)];
            child.upper_score_remaining = child.upper_score_remaining.saturating_sub(score as u8);
        }
        
        // set yahtzee eligibility
        if action_idx == EntryAction::YAHTZEE && YAHTZEE_DICE[dice_idx].is_some() {
            child.yahtzee_bonus_eligible = true;
        }
        child
    }    

    pub fn score_and_child(&self, action_idx: EntryAction, dice_idx: usize) -> (f32, State) {

        let child = self.child(action_idx, dice_idx);

        let mut normal_score = DICE_AND_ENTRY_SCORES[(action_idx.bits() as usize, dice_idx)] as f32;
        let upper_bonus = if !self.upper_complete() && child.upper_complete() {
            35_f32
        } else {
            0_f32
        };

        let yahtzee_bonus = if YAHTZEE_DICE[dice_idx].is_some() && self.yahtzee_bonus_eligible {
            100_f32
        } else {
            0_f32
        };
        

        // joker rule
        // yahtzee box filled
        if self.entries.contains(EntryAction::YAHTZEE) {
            // dice is yahtzee
            if let Some(yahtzee_idx) = YAHTZEE_DICE[dice_idx] {
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

    fn is_valid_action(&self, action_idx: EntryAction) -> bool {
        !self.entries.contains(action_idx)
    }

    fn upper_complete(&self) -> bool {
        self.upper_score_remaining == 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    pub state_scores: Array1<f32>,
}

impl Scores {

    pub fn new() -> Self {
        let state_scores = Array1::zeros(NUM_STATES);

        Scores { 
            state_scores,
        }
    }

    pub fn build(&mut self) {
    
        let valid_states = self.valid_states();

        // We go level-by-level, bottom to top, for correctness when multiprocessing
        for level in (0..NUM_ENTRY_ACTIONS).rev() {

            let mut new_scores = vec![0_f32; NUM_STATES];

            new_scores.par_iter_mut()
                .enumerate()
                .for_each(|(state_idx, score)| {
                    let state_level = (state_idx & 0b111111_1111111_0_000000).count_ones() as usize;            
                    if valid_states[state_idx] && state_level == level {
                        let state: State = state_idx.into();
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
        ExpectedValues{entry_actions: Vec::new(), third_dice: Vec::new(),
        second_keepers: Vec::new(),
        second_dice: Vec::new(),
        first_keepers: Vec::new(),
        first_dice: Vec::new(),
        value: 0_f32,
        }
    }

    fn widget(&self, state: State) -> f32 {

        // values of each entry for each final dice roll
        let entry_scores = Array2::from_shape_fn((13, 252), |(action_idx, dice_idx)| {
            if !state.is_valid_action(action_idx) {
                0_f32
            } else {
                let (score, child) = state.score_and_child(action_idx, dice_idx);
                let child_idx: usize = child.into();
                score + self.state_scores[child_idx]
            }
        });
        // value of each final dice roll
        let max_action_values = entry_scores.fold_axis(Axis(0), 0_f32, |acc, value| acc.max(*value));

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
                *val = (&dice_to_action * &avg_action_values).fold(0_f32, |acc, elem| acc.max(*elem));
            });

        Zip::from(&mut avg_action_values)
            .and(KEEPERS_TO_DICE_PROBABILITIES.genrows())
            .apply(|avg, act| {
                *avg = (&act * &dice_values).sum();
            });

        Zip::from(&mut dice_values)
            .and(DICE_TO_ALLOWED_KEEPERS.genrows())
            .apply(|val, dice_to_action| {
                *val = (&dice_to_action * &avg_action_values).fold(0_f32, |acc, elem| acc.max(*elem));
            });

        let first_roll = KEEPERS_TO_DICE_PROBABILITIES.index_axis(Axis(0), 0);
        first_roll.dot(&dice_values)
    }

    fn valid_states(&self) -> Box<[bool]> {

        let mut valid_markers = vec![false;  NUM_STATES];
        let default_idx: usize = State::default().into();
        valid_markers[default_idx] = true;

        let mut stack = VecDeque::new();

        // wait, I changed this to a queue at some point?
        stack.push_front(State::default());

        while let Some(elem) = stack.pop_back() {        

            for action_idx in 0..13 {

                if elem.is_valid_action(action_idx) {

                    for dice_idx in 0..252 {

                        let child = elem.child(action_idx, dice_idx);
                        let idx: usize = child.into();
                        if !valid_markers[idx] {
                            valid_markers[idx] = true;
                            stack.push_front(child);
                        }
                    }
                }
            }
        }
        valid_markers.into_boxed_slice()
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedValues {
    entry_actions: Vec<Option<f32>>,
    third_dice: Vec<f32>,
    second_keepers: Vec<f32>,
    second_dice: Vec<f32>,
    first_keepers: Vec<f32>,
    first_dice: Vec<f32>,
    value: f32,
}

#[wasm_bindgen]
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
        assert_eq!(num_valid, NUM_VALID_STATES);
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