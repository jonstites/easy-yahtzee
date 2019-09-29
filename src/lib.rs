use std::convert::From;
use std::fmt;
use std::collections::{HashMap, VecDeque};


#[macro_use]
extern crate lazy_static;
extern crate ndarray;
extern crate rayon;

use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

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

type DiceCounts = [usize; 6];

lazy_static! {
    static ref SCORES: Box<[[(u8, Option<usize>); 252]; 13]> = {
        let mut scores = Box::new([[(0, None); 252]; 13]);
        
        for (idx, dice) in dice_combinations(5).into_iter().enumerate() {
            let small_straight = dice[..4].iter().all(|&x| x > 0) || dice[1..5].iter().all(|&x| x > 0) || dice[2..6].iter().all(|&x| x > 0) ;
            for action in 0..13 {
                let score = match action {
                    idx if idx < 6 => (dice[action] * (action + 1)) as u8,
                    6 if *dice.iter().max().unwrap() >= 3_usize => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    7 if *dice.iter().max().unwrap() >= 4_usize => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    8 if *dice.iter().max().unwrap() == 3_usize && *dice.iter().filter(|&&i| i != 3_usize).max().unwrap() == 2_usize => 25,
                    9 if small_straight => 30,
                    10 if dice == [1, 1, 1, 1, 1, 0] || dice == [0, 1, 1, 1, 1, 1] => 40,
                    11 if *dice.iter().max().unwrap() == 5_usize => 50,
                    12 => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    _ => 0_u8,
                };
                
                let is_yahtzee = dice.iter().position(|&count| count == 5_usize);
                

                scores[action][idx] = (score, is_yahtzee);
            }
        }

        scores
    };

    static ref ACTIONS_TO_DICE_ARRAY: Array2<f64> = {
        keepers_to_dice()
    };

    static ref DICE_TO_ACTIONS_ARRAY: Array2<f64> = {
        dice_to_keepers()
    };    
}

// Matrix of 252x462 of allowed keepers from each dice roll
fn dice_to_keepers() -> Array2<f64> {
    let shape = (NUM_DICE_COMBINATIONS, NUM_KEEPERS);
    let mut dice_to_keepers: Array2<f64> = Array2::ones(shape);

    let dice: Vec<DiceCounts> = dice_combinations(NUM_DICE);
    let keepers: Vec<DiceCounts> = (0..=5).flat_map(|n| dice_combinations(n)).collect();    

    for (dice_idx, dice) in dice.iter().enumerate() {
        for (keeper_idx, keeper) in keepers.iter().enumerate() {
            for (die_count, keeper_die_count) in dice.iter().zip(keeper.iter()) {
                // Invalid action - cannot legitimately have keeper 
                // if count is greater than the dice roll
                if keeper_die_count > die_count {
                    dice_to_keepers[(dice_idx, keeper_idx)] = 0_f64;                    
                }
            }
        }
    }
    dice_to_keepers
}

// Matrix of 462x252 of transition probabilities from Keepers to Dice
fn keepers_to_dice() -> Array2<f64> {    

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

        let num_keeper_dice = keeper.iter().sum::<usize>();
        let num_remaining_dice = 5_usize - num_keeper_dice;

        for mut roll in dice_combinations(num_remaining_dice) {
            let roll_probability = dice_probability(&roll);
            // merge keeper with thrown roll
            for die_idx in 0..keeper.len() {
                roll[die_idx] += keeper[die_idx];
            }
            let dice_idx = dice_idx_lookup[&roll];
            keepers_to_dice[(keeper_idx, dice_idx)] = roll_probability;
        }
    }
    
    keepers_to_dice
}

// Odds of rolling a particular dice combination
// Can be computed from the values themselves - no need to consider permutations
pub fn dice_probability(dice: &[usize; 6]) -> f64 {
    let total_dice: usize = dice.iter().sum();
    let mut permutations_num = 1;
    let mut remaining_dice = total_dice;
    for count in dice.iter() {
        permutations_num *= choose(remaining_dice, *count);
        remaining_dice -= count;
    }

    let total_permutations = (NUM_DICE_FACES as f64).powf(total_dice as f64);
    (permutations_num as f64) / total_permutations
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

// Generate all dice combinations for a given number of dice
pub fn dice_combinations(num_dice: usize) -> Vec<DiceCounts> {
    let mut dice: DiceCounts = [0; NUM_DICE_FACES];
    dice[0] = num_dice;

    let mut dice_combinations = Vec::new();
    dice_combinations.push(dice);

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
        dice_combinations.push(dice);
    }
    dice_combinations
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct State(usize);

impl Default for State {

    fn default() -> State {
        // 63 is the number to hit to get upper score bonus
        // we keep the points remaining until it
        // everything else starts 0
        State(63_usize)
    }
}

impl fmt::Display for State {

    fn fmt(&self, dest: &mut fmt::Formatter) -> fmt::Result {
        // display in binary
        write!(dest, "{:#08b}_{:#09b}_{:#01b}_{:#06b}", 
            self.0 >> 14, 
            (self.0 >> 7) & 0b1111111,
            (self.0 >> 6) & 0b1,
            self.0 & 0b111111)
    }

}

impl State {    

    pub fn child(&self, action_idx: usize, dice_idx: usize) -> State {
        let mut child = *self;

        // set action
        child.0 |= 1 << (19 - action_idx);

        // set upper score
        if action_idx < 6 {
            let upper_score = (child.0 & 0b111111).saturating_sub(SCORES[action_idx][dice_idx].0 as usize);
                
            child.0 = (child.0 >> 6) << 6;
            child.0 |= upper_score;
        }
        
        // set yahtzee eligibility
        if action_idx == 11 && SCORES[action_idx][dice_idx].1.is_some() {
            child.0 |= 1 << 6;
        }
        child
    }    

    pub fn score_and_child(&self, action_idx: usize, dice_idx: usize) -> (f64, State) {

        let child = self.child(action_idx, dice_idx);

        let mut normal_score = SCORES[action_idx][dice_idx].0 as f64;
        let upper_bonus = if !self.upper_complete() && child.upper_complete() {
            35_f64
        } else {
            0_f64
        };

        let yahtzee_bonus = if SCORES[action_idx][dice_idx].1.is_some() && ((self.0 >> 6) & 1 == 1)   {
            100_f64
        } else {
            0_f64
        };
        

        // joker rule
        // yahtzee box filled
        if self.0 & (1 << (19 - 11)) != 0 {
            // dice is yahtzee
            if let Some(yahtzee_idx) = SCORES[action_idx][dice_idx].1 {
                // upper entry filled
                if !self.is_valid_action(yahtzee_idx) {
                    if action_idx == 8 {
                        normal_score = 25_f64;
                    } else if action_idx == 9 {
                        normal_score = 30_f64;
                    } else if action_idx == 10 {
                        normal_score = 40_f64;
                    }
                }
            }
        }         

        
        let score = normal_score + upper_bonus + yahtzee_bonus;
        (score, child)
    }

    fn is_valid_action(&self, action_idx: usize) -> bool {
        (self.0 >> (19 - action_idx) & 1) != 1
    }

    fn upper_complete(&self) -> bool {
        self.0 & 0b11_1111 == 0
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

pub fn valid_states() -> Box<[bool]> {

    let mut valid_markers = vec![false;  NUM_STATES];
    let default_idx: usize = State::default().into();
    valid_markers[default_idx] = true;

    let mut stack = VecDeque::new();
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

pub fn widget(state: State, scores: &Vec<f64>) -> f64 {
    // base case
    if !(0..13).map(|i| state.is_valid_action(i)).any(|i| i) {
        return 0_f64;
    }

    // values of each entry for each final dice roll
    let entry_scores = Array2::from_shape_fn((13, 252), |(action_idx, dice_idx)| {
        if !state.is_valid_action(action_idx) {
            0_f64
        } else {
            let (score, child) = state.score_and_child(action_idx, dice_idx);
            let child_idx: usize = child.into();
            score + scores[child_idx]
        }
    });
    // value of each final dice roll
    let max_action_values = entry_scores.fold_axis(Axis(0), 0_f64, |acc, value| acc.max(*value));

    let actions_to_dice: &Array2<f64> = &ACTIONS_TO_DICE_ARRAY;
    let mut avg_action_values: Array1<f64> = Array1::zeros(462);
    
    Zip::from(&mut avg_action_values)
        .and(actions_to_dice.genrows())
        .apply(|avg, act| {
            *avg = (&act * &max_action_values).sum();
        });


    let dice_to_actions: &Array2<f64> = &DICE_TO_ACTIONS_ARRAY;
    let mut dice_values: Array1<f64> = Array1::zeros(252);
    Zip::from(&mut dice_values)
        .and(dice_to_actions.genrows())
        .apply(|val, dice_to_action| {
            *val = (&dice_to_action * &avg_action_values).fold(0_f64, |acc, elem| acc.max(*elem));
        });

    Zip::from(&mut avg_action_values)
        .and(actions_to_dice.genrows())
        .apply(|avg, act| {
            *avg = (&act * &dice_values).sum();
        });

    Zip::from(&mut dice_values)
        .and(dice_to_actions.genrows())
        .apply(|val, dice_to_action| {
            *val = (&dice_to_action * &avg_action_values).fold(0_f64, |acc, elem| acc.max(*elem));
        });

    let first_roll = actions_to_dice.index_axis(Axis(0), 0);
    first_roll.dot(&dice_values)
}

pub fn scores() -> Vec<f64> {
    
    let valid_states = valid_states();
    let mut scores = vec![0.0; NUM_STATES];

   
    // totally unnecessary until implementing multiprocessing
    for level in (0..=14).rev() {
        let scores_ro = scores.clone();
        let bands : Vec<(usize, &mut [f64])> = scores.chunks_mut(1).enumerate().collect();

        bands
            //.rev()  unnecessary when using levels
            .into_par_iter()
            .for_each(|(state_idx, value)| {

            let state_level = (state_idx & 0b111111_1111111_0_000000).count_ones();            
            if valid_states[state_idx] && state_level == level {
                let state: State = state_idx.into();
                let score = widget(state, &scores_ro);
                value[0] = score;                
    }})};
    scores    
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedValues {
    entry_actions: Vec<Option<f64>>,
    third_dice: Vec<f64>,
    second_keepers: Vec<f64>,
    second_dice: Vec<f64>,
    first_keepers: Vec<f64>,
    first_dice: Vec<f64>,
    value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    pub state_scores: Array1<f64>,
    keepers_to_dice: Array2<f64>,
    dice_to_keepers: Array2<f64>,
}

impl Scores {

    pub fn new() -> Self {
        let keepers_to_dice = keepers_to_dice();
        let dice_to_keepers = dice_to_keepers();
        let state_scores = Array1::zeros(NUM_STATES);

        Scores { 
            state_scores,
            keepers_to_dice,
            dice_to_keepers,
        }
    }

    pub fn values(_state: State) -> ExpectedValues {
        ExpectedValues{entry_actions: Vec::new(), third_dice: Vec::new(),
        second_keepers: Vec::new(),
        second_dice: Vec::new(),
        first_keepers: Vec::new(),
        first_dice: Vec::new(),
        value: 0_f64,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_valid_states() {
        let states = valid_states();
        let num_valid = states.iter().filter(|x| **x).count();
        assert_eq!(num_valid, NUM_VALID_STATES);
    }

    #[test]
    fn test_expected_value() {
        let default_idx: usize = State::default().into();
        let scores = scores();
        let expected_value = scores[default_idx];
        assert!((expected_value - 254.5896).abs() < 0.0001);
    }
}