
use std::convert::From;
use std::fmt;
use std::collections::VecDeque;

extern crate nalgebra as na;
extern crate typenum;
use na::{U1, U2, U13, Dynamic, ArrayStorage, RowVectorN, Matrix, Matrix2x3};
use typenum::{U252, U462};

#[macro_use]
extern crate itertools;
use itertools::Itertools;

#[macro_use]
extern crate lazy_static;
extern crate ndarray;
use ndarray::{Array2, Zip};
const NUM_STATES: usize = 1048576;

use nalgebra::*;
extern crate ndarray_parallel;
use ndarray_parallel::prelude::*;
use ndarray::*;
// note: request doc changes for deprecated matrix array
type Matrix252x13  = MatrixMN<f64, U252, U13>;
type Matrix13x252  = MatrixMN<f64, U13, U252>;
type Matrix252x462 = MatrixMN<f64, U252, U462>;
type Matrix462x252 = MatrixMN<f64, U462, U252>;


lazy_static! {
    static ref SCORES: Box<[[(u8, Option<usize>); 252]; 13]> = {
        let mut scores = Box::new([[(0, None); 252]; 13]);
        
        for (idx, dice) in dice_combinations(5).into_iter().enumerate() {
            for action in 0..13 {
                let score = match action {
                    idx if idx < 6 => (dice[action] * (action + 1)) as u8,
                    6 if *dice.iter().max().unwrap() >= 3_usize => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    7 if *dice.iter().max().unwrap() >= 4_usize => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    8 if *dice.iter().max().unwrap() == 3_usize && *dice.iter().filter(|&&i| i != 3_usize).max().unwrap() == 2_usize => 25,
                    9 if dice[..4] == [1, 1, 1, 1] || dice[1..5] == [1, 1, 1, 1] || dice[2..6] == [1, 1, 1, 1] => 35,
                    10 if dice == [1, 1, 1, 1, 1, 0] || dice == [0, 1, 1, 1, 1, 1] => 45,
                    11 if *dice.iter().max().unwrap() == 5_usize => 50,
                    12 => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    _ => 0_u8,
                };
                
                let is_yahtzee = dice.iter().find(|&&i| i == 5_usize).map(|i| *i);
                

                scores[action][idx] = (score, is_yahtzee);
            }
        }

        scores
    };

    static ref SCORES_MATRIX: Matrix13x252 = {
        let combinations = dice_combinations(5);
        let scores = Matrix13x252::from_fn(|action, dice_idx| {
            let dice = combinations[dice_idx];
            let score = match action {
                    idx if idx < 6 => (dice[action] * (action + 1)) as u8,
                    6 if *dice.iter().max().unwrap() >= 3_usize => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    7 if *dice.iter().max().unwrap() >= 4_usize => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    8 if *dice.iter().max().unwrap() == 3_usize && *dice.iter().filter(|&&i| i != 3_usize).max().unwrap() == 2_usize => 25,
                    9 if dice[..4] == [1, 1, 1, 1] || dice[1..5] == [1, 1, 1, 1] || dice[2..6] == [1, 1, 1, 1] => 35,
                    10 if dice == [1, 1, 1, 1, 1, 0] || dice == [0, 1, 1, 1, 1, 1] => 45,
                    11 if *dice.iter().max().unwrap() == 5_usize => 50,
                    12 => dice.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>() as u8,
                    _ => 0_u8,
                };
            score as f64
        });
        scores
    };

    static ref ACTIONS_TO_DICE_MATRIX: Matrix462x252 = {
        let mut actions_to_dice = Box::new([[0_f64; 252]; 462]);
        let mut totals = vec![0_f64; 462];
        for (action_idx, action) in (0..=5).flat_map(|n| dice_combinations(n)).enumerate() {
            let left_to_roll = 5_usize - action.iter().sum::<usize>();
            'outer:
            for mut dice_permutation in dice_permutations(left_to_roll) {
                for idx in 0..=5 {
                    dice_permutation[idx] += action[idx];
                }

                let dice_idx = dice_combination_idx(dice_permutation);
                actions_to_dice[action_idx][dice_idx] += 1_f64;
                totals[action_idx] += 1_f64;
                //println!("{:?} {:?} ", dice_idx, actions_to_dice[action_idx][dice_idx] )
            }
        }

        for action_idx in 0..462 {
            for dice_idx in 0..252 {
                                //println!("{} {} {} ", action_idx, dice_idx, actions_to_dice[action_idx][dice_idx]);

                actions_to_dice[action_idx][dice_idx] /= totals[action_idx];
                //println!("{} {} {} ", action_idx, dice_idx, actions_to_dice[action_idx][dice_idx]);
            }
        }
        
        Matrix462x252::from_fn(|dice_action_idx, dice_idx| actions_to_dice[dice_action_idx][dice_idx])
    };

    static ref ACTIONS_TO_DICE_ARRAY: Array2<f64> = {
        let mut actions_to_dice = Box::new([[0_f64; 252]; 462]);
        let mut totals = vec![0_f64; 462];
        for (action_idx, action) in (0..=5).flat_map(|n| dice_combinations(n)).enumerate() {
            let left_to_roll = 5_usize - action.iter().sum::<usize>();
            'outer:
            for mut dice_permutation in dice_permutations(left_to_roll) {
                for idx in 0..=5 {
                    dice_permutation[idx] += action[idx];
                }

                let dice_idx = dice_combination_idx(dice_permutation);
                actions_to_dice[action_idx][dice_idx] += 1_f64;
                totals[action_idx] += 1_f64;
                //println!("{:?} {:?} ", dice_idx, actions_to_dice[action_idx][dice_idx] )
            }
        }

        let mut shape_vec = Vec::new();
        for action_idx in 0..462 {
            for dice_idx in 0..252 {
                                //println!("{} {} {} ", action_idx, dice_idx, actions_to_dice[action_idx][dice_idx]);

                actions_to_dice[action_idx][dice_idx] /= totals[action_idx];
                shape_vec.push(actions_to_dice[action_idx][dice_idx]);
                //println!("{} {} {} ", action_idx, dice_idx, actions_to_dice[action_idx][dice_idx]);
            }
        }
        
        Array2::from_shape_vec((462,252), shape_vec).expect("womp womp")
    };


    static ref DICE_TO_ACTIONS_MATRIX: Matrix462x252 = {
        let mut dice_to_actions = Box::new([[1_f64; 462]; 252]);

        for (dice_idx, dice) in dice_combinations(5).into_iter().enumerate() {
            for (action_idx, action) in (0..=5).flat_map(|n| dice_combinations(n)).enumerate() {

                for idx in 0..=5 {
                    if action[idx] > dice[idx] {
                        dice_to_actions[dice_idx][action_idx] = 0_f64;
                    }
                }
            }
        }

        Matrix462x252::from_fn(|action_idx, dice_idx| dice_to_actions[dice_idx][action_idx])
    };    

    static ref DICE_TO_ACTIONS_ARRAY: Array2<f64> = {
        
        let mut dice_to_actions_vec = vec![1_f64; 462*252];
        for (dice_idx, dice) in dice_combinations(5).into_iter().enumerate() {
            for (action_idx, action) in (0..=5).flat_map(|n| dice_combinations(n)).enumerate() {

                for idx in 0..=5 {
                    if action[idx] > dice[idx] {
                        dice_to_actions_vec[dice_idx*462 + action_idx] = 0_f64;
                        
                    }
                }
            }
        }


        Array2::from_shape_vec((252, 462), dice_to_actions_vec).unwrap()
    };    
}

fn dice_combination_idx(dice: [usize; 6]) -> usize {
    for (dice_idx, combination) in dice_combinations(5).into_iter().enumerate() {
        if combination == dice {
            return dice_idx;
        }
    }

    panic!("dice idx not found");
}

fn generate_dice(n: usize) -> Vec<Vec<usize>> {
    let mut dice = Vec::new();
    if n == 0 {
        dice.push(Vec::new());
    }

    for roll in (0..n).map(|_| 0..6).multi_cartesian_product() {
        dice.push(roll);
    }

    dice
}

fn dice_permutations(n: usize) -> Vec<[usize; 6]> {
    if n == 0 {
        return vec!([0; 6]);
    }

    let mut permutations = Vec::new();
    for dice in generate_dice(n) {
        let mut dice_array = [0; 6];
        for die in dice.into_iter() {
            dice_array[die] += 1;
        }
        permutations.push(dice_array);
    }
    permutations
}

fn dice_combinations(n: usize) -> Vec<[usize; 6]> {

    if n == 0 {
        return vec!([0; 6]);
    }

    let mut combinations = Vec::new();

    for dice in generate_dice(n) {

        // canonical combination is sorted
        let mut dice_sorted = dice.clone();
        dice_sorted.sort();
        if dice != dice_sorted {
            continue;
        }

        let mut dice_array = [0; 6];
        for die in dice.into_iter() {
            dice_array[die] += 1;
        }
        combinations.push(dice_array);
    }
    combinations
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
        write!(dest, "{:#020b}", self.0)
    }

}

impl State {
    pub fn child(&self, action_idx: usize, dice_idx: usize) -> State {
        let mut child = *self;

        // set action
        child.0 |= 1 << (19 - action_idx);

        // set upper score
        if action_idx < 6 {
            let upper_score = (child.0 & 0b11_1111).saturating_sub(SCORES[action_idx][dice_idx].0 as usize);
                
            child.0 = (child.0 >> 6) << 6;
            child.0 |= upper_score;
        }
        
        // set yahtzee eligibility
        if action_idx == 11 && SCORES[action_idx][dice_idx].1.is_some() {
            child.0 |= 1 << 6;
        }
        child
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

fn valid_states() -> Box<[bool]> {

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
            return 0_f64;
        }
        let child = state.child(action_idx, dice_idx);
        let child_idx: usize = child.into();
        let child_value = scores[child_idx];

        let normal_score = SCORES[action_idx][dice_idx].0 as f64;
        let upper_bonus = if !state.upper_complete() && child.upper_complete() {
            35_f64
        } else {
            0_f64
        };

        let yahtzee_bonus = if SCORES[action_idx][dice_idx].1.is_some() && ((state.0 & (1 << 6)) >> 6 == 1)  {
            100_f64
        } else {
            0_f64
        };


        let joker_points = match action_idx {
            8 => 25_f64,
            9 => 35_f64,
            10 => 45_f64,
            _ => 0_f64,
        };

        let joker_rule = match SCORES[action_idx][dice_idx].1 {
            Some(idx) if !state.is_valid_action(idx) => joker_points,
            _ => 0_f64,
        };
        
        let score = child_value + normal_score + upper_bonus;// + yahtzee_bonus + joker_rule;
        score
    });
    // value of each final dice roll
    let max_action_values = entry_scores.fold_axis(Axis(0), 0_f64, |acc, value| acc.max(*value));

    let actions_to_dice: &Array2<f64> = &ACTIONS_TO_DICE_ARRAY;
    let mut avg_action_values: Array1<f64> = Array1::zeros(462);
    
    Zip::from(&mut avg_action_values)
        .and(actions_to_dice.genrows())
        .par_apply(|avg, act| {
            *avg = (&act * &max_action_values).sum();
        });

    let dice_to_actions: &Array2<f64> = &DICE_TO_ACTIONS_ARRAY;
    let mut dice_values: Array1<f64> = Array1::zeros(252);
    Zip::from(&mut dice_values)
        .and(dice_to_actions.genrows())
        .par_apply(|val, dice_to_action| {
            *val = (&dice_to_action * &avg_action_values).fold(0_f64, |acc, elem| acc.max(*elem));
        });

    Zip::from(&mut avg_action_values)
        .and(actions_to_dice.genrows())
        .par_apply(|avg, act| {
            *avg = (&act * &dice_values).sum();
        });

    Zip::from(&mut dice_values)
        .and(dice_to_actions.genrows())
        .par_apply(|val, dice_to_action| {
            *val = (&dice_to_action * &avg_action_values).fold(0_f64, |acc, elem| acc.max(*elem));
        });

    let first_roll = actions_to_dice.index_axis(Axis(0), 0);
    first_roll.dot(&dice_values)
}

pub fn scores() -> Vec<f64> {
    

    println!("before valid states");
    let valid_states = valid_states();
    println!("after valid states");
    let mut scores = vec![0.0; NUM_STATES];
    for state_idx in (0..NUM_STATES).rev() {

        if valid_states[state_idx] {
            let state: State = state_idx.into();
            // ones open only
            //let state = State (0b011111_1111111_0_111111);
            let score = widget(state, &scores);
            scores[state_idx] = score;
                    if state_idx % 10000 == 0 {
            println!("{} {} {}", state, state_idx, score);
            // return scores;
        }
        }
    }

    scores
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_valid_states() {
        let states = valid_states();
        let num_valid = states.iter().filter(|x| **x).count();
        assert_eq!(num_valid, 536448);
    }

    #[test]
    fn test_expected_value() {
        println!("abc");
let default_idx: usize = State::default().into();
        assert_eq!(scores()[default_idx], 254.5896);
    }
}