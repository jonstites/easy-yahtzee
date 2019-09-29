
use std::convert::From;
use std::fmt;
use std::collections::VecDeque;


#[macro_use]
extern crate lazy_static;
extern crate ndarray;
use ndarray::{Array2, Zip};
const NUM_STATES: usize = 1048576;
use ndarray::prelude::*;
extern crate rayon;
use rayon::prelude::*;

const NUM_DICE_FACES: usize = 6;
const NUM_DICE: usize = 5;

lazy_static! {
    static ref SCORES: Box<[[(u8, Option<usize>); 252]; 13]> = {
        let mut scores = Box::new([[(0, None); 252]; 13]);
        
        for (idx, dice) in dice_combinations2(5).into_iter().enumerate() {
            println!("{:?} {:?}", idx, dice);
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
                //println!("{:?} {:?} {:?} {:?} {:?}", idx, dice, action, score, is_yahtzee);
            }
        }

        scores
    };

    static ref ACTIONS_TO_DICE_ARRAY: Array2<f64> = {
        let mut actions_to_dice = vec![0_f64; 252*462];
        
        for (action_idx, action) in (0..=5).flat_map(|n| dice_combinations2(n)).enumerate() {
            let left_to_roll = 5_usize - action.iter().sum::<usize>();
            println!("action: {:?} {:?}", action_idx, action);
            for mut roll in dice_combinations2(left_to_roll) {

                let probability = dice_probability(&roll);

                for idx in 0..NUM_DICE_FACES {
                    roll[idx] += action[idx];

                }

                for (idx, other) in dice_combinations2(5).into_iter().enumerate() {
                    if other == roll {
                        actions_to_dice[action_idx * 252 + idx] = probability;
                    }
                }

            }
        }
        let arr2 = Array2::from_shape_vec((462,252), actions_to_dice).unwrap();
        
        arr2
    };

    static ref DICE_TO_ACTIONS_ARRAY: Array2<f64> = {
        
        let mut dice_to_actions_vec = vec![1_f64; 462*252];
        for (dice_idx, dice) in dice_combinations2(5).into_iter().enumerate() {
            for (action_idx, action) in (0..=5).flat_map(|n| dice_combinations2(n)).enumerate() {

                for idx in 0..=5 {
                    if action[idx] > dice[idx] {
                        dice_to_actions_vec[dice_idx*462 + action_idx] = 0_f64;
                        
                    }
                }
            }
        }


        let arr2 = Array2::from_shape_vec((252, 462), dice_to_actions_vec).unwrap();
        arr2
    };    
}

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

pub fn dice_combinations2(num_dice: usize) -> Vec<[usize; NUM_DICE_FACES]> {
    let mut dice = [0; NUM_DICE_FACES];
    dice[0] = num_dice;
    let mut v = Vec::new();
    v.push(dice);

    while dice[NUM_DICE_FACES - 1] != num_dice {

        let rightmost = NUM_DICE_FACES - 1 - dice.iter().rev().position(|&x| x > 0_usize).unwrap();

        if rightmost + 1 < dice.len() {
            dice[rightmost] -= 1;
            dice[rightmost + 1] += 1;
        } else {
            // use simple while loop
            let next_rightmost = NUM_DICE_FACES - 1 - (NUM_DICE_FACES - rightmost + dice[0..rightmost].iter().rev().position(|&x| x > 0_usize).unwrap());
            let num_rightmost = dice[rightmost];
            dice[next_rightmost + 1] += num_rightmost;
            dice[rightmost] -= num_rightmost;
            dice[next_rightmost] -= 1;
            dice[next_rightmost + 1] += 1;

        }
        v.push(dice);
    }
    v
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
            0_f64
        } else {
            let (score, child) = state.score_and_child(action_idx, dice_idx);
            let child_idx: usize = child.into();
            score + scores[child_idx]
        }
    });
    // value of each final dice roll
    let max_action_values = entry_scores.fold_axis(Axis(0), 0_f64, |acc, value| acc.max(*value));
    //println!("{:?}", max_action_values);

    let actions_to_dice: &Array2<f64> = &ACTIONS_TO_DICE_ARRAY;
    let mut avg_action_values: Array1<f64> = Array1::zeros(462);
    
    Zip::from(&mut avg_action_values)
        .and(actions_to_dice.genrows())
        .par_apply(|avg, act| {
            *avg = (&act * &max_action_values).sum();
        });

    //println!("{:?}", avg_action_values);

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
    
    /*
    println!("before valid states");
    let valid_states = valid_states();
    println!("after valid states");
    let mut scores = vec![0.0; NUM_STATES];

  
    // totally unnecessary until implementing multiprocessing
    for level in (0..=14).rev() {
        println!("level: {}", level);
        for state_idx in (0..NUM_STATES).rev() {
            let state_level = (state_idx & 0b111111_1111111_0_000000).count_ones();            
            if valid_states[state_idx] && state_level == level {
                let state: State = state_idx.into();
                // ones open only, yahtzee eligible
                //let state = State (0b011111_1111111_1_111111);
                let score = widget(state, &scores);
                scores[state_idx] = score;                
                //return scores; 
            }
        }
    }
        scores

    */

    
    println!("before valid states");
    let valid_states = valid_states();
    println!("after valid states");
    let mut scores = vec![0.0; NUM_STATES];

   
    // totally unnecessary until implementing multiprocessing
    for level in (0..=14).rev() {
        println!("level: {}", level);
        let scores_ro = scores.clone();
        let bands : Vec<(usize, &mut [f64])> = scores.chunks_mut(1).enumerate().collect();

        bands
            //.rev()  unnecessary when using levels
            .into_par_iter()
            .for_each(|(state_idx, value)| {

            let state_level = (state_idx & 0b111111_1111111_0_000000).count_ones();            
            if valid_states[state_idx] && state_level == level {
                let state: State = state_idx.into();
                // ones open only, yahzee bonus eligible
                //let state = State (0b011111_1111111_1_111111);
                let score = widget(state, &scores_ro);
                value[0] = score;                
    }})};
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