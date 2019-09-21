
use std::convert::From;
use std::fmt;
use std::collections::VecDeque;

extern crate nalgebra as na;
#[macro_use]
extern crate itertools;
use itertools::Itertools;

#[macro_use]
extern crate lazy_static;

const NUM_STATES: usize = 1048576;


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

    static ref ACTIONS_TO_DICE: Box<[[f32; 252]; 462]> = {
        let mut actions_to_dice = Box::new([[0_f32; 252]; 462]);
        let mut totals = vec![0_f32; 462];
        for (action_idx, action) in (0..=5).flat_map(|n| dice_combinations(n)).enumerate() {
            let left_to_roll = 5_usize - action.iter().sum::<usize>();
            'outer:
            for mut dice_permutation in dice_permutations(left_to_roll) {
                for idx in 0..=5 {
                    dice_permutation[idx] += action[idx];
                }

                let dice_idx = dice_combination_idx(dice_permutation);
                actions_to_dice[action_idx][dice_idx] += 1_f32;
                totals[action_idx] += 1_f32;
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
        
        actions_to_dice
    };

    static ref DICE_TO_ACTIONS: Box<[[bool; 462]; 252]> = {
        let mut dice_to_actions = Box::new([[true; 462]; 252]);

        for (dice_idx, dice) in dice_combinations(5).into_iter().enumerate() {
            for (action_idx, action) in (0..=5).flat_map(|n| dice_combinations(n)).enumerate() {

                for idx in 0..=5 {
                    if action[idx] > dice[idx] {
                        dice_to_actions[dice_idx][action_idx] = false;
                    }
                }
            }
        }

        dice_to_actions
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

pub fn widget(state: State, scores: &Vec<f32>) -> f32 {
    if !(0..13).map(|i| state.is_valid_action(i)).any(|i| i) {
        return 0_f32;
    }

    let mut action_values = Box::new([[0_f32; 13]; 252]);

    let debug = false;
    for action_idx in 0..13 {
        if !state.is_valid_action(action_idx) {
            continue;
        }
        for dice_idx in 0..252 {
            let child = state.child(action_idx, dice_idx);
            let child_idx: usize = child.into();
            let child_value = scores[child_idx];

            let normal_score = SCORES[action_idx][dice_idx].0 as f32;
            let upper_bonus = if !state.upper_complete() && child.upper_complete() {
                35_f32
            } else {
                0_f32
            };

            let yahtzee_bonus = if SCORES[action_idx][dice_idx].1.is_some() && ((state.0 & (1 << 6)) >> 6 == 1)  {
                100_f32
            } else {
                0_f32
            };


            let joker_points = match action_idx {
                8 => 25_f32,
                9 => 35_f32,
                10 => 45_f32,
                _ => 0_f32,
            };

            let joker_rule = match SCORES[action_idx][dice_idx].1 {
                Some(idx) if !state.is_valid_action(idx) => joker_points,
                _ => 0_f32,
            };
            
            let score = child_value + normal_score + upper_bonus + yahtzee_bonus + joker_rule;
            if debug {
            println!("{} {} {} {} {} {} {}", state, child_value, normal_score, upper_bonus, yahtzee_bonus, action_idx, dice_idx);
            }
            action_values[dice_idx][action_idx] = score;
        }
    }

    if debug {
    for action_idx in 0..252 {
        println!("{:?} {:?}", action_idx, action_values[action_idx].clone().into_iter().collect::<Vec<_>>());
    }
    }
    let mut max_action_values = Box::new([0_f32; 252]);
    for dice_idx in 0..252 {
        let max = action_values[dice_idx].iter().max_by(|&lhs, &rhs| lhs.partial_cmp(rhs).unwrap()).unwrap();
        max_action_values[dice_idx] = *max;
    }

    if debug {
    println!("{:?}", max_action_values.clone().into_iter().collect::<Vec<_>>());
    }
    let mut avg_action_values = Box::new([0_f32; 462]);
    for action_idx in 0..462 {
        for dice_idx in 0..252 {
            avg_action_values[action_idx] += ACTIONS_TO_DICE[action_idx][dice_idx] * max_action_values[dice_idx];
            if debug {
                println!("{:?} {:?} {:?} {:?} {:?}", action_idx, dice_idx, avg_action_values[action_idx],ACTIONS_TO_DICE[action_idx][dice_idx], max_action_values[dice_idx]);
            }
        }
    }
    if debug {
    println!("{:?}", avg_action_values.clone().into_iter().collect::<Vec<_>>());
    }
    let mut dice_values = Box::new([0_f32; 252]);
    for dice_idx in 0..252 {
        for action_idx in 0..462 {
            if DICE_TO_ACTIONS[dice_idx][action_idx] {
                dice_values[dice_idx] = dice_values[dice_idx].max(avg_action_values[action_idx]);
            }
        }
    }
    if debug {
    println!("{:?}", dice_values.clone().into_iter().collect::<Vec<_>>());
    }
    let mut avg_action_values2 = Box::new([0_f32; 462]);
    for action_idx in 0..462 {
        for dice_idx in 0..252 {
            avg_action_values2[action_idx] += ACTIONS_TO_DICE[action_idx][dice_idx] * dice_values[dice_idx];
        }
    }
    if debug {
    println!("{:?}", avg_action_values2.clone().into_iter().collect::<Vec<_>>());
    }
    let mut dice_values2 = Box::new([0_f32; 252]);
    for dice_idx in 0..252 {
        for action_idx in 0..462 {
            if DICE_TO_ACTIONS[dice_idx][action_idx] {
                dice_values2[dice_idx] = dice_values2[dice_idx].max(avg_action_values2[action_idx]);
            }
        }
    }
    if debug {
    println!("{:?}", dice_values2.clone().into_iter().collect::<Vec<_>>());
    }
    let mut score = 0_f32;
    for dice_idx in 0..252 {
        score += ACTIONS_TO_DICE[0][dice_idx] * dice_values2[dice_idx];
    }
    if debug {
    println!("{}", score);
    }
    score
}

pub fn scores() -> Vec<f32> {
    for action in 0..13 {
        println!("{:?}", SCORES[action].iter().collect::<Vec<_>>());
    }

    for i in 0..ACTIONS_TO_DICE.len() {
        println!("{:?} {:?}", ACTIONS_TO_DICE[i].iter().collect::<Vec<_>>(), ACTIONS_TO_DICE[i].iter().sum::<f32>());
    }

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