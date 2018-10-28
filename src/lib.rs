#![allow(dead_code)]

extern crate itertools;

use itertools::Itertools;
use std::collections::HashMap;

struct Config {
    dice_sides: i32,
    dice_to_roll: i32,
    upper_section_bonus: i32,
    yahtzee_bonus: i32,    
}

impl Default for Config {

    fn default() -> Config {
        Config {
            dice_sides: 6,
            dice_to_roll: 5,
            upper_section_bonus: 35,
            yahtzee_bonus: 100,
        }
    }
}

struct Advisor {
    config: Config,
    // A lookup table where the first key is the number of dice rolled
    // and the value is another lookup of vector of dice counts to the
    // probability.
    roll_probabilities: Vec<HashMap<DiceCombination, f64>>,
    expected_values: HashMap<State, f64>
}

impl Advisor {
    
    fn new() -> Advisor {
        Advisor {
            config: Config::default(),
            roll_probabilities: Vec::new(),
            expected_values: HashMap::new(),
        }
    }
    
    fn with_options(config: Config) -> Advisor {

        Advisor {
            config,
            roll_probabilities: Vec::new(),
            expected_values: HashMap::new(),            
        }
    }

    fn run(&mut self) {
        self.set_roll_probabilities();
    }

    fn set_roll_probabilities(&mut self) {
        
        for dice_number in 0..=self.config.dice_to_roll {
            let probabilities = DiceCombination::probabilities(
                dice_number,
                self.config.dice_sides,
            );
            self.roll_probabilities.push(probabilities);
        }
    }



    fn set_expected_values(&mut self) {
        let start_state = State::default();

        //self.calculate(start_state);
        
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct DiceCombination {
    dice: Vec<i32>,
}

impl DiceCombination {

    fn new() -> DiceCombination {

        DiceCombination {
            dice: vec!(),
        }
    }

    fn from_permutation(permutation: Vec<i32>) -> DiceCombination {
        let mut combination = permutation;
        combination.sort();
        
        DiceCombination {
            dice: combination,
        }
    }

    fn probabilities(
        dice_number: i32,
        dice_sides: i32
    ) -> HashMap<DiceCombination, f64> {

        // Special case to handle the ability to "roll" 0 dice.
        if dice_number == 0 {
            let mut probabilities = HashMap::new();
            probabilities.insert(DiceCombination::new(), 1.0);
            return probabilities;
        }


        let divisor = (dice_sides as f64).powf(dice_number as f64);
        let mut probabilities = HashMap::new();

        let permutation_it = (0..dice_number)
            .map(|_| 1..=(dice_sides))
            .multi_cartesian_product();

        
        
        for permutation in permutation_it {
            let combination = DiceCombination::from_permutation(permutation);
                
            let probability = probabilities
                .entry(combination)
                .or_insert(0.0);

            *probability += 1.0 / divisor;
        }
        probabilities
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct State {
    entries_taken: [bool; 13],
    positive_yahtzee: bool,
    upper_score_total: i32,    
}

impl State {

    fn default() -> State {
        State {
            entries_taken: [false; 13],
            positive_yahtzee: false,
            upper_score_total: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roll_probabilities_empty_roll() {
        let mut advisor = Advisor::new();
        advisor.set_roll_probabilities();

        let empty_roll = DiceCombination::new();
        let result = advisor.roll_probabilities.get(0 as usize).unwrap();

        assert!(result.contains_key(&empty_roll));

        let expected_prob = 1.0;
        assert_eq!(result.get(&empty_roll).unwrap(), &expected_prob);
    }

    #[test]
    fn test_roll_probabilities_length() {
        let mut advisor = Advisor::new();
        advisor.set_roll_probabilities();
        
        let number_dice = advisor.roll_probabilities.len();

        // 5 dice rolled by default, plus a 0 dice roll
        assert_eq!(number_dice, 6);
    }

    #[test]
    fn test_roll_probabilities_correct() {
        let mut advisor = Advisor::new();
        advisor.set_roll_probabilities();
        
        let total_keys = advisor.roll_probabilities.get(5 as usize).unwrap().len();
        assert_eq!(total_keys, 252);

        
        let total_prob: f64 = advisor.roll_probabilities.get(5 as usize).unwrap().values().into_iter().sum();
        assert!((total_prob - 1.0).abs() < 0.000001);
    }
}

