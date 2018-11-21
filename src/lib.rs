#![allow(dead_code, unused_variables, unused_parens, unused_imports)]
#![feature(nll)]
#![feature(try_trait)]


extern crate itertools;

use itertools::Itertools;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::slice::Iter;
use std::result;

mod states;

use self::Entry::*;

const DICE_TO_ROLL: i32 = 5;
const DICE_SIDES: i32 = 6;

type Result<T> = result::Result<T, YahtzeeError>;

#[derive(Debug, PartialEq)]
pub enum YahtzeeError {
    BadConfig,
    InternalError,
    MissingState(std::option::NoneError)
}

impl From<std::option::NoneError> for YahtzeeError {
    fn from(err: std::option::NoneError) -> YahtzeeError {
        YahtzeeError::MissingState(err)
    }
}

#[derive(Debug)]
pub struct ConfigBuilder {
    upper_section_bonus: i32,
    yahtzee_bonus: i32,
    dice_to_roll: i32,
    dice_sides: i32
}

impl ConfigBuilder {

    pub fn default() -> Config {
        ConfigBuilder::new().build().unwrap()
    }
    
    pub fn new() -> ConfigBuilder {
        ConfigBuilder {
            upper_section_bonus: 35,
            yahtzee_bonus: 100,
            dice_to_roll: 5,
            dice_sides: 6
        }
    }

    pub fn upper_section_bonus(&mut self, bonus: i32) -> &mut Self {
        self.upper_section_bonus = bonus;
        self
    }

    pub fn yahtzee_bonus(&mut self, bonus: i32) -> &mut Self {
        self.yahtzee_bonus = bonus;
        self
    }

    pub fn dice_to_roll(&mut self, number: i32) -> &mut Self {
        self.dice_to_roll = number;
        self
    }

    pub fn dice_sides(&mut self, number: i32) -> &mut Self {
        self.dice_sides = number;
        self
    }
    
    pub fn build(&self) -> Result<Config> {
        if self.dice_to_roll < 0 || self.dice_sides < 1 {
            Err(YahtzeeError::BadConfig)
        } else {
            Ok(Config {
                upper_section_bonus: self.upper_section_bonus,
                yahtzee_bonus: self.yahtzee_bonus,
                dice_to_roll: self.dice_to_roll,
                dice_sides: self.dice_sides
            })
        }
    }
}

#[derive(Debug)]
pub struct Config {
    upper_section_bonus: i32,
    yahtzee_bonus: i32,
    dice_to_roll: i32,
    dice_sides: i32
}

// also vector to box works just fine (?!), with hash too.
pub struct ActionScores {
    state_values: HashMap<State, f64>,
    state_builder: StateBuilder
}

impl ActionScores {

    pub fn new(config: Config) -> ActionScores {
        let state_values = HashMap::new();
        let state_builder = StateBuilder::new(config);
        
        ActionScores {
            state_values,
            state_builder,
        }
    }

    pub fn init(&mut self) {
        let starting_state = State::default();
        self.init_from_state(starting_state);
    }
    
    pub fn init_from_state(&mut self, starting_state: State) {
        let mut states = self.state_builder.children(starting_state);
        while let Some(state) = states.pop() {
            let score = self.get_score(state).unwrap();
            self.state_values.insert(state, score);
        }
    }

    pub fn value_of_state(&self, state: State) -> Result<f64> {
        self.get_score(state)
    }

    pub fn value_of_entry(
        &self,        
        entry: Entry,
        state: State,
        dice: Vec<i32>,
    ) -> Result<f64> {
        let dice = DiceCombination::from_vec(dice);
        let child = self.state_builder.child(state, entry, dice);
        let score = state.score(entry, dice) as f64;
        let child_score = *self.state_values.get(&child)?;
        Ok(score + child_score)
    }

    pub fn value_of_keepers(
        &self,        
        keepers: Vec<i32>,
        rolls_remaining: i32,
        state: State
    ) -> Result<f64> {
        let default_full_state = Fs::I(
            FullState {
                dice: DiceCombination::new(),
                rolls_remaining: 3
            });

        let mut fs =  FullStateCalculator {
            minimal_state: &state,
            minimal_state_values: &self.state_values,
            possible_rolls: &self.state_builder.possible_rolls,
            roll_probs: &self.state_builder.rolls,
            full_state_values: HashMap::new(),
            state_builder: &self.state_builder,
        };

        // Handle Error  here
        fs.full_state_calculation(default_full_state)?;

        let lookup_fs = Fs::I(
            FullState {
                dice: DiceCombination::from_vec(keepers),
                rolls_remaining: rolls_remaining,
            });
        Ok(*fs.full_state_values.get(&lookup_fs)?)
    }

    fn get_score(&self, state: State) -> Result<f64> {
        let default_full_state = Fs::I(
            FullState {
                dice: DiceCombination::new(),
                rolls_remaining: 3 
            });

        let score =  FullStateCalculator {
            minimal_state: &state,
            minimal_state_values: &self.state_values,
            possible_rolls: &self.state_builder.possible_rolls,
            roll_probs: &self.state_builder.rolls,
            full_state_values: HashMap::new(),
            state_builder: &self.state_builder,
        }.full_state_calculation(default_full_state);
        score
    }
}


#[derive(Debug)]
struct StateBuilder {
    config: Config,
    rolls: Vec<HashMap<DiceCombination, f64>>,
    possible_rolls: HashSet<DiceCombination>
}

impl StateBuilder {

    // precondition: dice_to_roll must be non-negative
    // precondition: dice_sides must be positive
    // This is guaranteed by checks in ConfigBuilder
    // TODO: Remove the hard-coded number of dice in DiceCombination
    // This will require using a Box<i32> rather than array
    fn new(config: Config) -> StateBuilder {
        
        assert!(config.dice_to_roll >= 0);
        assert!(config.dice_sides > 0);

        let mut rolls = Vec::new();
        for dice_number in 0..=config.dice_to_roll {
            let probabilities = DiceCombination::probabilities(dice_number);
            rolls.push(probabilities);
        }

        let possible_rolls = rolls
            .last()
            .unwrap()
            .iter()
            .map(|(key, _)| *key)
            .collect();
        
        StateBuilder {
            config,
            rolls,
            possible_rolls
        }
    }

    pub fn children(&self, starting_state: State) -> Vec<State> {
        let mut queue = VecDeque::new();
        queue.push_back(starting_state);
        
        let mut added = HashSet::new();
        added.insert(starting_state);
        let mut states = Vec::new();

        while let Some(state) = queue.pop_front() {
            states.push(state);
            for entry in Entry::iterator().filter(|&e| state.is_valid(&e)) {
                for roll in self.possible_rolls.clone() {
                    let child = self.child(state, *entry, roll);
                    if !added.contains(&child) {
                        queue.push_back(child);
                        added.insert(child);                        
                    }
                }
            }
        }
        states
    }

    fn child(&self, parent: State, entry: Entry, roll: DiceCombination) -> State {
        let mut entries_taken = parent.entries_taken.clone();

        let index = entry as usize;
        entries_taken[index] = true;     

        let upper_score_total = parent.new_upper_score(entry, roll);
        let positive_yahtzee = parent.positive_yahtzee || ((entry == Yahtzee) && (roll.is_yahtzee()));
        
        State {
            entries_taken,
            upper_score_total,
            positive_yahtzee
        }
    }
    
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct State {
    pub entries_taken: [bool; 13],
    positive_yahtzee: bool,
    upper_score_total: i32,
}

impl State {

    pub fn default() -> State {
        State {
            entries_taken: [false; 13],
            positive_yahtzee: false,
            upper_score_total: 0,
        }
    }

    fn is_valid(&self, entry: &Entry) -> bool {
        !self.entries_taken[*entry as usize]
    }

    fn new_upper_score(&self, entry: Entry, roll: DiceCombination) -> i32 {
        let index = entry as usize;
        let additional_upper_score = match entry {
            Ones => 1 * roll.dice[index],
            Twos => 2 * roll.dice[index],
            Threes => 3 * roll.dice[index],
            Fours => 4 * roll.dice[index],
            Fives => 5 * roll.dice[index],
            Sixes => 6 * roll.dice[index],
            _ => 0,
        };
        
        63.min(self.upper_score_total + additional_upper_score)
        
    }

    fn score(&self, entry: Entry, roll: DiceCombination) -> i32 {
        let new_upper_score = self.new_upper_score(entry, roll);
        let old_upper_score = self.upper_score_total;
            
        let upper_score_bonus = match (old_upper_score, new_upper_score) {
            (63, 63) => 0,
            (_, 63) => 35, // TODO: use config here
            _ => 0,
        };
        
        let yahtzee_bonus = match (roll.is_yahtzee() && self.positive_yahtzee) {
            true => 100,
            false => 0,
        };

        let dice_score = self.score_dice(entry, roll);
        //println!("{:?} {:?} {:?} {:?} {:?}", upper_score_bonus, yahtzee_bonus, dice_score, entry, roll);
        let total_score = upper_score_bonus + yahtzee_bonus + dice_score;
        total_score
    }

    fn score_dice(&self, entry: Entry, roll: DiceCombination) -> i32 {
        // implemented as free choice joker rule
        // todo: support configuration for joker rule
        let index = entry as usize;
        
        let yahtzee_index = roll.dice.iter().position(|&x| x == 5);
            
        let joker = roll.is_yahtzee() &&
                self.entries_taken[Yahtzee as usize] &&
                self.entries_taken[yahtzee_index.unwrap()];


        match entry {
            Ones => 1 * roll.dice[index],
            Twos => 2 * roll.dice[index],
            Threes => 3 * roll.dice[index],
            Fours => 4 * roll.dice[index],
            Fives => 5 * roll.dice[index],
            Sixes => 6 * roll.dice[index],
            ThreeOfAKind => {
                if roll.dice.iter().max().unwrap() >= &3 {
                    return roll.dice.iter()
                        .enumerate()
                        .map(|(i, count)| count * (i as i32 + 1))
                        .sum();
                }
                return 0;
            },
            FourOfAKind => {
                if roll.dice.iter().max().unwrap() >= &4 {
                    return roll.dice.iter()
                        .enumerate()
                        .map(|(i, count)| count * (i as i32 + 1))
                        .sum();
                }
                return 0;
            }
            FullHouse => {
                let counts = roll.dice.iter().collect::<HashSet<_>>();
                if joker || (counts.contains(&3) && counts.contains(&2)) {
                    return 25;
                }
                return 0;
            }
            SmallStraight => {
                let runs = roll.dice.into_iter().group_by(|&die_count| die_count > &0);
                let longest_run = runs.into_iter()
                    .filter(|(is_positive, _)| *is_positive)
                    .map(|(_, group)| group.collect::<Vec<_>>().len())
                    .max().unwrap();

                if joker || longest_run >= 4 {
                    return 30;
                }
                return 0;
            }
            LargeStraight => {
                let runs = roll.dice.iter().group_by(|&die_count| die_count > &0);
                let longest_run = runs.into_iter()
                    .filter(|(is_positive, _)| *is_positive)
                    .map(|(_, group)| group.collect::<Vec<_>>().len())
                    .max().unwrap();

                if joker || longest_run >= 5 {
                    return 40;
                }
                return 0;

            }
            Yahtzee => {
                if roll.is_yahtzee() {
                    return 50;
                }
                return 0;
            },
            Chance => roll.dice.iter()
                .enumerate()
                .map(|(i, count)| count * (i as i32 + 1))
                .sum(),
        }
    }

    
    fn is_terminal(&self) -> bool {
        self.entries_taken.iter().all(|&x| x)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
struct FullState {
    rolls_remaining: i32,
    dice: DiceCombination
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
enum Fs {
    I(FullState),
    C(FullState)
}

struct FullStateCalculator<'a> {
    minimal_state: &'a State,
    minimal_state_values: &'a HashMap<State, f64>,
    possible_rolls: &'a HashSet<DiceCombination>,
    roll_probs: &'a Vec<HashMap<DiceCombination, f64>>,
    full_state_values: HashMap<Fs, f64>,
    state_builder: &'a StateBuilder
}

impl<'a> FullStateCalculator<'a> {
    
    
    fn best_entry_score(&self, full_state: &FullState) -> Result<f64> {

        let output = Entry::iterator()
            .filter(|&e| self.minimal_state.is_valid(&e))                
            .map(|&e| -> Result<_> {
                let score = self.minimal_state.score(e, full_state.dice) as f64;
                let child = self.state_builder.child(*self.minimal_state, e, full_state.dice);
                let child_score = self.minimal_state_values.get(&child)?;
                Ok(score + child_score)
            }).collect::<Result<Vec<f64>>>()?;

        let best = output.into_iter().fold(std::f64::NAN, f64::max); // Find the largest non-NaN in vector, or NaN otherwise
        Ok(best)
    }


    fn average_rolled_dice_score(
        &mut self,
        full_state: &FullState,
    ) -> Result<f64> {

        let keeper_fs = Fs::I(*full_state);
        let keeper = full_state.dice;
        let dice_to_roll: i32 = DICE_TO_ROLL - (keeper.dice.iter().sum::<i32>());
        let mut expected_value = 0.0;
        for (keeper_roll, keeper_roll_probability) in self.roll_probs[dice_to_roll as usize].iter() {
            let new_dice = keeper.add(keeper_roll);
            let new_fs = Fs::C(FullState{dice: new_dice, rolls_remaining: full_state.rolls_remaining - 1});
            let new_dice_expected_value = self.full_state_calculation(new_fs)?;
            expected_value += new_dice_expected_value * keeper_roll_probability;
        }
        Ok(expected_value)
    }

    fn best_keeper_score(
        &mut self,
        full_state: &FullState,
    ) -> Result<f64> {

        let output = full_state.dice.possible_keepers().iter()
            .map(|&keeper| Fs::I(FullState{dice: keeper, rolls_remaining: full_state.rolls_remaining}))
            .map(|new_fs| -> Result<_> {
                let score = self.full_state_calculation(new_fs)?;
                Ok(score)
            }).collect::<Result<Vec<f64>>>()?;
        Ok(output.into_iter().fold(std::f64::NAN, f64::max)) // Find the largest non-NaN in vector, or NaN otherwise
    }

    fn full_state_calculation(&mut self, full_state: Fs) -> Result<f64> {

        if self.minimal_state.is_terminal() {
            return Ok(0.0);
        }

        match full_state {
            Fs::C(ref s)
                if s.rolls_remaining == 0
                => {
                    if !self.full_state_values.contains_key(&full_state) {
                        let score = self.best_entry_score(s)?;
                        self.full_state_values.insert(full_state, score);
                    }
                    return Ok(*self.full_state_values.get(&full_state).unwrap());
                },
            Fs::I(ref s) => {
                if !self.full_state_values.contains_key(&full_state) {
                    let score = self.average_rolled_dice_score(s)?;
                    self.full_state_values.insert(full_state, score);
                }
                return Ok(*self.full_state_values.get(&full_state).unwrap());
            },
            Fs::C(ref s) => {
                if !self.full_state_values.contains_key(&full_state) {
                    let score = self.best_keeper_score(s)?;
                    self.full_state_values.insert(full_state, score);
                }
                return Ok(*self.full_state_values.get(&full_state)?);
            },            
        }
    }
}
    

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
struct DiceCombination {
    dice: [i32; DICE_SIDES as usize],
}

impl DiceCombination {

    fn new() -> DiceCombination {

        DiceCombination {
            dice: [0; DICE_SIDES as usize],
        }
    }

    fn from_permutation(permutation: Vec<i32>) -> DiceCombination {
        let mut combination = [0; DICE_SIDES as usize];
        for die in permutation {
            let index: usize = die as usize;
            let count = combination[index];
            combination[index] = count + 1;
        }
        
        DiceCombination {
            dice: combination,
        }
    }

    fn probabilities(
        dice_number: i32
    ) -> HashMap<DiceCombination, f64> {

        // Special case to handle the ability to "roll" 0 dice.
        if dice_number == 0 {
            let mut probabilities = HashMap::new();
            probabilities.insert(DiceCombination::new(), 1.0);
            return probabilities;
        }

        let divisor = (DICE_SIDES as f64).powf(dice_number as f64);
        let mut probabilities = HashMap::new();

        let permutation_it = (0..dice_number)
            .map(|_| 0..(DICE_SIDES as i32))
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

    fn is_yahtzee(&self) -> bool {
        self.dice.iter().max().unwrap() >= &5
    }

    fn possible_keepers(&self) -> Vec<DiceCombination> {
        self.dice.iter()
            .map(|i| 0..=*i)
            .multi_cartesian_product()
            .map(|new_counts| DiceCombination::from_vec(new_counts))
            .collect()
    }

    fn add(&self, other: &DiceCombination) -> DiceCombination {
        let mut new_dice = self.dice.clone();

        for (index, count) in other.dice.iter().enumerate() {
            new_dice[index] += count;
        }

        DiceCombination {
            dice: new_dice,
        }
    }


    fn from_vec(vec_counts: Vec<i32>) -> DiceCombination {
        let mut counts = [0; DICE_SIDES as usize];

        for (index, count) in vec_counts.iter().enumerate() {
            counts[index] = *count;
        }

        DiceCombination {dice: counts }

    }
    
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum Entry {
    Ones,
    Twos,
    Threes,
    Fours,
    Fives,
    Sixes,    
    ThreeOfAKind,
    FourOfAKind,
    FullHouse,
    SmallStraight,
    LargeStraight,
    Yahtzee,
    Chance
}

impl Entry {

    fn iterator() -> Iter<'static, Entry> {
        static ENTRIES: [Entry;  13] = [
            Ones,
            Twos,
            Threes,
            Fours,
            Fives,
            Sixes,    
            ThreeOfAKind,
            FourOfAKind,
            FullHouse,
            SmallStraight,
            LargeStraight,
            Yahtzee,
            Chance
        ];
        ENTRIES.into_iter()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder_upper_section_bonus_ok() {
        let bonus = 45;
        let config = ConfigBuilder::new()
            .upper_section_bonus(bonus)
            .build();

        assert_eq!(config.unwrap().upper_section_bonus, bonus);
    }

    #[test]
    fn test_config_builder_yahtzee_bonus_ok() {
        let bonus = 45;
        let config = ConfigBuilder::new()
            .yahtzee_bonus(bonus)
            .build();

        assert_eq!(config.unwrap().yahtzee_bonus, bonus);
    }

    #[test]
    fn test_config_builder_dice_to_roll_ok() {
        let dice_to_roll = 11;
        let config = ConfigBuilder::new()
            .dice_to_roll(dice_to_roll)
            .build();

        assert_eq!(config.unwrap().dice_to_roll, dice_to_roll);
    }

    #[test]
    fn test_config_builder_dice_to_roll_err() {
        let dice_to_roll = -4;
        let config = ConfigBuilder::new()
            .dice_to_roll(dice_to_roll)
            .build();

        assert_eq!(config.expect_err(""), YahtzeeError::BadConfig);
    }

    #[test]
    fn test_config_builder_dice_sides_ok() {
        let dice_sides = 11;
        let config = ConfigBuilder::new()
            .dice_sides(dice_sides)
            .build();

        assert_eq!(config.unwrap().dice_sides, dice_sides);
    }

    #[test]
    fn test_config_builder_dice_sides_err() {
        let dice_sides = 0;
        let config = ConfigBuilder::new()
            .dice_sides(dice_sides)
            .build();

        assert_eq!(config.expect_err(""), YahtzeeError::BadConfig);
    }

    
    
    #[test]
    fn test_roll_probabilities_empty_roll() {
        let empty_roll = DiceCombination::new();
        let actual = DiceCombination::probabilities(0);

        assert!(actual.contains_key(&empty_roll));

        let expected_prob = 1.0;
        assert_eq!(actual.get(&empty_roll).unwrap(), &expected_prob);
    }

    #[test]
    fn test_roll_probabilities_correct() {
        let probabilities = DiceCombination::probabilities(5);

        let actual_dice_combinations = probabilities.len();
        let expected_dice_combinations = 252;
        assert_eq!(actual_dice_combinations, expected_dice_combinations);
        
        let total_prob: f64 = probabilities.values().into_iter().sum();
        let abs_difference = (total_prob - 1.0).abs();
        let tolerance = 0.0000001;
        assert!(abs_difference < tolerance);
    }

    #[test]
    fn test_correct_children() {
        let state_builder =StateBuilder::new(ConfigBuilder::default());
        let mut starting_state = State::default();
        for i in 0..5 {
            starting_state.entries_taken[i] = true;
        }
        let children = state_builder.children(starting_state);

        assert_eq!(1344, children.len());
    }

    #[test]
    fn test_state_value() {
        let mut action_scores = ActionScores::new(ConfigBuilder::default());
        let mut starting_state = State::default();
        for i in 1..10 {
            starting_state.entries_taken[i] = true;
        }
        action_scores.init_from_state(starting_state);
        let actual_value = action_scores.value_of_state(starting_state).unwrap();
        let expected_value = 55.581619_f64;
        let abs_difference = (actual_value - expected_value).abs();
        let tolerance = 0.00001;
        assert!(abs_difference < tolerance);
    }

    #[test]
    fn test_state_value_bad() {
        let mut action_scores = ActionScores::new(ConfigBuilder::default());
        let mut starting_state = State::default();
        for i in 1..10 {
            starting_state.entries_taken[i] = true;
        }
        action_scores.init_from_state(starting_state);
        let result = action_scores.value_of_state(State::default());
        match result {
            Err(YahtzeeError::MissingState(_)) => {},
            _ => assert!(false),
        }
    }

    
    #[test]
    fn test_entry_value() {
        let mut action_scores = ActionScores::new(ConfigBuilder::default());
        let mut starting_state = State::default();
        for i in 2..10 {
            starting_state.entries_taken[i] = true;
        }
        action_scores.init_from_state(starting_state);
        let actual_value = action_scores.value_of_keepers(vec!(0, 4, 0, 0, 0, 0), 1, starting_state).unwrap();
        let expected_value = 72.42314_f64;
        let abs_difference = (actual_value - expected_value).abs();
        let tolerance = 0.00001;
        assert!(abs_difference < tolerance);
    }


    #[test]
    fn test_entry_value_bad() {
        let mut action_scores = ActionScores::new(ConfigBuilder::default());
        let mut starting_state = State::default();
        for i in 2..10 {
            starting_state.entries_taken[i] = true;
        }
        action_scores.init_from_state(starting_state);
        let result = action_scores.value_of_keepers(vec!(0, 4, 0, 0, 0, 0), 1, State::default());
        match result {
            Err(YahtzeeError::MissingState(_)) => {},
            _ => assert!(false),
        }
    }
    
}




