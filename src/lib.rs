#![allow(dead_code, unused_variables, unused_parens, unused_imports)]
#![feature(nll)]
#![feature(try_trait)]
#![feature(test)]

extern crate itertools;
extern crate test;

use test::Bencher;
use itertools::Itertools;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::slice::Iter;
use std::result;



use self::Entry::*;

const DICE_TO_ROLL: i32 = 5;
const DICE_SIDES: i32 = 6;

type Result<T> = result::Result<T, YahtzeeError>;

#[derive(Debug, PartialEq)]
pub enum YahtzeeError {
    InternalError,
    MissingState(std::option::NoneError)
}

impl From<std::option::NoneError> for YahtzeeError {
    fn from(err: std::option::NoneError) -> YahtzeeError {
        YahtzeeError::MissingState(err)
    }
}


pub struct ScoreData {
    expected_values: HashMap<State, f64>,
    roll_probabilities: Box<[HashMap<DiceCombination, f64>]>,
    distinct_rolls: HashSet<DiceCombination>,
    widget_states: Box<[WidgetState]>,
}

impl ScoreData {

    pub fn new() -> ScoreData {
        let expected_values = HashMap::new();
        let roll_probabilities = Box::new(DiceCombination::probabilities_up_to(DICE_SIDES));
        let distinct_rolls = roll_probabilities
            .last()
            .unwrap()
            .iter()
            .map(|(key, _)| *key)
            .collect();

        
        let widget_states = Box::new(
            full_state_children(&distinct_rolls, &roll_probabilities));

        
        let mut data = ScoreData {
            expected_values,
            roll_probabilities,
            distinct_rolls,
            widget_states,
        };

        data.init();
        data
    }

    pub fn init(&mut self) {
        let starting_state = State::default();
        self.init_from_state(starting_state);
    }
    
    pub fn init_from_state(&mut self, starting_state: State) {
        for &state in starting_state.children() {
            let score = full_state_score(
                &state,
                &self.full_state_children,
                &self.expected_values);

            let score = self.get_score(state).expect("fail in new place here");
            self.expected_values.insert(state, score);
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
        states.reverse();
        states
    }

    fn child(&self, entry: Entry, roll: DiceCombination) -> State {
        let mut entries_taken = self.entries_taken.clone();

        let index = entry as usize;
        entries_taken[index] = true;     

        let upper_score_total = self.new_upper_score(entry, roll);
        let positive_yahtzee = self.positive_yahtzee || ((entry == Yahtzee) && (roll.is_yahtzee()));
        
        State {
            entries_taken,
            upper_score_total,
            positive_yahtzee
        }
    }
    
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
enum WidgetState {
    Dice(DiceCombination, i32),
    Keepers(DiceCombination, i32),
}

fn full_state_children(
    possible_rolls: &HashSet<DiceCombination>,
    roll_probs: &Box<[HashMap<DiceCombination, f64>]>,
) -> Vec<WidgetState> {
    let mut children = Vec::new();

    // Entries
    possible_rolls.iter()
        .foreach(|&roll| children.push(WidgetState::Dice(roll, 0)));

    for rolls_remaining in (1..=2) {
        // Keepers
        roll_probs.iter()
            .flat_map(|all_rolls| all_rolls.keys())
            .map(|(&keeper)| WidgetState::Keepers(keeper, rolls_remaining))
            .foreach(|full_state| children.push(full_state));

    // Dice        
        possible_rolls.iter()
            .map(|(&dice)| WidgetState::Dice(dice, rolls_remaining))
            .foreach(|full_state| children.push(full_state));
    };

    let initial = WidgetState::Keepers(DiceCombination::new(), 3);
    children.push(initial);
    //println!("{:?}", children);
    children
}

fn full_state_score(
    state: &State,
    children: &Vec<WidgetState>,
    minimal_state_values: &HashMap<State, f64>,
    roll_probs: &Vec<HashMap<DiceCombination, f64>>,    
) -> f64 {
    if state.is_terminal() {
        return 0_f64;
    }
    let mut full_state_values = HashMap::new();
    
    children.iter().foreach(
        |&full_state| {
            let score = match full_state {
                WidgetState::Dice(dice, 0) => 
                    max_entry_score(
                        state,
                        &dice,
                        minimal_state_values).expect("whoopsie"),
                
                WidgetState::Dice(dice, rolls_remaining) =>
                    max_keeper_score(
                        &dice,
                        rolls_remaining,
                        &full_state_values),
                
                WidgetState::Keepers(dice, rolls_remaining) =>
                    average_rolled_dice_score(
                        &dice,
                        rolls_remaining,
                        roll_probs,
                        &full_state_values),
            };
            //println!("{:?} {:?}", full_state, score);
            full_state_values.insert(full_state, score);
        });

    let initial = WidgetState::Keepers(DiceCombination::new(), 3);
    //println!("full state score: {:}",     *full_state_values.get(&initial).expect("weird!"));
    *full_state_values.get(&initial).expect("weird!")
}

fn max_entry_score(
    state: &State,
    dice: &DiceCombination,
    minimal_state_values: &HashMap<State, f64>
) -> Result<f64> {
    let output = Entry::iterator()
        .filter(|&e| state.is_valid(&e))                
        .map(|&e| -> Result<_> {
            let score = state.score(e, *dice) as f64;
            let child = *state.child(e, *dice);
            let child_score = minimal_state_values.get(&child)?;
            Ok(score + child_score)
        }).collect::<Result<Vec<f64>>>()?;

    let best = output.into_iter().fold(std::f64::NAN, f64::max); // Find the largest non-NaN in vector, or NaN otherwise
    //println!("best entry of {:?} with {:?}", state, dice);
    Ok(best)
}

fn max_keeper_score(
    dice: &DiceCombination,
    rolls_remaining: i32,
    full_state_values: &HashMap<WidgetState, f64>
) -> f64 {

    dice.possible_keepers().iter()
        .map(|&keeper_dice| WidgetState::Keepers(keeper_dice, rolls_remaining))
        .map(|keeper| *full_state_values.get(&keeper).expect("Some internal error calculating score..."))
        .fold(std::f64::NAN, f64::max) // Find the largest non-NaN in vector, or NaN otherwise
}

fn average_rolled_dice_score(
    dice: &DiceCombination,
    rolls_remaining: i32,
    roll_probs: &Vec<HashMap<DiceCombination, f64>>,
    full_state_values: &HashMap<WidgetState, f64>,
    ) -> f64 {

    let dice_to_roll = DICE_TO_ROLL - dice.total_dice();
    let resulting_rolls = &roll_probs[dice_to_roll as usize];

    let score = resulting_rolls.iter()
        .map(|(keeper_roll, probability)| -> f64 {
            let combined_dice = dice.add(keeper_roll);
            let next_state = WidgetState::Dice(combined_dice, rolls_remaining - 1);
            let next_state_value = full_state_values.get(&next_state).expect("Some internal error with averages");
            probability * next_state_value
        })
        .sum();
    //println!("average roll value of {:?} with {:?}", dice, rolls_remaining);    
    score
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

    fn probabilities_up_to(max_dice_number: i32) -> Box<[HashMap<DiceCombination, f64>]> {
        let mut rolls = Vec::new();
        for dice_number in 0..=max_dice_number {
            let probabilities = DiceCombination::probabilities(dice_number);
            rolls.push(probabilities);
        }

        rolls.into_boxed_slice()
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

    fn total_dice(self) -> i32 {
        self.dice.iter().sum::<i32>()
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
    /*
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
        let state_builder =StateBuilder::new(Config::new());
        let mut starting_state = State::default();
        for i in 0..5 {
            starting_state.entries_taken[i] = true;
        }
        let children = state_builder.children(starting_state);

        assert_eq!(1344, children.len());
    }

    #[bench]
    fn bench_full_state(b: &mut Bencher) {
        let mut action_scores = ActionScores::new(Config::new());
        let mut starting_state = State::default();
        
        for i in 1..13 {
            starting_state.entries_taken[i] = true;
        }
        
        action_scores.init_from_state(starting_state);

        b.iter(|| {
            action_scores.value_of_keepers(vec!(0, 0, 3, 1, 0, 0), 2, starting_state)
        })
    }

    
    #[test]
    fn test_state_value() {
        let mut action_scores = ActionScores::new(Config::new());
        let mut starting_state = State::default();
        for i in 1..10 {
            starting_state.entries_taken[i] = true;
        }
        action_scores.init_from_state(starting_state);
        let actual_value = action_scores.value_of_state(starting_state).unwrap();
        let expected_value = 55.581619_f64;
        println!("Actual: {:+} Expected: {:?}", actual_value, expected_value);
        let abs_difference = (actual_value - expected_value).abs();
        let tolerance = 0.00001;
        assert!(abs_difference < tolerance);
    }

    #[test]
    fn test_state_value_bad() {
        let mut action_scores = ActionScores::new(Config::new());
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
        let mut action_scores = ActionScores::new(Config::new());
        let mut starting_state = State::default();
        for i in 2..10 {
            starting_state.entries_taken[i] = true;
        }
        action_scores.init_from_state(starting_state);
        let actual_value = action_scores.value_of_keepers(vec!(0, 4, 0, 0, 0, 0), 1, starting_state).unwrap();
        println!("{:?}", actual_value);
        let expected_value = 72.42314_f64;
        let abs_difference = (actual_value - expected_value).abs();
        let tolerance = 0.00001;
        assert!(abs_difference < tolerance);
    }


    #[test]
    fn test_entry_value_bad() {
        let mut action_scores = ActionScores::new(Config::new());
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

    #[test]
    fn test_full_state_children() {

        let state_builder = StateBuilder::new(Config::new());

        let children = full_state_children(
            &state_builder.possible_rolls,
            &state_builder.rolls);

        assert_eq!(children.len(), 1681);

    }

    #[bench]
    fn bench_full_state_children(b: &mut Bencher) {
        let state = State::default();
        let state_builder = StateBuilder::new(Config::new());

        b.iter(|| {
            full_state_children(
                &state_builder.possible_rolls,
                &state_builder.rolls)
        })
    }

    #[bench]
    fn bench_full_state_children_clone(b: &mut Bencher) {
        let state = State::default();
        let state_builder = StateBuilder::new(Config::new());
        let children = full_state_children(
                &state_builder.possible_rolls,
                &state_builder.rolls);

        b.iter(|| {
            test::black_box(children.clone())
        })
    }


    #[bench]
    fn bench_max_entry_score(b: &mut Bencher) {
        let mut action_scores = ActionScores::new(Config::new());
        let mut starting_state = State::default();
        for i in 1..10 {
            starting_state.entries_taken[i] = true;
        }
        action_scores.init_from_state(starting_state);

        b.iter(|| {
            max_entry_score(
                &starting_state,
                &DiceCombination::from_vec(vec!(0, 3, 2, 0, 0, 0)),
                &action_scores.state_builder,
                &action_scores.state_values)
        })
    }

    #[bench]
    fn bench_max_keeper_score(b: &mut Bencher) {
        b.iter(|| {
            max_keeper_score(
                &DiceCombination::from_vec(vec!(0, 3, 2, 0, 0, 0)),
                2,
                &HashMap::new()
            )
        })
    }

    #[bench]
    fn bench_average_rolled_dice_score(b: &mut Bencher) {
        let state_builder = StateBuilder::new(Config::new());

        b.iter(|| {
            average_rolled_dice_score(
                &DiceCombination::from_vec(vec!(0, 0, 0, 0, 0, 0)),
                2,
                &state_builder.rolls,
                &HashMap::new())
        })
    }

    #[bench]
    fn bench_full_score_calculation(b: &mut Bencher) {
        let mut action_scores = ActionScores::new(Config::new());
        let mut starting_state = State::default();
        for i in 1..10 {
            starting_state.entries_taken[i] = true;
        }
        action_scores.init_from_state(starting_state);

        let children = full_state_children(
            &action_scores.state_builder.possible_rolls,
            &action_scores.state_builder.rolls);
        
        b.iter(|| {
            full_state_score(
                &starting_state,
                &children,
                &action_scores.state_builder,
                &action_scores.state_values,
                &action_scores.state_builder.rolls)
        })
    }*/
}

