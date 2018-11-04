#![allow(dead_code, unused_variables)]


extern crate itertools;

use itertools::Itertools;
use std::collections::HashMap;
use std::collections::HashSet;
use std::slice::Iter;

use self::Entry::*;

const DICE_TO_ROLL: i32 = 5;
const DICE_SIDES: i32 = 6;


struct Config {
    upper_section_bonus: i32,
    yahtzee_bonus: i32,    
}

impl Default for Config {

    fn default() -> Config {
        Config {
            upper_section_bonus: 35,
            yahtzee_bonus: 100,
        }
    }
}

// maybe here define intermediate evs, create it first in
// action scores new. that way there's NO mutable action score
// functions...


// also vector to box works just fine (?!), with hash too.


pub struct ActionScores {
    state_values: HashMap<State, f64>,
    rolls: Vec<HashMap<DiceCombination, f64>>,
    possible_rolls: Vec<DiceCombination>
}

impl ActionScores {

    pub fn new() -> ActionScores {
        let mut rolls = Vec::new();
        for dice_number in 0..=DICE_TO_ROLL {
            let probabilities = DiceCombination::probabilities(dice_number);
            rolls.push(probabilities);
        }

        let possible_rolls = DiceCombination::probabilities(5).iter().map(|(key, _)| *key).collect();
        
        let state_values = HashMap::new();
        
        let mut action_scores = ActionScores {
            state_values,
            rolls,
            possible_rolls
        };
        let mut starting_state = State::default();
        starting_state.entries_taken = [true; 13];
        starting_state.entries_taken[1] = false;
        starting_state.entries_taken[11] = false;
        action_scores.set_scores(starting_state);
        action_scores
            
    }

    pub fn num_states(&self) -> usize {
        self.state_values.keys().len()
    }

    fn set_scores(&mut self, state: State) {
        if self.state_values.contains_key(&state) {
            return;
        }
        
        for entry in Entry::iterator().filter(|&e| state.is_valid(*e)) {
            for roll in self.possible_rolls.clone() {
                //println!("current {:?} {:?} {:?}", state, *entry, roll);
                //println!("Setting score for {:?}", state.child(*entry, roll));
                self.set_scores(state.child(*entry, roll));
            }
        }

        let score = self.calculate_score(state);
        self.state_values.insert(state, score);

        println!("{:?} {:?}", score, state);
                 
    }

    fn calculate_score(&self, parent_state: State) -> f64 {
        if parent_state.is_terminal() {
            return 0.0;
        }

        for roll in self.possible_rolls.clone() {
            let score = Entry::iterator()
                .filter(|&e| parent_state.is_valid(*e))                
                .map(|&e| parent_state.score(e, roll) as f64 + self.state_values.get(&parent_state.child(e, roll)).unwrap())
                .fold(0./0., f64::max); // Find the largest non-NaN in vector, or NaN otherwise
            return score;
            /*for entry in Entry::iterator().filter(|&e| parent_state.is_valid(*e)) {
                return 1.0;
            }*/
        }
        2.0
    }

    fn get_entry_scores(
        &self,
        entries_taken: [bool; 13],
        positive_yahtzee: bool,
        upper_score_total: i32,
        rolls_left: i32,
        dice: DiceCombination
    ) -> HashMap<Entry, f64> {
        HashMap::new()
    }


    fn get_keepers_scores(
        &self,
        entries_taken: [bool; 13],
        positive_yahtzee: bool,
        upper_score_total: i32,
        rolls_left: i32,
        dice: DiceCombination        
    ) -> HashMap<DiceCombination, f64> {
        HashMap::new()
    }

    fn get_score(
        &self,
        entries_taken: [bool; 13],
        positive_yahtzee: bool,
        upper_score_total: i32,
    ) -> f64 {
        0.0
    }
    
        
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
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

    fn is_valid(&self, entry: Entry) -> bool {
        !self.entries_taken[State::index(entry)]
    }
    
    fn child(&self, entry: Entry, roll: DiceCombination) -> State {
        let mut entries_taken = self.entries_taken.clone();

        let index = State::index(entry);
        entries_taken[index] = true;     

        let upper_score_total = self.new_upper_score(entry, roll);
        let positive_yahtzee = self.positive_yahtzee || ((entry == Yahtzee) && (self.positive_yahtzee));
        
        State {
            entries_taken,
            upper_score_total,
            positive_yahtzee
        }
    }

    fn new_upper_score(&self, entry: Entry, roll: DiceCombination) -> i32 {
        let index = State::index(entry);
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

        let total_score = upper_score_bonus + yahtzee_bonus + dice_score;
        total_score
    }

    fn score_dice(&self, entry: Entry, roll: DiceCombination) -> i32 {
        // implemented as free choice joker rule
        // todo: support configuration for joker rule
        let index = State::index(entry);
        
        let joker = (
            roll.is_yahtzee() &&
                self.positive_yahtzee &&
                self.entries_taken[index]
        );

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
                if counts.contains(&3) && counts.contains(&2) {
                    return 25;
                }
                return 0;
            }
            SmallStraight => 0,
            LargeStraight => 0,
            Yahtzee => 0,
            Chance => roll.dice.iter()
                .enumerate()
                .map(|(i, count)| count * (i as i32 + 1))
                .sum(),
        }
    }

    
    fn is_terminal(&self) -> bool {
        self.entries_taken.iter().all(|&x| x)
    }

    fn index(entry: Entry) -> usize {
        match entry {
            Ones => 0,
            Twos => 1,
            Threes => 2,
            Fours => 3,
            Fives => 4,
            Sixes => 5,
            ThreeOfAKind => 6,
            FourOfAKind => 7,
            FullHouse => 8,
            SmallStraight => 9,
            LargeStraight => 10,
            Yahtzee => 11,
            Chance => 12,
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
            let index: usize = (die - 1) as usize;
            let mut count = combination[index];
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
            .map(|_| 1..=(DICE_SIDES as i32))
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
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
enum Entry {
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
/*
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
        self.set_expected_values()
    }

    fn set_roll_probabilities(&mut self) {
        for dice_number in 0..=DICE_TO_ROLL {
            let probabilities = DiceCombination::probabilities(dice_number);
            self.roll_probabilities.push(probabilities);
        }
    }

    fn set_expected_values(&mut self) {
        let mut stack = vec!(State::default());
        let mut seen: HashSet<State> = HashSet::new();
        seen.insert(State::default());

        while let Some(state) = stack.pop() {
            let mut unseen_children: Vec<_> = state.children()
                .into_iter()
                .filter(|child| !seen.contains(child))
                .collect();

            if unseen_children.len() == 0 {
                let mut widget = Widget::new(state);
                self.expected_values.insert(state, widget.run());
                                            
            } else {
                stack.push(state);
                for child in unseen_children.iter().cloned() {
                    seen.insert(child);
                    stack.push(child);
                }
            }
            println!("values {:?}", self.expected_values);
        }
    }
}

struct Widget {
    parent: State,
    local_state_values: HashMap<WidgetState, f64>,
    child_values: HashMap<State, f64>,
}

impl Widget {

    fn new(state: State) -> Widget {
        Widget {
            parent: state,
            local_state_values: HashMap::new(),
            child_values: HashMap::new(),
        }
    }
    
    fn run(&mut self) -> f64 {
        let widget_state = WidgetState::Roll(LocalState::default());
        self.widget(widget_state);

        println!("{:?} local widget cache: {:?}", self.local_state_values, widget_state);
        5.7
        //*self.local_state_values.get(&widget_state).unwrap()
    }

    fn widget(&mut self, widget_state: WidgetState) {

        match widget_state {
            WidgetState::Child(state) => {}, //widget_state.insert(*self.child_values.get(&state).unwrap(),
            WidgetState::Keeper(local_state) => {}, //5.0,
                /*for roll in local_state.make_roll() \\
                total roll = roll + dice;
                new_state = make new state;
                something something
                 */

            WidgetState::Roll(local_state) => {

            } //4.0,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
struct DiceCombination {
    dice: [i32; DICE_SIDES],
}

impl DiceCombination {

    fn new() -> DiceCombination {

        DiceCombination {
            dice: [0; DICE_SIDES],
        }
    }

    fn from_permutation(permutation: Vec<i32>) -> DiceCombination {
        let mut combination = [0; DICE_SIDES];
        for die in permutation {
            let index: usize = (die - 1) as usize;
            let mut count = combination[index];
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
            .map(|_| 1..=(DICE_SIDES as i32))
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

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
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

    fn children(&self) -> Vec<State> {
        let mut child = State::default();
        child.upper_score_total = 100;
        vec!(child)
    }

    fn child(&self, dice: DiceCombination, entry: Entry) {

    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
struct LocalState {
    rolls_taken: i32,
    dice: DiceCombination,
}

impl LocalState {

    fn default() -> LocalState {
        LocalState {
            rolls_taken: 0,
            dice: DiceCombination::new(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
enum WidgetState {
    Child(State),
    Keeper(LocalState),
    Roll(LocalState),
}

enum Entry {
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_expected_values() {
        let mut advisor = Advisor::new();
        advisor.set_expected_values();

        assert!(false);
    }
}

*/

*/ 
