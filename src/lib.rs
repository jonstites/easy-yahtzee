#![allow(dead_code, unused_variables, unused_parens, unused_imports)]


extern crate itertools;

use itertools::Itertools;
use std::collections::HashMap;
use std::collections::HashSet;
use std::slice::Iter;

mod states;

use self::Entry::*;

const DICE_TO_ROLL: i32 = 5;
const DICE_SIDES: i32 = 6;


struct Config {
    upper_section_bonus: i32,
    yahtzee_bonus: i32,
    dice_to_roll: i32,
    dice_sides: i32
}

impl Default for Config {

    fn default() -> Config {
        Config {
            upper_section_bonus: 35,
            yahtzee_bonus: 100,
            dice_to_roll: 5,
            dice_sides: 6
        }
    }
}

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

        action_scores.set_scores(starting_state);
        println!("{:?}", action_scores.state_values.get(&starting_state));
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
                self.set_scores(state.child(*entry, roll));
            }
        }

        let score = self.calculate_score(state);
        self.state_values.insert(state, score);
    }

    fn calculate_score(&self, parent_state: State) -> f64 {
        if parent_state.is_terminal() {
            return 0.0;
        }

        let mut roll_values = HashMap::new();
        for roll in self.possible_rolls.clone() {
            let score = Entry::iterator()
                .filter(|&e| parent_state.is_valid(*e))                
                .map(|&e| parent_state.score(e, roll) as f64 + self.state_values.get(&parent_state.child(e, roll)).unwrap())
                .fold(std::f64::NAN, f64::max); // Find the largest non-NaN in vector, or NaN otherwise

            roll_values.insert(roll, score);
            //println!("last roll {:?} {:?}", roll, score);
        }

        let mut keeper_values = HashMap::new();
        let mut new_roll_values = HashMap::new();
        
        for roll in self.possible_rolls.clone() {
            let mut best_keeper_value: f64 = 0.0;
            for keeper in roll.possible_keepers() {
                if !keeper_values.contains_key(&keeper) {
                    let dice_to_roll: i32 = DICE_TO_ROLL - (keeper.dice.iter().sum::<i32>());
                    let mut expected_value = 0.0;
                    //println!("{:?} {:?} {:?}", roll, keeper, dice_to_roll);
                    for (keeper_roll, keeper_roll_probability) in self.rolls[dice_to_roll as usize].iter() {
                        let new_dice = keeper.add(keeper_roll);
                        //println!("{:?} {:?} {:?} {:?}", keeper, keeper_roll, dice_to_roll, new_dice);                        
                        let new_dice_expected_value = roll_values.get(&new_dice).unwrap();
                        expected_value += new_dice_expected_value * keeper_roll_probability;
                    }
                    keeper_values.insert(keeper, expected_value);
                    //println!("keeper value {:?} {:?}", expected_value, keeper);
                }
                best_keeper_value = best_keeper_value.max(*keeper_values.get(&keeper).unwrap());

            }
            new_roll_values.insert(roll, best_keeper_value);
            //println!("roll the second {:?} {:?}", roll, best_keeper_value);
        }

        roll_values = new_roll_values;

        let mut keeper_values = HashMap::new();
        let mut new_roll_values = HashMap::new();
        
        for roll in self.possible_rolls.clone() {
            let mut best_keeper_value: f64 = 0.0;
            for keeper in roll.possible_keepers() {
                if !keeper_values.contains_key(&keeper) {
                    let dice_to_roll: i32 = DICE_TO_ROLL - (keeper.dice.iter().sum::<i32>());
                    let mut expected_value = 0.0;
                    //println!("{:?} {:?} {:?}", roll, keeper, dice_to_roll);
                    for (keeper_roll, keeper_roll_probability) in self.rolls[dice_to_roll as usize].iter() {
                        let new_dice = keeper.add(keeper_roll);
                        //println!("{:?} {:?} {:?} {:?}", keeper, keeper_roll, dice_to_roll, new_dice);                        
                        let new_dice_expected_value = roll_values.get(&new_dice).unwrap();
                        expected_value += new_dice_expected_value * keeper_roll_probability;
                    }
                    keeper_values.insert(keeper, expected_value);
                    //println!("keeper value {:?} {:?}", expected_value, keeper);
                }
                best_keeper_value = best_keeper_value.max(*keeper_values.get(&keeper).unwrap());
                new_roll_values.insert(roll, best_keeper_value);
            }
        }
        roll_values = new_roll_values;
        
        let mut expected_value = 0.0;
        let keeper = DiceCombination::new();
        for (keeper_roll, keeper_roll_probability) in self.rolls[5 as usize].iter() {
            let new_dice = keeper.add(keeper_roll);
            let new_dice_expected_value = roll_values.get(&new_dice).unwrap();
            expected_value += new_dice_expected_value * keeper_roll_probability;
            //println!("{:?} {:?} {:?} {:?}", new_dice, new_dice_expected_value, keeper_roll_probability, parent_state);
        }
        expected_value
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
        let positive_yahtzee = self.positive_yahtzee || ((entry == Yahtzee) && (roll.is_yahtzee()));
        
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
        //println!("{:?} {:?} {:?} {:?} {:?}", upper_score_bonus, yahtzee_bonus, dice_score, entry, roll);
        let total_score = upper_score_bonus + yahtzee_bonus + dice_score;
        total_score
    }

    fn score_dice(&self, entry: Entry, roll: DiceCombination) -> i32 {
        // implemented as free choice joker rule
        // todo: support configuration for joker rule
        let index = State::index(entry);
        
        let yahtzee_index = roll.dice.iter().position(|&x| x == 5);
            
        let joker = roll.is_yahtzee() &&
                self.entries_taken[State::index(Yahtzee)] &&
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
struct FullState {
    rolls_remaining: i32,
    dice: DiceCombination
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
enum Fs {
    I(FullState),
    C(FullState)
}

fn best_entry_score(
    minimal_state: &State,
    minimal_state_values: &HashMap<State, f64>,
    full_state: &FullState) -> f64 {

    Entry::iterator()
        .filter(|&e| minimal_state.is_valid(*e))                
        .map(|&e| minimal_state.score(e, full_state.dice) as f64 + minimal_state_values.get(&minimal_state.child(e, full_state.dice)).unwrap())
        .fold(std::f64::NAN, f64::max) // Find the largest non-NaN in vector, or NaN otherwise
        
}

fn average_rolled_dice_score(
    minimal_state: &State,
    roll_probs: &Vec<HashMap<DiceCombination, f64>>,    
    full_state: &FullState,
    full_state_values: &mut HashMap<Fs, f64>)
    -> f64 {


    let keeper_fs = Fs::I(*full_state);
    let keeper = full_state.dice;
    let dice_to_roll: i32 = DICE_TO_ROLL - (keeper.dice.iter().sum::<i32>());
    let mut expected_value = 0.0;
    for (keeper_roll, keeper_roll_probability) in roll_probs[dice_to_roll as usize].iter() {
        let new_dice = keeper.add(keeper_roll);
        let new_fs = Fs::C(FullState{dice: new_dice, rolls_remaining: full_state.rolls_remaining - 1});
        let new_dice_expected_value = *full_state_values.entry(new_fs).or_insert(
            best_keeper_score());
        expected_value += new_dice_expected_value * keeper_roll_probability;
    }
    expected_value
}

fn best_keeper_score() -> f64 {
    0.0
}

fn full_state_calculation(
    full_state: Fs,
    minimal_state: &State,
    minimal_state_values: &HashMap<State, f64>,
    possible_rolls: &Vec<DiceCombination>,
    roll_probs: &Vec<HashMap<DiceCombination, f64>>,
    full_state_values: &mut HashMap<Fs, f64>) -> f64 {

    if minimal_state.is_terminal() {
        return 0.0;
    }

    match full_state {
        Fs::I(ref s)
            if s.rolls_remaining == 0
            => *full_state_values.entry(full_state).or_insert(best_entry_score(minimal_state, minimal_state_values, &s)),
        Fs::I(_) => 2.0,
        Fs::C(_) => 3.0,
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
*/
*/
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


}




