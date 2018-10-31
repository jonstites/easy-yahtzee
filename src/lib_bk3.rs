#![feature(nll)]

extern crate itertools;

use itertools::Itertools;
use std::collections::HashMap;

use std::cmp::max;
use std::cmp::min;

const MAX_DICE: i32 = 5;
const DICE_SIDES: usize = 6;
const UPPER_BONUS: f32 = 35.0;
const YAHTZEE_BONUS: f32 = 100.0;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct DiceCounts {
    counts: [i32; DICE_SIDES]
}

impl DiceCounts {

    pub fn new(counts: [i32; DICE_SIDES]) -> DiceCounts {
        DiceCounts {
            counts
        }
    }

    fn from_vec(vec_counts: Vec<i32>) -> DiceCounts {
        let mut counts = [0; DICE_SIDES];
        //let mut vec_counts_clone = vec_counts.clone();
        for (index, count) in vec_counts.iter().enumerate() {
            counts[index] = *count;
        }

        DiceCounts::new(counts)

    }
    
    fn from_dice(dice: Vec<i32>) -> DiceCounts {
        let mut counts = [0; 6];
        for die in dice {
            let index: usize = (die - 1) as usize;
            let mut count = counts[index];
            counts[index] = count + 1;
        }
        DiceCounts::new(counts)
    }

    fn num_dice(&self) -> i32 {
        self.counts.iter().sum()
    }
    
    fn sum(&self) -> i32 {
        self.counts.iter()
            .enumerate()
            .map(|(value, count)| (value as i32 + 1) * (count))
            .sum()
    }

    fn is_yahtzee(&self) -> bool {
        self.counts.iter().max() >= Some(&5)
    }

    fn largest_sequence_length(&self) -> i32 {
        let mut largest_length = 0;
        let mut current_length = 0;
        
        for die_count in self.counts.iter() {
            if die_count > &0 {
                current_length += 1;
            }
            else {
                current_length = 0;
            }
//            println!("{:?} {:?} {:?} {:?}", self.counts, die_count, current_length, largest_length);
            largest_length = max(current_length, largest_length);
        }
        largest_length
    }
    

    fn possible_keepers(&self) -> Vec<DiceCounts> {
        self.counts.iter()
            .map(|i| 0..=*i)
            .multi_cartesian_product()
            .map(|i| DiceCounts::from_vec(i))
            .collect()
    }
    
}

pub struct DiceCombinations {
    lookup: HashMap<i32, HashMap<DiceCounts, i32>>
}

impl DiceCombinations {

    pub fn new() -> DiceCombinations {
        let mut lookup = HashMap::new();

        
        lookup.entry(0)
            .or_insert(HashMap::new())
            .entry(DiceCounts::new([0; DICE_SIDES]))
            .or_insert(1);
        
        for n in 1..=MAX_DICE {
        
            let mut permutation_it = (0..n)
                .map(|_| 1..=(DICE_SIDES as i32))
                .multi_cartesian_product();
        
            for permutation in permutation_it {
                let counts = DiceCounts::from_dice(permutation);
                let combination_count = lookup.entry(n)
                    .or_insert(HashMap::new())
                    .entry(counts)
                    .or_insert(0);
                *combination_count += 1;
            }
        }
        DiceCombinations {
            lookup
        }
    }

    fn num_dice(&self, n: &i32) -> Option<&HashMap<DiceCounts, i32>> {
        self.lookup.get(n)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Upper {
    Ones = 0,
    Twos = 1,
    Threes = 2,
    Fours = 3,
    Fives = 4,
    Sixes = 5,    
}

#[derive(Clone, Copy, PartialEq)]
enum Lower {
    ThreeOfAKind = 6,
    FourOfAKind = 7,
    FullHouse = 8,
    SmallStraight = 9,
    LargeStraight = 10,
    Yahtzee = 11,
    Chance = 12
}

#[derive(Clone, Copy, PartialEq)]
enum Entry {
    U(Upper),
    L(Lower)
}


impl Entry {
    fn index(&self) -> usize {
        match self {
            Entry::U(e) => *e as usize,
            Entry::L(e) => *e as usize,
        }
    }
    // TODO: Free choice Joker rule!
    fn score(self, dice: DiceCounts) -> i32 {
        use Entry::*;
        use Lower::*;

        match self {
            U(upper) => {
                let index = upper as usize;
                let count = dice.counts[index];
                let value = upper as i32 + 1;
                return value * count;
            }

            L(ThreeOfAKind) => {
                if dice.counts.iter().max() >= Some(&3) {
                    return dice.sum();
                }
                return 0;
            }
            
            L(FourOfAKind) => {
                if dice.counts.iter().max() >= Some(&4) {
                    return dice.sum();
                }
                return 0;
            }

            L(FullHouse) => {
                let counts = dice.counts.iter().collect::<Vec<_>>();
                if counts.contains(&&3) && counts.contains(&&2) {
                    return 25;
                }
                // TODO: This is non-standard!
                if counts.contains(&&5) {
                    return 25;
                }
                return 0;
            }
            
            L(SmallStraight) => {
                if dice.largest_sequence_length() >= 4 {
                    return 30;
                }
                return 0;
            }
            
            L(LargeStraight) => {
                if dice.largest_sequence_length() >= 5 {
                    return 40;
                }
                return 0;
            }

            L(Yahtzee) => {
                if dice.is_yahtzee() {
                    return 50;
                }
                return 0;
            }
            
            L(Chance) => dice.sum(),
        }
    }

}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct State {
    entries: [bool; 14],
    upper_score: i32,
    rolls_left: i32,
    dice: DiceCounts,
}

impl State {

    pub fn new(entries: [bool; 14], upper_score: i32, rolls_left: i32, dice: DiceCounts) -> State {
        State {
            entries,
            upper_score,
            rolls_left,
            dice
        }
    }

    fn game_over(&self) -> bool {
        self.entries[0..13].iter().all(|i| *i)
    }

    fn dice_to_roll(&self) -> i32 {
        MAX_DICE - self.dice.num_dice()
    }

    fn make_roll(&self, other_dice: DiceCounts) -> State {
        let mut dice = self.dice;
        for (index, value) in other_dice.counts.iter().enumerate() {
            dice.counts[index] += value;
        }
        let entries = self.entries;
        let upper_score = self.upper_score;
        let rolls_left = self.rolls_left - 1;

        State {
            entries,
            upper_score,
            rolls_left,
            dice
        }
    }

    fn valid_entries(&self) -> Vec<Entry> {
        use Entry::*;
        use Upper::*;
        use Lower::*;
        static POSSIBILITIES : [Entry; 13] =
            [
                U(Ones),
                U(Twos),
                U(Threes),
                U(Fours),
                U(Fives),
                U(Sixes),
                L(ThreeOfAKind),
                L(FourOfAKind),
                L(FullHouse),
                L(SmallStraight),
                L(LargeStraight),
                L(Yahtzee),
                L(Chance),
            ];

        let mut valid = Vec::new();
        for (index, entry) in POSSIBILITIES.into_iter().enumerate() {
            
            if !self.entries[index] {
                valid.push(*entry);
            }
        }
        valid
    }

    fn child(&self, entry: Entry, dice: DiceCounts) -> State {

        let upper_score = match entry {
            Entry::U(_) => min(63, entry.score(dice) + self.upper_score),
            Entry::L(_) => self.upper_score,
        };

        let mut entries = self.entries;
        entries[entry.index()] = true;

        if (entry == Entry::L(Lower::Yahtzee)) && (dice.is_yahtzee()) {
            entries[13] = true;
        }

        let dice = DiceCounts::new([0; DICE_SIDES]);
        let rolls_left = 3;

        State {
            entries,
            upper_score,
            rolls_left,
            dice
        }
    }

}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
enum GameState {
    Full(State),
    Keepers(State),
}

pub struct ExpectedValues {
    pub ev: HashMap<State, f32>,
}


pub fn widget(
    state: State,
    cache: &mut ExpectedValues,
    dice_combinations: &DiceCombinations) -> f32 {

    let entries = state.entries;
    let upper_score = state.upper_score;

    let mut local_cache = HashMap::new();
    let dice_it = dice_combinations.num_dice(&MAX_DICE).unwrap();
    for dice in dice_it.keys() {
        let mut best_value: f32 = 0.0;
        for entry in state.valid_entries() {
            let child = state.child(entry, *dice);

            let child_score = match child.game_over() {
                true => 0_f32,
                false => {
                    if !cache.ev.contains_key(&child) {
                        let result = widget(child, cache,  dice_combinations);
                        cache.ev.insert(child, result);
                    }
                    *cache.ev.get(&child).unwrap()
                }
            };

            // PLUS will have to add upper score and yahtzee score.
            let dice_score = entry.score(*dice) as f32;
            let upper_bonus = match (child.upper_score, state.upper_score) {
                (63, 63) => 0.0,
                (63, _) => UPPER_BONUS,
                (_, _) => 0.0,
            };

            let yahtzee_bonus = match (dice.is_yahtzee(), state.entries[13]) {
                (true, true) => YAHTZEE_BONUS,
                _ => 0.0,
            };
            let score = child_score + dice_score + upper_bonus + yahtzee_bonus;
            best_value = best_value.max(score);
        }

        let new_state = State {
            entries,
            upper_score,
            rolls_left: 0,
            dice: *dice
        };
        //println!("mistake here? value {:?} {:?}", best_value, new_state);
        local_cache.insert(GameState::Full(new_state), best_value);
    }




    for rolls_left in 1..=3 {
        for num_dice in 0..=MAX_DICE {
            if rolls_left == 3 && num_dice != 0 {
                continue
            }
            
            let dice_it = dice_combinations.num_dice(&num_dice).unwrap();
            for dice in dice_it.keys() {
                let new_state = State {
                    entries,
                    upper_score,
                    rolls_left,
                    dice: *dice
                };
                
                let mut value = 0.0;
                let dice_it2 = dice_combinations.num_dice(&(MAX_DICE as i32 - num_dice)).unwrap();
                for (dice2, num) in dice_it2 {
                    let new_state2 = new_state.make_roll(*dice2);
                    let score = local_cache.get(&GameState::Full(new_state2)).unwrap();
                    value += (*num as f32) * score;
                    //println!("adding new value for keeper from: {:?}, value: {:?}", new_state2, score);                    
                }
                // divide value exponentially let value = value / (DICE_SIDES^ max_dice)
                value = value / dice_combinations.num_dice(&(MAX_DICE as i32 - num_dice)).unwrap().values().sum::<i32>() as f32;
                local_cache.insert(GameState::Keepers(new_state), value);
                //println!("keeprs: {:?}, value: {:?}", new_state, value);
              }
        }

        if rolls_left == 3 {
            continue
        }

        
        let dice_it = dice_combinations.num_dice(&5).unwrap();
        for dice in dice_it.keys() {
            let mut best_value: f32 = 0.0;

            for keepers in dice.possible_keepers() {
                let new_state = State {
                    entries,
                    upper_score,
                    rolls_left,
                    dice: keepers
                };

                let score = local_cache.get(&GameState::Keepers(new_state)).unwrap();
                //println!("a keeper for later is {:?} with {:?}", new_state, score);
                best_value = best_value.max(*score);                
            }

            let new_state = State {
                entries,
                upper_score,
                rolls_left,
                dice: *dice
            };
            
            local_cache.insert(GameState::Full(new_state), best_value);
            //println!("full: {:?}, value: {:?}", new_state, best_value);            
        }
    }

    for (key, value) in local_cache.iter() {
        //println!("{:?}", key);
        //println!("{:?}", value);        
    }
    //println!("set: {:?} {:?}", state, local_cache.get(&GameState::Keepers(state)).unwrap());
    *local_cache.get(&GameState::Keepers(state)).unwrap()

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dicecounts_new_test() {
        let mut counts = [0; DICE_SIDES];
        let dice = DiceCounts::new(counts);
        assert_eq!(dice.counts[0], 0);

        counts[2] = 4;
        let dice = DiceCounts::new(counts);        
        assert_eq!(dice.counts[2], 4);

        counts[2] = 0;
        counts[1] = 1;
        let dice = DiceCounts::new(counts);
        assert_eq!(dice.counts[2], 0);
        assert_eq!(dice.counts[1], 1);        
    }

    #[test]
    fn dicecounts_sum_test() {
        let mut counts = [0; DICE_SIDES];
        let dice = DiceCounts::new(counts);
        assert_eq!(dice.sum(), 0);

        counts[2] = 4;
        let dice = DiceCounts::new(counts);        
        assert_eq!(dice.sum(), 12);

        counts[2] = 0;
        counts[4] = 2;
        let dice = DiceCounts::new(counts);
        assert_eq!(dice.sum(), 10);
    }

    #[test]
    fn dicecounts_is_yahtzee_test() {
        let mut counts = [0; DICE_SIDES];
        let dice = DiceCounts::new(counts);
        assert!(!dice.is_yahtzee());

        counts[2] = 4;
        let dice = DiceCounts::new(counts);        
        assert!(!dice.is_yahtzee());


        counts[3] = 5;
        let dice = DiceCounts::new(counts);        
        assert!(dice.is_yahtzee());

    }

    #[test]
    fn score_upper_test() {
        let counts = [1, 1, 1, 0, 3, 0];
        let dice = DiceCounts::new(counts);

        let entry = Entry::U(Upper::Fives);
        let score = entry.score(dice);
        assert_eq!(score, 15);

        let entry = Entry::U(Upper::Sixes);
        let score = entry.score(dice);
        assert_eq!(score, 0);
    }


    #[test]
    fn score_three_of_a_kind_test() {
        let counts = [1, 1, 1, 0, 2, 0];
        let dice = DiceCounts::new(counts);
        
        let entry = Entry::L(Lower::ThreeOfAKind);
        let score = entry.score(dice);
        assert_eq!(score, 0);

        let counts = [1, 1, 1, 0, 3, 0];
        let dice = DiceCounts::new(counts);

        let score = entry.score(dice);
        assert_eq!(score, 21);
    }

    #[test]
    fn score_four_of_a_kind_test() {
        let counts = [1, 1, 1, 0, 3, 0];
        let dice = DiceCounts::new(counts);
        
        let entry = Entry::L(Lower::FourOfAKind);
        let score = entry.score(dice);
        assert_eq!(score, 0);

        let counts = [1, 1, 1, 0, 4, 0];
        let dice = DiceCounts::new(counts);

        let score = entry.score(dice);
        assert_eq!(score, 26);
    }

    #[test]
    fn score_small_straight_test() {
        let entry = Entry::L(Lower::SmallStraight);

        let counts = [1, 1, 5, 0, 0, 0];
        let dice = DiceCounts::new(counts);
        let score = entry.score(dice);
        assert_eq!(score, 0);

        let counts = [1, 1, 5, 1, 0, 0];
        let dice = DiceCounts::new(counts);
        let score = entry.score(dice);
        assert_eq!(score, 30);

        let counts = [0, 2, 1, 1, 0, 1];
        let dice = DiceCounts::new(counts);
        let score = entry.score(dice);        
        assert_eq!(score, 0);
            
    }

    #[test]
    fn score_large_straight_test() {
        let counts = [1, 1, 5, 1, 0, 0];
        let dice = DiceCounts::new(counts);

        let entry = Entry::L(Lower::LargeStraight);
        let score = entry.score(dice);
        assert_eq!(score, 0);

        let counts = [1, 1, 5, 1, 2, 0];
        let dice = DiceCounts::new(counts);

        let score = entry.score(dice);
        assert_eq!(score, 40);
    }

    #[test]
    fn score_yahtzee_test() {
        let counts = [1, 2, 4, 1, 0, 0];
        let dice = DiceCounts::new(counts);

        let entry = Entry::L(Lower::Yahtzee);
        let score = entry.score(dice);
        assert_eq!(score, 0);

        let counts = [1, 2, 5, 1, 0, 0];
        let dice = DiceCounts::new(counts);

        let score = entry.score(dice);
        assert_eq!(score, 50);
    }

    #[test]
    fn score_chance_test() {
        let counts = [1, 1, 4, 5, 0, 0];
        let dice = DiceCounts::new(counts);

        let entry = Entry::L(Lower::Chance);
        let score = entry.score(dice);
        assert_eq!(score, 35);

        let counts = [1, 1, 4, 4, 0, 0];
        let dice = DiceCounts::new(counts);

        let score = entry.score(dice);
        assert_eq!(score, 31);

        let counts = [1, 1, 4, 4, 5, 0];
        let dice = DiceCounts::new(counts);

        let score = entry.score(dice);
        assert_eq!(score, 56);
    }
    
    
    #[test]
    fn dice_lookups_test() {
        let lookups = DiceCombinations::new();
        println!("zeros: {:?}", lookups.num_dice(&0));
        assert_eq!(lookups.lookup.len() as i32, MAX_DICE + 1);

        match lookups.num_dice(&0) {
            Some(entry) => assert_eq!(entry.len(), 1),
            None => assert!(false),
        }
        
        match lookups.num_dice(&1) {
            Some(entry) => assert_eq!(entry.len(), 6),
            None => assert!(false),
        }
        
        match lookups.num_dice(&2) {
            Some(entry) => assert_eq!(entry.len(), 21),
            None => assert!(false),
        }

        match lookups.num_dice(&5) { 
           Some(entry) => {
                assert_eq!(entry.len(), 252);
                assert_eq!(entry.values().sum::<i32>(), 7776);
            },
            None => assert!(false),
        }        
    }

    #[test]
    fn game_over_test() {
        let entries = [false; 14];
        let upper_score = 10;
        let rolls_left = 0;
        let dice = DiceCounts::new([0; DICE_SIDES]);
        
        let mut state = State {
            entries,
            upper_score,
            rolls_left,
            dice
        };

        assert!(!state.game_over());

        let mut entries = [true; 14];
        state.entries = entries;
        assert!(state.game_over());

        entries[13] = false;
        state.entries = entries;
        assert!(state.game_over());

        entries[4] = false;
        state.entries = entries;
        assert!(!state.game_over());

    }

    //#[test]
    fn widget_test() {
        let mut entries =             [false;14];
        //entries[0] = false;
        let state = State::new(
            entries,
            0,
            3,
            DiceCounts::new([0;6]));

        let mut cache = ExpectedValues { ev: HashMap::new() };
        let dc = DiceCombinations::new();


        let ev = widget(state, &mut cache, &dc);
        println!("{:?}", ev);
        assert_eq!(ev, 1.0);
        
    }
    
}

