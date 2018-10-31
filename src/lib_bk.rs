extern crate itertools;

use itertools::Itertools;
use std::collections::HashMap;
use std::error::Error;
use std::cmp::max;

const MAX_DICE: i8 = 5;
const NUM_DIE_SIDES: usize = 6;
const UPPER_SCORE_BONUS: i32 = 35;
const YAHTZEE_BONUS: i32 = 100;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct DiceCounts {
    counts: [i8; NUM_DIE_SIDES],
}

pub type DiceCount2 = HashMap<i32, i32>;

pub type DiceCounts3 = [i8; NUM_DIE_SIDES];

pub enum YahtzeeState {
    UNSET,
    ZERO,
    NON_ZERO,
}
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum UpperEntry {
    ONES = 1,
    TWOS = 2,
    THREES = 3,
    FOURS = 4,
    FIVES = 5,
    SIXES = 6,
}

pub enum LowerEntry {
    THREE_OF_A_KIND,
    FOUR_OF_A_KIND,
    SMALL_STRAIGHT,
    LARGE_STRAIGHT,
}


pub enum Entry2 {
    Upper(UpperEntry),
    Lower(LowerEntry),
    YAHTZEE,
}

impl Entry2 {

    fn get_upper_score(&self, dc: DiceCounts3) -> i32 {
        match self {
            Entry2::Upper(value) => {
                let value = *value as usize;
                (value as i8 * dc[(value - 1)]) as i32
            },
            _ => 0
        }

    }
}

pub enum EntryState {
    Entry2(bool),
}



#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct GameState {
    pub entries: [bool; 15],
    pub upper_score: i32,    
    pub rolls_left: i8,
    pub dice_kept: DiceCounts,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Action {
    Entry(usize),
    DiceCounts,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Entry {
    value: usize
}

impl GameState {

    pub fn get_valid_entry_actions(&self) -> Vec<Action> {
        let mut entries = vec!();
        for (index, entry) in self.entries.into_iter().enumerate().filter(|(i, entry)| !**entry) {
            entries.push(Action::Entry(index));
        }
        entries
    }

    pub fn new_child(&self, action: Action, dice_roll: &DiceCounts) -> GameState {
        match action {
            Action::Entry(x) => {
                GameState {
                    entries: self.entries,
                    upper_score: max(self.upper_score + self.upper_score(action, dice_roll), 63),
                    rolls_left: self.rolls_left,
                    dice_kept: self.dice_kept,
                }
            },
            Action::DiceCounts => {println!("dc"); *self},
        }
    }

    pub fn upper_score(&self, entry: Action, dice_roll: &DiceCounts) -> i32 {
        let mut score = 0;
        if let Action::Entry(value) = entry {
            if value < 6 {
                score = ((*dice_roll).counts[value] as usize) * (value + 1);
            }
        }
        score as i32
    }
}

impl DiceCounts {

    pub fn zeros() -> DiceCounts {
        DiceCounts::new([0; 6])
    }

    pub fn new(counts: [i8; 6]) -> DiceCounts {
        let total: i8 = counts.iter().sum();
        if total > MAX_DICE {
            panic!("There should only be {} dice, got {}", MAX_DICE, total);
        }

        for count in counts.iter() {
            if *count < 0 {
                panic!("There should never be negative dice counts, but got {}", count);
            }
        }
        DiceCounts{ counts }
    }

    pub fn from_dice(dice: Vec<i32>) -> DiceCounts {
        let mut counts: [i8; 6] = [0; 6];
        for die in dice {
            let index: usize = (die - 1) as usize;
            let mut count = counts[index];
            counts[index] = count + 1;
        }
        DiceCounts::new(counts)
    }
}


pub fn fill_dice_cache() -> HashMap<i32, HashMap<DiceCounts, i32>> {
    let mut dice_cache = HashMap::new();

    for n in 0..=5 {
        
        let mut dice_it = (0..n).map(|i| 1..=6 ).multi_cartesian_product();
        
        for dice_permutation in dice_it {
            let dice_counts = DiceCounts::from_dice(dice_permutation);
            let combination_count = dice_cache.entry(n)
                .or_insert(HashMap::new())
                .entry(dice_counts)
                .or_insert(0);
            *combination_count += 1;
        }
    }
    dice_cache
}

pub fn fill_widget_cache(state: GameState, cache: &HashMap<GameState, i32>,
    probs: &HashMap<i32, HashMap<DiceCounts, i32>>) -> HashMap<GameState, i32> {

    let mut widget_cache = HashMap::new();
    //let mut keeper_cache = HashMap::new();

    if let Some(dice_map) = probs.get(&1) {
        for dice_roll in dice_map.keys() {
            let widget_state = GameState {
                entries: state.entries,
                upper_score: state.upper_score,
                dice_kept: *dice_roll,
                rolls_left: 0,
            };
            println!("widget state: {:?}", widget_state);

            let mut best_value = 0;
            let mut best_child = widget_state;
            for entry in state.get_valid_entry_actions() {
                //let score = widget_state.upper_score(entry, dice_roll);
                //println!("upper score: {:?}", score);

                let new_state = widget_state.new_child(entry, dice_roll);
                if new_state.upper_score >= best_value {
                    best_value = new_state.upper_score;
                    best_child = new_state;
                }

                //let mut entries = widget_state.entries;
                //entries[entry] = true;
            }
            println!("best child state: {:?}", best_child);
            
            widget_cache.insert(widget_state, best_value);
        }
    }
    widget_cache
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dice_cache_full() {
        let dice_cache = fill_dice_cache();
        assert_eq!(dice_cache.len(), 5);

        match dice_cache.get(&1) {
            Some(entry) => assert_eq!(entry.len(), 6),
            None => assert!(false),
        }
        
        match dice_cache.get(&2) {
            Some(entry) => assert_eq!(entry.len(), 21),
            None => assert!(false),
        }

        match dice_cache.get(&5) { 
           Some(entry) => {
                assert_eq!(entry.len(), 252);
                assert_eq!(entry.values().sum::<i32>(), 7776);
            },
            None => assert!(false),
        }        

    }

    #[test]
    fn test_widget_cache_full() {


        let state = GameState {
            entries: [false; 15],
            upper_score: 0,
            rolls_left: 3,
            dice_kept: DiceCounts::zeros(),
        };

        let cache : HashMap<GameState, i32> = HashMap::new();
        let dice_cache = fill_dice_cache();
        let widget_cache = fill_widget_cache(state, &cache, &dice_cache);


        let e2 = Entry2::Upper(UpperEntry::ONES);

        println!("entry 2 upper score: {:?}", e2.get_upper_score([3, 0, 6, 0, 0, 0]));
        
        assert_eq!(widget_cache.len(), 756);
    }
}
