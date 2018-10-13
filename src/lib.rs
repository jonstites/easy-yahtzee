extern crate itertools;

use itertools::Itertools;
use std::collections::HashMap;
use std::error::Error;

const MAX_DICE: i8 = 5;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct DiceCounts {
    counts: [i8; 6],
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct GameState {
    pub entries: [bool; 15],
    pub upper_score: i8,    
    pub rolls_left: i8,
    pub dice_kept: DiceCounts,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Entry {
    ONES,
    TWOS,
    THREES,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Action {
    Entry,
    DiceCounts
}

impl GameState {

    pub fn get_entries(&self) -> Vec<usize> {
        let mut entries = vec!();
        for (index, entry) in self.entries.into_iter().enumerate().filter(|(i, entry)| **entry) {
            entries.push(index);
        }
        entries
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
    let mut keeper_cache = HashMap::new();

    if let Some(dice_map) = probs.get(&1) {
        for dice_roll in dice_map.keys() {
            let widget_state = GameState {
                entries: state.entries,
                upper_score: state.upper_score,
                dice_kept: *dice_roll,
                rolls_left: 0,
            };
            println!("{:?}", widget_state);

            let best_value = 0;
            for entry in state.get_entries() {
                let mut entries = widget_state.entries;
                entries[entry] = true;
                let new_state = GameState {
                    entries: entries,
                    upper_score: state.upper_score,
                    dice_kept: 
                    
                }
                let score = score_entry(state, new_state, dice);
            }
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

        assert_eq!(widget_cache.len(), 756);
    }
}
