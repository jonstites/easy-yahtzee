extern crate itertools;

use itertools::Itertools;
use std::collections::HashMap;
use std::error::Error;

const MAX_DICE: u8 = 5;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct ScoreCard {
    pub entries: [bool; 15],
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct DiceCounts {
    counts: [u8; 6],
}

impl DiceCounts {
    pub fn new(counts: [u8; 6]) -> DiceCounts {
        let total: u8 = counts.iter().sum();
        if total > MAX_DICE {
            panic!("There should only be {} dice, got {}", MAX_DICE, total);
        }
        DiceCounts{ counts }
    }

    pub fn from_dice(dice: Vec<i32>) -> DiceCounts {
        let mut counts: [u8; 6] = [0; 6];
        for die in dice {
            let index: usize = (die - 1) as usize;
            let mut count = counts[index];
            counts[index] = count + 1;
        }
        DiceCounts::new(counts)
    }

    pub fn counts(&self) -> [u8; 6] {
        self.counts
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct State {
    pub score_card: ScoreCard,
    pub dice: DiceCounts,
    pub rolls_left: i8,
    pub upper_score: i8,
}


pub fn get_best_category(entries: [bool; 15], upper_score: i32, dice: DiceCounts, rolls_left: i8) -> i32 {
     5
}

pub fn fill_dice_cache() -> HashMap<i32, HashMap<DiceCounts, i32>> {
    let mut prob_cache = HashMap::new();

    for n in 0..=5 {
        
        let mut dice_it = (0..n).map(|i| 1..=6 ).multi_cartesian_product();
        
        for dice_permutation in dice_it {
            let dice_counts = DiceCounts::from_dice(dice_permutation);
            let combination_count = prob_cache
                .entry(n)
                .or_insert(HashMap::new())
                .entry(dice_counts)
                .or_insert(0);
            *combination_count += 1;
        }
    }
    prob_cache
}

 

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn fill_prob_cache_one() {
        let prob_cache = fill_dice_cache();
        assert_eq!(prob_cache.len(), 5);

        match prob_cache.get(&1) {
            Some(entry) => assert_eq!(entry.len(), 6),
            None => assert!(false),
        }
        match prob_cache.get(&2) {
            Some(entry) => assert_eq!(entry.len(), 21),
            None => assert!(false),
        }

        match prob_cache.get(&5) {
            Some(entry) => {
                assert_eq!(entry.len(), 252);
                assert_eq!(entry.values().sum::<i32>(), 7776);
            },
            None => assert!(false),
        }        

    }
}
