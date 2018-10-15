extern crate itertools;

use itertools::Itertools;

use std::collections::HashMap;
use std::collections::HashSet;
//use std::iter::Iterator;
use std::cmp::max;


const MAX_DICE: i32 = 5;
const NUM_DIE_FACES: i32 = 6;
    
#[derive(Debug)]
pub struct DiceCounts {
    counts: HashMap<i32, i32>
}

impl DiceCounts {

    pub fn new() -> DiceCounts {
        DiceCounts {
            counts: HashMap::new()
        }
    }

    pub fn insert(&mut self, key: i32, value: i32) {
        // normalize here so zeros are removed
        if value == 0 {
            self.counts.remove(&key);
        }
        else {
            self.counts.insert(key, value);
        }
    }

    pub fn add(&mut self, dice: Vec<i32>) {
        for die in dice {
            let entry = self.counts.entry(die).or_insert(0);
            *entry += 1;
        }
    }

    pub fn sum(&self) -> i32 {
        self.counts.iter()
            .map(|(die, value)| (*die) * (*value))
            .sum()
    }

    pub fn largest_sequence_length(&self) -> i32 {
        let mut dice_sequence = self.counts.iter()
            .filter(|(_die, value)| *value > &0)
            .map(|(die, _value)| die)
            .collect::<Vec<_>>();
        dice_sequence.sort();

        let mut previous_die = 1;
        let mut largest_length = 0;
        let mut current_length = 0;
        
        for die in dice_sequence {
            let die = *die;
            if die == previous_die + 1 {
                current_length += 1;
            }
            else {
                current_length = 1;
            }
            largest_length = max(current_length, largest_length);
            previous_die = die;
        }
        largest_length
    }

    pub fn is_yahtzee(&self) -> bool {
        self.counts.values().max() >= Some(&5)
    }
    
    pub fn value(&self, category: Category) -> i32 {
        match category {
            Category::Upper(upper_category) => {
                let die = upper_category as i32;
                match self.counts.get(&die) {
                    Some(count) => {
                        die * count
                    }
                    None => 0,
                }
            }
            Category::Lower(LowerCategory::ThreeOfAKind) => {
                if self.counts.values().max() >= Some(&3) {
                    return self.sum();
                }
                return 0;
            }
            Category::Lower(LowerCategory::FourOfAKind) => {
                if self.counts.values().max() >= Some(&4) {
                    return self.sum();
                }
                return 0;
            },
            Category::Lower(LowerCategory::FullHouse) => {
                let values = self.counts.values().collect::<HashSet<_>>();
                if values.contains(&3) && values.contains(&2) {
                    return 25;
                }
                return 0;
            }
            Category::Lower(LowerCategory::SmallStraight) => {
                if self.largest_sequence_length() >= 4 {
                    return 35;
                }
                return 0;
            }
            Category::Lower(LowerCategory::LargeStraight) => {
                if self.largest_sequence_length() >= 5 {
                    return 45;
                }
                return 0;
            }
            Category::Lower(LowerCategory::Yahtzee) => {
                if self.is_yahtzee() {
                    return 50;
                }
                return 0;
            }
            Category::Lower(LowerCategory::Chance) => {
                self.sum()
            }
        }
    }
}


/*
struct DiceCombinations {
    
    counts: HashMap<i32, HashMap<DiceCounts, i32>>,
}   

impl DiceCombinations {

    pub fn new(max_dice: i32, num_die_faces: i32) -> DiceCombinations {
        let mut combinations = DiceCombinations {
            counts: HashMap::new()
        };

        combinations.build(max_dice, num_die_faces);
        combinations
    }

    pub fn build(&mut self, max_dice: i32, num_die_faces: i32) {

        for n in 0..=max_dice {
        
            let mut dice_it = (0..n).map(|i| 1..=num_die_faces).multi_cartesian_product();
            
            for dice_permutation in dice_it {
                let dice_counts = DiceCounts::new();
                dice_counts.add(dice_permutation.into_iter().collect());
                
                let combination_count = self.counts.entry(n)
                    .or_insert(HashMap::new())
                    .entry(dice_counts)
                    .or_insert(0);
                *combination_count += 1;
            }
        }
    }
}*/


#[derive(PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Debug)]
pub enum UpperCategory {
    Ones = 1,
    Twos = 2,
    Threes = 3,
    Fours = 4,
    Fives = 5,
    Sixes = 6
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum LowerCategory {
    ThreeOfAKind,
    FourOfAKind,
    FullHouse,
    SmallStraight,
    LargeStraight,
    Yahtzee,
    Chance
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum Category {
    Upper(UpperCategory),
    Lower(LowerCategory),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum YahtzeeState {
    Zero,
    NonZero,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum CategoryState {
    Standard(Category),
    Yahtzee(Category, YahtzeeState),
}

#[derive(Debug)]
pub struct PartialGameState {
    categories: HashSet<CategoryState>,
    upper_value: i32,
}

impl PartialGameState {

    pub fn new() -> PartialGameState {
        PartialGameState {
            categories: HashSet::new(),
            upper_value: 0,
        }
    }

    pub fn child(&self, dice: DiceCounts, category: Category) -> PartialGameState {

        let upper_value = match category {
            Category::Upper(_) => max(dice.value(category) + self.upper_value, 63),
            Category::Lower(_) => self.upper_value
        };

        let category_state = match category {
            Category::Lower(LowerCategory::Yahtzee) => {
                if dice.is_yahtzee() {
                    CategoryState::Yahtzee(category, YahtzeeState::NonZero)
                }
                else {
                    CategoryState::Yahtzee(category, YahtzeeState::Zero)
                }
            }
            _ => CategoryState::Standard(category),
        };
        
        let mut categories = self.categories.clone();
        categories.insert(category_state);
        
        PartialGameState {
            categories,
            upper_value,
        }
    }

    pub fn valid_categories(&self) -> Vec<Category> {
        use UpperCategory::*;
        use LowerCategory::*;
        use Category::*;
        static POSSIBILITIES : [Category; 13] =
            [
                Upper(Ones),
                Upper(Twos),
                Upper(Threes),
                Upper(Fours),
                Upper(Fives),
                Upper(Sixes),
                Lower(ThreeOfAKind),
                Lower(FourOfAKind),
                Lower(FullHouse),
                Lower(SmallStraight),
                Lower(LargeStraight),
                Lower(Yahtzee),
                Lower(Chance),
            ];

        let mut valid = Vec::new();
        for category in POSSIBILITIES.into_iter() {
            if let Lower(Yahtzee) = category {
                let zero = CategoryState::Yahtzee(Lower(Yahtzee), YahtzeeState::Zero);
                let nonzero = CategoryState::Yahtzee(Lower(Yahtzee), YahtzeeState::NonZero);
                if !self.categories.contains(&zero) &&
                    !self.categories.contains(&nonzero) {
                        valid.push(*category);
                }
            }
            else {
                let state = CategoryState::Standard(*category);
                if !self.categories.contains(&state) {
                    valid.push(*category);
                }
            }
        }
        valid
    }
}



#[derive(Debug)]
pub struct LocalRoundState {
    dice: DiceCounts,
    rolls_left: i32,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_largest_sequence_length() {
        let mut dice = DiceCounts::new();
        dice.counts.insert(1, 1);
        dice.insert(2, 1);
        dice.insert(3, 1);
        dice.insert(5, 1);

        let seq_len = dice.largest_sequence_length();
        assert_eq!(seq_len, 3);


        let mut dice = DiceCounts::new();
        dice.insert(1, 1);
        dice.insert(2, 1);
        dice.insert(3, 4);
        dice.insert(4, 1);

        let seq_len = dice.largest_sequence_length();
        assert_eq!(seq_len, 4);
    }

    #[test]
    fn test_score_upper() {

        let mut dice = DiceCounts::new();
        dice.insert(1, 1);
        dice.insert(2, 1);
        dice.insert(3, 1);
        dice.insert(5, 3);

        let category = Category::Upper(UpperCategory::Fives);
        let score = dice.value(category);
        assert_eq!(score, 15);

        let category = Category::Upper(UpperCategory::Sixes);
        let score = dice.value(category);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_score_three_of_a_kind() {
        let mut dice = DiceCounts::new();
        dice.insert(1, 3);
        dice.insert(6, 2);

        let category = Category::Lower(LowerCategory::ThreeOfAKind);
        let score = dice.value(category);
        assert_eq!(score, 15);

        dice.insert(1, 2);
        let score = dice.value(category);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_score_four_of_a_kind() {
        let mut dice = DiceCounts::new();
        dice.insert(1, 4);
        dice.insert(6, 2);

        let category = Category::Lower(LowerCategory::FourOfAKind);
        let score = dice.value(category);
        assert_eq!(score, 16);

        dice.insert(1, 3);
        let score = dice.value(category);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_score_small_straight() {
        let mut dice = DiceCounts::new();
        dice.insert(1, 1);
        dice.insert(2, 1);
        dice.insert(3, 4);
        dice.insert(4, 1);

        let category = Category::Lower(LowerCategory::SmallStraight);
        let score = dice.value(category);
        assert_eq!(score, 35);

        dice.insert(1, 0);
        let score = dice.value(category);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_score_large_straight() {
        let mut dice = DiceCounts::new();
        dice.insert(1, 1);
        dice.insert(2, 1);
        dice.insert(3, 4);
        dice.insert(4, 1);

        let category = Category::Lower(LowerCategory::LargeStraight);
        let score = dice.value(category);
        assert_eq!(score, 0);

        dice.insert(5, 4);
        let score = dice.value(category);
        assert_eq!(score, 45);

        dice.insert(5, 0);
        let score = dice.value(category);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_score_yahtzee() {
        let mut dice = DiceCounts::new();
        dice.insert(1, 1);
        dice.insert(2, 1);
        dice.insert(3, 4);
        dice.insert(4, 5);

        let category = Category::Lower(LowerCategory::Yahtzee);
        let score = dice.value(category);
        assert_eq!(score, 50);

        dice.insert(4, 4);
        let score = dice.value(category);
        assert_eq!(score, 0);

        dice.insert(5, 5);
        let score = dice.value(category);
        assert_eq!(score, 50);
    }

    #[test]
    fn test_score_chance() {
        let mut dice = DiceCounts::new();
        dice.insert(1, 1);
        dice.insert(2, 1);
        dice.insert(3, 4);
        dice.insert(4, 5);

        let category = Category::Lower(LowerCategory::Chance);
        let score = dice.value(category);
        assert_eq!(score, 35);

        dice.insert(4, 4);
        let score = dice.value(category);
        assert_eq!(score, 31);

        dice.insert(5, 5);
        let score = dice.value(category);
        assert_eq!(score, 56);
    }

    #[test]
    fn test_partial_game_state() {
        let state = PartialGameState::new();
        println!("state: {:?}", state);

        assert_eq!(1, 0);
    }
}
