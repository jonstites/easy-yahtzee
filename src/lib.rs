use std::collections::HashMap;
use std::collections::HashSet;
use std::cmp::max;

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
        self.counts.insert(key, value);
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
                if self.counts.values().max() >= Some(&5) {
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

#[derive(PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
pub enum UpperCategory {
    Ones = 1,
    Twos = 2,
    Threes = 3,
    Fours = 4,
    Fives = 5,
    Sixes = 6
}

#[derive(Clone, Copy)]
pub enum LowerCategory {
    ThreeOfAKind,
    FourOfAKind,
    FullHouse,
    SmallStraight,
    LargeStraight,
    Yahtzee,
    Chance
}

#[derive(Clone, Copy)]
pub enum Category {
    Upper(UpperCategory),
    Lower(LowerCategory),
}

pub enum YahtzeeState {
    Unset,
    Zero,
    NonZero,
}

pub enum CategoryState {
    Standard(bool),
    Yahtzee(YahtzeeState),
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
    
}
