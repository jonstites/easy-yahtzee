use std::collections::HashMap;
use std::collections::HashSet;

pub struct DiceCounts {
    counts: HashMap<UpperCategory, i32>
}

impl DiceCounts {

    pub fn sum(&self) -> i32 {
        let mut total = 0;
        for (category, count) in self.counts.iter() {
            let die_value = (*category) as i32;
            total += die_value * count;
        }
        total
    }

    pub fn largest_sequence(&self) -> i32 {
        3
    }
    
    pub fn value(&self, category: Category) -> i32 {
        match category {
            Category::Upper(upper_category) =>
                match self.counts.get(&upper_category) {
                    Some(count) => {
                        let die_value = (upper_category) as i32;
                        die_value * count
                    }
                    None => 0,
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
                if self.largest_sequence() >= 4 {
                    return 35;
                }
                return 0;
            }
            Category::Lower(LowerCategory::LargeStraight) => {
                if self.largest_sequence() >= 5 {
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
                self.counts.values().sum()
            }
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum UpperCategory {
    Ones = 1,
    Twos = 2,
    Threes = 3,
    Fours = 4,
    Fives = 5,
    Sixes = 6
}

pub enum LowerCategory {
    ThreeOfAKind,
    FourOfAKind,
    FullHouse,
    SmallStraight,
    LargeStraight,
    Yahtzee,
    Chance
}


pub enum Category {
    Upper(UpperCategory),
    Lower(LowerCategory),
}


pub fn a() -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        assert_eq!(a(), 1);
    }
}
