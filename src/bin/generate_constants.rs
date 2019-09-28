use itertools::Itertools;

const NUM_ENTRY_ACTIONS: usize = 13_usize;
const NUM_DICE_ACTIONS: usize = 462_usize;
const NUM_DICE_COMBINATIONS: usize = 252_usize;
const NUM_DICE: usize = 5_usize;
const NUM_DICE_FACES: usize = 6_usize;


fn main() {
    generate_scores();
}

struct Dice(Vec<usize>);

struct DiceCounts([usize; 6]);

impl DiceCounts {

    fn from_vec(v: Vec<usize>) -> Self {
        let mut counts = [0; 6];
        for die in v.into_iter() {
            counts[die] += 1;
        }
        DiceCounts(counts)
    }

    fn score(&self, entry_action: usize) -> usize {
        match entry_action {
            idx if idx < 6 => self.0[entry_action] * (entry_action + 1),
                6 if *self.0.iter().max().unwrap() >= 3_usize => self.0.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>(),
                7 if *self.0.iter().max().unwrap() >= 4_usize => self.0.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>(),
                8 if *self.0.iter().max().unwrap() == 3_usize && *self.0.iter().filter(|&&i| i != 3_usize).max().unwrap() == 2_usize => 25,
                9 if self.0[..4] == [1, 1, 1, 1] || self.0[1..5] == [1, 1, 1, 1] || self.0[2..6] == [1, 1, 1, 1] => 35,
                10 if self.0 == [1, 1, 1, 1, 1, 0] || self.0 == [0, 1, 1, 1, 1, 1] => 45,
                11 if *self.0.iter().max().unwrap() == 5_usize => 50,
                12 => self.0.iter().enumerate().map(|(idx, count)| count * (idx + 1)).sum::<usize>(),
                _ => 0_usize,
        }
    }
}

fn generate_scores() {
    let dice_combinations = (0..NUM_DICE_FACES).map(|_i| 0..NUM_DICE).multi_cartesian_product();
    for (dice_idx, dice_combination) in dice_combinations.enumerate() {
        let dice_counts = DiceCounts::from_vec(dice_combination.clone());

        for entry_action in 0..NUM_ENTRY_ACTIONS {
            let score = dice_counts.score(entry_action);
            println!("{:?} {} {} {}", dice_combination, dice_idx, entry_action, score);
               
            };
        }
    }
