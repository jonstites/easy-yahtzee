extern crate optizee;

fn main() {
    println!("starting!");
    let idx: usize = optizee::State::default().into();
    println!("{:?}", optizee::dice_probability(&[1, 1, 0, 0, 0, 0]));
    println!("{:?}", optizee::dice_probability(&[2, 0, 0, 0, 0, 0]));

    let scores = optizee::scores();
    let idx: usize = optizee::State::default().into();
    println!("expected value: {:?}", scores[idx]);
    // 3 sixes
    let (score, child) = optizee::State::default().score_and_child(5, 90);
    println!("score 3 6s: {}", score);

    // 4 fives
    let (score, child) = child.score_and_child(4, 121);
    println!("score 4 5s: {}", score);

    // 3 fours
    let (score, child) = child.score_and_child(3, 81);
    println!("score 3 4s: {}", score);

    // 3 threes
    let (score, child) = child.score_and_child(2, 92);
    println!("score 3 3s: {}", score);

    // 3 twos, upper bonus here?
    let (score, child) = child.score_and_child(1, 58);
    println!("score 3 2s: {}", score);
    // 3 of a kind
    let (score, child) = child.score_and_child(6, 19);
    println!("score 3 of a kind: {}", score);
    // 4 of a kind
    let (score, child) = child.score_and_child(7, 1);
    println!("score 4 of a kind: {}", score);
    // full house
    let (score, child) = child.score_and_child(8, 6);
    println!("score full house: {}", score);
    // small straight
    let (score, child) = child.score_and_child(9, 27);
    println!("score small straight: {}", score);
    // large straight
    let (score, child) = child.score_and_child(10, 76);
    println!("score large straight: {}", score);
    // yahtzee
    let (score, child) = child.score_and_child(11, 126);
    println!("score yahtzee: {}", score);
    // chance
    let (score, child) = child.score_and_child(12, 58);
    println!("score chance: {}", score);

    
    /*for val in optizee::dice_combinations2(5) {
        println!("{:?}", val);
    }*/
}
