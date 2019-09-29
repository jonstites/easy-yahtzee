extern crate optizee;

fn main() {
    println!("starting!");
    let idx: usize = optizee::State::default().into();
    println!("{:?}", optizee::dice_probability(&[1, 1, 0, 0, 0, 0]));
    println!("{:?}", optizee::dice_probability(&[2, 0, 0, 0, 0, 0]));

    let scores = optizee::scores();
    let idx: usize = optizee::State::default().into();
    println!("expected value: {:?}", scores[idx]);

    let idx = 0b011111_1100011_0_111111_usize;
    println!("expected value: {:?}", scores[idx]);
    
    let state: optizee::State = 0b000000_1111111_0_111111_usize.into();
    let (score, child) = state.score_and_child(0, 56);
    println!("{} to {} score {}", state, child, score);
    let indexes = vec!(
        /*0b111111_1111011_0_111111_usize,
        0b111111_1110011_0_111111_usize,
        0b111111_0000000_0_111111_usize,
        0b000000_1111111_0_111111_usize,
        0b111111_1111011_1_111111_usize,
        0b111111_1110011_1_111111_usize,
        0b111111_0000000_1_111111_usize,
        
        0b000000_1111101_0_111111_usize,
        0b000000_1111100_0_111111_usize,
        0b000000_1111001_0_111111_usize,
        0b000000_1110101_0_111111_usize,
        0b000000_1101101_0_111111_usize,
        0b000000_1011101_0_111111_usize,
        0b000000_0111101_0_111111_usize,
        */

        // with yahtzee bonus and joker rule   140.37896121765158

        // without yahtzee bonus or joker rule 140.37896121765158
        //0b000000_0011110_0_111111_usize,

        // with yahtzee bonus and joker rule   175.09179481140498
        // seems fine
        //0b000000_0011110_1_111111_usize,

        /// with yahtzee bonus and joker rule  270.60719231079656
        /// wrong!
        0b000000_0000010_1_111111_usize
    );

    for idx in indexes {
        let state: optizee::State = idx.into();
        println!("{} ev: {}", state, scores[idx]);
    }

    // 3 sixes
    //let (score, child) = optizee::State::default().score_and_child(5, 90);
    //println!("score 3 6s: {}", score);

    // yahtzee
    //let (score, child) = child.score_and_child(11, 126);
    //println!("score yahtzee: {}", score);

    // yahtzee of 1, large straight
    // large straight
    //let child: optizee::State = 0b011111_1100011_0_111111_usize.into();
    //let (score, child) = child.score_and_child(8, 126);
    //println!("score large straight: {}", score);
    
    /*
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


    // chance
    let (score, child) = child.score_and_child(12, 58);
    println!("score chance: {}", score);

    */
    /*for val in optizee::dice_combinations2(5) {
        println!("{:?}", val);
    }*/
}
