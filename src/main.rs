extern crate optizee;
use std::io::prelude::*;
use std::io;

fn main() {
    let mut scores = optizee::Scores::new();
    scores.build();
    println!("{:?}", optizee::State::default());
    let idx: optizee::State = 536448_usize.into();
    println!("{:?}", idx);
    let idx: optizee::State = 1048576_usize.into();
        println!("{:?}", idx);

    /*let valid = optizee::valid_states();

    let block_size = 2.11058197_f64 - 2.10648148_f64 + 0.0001_f64;
    let mut scores2 = Vec::new();
    for idx in 0..scores.len() {
        if valid[idx] {
            let block_num_u32 = (scores[idx] / block_size) as u32;
            let block_num = block_num_u32 as u16;
            if block_num as u32 != block_num_u32 {
                panic!();
            }
            scores2.push(block_num);
        }
    }*/
    //let scores2 = scores.iter().map(|&f| (f as f32) as f64).collect::<Vec<f64>>();
    //println!("{}", scores.iter().zip(scores2.iter()).map(|(f1, f2)| (f1 - f2).abs()).max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap());
    
    /*extern crate flate2;
    use flate2::Compression;
    use flate2::read::{GzDecoder};
    use flate2::write::{GzEncoder};

    let read = true;

    if read {
        let f = std::fs::File::create("scores.yht").unwrap();
        let mut e = GzEncoder::new(f, Compression::default());

        let mut scores = optizee::Scores::new();
        scores.build();
        let encoded: Vec<u8> = bincode::serialize(&scores).unwrap();
        e.write_all(&encoded).unwrap();
    }

    else {    
        let mut f = std::fs::File::open("scores.yht").unwrap();
        let mut bytes: Vec<u8> = Vec::new();
        
        f.read_to_end(&mut bytes).unwrap();

        let mut e = GzDecoder::new(&bytes[..]);
        let mut decoded_bytes: Vec<u8> = Vec::new();
        e.read_to_end(&mut decoded_bytes).unwrap();
        let scores: optizee::Scores = bincode::deserialize(&decoded_bytes[..]).unwrap();
        let idx: usize = optizee::State::default().into();
        println!("expected value: {}", scores.state_scores[idx]);
    }*/
}
