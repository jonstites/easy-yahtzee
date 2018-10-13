extern crate optizee;
extern crate permutohedron;
extern crate itertools;

use itertools::Itertools;
use permutohedron::Heap;
use std::collections::HashSet;
use optizee::GameState;
use optizee::DiceCounts;

fn recursive(i: i32, cache: &mut HashSet<i32>) -> i32 {
    cache.insert(i);
    if(i <= 0) {
        return 0;
    }

    i + recursive(i-1, cache)
}

fn main() {
    println!("Hello, world!!@");
    let mut states: Vec<GameState> = Vec::new();
    let mut cache: HashSet<i32> = HashSet::new();
    println!("{}", recursive(5, &mut cache));
    println!("{:?}", cache);
    for i in 0..2100000 {
        let state = GameState {
            entries: [false; 15],
            upper_score: 2,
            dice_kept: DiceCounts::new([0; 6]),
            rolls_left: 1
        };

        states.push(state);
    }

    //let mut data = Vec::new();
    let d = vec![0, 0, 0, 0, 0, 1];
    let mut it = (0..2).map(|i| 1..=6 ).multi_cartesian_product();
    for data in it {
        println!("{:?}", data);
    }
}
