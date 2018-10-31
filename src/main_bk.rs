extern crate optizee;
extern crate permutohedron;
extern crate itertools;

use itertools::Itertools;
use permutohedron::Heap;
use std::collections::HashSet;
use optizee::GameState;
use optizee::DiceCounts;

fn main() {
    let mut it = (0..2).map(|i| 1..=6 ).multi_cartesian_product();
    for data in it {
        println!("{:?}", data);
    }
}
