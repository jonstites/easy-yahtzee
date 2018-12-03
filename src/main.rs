extern crate optizee;

extern crate time;

use time::PreciseTime;
use optizee::ScoreData;

fn main() {
    let start = PreciseTime::now();
    let action_scores = ScoreData::new();
    let end = PreciseTime::now();
    println!("{} seconds for entire thing.", start.to(end));
}
