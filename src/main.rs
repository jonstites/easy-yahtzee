extern crate optizee;
use std::fs::File;
extern crate flame;
extern crate time;

use time::PreciseTime;
use optizee::{ActionScores, Config};

fn main() {
    let mut action_scores = ActionScores::new(Config::new());
    let mut state = optizee::State::default();
    for i in 0..4 {
        state.entries_taken[i] = true;
    };

    let start = PreciseTime::now();
    action_scores.init_from_state(state);
    let end = PreciseTime::now();
    println!("{} seconds for iterative action scores.", start.to(end));
}
