extern crate optizee;
use std::fs::File;
extern crate flame;
extern crate time;

use time::PreciseTime;

fn main() {
    let mut action_scores = optizee::ActionScores::new();
    let mut state = optizee::State::default();
    for i in 0..4 {
        state.entries_taken[i] = true;
    };

    let start = PreciseTime::now();
    let states = action_scores.children(state);
    println!("number of states: {:?}", states.len());
    let end = PreciseTime::now();
    println!("{} seconds for children.", start.to(end));

    //println!("{:?}", states);
    let start = PreciseTime::now();
    action_scores.init_from_state(state);//optizee::State::default());
    //println!("number of states: {:?}", action_scores.num_states());
    let end = PreciseTime::now();
    println!("{} seconds for iterative action scores.", start.to(end));
}
