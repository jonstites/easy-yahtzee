extern crate optizee;


fn main() {
    let action_scores = optizee::ActionScores::new();
    println!("number of states: {:?}", action_scores.num_states());
    //action_scores.set_scores(optizee::State::default());
}
