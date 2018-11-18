extern crate optizee;

use optizee::{ActionScores, State};

#[test]
fn test_state_value() {
    let mut action_scores = ActionScores::new();
    let mut starting_state = State::default();
    for i in 1..10 {
        starting_state.entries_taken[i] = true;
    }
    action_scores.init_from_state(starting_state);
    let actual_value = action_scores.value_of_state(starting_state);
    let expected_value = 55.581619_f64;
    let abs_difference = (actual_value - expected_value).abs();
    let tolerance = 0.00001;
    println!("{}", actual_value);
    assert!(abs_difference < tolerance);
}

#[test]
fn test_entry_value() {
    let mut action_scores = ActionScores::new();
    let mut starting_state = State::default();
    for i in 2..10 {
        starting_state.entries_taken[i] = true;
    }
    action_scores.init_from_state(starting_state);
    let actual_value = action_scores.value_of_keepers(vec!(0, 4, 0, 0, 0, 0), 1, starting_state);
    let expected_value = 72.42314_f64;
    let abs_difference = (actual_value - expected_value).abs();
    let tolerance = 0.00001;
    println!("{}", actual_value);
    assert!(abs_difference < tolerance);
}
