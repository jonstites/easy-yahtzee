use yahtzee_core::Scores;
use yahtzee_wasm::{core_recommend, StateInput};

#[test]
fn opening_large_straight_roll_3() {
    let scores = Scores::new();

    // Round-trip through bincode to exercise the wire format Solver::new uses.
    let bytes = bincode::serialize(&scores).expect("serialize");
    let round_tripped: Scores = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(scores, round_tripped);

    let input = StateInput {
        entries: [false; 13],
        yahtzee_bonus_eligible: false,
        upper_score_remaining: 63,
    };

    // Opening roll 1-2-3-4-5 on roll 3: large straight dominates.
    let rec = core_recommend(&round_tripped, &input, &[1, 2, 3, 4, 5], 3).unwrap();

    assert!((rec.value - 254.5896).abs() < 0.0001, "value = {}", rec.value);

    let entries = rec.entries.expect("roll 3 returns entries");
    let top = &entries[0];
    assert_eq!(top.entry, "large_straight");
    assert!(
        (top.ev - 261.53375).abs() < 0.01,
        "top EV = {}",
        top.ev
    );
}
