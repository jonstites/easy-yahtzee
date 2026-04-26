use yahtzee_core::Scores;
use yahtzee_wasm::{core_recommend, StateInput};

#[test]
fn opening_large_straight_roll_3() {
    let scores = Scores::new();

    // Round-trip through bincode to exercise the wire format Solver::new uses.
    let bytes = bincode::serde::encode_to_vec(&scores, bincode::config::standard())
        .expect("serialize");
    let (round_tripped, _): (Scores, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .expect("deserialize");
    assert_eq!(scores, round_tripped);

    let input = StateInput {
        entries: [false; 13],
        yahtzee_bonus_eligible: false,
        upper_score_remaining: 63,
    };

    // Opening roll 1-2-3-4-5 on roll 3: large straight dominates.
    let rec = core_recommend(&round_tripped, &input, &[1, 2, 3, 4, 5], 3).unwrap();

    assert!((rec.value - 254.5896).abs() < 0.0001, "value = {}", rec.value);

    // Roll 3 has no keepers; the choice list is `entries`.
    assert!(rec.keepers.is_none(), "roll 3 should have no keepers");
    let top = &rec.entries[0];
    // Index 10 == LARGE_STRAIGHT in ENTRY_ACTIONS.
    assert_eq!(top.entry, 10);
    assert!((top.ev - 261.53375).abs() < 0.01, "top EV = {}", top.ev);
}
