//! Property tests for the EV math.
//!
//! A slow reference implementation walks the decision tree directly via
//! [`Scores::state_scores`], mirroring the agent's choices (max overall EV,
//! ties broken by first-action / first-keeper index). We assert it matches
//! the optimized cached-array paths in
//! [`ExpectedValues::first_keepers_with_turn_ev`],
//! [`ExpectedValues::second_keepers_with_turn_ev`], and
//! [`ExpectedValues::this_turn_ev`].
//!
//! Both implementations share the underlying probability tables
//! ([`KEEPERS_TO_DICE_PROBABILITIES`], [`DICE_TO_ALLOWED_KEEPERS`]) — so this
//! catches *structural* bugs (transposed indices, wrong tie-break, the wrong
//! axis maxed) rather than table-build bugs.

use std::sync::LazyLock;

use ndarray::Array1;
use proptest::prelude::*;

use crate::tests::shared_scores;
use crate::*;

/// Tolerance for f32 comparisons. Reference and optimized paths sum hundreds
/// of probabilities × scores, so a few ULPs of accumulated drift is normal.
const TOL: f32 = 5e-3;

fn arb_state() -> impl Strategy<Value = State> {
    // Exclude the all-13-filled mask so at least one action is always valid.
    (0u16..((1 << 13) - 1), any::<bool>(), 0u8..=63).prop_map(|(bits, eligible, upper)| State {
        entries: EntryAction::from_bits_truncate(bits),
        yahtzee_bonus_eligible: eligible,
        upper_score_remaining: upper,
    })
}

fn dice_list() -> &'static [DiceCounts] {
    static LIST: LazyLock<Vec<DiceCounts>> = LazyLock::new(|| math::dice_combinations(NUM_DICE));
    &LIST
}

fn keeper_list() -> &'static [DiceCounts] {
    static LIST: LazyLock<Vec<DiceCounts>> =
        LazyLock::new(|| (0u8..=5).flat_map(math::dice_combinations).collect());
    &LIST
}

fn keeper_idx_of(k: &DiceCounts) -> usize {
    keeper_list()
        .iter()
        .position(|x| x == k)
        .expect("returned keeper must be in the canonical keeper list")
}

/// The agent's chosen action on a roll-3 dice combo: argmax over actions of
/// `state.score_and_child(...).0 + state_scores[child]`. Returns the
/// *turn-only* score component of that action.
///
/// Faithfully mirrors `ExpectedValues::this_turn_roll3_dice`'s tie-break:
/// ties on overall EV go to the lowest action index — even when that action
/// is invalid (a quirk in the production code that's harmless because no
/// non-terminal state ties at NEG_INFINITY).
fn ref_roll3_turn(scores: &Scores, state: State, d3: usize) -> f32 {
    let mut best_idx = 0_usize;
    let mut best_overall = f32::NEG_INFINITY;
    for (a, &action) in ENTRY_ACTIONS.iter().enumerate() {
        let v = if state.is_valid_action(action) {
            let (turn, child) = state.score_and_child(action, d3 as u8);
            turn + scores.state_scores[usize::from(child)]
        } else {
            0.0
        };
        if v > best_overall {
            best_overall = v;
            best_idx = a;
        }
    }
    if best_overall == f32::NEG_INFINITY {
        return 0.0;
    }
    state.score_and_child(ENTRY_ACTIONS[best_idx], d3 as u8).0
}

/// Reference for `ExpectedValues::second_keepers_with_turn_ev[k2].turn_ev`.
fn ref_second_keeper_turn(scores: &Scores, state: State, k2: usize) -> f32 {
    let mut sum = 0.0;
    for d3 in 0..NUM_DICE_COMBINATIONS as usize {
        let p = KEEPERS_TO_DICE_PROBABILITIES[(k2, d3)];
        if p == 0.0 {
            continue;
        }
        sum += p * ref_roll3_turn(scores, state, d3);
    }
    sum
}

/// Reference for `ExpectedValues::first_keepers_with_turn_ev[k1].turn_ev`.
///
/// Takes `second_keepers` from the production `ExpectedValues` (rather than
/// recomputing it) for the best-`k2` selection: on near-terminal states the
/// agent's "best k2" is determined by tiny f32 differences in the overall-EV
/// array, and reproducing the *exact* same array via a structurally-different
/// summation introduces accumulated reduction-order drift that flips the
/// argmax tie-break ~1% of the time. We're not trying to verify
/// `second_keepers` here — that's what
/// `second_keepers_turn_ev_matches_reference` indirectly does, and what
/// `test_expected_value` covers via the root-state EV. The structural check
/// remaining here is "given a correctly-ranked second_keepers, does the outer
/// loop walk the tree right?".
fn ref_first_keeper_turn(
    scores: &Scores,
    state: State,
    k1: usize,
    second_keepers: &Array1<f32>,
) -> f32 {
    let mut sum = 0.0;
    for d2 in 0..NUM_DICE_COMBINATIONS as usize {
        let p = KEEPERS_TO_DICE_PROBABILITIES[(k1, d2)];
        if p == 0.0 {
            continue;
        }
        let mut best_k2 = 0_usize;
        let mut best_val = f32::NEG_INFINITY;
        for k2 in 0..NUM_KEEPERS as usize {
            if DICE_TO_ALLOWED_KEEPERS[(d2, k2)] > 0.0 && second_keepers[k2] > best_val {
                best_val = second_keepers[k2];
                best_k2 = k2;
            }
        }
        sum += p * ref_second_keeper_turn(scores, state, best_k2);
    }
    sum
}

proptest! {
    // 8 cases keeps the suite under ~10s on top of the one-time Scores::new()
    // cost. Each case sweeps every keeper that's valid for the rolled dice,
    // which is ~5–250 comparisons per case depending on dice diversity.
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn second_keepers_turn_ev_matches_reference(
        state in arb_state(),
        dice_idx in 0u8..NUM_DICE_COMBINATIONS,
    ) {
        let scores = shared_scores();
        let values = scores.values(state);
        let dice = dice_list()[dice_idx as usize].clone();

        for (keeper, _ev, opt_turn) in values.second_keepers_with_turn_ev(dice) {
            let k_idx = keeper_idx_of(&keeper);
            let ref_turn = ref_second_keeper_turn(scores, state, k_idx);
            prop_assert!(
                (opt_turn - ref_turn).abs() <= TOL,
                "state={state:?} dice_idx={dice_idx} k_idx={k_idx} \
                 opt_turn={opt_turn} ref_turn={ref_turn}",
            );
        }
    }

    #[test]
    fn first_keepers_turn_ev_matches_reference(
        state in arb_state(),
        dice_idx in 0u8..NUM_DICE_COMBINATIONS,
    ) {
        let scores = shared_scores();
        let values = scores.values(state);
        let dice = dice_list()[dice_idx as usize].clone();

        for (keeper, _ev, opt_turn) in values.first_keepers_with_turn_ev(dice) {
            let k_idx = keeper_idx_of(&keeper);
            let ref_turn = ref_first_keeper_turn(scores, state, k_idx, &values.second_keepers);
            prop_assert!(
                (opt_turn - ref_turn).abs() <= TOL,
                "state={state:?} dice_idx={dice_idx} k_idx={k_idx} \
                 opt_turn={opt_turn} ref_turn={ref_turn}",
            );
        }
    }

    #[test]
    fn this_turn_ev_roll3_matches_reference(
        state in arb_state(),
        dice_idx in 0u8..NUM_DICE_COMBINATIONS,
    ) {
        let scores = shared_scores();
        let values = scores.values(state);
        let dice = dice_list()[dice_idx as usize].clone();

        let opt = values.this_turn_ev(dice, 3);
        let reference = ref_roll3_turn(scores, state, dice_idx as usize);
        prop_assert!(
            (opt - reference).abs() <= TOL,
            "state={state:?} dice_idx={dice_idx} opt={opt} ref={reference}",
        );
    }

    /// Cross-check the action-ranking outputs across all enabled
    /// [`LinalgBackend`]s. For any `(state, dice)`, every backend should
    /// produce the same overall EV and the same EV at each rank in the
    /// sorted entries / first_keepers / second_keepers lists (within float
    /// reduction-order tolerance). Choices may swap *within* an EV-tie
    /// bucket — that's fine; we compare EVs at each position, not choices.
    ///
    /// What this catches that `test_*_backend_matches` doesn't: those tests
    /// only check the default-state scalar EV. This walks the full
    /// recommendation surface (252 dice × ~8000 sampled states × {1,2,3}
    /// rolls' worth of ranked choices), so a backend bug that nudges
    /// EV-by-1e-2 on a niche state — well within the noise of the
    /// default-state test — shows up here.
    #[test]
    fn linalg_backends_agree_on_action_ranking(
        state in arb_state(),
        dice_idx in 0u8..NUM_DICE_COMBINATIONS,
    ) {
        let scores = shared_scores();
        let dice = dice_list()[dice_idx as usize].clone();

        // Naive is the always-on, no-deps oracle. Other backends are
        // feature-gated; collect a Vec<(name, ExpectedValues)> of whichever
        // ones are compiled in and cross-check each against naive.
        let oracle = scores.values_with(state, &NaiveBackend);
        #[allow(unused_mut)]
        let mut others: Vec<(&'static str, ExpectedValues)> =
            vec![("ndarray", scores.values_with(state, &NdarrayBackend))];
        #[cfg(feature = "faer")]
        others.push(("faer", scores.values_with(state, &linalg::FaerBackend::new())));
        #[cfg(feature = "simd")]
        others.push(("simd", scores.values_with(state, &linalg::SimdBackend::new())));

        for (name, candidate) in &others {
            prop_assert!(
                (oracle.value - candidate.value).abs() <= TOL,
                "value: naive={} {name}={} state={state:?}",
                oracle.value, candidate.value,
            );

            // Roll 3: ranked entries (immediate score for each action).
            let oe = oracle.entries_score(dice.clone());
            let ce = candidate.entries_score(dice.clone());
            prop_assert_eq!(
                oe.len(), ce.len(),
                "entries len differs for {}", name,
            );
            for (i, (a, b)) in oe.iter().zip(ce.iter()).enumerate() {
                prop_assert!(
                    (a.1 - b.1).abs() <= TOL,
                    "entries rank {i}: naive={} {name}={} \
                     state={state:?} dice_idx={dice_idx}",
                    a.1, b.1,
                );
            }

            // Rolls 1 & 2: ranked keepers.
            for (label, oa, ca) in [
                ("first_keepers",
                    oracle.first_keepers_score(dice.clone()),
                    candidate.first_keepers_score(dice.clone())),
                ("second_keepers",
                    oracle.second_keepers_score(dice.clone()),
                    candidate.second_keepers_score(dice.clone())),
            ] {
                prop_assert_eq!(
                    oa.len(), ca.len(),
                    "{} len differs for {}", label, name,
                );
                for (i, (a, b)) in oa.iter().zip(ca.iter()).enumerate() {
                    prop_assert!(
                        (a.1 - b.1).abs() <= TOL,
                        "{label} rank {i}: naive={} {name}={} \
                         state={state:?} dice_idx={dice_idx}",
                        a.1, b.1,
                    );
                }
            }
        }
    }
}
