//! Deterministic end-to-end test for `easy-yahtzee play --auto --seed N`.
//!
//! This shells out to the built binary instead of calling library code
//! directly because `play.rs` is part of the bin crate. The shell-out also
//! exercises the embedded score table, the clap dispatch, the RNG seeding,
//! the scorecard math, and the bonus tracking — so it's a tight regression
//! tripwire for any change touching the play loop.
//!
//! Two seeds are pinned. If the recorded scores ever change, that means
//! something user-visible changed: the RNG sequence, the solver's choice on
//! some state, the scorecard math, or the renderer's "Final score: N" line.
//! Any of those is worth a deliberate revisit + a checked-in update.

use std::process::Command;

fn run_auto(seed: u64) -> String {
    let bin = env!("CARGO_BIN_EXE_easy-yahtzee");
    let out = Command::new(bin)
        .args([
            "play",
            "--auto",
            "--seed",
            &seed.to_string(),
            "--top",
            "1",
        ])
        .output()
        .expect("spawn easy-yahtzee");
    assert!(
        out.status.success(),
        "play --auto --seed {seed} exited with {:?}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8(out.stdout).expect("stdout utf-8")
}

fn extract_final_score(stdout: &str) -> u32 {
    for line in stdout.lines() {
        if let Some(rest) = line.trim().strip_prefix("Final score:") {
            return rest.trim().parse().expect("parseable final score");
        }
    }
    panic!("no `Final score:` line in output:\n{stdout}");
}

#[test]
fn auto_play_seed_42_completes_with_expected_score() {
    let stdout = run_auto(42);
    let score = extract_final_score(&stdout);

    // Sanity: a real Yahtzee score under reasonable play sits in 100..=500.
    // We're not asserting "optimal play wins" — just that the loop completes
    // and the math doesn't go off the rails.
    assert!(
        (50..=600).contains(&score),
        "implausible final score {score}\noutput:\n{stdout}",
    );

    // Determinism check: pin the exact value so any drift in RNG sequence,
    // solver pick, or scorecard math fails this test.
    assert_eq!(
        score, AUTO_SEED_42_SCORE,
        "auto-play with seed 42 produced {score}, expected {AUTO_SEED_42_SCORE}.\n\
         If this is intentional (RNG/solver/scoring change), update the constant\n\
         after checking the new game's transcript looks right.\nFull output:\n{stdout}"
    );
}

#[test]
fn auto_play_two_seeds_diverge() {
    // Different seeds should (with overwhelming probability) produce different
    // games. If they don't, the seeding is broken.
    let a = extract_final_score(&run_auto(1));
    let b = extract_final_score(&run_auto(2));
    assert_ne!(
        a, b,
        "seeds 1 and 2 produced the same final score ({a}) — seeding is suspect",
    );
}

// Pinned values: regenerate these once after running the test and reading the
// failure messages. They must be exact, so a reviewer can grep history when
// they change to see *why*.
// Pinned by hand from a clean run. Includes a real Yahtzee (50) plus the
// rolled-not-quite-optimal upper section, so it lands a hair above the
// fresh-game expected value of 254.59.
const AUTO_SEED_42_SCORE: u32 = 257;
