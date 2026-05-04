//! Times `Scores::new()` (the rayon-parallelized DP table build) and prints
//! the wall time and the EV of the default state. Used for ad-hoc rayon
//! scaling measurements; not part of the normal test/bench suite.
//!
//! Run with e.g.
//!
//!     RAYON_NUM_THREADS=1  cargo run -p yahtzee-core --release --example time_build
//!     RAYON_NUM_THREADS=8  cargo run -p yahtzee-core --release --example time_build
//!
//! Output is one line: `threads=<n> elapsed_ms=<ms> default_ev=<ev>`.

use std::time::Instant;

use yahtzee_core::{Scores, State};

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string());
    let t0 = Instant::now();
    let scores = Scores::new();
    let elapsed = t0.elapsed();
    let ev = scores.state_value(State::default());
    println!(
        "threads={threads} elapsed_ms={:.1} default_ev={ev:.4}",
        elapsed.as_secs_f64() * 1000.0
    );
}
