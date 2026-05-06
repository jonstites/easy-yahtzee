//! Times `Scores::new()` (the rayon-parallelized DP table build) and prints
//! the wall time and the EV of the default state. Used for ad-hoc rayon
//! scaling measurements; not part of the normal test/bench suite.
//!
//! Run with e.g.
//!
//!     RAYON_NUM_THREADS=1  cargo run -p yahtzee-core --release --example time_build
//!     RAYON_NUM_THREADS=8  cargo run -p yahtzee-core --release --example time_build
//!
//! Set `YAHTZEE_UNVALIDATED=1` to drive `Scores::new_with_unvalidated` instead
//! (skips the BFS-reachability filter; ~2× build wall-clock; correctness-
//! identical for reachable states).
//!
//! Set `YAHTZEE_BACKEND=ndarray|naive|simd|faer` to swap the per-state
//! `LinalgBackend` driving the build (`ndarray` is the default; `simd` and
//! `faer` require their respective Cargo features). The 2×2 matrix:
//!
//!     cargo run -p yahtzee-core --release --features simd --example time_build
//!     YAHTZEE_UNVALIDATED=1 cargo run -p yahtzee-core --release --features simd --example time_build
//!     YAHTZEE_BACKEND=simd cargo run -p yahtzee-core --release --features simd --example time_build
//!     YAHTZEE_UNVALIDATED=1 YAHTZEE_BACKEND=simd cargo run -p yahtzee-core --release --features simd --example time_build
//!
//! Output is one line: `threads=<n> backend=<b> mode=<m> elapsed_ms=<ms> default_ev=<ev>`.

use std::time::Instant;

use yahtzee_core::{
    BuildBackend, CpuBuildBackend, CpuBuildBackendWith, NaiveBackend, Scores, State,
};

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string());
    let unvalidated = std::env::var("YAHTZEE_UNVALIDATED").ok().is_some();
    let backend_name = std::env::var("YAHTZEE_BACKEND").unwrap_or_else(|_| "ndarray".to_string());

    let t0 = Instant::now();
    let scores = match backend_name.as_str() {
        "ndarray" => run(&CpuBuildBackend, unvalidated),
        "naive" => run(&CpuBuildBackendWith(NaiveBackend), unvalidated),
        #[cfg(feature = "simd")]
        "simd" => run(
            &CpuBuildBackendWith(yahtzee_core::linalg::SimdBackend::new()),
            unvalidated,
        ),
        #[cfg(feature = "faer")]
        "faer" => run(
            &CpuBuildBackendWith(yahtzee_core::linalg::FaerBackend::new()),
            unvalidated,
        ),
        other => panic!("unknown YAHTZEE_BACKEND={other:?} (or feature not enabled)"),
    };
    let elapsed = t0.elapsed();

    let ev = scores.state_value(State::default());
    let mode = if unvalidated { "unvalidated" } else { "validated" };
    println!(
        "threads={threads} backend={backend_name} mode={mode} elapsed_ms={:.1} default_ev={ev:.4}",
        elapsed.as_secs_f64() * 1000.0
    );
}

fn run<B>(backend: &B, unvalidated: bool) -> Scores
where
    B: BuildBackend,
    B::Error: std::fmt::Debug,
{
    if unvalidated {
        Scores::new_with_unvalidated(backend).expect("build failed")
    } else {
        Scores::new_with(backend).expect("build failed")
    }
}
