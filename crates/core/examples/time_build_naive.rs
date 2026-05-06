//! Times `Scores::new_with(&CpuBuildBackendWith(NaiveBackend))`. Companion to
//! `time_build` (default ndarray) and `cuda_smoke` (CUDA). Used to land my
//! prediction-vs-reality numbers when adding `NaiveBackend`.

use std::time::Instant;

use yahtzee_core::{CpuBuildBackendWith, NaiveBackend, Scores, State};

fn main() {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_string());
    let backend = CpuBuildBackendWith(NaiveBackend);
    let t0 = Instant::now();
    let scores = Scores::new_with(&backend).expect("naive build is infallible");
    let elapsed = t0.elapsed();
    let ev = scores.state_value(State::default());
    println!(
        "threads={threads} backend=naive elapsed_ms={:.1} default_ev={ev:.4}",
        elapsed.as_secs_f64() * 1000.0
    );
}
