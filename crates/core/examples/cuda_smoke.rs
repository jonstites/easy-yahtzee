//! CUDA wiring smoke test. Initializes [`CudaBuildBackend`], runs the full DP
//! fill via `Scores::new_with`, and asserts the default-state EV matches the
//! known good value (~254.5897). With the skeleton's CPU-fallback
//! `compute_level`, this just proves the cudarc context / table uploads work
//! and the feature-gated wiring compiles into a runnable binary. With the
//! real kernels in place, this becomes the correctness smoke test.
//!
//! Run with `cargo run -p yahtzee-core --release --example cuda_smoke --features cuda`.

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;
    use yahtzee_core::{linalg::cuda::CudaBuildBackend, Scores, State};

    let t0 = Instant::now();
    let backend = CudaBuildBackend::new().expect("CUDA init failed");
    println!("CUDA context + table upload: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);

    let t1 = Instant::now();
    let scores = Scores::new_with(&backend);
    let elapsed = t1.elapsed();
    let ev = scores.state_value(State::default());
    println!(
        "Scores::new_with(CudaBuildBackend) elapsed_ms={:.1} default_ev={ev:.4}",
        elapsed.as_secs_f64() * 1000.0
    );

    let want = 254.5896;
    assert!(
        (ev - want).abs() < 1e-3,
        "default-state EV diverged: got {ev}, want ~{want}",
    );
    println!("OK");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Build with --features cuda to run this example.");
    std::process::exit(1);
}
