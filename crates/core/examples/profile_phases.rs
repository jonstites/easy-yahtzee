//! Callgrind-friendly hot-loop driver for the `simd_batch` phase functions.
//!
//! Each phase runs in its own `#[inline(never)]` wrapper so callgrind
//! attribution is clean — the per-function instruction / cache / branch
//! counters in the callgrind output map 1:1 to the underlying phase.
//!
//! State is synthetic (`vec![0.0; 1 << 20]` for `state_scores`, 8 default
//! states per batch). The cache-access *pattern* is what callgrind cares
//! about, and that's determined by the `(state, action, dice) → child_idx`
//! computation in `score_and_child` — not by the values being read. So
//! synthetic state_scores give the same memory-access profile as a real
//! built table, at zero setup cost.
//!
//! Run with:
//!
//!     cargo build -p yahtzee-core --release --features simd --example profile_phases
//!     valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes \
//!         --callgrind-out-file=callgrind.out \
//!         ./target/release/examples/profile_phases
//!     callgrind_annotate callgrind.out > callgrind.txt
//!
//! Tune iter counts via env: `ITERS_ENTRY_ACTIONS`, `ITERS_FUSED`,
//! `ITERS_DENSE_GEMV`, `ITERS_DENSE_MASKED_MAX`. Defaults give roughly
//! equal wall time per phase (~10–20 s wall under callgrind's ~30× slow-
//! down).

#[cfg(not(feature = "simd"))]
fn main() {
    eprintln!("Build with --features simd to run this example.");
    std::process::exit(1);
}

#[cfg(feature = "simd")]
fn main() {
    use std::hint::black_box;
    use wide::f32x8;
    use yahtzee_core::linalg::simd_batch::{
        phase_entry_actions_fuse, phase_gemv, phase_masked_max,
    };
    use yahtzee_core::{State, ENTRY_ACTIONS};

    const N_DICE: usize = 252;
    const N_KEEPERS: usize = 462;
    const NUM_STATES: usize = 1 << 20; // matches yahtzee_core's NUM_STATES (private)

    // Synthetic state_scores. Values irrelevant — what matters is that
    // it's a 4 MiB f32 array indexed by child_idx, so the cache footprint
    // matches production.
    let state_scores: Vec<f32> = vec![0.0; NUM_STATES];

    // Eight *different* states at level 1 (each is the default state after
    // taking a different first action on dice 0). Picked this way because
    // production batches are 8 different states at the same level, so the
    // per-batch cache footprint of `state_scores[child_idx]` reads is much
    // bigger than 8 copies of one state would suggest. Using diverse states
    // here makes callgrind's D1mr / DLmr counts meaningful proxies for the
    // production cache behavior.
    let s0 = State::default();
    let states: [State; 8] = std::array::from_fn(|i| s0.score_and_child(ENTRY_ACTIONS[i], 0).1);

    // Realistic intermediate buffers: pre-fill once so each phase below
    // sees a representative input.
    let mut third_dice = vec![f32x8::splat(0.0); N_DICE];
    let mut second_keepers = vec![f32x8::splat(0.0); N_KEEPERS];
    let mut second_dice = vec![f32x8::splat(0.0); N_DICE];
    phase_entry_actions_fuse(&states, &state_scores, &mut third_dice);
    phase_gemv(&mut second_keepers, &third_dice);
    phase_masked_max(&mut second_dice, &second_keepers);

    fn iters(var: &str, default: usize) -> usize {
        std::env::var(var)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    // Default iter counts: roughly 1.0–1.5 s native wall per arm, scaling
    // to 30–50 s under callgrind. entry_actions_fuse is ~30× slower per
    // call than fused_keeper_round, so it gets fewer iters.
    let n_entry = iters("ITERS_ENTRY_ACTIONS", 2_000);
    let n_fused = iters("ITERS_FUSED", 60_000);
    let n_gemv = iters("ITERS_DENSE_GEMV", 5_000);
    let n_max = iters("ITERS_DENSE_MASKED_MAX", 5_000);

    eprintln!(
        "iters: entry_actions={n_entry} fused={n_fused} dense_gemv={n_gemv} \
         dense_masked_max={n_max}"
    );

    bench_entry_actions(n_entry, &states, &state_scores, &mut third_dice);
    bench_fused(n_fused, &mut second_dice, &third_dice);
    bench_dense_gemv(n_gemv, &mut second_keepers, &third_dice);
    bench_dense_masked_max(n_max, &mut second_dice, &second_keepers);

    // Pin the buffers so the optimizer can't elide the loops.
    black_box(&third_dice);
    black_box(&second_keepers);
    black_box(&second_dice);
    eprintln!("OK");
}

#[cfg(feature = "simd")]
#[inline(never)]
fn bench_entry_actions(
    iters: usize,
    states: &[yahtzee_core::State; 8],
    state_scores: &[f32],
    out: &mut [wide::f32x8],
) {
    use std::hint::black_box;
    use yahtzee_core::linalg::simd_batch::phase_entry_actions_fuse;
    for _ in 0..iters {
        phase_entry_actions_fuse(black_box(states), black_box(state_scores), black_box(out));
    }
}

#[cfg(feature = "simd")]
#[inline(never)]
fn bench_fused(iters: usize, out: &mut [wide::f32x8], input: &[wide::f32x8]) {
    use std::hint::black_box;
    use yahtzee_core::linalg::simd_batch::phase_fused_keeper_round;
    for _ in 0..iters {
        phase_fused_keeper_round(black_box(out), black_box(input));
    }
}

#[cfg(feature = "simd")]
#[inline(never)]
fn bench_dense_gemv(iters: usize, out: &mut [wide::f32x8], input: &[wide::f32x8]) {
    use std::hint::black_box;
    use yahtzee_core::linalg::simd_batch::phase_gemv;
    for _ in 0..iters {
        phase_gemv(black_box(out), black_box(input));
    }
}

#[cfg(feature = "simd")]
#[inline(never)]
fn bench_dense_masked_max(iters: usize, out: &mut [wide::f32x8], input: &[wide::f32x8]) {
    use std::hint::black_box;
    use yahtzee_core::linalg::simd_batch::phase_masked_max;
    for _ in 0..iters {
        phase_masked_max(black_box(out), black_box(input));
    }
}
