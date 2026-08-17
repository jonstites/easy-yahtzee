//! Benchmarks for the per-state EV computation (`Scores::values`), the
//! UI-facing recommendation entry points (`recommend`), and the
//! `*_with_turn_ev` paths that walk the cached arrays.
//!
//! Run with `cargo bench -p yahtzee-core`. Output goes under
//! `target/criterion/<name>/`. To compare runs:
//!
//!     cargo bench -p yahtzee-core -- --save-baseline before
//!     # ...make changes...
//!     cargo bench -p yahtzee-core -- --baseline before
//!
//! `Scores::new()` runs once and is shared across all benches via a
//! `LazyLock`; rebuilding it inside each bench would dominate runtime and
//! drown out the per-call cost we're measuring.

use std::hint::black_box;
use std::sync::LazyLock;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use yahtzee_core::{
    dice_to_counts, recommend, CpuBuildBackendWith, NaiveBackend, NdarrayBackend, Scores, State,
    StateInput,
};

static SHARED_SCORES: LazyLock<Scores> = LazyLock::new(Scores::new);

fn bench_state_values(c: &mut Criterion) {
    let scores = &*SHARED_SCORES;
    let state = State::default();
    c.bench_function("Scores::values(default)", |b| {
        b.iter(|| {
            let v = scores.values(black_box(state));
            black_box(v.value)
        })
    });
    c.bench_function("Scores::state_value(default)", |b| {
        b.iter(|| black_box(scores.state_value(black_box(state))))
    });
}

fn bench_recommend(c: &mut Criterion) {
    let scores = &*SHARED_SCORES;
    let input = StateInput {
        entries: [false; 13],
        yahtzee_bonus_eligible: false,
        upper_score_remaining: 63,
    };
    let dice = [1u8, 2, 3, 4, 5];

    let mut group = c.benchmark_group("recommend(default,[1..5])");
    for roll in [1u8, 2, 3] {
        group.bench_function(format!("roll={roll}"), |b| {
            b.iter(|| {
                let r = recommend(
                    black_box(scores),
                    black_box(&input),
                    black_box(&dice),
                    black_box(roll),
                )
                .expect("valid input");
                black_box(r.value)
            })
        });
    }
    group.finish();
}

fn bench_with_turn_ev(c: &mut Criterion) {
    let scores = &*SHARED_SCORES;
    let values = scores.values(State::default());
    let dice = dice_to_counts(&[1, 2, 3, 4, 5]).unwrap();

    c.bench_function("first_keepers_with_turn_ev(default,[1..5])", |b| {
        b.iter(|| black_box(values.first_keepers_with_turn_ev(black_box(dice.clone()))))
    });
    c.bench_function("second_keepers_with_turn_ev(default,[1..5])", |b| {
        b.iter(|| black_box(values.second_keepers_with_turn_ev(black_box(dice.clone()))))
    });
    c.bench_function("entries_with_turn_ev(default,[1..5])", |b| {
        b.iter(|| black_box(values.entries_with_turn_ev(black_box(dice.clone()))))
    });
}

/// Side-by-side comparison of `LinalgBackend` impls on the same per-state
/// EV path. `naive` is the scalar-`for`-loop reference implementation;
/// `ndarray` is the default (matrixmultiply, or OpenBLAS under
/// `--features blas`); the others are opt-in feature gates. Run with
/// `cargo bench -p yahtzee-core --features faer,simd` to include all CPU
/// variants in one go.
fn bench_backends(c: &mut Criterion) {
    let scores = &*SHARED_SCORES;
    let state = State::default();

    let mut group = c.benchmark_group("state_value/backends");

    let nv = NaiveBackend;
    group.bench_function("naive", |b| {
        b.iter(|| black_box(scores.state_value_with(black_box(state), &nv)))
    });

    let nd = NdarrayBackend;
    group.bench_function("ndarray", |b| {
        b.iter(|| black_box(scores.state_value_with(black_box(state), &nd)))
    });

    #[cfg(feature = "faer")]
    {
        let fa = yahtzee_core::linalg::FaerBackend::new();
        group.bench_function("faer", |b| {
            b.iter(|| black_box(scores.state_value_with(black_box(state), &fa)))
        });
    }

    #[cfg(feature = "simd")]
    {
        let si = yahtzee_core::linalg::SimdBackend::new();
        group.bench_function("simd", |b| {
            b.iter(|| black_box(scores.state_value_with(black_box(state), &si)))
        });
    }

    group.finish();
}

/// End-to-end `Scores::new_with(&backend)` head-to-head. Each iteration
/// runs the full DP table build (`set_valid_states` + 13-level DP fill),
/// so we override Criterion's defaults: sample_size = 10 (otherwise ~100 s
/// per backend) and a long measurement time. Naive can take ~14 s per
/// iter, so the group can run for several minutes total — that's fine,
/// criterion will exceed `measurement_time` to collect its 10 samples.
///
/// Each variant uses [`CpuBuildBackendWith`] so the comparison is a clean
/// "same outer rayon scaffolding, swap only the per-state linalg." The
/// CUDA arm is the exception: it's a different `BuildBackend` impl
/// entirely (per-level GPU pipeline, not per-state CPU walk), reused
/// across iterations via `LazyLock` so we time per-build work and not
/// context init + NVRTC compile (~200 ms one-time).
fn bench_build_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scores::new_with");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let naive = CpuBuildBackendWith(NaiveBackend);
    group.bench_function("naive", |b| {
        b.iter(|| {
            let s = Scores::new_with(black_box(&naive)).unwrap();
            black_box(s.state_value(State::default()))
        })
    });

    let ndarray = CpuBuildBackendWith(NdarrayBackend);
    group.bench_function("ndarray", |b| {
        b.iter(|| {
            let s = Scores::new_with(black_box(&ndarray)).unwrap();
            black_box(s.state_value(State::default()))
        })
    });

    #[cfg(feature = "faer")]
    {
        let faer = CpuBuildBackendWith(yahtzee_core::linalg::FaerBackend::new());
        group.bench_function("faer", |b| {
            b.iter(|| {
                let s = Scores::new_with(black_box(&faer)).unwrap();
                black_box(s.state_value(State::default()))
            })
        });
    }

    #[cfg(feature = "simd")]
    {
        let simd = CpuBuildBackendWith(yahtzee_core::linalg::SimdBackend::new());
        group.bench_function("simd", |b| {
            b.iter(|| {
                let s = Scores::new_with(black_box(&simd)).unwrap();
                black_box(s.state_value(State::default()))
            })
        });
    }

    #[cfg(feature = "cuda")]
    {
        static CUDA_BACKEND: LazyLock<yahtzee_core::linalg::cuda::CudaBuildBackend> =
            LazyLock::new(|| {
                yahtzee_core::linalg::cuda::CudaBuildBackend::new()
                    .expect("CUDA backend init failed")
            });
        let cuda = &*CUDA_BACKEND;
        group.bench_function("cuda", |b| {
            b.iter(|| {
                let s = Scores::new_with(black_box(cuda)).expect("CUDA build failed");
                black_box(s.state_value(State::default()))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_state_values,
    bench_recommend,
    bench_with_turn_ev,
    bench_backends,
    bench_build_backends,
);
criterion_main!(benches);
