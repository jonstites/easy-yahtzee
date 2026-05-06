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

use criterion::{criterion_group, criterion_main, Criterion};
use yahtzee_core::{
    dice_to_counts, recommend, NdarrayBackend, Scores, State, StateInput,
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
/// EV path. NdarrayBackend is the default; whether it dispatches GEMV through
/// `matrixmultiply` or OpenBLAS is a Cargo-feature decision (`--features blas`),
/// so this group really compares "current ndarray build vs. faer". Run with
/// `cargo bench -p yahtzee-core --features faer` to include FaerBackend.
fn bench_backends(c: &mut Criterion) {
    let scores = &*SHARED_SCORES;
    let state = State::default();

    let mut group = c.benchmark_group("state_value/backends");

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

criterion_group!(
    benches,
    bench_state_values,
    bench_recommend,
    bench_with_turn_ev,
    bench_backends,
);
criterion_main!(benches);
