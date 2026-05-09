//! Pluggable linear-algebra backends for the per-state EV pipeline.
//!
//! `Scores::state_value` and `Scores::values` walk the three-roll decision
//! tree by alternating two shape-changing ops on the dice-value array:
//!
//! 1. **Per-keeper expected dice value** (`keepers_from_dice`) — a true GEMV:
//!    `out[k] = sum_d KEEPERS_TO_DICE_PROBABILITIES[k,d] * dice_values[d]`,
//!    shape (462, 252) × (252,) → (462,). Called twice per state.
//! 2. **Per-dice "best valid keeper"** (`dice_from_keepers`) — a *masked*
//!    max reduction: `out[d] = max over keepers k where DICE_TO_ALLOWED_KEEPERS[d,k] != 0
//!    of keeper_values[k]`. Not linear; BLAS / matrixmultiply / faer can't help.
//!    Also called twice per state.
//!
//! Plus a final scalar dot product against the initial-roll distribution
//! (`initial_roll_ev`).
//!
//! This module abstracts those three ops behind [`LinalgBackend`] so different
//! GEMV implementations can be benchmarked side-by-side. The trait stays
//! narrow — Yahtzee-specific scoring (entry-action table, joker rule, upper
//! bonus) lives in `Scores` because it's tied to the state encoding rather
//! than dense linalg.
//!
//! `NdarrayBackend` is the default and is what `Scores::state_value` /
//! `Scores::values` use. Other backends (e.g. `FaerBackend` under the `faer`
//! feature) are typically only reached via `Scores::state_value_with` /
//! `Scores::values_with`, which the benches use to compare apples-to-apples.
//!
//! `NaiveBackend` is a bonus: a scalar-`for`-loop reference impl that reads
//! exactly like the math in the trait docstrings. No ndarray, BLAS, faer or
//! SIMD machinery. Useful as a meaningfully-independent oracle for tests
//! (it shares no code with the optimized backends) and as a "honest"
//! baseline so other backends' speedups can be quoted relative to a clear
//! reference rather than to each other.
//!
//! There is a second, coarser-grained trait in this module: [`BuildBackend`].
//! `LinalgBackend` is the *per-state* abstraction (one state, three GEMV/max
//! ops). `BuildBackend` is the *per-level* abstraction: given a batch of
//! states at the same DP level and the read-only state-scores buffer, return
//! the EV of every state in the batch. The CPU implementation
//! ([`CpuBuildBackend`]) is a thin rayon `par_iter` wrapper over the per-state
//! path, so it costs nothing relative to the previous in-line loop. The
//! coarser granularity exists so a future GPU backend can amortize kernel
//! launches across a whole level instead of paying launch overhead per state.

use ndarray::{Array1, Zip};
use rayon::prelude::*;

use crate::{
    state_value_with, DICE_TO_ALLOWED_KEEPERS, KEEPERS_TO_DICE_PROBABILITIES, NUM_DICE_COMBINATIONS,
    NUM_KEEPERS, State,
};

/// The three swappable linear-algebra steps in the per-state EV pipeline.
/// Implementations should be cheap to construct (`Default::default()` is
/// recommended where possible) and stateless or holding only preprocessed
/// matrices; one instance can be reused across many calls.
pub trait LinalgBackend: Send + Sync {
    /// 252 → 462. `out[k] = sum over d of KEEPERS_TO_DICE_PROBABILITIES[k,d] * dice_values[d]`.
    fn keepers_from_dice(&self, dice_values: &Array1<f32>) -> Array1<f32>;

    /// 462 → 252. `out[d] = max over keepers k allowed for d of keeper_values[k]`.
    fn dice_from_keepers(&self, keeper_values: &Array1<f32>) -> Array1<f32>;

    /// 252 → scalar. Marginal of `first_dice` over the initial-roll distribution
    /// (= row 0 of `KEEPERS_TO_DICE_PROBABILITIES`, the "kept nothing → roll 5
    /// fresh dice" probabilities).
    fn initial_roll_ev(&self, first_dice: &Array1<f32>) -> f32;
}

/// Default backend: GEMV via `ndarray::Array2::dot` (which dispatches to
/// `matrixmultiply` by default, or to `cblas_sgemv` under `--features blas`),
/// masked max via a hand-rolled `Zip`, and the scalar dot product via
/// `ArrayView1::dot`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NdarrayBackend;

impl LinalgBackend for NdarrayBackend {
    #[inline]
    fn keepers_from_dice(&self, dice_values: &Array1<f32>) -> Array1<f32> {
        KEEPERS_TO_DICE_PROBABILITIES.dot(dice_values)
    }

    #[inline]
    fn dice_from_keepers(&self, keeper_values: &Array1<f32>) -> Array1<f32> {
        let mut dice: Array1<f32> = Array1::zeros(NUM_DICE_COMBINATIONS as usize);
        Zip::from(&mut dice)
            .and(DICE_TO_ALLOWED_KEEPERS.rows())
            .for_each(|val, mask_row| {
                *val = (&mask_row * keeper_values).fold(0_f32, |acc, elem| acc.max(*elem));
            });
        dice
    }

    #[inline]
    fn initial_roll_ev(&self, first_dice: &Array1<f32>) -> f32 {
        // Row 0 = "kept nothing" = initial-roll distribution over the 252
        // five-dice combos.
        KEEPERS_TO_DICE_PROBABILITIES.row(0).dot(first_dice)
    }
}

/// Reference impl: nested scalar `for` loops over `&[f32]` slices, no
/// ndarray ops, no SIMD, no BLAS, no faer. Reads exactly like the math in
/// the trait docstrings.
///
/// Two purposes:
///
/// 1. **Cross-check oracle.** The other backends all share some flavor of
///    the same matrix machinery (ndarray's `.dot()`, with optional BLAS or
///    SIMD speedups, or faer's GEMV) — none are *meaningfully* independent.
///    A naive scalar loop is independent code; if it disagrees with the
///    others by more than fp tolerance, something's wrong with one of them.
///
/// 2. **Calibration baseline.** Every other backend's speedup is "vs. some
///    optimized baseline." The naive backend is the "if you'd just written
///    the math" baseline, so the speedups can be quoted relative to *that*.
///    At our small problem sizes (252×462 and 462×252) the constant
///    overhead in matrixmultiply / BLAS / faer can eat their asymptotic
///    advantage, so the gap may be smaller than you'd expect.
#[derive(Debug, Default, Clone, Copy)]
pub struct NaiveBackend;

impl LinalgBackend for NaiveBackend {
    #[inline]
    fn keepers_from_dice(&self, dice_values: &Array1<f32>) -> Array1<f32> {
        // out[k] = Σ_d KEEPERS_TO_DICE_PROBABILITIES[k, d] * dice_values[d]
        let probs = KEEPERS_TO_DICE_PROBABILITIES
            .as_slice()
            .expect("KEEPERS_TO_DICE_PROBABILITIES is contiguous");
        let dice = dice_values.as_slice().expect("dice_values is contiguous");
        let n_d = NUM_DICE_COMBINATIONS as usize;
        let mut out = vec![0.0_f32; NUM_KEEPERS as usize];
        for k in 0..NUM_KEEPERS as usize {
            let row = &probs[k * n_d..(k + 1) * n_d];
            let mut acc = 0.0_f32;
            for d in 0..n_d {
                acc += row[d] * dice[d];
            }
            out[k] = acc;
        }
        Array1::from_vec(out)
    }

    #[inline]
    fn dice_from_keepers(&self, keeper_values: &Array1<f32>) -> Array1<f32> {
        // out[d] = max over keepers k where DICE_TO_ALLOWED_KEEPERS[d, k] != 0
        //          of keeper_values[k]
        let mask = DICE_TO_ALLOWED_KEEPERS
            .as_slice()
            .expect("DICE_TO_ALLOWED_KEEPERS is contiguous");
        let kv = keeper_values
            .as_slice()
            .expect("keeper_values is contiguous");
        let n_d = NUM_DICE_COMBINATIONS as usize;
        let n_k = NUM_KEEPERS as usize;
        let mut out = vec![0.0_f32; n_d];
        for d in 0..n_d {
            let row = &mask[d * n_k..(d + 1) * n_k];
            let mut best = 0.0_f32;
            for k in 0..n_k {
                if row[k] != 0.0 && kv[k] > best {
                    best = kv[k];
                }
            }
            out[d] = best;
        }
        Array1::from_vec(out)
    }

    #[inline]
    fn initial_roll_ev(&self, first_dice: &Array1<f32>) -> f32 {
        // out = Σ_d KEEPERS_TO_DICE_PROBABILITIES[0, d] * first_dice[d]
        let probs = KEEPERS_TO_DICE_PROBABILITIES
            .as_slice()
            .expect("KEEPERS_TO_DICE_PROBABILITIES is contiguous");
        let n_d = NUM_DICE_COMBINATIONS as usize;
        let row0 = &probs[..n_d];
        let fd = first_dice.as_slice().expect("first_dice is contiguous");
        let mut acc = 0.0_f32;
        for d in 0..n_d {
            acc += row0[d] * fd[d];
        }
        acc
    }
}

/// Coarser-grained "given a level's worth of states, return their EVs" trait.
/// Used by [`crate::Scores::new_with`] to drive the level-by-level DP fill.
///
/// The CPU implementation ([`CpuBuildBackend`]) is a rayon `par_iter` over
/// the existing per-state pipeline; it's morally identical to the in-line
/// loop the DP previously used.
///
/// The CUDA implementation lives behind the `cuda` Cargo feature and owns
/// the full batched pipeline internally (custom kernels + cuBLAS sgemm),
/// so it can amortize kernel-launch overhead across all states at a level
/// instead of paying it per state.
///
/// `compute_level` is fallible because GPU backends can fail mid-build
/// (out-of-memory, driver disconnect, kernel launch error, etc.). The
/// associated [`Self::Error`] type is `std::convert::Infallible` for CPU
/// backends and a richer error enum for GPU backends.
pub trait BuildBackend: Send + Sync {
    /// Errors a `compute_level` call can raise. CPU backends typically use
    /// `std::convert::Infallible` (which lets callers `.unwrap()` knowing
    /// the call cannot fail at runtime); GPU backends use a richer enum
    /// surfacing driver / cuBLAS / kernel errors.
    type Error: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static;

    /// Compute the overall EV of every state in `states`. The returned
    /// vector is in the same order as the input. `state_scores` is the
    /// (read-only) DP buffer holding EVs for already-computed levels —
    /// every state in `states` is at the same level, and the only entries
    /// of `state_scores` they read are at strictly higher levels (more
    /// entries filled), so the contents at the level-being-computed don't
    /// matter.
    fn compute_level(
        &self,
        states: &[State],
        state_scores: &[f32],
    ) -> Result<Vec<f32>, Self::Error>;
}

/// Default [`BuildBackend`] impl: rayon `par_iter` over the per-state EV
/// path using [`NdarrayBackend`]. Behaves identically to the in-line loop
/// `Scores::set_scores` used before the trait was extracted.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBuildBackend;

impl BuildBackend for CpuBuildBackend {
    // Pure-CPU rayon work: no I/O, no allocation that can fail in any way
    // we'd want to recover from. `Infallible` lets `Scores::new()` (which
    // calls `Scores::new_with(&CpuBuildBackend)`) stay infallible without
    // any `Result`-flavored noise at the public API.
    type Error = std::convert::Infallible;

    fn compute_level(
        &self,
        states: &[State],
        state_scores: &[f32],
    ) -> Result<Vec<f32>, Self::Error> {
        let backend = NdarrayBackend;
        Ok(states
            .par_iter()
            .map(|state| state_value_with(*state, state_scores, &backend))
            .collect())
    }
}

/// Like [`CpuBuildBackend`], but parameterized over the [`LinalgBackend`]
/// used inside the per-state walk. Lets benches and external callers
/// drive a full table build through a non-default linear-algebra
/// implementation (e.g. [`NaiveBackend`], `SimdBackend`, `FaerBackend`)
/// without copy-pasting the rayon scaffolding.
///
/// `CpuBuildBackend` is exactly `CpuBuildBackendWith(NdarrayBackend)` plus
/// a shorter name; both are kept for API ergonomics.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBuildBackendWith<L: LinalgBackend>(pub L);

impl<L: LinalgBackend> BuildBackend for CpuBuildBackendWith<L> {
    type Error = std::convert::Infallible;

    fn compute_level(
        &self,
        states: &[State],
        state_scores: &[f32],
    ) -> Result<Vec<f32>, Self::Error> {
        Ok(states
            .par_iter()
            .map(|state| state_value_with(*state, state_scores, &self.0))
            .collect())
    }
}

/// faer-based backend (opt-in via `--features faer`). Uses faer's GEMV for
/// `keepers_from_dice` and scalar dot for `initial_roll_ev`. The masked-max
/// in `dice_from_keepers` isn't a linear op so faer doesn't help — we reuse
/// the same Zip-based implementation as `NdarrayBackend`.
///
/// Holds preprocessed `faer::Mat` / `faer::Col` copies of the static
/// probability tables, built once at `FaerBackend::new()` time.
#[cfg(feature = "faer")]
pub use faer_impl::FaerBackend;

/// Hand-vectorized backend (opt-in via `--features simd`). Uses ndarray's
/// `.dot()` for the GEMV and a `wide::f32x8`-based loop for the masked-max
/// `dice_from_keepers`. The latter is the op BLAS / faer can't help with,
/// so it's the only place new SIMD code can move the needle.
///
/// Holds a copy of `DICE_TO_ALLOWED_KEEPERS` row-padded to a multiple of 8
/// columns so each row is a clean sequence of `f32x8` lanes.
#[cfg(feature = "simd")]
pub use simd_impl::SimdBackend;

/// Batched-across-states SIMD `BuildBackend` (8 states per `f32x8` lane group).
/// Companion to [`SimdBackend`] (which vectorizes within a single state's
/// masked-max). See [`simd_batch::SimdBatchBuildBackend`] for the design;
/// pairs with [`crate::Scores::new_with_unvalidated`] for regular per-level
/// batches.
#[cfg(feature = "simd")]
pub mod simd_batch;
#[cfg(feature = "simd")]
pub use simd_batch::SimdBatchBuildBackend;

/// CUDA backend (opt-in via `--features cuda`). Implements [`BuildBackend`]
/// (per-level batched), not [`LinalgBackend`] (per-state). See
/// [`cuda::CudaBuildBackend`] for the gory details.
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaBuildBackend;

#[cfg(feature = "simd")]
mod simd_impl {
    use super::{LinalgBackend, DICE_TO_ALLOWED_KEEPERS, KEEPERS_TO_DICE_PROBABILITIES,
        NUM_DICE_COMBINATIONS};
    use ndarray::Array1;
    use wide::f32x8;

    const LANES: usize = 8;
    // 462 keepers padded up to the next multiple of 8 = 464.
    const KEEPER_LANES: usize = ((462 + LANES - 1) / LANES) * LANES;

    pub struct SimdBackend {
        // Row-major, 252 rows × KEEPER_LANES cols, padded with 0.0.
        mask_padded: Vec<f32>,
    }

    impl SimdBackend {
        pub fn new() -> Self {
            let nd = &*DICE_TO_ALLOWED_KEEPERS;
            let (n_dice, n_keepers) = nd.dim();
            assert_eq!(n_dice, NUM_DICE_COMBINATIONS as usize);
            let mut mask_padded = vec![0.0_f32; n_dice * KEEPER_LANES];
            for d in 0..n_dice {
                for k in 0..n_keepers {
                    mask_padded[d * KEEPER_LANES + k] = nd[(d, k)];
                }
                // Trailing KEEPER_LANES - n_keepers slots stay 0.0.
            }
            Self { mask_padded }
        }
    }

    impl Default for SimdBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl LinalgBackend for SimdBackend {
        #[inline]
        fn keepers_from_dice(&self, dice_values: &Array1<f32>) -> Array1<f32> {
            // Same as NdarrayBackend; SIMD doesn't add anything ndarray's
            // GEMV doesn't already do.
            KEEPERS_TO_DICE_PROBABILITIES.dot(dice_values)
        }

        #[inline]
        fn dice_from_keepers(&self, keeper_values: &Array1<f32>) -> Array1<f32> {
            let kv = keeper_values
                .as_slice()
                .expect("keeper_values is contiguous");

            // Copy into a length-KEEPER_LANES scratch with trailing zeros so
            // the multiply-against-mask doesn't pick up garbage in the tail.
            // (Mask tail is already 0; any value here would survive the
            // 0 * x = 0 max contribution. Padding still matters because we
            // load 8 lanes regardless.)
            let mut padded_kv = [0.0_f32; KEEPER_LANES];
            padded_kv[..kv.len()].copy_from_slice(kv);

            let mut out: Array1<f32> = Array1::zeros(NUM_DICE_COMBINATIONS as usize);
            // SAFETY: KEEPER_LANES is divisible by LANES; both arrays are aligned f32.
            let kv_chunks: &[[f32; LANES]] = bytemuck::cast_slice(&padded_kv);

            for d in 0..NUM_DICE_COMBINATIONS as usize {
                let row_start = d * KEEPER_LANES;
                let mask_row = &self.mask_padded[row_start..row_start + KEEPER_LANES];
                let mask_chunks: &[[f32; LANES]] = bytemuck::cast_slice(mask_row);

                let mut acc = f32x8::splat(0.0);
                for (m_chunk, k_chunk) in mask_chunks.iter().zip(kv_chunks.iter()) {
                    let m = f32x8::from(*m_chunk);
                    let k = f32x8::from(*k_chunk);
                    acc = acc.fast_max(m * k);
                }
                // wide 1.3 has no horizontal max; scalar fold over 8 lanes.
                let lanes: [f32; LANES] = acc.into();
                out[d] = lanes.iter().copied().fold(0.0_f32, f32::max);
            }
            out
        }

        #[inline]
        fn initial_roll_ev(&self, first_dice: &Array1<f32>) -> f32 {
            // Same as NdarrayBackend.
            KEEPERS_TO_DICE_PROBABILITIES.row(0).dot(first_dice)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_impl {
    use super::{LinalgBackend, DICE_TO_ALLOWED_KEEPERS, KEEPERS_TO_DICE_PROBABILITIES,
        NUM_DICE_COMBINATIONS};
    use faer::{Col, Mat};
    use ndarray::{Array1, Zip};

    pub struct FaerBackend {
        // (462, 252) preprocessed copy of KEEPERS_TO_DICE_PROBABILITIES.
        keepers_to_dice: Mat<f32>,
        // Length-252 row-0 of the same matrix (initial-roll distribution).
        first_roll: Col<f32>,
    }

    impl FaerBackend {
        pub fn new() -> Self {
            let nd = &*KEEPERS_TO_DICE_PROBABILITIES;
            let (rows, cols) = nd.dim();
            let keepers_to_dice = Mat::from_fn(rows, cols, |i, j| nd[(i, j)]);
            let first_roll = Col::from_fn(cols, |j| nd[(0, j)]);
            Self {
                keepers_to_dice,
                first_roll,
            }
        }
    }

    impl Default for FaerBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl LinalgBackend for FaerBackend {
        #[inline]
        fn keepers_from_dice(&self, dice_values: &Array1<f32>) -> Array1<f32> {
            // Borrow ndarray slice as a faer column without copying.
            let slice = dice_values.as_slice().expect("dice_values contiguous");
            let dice_col = faer::ColRef::from_slice(slice);

            // GEMV via faer's Mul overload (Mat * Col -> Col).
            let out: Col<f32> = &self.keepers_to_dice * dice_col;

            // Convert faer Col -> ndarray Array1 (one allocation; same shape as
            // ndarray's `.dot()` path, which also returns an owned Array1).
            Array1::from_iter(out.iter().copied())
        }

        #[inline]
        fn dice_from_keepers(&self, keeper_values: &Array1<f32>) -> Array1<f32> {
            // Masked max-reduction; faer doesn't help. Identical to NdarrayBackend.
            let mut dice: Array1<f32> = Array1::zeros(NUM_DICE_COMBINATIONS as usize);
            Zip::from(&mut dice)
                .and(DICE_TO_ALLOWED_KEEPERS.rows())
                .for_each(|val, mask_row| {
                    *val = (&mask_row * keeper_values).fold(0_f32, |acc, elem| acc.max(*elem));
                });
            dice
        }

        #[inline]
        fn initial_roll_ev(&self, first_dice: &Array1<f32>) -> f32 {
            let slice = first_dice.as_slice().expect("first_dice contiguous");
            let v = faer::ColRef::from_slice(slice);
            // Inner product via faer's column-vs-column dot.
            (0..v.nrows()).map(|i| self.first_roll[i] * v[i]).sum()
        }
    }
}
