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

use ndarray::{Array1, Zip};

use crate::{DICE_TO_ALLOWED_KEEPERS, KEEPERS_TO_DICE_PROBABILITIES, NUM_DICE_COMBINATIONS};

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

/// faer-based backend (opt-in via `--features faer`). Uses faer's GEMV for
/// `keepers_from_dice` and scalar dot for `initial_roll_ev`. The masked-max
/// in `dice_from_keepers` isn't a linear op so faer doesn't help — we reuse
/// the same Zip-based implementation as `NdarrayBackend`.
///
/// Holds preprocessed `faer::Mat` / `faer::Col` copies of the static
/// probability tables, built once at `FaerBackend::new()` time.
#[cfg(feature = "faer")]
pub use faer_impl::FaerBackend;

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
