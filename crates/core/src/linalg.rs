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
