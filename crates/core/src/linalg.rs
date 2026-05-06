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

/// Hand-vectorized backend (opt-in via `--features simd`). Uses ndarray's
/// `.dot()` for the GEMV and a `wide::f32x8`-based loop for the masked-max
/// `dice_from_keepers`. The latter is the op BLAS / faer can't help with,
/// so it's the only place new SIMD code can move the needle.
///
/// Holds a copy of `DICE_TO_ALLOWED_KEEPERS` row-padded to a multiple of 8
/// columns so each row is a clean sequence of `f32x8` lanes.
#[cfg(feature = "simd")]
pub use simd_impl::SimdBackend;

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
