//! Outer-loop SIMD: vectorize the per-state EV pipeline *across* states
//! rather than within a single state's masked-max, plus a sparse-fused
//! keeper round that merges each GEMV→masked_max pair into one CSR walk.
//!
//! [`SimdBackend`] (in the parent module) is the per-state intra-state
//! vectorizer: `wide::f32x8` lanes are the 8 keepers being max-reduced for
//! one state. [`SimdBatchBuildBackend`] flips the axis: `f32x8` lanes are 8
//! *different states* at the same DP level, and the same matrix /
//! probability tables are broadcast scalar across lanes. This works because
//! the per-state pipeline's structure is identical state-to-state — the
//! only state-dependent inputs are (a) which entry actions are legal, (b)
//! the per-action `(score, child)` look-up, and (c) the resulting
//! `state_scores[child]` reads. Everything downstream (the GEMVs, the
//! masked-maxes, the final dot) sees the matrices as constants.
//!
//! On top of that outer-loop SIMD, [`phase_fused_keeper_round`] exploits the
//! 3.75% structural sparsity of `KEEPERS_TO_DICE_PROBABILITIES` (4,368 of
//! 116,424 entries are nonzero — each keeper `k` is compatible only with
//! full rolls `d` that contain it as a sub-multiset). A single CSR drives
//! both the GEMV (sum over `d` in supp(K2D[k, ·])) and the masked-max
//! (max over compatible `d`s) in one walk, scatter-maxing the dot-product
//! result directly into `out_dice` without ever materializing the 462-wide
//! `second_keepers` / `first_keepers` intermediate. End-to-end this is
//! ~2.9× over the dense outer-loop SIMD path, and the fused phase alone is
//! 26.3× vs the dense `phase_gemv` + `phase_masked_max` pair (matching the
//! 1/0.0375 = 26.7× theoretical density-ratio ceiling).
//!
//! This backend works on both [`Scores::new_with`](crate::Scores::new_with)
//! (validated) and [`Scores::new_with_unvalidated`](crate::Scores::new_with_unvalidated):
//! the unvalidated per-level batches are exactly `C(13, L) × 128` states,
//! all divisible by 8, so no scalar tail handling is needed; the validated
//! path's tail (<8 states) falls back to the per-state [`SimdBackend`].
//!
//! Allocations live in a per-thread [`Scratch`] reused across chunks via
//! rayon's `map_init`. Working set is ~24 KiB per thread (three 252-wide
//! `f32x8` buffers); the sparse-fused round eliminated the 462-wide
//! intermediates that previously pushed it to ~50 KiB.

use std::convert::Infallible;
use std::sync::LazyLock;

use rayon::prelude::*;
use wide::f32x8;

use super::{BuildBackend, SimdBackend};
use crate::{
    state_value_with, DICE_TO_ALLOWED_KEEPERS, ENTRY_ACTIONS, KEEPERS_TO_DICE_PROBABILITIES,
    NUM_DICE_COMBINATIONS, NUM_KEEPERS, State,
};

const LANES: usize = 8;
const N_DICE: usize = NUM_DICE_COMBINATIONS as usize; // 252
const N_KEEPERS: usize = NUM_KEEPERS as usize; // 462
const N_ACTIONS: usize = 13;

/// CSR-format sparse representation of `KEEPERS_TO_DICE_PROBABILITIES`.
///
/// `KEEPERS_TO_DICE_PROBABILITIES` is structurally sparse — `K2D[k, d] > 0`
/// iff `d ⊇ k` (i.e., the kept multiset `k` is a sub-multiset of the full
/// roll `d`). Counting per keeper-size: 1·252 + 6·126 + 21·56 + 56·21 +
/// 126·6 + 252·1 = **4,368 nonzeros** out of 462 × 252 = 116,424 ≈ **3.75%
/// density**, mean ≈9.5 nz per row.
///
/// Crucially, **the same nonzero pattern serves both ops** in the per-level
/// pipeline:
///
/// - GEMV step: `out[k] = sum over d in nonzeros[k] of K2D[k, d] * input[d]`.
/// - Masked-max step: `mask[d, k] = 1 ⟺ k ⊆ d ⟺ K2D[k, d] > 0`, so the
///   list of `d`s compatible with each `k` is exactly the same column
///   indices.
///
/// So one CSR over K2D drives both ops, which lets `phase_fused_keeper_round`
/// compute one row of K2D and immediately scatter-max it into the output
/// without ever materializing the 462-wide `second_keepers` intermediate.
///
/// Built once via `LazyLock` — read-only afterwards, shared across all
/// threads / pipeline calls.
struct SparseK2D {
    /// Row offsets, length `N_KEEPERS + 1`. Row `k` spans
    /// `col_idx[row_ptr[k]..row_ptr[k+1]]` (and same for `values`).
    row_ptr: Vec<u32>,
    /// Dice-index of each nonzero, length 4,368. Sorted ascending within
    /// each row so the scatter-max walks `out_dice` in cache-friendly order.
    col_idx: Vec<u16>,
    /// `K2D[k, d]` probabilities, parallel to `col_idx`.
    values: Vec<f32>,
}

static SPARSE_K2D: LazyLock<SparseK2D> = LazyLock::new(|| {
    let m = &*KEEPERS_TO_DICE_PROBABILITIES;
    debug_assert_eq!(m.shape(), &[N_KEEPERS, N_DICE]);
    let mut row_ptr = Vec::with_capacity(N_KEEPERS + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for k in 0..N_KEEPERS {
        for d in 0..N_DICE {
            let p = m[(k, d)];
            if p > 0.0 {
                col_idx.push(d as u16);
                values.push(p);
            }
        }
        row_ptr.push(col_idx.len() as u32);
    }
    debug_assert_eq!(col_idx.len(), 4368, "expected 4368 nonzeros");
    debug_assert_eq!(values.len(), 4368);
    SparseK2D {
        row_ptr,
        col_idx,
        values,
    }
});

/// Per-thread scratch for the 8-wide pipeline. Reused across batches via
/// rayon's `map_init` so the working set isn't reallocated 27k times per
/// heavy DP level.
///
/// Note: since [`phase_fused_keeper_round`] computes `second_dice` directly
/// from `third_dice` (and `first_dice` directly from `second_dice`) without
/// ever materializing the intermediate `second_keepers` / `first_keepers`
/// 462-wide vectors, only three 252-wide buffers remain. Working set is now
/// ~24 KiB per thread (down from ~50 KiB).
struct Scratch {
    third_dice: Vec<f32x8>,  // length N_DICE
    second_dice: Vec<f32x8>, // length N_DICE
    first_dice: Vec<f32x8>,  // length N_DICE
}

impl Scratch {
    fn new() -> Self {
        Self {
            third_dice: vec![f32x8::splat(0.0); N_DICE],
            second_dice: vec![f32x8::splat(0.0); N_DICE],
            first_dice: vec![f32x8::splat(0.0); N_DICE],
        }
    }
}

/// `BuildBackend` that processes 8 states per `f32x8` lane group through the
/// full per-state EV pipeline. Drives `Scores::new_with_unvalidated` at
/// per-thread parallelism via `par_chunks(8)`. Falls back to the per-state
/// `SimdBackend` for any tail `< 8` (only relevant on the validated path;
/// unvalidated batches are always a multiple of 128).
pub struct SimdBatchBuildBackend;

impl BuildBackend for SimdBatchBuildBackend {
    type Error = Infallible;

    fn compute_level(
        &self,
        states: &[State],
        state_scores: &[f32],
    ) -> Result<Vec<f32>, Infallible> {
        // par_chunks_exact(8) feeds the SIMD core; the remainder (0..8) goes
        // through the scalar per-state SimdBackend so the same code path
        // serves validated batches (which can have an arbitrary tail).
        let chunks = states.par_chunks_exact(LANES);
        let remainder: Vec<State> = chunks.remainder().to_vec();

        let mut out: Vec<f32> = chunks
            .map_init(Scratch::new, |scratch, chunk| {
                let group: &[State; LANES] = chunk.try_into().unwrap();
                compute_lanes(group, state_scores, scratch)
            })
            .flat_map_iter(|arr| arr.into_iter())
            .collect();

        for state in &remainder {
            out.push(state_value_with(*state, state_scores, &SimdBackend::new()));
        }
        Ok(out)
    }
}

/// The 8-wide pipeline. Returns the EV of each of the 8 input states.
///
/// Each `f32x8` lane is one of the 8 input states. The matrices and masks
/// are broadcast scalar-into-lanes via `f32x8::splat`. Steps 1+2 fuse the
/// per-state `entry_actions` lookup with the max-over-actions reduce —
/// never materializes the 13 × 252 × 8 intermediate, streams into
/// `third_dice` directly. Steps 3+4 (and 5+6) fuse via
/// [`phase_fused_keeper_round`]: sparse CSR over `K2D` lets each row's
/// dot product be computed and immediately scatter-maxed into `out_dice`
/// without ever materializing the 462-wide `second_keepers` /
/// `first_keepers` intermediate.
///
/// Phases are factored into [`phase_entry_actions_fuse`],
/// [`phase_fused_keeper_round`], and [`phase_final_dot`] (plus the dense
/// [`phase_gemv`] / [`phase_masked_max`] kept around as the non-fused
/// baseline for the criterion `simd_batch_phases` group).
fn compute_lanes(
    states: &[State; LANES],
    state_scores: &[f32],
    scratch: &mut Scratch,
) -> [f32; LANES] {
    // Steps 1+2: per-state entry_actions × max-over-actions, into third_dice.
    phase_entry_actions_fuse(states, state_scores, &mut scratch.third_dice);

    // Steps 3+4 fused: second_dice[d] = max over k where d ⊇ k of
    //   sum over d' in supp(K2D[k, ·]) of K2D[k, d'] * third_dice[d'].
    // Sparse CSR over K2D drives both the dot product and the scatter-max
    // in one pass; second_keepers is never materialized.
    phase_fused_keeper_round(&mut scratch.second_dice, &scratch.third_dice);

    // Steps 5+6: same as 3+4 but starting from second_dice.
    phase_fused_keeper_round(&mut scratch.first_dice, &scratch.second_dice);

    // Step 7: ev = sum over d of initial_dist[d] * first_dice[d].
    phase_final_dot(&scratch.first_dice)
}

/// Steps 1+2 of the per-level pipeline: fill `out_third_dice[d]` with the
/// max over actions of `entry_score(action, d) + state_scores[child(d, action)]`,
/// where each `f32x8` lane corresponds to one of the 8 input states.
///
/// Per-state work (`is_valid_action`, `score_and_child`, `state_scores[child_idx]`)
/// is scalar — the lane axis is purely outer-loop SIMD over states. Invalid
/// actions contribute 0.0 to their lane, matching `entry_actions_array`'s
/// semantics; the max reduction picks them up only if every action is
/// invalid (terminal state), which can't happen in `compute_level` since
/// `set_scores` never hits level 13.
///
/// Exposed as `#[doc(hidden)] pub` so the criterion `simd_batch_phases`
/// group can call it directly. Not part of the stable API.
#[doc(hidden)]
#[inline]
pub fn phase_entry_actions_fuse(
    states: &[State; LANES],
    state_scores: &[f32],
    out_third_dice: &mut [f32x8],
) {
    debug_assert_eq!(out_third_dice.len(), N_DICE);
    for d in 0..N_DICE {
        let mut acc = f32x8::splat(0.0);
        for action_idx in 0..N_ACTIONS {
            let action = ENTRY_ACTIONS[action_idx];
            let mut lanes = [0.0_f32; LANES];
            for s in 0..LANES {
                let state = states[s];
                if state.is_valid_action(action) {
                    let (score, child) = state.score_and_child(action, d as u8);
                    let child_idx: usize = child.into();
                    lanes[s] = score + state_scores[child_idx];
                }
            }
            acc = acc.fast_max(f32x8::from(lanes));
        }
        out_third_dice[d] = acc;
    }
}

/// Batched GEMV: `out[k] = sum_d M[k, d] * input[d]`, with `M =
/// KEEPERS_TO_DICE_PROBABILITIES` and the lane dimension being states.
///
/// Exposed as `#[doc(hidden)] pub` for the criterion phase benches.
#[doc(hidden)]
#[inline]
pub fn phase_gemv(out: &mut [f32x8], input: &[f32x8]) {
    debug_assert_eq!(out.len(), N_KEEPERS);
    debug_assert_eq!(input.len(), N_DICE);
    let m = &*KEEPERS_TO_DICE_PROBABILITIES;
    for k in 0..N_KEEPERS {
        let mut acc = f32x8::splat(0.0);
        for d in 0..N_DICE {
            acc += f32x8::splat(m[(k, d)]) * input[d];
        }
        out[k] = acc;
    }
}

/// Batched masked-max: `out[d] = max over k where mask[d, k] > 0 of input[k]`.
/// Uses the multiplication trick (0-mask contributes 0, which is the correct
/// floor for our non-negative EVs).
///
/// Exposed as `#[doc(hidden)] pub` for the criterion phase benches.
#[doc(hidden)]
#[inline]
pub fn phase_masked_max(out: &mut [f32x8], input: &[f32x8]) {
    debug_assert_eq!(out.len(), N_DICE);
    debug_assert_eq!(input.len(), N_KEEPERS);
    let mask = &*DICE_TO_ALLOWED_KEEPERS;
    for d in 0..N_DICE {
        let mut acc = f32x8::splat(0.0);
        for k in 0..N_KEEPERS {
            acc = acc.fast_max(f32x8::splat(mask[(d, k)]) * input[k]);
        }
        out[d] = acc;
    }
}

/// Fused sparse GEMV + scatter-max: computes
/// `out_dice[d] = max over k where d ⊇ k of (sum over d' in supp(K2D[k, ·]) of K2D[k, d'] * in_dice[d'])`
/// in one pass, equivalent to back-to-back [`phase_gemv`] +
/// [`phase_masked_max`] but skipping the 462-wide `second_keepers`
/// intermediate.
///
/// **Algorithm.** For each keeper `k`, walk the CSR row of
/// `KEEPERS_TO_DICE_PROBABILITIES` *once*: first compute the sparse dot
/// product `val = sum_{(d', p) ∈ row k} p * in_dice[d']`, then scatter-max
/// `val` into every `out_dice[d]` for `d` in that same column-index list.
/// The two passes share the same row's `col_idx` slice so it stays in L1.
///
/// **Why this fuses GEMV and masked_max.** The masked-max compatibility
/// `mask[d, k] = 1 ⟺ k ⊆ d ⟺ K2D[k, d] > 0`, so the column indices of
/// K2D row `k` are also exactly the dice that the masked-max needs to
/// max-update. One CSR drives both ops.
///
/// **Why this is a sparse win.** K2D is 3.75% dense (4,368 nonzeros of
/// 116,424). The dense [`phase_gemv`] does 462 × 252 = 116,424 FMAs per
/// 8-state batch; this fused path does 4,368 FMAs (the dot products) plus
/// 4,368 max-updates. Both inputs to the FMA are register-sized loads:
/// `splat(p)` is a scalar broadcast, and `in_dice[col_idx[i]]` is one
/// aligned 256-bit load (lanes are pre-packed 8-states-contiguous in the
/// `f32x8` buffer, so no gather is needed even when `col_idx[i]` strides).
///
/// Exposed as `#[doc(hidden)] pub` for the criterion phase benches.
#[doc(hidden)]
#[inline]
pub fn phase_fused_keeper_round(out_dice: &mut [f32x8], in_dice: &[f32x8]) {
    debug_assert_eq!(out_dice.len(), N_DICE);
    debug_assert_eq!(in_dice.len(), N_DICE);
    let sparse = &*SPARSE_K2D;

    // The masked-max trick (multiply by 0-mask) used 0 as the floor for the
    // max because EVs are non-negative. We preserve that invariant by zeroing
    // out_dice up front: the scatter-max only ever sees `val ≥ 0` updates,
    // and any d that no k ever updates stays at 0. (Empirically, every d is
    // covered by at least one k — d ⊇ k for k = 0-keeper is always true —
    // but the zero-init is the principled floor.)
    for slot in out_dice.iter_mut() {
        *slot = f32x8::splat(0.0);
    }

    for k in 0..N_KEEPERS {
        let row_start = sparse.row_ptr[k] as usize;
        let row_end = sparse.row_ptr[k + 1] as usize;
        let row_cols = &sparse.col_idx[row_start..row_end];
        let row_vals = &sparse.values[row_start..row_end];

        // Pass 1: sparse dot product. `val[s]` = K2D[k, ·] · in_dice for lane s.
        let mut val = f32x8::splat(0.0);
        for i in 0..row_cols.len() {
            let d = row_cols[i] as usize;
            val += f32x8::splat(row_vals[i]) * in_dice[d];
        }

        // Pass 2: scatter-max `val` into out_dice[d] for every d in this row.
        // Distinct d's within one row, so no aliasing within the inner loop;
        // aliasing across k's is fine since updates are strictly sequential.
        for &d_u16 in row_cols {
            let d = d_u16 as usize;
            out_dice[d] = out_dice[d].fast_max(val);
        }
    }
}

/// Step 7: `ev[s] = sum over d of initial_dist[d] * first_dice[d][s]` for
/// each lane `s`. `initial_dist` is row 0 of `KEEPERS_TO_DICE_PROBABILITIES`
/// (the "kept-nothing → roll 5 fresh dice" distribution).
///
/// Exposed as `#[doc(hidden)] pub` for the criterion phase benches.
#[doc(hidden)]
#[inline]
pub fn phase_final_dot(first_dice: &[f32x8]) -> [f32; LANES] {
    debug_assert_eq!(first_dice.len(), N_DICE);
    let m = &*KEEPERS_TO_DICE_PROBABILITIES;
    let mut ev = f32x8::splat(0.0);
    for d in 0..N_DICE {
        ev += f32x8::splat(m[(0, d)]) * first_dice[d];
    }
    ev.into()
}
