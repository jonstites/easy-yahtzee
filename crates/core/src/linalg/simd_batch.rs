//! Outer-loop SIMD: vectorize the per-state EV pipeline *across* states
//! rather than within a single state's masked-max.
//!
//! [`SimdBackend`] (in the parent module) is the per-state intra-state
//! vectorizer: `wide::f32x8` lanes are the 8 keepers being max-reduced for
//! one state. [`SimdBatchBuildBackend`] flips the axis: `f32x8` lanes are 8
//! *different states* at the same DP level, and the same matrix /
//! probability tables are broadcast scalar across lanes. This works because
//! the per-state pipeline's structure is identical state-to-state — the
//! only state-dependent inputs are (a) which entry actions are legal, (b)
//! the per-action `(score, child)` look-up, and (c) the resulting
//! `state_scores[child]` reads. Everything downstream (the two GEMVs, the
//! two masked-maxes, the final dot) sees the matrices as constants.
//!
//! This backend is intended to drive
//! [`Scores::new_with_unvalidated`](crate::Scores::new_with_unvalidated): the
//! unvalidated per-level batches are exactly `C(13, L) × 128` states, all
//! divisible by 8, so no scalar tail handling is needed. It also works on
//! the validated path (any tail `< 8` falls back to the per-state SIMD
//! pipeline), but the win there is smaller because validated batches are
//! ~half the size.
//!
//! Allocations live in a per-thread [`Scratch`] reused across chunks via
//! rayon's `map_init`.

use std::convert::Infallible;

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

/// Per-thread scratch for the 8-wide pipeline. Reused across batches via
/// rayon's `map_init` so the ~50 KiB working set isn't reallocated 27k
/// times per heavy DP level.
struct Scratch {
    third_dice: Vec<f32x8>,    // length N_DICE
    second_keepers: Vec<f32x8>, // length N_KEEPERS
    second_dice: Vec<f32x8>,    // length N_DICE
    first_keepers: Vec<f32x8>,  // length N_KEEPERS
    first_dice: Vec<f32x8>,     // length N_DICE
}

impl Scratch {
    fn new() -> Self {
        Self {
            third_dice: vec![f32x8::splat(0.0); N_DICE],
            second_keepers: vec![f32x8::splat(0.0); N_KEEPERS],
            second_dice: vec![f32x8::splat(0.0); N_DICE],
            first_keepers: vec![f32x8::splat(0.0); N_KEEPERS],
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
/// are broadcast scalar-into-lanes via `f32x8::splat`. Step 1 (per-state
/// `entry_actions` lookup) is fused with step 2 (max-over-actions reduce):
/// we never materialize the 13 × 252 × 8 intermediate, just stream into
/// `third_dice` directly. Saves 105 KiB of working set per chunk.
///
/// The four phases are factored into [`phase_entry_actions_fuse`],
/// [`phase_gemv`], [`phase_masked_max`], and [`phase_final_dot`] so the
/// criterion `simd_batch_phases` group can time them in isolation. Inlining
/// is preserved at the function-call boundary via `#[inline]`, so the
/// production path is bit-identical to the previous monolithic body.
fn compute_lanes(
    states: &[State; LANES],
    state_scores: &[f32],
    scratch: &mut Scratch,
) -> [f32; LANES] {
    // Steps 1+2 fused into third_dice.
    phase_entry_actions_fuse(states, state_scores, &mut scratch.third_dice);

    // Step 3: second_keepers[k] = sum over d of M[k, d] * third_dice[d].
    // M = KEEPERS_TO_DICE_PROBABILITIES, shape (462, 252).
    phase_gemv(&mut scratch.second_keepers, &scratch.third_dice);

    // Step 4: second_dice[d] = max over k of mask[d, k] * second_keepers[k].
    // Multiplication trick for the masked-max: 0-mask zeros the contribution,
    // and second_keepers is non-negative (sum of probabilities × non-negative
    // EVs), so 0 is the correct floor of the max.
    phase_masked_max(&mut scratch.second_dice, &scratch.second_keepers);

    // Steps 5 & 6: same shape as 3 & 4 but starting from second_dice.
    phase_gemv(&mut scratch.first_keepers, &scratch.second_dice);
    phase_masked_max(&mut scratch.first_dice, &scratch.first_keepers);

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
