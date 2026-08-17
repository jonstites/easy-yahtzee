//! Outer-loop SIMD: vectorize the per-state EV pipeline *across* states
//! rather than within a single state's masked-max, plus a sparse-fused
//! keeper round that merges each GEMV→masked_max pair into one CSR walk,
//! plus a vectorized `score_and_child` that computes all 8 lanes' scores
//! and child indices in one SIMD pass per `(action, dice)` cell.
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

use bytemuck::cast;
use rayon::prelude::*;
use wide::{f32x8, i32x8, CmpEq, CmpGt};

use super::{BuildBackend, SimdBackend};
use crate::{
    state_value_with, DICE_TO_ALLOWED_KEEPERS, ENTRY_ACTIONS, EntryAction,
    KEEPERS_TO_DICE_PROBABILITIES, NUM_DICE_COMBINATIONS, NUM_KEEPERS, State,
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
/// `entry_actions_table` is the precomputed `(score + state_scores[child])`
/// table consumed by [`phase_entry_actions_precomputed`]. Layout is
/// action-major, dice-inner: cell `(a, d)` is at index `a * N_DICE + d`.
/// Each `f32x8` cell holds the resolved values across all 8 lanes (one per
/// state). 13 × 252 × 32 = 105 KiB per thread; fits in L2 (1 MiB on Zen 5).
///
/// The other three 252-wide buffers carry the dice-axis intermediates
/// between pipeline phases. The 462-wide keeper-axis intermediates were
/// eliminated by [`phase_fused_keeper_round`].
struct Scratch {
    /// Precomputed `score + state_scores[child]` table for the entry-actions
    /// phase. Layout `[f32x8; N_ACTIONS * N_DICE]`, indexed `a * N_DICE + d`.
    entry_actions_table: Vec<f32x8>,
    third_dice: Vec<f32x8>,  // length N_DICE
    second_dice: Vec<f32x8>, // length N_DICE
    first_dice: Vec<f32x8>,  // length N_DICE
}

impl Scratch {
    fn new() -> Self {
        Self {
            entry_actions_table: vec![f32x8::splat(0.0); N_ACTIONS * N_DICE],
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
/// Each `f32x8` lane is one of the 8 input states. Steps 1+2 use a
/// precomputed `(score + state_scores[child])` table whose cells are
/// computed by [`phase_entry_actions_vectorized`] — all 8 lanes are
/// computed in parallel via SIMD per `(action, dice)` cell. The table
/// then feeds a branchless SIMD max-reduce over actions. Steps 3+4 (and
/// 5+6) fuse via [`phase_fused_keeper_round`]: sparse CSR over `K2D` lets
/// each row's dot product be computed and immediately scatter-maxed into
/// `out_dice` without ever materializing the 462-wide `second_keepers` /
/// `first_keepers` intermediate.
///
/// Phases are factored into [`phase_entry_actions_vectorized`],
/// [`phase_fused_keeper_round`], and [`phase_final_dot`]. The older
/// scalar-per-lane variant [`phase_entry_actions_precomputed`] (Round 1)
/// and the inline build+reduce variant [`phase_entry_actions_fuse`]
/// (Round 0) are kept as bench-only baselines for the criterion
/// `simd_batch_phases` group, so the vectorization and precompute wins
/// stay measurable on every CI run.
fn compute_lanes(
    states: &[State; LANES],
    state_scores: &[f32],
    scratch: &mut Scratch,
) -> [f32; LANES] {
    // Steps 1+2: precompute (score + state_scores[child]) table per (s, a, d),
    // then max-reduce over actions per dice into third_dice. The vectorized
    // path computes all 8 lanes in parallel via SIMD; see
    // [`phase_entry_actions_vectorized`].
    phase_entry_actions_vectorized(
        states,
        state_scores,
        &mut scratch.entry_actions_table,
        &mut scratch.third_dice,
    );

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

/// Round 1 — bench-only baseline. Same precompute layout as
/// [`phase_entry_actions_vectorized`] but the per-lane work is scalar
/// (one lane at a time, locality-preserving `(s outer, a middle, d inner)`
/// nest). Kept so the `simd_batch_phases` group can measure the
/// vectorization win as a delta.
///
/// **Why precompute.** The phase has two kinds of work: (a) per-(s, a, d)
/// scalar work to compute `(score, child_idx)` and gather `state_scores[child_idx]`
/// (random access through ~4 MiB), and (b) a 13-way max reduction across
/// actions to produce `out_third_dice[d]`. The naive interleaved loop nest
/// (see [`phase_entry_actions_fuse`]) does both inside one `for d` body,
/// which fights the cache prefetcher: each d iteration touches 8 × 13 = 104
/// random `state_scores` cells.
///
/// Splitting the work into Phase A (build table, locality-preserving nest)
/// and Phase B (branchless SIMD max-reduce) lets each phase have a clean
/// memory access pattern. The total compute is the same, but Phase A's
/// inner d-loop now sees *highly localized* `state_scores[child]` reads:
///
/// - **Non-upper actions** (7 of 13): for fixed `(s, a)`, `child` is
///   constant across all d (the child state's `entries` and
///   `upper_score_remaining` don't change with the dice). Same address read
///   252 times → guaranteed L1 hit after the first.
/// - **Upper actions** (6 of 13): `child.upper_score_remaining` varies with
///   the score on `d` (range 0..5k for face k). 252 reads land within a
///   ~30-value band of the parent's `idx` → small working set, prefetcher
///   catches it easily.
///
/// **Phase B's win.** Pure branchless SIMD: 13 `fast_max` ops per d on
/// pre-resolved `f32x8` cells. No scalar work, no random gather, no
/// validity branch. The compiler emits the optimal reduce-max pattern.
///
/// **Layout.** Action-major, dice-inner: cell `(a, d)` lives at table index
/// `a * N_DICE + d`. Phase A writes one lane at a time via `as_mut_array()`
/// (stride-32 across 252 cells per (s, a)); Phase B reads whole `f32x8`
/// cells. 13 × 252 × 32 = 105 KiB scratch per thread.
///
/// **Invalid actions.** Phase A unconditionally writes every `(s, a, d)`
/// cell — either the resolved value (valid) or 0.0 (invalid) — so no
/// pre-zero is needed and Phase B doesn't need a validity mask. The 0.0
/// floor is correct because EVs are non-negative.
///
/// Exposed as `#[doc(hidden)] pub` so the criterion `simd_batch_phases`
/// group can call it directly. Not part of the stable API.
#[doc(hidden)]
#[inline]
pub fn phase_entry_actions_precomputed(
    states: &[State; LANES],
    state_scores: &[f32],
    table: &mut [f32x8],
    out_third_dice: &mut [f32x8],
) {
    debug_assert_eq!(table.len(), N_ACTIONS * N_DICE);
    debug_assert_eq!(out_third_dice.len(), N_DICE);

    // Phase A: build table. Lane-by-lane fill in a (s outer, a middle, d
    // inner) nest so state_scores[child_idx] reads stay localized for fixed
    // (s, a). The validity branch hoists out of the d-loop.
    for s in 0..LANES {
        let state = states[s];
        for action_idx in 0..N_ACTIONS {
            let action = ENTRY_ACTIONS[action_idx];
            let row = &mut table[action_idx * N_DICE..(action_idx + 1) * N_DICE];
            if state.is_valid_action(action) {
                for d in 0..N_DICE {
                    let (score, child) = state.score_and_child(action, d as u8);
                    let child_idx: usize = child.into();
                    row[d].as_mut_array()[s] = score + state_scores[child_idx];
                }
            } else {
                for d in 0..N_DICE {
                    row[d].as_mut_array()[s] = 0.0;
                }
            }
        }
    }

    // Phase B: max-reduce. For each d, max over actions of table[a][d].
    // Pure SIMD — no scalar work, no branches.
    for d in 0..N_DICE {
        let mut acc = table[d]; // a = 0
        for action_idx in 1..N_ACTIONS {
            acc = acc.fast_max(table[action_idx * N_DICE + d]);
        }
        out_third_dice[d] = acc;
    }
}

/// Bit values from `EntryAction` (bitflags<u16>) that the vectorized phase
/// uses directly. Kept as `const` so they fold into the SIMD splats at
/// compile time.
const UPPER_ACTIONS_MASK: i32 = (EntryAction::ONE.bits()
    | EntryAction::TWO.bits()
    | EntryAction::THREE.bits()
    | EntryAction::FOUR.bits()
    | EntryAction::FIVE.bits()
    | EntryAction::SIX.bits()) as i32;
const YAHTZEE_BIT: i32 = EntryAction::YAHTZEE.bits() as i32;

/// Round 2 — vectorized variant of [`phase_entry_actions_precomputed`].
///
/// Same semantics: build a `[f32x8; N_ACTIONS * N_DICE]` table of
/// `(score + state_scores[child]) * valid_mask` per `(action, dice)` cell,
/// where each `f32x8` lane is one of the 8 input states. Then max-reduce
/// over actions per dice into `out_third_dice`.
///
/// **Why vectorize.** Round 1's precomputed variant runs scalar
/// `score_and_child` 26,208 times per batch (8 lanes × 13 actions × 252
/// dice). The body has ~30 instructions of branchy state-level logic
/// (upper bonus, Yahtzee bonus, joker rule); doing it scalar one lane at
/// a time leaves ~6× IPC on the table at AVX2 width. This variant flips
/// the loop nest to put the lane axis innermost — `(a outer, d middle,
/// s inner-vectorized)` — and computes all 8 lanes' `(score, child_idx)`
/// in one SIMD pass per `(action, dice)` cell.
///
/// **Hoisting.** Per-batch state-only quantities (`entries`, `upper`,
/// `yahtzee_eligible`, `contains_yahtzee_box`, `upper_complete`) live in
/// `i32x8` registers across the whole loop. Per-action quantities (action
/// bits, joker score, `is_upper_action`, `is_yahtzee_action`, per-lane
/// validity) hoist out of the d-loop. Only the `(a, d)` cell-specific
/// `raw_score` and the `YAHTZEE_DICE[d]` flag are loaded per d, both
/// once.
///
/// **Joker rule** (Yahtzee bonus categories take guaranteed 25/30/40 if
/// the matching upper category was already filled when a Yahtzee is
/// rolled): per-lane mask = `contains_yahtzee_box & entries_have_yahtzee_face`,
/// only enabled when the action's `joker_lower_score` is `Some`. The mask
/// `blend`s the joker constant in for matching lanes, leaves
/// `raw_score` for the rest.
///
/// **Gather.** `state_scores[child_idx]` is still 8 scalar loads per cell
/// — `wide` doesn't expose VGATHERDPS. The locality argument from Round 1
/// still holds per lane: for fixed `(s, a)`, child_idx varies in a small
/// band across d, so the L1 cache absorbs the gather even though the
/// loads aren't a single SIMD instruction. Round 1 walked d innermost
/// for fixed `(s, a)`; this variant walks d in the middle but the
/// per-lane access pattern across consecutive d's is the same — child_idx
/// for `(s, a, d)` and `(s, a, d+1)` differ by at most a small upper-score
/// delta.
///
/// **Validity.** Per-lane `is_valid_action(a)` is precomputed at the top
/// of the action loop as an `i32x8` bitmask. The final cell value is
/// `total * valid_mask`, where `valid_mask` is 0.0 / 1.0 per lane (rather
/// than `blend`-replacing with 0 — equivalent, lets us skip a separate
/// `f32x8::ZERO` operand).
#[doc(hidden)]
#[inline]
pub fn phase_entry_actions_vectorized(
    states: &[State; LANES],
    state_scores: &[f32],
    table: &mut [f32x8],
    out_third_dice: &mut [f32x8],
) {
    debug_assert_eq!(table.len(), N_ACTIONS * N_DICE);
    debug_assert_eq!(out_third_dice.len(), N_DICE);

    // Hoist per-batch state-only quantities into i32x8 SoA. Each lane is one
    // input state; the lane stays put for the entire pipeline.
    let entries_v = i32x8::new(std::array::from_fn(|s| states[s].entries.bits() as i32));
    let upper_remaining_v =
        i32x8::new(std::array::from_fn(|s| states[s].upper_score_remaining as i32));
    let yahtzee_eligible_v = i32x8::new(std::array::from_fn(|s| {
        if states[s].yahtzee_bonus_eligible {
            1
        } else {
            0
        }
    }));
    // upper_complete_mask: all-1s lanes where parent already had upper bonus
    // (so we can't fire the +35 again).
    let upper_complete_mask = upper_remaining_v.simd_eq(i32x8::splat(0));
    // contains_yahtzee_mask: all-1s lanes where parent has the Yahtzee box
    // already filled (gate for joker rule and for whether re-rolling a
    // Yahtzee qualifies for the +100 bonus).
    let contains_yahtzee_mask =
        (entries_v & i32x8::splat(YAHTZEE_BIT)).simd_gt(i32x8::splat(0));
    // yahtzee_eligible_mask: all-1s lanes where state.yahtzee_bonus_eligible
    // is true (the +100 bonus precondition).
    let yahtzee_eligible_mask = yahtzee_eligible_v.simd_gt(i32x8::splat(0));

    let dice_and_entry_scores = &*crate::DICE_AND_ENTRY_SCORES;
    let yahtzee_dice = &*crate::YAHTZEE_DICE;

    for action_idx in 0..N_ACTIONS {
        let action = ENTRY_ACTIONS[action_idx];
        let action_bits = action.bits() as i32;
        let is_upper_action = (action_bits & UPPER_ACTIONS_MASK) != 0;
        let is_yahtzee_action = action == EntryAction::YAHTZEE;
        let joker_score: Option<u8> = match action {
            EntryAction::FULL_HOUSE => Some(25),
            EntryAction::SMALL_STRAIGHT => Some(30),
            EntryAction::LARGE_STRAIGHT => Some(40),
            _ => None,
        };

        // Per-lane validity = !entries.contains(action) = ((entries & action) == 0).
        // valid_mask is an f32x8 with 1.0 in valid lanes, 0.0 in invalid.
        let valid_int_mask =
            (entries_v & i32x8::splat(action_bits)).simd_eq(i32x8::splat(0));
        let valid_mask_f: f32x8 =
            cast::<i32x8, f32x8>(valid_int_mask).blend(f32x8::splat(1.0), f32x8::splat(0.0));

        // entries with this action OR'd in (used for child_idx).
        let entries_with_action_v = entries_v | i32x8::splat(action_bits);

        let row = &mut table[action_idx * N_DICE..(action_idx + 1) * N_DICE];
        let scores_row = &dice_and_entry_scores[action_idx];

        for d in 0..N_DICE {
            let raw_score_i = scores_row[d] as i32;
            let raw_score_v = i32x8::splat(raw_score_i);
            let raw_score_f = f32x8::splat(raw_score_i as f32);

            let yahtzee_face_opt = yahtzee_dice[d];
            let yahtzee_face_bits =
                yahtzee_face_opt.map(|f| f.bits() as i32).unwrap_or(0);
            let dice_is_yahtzee = yahtzee_face_opt.is_some();

            // child.upper_score_remaining = saturating_sub(parent_upper, raw_score) for upper
            // actions, unchanged otherwise. (raw_score - parent_upper) clamps via .max(0).
            let new_upper_v = if is_upper_action {
                (upper_remaining_v - raw_score_v).max(i32x8::splat(0))
            } else {
                upper_remaining_v
            };

            // child.yahtzee_bonus_eligible = parent.eligible | (action==YAHTZEE && dice_is_yahtzee).
            // Only the second term is per-action-and-dice; first term is parent-only.
            let new_yahtzee_eligible_v = if is_yahtzee_action && dice_is_yahtzee {
                yahtzee_eligible_v | i32x8::splat(1)
            } else {
                yahtzee_eligible_v
            };

            // child_idx = (entries_with_action << 7) | (new_yahtzee_eligible << 6) | new_upper.
            let child_idx_v: i32x8 = (entries_with_action_v << 7_u32)
                | (new_yahtzee_eligible_v << 6_u32)
                | new_upper_v;

            // Scalar gather. wide doesn't expose VGATHERDPS, so we do 8 scalar
            // loads. The 8 child_idx values are within a small band per lane
            // for fixed (s, a) (child differs from parent only by upper or
            // yahtzee-eligible bit), so the L1 absorbs them.
            let idx_arr = child_idx_v.to_array();
            let child_evs_f = f32x8::new([
                state_scores[idx_arr[0] as usize],
                state_scores[idx_arr[1] as usize],
                state_scores[idx_arr[2] as usize],
                state_scores[idx_arr[3] as usize],
                state_scores[idx_arr[4] as usize],
                state_scores[idx_arr[5] as usize],
                state_scores[idx_arr[6] as usize],
                state_scores[idx_arr[7] as usize],
            ]);

            // Joker rule: if action ∈ {FULL_HOUSE, SMALL_STRAIGHT, LARGE_STRAIGHT}
            // AND dice is a Yahtzee AND parent has Yahtzee box filled AND
            // parent has the matching upper face filled, override raw_score with
            // the joker constant (25/30/40). The first three conditions are
            // action+d-only; the fourth is the per-lane bit test.
            let normal_score_f = if let Some(joker_val) = joker_score
                && dice_is_yahtzee
            {
                let entries_have_face_mask = (entries_v
                    & i32x8::splat(yahtzee_face_bits))
                    .simd_gt(i32x8::splat(0));
                let joker_lane_mask = contains_yahtzee_mask & entries_have_face_mask;
                cast::<i32x8, f32x8>(joker_lane_mask)
                    .blend(f32x8::splat(joker_val as f32), raw_score_f)
            } else {
                raw_score_f
            };

            // Upper bonus: !parent.upper_complete && child.upper_complete.
            // child.upper_complete = (new_upper == 0).
            let new_upper_zero_mask = new_upper_v.simd_eq(i32x8::splat(0));
            let upper_fires_mask = !upper_complete_mask & new_upper_zero_mask;
            let upper_bonus_f = cast::<i32x8, f32x8>(upper_fires_mask)
                .blend(f32x8::splat(35.0), f32x8::splat(0.0));

            // Yahtzee bonus: dice is Yahtzee && parent.yahtzee_bonus_eligible.
            let yahtzee_bonus_f = if dice_is_yahtzee {
                cast::<i32x8, f32x8>(yahtzee_eligible_mask)
                    .blend(f32x8::splat(100.0), f32x8::splat(0.0))
            } else {
                f32x8::splat(0.0)
            };

            let total_f = normal_score_f + upper_bonus_f + yahtzee_bonus_f + child_evs_f;

            // Apply validity mask: 0.0 in invalid lanes (so they lose the
            // max-reduce in Phase B), original total in valid lanes.
            row[d] = total_f * valid_mask_f;
        }
    }

    // Phase B: same as Round 1 — pure SIMD max-reduce over actions per dice.
    for d in 0..N_DICE {
        let mut acc = table[d];
        for action_idx in 1..N_ACTIONS {
            acc = acc.fast_max(table[action_idx * N_DICE + d]);
        }
        out_third_dice[d] = acc;
    }
}

/// Bench-only baseline for steps 1+2: same semantics as
/// [`phase_entry_actions_precomputed`] but with the build and reduce fused
/// inside a single `for d` loop. Kept so the criterion
/// `simd_batch_phases` group can measure the precompute win as a delta.
///
/// Per-state work (`is_valid_action`, `score_and_child`, `state_scores[child_idx]`)
/// is scalar — the lane axis is purely outer-loop SIMD over states. Invalid
/// actions contribute 0.0 to their lane; the max reduction picks them up
/// only if every action is invalid (terminal state), which can't happen in
/// `compute_level` since `set_scores` never hits level 13.
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
