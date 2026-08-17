//! CUDA implementation of [`crate::BuildBackend`].
//!
//! The level-batched DP fill mapped to the GPU. Per [`crate::BuildBackend`]'s
//! contract, `compute_level` receives a batch of states all at the same DP
//! level plus the read-only `state_scores` buffer, and returns one EV per
//! state. We exploit the batch in two ways:
//!
//! 1. The two GEMVs in the per-state pipeline (`keepers_from_dice`, 252→462)
//!    become one batched GEMM each: `(B × 252) · (252 × 462)`. cuBLAS does
//!    them as a single launch per batch.
//! 2. The masked-max reduction in `dice_from_keepers` (462→252, gated by the
//!    252×462 mask) becomes a single custom kernel over the whole batch: one
//!    block per `(dice_idx, state_b)`, 64-wide reduction across keepers.
//!
//! The branchy "build entry-actions matrix and fold-axis to roll-3 dice" step
//! is also a custom kernel — per state it walks 13 entries, branches on
//! validity / joker rule / upper bonus, and gathers child-state EVs from
//! device-resident `state_scores`.
//!
//! All static tables (`KEEPERS_TO_DICE_PROBABILITIES`, `DICE_TO_ALLOWED_KEEPERS`,
//! `DICE_AND_ENTRY_SCORES`, `YAHTZEE_DICE`) are uploaded once at backend
//! construction time. `state_scores` is uploaded per `compute_level` call
//! (one ~4 MiB H→D copy per level — small relative to the kernel work and
//! easy to reason about).
//!
//! cudarc loads libcuda / libcublas / libnvrtc dynamically (the
//! `fallback-dynamic-loading` feature). No CUDA headers or `nvcc` are
//! required at build time, only the runtime `.so`s at run time.

use std::sync::{Arc, Mutex};

use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::{BuildBackend, State, NUM_STATES};

/// Errors raised by the CUDA backend during construction (`new`) or per-level
/// compute (`compute_level`).
#[derive(Debug)]
pub enum CudaError {
    /// Failed to load libcuda.so / libnvrtc.so / libcublas.so, the GPU is
    /// unreachable, an allocation failed, a kernel launch returned a CUDA
    /// error, etc. The string is the cudarc driver-API error message.
    Driver(String),
    /// NVRTC failed to compile a kernel. Almost always a code bug in this
    /// crate's kernel sources rather than something the user can fix.
    Compile(String),
    /// cuBLAS initialization or sgemm call failed.
    Cublas(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Driver(msg) => write!(f, "CUDA driver error: {msg}"),
            CudaError::Compile(msg) => write!(f, "NVRTC compile error: {msg}"),
            CudaError::Cublas(msg) => write!(f, "cuBLAS error: {msg}"),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        CudaError::Driver(format!("{err:?}"))
    }
}

impl From<cudarc::nvrtc::CompileError> for CudaError {
    fn from(err: cudarc::nvrtc::CompileError) -> Self {
        CudaError::Compile(format!("{err:?}"))
    }
}

impl From<cudarc::cublas::result::CublasError> for CudaError {
    fn from(err: cudarc::cublas::result::CublasError) -> Self {
        CudaError::Cublas(format!("{err:?}"))
    }
}

/// Source of the three custom kernels — compiled once at construction via
/// NVRTC. See module docs for the math each one implements; the kernels
/// themselves are intentionally written to map 1:1 onto the per-state CPU
/// pipeline so a per-state diff against `state_value_with(NdarrayBackend)`
/// is a useful debug step.
const KERNEL_SRC: &str = r#"
// Build the (B, 252) `third_dice` array: per state in the batch, fold the
// 13-entry × 252-dice "entry_actions" matrix on its action axis. Each entry
// in that matrix is `score + state_scores[child]` for valid `(action, dice)`
// pairs (joker-aware, upper-bonus-aware, yahtzee-bonus-aware), 0 for invalid.
// We don't materialize the (13 × 252) matrix; we fold it on the fly.
//
// Grid shape: gridDim = (batch, 1, 1), blockDim = (256, 1, 1). batch goes on
// gridDim.x (limit 2^31) since gridDim.y is capped at 65535 and the build's
// largest level can exceed that. dice_idx maps to threadIdx.x; threads with
// threadIdx.x >= 252 early-out.
extern "C" __global__ void build_third_dice(
    const unsigned int *state_indices,           // [B]
    const float *state_scores,                   // [NUM_STATES] (1<<20 = 1M)
    const unsigned char *dice_and_entry_scores,  // [13 * 252]
    const signed char *yahtzee_dice,             // [252]; -1 for non-Yahtzee, 0..5 for upper
    float *third_dice,                           // out [B * 252]
    int batch_size
) {
    int dice_idx = threadIdx.x;
    int state_b  = blockIdx.x;
    if (dice_idx >= 252 || state_b >= batch_size) return;

    unsigned int sidx = state_indices[state_b];
    // Same packing as `impl From<State> for usize` in lib.rs:
    //   bits 0..5 = upper_score_remaining (0..=63)
    //   bit 6     = yahtzee_bonus_eligible
    //   bits 7..19 = entries (13-bit EntryAction bitset)
    unsigned int entries_bits = (sidx >> 7) & 0x1FFFu;
    unsigned int eligible     = (sidx >> 6) & 1u;
    unsigned int upper_rem    = sidx & 0x3Fu;

    int yd = (int) yahtzee_dice[dice_idx];   // -1 or 0..5
    int is_yahtzee = (yd >= 0);

    // Joker rule preconditions: YAHTZEE box (bit 11) is filled, the dice
    // is a Yahtzee, and the matching upper category (bit yd) is also filled.
    int yahtzee_filled = (entries_bits >> 11) & 1;
    int joker_active = is_yahtzee && yahtzee_filled
        && (yd >= 0 ? ((entries_bits >> yd) & 1) : 0);

    // Yahtzee +100 bonus depends only on (dice, eligibility), not on action,
    // so it's a turn-level constant we can hoist out of the action loop.
    float yahtzee_bonus = (is_yahtzee && eligible) ? 100.0f : 0.0f;

    float best = 0.0f;

    #pragma unroll
    for (int action_idx = 0; action_idx < 13; action_idx++) {
        // is_valid_action: !entries.contains(action)
        if ((entries_bits >> action_idx) & 1) continue;

        unsigned int action_bit = 1u << action_idx;
        unsigned char raw_score = dice_and_entry_scores[action_idx * 252 + dice_idx];
        unsigned int normal_score;

        // Joker rule for lower-section fixed scores (matches
        // `joker_lower_score` in lib.rs):
        //   FULL_HOUSE (8)     -> 25
        //   SMALL_STRAIGHT (9) -> 30
        //   LARGE_STRAIGHT (10)-> 40
        if (joker_active) {
            switch (action_idx) {
                case 8:  normal_score = 25; break;
                case 9:  normal_score = 30; break;
                case 10: normal_score = 40; break;
                default: normal_score = (unsigned int) raw_score;
            }
        } else {
            normal_score = (unsigned int) raw_score;
        }

        // Compute child state. Mirror `State::child` exactly:
        //   entries |= action_bit
        //   if upper category: upper_rem = saturating_sub(upper_rem, raw_score)
        //   if YAHTZEE && is_yahtzee: eligible = 1
        unsigned int child_entries = entries_bits | action_bit;
        unsigned int child_upper = upper_rem;
        if (action_idx < 6) {
            unsigned int s = (unsigned int) raw_score;
            child_upper = (child_upper > s) ? (child_upper - s) : 0;
        }
        unsigned int child_eligible = eligible;
        if (action_idx == 11 && is_yahtzee) {
            child_eligible = 1;
        }

        unsigned int child_idx = (child_entries << 7) | (child_eligible << 6) | child_upper;

        // Upper bonus fires when this entry completes the upper section.
        float upper_bonus = ((upper_rem != 0) && (child_upper == 0)) ? 35.0f : 0.0f;

        float total = (float) normal_score + upper_bonus + yahtzee_bonus
                    + state_scores[child_idx];

        if (total > best) best = total;
    }

    third_dice[state_b * 252 + dice_idx] = best;
}

// Masked max-reduction: for each (state_b, dice_idx), output the max over
// keepers k of `mask[dice_idx, k] * keeper_values[state_b, k]`. The mask is
// 0 / 1, so this is "max over keepers allowed for this dice combo." The
// masked product (rather than a branch) matches the CPU implementation
// exactly and stays branchless inside the inner loop.
//
// Grid shape: gridDim = (batch, 252, 1), blockDim = (64, 1, 1). batch is on
// gridDim.x (no limit); dice_idx is on gridDim.y (252 < 65535). 64 threads
// cooperate on the 462-wide reduction (~7 keepers per thread). Shared memory
// holds the per-thread max.
extern "C" __global__ void masked_max(
    const float *keeper_values,  // [B * 462]
    const float *mask,           // [252 * 462] (DICE_TO_ALLOWED_KEEPERS, dice-major)
    float *out,                  // [B * 252]
    int batch_size
) {
    int state_b  = blockIdx.x;
    int dice_idx = blockIdx.y;
    int tid      = threadIdx.x;
    if (state_b >= batch_size) return;

    extern __shared__ float sdata[];

    float my_max = 0.0f;
    const float *kv = keeper_values + state_b * 462;
    const float *mr = mask + dice_idx * 462;

    for (int k = tid; k < 462; k += blockDim.x) {
        float v = kv[k] * mr[k];
        if (v > my_max) my_max = v;
    }

    sdata[tid] = my_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[state_b * 252 + dice_idx] = sdata[0];
    }
}

// Final marginal: for each state_b, compute sum_d first_dice[b, d] * row0[d],
// where row0 is row 0 of KEEPERS_TO_DICE_PROBABILITIES (the initial-roll
// distribution over the 252 five-dice combos). One block per state, 64
// threads cooperate on the 252-wide sum.
extern "C" __global__ void initial_roll_ev(
    const float *first_dice,   // [B * 252]
    const float *row0,         // [252]
    float *out,                // [B]
    int batch_size
) {
    int state_b = blockIdx.x;
    int tid     = threadIdx.x;
    if (state_b >= batch_size) return;

    extern __shared__ float sdata[];

    float my_sum = 0.0f;
    const float *fd = first_dice + state_b * 252;

    for (int d = tid; d < 252; d += blockDim.x) {
        my_sum += fd[d] * row0[d];
    }

    sdata[tid] = my_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[state_b] = sdata[0];
    }
}

// Scatter the per-state EVs we just computed into the device-resident
// `state_scores` buffer at the indices `state_indices[b]`. This is what
// lets us keep `state_scores` on the GPU across `compute_level` calls,
// rather than re-uploading the host buffer (which has the same values)
// every level.
//
// One thread per batch element, flat 1D launch. Trivial.
extern "C" __global__ void scatter_results(
    const unsigned int *state_indices,  // [B]
    const float *result,                 // [B]
    float *state_scores,                 // [NUM_STATES]
    int batch_size
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    state_scores[state_indices[b]] = result[b];
}
"#;

const NUM_DICE_F: usize = 252;
const NUM_KEEPERS_F: usize = 462;
const NUM_ENTRIES_F: usize = 13;
/// Threads per block for the masked-max and final-marginal kernels. Picked
/// for divisibility (462 / 64 = 8 inner iters with one tail; 252 / 64 = 4
/// inner iters with one tail) and to keep shared-mem reductions short.
const REDUCE_BLOCK_DIM: u32 = 64;
/// Threads per block for the entry-actions kernel. 256 covers all 252 dice
/// combos in one block per state, with 4 idle threads at the tail.
const ENTRY_BLOCK_DIM: u32 = 256;

/// Mutable, lazily-allocated device-side buffers that persist across
/// `compute_level` calls.
///
/// `state_scores` is allocated once at backend construction (NUM_STATES f32s
/// = 4 MiB), zero-initialized, and updated in place by the `scatter_results`
/// kernel after each level so the next level's `build_third_dice` can read
/// it directly from device memory — no per-level H→D upload of the host
/// state_scores buffer needed.
///
/// The five intermediate buffers (`third_dice`, `second_keepers`,
/// `second_dice`, `first_keepers`, `first_dice`) and the per-state `result`
/// scale with batch size and are lazily grown to the largest batch we've
/// seen (with a small headroom). We never shrink — once allocated, the
/// buffers live until the backend drops.
struct DeviceState {
    /// 4 MiB = NUM_STATES × f32. Persistent across all `compute_level`
    /// calls (and across multiple `Scores::new_with` builds — see comment
    /// in `compute_level_inner` about why that's safe given deterministic
    /// per-state EVs).
    state_scores: CudaSlice<f32>,

    // Per-call intermediates, sized to `capacity_batch`. None until the
    // first `compute_level` call grows them.
    third_dice: Option<CudaSlice<f32>>,
    second_keepers: Option<CudaSlice<f32>>,
    second_dice: Option<CudaSlice<f32>>,
    first_keepers: Option<CudaSlice<f32>>,
    first_dice: Option<CudaSlice<f32>>,
    result: Option<CudaSlice<f32>>,
    /// Number of states the buffers above can currently hold. 0 means
    /// uninitialized.
    capacity_batch: usize,

    /// Per-call upload of state indices: lazily grown like the
    /// intermediates. The host-side state_indices Vec<u32> is rebuilt per
    /// call; only the device buffer reuses memory.
    state_indices: Option<CudaSlice<u32>>,
}

/// GPU-resident [`BuildBackend`]. Construct once with [`CudaBuildBackend::new`]
/// and reuse across multiple `Scores::new_with` calls (each call makes its
/// own per-level scatter; the static tables, kernels, cuBLAS handle, and
/// device buffers are all reused).
pub struct CudaBuildBackend {
    /// CUDA context — keeps the device alive and the loaded module / cuBLAS
    /// handle valid. Held as an `Arc` so the stream and module references
    /// stay valid for the backend's lifetime.
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    /// Default stream for all H→D copies, kernel launches, and D→H copies.
    /// Everything is serialized on this stream — concurrent streams aren't
    /// useful here because every level depends on the previous one.
    stream: Arc<CudaStream>,
    /// cuBLAS handle, bound to `stream`. Used for the two batched GEMMs that
    /// implement `keepers_from_dice` (252→462) per level.
    blas: CudaBlas,

    // Compiled kernel handles. Loaded once at construction.
    f_build_third_dice: CudaFunction,
    f_masked_max: CudaFunction,
    f_initial_roll_ev: CudaFunction,
    f_scatter_results: CudaFunction,

    // Device-resident static tables, uploaded once.
    /// 462 × 252 row-major. cuBLAS reads it as a 252×462 column-major view of
    /// the transpose; that matches what we want (see `compute_level` for the
    /// transpose-juggling explanation).
    keepers_to_dice: CudaSlice<f32>,
    /// 252 × 462 row-major mask, indexed by `(dice_idx, keeper_idx)`. The
    /// `masked_max` kernel reads one row per `(dice_idx, state_b)` block.
    dice_to_keepers: CudaSlice<f32>,
    /// 13 × 252 row-major raw entry scores (no joker rule applied — the
    /// kernel layers it on at lookup time).
    dice_and_entry_scores: CudaSlice<u8>,
    /// 252 entries: -1 for non-Yahtzee dice, 0..5 = matching upper
    /// EntryAction index (`EntryAction::ONE.as_idx() == 0`, …,
    /// `EntryAction::SIX.as_idx() == 5`).
    yahtzee_dice: CudaSlice<i8>,

    /// Lazily-grown per-call device buffers. Mutex (rather than
    /// `RefCell`) so `BuildBackend: Sync` holds; in practice
    /// `Scores::new_with` calls `compute_level` serially.
    device_state: Mutex<DeviceState>,
}

impl CudaBuildBackend {
    /// Initialize the CUDA backend on device 0: load the driver, allocate a
    /// stream, compile and load the custom kernels, upload the static
    /// tables, and zero-allocate the persistent device-side state_scores
    /// buffer.
    pub fn new() -> Result<Self, CudaError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;

        // Compile and load kernels via NVRTC. ~50ms first time; PTX is cached
        // by the driver so re-running the same binary is faster.
        let ptx = compile_ptx(KERNEL_SRC)?;
        let module = ctx.load_module(ptx)?;
        let f_build_third_dice = module.load_function("build_third_dice")?;
        let f_masked_max = module.load_function("masked_max")?;
        let f_initial_roll_ev = module.load_function("initial_roll_ev")?;
        let f_scatter_results = module.load_function("scatter_results")?;

        // Upload static tables.
        let kt = crate::KEEPERS_TO_DICE_PROBABILITIES
            .as_slice()
            .expect("KEEPERS_TO_DICE_PROBABILITIES is contiguous");
        let dk = crate::DICE_TO_ALLOWED_KEEPERS
            .as_slice()
            .expect("DICE_TO_ALLOWED_KEEPERS is contiguous");
        let des: Vec<u8> = crate::DICE_AND_ENTRY_SCORES.iter().copied().collect();
        let yd: Vec<i8> = crate::YAHTZEE_DICE
            .iter()
            .map(|opt| match opt {
                None => -1_i8,
                Some(action) => action.as_idx() as i8,
            })
            .collect();

        debug_assert_eq!(kt.len(), NUM_KEEPERS_F * NUM_DICE_F);
        debug_assert_eq!(dk.len(), NUM_DICE_F * NUM_KEEPERS_F);
        debug_assert_eq!(des.len(), NUM_ENTRIES_F * NUM_DICE_F);
        debug_assert_eq!(yd.len(), NUM_DICE_F);

        let keepers_to_dice = stream.clone_htod(kt)?;
        let dice_to_keepers = stream.clone_htod(dk)?;
        let dice_and_entry_scores = stream.clone_htod(&des)?;
        let yahtzee_dice = stream.clone_htod(&yd)?;

        // Persistent state_scores: NUM_STATES f32 = 4 MiB, zero-initialized.
        // Lives until the backend drops.
        let state_scores = stream.alloc_zeros::<f32>(NUM_STATES as usize)?;
        stream.synchronize()?;

        Ok(Self {
            ctx,
            stream,
            blas,
            f_build_third_dice,
            f_masked_max,
            f_initial_roll_ev,
            f_scatter_results,
            keepers_to_dice,
            dice_to_keepers,
            dice_and_entry_scores,
            yahtzee_dice,
            device_state: Mutex::new(DeviceState {
                state_scores,
                third_dice: None,
                second_keepers: None,
                second_dice: None,
                first_keepers: None,
                first_dice: None,
                result: None,
                capacity_batch: 0,
                state_indices: None,
            }),
        })
    }

    /// Ensure the per-call device buffers can hold at least `batch` states.
    /// Grows by reallocating to `batch` exactly the first time we hit a new
    /// max; subsequent calls with smaller batches reuse the existing
    /// allocation. We never shrink.
    fn ensure_capacity(
        &self,
        batch: usize,
        ds: &mut DeviceState,
    ) -> Result<(), CudaError> {
        if batch <= ds.capacity_batch {
            return Ok(());
        }
        let stream = &self.stream;
        ds.third_dice = Some(stream.alloc_zeros::<f32>(batch * NUM_DICE_F)?);
        ds.second_keepers = Some(stream.alloc_zeros::<f32>(batch * NUM_KEEPERS_F)?);
        ds.second_dice = Some(stream.alloc_zeros::<f32>(batch * NUM_DICE_F)?);
        ds.first_keepers = Some(stream.alloc_zeros::<f32>(batch * NUM_KEEPERS_F)?);
        ds.first_dice = Some(stream.alloc_zeros::<f32>(batch * NUM_DICE_F)?);
        ds.result = Some(stream.alloc_zeros::<f32>(batch)?);
        ds.state_indices = Some(stream.alloc_zeros::<u32>(batch)?);
        ds.capacity_batch = batch;
        Ok(())
    }

    /// The kernel-and-cuBLAS pipeline for a single level. Returns the EV per
    /// state in `states`. `_state_scores` is unused — the device-resident
    /// `device_state.state_scores` is kept in sync via the `scatter_results`
    /// kernel, so we don't re-upload the host buffer every level.
    ///
    /// Note on multi-build reuse: the device-side `state_scores` carries
    /// values from prior `Scores::new_with` calls. That's safe because the
    /// per-state EV is deterministic — any state's child indices read at
    /// the level the kernel cares about hold the same values either way.
    /// Indices BELOW the current level may hold stale (final-EV) values,
    /// but the kernel never reads them: `build_third_dice` only reads
    /// state_scores at child indices, which are at strictly higher levels.
    fn compute_level_inner(
        &self,
        states: &[State],
        _state_scores: &[f32],
    ) -> Result<Vec<f32>, CudaError> {
        let stream = &self.stream;
        let batch = states.len();
        let batch_i32 = batch as i32;

        let mut ds = self
            .device_state
            .lock()
            .expect("CudaBuildBackend mutex poisoned");

        // Lazy-grow the per-call buffers if this batch is the largest yet.
        self.ensure_capacity(batch, &mut ds)?;

        // Split-borrow the persistent buffers into disjoint mutable
        // references. Without this we can't both read one buffer and write
        // another in the same statement (the borrow checker treats methods
        // on the same `MutexGuard` as overlapping borrows).
        let DeviceState {
            ref mut state_scores,
            ref mut third_dice,
            ref mut second_keepers,
            ref mut second_dice,
            ref mut first_keepers,
            ref mut first_dice,
            ref mut result,
            ref mut state_indices,
            ..
        } = *ds;
        let third_dice = third_dice.as_mut().expect("ensured");
        let second_keepers = second_keepers.as_mut().expect("ensured");
        let second_dice = second_dice.as_mut().expect("ensured");
        let first_keepers = first_keepers.as_mut().expect("ensured");
        let first_dice = first_dice.as_mut().expect("ensured");
        let result = result.as_mut().expect("ensured");
        let state_indices = state_indices.as_mut().expect("ensured");

        // ------- H→D upload of state indices ------------------------------
        // Pack each State to its u32 index (matches `From<State> for usize`
        // in lib.rs). The kernel unpacks. Reuses the persistent device
        // buffer; only the first `batch` elements are written.
        let state_indices_h: Vec<u32> = states.iter().map(|s| usize::from(*s) as u32).collect();
        stream.memcpy_htod(&state_indices_h, state_indices)?;

        // ------- Step 1: build_third_dice ---------------------------------
        // Grid: (batch, 1, 1) blocks of (256, 1, 1) threads. batch on x to
        // dodge the gridDim.y 65535 cap. ENTRY_BLOCK_DIM=256 covers all 252
        // dice in one block; tail threads (252..255) early-out.
        let cfg_entry = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (ENTRY_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.f_build_third_dice)
                .arg(&*state_indices)
                .arg(&*state_scores)
                .arg(&self.dice_and_entry_scores)
                .arg(&self.yahtzee_dice)
                .arg(&mut *third_dice)
                .arg(&batch_i32)
                .launch(cfg_entry)?;
        }

        // ------- Step 2: keepers_from_dice via cuBLAS sgemm ---------------
        // Logical: second_keepers[b,k] = Σ_d KEEPERS_TO_DICE_PROBABILITIES[k,d] * third_dice[b,d]
        //
        // Memory layout (row-major):
        //   keepers_to_dice (M):  shape (462, 252)
        //   third_dice (X):       shape (B, 252)
        //   second_keepers (Y):   shape (B, 462)
        //
        // cuBLAS is column-major. Trick: row-major data of logical shape
        // (rows, cols) read as cm with shape (cols, rows) gives the
        // transpose. To produce Y_rm (B×462) we compute Y^T_cm (462×B),
        // since Y_rm and Y^T_cm share storage. Y^T = M · X^T, so call
        // sgemm with op_a=T (read M_rm as cm gives M^T → transpose back to
        // M), op_b=N (read X_rm as cm already gives X^T, no transpose).
        //   m = K = 462, n = B, k = D = 252
        //   lda = D = 252 (cm rows of M_rm)
        //   ldb = D = 252 (cm rows of X_rm)
        //   ldc = K = 462 (cm rows of Y_rm-as-Y^T_cm)
        let gemm_cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: NUM_KEEPERS_F as i32,
            n: batch_i32,
            k: NUM_DICE_F as i32,
            alpha: 1.0,
            lda: NUM_DICE_F as i32,
            ldb: NUM_DICE_F as i32,
            beta: 0.0,
            ldc: NUM_KEEPERS_F as i32,
        };
        unsafe {
            self.blas.gemm(
                gemm_cfg,
                &self.keepers_to_dice,
                &*third_dice,
                &mut *second_keepers,
            )?;
        }

        // ------- Step 3: masked_max → second_dice -------------------------
        // Grid: (batch, 252, 1) blocks of (REDUCE_BLOCK_DIM, 1, 1) threads.
        // batch on x (no cap), dice_idx on y (252 < 65535).
        let cfg_reduce = LaunchConfig {
            grid_dim: (batch as u32, NUM_DICE_F as u32, 1),
            block_dim: (REDUCE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK_DIM * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.f_masked_max)
                .arg(&*second_keepers)
                .arg(&self.dice_to_keepers)
                .arg(&mut *second_dice)
                .arg(&batch_i32)
                .launch(cfg_reduce)?;
        }

        // ------- Step 4: keepers_from_dice via cuBLAS sgemm (again) -------
        unsafe {
            self.blas.gemm(
                gemm_cfg,
                &self.keepers_to_dice,
                &*second_dice,
                &mut *first_keepers,
            )?;
        }

        // ------- Step 5: masked_max → first_dice --------------------------
        unsafe {
            stream
                .launch_builder(&self.f_masked_max)
                .arg(&*first_keepers)
                .arg(&self.dice_to_keepers)
                .arg(&mut *first_dice)
                .arg(&batch_i32)
                .launch(cfg_reduce)?;
        }

        // ------- Step 6: initial_roll_ev → result -------------------------
        // row0 of KEEPERS_TO_DICE_PROBABILITIES occupies offsets [0..252] of
        // the row-major (462, 252) buffer, so we just pass `keepers_to_dice`
        // and the kernel reads `row0[d]` from the first 252 elements.
        let cfg_marg = LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (REDUCE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK_DIM * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            stream
                .launch_builder(&self.f_initial_roll_ev)
                .arg(&*first_dice)
                .arg(&self.keepers_to_dice)
                .arg(&mut *result)
                .arg(&batch_i32)
                .launch(cfg_marg)?;
        }

        // ------- Step 7: scatter result into device state_scores ----------
        // After this, the next level's `build_third_dice` can read its
        // children's EVs directly from device memory; no host round trip.
        const SCATTER_BLOCK: u32 = 256;
        let scatter_blocks = ((batch as u32) + SCATTER_BLOCK - 1) / SCATTER_BLOCK;
        let cfg_scatter = LaunchConfig {
            grid_dim: (scatter_blocks, 1, 1),
            block_dim: (SCATTER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.f_scatter_results)
                .arg(&*state_indices)
                .arg(&*result)
                .arg(&mut *state_scores)
                .arg(&batch_i32)
                .launch(cfg_scatter)?;
        }

        // ------- D→H of just the per-state result -------------------------
        // The trait still wants Vec<f32>; the host-side `Scores::set_scores`
        // also writes these into its own state_scores Array1. (That host-
        // side write is now redundant work for the CUDA path, but keeping
        // the trait shape lets `Scores` remain backend-agnostic and the
        // CPU path keep its straight-line read of `state_scores: &[f32]`.)
        //
        // The persistent `result` buffer may be larger than `batch`; copy
        // only the first `batch` floats out.
        let mut host_result = vec![0.0_f32; batch];
        stream.memcpy_dtoh(&result.slice(0..batch), &mut host_result)?;
        stream.synchronize()?;
        Ok(host_result)
    }
}

impl BuildBackend for CudaBuildBackend {
    type Error = CudaError;

    fn compute_level(
        &self,
        states: &[State],
        state_scores: &[f32],
    ) -> Result<Vec<f32>, Self::Error> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        // Errors propagate to `Scores::new_with`'s caller; mid-build CUDA
        // failure leaves the partial DP buffer in an inconsistent state but
        // the `Scores` value is dropped so it's never observed.
        self.compute_level_inner(states, state_scores)
    }
}
