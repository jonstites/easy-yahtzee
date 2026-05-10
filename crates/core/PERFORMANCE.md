# Performance characteristics

Canonical home for `yahtzee-core` performance numbers. CLAUDE.md links here
instead of carrying the tables itself, so updates land in one place.

Three sections:

1. **[Backend sweep](#backend-sweep)** — `Scores::new_with(&backend)` head-to-head, criterion `Scores::new_with` group. The summary table.
2. **[Per-level breakdown](#per-level-breakdown)** — `YAHTZEE_TRACE_LEVELS=1` output for each backend. Where the time goes inside one build.
3. **[Per-stage micro-benches](#per-stage-micro-benches)** — criterion `simd_batch_phases` group. Within-pipeline attribution: dense GEMV vs masked_max vs the sparse fused round vs entry_actions fuse vs final dot.

Plus [hardware config](#hardware-config) and the [structural-sparsity facts](#structural-sparsity-of-the-probability-tables) that the current `simd_batch` exploits.

---

## Hardware config

Numbers below are from one host running cold-fans, no other load:

| | |
|---|---|
| CPU | AMD Ryzen 7 9800X3D (Zen 5, 8 physical / 16 logical cores, 3D-VCache) |
| GPU | NVIDIA GeForce RTX 5070 Ti (16 GiB) |
| OS / kernel | Linux 7.0.0-15-generic |
| rustc | 1.95.0 |
| RAYON_NUM_THREADS | unset (rayon defaults to 16 logical cores) |
| Driver / CUDA | 595.58.03 / 13.2 |

Re-run the sweep on a different host: numbers will move. Treat the *ratios* as more stable than absolute ms.

---

## Backend sweep

Criterion `Scores::new_with` group, `sample_size=10`, `measurement_time=60s` (criterion may exceed to collect samples — naive ran ~88 s total).

```bash
cargo bench -p yahtzee-core --features "simd faer cuda" -- "Scores::new_with"
```

| Backend | Build time | vs naive | Notes |
|---|---|---|---|
| `naive` | 8.82 s | 1.00× | Reference scalar `for`-loop `LinalgBackend`. Honest baseline. |
| `ndarray` | 5.81 s | 1.52× | `matrixmultiply` GEMV via ndarray's `.dot()`. Default `Scores::new()`. |
| `faer` | 4.88 s | 1.81× | faer's `Mat * Col` GEMV; quietly the best CPU GEMV at 252×462. |
| `simd` | 3.13 s | 2.82× | Per-state intra-state SIMD on the masked-max via `wide::f32x8`. |
| `simd_batch` | **680 ms** | **12.97×** | **Best end-to-end.** Outer-loop SIMD across 8 states + sparse-fused GEMV/masked_max + flat-array `DICE_AND_ENTRY_SCORES` indexing. CLI default. |
| `cuda` | 951 ms | 9.28× | Per-level batched on GPU; cuBLAS sgemm + 3 NVRTC kernels. |

`simd_batch` overtook CUDA after the sparse fused-keeper-round landed (see [`phase_fused_keeper_round`](#per-stage-micro-benches) below). The previous dense `simd_batch` was 2.29 s; the current row is **3.37× faster than that** thanks to (a) a CSR-format K2D combined with merged GEMV + masked_max in one pass per keeper, and (b) replacing `Array2<u8>` with `[[u8; 252]; 13]` for `DICE_AND_ENTRY_SCORES` — callgrind showed ndarray's `dimension_trait.rs` stride math + bounds checks were ~10% of `score_and_child` / `State::child` Ir, and a flat 2D array compiles to one `lea` per access.

Per-state (`state_value/backends` group, single-thread, default state — these are the per-state `LinalgBackend` impls, unchanged by the fused-round work):

```bash
cargo bench -p yahtzee-core --features "simd faer" -- "state_value/backends"
```

| Backend | Per-state EV | vs naive |
|---|---|---|
| `naive` | 161 µs | 1.00× |
| `ndarray` | 116 µs | 1.39× |
| `faer` | 98 µs | 1.64× |
| `simd` | 53 µs | 3.04× |

(No `simd_batch` row: it's a `BuildBackend`, not a `LinalgBackend` — vectorizes across states, so single-state-per-call is meaningless.)

External comparison: the `timpalpant/yahtzee` Go reference takes ~45 s for the same table on the same hardware (16 threads). Our naive Rust is 5.1× faster than that, default Rust 7.7×, simd_batch **66×**, CUDA 47×. (Their start-state EV converges to 254.49, ours to 254.5896 — likely a small game-rule difference, not investigated.)

---

## Per-level breakdown

`YAHTZEE_TRACE_LEVELS=1` dumps `level=N batch=B collect=…ms compute=…ms write=…ms` per DP level. `collect` is the BFS-aware state enumeration, `compute` is the `BuildBackend::compute_level` call, `write` is the post-level scatter back into `state_scores`.

Reproducer:

```bash
YAHTZEE_TRACE_LEVELS=1 YAHTZEE_BACKEND=simd_batch \
  cargo run -p yahtzee-core --release --features simd --example time_build
YAHTZEE_TRACE_LEVELS=1 \
  cargo run -p yahtzee-core --release --features cuda --example cuda_smoke
```

`compute` is 88–95% of CPU wall; collect/write are noise (sub-ms each level). Numbers below are `compute_ms` only.

| Level | States | ndarray | simd | simd_batch | cuda |
|---:|---:|---:|---:|---:|---:|
| 12 | 1,598 | 15.6 | 7.9 | 1.6 | 2.3 |
| 11 | 9,135 | 86.3 | 44.2 | 4.6 | 6.2 |
| 10 | 31,322 | 303.3 | 154.5 | 18.2 | 41.3 |
| 9 | 71,237 | 710.2 | 361.5 | 50.9 | 95.7 |
| **8** | **112,596** | **1145.6** | **585.6** | **94.0** | **150.1** |
| **7** | **126,219** | **1288.3** | **671.7** | **119.9** | **169.4** |
| **6** | **100,619** | **1042.1** | **543.5** | **108.3** | **132.8** |
| 5 | 56,283 | 586.4 | 309.8 | 67.6 | 75.2 |
| 4 | 21,377 | 223.7 | 124.3 | 28.6 | 12.9 |
| 3 | 5,178 | 56.5 | 29.5 | 7.6 | 3.4 |
| 2 | 711 | 7.9 | 4.6 | 1.7 | 0.4 |
| 1 | 44 | 0.5 | 0.3 | 0.4 | 0.1 |
| 0 | 1 | 0.1 | 0.1 | 0.1 | 4.7* |
| **total compute** | | 5466 | 2837 | 503 | 694 |
| wall (incl. BFS / GPU init) | | 5742 | 3113 | 681 | 965 |

`*` CUDA's L=0 outlier is kernel-launch overhead on a 1-state batch — irrelevant in absolute terms.

The bolded levels (L=6, 7, 8) are 64% of compute on every backend. **simd_batch beats CUDA at every level from L=5 upward**; CUDA only wins at L=4 and below where its kernel-launch latency amortizes worse. Net: simd_batch wins by 184 ms wall.

The previous dense `simd_batch` column at the heavy levels was L=8 403.5 ms / L=7 467.2 ms / L=6 389.6 ms — sparse fused gives **3.6×–4.3× per heavy level**.

---

## Per-stage micro-benches

Criterion `simd_batch_phases` group. Each bench drives one phase of `simd_batch::compute_lanes` on a pre-filled scratch (so no warmup work shows up in the timing). Single-thread, cache-warm — these are *relative* attribution numbers, not absolute end-to-end. See ["Translating phase numbers"](#translating-phase-numbers-to-end-to-end) below.

```bash
cargo bench -p yahtzee-core --features simd -- "simd_batch_phases"
```

The production pipeline is now `phase_entry_actions_fuse` + 2× `phase_fused_keeper_round` + `phase_final_dot`. The dense `phase_gemv` / `phase_masked_max` are kept as bench-only baselines so the sparse-fusion win stays measurable on every CI run.

| Phase | Time per 8-state batch | Calls per batch | Subtotal | % of pipeline |
|---|---:|---:|---:|---:|
| `entry_actions_fuse` (steps 1+2) | 106.5 µs | 1 | 106.5 µs | **93.7%** |
| `fused_keeper_round` (sparse, steps 3+4 / 5+6) | 3.53 µs | 2 | 7.05 µs | 6.2% |
| `final_dot` (step 7) | 0.10 µs | 1 | 0.10 µs | 0.1% |
| **pipeline total** | | | **~114 µs** | 100% |
| | | | | |
| dense `gemv` (bench-only baseline) | 68.9 µs | — | — | — |
| dense `masked_max` (bench-only baseline) | 45.5 µs | — | — | — |

Sparse fused vs dense: `phase_fused_keeper_round` at 3.53 µs replaces the dense `gemv` (68.9) + `masked_max` (45.5) = 114.4 µs pair → **32× speedup on this stage**, beating the 3.75% density ratio's 26.7× theoretical FMA reduction (the extra factor comes from skipping the materialization of the 462-wide intermediate buffer). Pipeline total dropped 297 µs → 114 µs (**2.6×**); end-to-end build dropped 2.29 s → 680 ms (**3.4×**).

### What this says about the *next* round of optimizations

Callgrind on `phase_entry_actions_fuse` (see [Profiling](#profiling)) attributed its ~2.4M Ir/call as: ~80% inside `score_and_child` + `State::child` (per-state DP recurrence math), ~15% on the `state_scores[child_idx]` indexed load (mostly L1 misses absorbed by V-Cache; only ~0.06 LL misses/call), ~5% on the `is_valid_action` branch (6.7% mispredict rate on the bench's level-1 states; presumably higher on heavier production levels). It's compute-bound, not memory-bound.

- **`entry_actions_fuse` is now 94% of the pipeline.** Everything else is rounding error. The biggest remaining win is **vectorizing `score_and_child` across the 8 lanes** — every conditional inside it (`upper_complete`, `yahtzee_bonus_eligible`, joker rule) is data-dependent per lane but uniformly bool-mask-able. A SoA `State8 { entries: [u16; 8], upper: [u8; 8], yahtzee_eligible: [bool; 8] }` representation plus a `score_and_child_8x` SIMD primitive could compute all 8 lanes' (score, child_idx) in roughly one current scalar call's worth of Ir. Theoretical upper bound: 8× on the dominant cost → ~3-4× on `entry_actions_fuse` → ~3× e2e. Comparable scope to the sparse fused work; deferred.
- A simpler change already landed: **flat `[[u8; 252]; 13]` for `DICE_AND_ENTRY_SCORES`** (was `Array2<u8>`). Callgrind showed ndarray's `dimension_trait.rs` stride math + bounds checks were ~10% of `score_and_child` / `State::child` self time — a flat 2D array compiles to one `lea` per access. Net: 13.5% wall-clock improvement, single-PR change.
- **`fused_keeper_round` is unlikely to drop further.** The dot product walks 4,368 nonzeros across all 462 keepers — that's 9.5 nz/row average. A k=0 row with 252 nz dominates a single k=5 row with 1 nz. With sequential FMAs at one cycle each, 3.5 µs ≈ 18,000 cycles ≈ 8 cycles/nz, which is roughly memory-bandwidth-bound on the `in_dice` load + L1 hit on the CSR arrays. Squeezing this further would need either (a) a layout change that batches keepers of the same size to enable horizontal SIMD across rows, or (b) folding the scatter-max into a full L1-resident running max — both speculative.
- **`final_dot` is free** — 0.09% of the pipeline. Don't touch.

### Translating phase numbers to end-to-end

The 114 µs/batch number is single-thread, cache-warm. End-to-end on simd_batch is 680 ms wall × 16 threads ≈ 10.9 s of CPU time across 67,056 batches (= 536,448 valid states / 8) → ~162 µs/batch effective. The ~1.42× gap vs the cache-warm bench is the parallel scaling penalty (memory bandwidth contention on the shared 4 MB `state_scores` slice across 16 cores). Phase *ratios* survive this multiplier; the absolute numbers don't.

Pre-sparse-fused (when entry_actions_fuse was 37% of pipeline), this gap was 1.84×. The smaller scaling penalty post-sparse is consistent with the smaller working set: dropping `second_keepers` / `first_keepers` (462 × 8 × 4 = 14.7 KiB each, ×2) shrunk per-thread scratch from ~50 KiB to ~24 KiB, easing L1/L2 pressure across cores.

---

## Structural sparsity of the probability tables

Both `KEEPERS_TO_DICE_PROBABILITIES` (462 × 252) and its transposed-support sibling `DICE_TO_ALLOWED_KEEPERS` (252 × 462) are mostly zero. A keeper of size *k* is compatible only with full rolls that contain it as a sub-multiset, i.e. you re-roll (5−k) dice. Counting per keeper-size:

| k | # keepers | compatible dice / row | nz |
|:-:|:-:|:-:|:-:|
| 0 | 1 | 252 | 252 |
| 1 | 6 | 126 | 756 |
| 2 | 21 | 56 | 1176 |
| 3 | 56 | 21 | 1176 |
| 4 | 126 | 6 | 756 |
| 5 | 252 | 1 | 252 |
| **total** | **462** | | **4,368** |

Total nonzeros = **4,368** out of 462 × 252 = 116,424 → **3.75% density**, mean ≈9.5 nz per row.

A key invariant: `K2D[k, d] > 0 ⟺ k ⊆ d ⟺ DICE_TO_ALLOWED_KEEPERS[d, k] = 1`. Same predicate. So one CSR over K2D drives both ops in `compute_lanes` — the column indices of K2D row `k` are *exactly* the dice that the masked-max for that `k` needs to update. That equivalence is what `phase_fused_keeper_round` exploits: walk K2D by row, compute the sparse dot product `val`, and immediately scatter-max `val` into every `out_dice[d]` for `d` in the same column-index list. The dense `second_keepers` 462-wide intermediate is never materialized.

The realized speedup (26.3×, see [phase benches](#per-stage-micro-benches)) tracked the density-ratio prediction (26.7×) almost exactly — the rare case where a perf calculation cashes out at the theoretical limit.

---

## Reproducing all the above

```bash
# Backend sweep (~7 min total).
cargo bench -p yahtzee-core --features "simd faer cuda" -- "Scores::new_with"

# Per-state sweep (~30 s).
cargo bench -p yahtzee-core --features "simd faer" -- "state_value/backends"

# Per-stage micro-benches (~30 s).
cargo bench -p yahtzee-core --features simd -- "simd_batch_phases"

# Per-level traces (one shot per backend, < 10 s each).
YAHTZEE_TRACE_LEVELS=1 YAHTZEE_BACKEND=simd_batch \
  cargo run -p yahtzee-core --release --features simd --example time_build
YAHTZEE_TRACE_LEVELS=1 \
  cargo run -p yahtzee-core --release --features cuda --example cuda_smoke
```

To compare runs, use criterion baselines:

```bash
cargo bench -p yahtzee-core --features simd -- --save-baseline before
# ...make changes...
cargo bench -p yahtzee-core --features simd -- --baseline before
```

---

## Profiling

For drilling into a specific phase (e.g. when picking the next bottleneck), use `valgrind --tool=callgrind` against the `profile_phases` example, which runs each phase in its own `#[inline(never)]` wrapper so callgrind attribution is clean.

```bash
cargo build -p yahtzee-core --release --features simd --example profile_phases
valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes \
    --callgrind-out-file=callgrind.out \
    ./target/release/examples/profile_phases
callgrind_annotate callgrind.out --auto=yes
```

`Scores::new_with(...)` is *not* called by this example — it uses synthetic 4 MiB `state_scores` (zeroed) instead, since the access pattern (which addresses are read) is what cache simulation cares about, not the values. The 8-state batch is constructed by taking the first 8 children of `State::default()` (one different action filled per lane), so the per-batch cache footprint of `state_scores[child_idx]` reads is realistic for production-shaped batches.

Cache+branch simulation slows the run ~150-200× — default iter counts (set via env: `ITERS_ENTRY_ACTIONS=2000`, `ITERS_FUSED=60000`, `ITERS_DENSE_GEMV=5000`, `ITERS_DENSE_MASKED_MAX=5000`) keep the wall-clock under 4 minutes. Drop them another 5× for a ~30s instructions-only run by passing `--cache-sim=no --branch-sim=no` to valgrind.

Release builds carry `debug = "line-tables-only"` (set in the workspace `Cargo.toml`) so `callgrind_annotate --auto=yes` produces source-line attribution. Release binary size impact: tens of KB of `.debug_line`; strips fine.
