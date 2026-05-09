# Performance characteristics

Canonical home for `yahtzee-core` performance numbers. CLAUDE.md links here
instead of carrying the tables itself, so updates land in one place.

Three sections:

1. **[Backend sweep](#backend-sweep)** — `Scores::new_with(&backend)` head-to-head, criterion `Scores::new_with` group. The summary table.
2. **[Per-level breakdown](#per-level-breakdown)** — `YAHTZEE_TRACE_LEVELS=1` output for each backend. Where the time goes inside one build.
3. **[Per-stage micro-benches](#per-stage-micro-benches)** — criterion `simd_batch_phases` group. Within-pipeline attribution: GEMV vs masked_max vs entry_actions fuse vs final dot.

Plus [hardware config](#hardware-config) and the [structural-sparsity facts](#structural-sparsity-of-the-probability-tables) that motivate the next round of optimization.

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
| `simd_batch` | **2.29 s** | **3.85×** | **Best CPU.** Outer-loop SIMD across 8 states. CLI default. |
| `cuda` | 950 ms | 9.28× | Per-level batched on GPU; cuBLAS sgemm + 3 NVRTC kernels. |

Per-state (`state_value/backends` group, single-thread, default state):

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

External comparison: the `timpalpant/yahtzee` Go reference takes ~45 s for the same table on the same hardware (16 threads). Our naive Rust is 5.1× faster than that, default Rust 7.7×, simd_batch 19.7×, CUDA 47×. (Their start-state EV converges to 254.49, ours to 254.5896 — likely a small game-rule difference, not investigated.)

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
| 12 | 1,598 | 15.6 | 7.9 | 6.2 | 2.3 |
| 11 | 9,135 | 86.3 | 44.2 | 29.9 | 6.2 |
| 10 | 31,322 | 303.3 | 154.5 | 104.3 | 41.3 |
| 9 | 71,237 | 710.2 | 361.5 | 247.4 | 95.7 |
| **8** | **112,596** | **1145.6** | **585.6** | **403.5** | **150.1** |
| **7** | **126,219** | **1288.3** | **671.7** | **467.2** | **169.4** |
| **6** | **100,619** | **1042.1** | **543.5** | **389.6** | **132.8** |
| 5 | 56,283 | 586.4 | 309.8 | 226.1 | 75.2 |
| 4 | 21,377 | 223.7 | 124.3 | 88.7 | 12.9 |
| 3 | 5,178 | 56.5 | 29.5 | 22.4 | 3.4 |
| 2 | 711 | 7.9 | 4.6 | 4.0 | 0.4 |
| 1 | 44 | 0.5 | 0.3 | 0.7 | 0.1 |
| 0 | 1 | 0.1 | 0.1 | 0.1 | 4.7* |
| **total compute** | | 5466 | 2837 | 1990 | 694 |
| wall (incl. BFS / GPU init) | | 5742 | 3113 | 2269 | 965 |

`*` CUDA's L=0 outlier is kernel-launch overhead on a 1-state batch — irrelevant in absolute terms.

The bolded levels (L=6, 7, 8) are 63–64% of compute on every backend. **Optimization wins at those levels move the needle; wins at L≤4 don't.**

CPU vs CUDA per heavy level:
- L=7: simd_batch 467 ms, cuda 169 ms — **2.76×**
- L=8: simd_batch 403 ms, cuda 150 ms — **2.69×**
- L=6: simd_batch 390 ms, cuda 133 ms — **2.93×**

That ~2.8× ratio is the "remaining headroom" between the best CPU backend and the GPU on the dominant levels.

---

## Per-stage micro-benches

Criterion `simd_batch_phases` group. Each bench drives one of the four phases of `simd_batch::compute_lanes` on a pre-filled scratch (so no warmup work shows up in the timing). Single-thread, cache-warm — these are *relative* attribution numbers, not absolute end-to-end. See ["Translating phase numbers"](#translating-phase-numbers-to-end-to-end) below.

```bash
cargo bench -p yahtzee-core --features simd -- "simd_batch_phases"
```

| Phase | Time per 8-state batch | Calls per batch | Subtotal | % of pipeline |
|---|---:|---:|---:|---:|
| `entry_actions_fuse` (steps 1+2) | 110.8 µs | 1 | 110.8 µs | **37.3%** |
| `gemv` (step 3 & step 5) | 47.2 µs | 2 | 94.4 µs | **31.8%** |
| `masked_max` (step 4 & step 6) | 45.9 µs | 2 | 91.8 µs | **30.9%** |
| `final_dot` (step 7) | 0.10 µs | 1 | 0.10 µs | 0.0% |
| **pipeline total** | | | **~297 µs** | 100% |

### What this says about future optimizations

- **GEMV + masked_max = 62.7% of pipeline.** Sparsifying both (matrix is 3.75% dense, see below) attacks ~26× theoretical speedup on those phases. End-to-end ceiling: pipeline drops 297 µs → ~115 µs, roughly **2.6× CPU build wall-clock** if the speedup is fully realized. That'd land simd_batch at ~880 ms, vs cuda at 950 ms.
- **`entry_actions_fuse` is the irreducible per-state work** (`is_valid_action`, `score_and_child`, `state_scores[child_idx]` reads). It's already as scalar-tight as it gets in a per-state walk. Wins here would need a structural change — e.g. precomputing `(score, child_idx)` tables across batches.
- **`final_dot` is free** — 0.03% of the pipeline. Don't touch.

### Translating phase numbers to end-to-end

The 297 µs/batch number is single-thread, cache-warm. End-to-end on simd_batch is 2.29 s wall × 16 threads ≈ 36.6 s of CPU time across 67,056 batches (= 536,448 valid states / 8) → ~546 µs/batch effective. The ~1.84× gap vs the cache-warm bench is the parallel scaling penalty (memory bandwidth contention on the shared 4 MB `state_scores` slice across 16 cores). Phase *ratios* survive this multiplier; the absolute numbers don't.

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

Total nonzeros = **4,368** out of 462 × 252 = 116,424 → **3.75% density**, mean ≈9.5 nz per row. `DICE_TO_ALLOWED_KEEPERS` has the same support count (transpose).

The current `phase_gemv` and `phase_masked_max` iterate the full 462×252 every level. A CSR-like format that only visits the 4,368 nonzeros — combined with the existing state-as-lane SoA layout (each "input[d] across 8 lanes" is already a contiguous 256-bit load, no gather needed) — is the natural next optimization. See the GEMV+masked_max fusion idea in the design notes.

---

## Reproducing all the above

```bash
# Backend sweep (~7 min total).
cargo bench -p yahtzee-core --features "simd faer cuda" -- "Scores::new_with"

# Per-state sweep (~30 s).
cargo bench -p yahtzee-core --features "simd faer" -- "state_value/backends"

# Per-stage micro-benches (~25 s).
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
