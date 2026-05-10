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
| `simd_batch` | **229 ms** | **38.5×** | **Best end-to-end.** Outer-loop SIMD across 8 states + sparse-fused GEMV/masked_max + flat-array `DICE_AND_ENTRY_SCORES` + `#[inline]` on the per-state DP fns + precomputed `(score + state_scores[child])` table + vectorized `score_and_child` across the 8 lanes + parallel BFS reachability filter. CLI default. |
| `cuda` | 951 ms | 9.28× | Per-level batched on GPU; cuBLAS sgemm + 3 NVRTC kernels. |

`simd_batch` overtook CUDA after the sparse fused-keeper-round landed and has pulled progressively further ahead with each subsequent optimization. Walking the arc, end-to-end:

| Stage | Wall (criterion) | vs prior |
|---|---:|---|
| dense `simd_batch` (post-flat-array) | 2.29 s | baseline |
| + sparse fused keeper round | 784 ms | 2.92× |
| + `Array2<u8>` → `[[u8; 252]; 13]` | 680 ms | 1.15× |
| + `#[inline]` on `score_and_child` / `State::child` | 476 ms | 1.43× |
| + precomputed `(score + state_scores[child])` table | 439 ms | 1.09× |
| + vectorized `score_and_child` across 8 lanes | 349 ms | 1.26× |
| + parallel BFS reachability filter | **229 ms** | 1.52× |

The current row is **10.0× faster than the dense `simd_batch` starting point**. The two most recent stages share a common shape — split the inline-per-iteration work into a Phase A (build a per-batch table of resolved values) and Phase B (branchless SIMD max-reduce over the table). Round 1 just changed the loop nest of Phase A (locality-preserving `(s outer, a middle, d inner)` so `state_scores[child]` reads stay in L1 for fixed `(s, a)`); Round 2 flipped it to `(a outer, d middle, s inner-vectorized)` so all 8 lanes' `score_and_child` work runs in one `i32x8`/`f32x8` pass per `(action, dice)` cell. Round 2 alone is 5.7× on the entry-actions phase in cache-warm benches and 1.26× e2e.

Earlier wins came from elsewhere: the sparse fused keeper round dropped K2D's 116,424-FMA dense GEMV+masked_max pair to a 4,368-nonzero CSR walk that scatter-maxes directly into `out_dice` (26.3× on that phase, matching the 1/0.0375 = 26.7× density ceiling). The `Array2<u8>` → `[[u8; 252]; 13]` flatten removed ndarray's stride math from the hot lookup. And the `#[inline]` on `score_and_child` / `State::child` recovered −30% wall-clock from a two-line change — callgrind showed ~12% of total program Ir was the function-prologue/epilogue overhead alone (push/pop register saves on a 30-line body the inliner refused to touch without a hint at -C opt-level=3 + no-LTO), and the inliner also did cross-fn CSE on the `DICE_AND_ENTRY_SCORES` lookup that both functions were doing redundantly.

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

External comparison: the `timpalpant/yahtzee` Go reference takes ~45 s for the same table on the same hardware (16 threads). Our naive Rust is 5.1× faster than that, default Rust 7.7×, simd_batch **196×**, CUDA 47×. (Their start-state EV converges to 254.49, ours to 254.5896 — likely a small game-rule difference, not investigated.)

---

## Thread scaling

`simd_batch` scaling on the 9800X3D (8 physical / 16 logical):

| Threads | Wall (ms) | Speedup vs 1T | Efficiency |
|---:|---:|---:|---:|
| 1 | 1570 | 1.00× | 100% |
| 2 | 794 | 1.98× | **99%** |
| 4 | 406 | 3.87× | **97%** |
| 8 | 229 | 6.86× | **86%** |
| 12 | 228 | 6.89× | 57% |
| 16 | 242 | 6.49× | 41% |

Reproduce: `for n in 1 2 4 8 12 16; do RAYON_NUM_THREADS=$n YAHTZEE_BACKEND=simd_batch cargo run -p yahtzee-core --release --features simd --example time_build; done`

**Sweet spot is 8-12 threads.** Going from 8→16 threads adds nothing (and slightly *hurts*: 229 → 242 ms, ~6%) — the 9800X3D's SMT can't hide behind any execution-unit slack because vectorized SIMD already saturates the FPU. The second SMT thread per core just contests for L1d / L2 / store buffer slots.

The 86% efficiency at 8T is the "headroom" from the parallel BFS landing in commit 4ae79d9. Before that change, 8T efficiency was 54%: a fixed ~175 ms sequential BFS dragged Amdahl's law on every config from 4T upward. Once that disappeared, the per-level DP work (which scaled at ~7× on heavy levels at 8T even *before* the fix) was finally visible end-to-end.

The remaining 14% gap at 8T is fixed-cost overhead: the post-build `state_scores` write-back (single-threaded), the `into_par_iter` startup on tiny levels (L=12 has 1598 states, ~1.5 µs of work split 8 ways gets dwarfed by rayon dispatch), and the levels of trace prints when enabled.

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

Per-level `compute=` is wall-time of the parallel `compute_level` call. Note the BFS reachability filter (`set_valid_states`) runs *before* level 12 and isn't shown in any column — see the [Thread scaling](#thread-scaling) section for the relevant numbers.

| Level | States | ndarray | simd | simd_batch (R1) | simd_batch (R2) | simd_batch (R3) | cuda |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1,598 | 15.6 | 7.9 | 1.6 | 1.8 | 1.8 | 2.3 |
| 11 | 9,135 | 86.3 | 44.2 | 3.7 | 3.3 | 3.3 | 6.2 |
| 10 | 31,322 | 303.3 | 154.5 | 14.1 | 10.2 | 10.2 | 41.3 |
| 9 | 71,237 | 710.2 | 361.5 | 37.2 | 23.4 | 23.4 | 95.7 |
| **8** | **112,596** | **1145.6** | **585.6** | **67.4** | **36.4** | **36.4** | **150.1** |
| **7** | **126,219** | **1288.3** | **671.7** | **85.0** | **40.2** | **40.2** | **169.4** |
| **6** | **100,619** | **1042.1** | **543.5** | **76.1** | **32.7** | **32.7** | **132.8** |
| 5 | 56,283 | 586.4 | 309.8 | 47.1 | 18.5 | 18.5 | 75.2 |
| 4 | 21,377 | 223.7 | 124.3 | 20.4 | 7.1 | 7.1 | 12.9 |
| 3 | 5,178 | 56.5 | 29.5 | 5.5 | 1.8 | 1.8 | 3.4 |
| 2 | 711 | 7.9 | 4.6 | 1.3 | 0.6 | 0.6 | 0.4 |
| 1 | 44 | 0.5 | 0.3 | 0.4 | 0.3 | 0.3 | 0.1 |
| 0 | 1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 4.7* |
| total compute_level | | 5466 | 2837 | 359 | 176 | 176 | 694 |
| BFS (`set_valid_states`) | | 175 | 175 | 175 | 175 | **53** | 175 |
| **wall (criterion, warm)** | | — | — | **439** | **349** | **229** | **951** |

`*` CUDA's L=0 outlier is kernel-launch overhead on a 1-state batch — irrelevant in absolute terms.

The bolded levels (L=6, 7, 8) are 64% of `compute_level` time on every backend. **simd_batch (R3) beats CUDA at every level from L=5 upward**; CUDA only wins at L=4 and below where its kernel-launch latency amortizes worse. Net: simd_batch wins by 720 ms wall.

R3 vs R2 per-level `compute_level` is identical — Round 3 only changed `set_valid_states`, not the DP work. The 120 ms wall improvement (349 → 229 ms) comes entirely from the BFS line: 175 → 53 ms at 16 threads (3.3× on that piece, ~9× on the previously-sequential portion of total wall). The earlier rounds had the BFS as a hidden Amdahl ceiling: `compute_level` total was 176 ms but wall was 349 ms, a 2.0× gap whose cause was 175 ms of BFS the trace didn't surface. With the BFS down to 53 ms, the gap closes to 1.30× (176 + 53 = 229 ms).

R2 vs R1 per heavy level: roughly **2.0×** uniformly (67.4 → 36.4, 85.0 → 40.2, 76.1 → 32.7), tracking the per-phase `entry_actions_vectorized` 5.7× on cache-warm but giving back some of the ratio to gather latency that doesn't amortize as well in production-cold conditions. R1 vs the prior dense `simd_batch` at the heavy levels (403.5 / 467.2 / 389.6 ms) was 4.0–6.0× per level — sparse fused was the bigger structural win, vectorization is icing.

---

## Per-stage micro-benches

Criterion `simd_batch_phases` group. Each bench drives one phase of `simd_batch::compute_lanes` on a pre-filled scratch (so no warmup work shows up in the timing). Single-thread, cache-warm — these are *relative* attribution numbers, not absolute end-to-end. See ["Translating phase numbers"](#translating-phase-numbers-to-end-to-end) below.

```bash
cargo bench -p yahtzee-core --features simd -- "simd_batch_phases"
```

The production pipeline is now `phase_entry_actions_vectorized` + 2× `phase_fused_keeper_round` + `phase_final_dot`. The Round 0 inline-loop `phase_entry_actions_fuse`, the Round 1 scalar-per-lane `phase_entry_actions_precomputed`, and the dense `phase_gemv` / `phase_masked_max` are all kept as bench-only baselines so each successive win stays measurable on every CI run.

| Phase | Time per 8-state batch | Calls per batch | Subtotal | % of pipeline |
|---|---:|---:|---:|---:|
| `entry_actions_vectorized` (steps 1+2) | 12.32 µs | 1 | 12.32 µs | **63.0%** |
| `fused_keeper_round` (sparse, steps 3+4 / 5+6) | 3.56 µs | 2 | 7.11 µs | 36.5% |
| `final_dot` (step 7) | 0.10 µs | 1 | 0.10 µs | 0.5% |
| **pipeline total** | | | **~19.5 µs** | 100% |
| | | | | |
| `entry_actions_precomputed` (R1, bench-only) | 70.5 µs | — | — | — |
| `entry_actions_fuse` (R0, bench-only) | 72.0 µs | — | — | — |
| dense `gemv` (bench-only) | 46.4 µs | — | — | — |
| dense `masked_max` (bench-only) | 68.0 µs | — | — | — |

Sparse fused vs dense: `phase_fused_keeper_round` at 3.56 µs replaces the dense `gemv` (46.4) + `masked_max` (68.0) = 114.4 µs pair → **32× speedup on this stage**, matching the 1/0.0375 = 26.7× density-ratio ceiling plus the savings from never materializing the 462-wide intermediate buffer.

R2 vs R1 entry_actions: 12.32 µs vs 70.5 µs → **5.7× speedup on this stage**. The win is from running all 8 lanes' `score_and_child` in one SIMD pass per `(action, dice)` cell — `i32x8` for the bit-pack child_idx computation, `f32x8` for the score arithmetic and bonuses, mask-blend for the joker rule. The gather of `state_scores[child_idx]` is still 8 scalar loads (no VGATHERDPS through `wide`), but the locality argument from R1 still holds per lane: child_idx for `(s, a, d)` and `(s, a, d+1)` differ by at most a small upper-score delta.

R1 vs R0 entry_actions: 70.5 µs vs 72.0 µs → only ~2% on cache-warm, but the production e2e win was bigger (-9% wall) because R1's loop-nest reorder made `state_scores[child]` reads localize for the cache prefetcher under cold-cache conditions that the bench can't reproduce. Single-thread cache-warm benches under-measure cold-cache wins; the e2e wall is the truth.

End-to-end pipeline: 297 µs (R0 dense GEMVs) → 55 µs (R0 + sparse fused) → 19.5 µs (R2). **15.2× pipeline reduction over the dense baseline.**

### What this says about the *next* round of optimizations

The pipeline is now 63% `entry_actions_vectorized` (12.3 µs) and 36% `fused_keeper_round` (3.56 µs ×2 = 7.1 µs). `final_dot` is rounding error.

- **`fused_keeper_round`** is at the per-cycle FMA ceiling. 4,368 nonzeros at ~1 FMA/cycle = ~4,368 cycles ≈ 1.0 µs at 4 GHz, plus L1 hit latencies on the CSR arrays and the `in_dice` reads. 3.5 µs ≈ 14k cycles ≈ 3.2 cycles/nz is roughly memory-bandwidth-bound on the `in_dice` random reads (each row's column indices are sorted but vary across rows). Further wins would need a layout change — e.g., batching keepers of the same size to enable horizontal SIMD across rows — and feel speculative.
- **`entry_actions_vectorized`** at 12.3 µs cache-warm is plausibly close to its compute-bound floor: 13 actions × 252 dice = 3,276 cells × ~10 cycles each (vectorized FMA + masking) = ~33k cycles ≈ 8 µs at 4 GHz, so we're maybe 1.5× off the dense-FMA ceiling. The ~4 µs gap is probably the gather (8 scalar loads per cell, even if mostly L1 hits, eats issue slots). Reclaiming that would mean either VGATHERDPS via `safe_arch` / `std::simd` (1-2 µs win at best, complicates the scalar-fallback path on non-AVX2 targets), or restructuring to do the gather as a separate phase with prefetch hints — diminishing returns.
- **Parallel scaling is now near-linear on physical cores.** R3's headline change (parallel BFS, 175 → 53 ms at 16T) closed a hidden Amdahl ceiling that had been masking how well the per-level DP work was already scaling. 8T efficiency went 54% → 86%; the remaining 14% gap is fixed-cost overhead (rayon dispatch on tiny levels, single-threaded write-back). 12T is now 1ms faster than 16T — SMT genuinely costs us at this point, since vectorized SIMD already saturates the FPU and the second SMT thread per core just contests for the same execution-unit slots.
- **The big "make it faster" levers are diminishing.** Best apparent next moves: (a) a CUDA sparse-fused-keeper-round port — same algorithm, NVRTC-compiled, scatter-max via shared memory — could probably claw the GPU back into the lead at high core counts, since GPUs handle "many independent batches" much better than 8-12 saturated CPU cores. (b) Pin threads to physical cores and skip SMT (`taskset -c 0-7` or rayon's thread-pool builder) — would shave a few % off 8T at zero algorithmic cost. (c) The `Scores::new_with_unvalidated` path opens up new DP orderings (regular `C(13, L) × 128` batch shape), which we *could* go after but the prior experiment didn't pan out.

### Translating phase numbers to end-to-end

R3 cache-warm pipeline = 19.5 µs/batch. Per-level `compute_level` total is 176 ms; that's now 77% of wall (vs 50% before R3) because the previously-hidden BFS overhead is gone. Plus the 53 ms BFS = 229 ms total, which matches criterion. End-to-end on simd_batch is 229 ms wall × 16 threads ≈ 3.7 s of CPU time across 67,056 batches → ~55 µs/batch effective. The 2.8× gap vs cache-warm is the parallel scaling penalty (smaller now that per-level work is less dominant relative to coordination overhead).

The remaining "near-linear scaling on physical cores, SMT contributes nothing" pattern is the steady state. The 9800X3D's 8 cores are the real ceiling for this workload. To go further on a CPU you'd need either more cores or per-core dispatch width that we're not using (AVX-512, when exposed by `wide`).

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
