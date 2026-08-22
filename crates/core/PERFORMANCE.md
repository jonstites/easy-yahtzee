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

Criterion `Scores::new_with` group, `sample_size=10`, `measurement_time=60s` (criterion may exceed to collect samples — naive ran ~80 s total).

```bash
cargo bench -p yahtzee-core --features "simd faer cuda" -- "Scores::new_with"
```

| Backend | Build time | vs naive | Notes |
|---|---|---|---|
| `naive` | 7.74 s | 1.00× | Reference scalar `for`-loop `LinalgBackend`. Honest baseline. |
| `ndarray` | 4.13 s | 1.87× | `matrixmultiply` GEMV via ndarray's `.dot()`. Default `Scores::new()`. |
| `faer` | 3.20 s | 2.42× | faer's `Mat * Col` GEMV; quietly the best CPU GEMV at 252×462. |
| `simd` | 2.57 s | 3.01× | Per-state intra-state SIMD on the masked-max via `wide::f32x8`. |
| `simd_batch` | **101 ms** | **76.6×** | **Best end-to-end.** Outer-loop SIMD across 8 states + sparse-fused GEMV/masked_max + flat-array `DICE_AND_ENTRY_SCORES` + `#[inline]` on the per-state DP fns + precomputed `(score + state_scores[child])` table + vectorized `score_and_child` across the 8 lanes + parallel BFS reachability filter + d-independent gather hoist for 6 of the 13 actions + per-score-class precomputation for the slow arm (6 distinct child states per upper action, 2 per YAHTZEE) + bucket-sort by `!yahtzee_eligible` (R6.5) + LazyLock hoist out of BFS (R7). CLI default. |
| `cuda` | 712 ms | 10.87× | Per-level batched on GPU; cuBLAS sgemm + 3 NVRTC kernels. |

`simd_batch` overtook CUDA after the sparse fused-keeper-round landed and has pulled progressively further ahead — currently **7.05× faster than CUDA** end-to-end. Walking the arc:

| Stage | Wall (criterion) | vs prior |
|---|---:|---|
| dense `simd_batch` (post-flat-array, pre-target-cpu=native) | 2.29 s | baseline |
| + sparse fused keeper round | 784 ms | 2.92× |
| + `Array2<u8>` → `[[u8; 252]; 13]` | 680 ms | 1.15× |
| + `#[inline]` on `score_and_child` / `State::child` | 476 ms | 1.43× |
| + precomputed `(score + state_scores[child])` table (R1) | 439 ms | 1.09× |
| + vectorized `score_and_child` across 8 lanes (R2) | 349 ms | 1.26× |
| + parallel BFS reachability filter (R3) | 229 ms | 1.52× |
| + hoist d-independent gather, 6 of 13 actions (R4) | 193 ms | 1.19× |
| + per-score-class precompute in slow arm (R5) | 152 ms | 1.27× |
| + `target-cpu=native` (AVX2 + FMA + BMI2 enabled, was SSE2 only) | 124.5 ms | 1.22× |
| + bucket-sort by `!yahtzee_eligible` (R6.5) | 119 ms | 1.04× |
| + LazyLock hoist + d-independent action collapse in BFS (R7) | **101 ms** | 1.18× |

The current row is **22.7× faster than the dense `simd_batch` starting point**. R7 (the most recent landed round) is structurally different from R0..R6.5: every prior round optimized the *DP per-level compute_level*, but R7 optimizes the *BFS reachability filter* (`Scores::set_valid_states`). After R7, BFS is ~28% of e2e wall (was ~40% pre-R7), and the DP work — specifically `phase_fused_keeper_round` — is back to dominating. The handoff between R6.5 and R7 predicted a state-encoding restructure (different bijection from `State` to `usize`) as the next big win, estimating 20-40% e2e. Cachegrind disproved that hypothesis: D1 miss rate is 0.1% and LLd is 0.0%, so cache locality isn't the bottleneck. The real cost was ~21% of total Ir going to `sync/once.rs` + `sync/atomic.rs` — the BFS hot loop's per-iteration LazyLock atomic checks for `DICE_AND_ENTRY_SCORES` and `YAHTZEE_DICE`. The per-stage micro-benches below didn't catch this because they only bench `phase_*` functions, not `set_valid_states`.

R5 (per-score-class precompute) was the prior big structural win, sharing a pattern with R4 (gather hoist): ask "for which inputs does the gathered `state_scores[child_idx]` actually depend on `dice`?" R4 found that 6 of 13 actions have d-independent children (entire gather hoists out of d-loop); R5 found that the remaining 7 have very *low-cardinality* d-dependence (upper actions: 6 distinct values per parent, indexed by `count(face, d)`; YAHTZEE: 2, indexed by `is_yahtzee(d)`). The R5 entry-actions phase was **2.36× faster** cache-warm vs R4 and **1.27× e2e** — total slow-arm gathers per batch dropped from 14,112 to ~304.

The two stages before R4 share a different shape — split the inline-per-iteration work into a Phase A (build a per-batch table of resolved values) and Phase B (branchless SIMD max-reduce over the table). R1 just changed the loop nest of Phase A (locality-preserving `(s outer, a middle, d inner)` so `state_scores[child]` reads stay in L1 for fixed `(s, a)`); R2 flipped it to `(a outer, d middle, s inner-vectorized)` so all 8 lanes' `score_and_child` work runs in one `i32x8`/`f32x8` pass per `(action, dice)` cell. R2 alone was 5.7× on the entry-actions phase in cache-warm benches and 1.26× e2e.

Earlier wins came from elsewhere: the sparse fused keeper round dropped K2D's 116,424-FMA dense GEMV+masked_max pair to a 4,368-nonzero CSR walk that scatter-maxes directly into `out_dice` (~28× on that phase post-target-cpu, matching the 1/0.0375 = 26.7× density ceiling within noise). The `Array2<u8>` → `[[u8; 252]; 13]` flatten removed ndarray's stride math from the hot lookup. And the `#[inline]` on `score_and_child` / `State::child` recovered −30% wall-clock from a two-line change — callgrind showed ~12% of total program Ir was the function-prologue/epilogue overhead alone (push/pop register saves on a 30-line body the inliner refused to touch without a hint at -C opt-level=3 + no-LTO), and the inliner also did cross-fn CSE on the `DICE_AND_ENTRY_SCORES` lookup that both functions were doing redundantly.

`target-cpu=native` (between R5 and R6.5) was an unrelated find: the default rustc target for `x86_64-unknown-linux-gnu` is SSE2, so every `wide::f32x8` op was silently lowering to two `f32x4` SSE2 ops and `mul + add` never fused into `vfmadd*ps`. Enabling target-cpu=native in `.cargo/config.toml` recovered ~22% e2e. The .cargo/config.toml is `cfg`-scoped to x86_64/aarch64 to leave the wasm cross-compile untouched.

Per-state (`state_value/backends` group, single-thread, default state — these are the per-state `LinalgBackend` impls, unchanged by R7):

```bash
cargo bench -p yahtzee-core --features "simd faer" -- "state_value/backends"
```

| Backend | Per-state EV | vs naive |
|---|---|---|
| `naive` | 147 µs | 1.00× |
| `ndarray` | 85 µs | 1.72× |
| `faer` | 66 µs | 2.22× |
| `simd` | 42 µs | 3.49× |

(No `simd_batch` row: it's a `BuildBackend`, not a `LinalgBackend` — vectorizes across states, so single-state-per-call is meaningless.)

External comparison: the `timpalpant/yahtzee` Go reference takes ~45 s for the same table on the same hardware (16 threads). Our naive Rust is 5.8× faster than that, default Rust 10.9×, simd_batch **445×**, CUDA 63×. (Their start-state EV converges to 254.49, ours to 254.5896 — likely a small game-rule difference, not investigated.)

---

## Thread scaling

`simd_batch` scaling on the 9800X3D (8 physical / 16 logical), post-R7 + target-cpu=native, single `time_build` sample per thread count:

| Threads | Wall (ms) | Speedup vs 1T | Efficiency |
|---:|---:|---:|---:|
| 1 | 562 | 1.00× | 100% |
| 2 | 291 | 1.93× | **96%** |
| 4 | 154 | 3.65× | **91%** |
| 8 | 110 | 5.11× | 64% |
| 12 | **95** | 5.91× | 49% |
| 16 | 110 | 5.11× | 32% |

Reproduce: `for n in 1 2 4 8 12 16; do RAYON_NUM_THREADS=$n YAHTZEE_BACKEND=simd_batch cargo run -p yahtzee-core --release --features simd --example time_build; done`

**Sweet spot is 12 threads.** Going from 12→16 adds nothing (and slightly *hurts*: 95 → 110 ms) — the 9800X3D's SMT can't hide behind any execution-unit slack because vectorized SIMD already saturates the FPU. The second SMT thread per core just contests for L1d / L2 / store buffer slots. 8→12 threads picks up ~16% wall (110 → 95 ms) from OS scheduling flexibility without full SMT pairing.

The 64% efficiency at 8T (compared to ~84% pre-target-cpu) is mostly fixed-cost overhead becoming a larger relative share now that the parallel work shrunk dramatically. Total fixed costs: the post-build `state_scores` write-back (single-threaded), the `into_par_iter` startup on tiny levels (L=12 has 1598 states, ~0.5 µs of work split 8 ways gets dwarfed by rayon dispatch), the BFS reduce-OR merge between levels (sequential by design), and trace prints when enabled. R3's parallel BFS removed the previous ~175 ms sequential ceiling that capped 8T efficiency at 54%; R7 then halved the BFS wall itself (56 ms → 29 ms at 16T), so the BFS line is no longer a noticeable Amdahl ceiling.

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

Current measurements (post-R7, target-cpu=native, 16T):

| Level | States | ndarray | simd | sb (R5, pre-target-cpu) | sb (R7) | cuda |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1,598 | 12.0 | 13.8 | 1.3 | 1.6 | 2.0 |
| 11 | 9,135 | 65.9 | 44.6 | 2.1 | 1.8 | 5.7 |
| 10 | 31,322 | 232.1 | 153.7 | 5.7 | 4.8 | 30.0 |
| 9 | 71,237 | 540.4 | 391.4 | 13.3 | 9.6 | 69.7 |
| **8** | **112,596** | **861.3** | **564.9** | **19.6** | **14.5** | **109.6** |
| **7** | **126,219** | **991.1** | **674.3** | **21.5** | **16.3** | **123.8** |
| **6** | **100,619** | **781.9** | **502.7** | **17.0** | **12.9** | **97.0** |
| 5 | 56,283 | 442.1 | 300.8 | 10.1 | 7.6 | 55.0 |
| 4 | 21,377 | 167.8 | 123.0 | 3.9 | 3.3 | 9.4 |
| 3 | 5,178 | 41.2 | 33.3 | 1.1 | 1.3 | 2.5 |
| 2 | 711 | 5.7 | 4.8 | 0.6 | 0.6 | 0.3 |
| 1 | 44 | 0.4 | 0.8 | 0.3 | 0.3 | 0.1 |
| 0 | 1 | 0.1 | 0.1 | 0.1 | 0.1 | 4.7* |
| total compute_level | | 4142 | 2808 | 96 | 75 | 510 |
| BFS (`set_valid_states`) | | 29 | 65** | 56 | 29 | 29 |
| **wall (criterion, warm)** | | **4127 ms** | **2566 ms** | **152 ms** | **101 ms** | **712 ms** |

`*` CUDA's L=0 outlier is kernel-launch overhead on a 1-state batch — irrelevant in absolute terms.
`**` `simd`'s BFS is noisier in single-sample traces because target-cpu=native makes the DP fast enough that scheduler jitter at lower-thread-count workloads shows up as BFS variance. R7's BFS line is identical regardless of which backend drives `compute_level` (BFS runs *before* level 12) — quoted here as 29 ms across backends for clarity.

The bolded levels (L=6, 7, 8) are ~58% of `compute_level` time on every backend. **simd_batch (R7) beats CUDA at every level from L=3 upward** (R5 beat it from L=4); CUDA only wins at L=2 and below where its kernel-launch latency amortizes worse and the absolute numbers are sub-ms anyway. Net: simd_batch wins by 611 ms wall vs CUDA.

R7 vs R5 (with target-cpu=native applied between): per heavy level roughly **1.30–1.35×** uniformly (19.6 → 14.5, 21.5 → 16.3, 17.0 → 12.9). R7 itself didn't touch the DP (only `set_valid_states`); these per-level gains come from `target-cpu=native` enabling AVX2 + FMA + BMI2 in `wide::f32x8`'s lowering (previously SSE2-only fallback). R7's own win is in the BFS line: 56 ms → 29 ms at 16T (−48%) — a ~21 ms wall reduction that the per-level table can't show because BFS happens *outside* `compute_level`. Combined R5 → R7 wall: 152 → 101 ms (-33.6%), split roughly 1/3 from R6.5 + target-cpu + R7's BFS hoist, 2/3 from target-cpu's effect on the unchanged DP phases.

Pre-target-cpu progression (numbers in the 5466 → 96 ms range of the old table) — see [Backend sweep "stage" table](#backend-sweep) for the full arc. Brief summary of the per-heavy-level deltas: **R5 vs R4** was 1.40–1.65× from per-score-class precompute (slow-arm gathers dropped 14,112 → 304); **R4 vs R3** was 1.20–1.27× from d-independent gather hoist (46% fewer gathers); **R3 vs R2** was identical per-level (R3 only changed BFS); **R2 vs R1** was 2.0× from vectorizing `score_and_child` across 8 lanes.

---

## Per-stage micro-benches

Criterion `simd_batch_phases` group. Each bench drives one phase of `simd_batch::compute_lanes` on a pre-filled scratch (so no warmup work shows up in the timing). Single-thread, cache-warm — these are *relative* attribution numbers, not absolute end-to-end. See ["Translating phase numbers"](#translating-phase-numbers-to-end-to-end) below.

```bash
cargo bench -p yahtzee-core --features simd -- "simd_batch_phases"
```

The production pipeline is now `phase_entry_actions_per_count` + 2× `phase_fused_keeper_round` + `phase_final_dot`. The earlier `phase_entry_actions_*` variants and dense `phase_gemv` / `phase_masked_max` are all kept as bench-only baselines so each successive win stays measurable on every CI run.

Current measurements (post-R7, target-cpu=native):

| Phase | Time per 8-state batch | Calls per batch | Subtotal | % of pipeline |
|---|---:|---:|---:|---:|
| `entry_actions_per_count` (steps 1+2, both buckets) | 2.17 µs | 1 | 2.17 µs | **29%** |
| `entry_actions_per_count_not_yahtzee_elig` (R6.5 fast bucket) | 1.90 µs | — | — | — |
| `fused_keeper_round` (sparse, steps 3+4 / 5+6) | 2.60 µs | 2 | 5.20 µs | **70%** |
| `final_dot` (step 7) | 0.08 µs | 1 | 0.08 µs | 1% |
| **pipeline total** | | | **~7.45 µs** | 100% |
| | | | | |
| `entry_actions_hoisted` (R4, bench-only) | 5.35 µs | — | — | — |
| `entry_actions_vectorized` (R2, bench-only) | 8.01 µs | — | — | — |
| `entry_actions_precomputed` (R1, bench-only) | 76.9 µs | — | — | — |
| `entry_actions_fuse` (R0, bench-only) | 83.0 µs | — | — | — |
| dense `gemv` (bench-only) | 36.6 µs | — | — | — |
| dense `masked_max` (bench-only) | 40.2 µs | — | — | — |

Sparse fused vs dense: `phase_fused_keeper_round` at 2.60 µs replaces the dense `gemv` (36.6) + `masked_max` (40.2) = 76.8 µs pair → **30× speedup on this stage** (the dense pair benefits more from AVX-512 / FMA than the sparse path does, since the dense path's hot loop is just FMA throughput; the sparse path's loop is mostly load-modify-store bound on the scatter-max pass and gains less from wider SIMD).

R5 vs R4 entry_actions (post-target-cpu, current measurements): 2.17 vs 5.35 µs → **2.46× speedup on this stage**. The win is per-score-class precomputation in the slow arm: upper actions produce only 6 distinct child states per parent (indexed by `count(face, d)`), and YAHTZEE produces only 2 (indexed by `is_yahtzee(d)`). Slow-arm gathers per batch drop from 14,112 (R4) to ~304 (R5) — 46× fewer gathers in the slow arm, 40× total batch gathers.

R4 vs R2 entry_actions (post-target-cpu): 5.35 vs 8.01 µs → **1.50× speedup on this stage**. The win is hoisting `state_scores[child_idx]` out of the d-loop for the 6 of 13 actions (3oak / 4oak / FH / SS / LS / chance) where `child` is d-independent.

R2 vs R1 entry_actions (post-target-cpu): 8.01 vs 76.9 µs → **9.6× speedup**. (Pre-target-cpu was 5.7×; AVX-512 + FMA disproportionately helps R2's SIMD pass.)

R6.5 bucket fast-bucket entry_actions: 1.90 µs vs both-buckets-mixed 2.17 µs — **1.14× saving on the heavier bucket alone**, but ~52–95% of states across levels hit it, so the bucket-sorted average lands somewhere in between (closer to 1.90 µs for late-game levels).

End-to-end pipeline (post-target-cpu): R0 dense GEMVs ≈ 162 µs → R0 + sparse fused ≈ 25 µs → R2 ≈ 13 µs → R4 ≈ 11 µs → R5 ≈ 7.8 µs → **R7 (= R5 phases + bucket-sort) ≈ 7.45 µs**. **22× pipeline reduction over the dense baseline.**

`fused_keeper_round` is now **70% of the pipeline** (up from 67% pre-target-cpu) — `entry_actions` has shrunk so much that the keeper-round stage dominates 2.4:1. To move the e2e number further, AVX-512 widening (state batch 8 → 16 via `wide::f32x16`) is the obvious next lever: the keeper-round Pass 1 is FMA-throughput-bound on Zen 5's full-512-bit datapath, so a 2× wide pipeline plausibly halves Pass 1's contribution. The handoff specifically called out multi-accumulator unrolling as a *don't-do* (regressed -7% in testing — Zen 5's OoO already overlaps cross-row accumulator chains via the renamer); widening avoids that pitfall.

### What this says about the *next* round of optimizations

The pipeline is now ~29% `entry_actions_per_count`, ~70% `fused_keeper_round`, ~1% `final_dot`. **`fused_keeper_round` is the bottleneck and AVX-512 widening is the next big lever.**

- **AVX-512 via `wide::f32x16`.** `wide` 1.3.0 ships `f32x16` and `i32x16` that lower to native `__m512` ops + `fused_mul_add_m512` when `target_feature="avx512f","fma"` are set (both already on via `.cargo/config.toml target-cpu=native` on Zen 5). The earlier handoff predicted "below 70 ms probably needs AVX-512 hardware or a different problem decomposition" — we have the hardware *and* the type. Estimated 30–50% on `phase_fused_keeper_round` (FMA-throughput-bound), less on `phase_entry_actions_per_count` (which is gather-bound and gathers don't scale with SIMD width on Zen 5). E2e estimate: ~25% off the 101 ms wall, landing in the **70–80 ms range**.
- **Cache locality is *not* the bottleneck.** Cachegrind on the current build shows D1 miss rate 0.1%, LLd 0.0%. The handoff between R6.5 and R7 predicted a state-encoding restructure (~20–40% e2e) based on action CHANCE's 2 MiB child-idx offset — that diagnostic was wrong, because consecutive states in a batch share `entries` so the 8 gather addresses cluster within 128 idx of each other (~1 cache line), and consecutive batches share most cache lines too. Don't re-attempt encoding restructure unless a future profile actually shows a high miss rate.
- **Parallel scaling is excellent up to physical core count.** Sweet spot is 8–12 threads. SMT contributes negative on heavy levels (16T is reliably worse than 12T). R7 didn't change scaling characteristics — BFS is now small enough that its parallel-scaling profile barely affects the e2e curve.
- **Don't-do list from the R6.5→R7 handoff.** Re-iterating for the next session: don't multi-acc on `phase_fused_keeper_round` (regressed -7%; Zen 5's renamer already overlaps cross-row chains); don't use `mul_add` explicitly for FMA fusion on serial accumulators (the 4-cycle FMA dep chain loses to 3-cycle add-only on Zen 5); don't bucket-sort on `upper_complete` (fires <2% of states at heavy levels, can't amortize bucket-sort overhead).

### Translating phase numbers to end-to-end

R7 cache-warm pipeline ≈ 7.45 µs/batch. Per-level `compute_level` total is 75 ms across 16 threads; 67,056 batches × 7.45 µs / 16 threads = **31 ms theoretical floor**. The 2.4× gap (31 → 75 ms) is the cold-cache penalty (similar ratio to R5): production batches see different states with cold child_idx values. Cache-warm benches load the same 8 states many times and never miss after the first iteration; cache-cold production refills L1 per batch.

Plus the 29 ms BFS = 104 ms total, which matches the criterion wall (101 ms ±2). Standalone `time_build` is 107-110 ms because fresh process invocations don't get criterion's cache warmup, and the BFS bitmap allocation also doesn't amortize across iterations.

The remaining headroom on this hardware: at 12 threads × 7.45 µs / batch × 67k batches = 41 ms theoretical floor, vs current 75 ms compute_level + 29 ms BFS = 104 ms. AVX-512 widening would halve `fused_keeper_round` (the FMA-throughput-bound stage) → pipeline drops from 7.45 to ~4.85 µs → compute_level drops from 75 to ~49 ms → wall drops to ~78 ms. Below that, either a fundamentally different DP / problem decomposition, or explicit prefetch on the surviving cold-cache gathers.

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

The realized speedup (post-target-cpu=native: 76.8 / 2.60 = ~30×; pre-target-cpu was 26.3×) tracked the density-ratio prediction (26.7×) within noise — the rare case where a perf calculation cashes out at the theoretical limit, with AVX-512 / FMA helping the dense baseline more than the sparse path.

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

**AVX-512 caveat (valgrind 3.24 and older).** `.cargo/config.toml` sets `target-cpu=native`, which on Zen 5 enables AVX-512. The compiler's auto-vectorizer (and `wide::f32x16`, post-R8 if you've widened) emits EVEX-encoded instructions valgrind doesn't recognize, and the run dies on SIGILL inside startup glibc code (memset, etc.). To run cachegrind / callgrind, rebuild with AVX-512 disabled:

```bash
RUSTFLAGS="-C target-cpu=znver4 -C target-feature=-avx512f,-avx512vl,-avx512bw,-avx512dq,-avx512cd,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512vnni,-avx512bitalg,-avx512vpopcntdq,-avx512bf16,-avx512vp2intersect" \
  cargo build -p yahtzee-core --release --features simd --example time_build
```

The `wide::f32x8` path is unaffected (always 256-bit AVX2). Only AVX-512 codegen elsewhere (memset, scalar code that auto-vectorized at higher width, or `wide::f32x16` once introduced) requires this workaround.

`Scores::new_with(...)` is *not* called by this example — it uses synthetic 4 MiB `state_scores` (zeroed) instead, since the access pattern (which addresses are read) is what cache simulation cares about, not the values. The 8-state batch is constructed by taking the first 8 children of `State::default()` (one different action filled per lane), so the per-batch cache footprint of `state_scores[child_idx]` reads is realistic for production-shaped batches.

Cache+branch simulation slows the run ~150-200× — default iter counts (set via env: `ITERS_ENTRY_ACTIONS=2000`, `ITERS_FUSED=60000`, `ITERS_DENSE_GEMV=5000`, `ITERS_DENSE_MASKED_MAX=5000`) keep the wall-clock under 4 minutes. Drop them another 5× for a ~30s instructions-only run by passing `--cache-sim=no --branch-sim=no` to valgrind.

Release builds carry `debug = "line-tables-only"` (set in the workspace `Cargo.toml`) so `callgrind_annotate --auto=yes` produces source-line attribution. Release binary size impact: tens of KB of `.debug_line`; strips fine.
