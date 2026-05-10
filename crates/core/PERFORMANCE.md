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
| `simd_batch` | **78.9 ms** | **98.1×** | **Best end-to-end.** Outer-loop SIMD across 16 states (AVX-512 ZMM via `wide::f32x16`) + sparse-fused GEMV/masked_max + flat-array `DICE_AND_ENTRY_SCORES` + `#[inline]` on the per-state DP fns + precomputed `(score + state_scores[child])` table + vectorized `score_and_child` across the 16 lanes + parallel BFS reachability filter + d-independent gather hoist for 6 of the 13 actions + per-score-class precomputation for the slow arm (6 distinct child states per upper action, 2 per YAHTZEE) + bucket-sort by `!yahtzee_eligible` (R6.5) + LazyLock hoist out of BFS (R7) + AVX-512 widening (R8). CLI default. |
| `cuda` | 712 ms | 10.87× | Per-level batched on GPU; cuBLAS sgemm + 3 NVRTC kernels. |

`simd_batch` overtook CUDA after the sparse fused-keeper-round landed and has pulled progressively further ahead — currently **9.0× faster than CUDA** end-to-end. Walking the arc:

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
| + LazyLock hoist + d-independent action collapse in BFS (R7) | 101 ms | 1.18× |
| + AVX-512 widening via `wide::f32x16` (R8) | **78.9 ms** | 1.28× |

The current row is **29.0× faster than the dense `simd_batch` starting point**. R7 (the most recent landed round) is structurally different from R0..R6.5: every prior round optimized the *DP per-level compute_level*, but R7 optimizes the *BFS reachability filter* (`Scores::set_valid_states`). After R7, BFS is ~28% of e2e wall (was ~40% pre-R7), and the DP work — specifically `phase_fused_keeper_round` — is back to dominating. The handoff between R6.5 and R7 predicted a state-encoding restructure (different bijection from `State` to `usize`) as the next big win, estimating 20-40% e2e. Cachegrind disproved that hypothesis: D1 miss rate is 0.1% and LLd is 0.0%, so cache locality isn't the bottleneck. The real cost was ~21% of total Ir going to `sync/once.rs` + `sync/atomic.rs` — the BFS hot loop's per-iteration LazyLock atomic checks for `DICE_AND_ENTRY_SCORES` and `YAHTZEE_DICE`. The per-stage micro-benches below didn't catch this because they only bench `phase_*` functions, not `set_valid_states`.

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

External comparison: the `timpalpant/yahtzee` Go reference takes ~45 s for the same table on the same hardware (16 threads). Our naive Rust is 5.8× faster than that, default Rust 10.9×, simd_batch **570×**, CUDA 63×. (Their start-state EV converges to 254.49, ours to 254.5896 — likely a small game-rule difference, not investigated.)

---

## Thread scaling

`simd_batch` scaling on the 9800X3D (8 physical / 16 logical), post-R8 (AVX-512 LANES=16) + target-cpu=native, single `time_build` sample per thread count:

| Threads | Wall (ms) | Speedup vs 1T | Efficiency |
|---:|---:|---:|---:|
| 1 | 373 | 1.00× | 100% |
| 2 | 200 | 1.87× | **93%** |
| 4 | 112 | 3.33× | **83%** |
| 8 | **78** | 4.78× | 60% |
| 12 | 80 | 4.66× | 39% |
| 16 | 94 | 3.97× | 25% |

(Pre-R8 reference, LANES=8: 1T = 562 ms, 12T sweet-spot = 95 ms, 16T = 110 ms. R8 dropped 1T by 34% and 8T by 29%.)

Reproduce: `for n in 1 2 4 8 12 16; do RAYON_NUM_THREADS=$n YAHTZEE_BACKEND=simd_batch cargo run -p yahtzee-core --release --features simd --example time_build; done`

**Sweet spot is now 8 threads** (was 12 pre-R8). The sweet spot moved DOWN because per-batch work doubled in R8 (16 states per batch instead of 8), so each level produces half as many batches and the rayon work-stealing dispatcher has less to keep extra threads busy. Going 8→16 still regresses (78 → 94 ms) — same SMT-contention story as before, just at a lower thread-count threshold.

The 60% efficiency at 8T (compared to 64% pre-R8) is fixed-cost overhead's growing relative share: the parallel work shrunk again, and the absolute fixed costs (BFS reduce-OR merge between levels, scratch zero-init, rayon dispatch, post-build single-threaded write-back) stay constant. R3's parallel BFS removed the previous ~175 ms sequential ceiling that capped 8T efficiency at 54%; R7 then halved the BFS wall itself (56 ms → 29 ms at 16T); R8 didn't touch BFS. The remaining Amdahl ceiling is now spread thin across many small sequential costs rather than concentrated in one phase.

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

Current measurements (post-R8, target-cpu=native, 16T):

| Level | States | ndarray | simd | sb (R5, pre-target-cpu) | sb (R7) | sb (R8) | cuda |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1,598 | 12.0 | 13.8 | 1.3 | 1.6 | 2.5 | 2.0 |
| 11 | 9,135 | 65.9 | 44.6 | 2.1 | 1.8 | 1.7 | 5.7 |
| 10 | 31,322 | 232.1 | 153.7 | 5.7 | 4.8 | 3.8 | 30.0 |
| 9 | 71,237 | 540.4 | 391.4 | 13.3 | 9.6 | 7.9 | 69.7 |
| **8** | **112,596** | **861.3** | **564.9** | **19.6** | **14.5** | **11.2** | **109.6** |
| **7** | **126,219** | **991.1** | **674.3** | **21.5** | **16.3** | **11.4** | **123.8** |
| **6** | **100,619** | **781.9** | **502.7** | **17.0** | **12.9** | **9.0** | **97.0** |
| 5 | 56,283 | 442.1 | 300.8 | 10.1 | 7.6 | 5.4 | 55.0 |
| 4 | 21,377 | 167.8 | 123.0 | 3.9 | 3.3 | 3.6 | 9.4 |
| 3 | 5,178 | 41.2 | 33.3 | 1.1 | 1.3 | 1.1 | 2.5 |
| 2 | 711 | 5.7 | 4.8 | 0.6 | 0.6 | 1.5 | 0.3 |
| 1 | 44 | 0.4 | 0.8 | 0.3 | 0.3 | 0.7 | 0.1 |
| 0 | 1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 4.7* |
| total compute_level | | 4142 | 2808 | 96 | 75 | 60 | 510 |
| BFS (`set_valid_states`) | | 29 | 65** | 56 | 29 | 29 | 29 |
| **wall (criterion, warm)** | | **4127 ms** | **2566 ms** | **152 ms** | **101 ms** | **78.9 ms** | **712 ms** |

`*` CUDA's L=0 outlier is kernel-launch overhead on a 1-state batch — irrelevant in absolute terms.
`**` `simd`'s BFS is noisier in single-sample traces because target-cpu=native makes the DP fast enough that scheduler jitter at lower-thread-count workloads shows up as BFS variance. The BFS line is identical regardless of which backend drives `compute_level` (BFS runs *before* level 12) — quoted as 29 ms across backends for clarity.

The bolded levels (L=6, 7, 8) are ~52% of `compute_level` time on R8 (was ~58% pre-R8 — heavy levels shrunk slightly more than light levels). **simd_batch (R8) beats CUDA at every level from L=3 upward**; CUDA only wins at L=2 and below where its kernel-launch latency amortizes worse and the absolute numbers are sub-ms anyway. Net: simd_batch wins by 633 ms wall vs CUDA (9.0× faster).

R8 vs R7 per heavy level: roughly **1.25–1.43×** uniformly (14.5 → 11.2, 16.3 → 11.4, 12.9 → 9.0). R8 widened the SIMD batch from 8 states (f32x8 / AVX2) to 16 states (f32x16 / AVX-512 ZMM). Per-state work in the FMA-bound `phase_fused_keeper_round` drops ~41% because `vmulps zmm` + `vaddps zmm` directly double throughput vs `vmulps ymm` + `vaddps ymm`; per-state work in the gather-bound `phase_entry_actions_per_count` drops ~31% (gathers don't widen — each scalar load is still one cycle — but the per-batch SIMD math around them does). The small levels (L≤4) are noise-level: per-batch overhead amortizes slightly worse with 16-state batches at small populations (L=12 has 1,598 states → 100 batches at LANES=16 instead of 200 at LANES=8; tail handling matters more relatively).

R7 vs R5 (with target-cpu=native applied between, then R6.5 between again): per heavy level roughly **1.30–1.35×** uniformly. R7 itself didn't touch the DP (only `set_valid_states`); these per-level gains came from `target-cpu=native` enabling AVX2 + FMA + BMI2 in `wide::f32x8`'s lowering (previously SSE2-only fallback). R7's own win is in the BFS line: 56 ms → 29 ms at 16T (−48%) — a ~21 ms wall reduction that the per-level table can't show because BFS happens *outside* `compute_level`. Combined R5 → R8 wall: 152 → 78.9 ms (-48%), split roughly 1/4 from R6.5 + target-cpu + R7's BFS hoist, 1/4 from target-cpu's effect on the unchanged DP phases at f32x8 width, 1/4 from R8's f32x8 → f32x16 widening of the DP phases, and 1/4 from the secondary effects of all three.

Pre-target-cpu progression (numbers in the 5466 → 96 ms range of the old table) — see [Backend sweep "stage" table](#backend-sweep) for the full arc. Brief summary of the per-heavy-level deltas: **R5 vs R4** was 1.40–1.65× from per-score-class precompute (slow-arm gathers dropped 14,112 → 304); **R4 vs R3** was 1.20–1.27× from d-independent gather hoist (46% fewer gathers); **R3 vs R2** was identical per-level (R3 only changed BFS); **R2 vs R1** was 2.0× from vectorizing `score_and_child` across 8 lanes.

---

## Per-stage micro-benches

Criterion `simd_batch_phases` group. Each bench drives one phase of `simd_batch::compute_lanes` on a pre-filled scratch (so no warmup work shows up in the timing). Single-thread, cache-warm — these are *relative* attribution numbers, not absolute end-to-end. See ["Translating phase numbers"](#translating-phase-numbers-to-end-to-end) below.

```bash
cargo bench -p yahtzee-core --features simd -- "simd_batch_phases"
```

The production pipeline is now `phase_entry_actions_per_count` + 2× `phase_fused_keeper_round` + `phase_final_dot`. The earlier `phase_entry_actions_*` variants and dense `phase_gemv` / `phase_masked_max` are all kept as bench-only baselines so each successive win stays measurable on every CI run.

Current measurements (post-R8, target-cpu=native; 16-state batches):

| Phase | Time per 16-state batch | Calls per batch | Subtotal | % of pipeline |
|---|---:|---:|---:|---:|
| `entry_actions_per_count` (steps 1+2, both buckets) | 2.98 µs | 1 | 2.98 µs | **32%** |
| `entry_actions_per_count_not_yahtzee_elig` (R6.5 fast bucket) | 2.76 µs | — | — | — |
| `fused_keeper_round` (sparse, steps 3+4 / 5+6) | 3.06 µs | 2 | 6.11 µs | **67%** |
| `final_dot` (step 7) | 0.08 µs | 1 | 0.08 µs | 1% |
| **pipeline total per batch** | | | **~9.18 µs** | 100% |
| **pipeline total per state** | | | **~0.57 µs** | (was 0.93 µs at LANES=8) |
| | | | | |
| `entry_actions_hoisted` (R4 design, bench-only) | 10.22 µs | — | — | — |
| `entry_actions_vectorized` (R2 design, bench-only) | 15.76 µs | — | — | — |
| `entry_actions_precomputed` (R1 design, bench-only) | 154.5 µs | — | — | — |
| `entry_actions_fuse` (R0 design, bench-only) | 168.9 µs | — | — | — |
| dense `gemv` (bench-only) | 36.8 µs | — | — | — |
| dense `masked_max` (bench-only) | 40.3 µs | — | — | — |

**Reading per-batch vs per-state at LANES=16.** Every per-batch number is for 16 states' worth of work in one batch. To get per-state cost, divide by 16. R8 doubled the per-batch work (we now compute 16 lanes' EVs simultaneously) — but it didn't double the per-batch *time* on most stages, because each AVX-512 `vmulps zmm` / `vaddps zmm` does 2× the work of the AVX2 `vmulps ymm` / `vaddps ymm` it replaces. Effective per-state speedups vs the R7 column in the per-level breakdown:

- `entry_actions_per_count`: 2.98 / 16 = 0.19 µs/state vs R7's 2.17 / 8 = 0.27 µs/state → **-31% per state**. Mixed: the SIMD math doubled in throughput, but the per-batch scalar gathers (304 state_scores reads in the slow arm) doubled in count too. Net moderate win.
- `entry_actions_per_count_not_yahtzee_elig` (R6.5 fast bucket): 2.76 / 16 = 0.17 µs/state vs R7's 1.90 / 8 = 0.24 µs/state → **-28% per state**. Same shape.
- `fused_keeper_round`: 3.06 / 16 = 0.19 µs/state vs R7's 2.60 / 8 = 0.33 µs/state → **-41% per state**. The CSR Pass 1 inner loop is pure `+= splat(p) * in_dice[d]` — the throughput-bound FMA-equivalent doubles cleanly on AVX-512.
- dense `gemv`: 36.8 µs / 16 = 2.30 µs/state vs R7's 36.6 / 8 = 4.58 µs/state → **-50% per state**. Pure dense FMA loop — perfectly halves on the doubled SIMD width. (Not on the production path, but a useful AVX-512 efficiency reference.)
- dense `masked_max`: -50% per state, same shape.
- `final_dot`: unchanged in time per batch (it's small fixed scalar work) → -50% per state effectively, but negligible absolute.

Sparse fused vs dense: `phase_fused_keeper_round` at 3.06 µs replaces the dense `gemv` (36.8) + `masked_max` (40.3) = 77.1 µs pair → **25× speedup on this stage**. Slightly less than the 30× pre-R8 because both sides benefit from AVX-512, but the dense side benefits MORE (perfect 2× scaling) while the sparse-fused side benefits LESS (it has scalar bookkeeping and load-modify-store on the scatter-max pass that don't widen). Still within striking distance of the 1/0.0375 = 26.7× density-ratio ceiling, which the dense baseline is now closer to than the sparse path.

R5 vs R4 entry_actions (post-R8 measurements): 2.98 vs 10.22 µs per-batch → **3.43× speedup on this stage** at LANES=16 (was 2.46× at LANES=8 — the per-score-class precompute eliminates per-batch gather work that scales linearly with LANES, so its absolute win grows with width). Structural insight unchanged: upper actions produce only 6 distinct child states per parent (indexed by `count(face, d)`); YAHTZEE produces only 2 (indexed by `is_yahtzee(d)`); R5 collapses the slow-arm gather count by 40×.

R4 vs R2 entry_actions (post-R8): 10.22 vs 15.76 µs → **1.54× speedup on this stage** (was 1.50× at LANES=8). D-independent gather hoist; 46% fewer gathers per batch.

R2 vs R1 entry_actions (post-R8): 15.76 vs 154.5 µs → **9.8× speedup**. (Pre-target-cpu was 5.7× at LANES=8; AVX-512 + FMA + d-vectorization compound.)

R6.5 bucket fast-bucket: 2.76 µs vs both-buckets-mixed 2.98 µs — **1.08× saving on the fast bucket alone** at LANES=16 (was 1.14× at LANES=8 — the DCE'd yahtzee-bonus blend chain is a smaller relative share of a smaller phase). The cumulative impact across heavy levels is still meaningful because ~52-95% of states hit the fast bucket.

End-to-end pipeline: R0 dense GEMVs ≈ 162 µs → R0 + sparse fused ≈ 25 µs → R2 ≈ 13 µs → R4 ≈ 11 µs → R5 ≈ 7.8 µs → R7 (= R5 + R6.5) ≈ 7.45 µs → **R8 (16-state batch) ≈ 9.18 µs per 16-state batch = 0.57 µs per state**. Per-state pipeline reduction from the dense baseline: R0 ≈ 20.3 µs/state → R8 = 0.57 µs/state → **35.6× per-state pipeline reduction**.

`fused_keeper_round` is now **67% of the pipeline** (down from 70% post-R7) — `entry_actions_per_count` shrunk less than `fused_keeper_round` under the widening, so the pipeline rebalanced slightly back toward parity. AVX-512 widening was the predicted next lever and delivered as expected.

### What this says about the *next* round of optimizations

The pipeline is now ~32% `entry_actions_per_count`, ~67% `fused_keeper_round`, ~1% `final_dot`. **`fused_keeper_round` is still the bottleneck.**

- **What's still on the table for `fused_keeper_round`.** Pass 1 (sparse dot product, 4368 nonzeros) is now 1.53 µs per batch at LANES=16 → 0.10 µs/state, or ~8 cycles per nonzero per state at 5 GHz. The naive ceiling is 1 cycle/nz on the FMA-throughput side, and we're at 8× over that — but the cycles include the CSR walk's scalar bookkeeping (row_cols index loads, splat broadcasts, sum reduction), not just the FMA. Pass 2 (scatter-max RMW into out_dice) is the other 1.53 µs per batch, dominated by load-modify-store latency on `out_dice[d]` which AVX-512 doesn't help (each store still takes a port). The remaining sub-leverage levers: explicit prefetch on `in_dice[col_idx[i+N]]` ahead of N=4 iterations (might pull L2 → L1 just in time), or restructuring Pass 2 to scatter into a temporary first then merge. Neither obviously big.
- **`entry_actions_per_count`** at 2.98 µs / batch = 0.19 µs/state has structural floor in the scalar gather count: 304 gathers per batch for the slow arm, ~48 for the fast arm, can't easily be reduced further without state-space restructuring. AVX-512 won't help the gather throughput specifically (gather ports are width-invariant on Zen 5). Cache miss rate is still 0.1% per cachegrind.
- **State-space restructuring revisited.** With per-state work now at 0.57 µs and 67k batches × 0.57 µs = 38 ms theoretical floor for 16 threads of perfectly-saturated compute, the gap to current 60 ms compute_level is ~1.6× — much tighter than the 2.4× we had at R5. State encoding would only help if it shrank either gather count or the entry_actions_table scratch (105 KiB pre-R8, now 210 KiB at LANES=16 — but still L2-resident per thread). The earlier hypothesis "different bijection cuts cold-cache penalty" was refuted by cachegrind showing D1 = 0.1%; what *might* help instead is a more compact per-batch scratch layout that fits more of the table in L1.
- **Parallel scaling is excellent up to physical core count.** Sweet spot is 12 threads (95 → 89 ms on the f32x8 path; R8 hasn't been measured at all thread counts yet). SMT contributes negative on heavy levels (16T regresses).
- **Don't-do list.** Carried forward: don't multi-acc on `phase_fused_keeper_round` (regressed -7% at LANES=8; Zen 5's renamer already overlaps cross-row chains, and at LANES=16 the renamer has 32 ZMM registers to play with); don't use `mul_add` explicitly to fuse mul+add into FMA on serial accumulators (4-cycle FMA dep chain loses to 3-cycle add-only); don't bucket-sort on `upper_complete` (predicate fires <2% of states); don't re-attempt state-encoding restructure for cache locality (refuted by cachegrind).

### Translating phase numbers to end-to-end

R8 cache-warm pipeline ≈ 9.18 µs/16-state-batch = 0.57 µs/state. Per-level `compute_level` total is 60 ms across 16 threads; 67,056 states × 0.57 µs / 16 threads = **2.4 ms theoretical floor per thread** = 38 ms wall... but that assumes per-thread saturation and zero rayon overhead. The 1.58× gap (38 → 60 ms) is partly the cold-cache penalty (cache-warm phase benches don't see fresh state_scores per batch), partly per-batch setup (state SoA hoists, mask precomputes, scratch zero-init).

Plus the 29 ms BFS = 89 ms total. Criterion measures the build at 78.9 ms ±0.2 — slightly under the trace-sum estimate because criterion warms up across iterations, amortizing some of the BFS bitmap allocation cost.

The remaining headroom on this hardware: theoretical floor of 38 ms compute_level + 29 ms BFS = 67 ms vs current 78.9 ms wall = 1.18× away. Closing the cold-cache gap further requires either explicit prefetch on the surviving gathers or a more compact working set. Below 67 ms probably needs a fundamentally different DP / problem decomposition (e.g., hierarchical state factorization, see the R6.5→R7 handoff notes).

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

The realized speedup (post-R8 LANES=16: 77.1 / 3.06 = ~25×; post-R7 LANES=8: ~30×; pre-target-cpu was 26.3×) tracks the density-ratio prediction (26.7×) within noise — the rare case where a perf calculation cashes out at the theoretical limit. The dense baseline ((g)emv + masked_max) scales perfectly with SIMD width (-50% per state going 8→16 lanes), while the sparse fused path has scalar bookkeeping and scatter-max load-modify-store overhead that doesn't widen — so the dense baseline closed some of the gap with R8 widening, dropping the ratio from 30× back to 25×.

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
