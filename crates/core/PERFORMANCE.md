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
| `simd_batch` | **152 ms** | **58.0×** | **Best end-to-end.** Outer-loop SIMD across 8 states + sparse-fused GEMV/masked_max + flat-array `DICE_AND_ENTRY_SCORES` + `#[inline]` on the per-state DP fns + precomputed `(score + state_scores[child])` table + vectorized `score_and_child` across the 8 lanes + parallel BFS reachability filter + d-independent gather hoist for 6 of the 13 actions + per-score-class precomputation for the slow arm (6 distinct child states per upper action, 2 per YAHTZEE). CLI default. |
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
| + parallel BFS reachability filter | 229 ms | 1.52× |
| + hoist d-independent gather (6 of 13 actions) | 193 ms | 1.19× |
| + per-score-class precompute in slow arm (R5) | **152 ms** | 1.27× |

The current row is **15.1× faster than the dense `simd_batch` starting point**. The most recent two stages (R4 hoist, R5 per-count) share a structural pattern: ask "for which inputs does the gathered `state_scores[child_idx]` actually depend on `dice`?" R4 found that 6 of 13 actions have d-independent children (entire gather hoists out of d-loop); R5 found that the remaining 7 have very *low-cardinality* d-dependence (upper actions: 6 distinct values per parent, indexed by `count(face, d)`; YAHTZEE: 2, indexed by `is_yahtzee(d)`). The R5 entry-actions phase is **2.36× faster** cache-warm (7.69 → 3.26 µs) and **1.27× e2e** vs R4 — total slow-arm gathers per batch dropped from 14,112 to ~304.

The two stages before R4 share a different shape — split the inline-per-iteration work into a Phase A (build a per-batch table of resolved values) and Phase B (branchless SIMD max-reduce over the table). R1 just changed the loop nest of Phase A (locality-preserving `(s outer, a middle, d inner)` so `state_scores[child]` reads stay in L1 for fixed `(s, a)`); R2 flipped it to `(a outer, d middle, s inner-vectorized)` so all 8 lanes' `score_and_child` work runs in one `i32x8`/`f32x8` pass per `(action, dice)` cell. R2 alone was 5.7× on the entry-actions phase in cache-warm benches and 1.26× e2e.

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

External comparison: the `timpalpant/yahtzee` Go reference takes ~45 s for the same table on the same hardware (16 threads). Our naive Rust is 5.1× faster than that, default Rust 7.7×, simd_batch **296×**, CUDA 47×. (Their start-state EV converges to 254.49, ours to 254.5896 — likely a small game-rule difference, not investigated.)

---

## Thread scaling

`simd_batch` scaling on the 9800X3D (8 physical / 16 logical):

| Threads | Wall (ms) | Speedup vs 1T | Efficiency |
|---:|---:|---:|---:|
| 1 | 1282 | 1.00× | 100% |
| 2 | 645 | 1.99× | **99%** |
| 4 | 332 | 3.86× | **97%** |
| 8 | 192 | 6.68× | **84%** |
| 12 | **187** | 6.86× | 57% |
| 16 | 201 | 6.38× | 40% |

Reproduce: `for n in 1 2 4 8 12 16; do RAYON_NUM_THREADS=$n YAHTZEE_BACKEND=simd_batch cargo run -p yahtzee-core --release --features simd --example time_build; done`

**Sweet spot is 8-12 threads.** Going from 8→16 threads adds nothing (and slightly *hurts*: 192 → 201 ms) — the 9800X3D's SMT can't hide behind any execution-unit slack because vectorized SIMD already saturates the FPU. The second SMT thread per core just contests for L1d / L2 / store buffer slots. 12 is interesting: best wall by a few ms, suggesting the OS gets some scheduling flexibility from having more threads than physical cores without engaging full SMT pairing.

The 84% efficiency at 8T is fixed-cost overhead: the post-build `state_scores` write-back (single-threaded), the `into_par_iter` startup on tiny levels (L=12 has 1598 states, ~1.5 µs of work split 8 ways gets dwarfed by rayon dispatch), and the trace prints when enabled. R3's parallel BFS removed the previous ~175 ms sequential ceiling that capped 8T efficiency at 54%; R4's d-independent-gather hoist shrank the parallel work itself, so absolute fixed-cost is unchanged but its share grew slightly (84% vs 86% pre-R4).

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

| Level | States | ndarray | simd | sb (R1) | sb (R2) | sb (R3) | sb (R4) | sb (R5) | cuda |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1,598 | 15.6 | 7.9 | 1.6 | 1.8 | 1.8 | 1.4 | 1.3 | 2.3 |
| 11 | 9,135 | 86.3 | 44.2 | 3.7 | 3.3 | 3.3 | 2.8 | 2.1 | 6.2 |
| 10 | 31,322 | 303.3 | 154.5 | 14.1 | 10.2 | 10.2 | 8.0 | 5.7 | 41.3 |
| 9 | 71,237 | 710.2 | 361.5 | 37.2 | 23.4 | 23.4 | 17.7 | 13.3 | 95.7 |
| **8** | **112,596** | **1145.6** | **585.6** | **67.4** | **36.4** | **36.4** | **28.1** | **19.6** | **150.1** |
| **7** | **126,219** | **1288.3** | **671.7** | **85.0** | **40.2** | **40.2** | **31.7** | **21.5** | **169.4** |
| **6** | **100,619** | **1042.1** | **543.5** | **76.1** | **32.7** | **32.7** | **27.1** | **17.0** | **132.8** |
| 5 | 56,283 | 586.4 | 309.8 | 47.1 | 18.5 | 18.5 | 14.3 | 10.1 | 75.2 |
| 4 | 21,377 | 223.7 | 124.3 | 20.4 | 7.1 | 7.1 | 6.4 | 3.9 | 12.9 |
| 3 | 5,178 | 56.5 | 29.5 | 5.5 | 1.8 | 1.8 | 1.5 | 1.1 | 3.4 |
| 2 | 711 | 7.9 | 4.6 | 1.3 | 0.6 | 0.6 | 0.6 | 0.6 | 0.4 |
| 1 | 44 | 0.5 | 0.3 | 0.4 | 0.3 | 0.3 | 0.3 | 0.3 | 0.1 |
| 0 | 1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 4.7* |
| total compute_level | | 5466 | 2837 | 359 | 176 | 176 | 140 | **96** | 694 |
| BFS (`set_valid_states`) | | 175 | 175 | 175 | 175 | 53 | 58 | 56 | 175 |
| **wall (criterion, warm)** | | — | — | **439** | **349** | **229** | **193** | **152** | **951** |

`*` CUDA's L=0 outlier is kernel-launch overhead on a 1-state batch — irrelevant in absolute terms.

The bolded levels (L=6, 7, 8) are 60% of `compute_level` time on every backend. **simd_batch (R5) beats CUDA at every level from L=4 upward** (R4 only beat it from L=5); CUDA now only wins at L=2 and below where its kernel-launch latency amortizes worse and the absolute numbers are sub-ms anyway. Net: simd_batch wins by 799 ms wall.

R5 vs R4 per heavy level: roughly **1.40–1.65×** uniformly (28.1 → 19.6, 31.7 → 21.5, 27.1 → 17.0), matching the cache-warm per-phase improvement of 7.69 → 3.26 µs (2.36×) modulo the cold-cache scaling penalty. Effective ratio in production = √2.36 ≈ 1.54 — typical √-cold-cache pattern, attenuated by the now-67%-pipeline-share of `fused_keeper_round` (unchanged).

R4 vs R3 per heavy level: roughly **1.20–1.27×** uniformly (36.4 → 28.1, 40.2 → 31.7, 32.7 → 27.1), matching the cache-warm per-phase improvement of 12.34 → 7.81 µs (1.58×) modulo the cold-cache scaling penalty. Effective ratio in production = √1.58 ≈ 1.26 — same shape we've seen on every CPU round.

R3 vs R2 per-level `compute_level` is identical — Round 3 only changed `set_valid_states`, not the DP work. The 120 ms wall improvement (349 → 229 ms) came entirely from the BFS line: 175 → 53 ms at 16 threads. R4 then took out *another* 36 ms wall on top, this time from the per-level work (compute_level total 176 → 140 ms, plus the small BFS regression 53 → 58 ms is rounding error). R5 then took out *another* 41 ms wall on top of R4, again from the per-level work (140 → 96 ms).

R2 vs R1 per heavy level: roughly **2.0×** uniformly (67.4 → 36.4, 85.0 → 40.2, 76.1 → 32.7). R1 vs prior dense `simd_batch` heavy levels (403.5 / 467.2 / 389.6 ms): 4.0–6.0× per level — sparse fused was the bigger structural win, vectorization the icing, hoist the cherry on top.

---

## Per-stage micro-benches

Criterion `simd_batch_phases` group. Each bench drives one phase of `simd_batch::compute_lanes` on a pre-filled scratch (so no warmup work shows up in the timing). Single-thread, cache-warm — these are *relative* attribution numbers, not absolute end-to-end. See ["Translating phase numbers"](#translating-phase-numbers-to-end-to-end) below.

```bash
cargo bench -p yahtzee-core --features simd -- "simd_batch_phases"
```

The production pipeline is now `phase_entry_actions_per_count` + 2× `phase_fused_keeper_round` + `phase_final_dot`. The earlier `phase_entry_actions_*` variants and dense `phase_gemv` / `phase_masked_max` are all kept as bench-only baselines so each successive win stays measurable on every CI run.

| Phase | Time per 8-state batch | Calls per batch | Subtotal | % of pipeline |
|---|---:|---:|---:|---:|
| `entry_actions_per_count` (steps 1+2) | 3.26 µs | 1 | 3.26 µs | **30.6%** |
| `fused_keeper_round` (sparse, steps 3+4 / 5+6) | 3.56 µs | 2 | 7.11 µs | **66.7%** |
| `final_dot` (step 7) | 0.15 µs | 1 | 0.15 µs | 1.4% |
| **pipeline total** | | | **~10.5 µs** | 100% |
| | | | | |
| `entry_actions_hoisted` (R4, bench-only) | 7.69 µs | — | — | — |
| `entry_actions_vectorized` (R2, bench-only) | 12.34 µs | — | — | — |
| `entry_actions_precomputed` (R1, bench-only) | 70.5 µs | — | — | — |
| `entry_actions_fuse` (R0, bench-only) | 72.0 µs | — | — | — |
| dense `gemv` (bench-only) | 46.4 µs | — | — | — |
| dense `masked_max` (bench-only) | 68.0 µs | — | — | — |

Sparse fused vs dense: `phase_fused_keeper_round` at 3.56 µs replaces the dense `gemv` (46.4) + `masked_max` (68.0) = 114.4 µs pair → **32× speedup on this stage**, matching the 1/0.0375 = 26.7× density-ratio ceiling plus the savings from never materializing the 462-wide intermediate buffer.

R5 vs R4 entry_actions: 3.26 µs vs 7.69 µs → **2.36× speedup on this stage**. The win is per-score-class precomputation in the slow arm: upper actions produce only 6 distinct child states per parent (indexed by `count(face, d)`), and YAHTZEE produces only 2 (indexed by `is_yahtzee(d)`). Slow-arm gathers per batch drop from 14,112 (R4) to ~304 (R5) — 46× fewer gathers in the slow arm, 40× total batch gathers.

R4 vs R2 entry_actions: 7.69 µs vs 12.34 µs → **1.61× speedup on this stage**. The win is hoisting `state_scores[child_idx]` out of the d-loop for the 6 of 13 actions (3oak / 4oak / FH / SS / LS / chance) where `child` is d-independent — same `entries`, same `upper_score_remaining`, same `yahtzee_eligible` as parent regardless of dice. 46% fewer total gathers per batch.

R2 vs R1 entry_actions: 12.34 µs vs 70.5 µs → **5.7× speedup**. All 8 lanes' `score_and_child` in one SIMD pass via `i32x8` / `f32x8` / mask-blend.

R1 vs R0 entry_actions: 70.5 µs vs 72.0 µs → only ~2% cache-warm but the production e2e win was -9% wall, because R1's loop-nest reorder made `state_scores[child]` reads localize for the cache prefetcher under cold-cache conditions the bench can't reproduce.

End-to-end pipeline: 297 µs (R0 dense GEMVs) → 55 µs (R0 + sparse fused) → 19.5 µs (R2) → 15.0 µs (R4) → **10.5 µs (R5)**. **28.3× pipeline reduction over the dense baseline.**

`fused_keeper_round` is now 67% of the pipeline (up from 47% post-R4) — `entry_actions` has shrunk so much that the keeper-round stage dominates again. To move the e2e number further, the FMA-dependency-chain length in `phase_fused_keeper_round` Pass 1 is the main remaining lever (mean row length 9.5 nz, ~38 cycles per row of serial FMA chain — see R7 in the next section).

### What this says about the *next* round of optimizations

The pipeline is now ~31% `entry_actions_per_count` (3.26 µs), ~67% `fused_keeper_round` (3.56 µs × 2 = 7.11 µs), ~1% `final_dot`. **`fused_keeper_round` is the new bottleneck** — `entry_actions` shrunk so far past it that the keeper-round stage now dominates 2:1.

- **`fused_keeper_round`** is at ~3.5 µs / 14k cycles for 4,368 nonzeros = 3.2 cycles/nz. The naive ceiling is 1 cycle/nz (one FMA pipelined per cycle), so we're 3× over. Where the cycles go: each row's `val` accumulation has a serial-dependence chain through the FMAs (depth ≈ 9.5 nz/row × ~4-cycle FMA latency = 38 cycles per row × 462 rows would be 17.5k cycles; we get 14k because rows pipeline). Multiple-accumulator unrolling is the obvious next step (R7 in the queue): break the dep chain with 2-4 parallel accumulators per row, expect 1.2-1.5× on Pass 1, ~10% pipeline.
- **`entry_actions_per_count`** at 3.26 µs is fast but still has structure to exploit. The slow arm now does ~304 gathers per batch (down from 14,112 in R4); the fast arm does ~48. Most of the remaining 3.26 µs is per-d compute (score blend, joker rule, yahtzee bonus, valid-mask multiply) across 13 actions × 252 d × 8 lanes — pure SIMD work. Per-batch invariant specialization is the next structural lever: when all 8 lanes share `upper_complete = true`, upper actions become d-independent (`new_upper` stays at 0), so all 6 upper actions collapse from slow arm into fast arm — total batch gathers drop to ~64. When `!yahtzee_eligible` for any lane, the +100 bonus blend dies everywhere. Bucket-sort states within each level by these invariants and dispatch to const-generic specializations of `compute_lanes`. Estimated 10-15% e2e from R6 (`upper_complete` axis) and another 5% from R6.5 (`yahtzee_eligible` axis).
- **Parallel scaling is excellent up to physical core count.** R5 didn't change the BFS but did shrink per-level work, slightly *worsening* the fixed-cost ratio. Still very good. SMT contributes negative — 8T and 12T are within noise but 16T is reliably worse. **12 threads is the new sweet spot.**
- **The big "make it faster" levers are now structural-specialization (R6/R6.5) and pipeline-balance (R7).** Best apparent moves in priority order: (a) `upper_complete` axis specialization — collapses 6 of 13 actions from slow arm to fast arm when the predicate fires (frequent late-game). (b) Multi-accumulator in `phase_fused_keeper_round` Pass 1 — breaks the FMA dep chain. (c) Pin threads to physical cores and skip SMT. After R6+R6.5+R7, expected wall is ~120-130 ms; below that probably needs AVX-512 hardware or a different problem decomposition entirely.

### Translating phase numbers to end-to-end

R5 cache-warm pipeline = 10.5 µs/batch. Per-level `compute_level` total is 96 ms across 16 threads; 67,056 batches × 10.5 µs / 16 threads = 44 ms theoretical. The 2.2× gap is the cold-cache penalty (same ratio as R4): production batches see different states with cache-cold child_idx values, so the surviving gathers actually miss L1. Cache-warm benches load the same 8 states many times and never miss after the first iteration; cache-cold production has to refill L1 per batch. The R5 per-count precompute reduced the *number* of slow-arm gathers (14k → ~300) so the cold-cache work shrunk proportionally — but the ratio between cold and warm stayed roughly fixed at ~2.2× because the structure of the remaining work (sparse-fused keeper round) didn't change.

Plus the 56 ms BFS = 152 ms total, which matches the criterion wall (152 ms). Standalone is 159-165 ms because fresh process invocations don't get criterion's cache warmup.

The remaining headroom on this hardware is shrinking. 12 threads × 10.5 µs / batch × 67k batches = 59 ms theoretical floor, vs current 152 ms. To close the 2.6× cold-cache gap we'd need either smaller working set (fewer cache misses on `state_scores`) or explicit prefetch. With R6/R6.5 collapsing more upper actions to the fast arm and R7 breaking FMA dep chains, expected post-R7 wall is ~120-130 ms — within ~2× of theoretical floor.

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
