# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

Cargo workspace (`Cargo.toml` lists members; the root is not a crate):

- `crates/core` — `yahtzee-core` library. The DP solver, state/dice types, precomputed tables, and per-state `ExpectedValues` API.
- `crates/cli` — `yahtzee-cli` binary. Thin CLI wrapper for generating/querying scores (`clap`-based).
- `crates/wasm` — `yahtzee-wasm` crate. Wraps the core in a `wasm-bindgen` interface; built with `wasm-pack` to `crates/wasm/pkg/`.
- `web/` — Svelte + Vite frontend that imports `yahtzee-wasm` as a linked local package.

## Commands

- Build: `cargo build` (release: `cargo build --release`).
- Test: `cargo test` — `Cargo.toml` sets `[profile.test] opt-level = 3` because `Scores::new()` is used in tests and is minutes-to-hours slow at `opt-level = 0`.
- Run a single test: `cargo test -p yahtzee-core test_expected_value -- --nocapture`.
- CLI: `cargo run -p yahtzee-cli --release -- --help`. Subcommands are `solve`, `value`, `play`, and `build`. The CLI binary embeds a brotli-compressed score table at compile time (canonical artifact: `crates/cli/data/scores.bin.br`, ~1 MiB, tracked in git), so `solve` / `value` / `play` work out of the box with no `--scores` flag. `--scores PATH` overrides the embed with a raw bincode file from disk — useful when iterating on `crates/core` without rebuilding the CLI. `play` is an interactive game loop; pass `--auto --seed N` for a deterministic solver-vs-itself game (used by `tests/play_auto.rs`), `--manual-dice` for a real-game coach. `build` regenerates the table; the web bundle uses `build --output crates/cli/data/scores.bin --brotli`, which also writes the `.br` and a `MANIFEST` and updates the embed source for the next CLI rebuild.
- Rebuild wasm after touching `crates/core` or `crates/wasm`: `cd crates/wasm && wasm-pack build --target web --out-dir pkg`. The Svelte dev server picks up the new `pkg/` automatically.
- Web dev: `cd web && npx vite` (or `pnpm run dev` if pnpm is available). Type check: `npx svelte-check --tsconfig ./tsconfig.json`. Production build: `npx vite build`.
- Time `Scores::new()` ad-hoc (rayon scaling, perf experiments): `RAYON_NUM_THREADS=N cargo run -p yahtzee-core --release --example time_build` (default ndarray backend), or `... --example time_build_naive` (naive scalar backend), or `... --example cuda_smoke --features cuda` (GPU). All print one-line timings; `cuda_smoke` also asserts the default-state EV.
- Per-level DP-build trace: set `YAHTZEE_TRACE_LEVELS=1` on any of the above to print `level=N batch=B collect=…ms compute=…ms write=…ms` for each of the 13 DP levels. Useful when comparing backends at finer-than-total-wall-time granularity.
- Criterion benches: `cargo bench -p yahtzee-core --bench recommend [--features cuda,faer,simd] -- "Scores::new_with|state_value/backends"` runs the head-to-head across naive / ndarray / faer / simd / cuda. The full-build group has `sample_size=10` and `measurement_time=60s`, so it can run for ~5 min with all features enabled — that's expected. Other bench groups (`Scores::values`, `recommend(default,[1..5])`, `*_with_turn_ev`) are quick.

The web app expects `scores.bin.br` (brotli-compressed `Scores`) at `crates/cli/data/scores.bin.br` — the same file the CLI embeds. The dev-server middleware in `web/vite.config.ts` reads from there and streams it at `/scores.bin` with `Content-Encoding: br`. There is no longer a separate `web/static/scores.bin.br`.

## Core architecture

The solver uses backward-induction dynamic programming over scorecard states.

### State space and indexing

A `State` is `(entries: EntryAction bitflags, yahtzee_bonus_eligible: bool, upper_score_remaining: u8)`. States are packed into a `usize` via `From<State>` / `From<usize>`, giving `NUM_STATES = 2^13 * 2 * 64 = 1_048_576` slots. Only `NUM_VALID_STATES = 536_448` are reachable; `Scores::set_valid_states` computes reachability by BFS from the default state.

`EntryAction` is a `bitflags` type where each of the 13 scoring categories is a single bit; `ENTRY_ACTIONS` is the canonical ordered array. `DiceCounts([u8; 6])` represents a multiset of dice; there are `NUM_DICE_COMBINATIONS = 252` full 5-dice combinations and `NUM_KEEPERS = 462` possible kept subsets (0..=5 dice).

### Precomputed tables (`mod math`, wrapped in `lazy_static!`)

- `DICE_AND_ENTRY_SCORES` — `Array2<u8>` of raw score for each `(entry, dice_idx)`. Does *not* apply the joker rule (that's layered on in `State::score_and_child` / `State::entry_score`).
- `DICE_TO_ALLOWED_KEEPERS` — `Array2<f32>` indicator matrix: which keeper subsets can be chosen from each full dice roll.
- `KEEPERS_TO_DICE_PROBABILITIES` — `Array2<f32>` transition probabilities from a kept subset to each resulting full roll after re-rolling.
- `YAHTZEE_DICE` — maps dice index to `Some(upper EntryAction)` iff all 5 dice are equal (used for the joker rule and Yahtzee bonus eligibility).

### Solver (`Scores`)

`Scores::new()` allocates `state_scores: Array1<f32>` of length `NUM_STATES` and fills it level-by-level from 13 filled entries down to 0 (`set_scores`). Within a level all states are independent (a state only reads entries of `state_scores` at strictly higher levels), and that's the parallelism dispatched through the [`BuildBackend`](#linalg-and-build-backends) trait — by default `CpuBuildBackend` does it as a rayon `par_iter` over the per-state walk; the GPU backend does the whole batch in one custom-kernel + cuBLAS pipeline per level.

`Scores::new_with<B: BuildBackend>(backend: &B) -> Result<Self, B::Error>` is the generic constructor; `Scores::new()` calls it with `&CpuBuildBackend` and unwraps the `Infallible` result. Use `new_with` directly to opt into a non-default backend (e.g. `&CudaBuildBackend::new()?` under the `cuda` feature).

`Scores::new_with_unvalidated<B>` is the sibling that skips the BFS-reachability filter in the per-level batch enumeration: every structurally possible state at each level enters the DP, including the ~512k that no real game reaches. ~1.9× build wall-clock vs `new_with`, uniformly across all backends (naive, ndarray, faer, simd, simd_batch). The original hypothesis — that the regular `C(13,L) × 128` batch shape would unlock outer-loop SIMD wins beyond what semi-regular validated batches give — turned out empirically false: `simd_batch` already extracts most of the outer-loop win on validated batches (peak L=7 = 126k states is plenty for `par_chunks(8)` saturation), and the unvalidated path is ~1.88× slower than validated for `simd_batch` too. So validation pays its way and stays the default.

The unvalidated constructor is kept around for (a) BFS soundness — `test_unvalidated_matches_validated` (gated `#[ignore]`) cross-checks that the validated and unvalidated tables agree on every reachable state's EV (today: 0 mismatches across 536,448 reachable states), and (b) future experiments. The first round (outer-loop SIMD) didn't recover the 1.88× cost, but other directions could — different state encodings that change reachability density, alternate DP orderings, GPU kernels that benefit from the regular `C(13,L)×128` shape in ways the CPU SIMD path didn't. The path is opt-in and adds no cost to the default builds, so it stays until either a follow-on experiment lands a win or we conclude the design space has been exhausted.

`Scores::values(state)` returns an `ExpectedValues` by walking the three-roll decision tree in reverse, dispatching the linalg ops through a [`LinalgBackend`](#linalg-and-build-backends):

1. `entry_actions[action, dice]` = immediate score of taking `action` on `dice` + stored EV of the resulting child state (via `State::score_and_child`, which handles upper bonus, Yahtzee bonus, and the joker rule).
2. `third_dice[dice] = max_action entry_actions[action, dice]` (best entry to pick on the final roll).
3. Alternate keepers and dice: `second_keepers = KEEPERS_TO_DICE_PROBABILITIES · third_dice`; `second_dice[d] = max over allowed keepers of DICE_TO_ALLOWED_KEEPERS[d] * second_keepers`; repeat to get `first_keepers` and `first_dice`.
4. Overall EV = initial-roll distribution (row 0 of `KEEPERS_TO_DICE_PROBABILITIES`) · `first_dice`.

The expected value from the default state is ~254.5896 (asserted in `test_expected_value`).

Both per-state primitives also exist as **free functions** in the crate root: `pub fn state_value_with(state, state_scores: &[f32], &impl LinalgBackend) -> f32` and `pub fn values_with(...)`. The `Scores::*_with` methods are 1-line wrappers; the free-function form is what `BuildBackend` impls call so they don't need a `Scores` value.

### Linalg and build backends

There are **two** swappable abstractions, at different granularities:

`LinalgBackend` — *per-state* primitives. Three ops: `keepers_from_dice` (252→462 GEMV), `dice_from_keepers` (462→252 masked-max reduction), `initial_roll_ev` (252-dim dot). Implementations live in `crates/core/src/linalg.rs`:

| Backend | Cargo feature | What it does |
|---|---|---|
| `NaiveBackend` | (always-on, no deps) | Reference impl: nested scalar `for` loops over `&[f32]`. No ndarray ops. Independent of the others' code paths, so it doubles as the cross-check oracle (`test_naive_backend_matches`). |
| `NdarrayBackend` | (default) | GEMV via `ndarray::Array2::dot` (which is `matrixmultiply` by default; `cblas_sgemv` under `--features blas`). Masked-max via a hand-rolled `Zip`. **What `Scores::new()` and `Scores::values()` use by default.** |
| `FaerBackend` | `--features faer` | GEMV via faer's `Mat * Col`. Holds a preprocessed `faer::Mat` of `KEEPERS_TO_DICE_PROBABILITIES`. Reuses ndarray's masked-max (faer can't help with that). |
| `SimdBackend` | `--features simd` | Hand-vectorized masked-max via `wide::f32x8`. GEMV stays on ndarray's `.dot()`. The masked-max acceleration is the single biggest CPU win — see perf table below. |

`BuildBackend` — *per-level* batched. One method, `compute_level(states: &[State], state_scores: &[f32]) -> Result<Vec<f32>, Self::Error>`, with associated error type. CPU impls just dispatch the per-state walk in a rayon `par_iter`; the GPU impl owns the whole batched pipeline (uploads, NVRTC kernels, cuBLAS gemms, scatter back to device-resident state_scores).

| Backend | Cargo feature | What it does |
|---|---|---|
| `CpuBuildBackend` | (always-on) | Convenience unit-struct: rayon `par_iter` over the per-state walk using `NdarrayBackend`. What `Scores::new()` uses. `Error = Infallible`. |
| `CpuBuildBackendWith<L: LinalgBackend>(pub L)` | (always-on) | Generic over the per-state linalg, so benches and external callers can drive a full table build through any `LinalgBackend` (e.g. `CpuBuildBackendWith(NaiveBackend)`, `CpuBuildBackendWith(SimdBackend)`). `Error = Infallible`. |
| `SimdBatchBuildBackend` | `--features simd` | Outer-loop SIMD: vectorizes the per-state pipeline *across* 8 states with `wide::f32x8`, where each lane is a different state and the matrices/probabilities are broadcast scalar across lanes. Lives in `crates/core/src/linalg/simd_batch.rs`. Pairs `par_chunks(8)` for thread-level parallelism with thread-local scratch via `map_init`; tail (<8 states) falls back to the per-state `SimdBackend`. Four structural wins compound here: (a) the two GEMV→masked_max pairs are merged into a single sparse `phase_fused_keeper_round` (walks K2D as CSR, 4,368 nonzeros of 116,424, 3.75% dense, scatter-maxes directly into `out_dice` without materializing the 462-wide intermediate — 26.3× on this stage); (b) a precomputed `(score + state_scores[child])` table per batch (105 KiB scratch) splits the entry-actions phase into a build phase whose loop nest preserves locality on the random `state_scores[child_idx]` gather, and a branchless SIMD max-reduce phase; (c) `phase_entry_actions_vectorized` runs all 8 lanes' `score_and_child` in one SIMD pass per `(action, dice)` cell — `i32x8` for the bit-pack child_idx computation, `f32x8` for score arithmetic and bonuses, mask-blend for the joker rule; (d) `phase_entry_actions_hoisted` then splits that loop into fast/slow arms by whether `child` depends on dice — for the 6 of 13 actions where it doesn't (3oak / 4oak / FH / SS / LS / chance), `state_scores[child_idx]` hoists out of the d-loop, giving 46% fewer gathers per batch (1.58× on this stage). End-to-end 16.2× over `CpuBuildBackendWith(SimdBackend)` on the validated path (3.13 s → 193 ms); the entry-actions phase alone is 9.0× faster hoisted than scalar-per-lane (70.5 µs → 7.81 µs cache-warm). `Error = Infallible`. The parallel BFS reachability filter (`Scores::set_valid_states`) uses rayon fold-with-thread-local-bitmap + reduce-OR-merge to avoid serializing the build at 8+ threads — pre-fix it was 175 ms sequential and ~50% of wall at 16T, post-fix 53 ms at 16T. |
| `CudaBuildBackend` | `--features cuda` | Full GPU pipeline. Lives in `crates/core/src/linalg/cuda.rs`. Uses `cudarc` (driver API + cuBLAS + NVRTC), dynamic-loaded — no `nvcc` or CUDA headers needed at build time, only `libcuda` / `libcublas` / `libnvrtc` `.so`s on the runtime host. Three custom NVRTC kernels (`build_third_dice`, `masked_max`, `initial_roll_ev`) plus two cuBLAS sgemm calls per level for the two `keepers_from_dice` GEMVs. State_scores is allocated **once** on device and updated in place via a `scatter_results` kernel — no per-level H→D upload of the host buffer. Per-level intermediates are persistent and lazily grown to the largest batch seen. `Error = CudaError`. |

**Performance numbers, per-level breakdowns, per-stage micro-benches, and the structural-sparsity facts behind the current architecture live in [`crates/core/PERFORMANCE.md`](crates/core/PERFORMANCE.md).** That doc is the canonical home — when build wall-clock or backend ordering changes, update it, not this file. As a one-line summary at time of writing: on Ryzen 7 9800X3D + RTX 5070 Ti (16 threads), `Scores::new_with` is **8.82 s naive → 193 ms simd_batch → 951 ms cuda**. simd_batch is the fastest backend end-to-end (45.7× over naive, **4.93× faster than CUDA**) after the sparse fused-keeper-round + flat-array indexing + `#[inline]` on the per-state DP fns + a precomputed `(score + state_scores[child])` table + vectorized `score_and_child` across the 8 lanes + parallel BFS reachability filter + d-independent gather hoist landed; CUDA still wins per-level at L≤4 but loses everywhere else. Scaling is near-linear on physical cores (97% efficiency at 4T, 84% at 8T) but SMT adds nothing — 8-12 threads is the sweet spot.

Helpful entry points:
- **Examples**: `crates/core/examples/time_build.rs` (default), `time_build_naive.rs`, `cuda_smoke.rs`.
- **Benches**: `crates/core/benches/recommend.rs` — see the `bench_backends` (per-state), `bench_build_backends` (full builds), and `bench_simd_batch_phases` (per-stage micro-benches that time each of `entry_actions_fuse` / `gemv` / `masked_max` / `final_dot` in isolation against pre-filled scratch) groups. The phase fns are exposed `#[doc(hidden)] pub` from `simd_batch.rs` for that purpose.
- **Cross-check tests**: `test_naive_backend_matches` asserts naive and ndarray agree on the default-state EV within 1e-3 (single scalar tripwire). `proptests::linalg_backends_agree_on_action_ranking` is the broader cousin: for `arb_state() × dice_idx`, every enabled `LinalgBackend` (naive, ndarray, optionally faer/simd) must produce the same overall EV *and* the same EV at each rank in the sorted entries / first_keepers / second_keepers lists, within 5e-3. This catches structural backend bugs that nudge EVs by ~1e-2 on niche states — well below the default-state test's noise floor. `test_unvalidated_matches_validated` (gated `#[ignore]`) compares `Scores::new_with` against `Scores::new_with_unvalidated` across every reachable state — cross-checks the BFS soundness as a side effect. `test_simd_batch_matches` (gated `#[ignore]`, `--features simd`) compares `SimdBatchBuildBackend` against `CpuBuildBackend` across every reachable state, catching structural bugs in the outer-loop pipeline that the per-state `LinalgBackend` proptest can't see. Both ignored tests run with `cargo test -p yahtzee-core --release --features simd -- --ignored`.

### Scoring helpers

- `State::score_and_child(action, dice_idx) -> (f32, State)` — total points scored this turn (normal score + upper bonus if triggered + Yahtzee bonus if triggered, joker-aware) plus the resulting child state. This is the "full fidelity" scoring primitive.
- `State::entry_score(action, &DiceCounts) -> u8` — just the value that goes into the entry box itself (joker-aware), with no bonuses. Used for UI hints where bonuses are displayed separately.
- `achievable_scores(entry_idx: usize) -> Vec<u8>` — the set of legal score values for a category, derived from unique values of `DICE_AND_ENTRY_SCORES` row `i` plus 25 / 30 / 40 for Full House / Small Straight / Large Straight (joker values). The set is state-independent: it may miss joker-enabled upper values that aren't reachable via normal scoring. UI validation uses it as an allow-list.

### `ExpectedValues` API

Read off the current state's precomputed tables:

- `first_keepers_score(dice) -> Vec<(DiceCounts, f32)>` / `second_keepers_score(dice) -> ...` — per-keeper overall-EV rankings for rolls 1 and 2.
- `entries_score(dice) -> Vec<(EntryAction, f32)>` — per-entry overall-EV rankings for roll 3.
- `first_keepers_with_turn_ev(dice)` / `second_keepers_with_turn_ev(dice)` / `entries_with_turn_ev(dice)` — same shape as above but each tuple is `(choice, overall_ev, turn_ev)`. `turn_ev` is the expected points scored **on this turn only** if that choice is taken (and optimal play continues thereafter), including any +35 / +100 bonuses triggered this turn.
- `this_turn_ev(dice, roll) -> f32` — scalar expected-points-this-turn for the current dice/roll, assuming optimal play. Complements `value` (which is this turn + all future turns). On roll 3 it's deterministic (best action's immediate score); on rolls 1/2 it marginalizes through the optimal keeper selection and subsequent dice/action choices using the cached `*_keepers` / `*_dice` arrays.

Private helpers `turn_ev_by_roll3_dice()` and `turn_ev_by_roll2_dice(...)` build per-dice turn-EV arrays at the top of each `*_with_turn_ev` call (recomputed per call, not memoized across calls), so the per-keeper outer loop is a cheap O(num_dice_combos) inner product instead of a nested decision-tree walk (~300 kop on roll 1, ~120 kop on roll 2).

## Serialization, CLI, wasm

`Scores` derives `Serialize` / `Deserialize` and is persisted with `bincode`. The CLI (`crates/cli/src/main.rs`) is `clap`-subcommand-based — see the Commands section above for the four subcommands (`solve`, `value`, `play`, `build`) and how `--scores PATH` overrides the embedded score table. The `--entries` argument used by `solve` / `value` is a 13-character `0`/`1` string aligned with `ENTRY_ACTIONS` order.

CLI feature wiring (`crates/cli/Cargo.toml`):

- `default = ["simd"]` — the `build` subcommand dispatches through `SimdBatchBuildBackend` (outer-loop SIMD across 8 states + sparse fused keeper round + precomputed entry-actions table + vectorized `score_and_child` + d-independent gather hoist + parallel BFS reachability filter; ~30.1× wall-clock vs `NdarrayBackend`, no API change). The `solve` / `value` / `play` subcommands deserialize the embedded table and don't hit `Scores::new()` regardless of features, so this only affects table regeneration.
- `cuda` (opt-in) — when compiled (`cargo build -p yahtzee-cli --release --features cuda`), `build` dispatches through `CudaBuildBackend` instead, taking precedence over `simd`. Requires `libcuda` / `libcublas` / `libnvrtc` `.so`s on the runtime host (cudarc dynamic-loads, no CUDA SDK at build time). If GPU init fails the subcommand errors out — there's no automatic CPU fallback.
- `--no-default-features` — falls back to `Scores::new()` (`NdarrayBackend`). Useful for benching the unaccelerated path.

The selection is compile-time (`cfg`-gated `build_scores()` in `crates/cli/src/main.rs`), not a runtime flag.

The wasm crate intentionally consumes `yahtzee-core` with default features only. It deserializes the embedded brotli blob rather than calling `Scores::new()`, so `simd` / `faer` / `cuda` would only affect the per-state `Scores::values()` walk (~100 µs/call, single-call-per-UI-action) — not worth a feature surface or extra deps. The canonical `scores.bin.br` is regenerated through the CLI's `build` subcommand, so backend-driven build wins land via the CLI path.

`crates/wasm/src/lib.rs` exposes:

- `class Solver` — `new(bytes)` deserializes a `scores.bin`. Methods: `recommend(state, dice, roll)` returns `{ value, keepers?, entries? }` where keeper/entry rows carry both `ev` (overall) and `turn_ev` (this turn), plus `best_entry` on keepers that have all 5 dice kept (= "stop rolling and score here"). `stateValue(state)` and `thisTurnEv(state, dice, roll)` are thin passthroughs.
- Free functions `entryScore(state, dice, entry_idx) -> u8` and `achievableScores(entry_idx) -> Uint8Array` for UI scoring / validation without needing a `Solver` instance.

Wasm doesn't *build* score tables at runtime (it deserializes the embedded brotli blob), so backend choice only affects the canonical `scores.bin.br` regeneration path that runs through the CLI's `build` subcommand. Whether to enable `simd` for wasm anyway (e.g. for any future runtime computation) is a separate question — `wide::f32x8` falls back to scalar on wasm targets without `wasm-simd128`, so it's safe to enable but a no-op without the wasm-simd128 target feature.

## Web app

Svelte 4 + TypeScript + Vite. `web/src/App.svelte` is the whole UI: a scorecard on the left (score inputs with per-category validation against `achievableScores`, Yahtzee-bonus counter clamped to legal range) and dice + recommendation panel on the right (clickable SVG dice, roll 1/2/3 segmented control, a unified ranked choice list where each row says `keep [dice]` or `score in <Category>`). The top recommendation is visualised by either shading the kept dice or highlighting the target scorecard row, whichever matches the top choice kind. Joker-eligible lower rows link out to the Wikipedia joker-rule section.
