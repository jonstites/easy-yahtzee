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

`Scores::new()` allocates `state_scores: Array1<f32>` of length `NUM_STATES` and fills it level-by-level from 13 filled entries down to 0 (`set_scores`). Each level is parallelized with `rayon` (`par_iter_mut`); correctness of the parallelism depends on processing strictly higher levels first, since a state only depends on states with more entries filled.

`Scores::values(state)` returns an `ExpectedValues` by walking the three-roll decision tree in reverse:

1. `entry_actions[action, dice]` = immediate score of taking `action` on `dice` + stored EV of the resulting child state (via `State::score_and_child`, which handles upper bonus, Yahtzee bonus, and the joker rule).
2. `third_dice[dice] = max_action entry_actions[action, dice]` (best entry to pick on the final roll).
3. Alternate keepers and dice: `second_keepers = KEEPERS_TO_DICE_PROBABILITIES · third_dice`; `second_dice[d] = max over allowed keepers of DICE_TO_ALLOWED_KEEPERS[d] * second_keepers`; repeat to get `first_keepers` and `first_dice`.
4. Overall EV = initial-roll distribution (row 0 of `KEEPERS_TO_DICE_PROBABILITIES`) · `first_dice`.

The expected value from the default state is ~254.5896 (asserted in `test_expected_value`).

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

Private helpers `turn_ev_by_roll3_dice()` and `turn_ev_by_roll2_dice(...)` precompute per-dice turn-EV arrays so the `*_with_turn_ev` outer loops stay cheap (~300 kop on roll 1, ~120 kop on roll 2).

## Serialization, CLI, wasm

`Scores` derives `Serialize` / `Deserialize` and is persisted with `bincode`. The CLI (`crates/cli/src/main.rs`) either regenerates scores (`-o`) or loads them (`-i`), then for `--roll 1|2|3` prints ranked keepers / entries by EV. The `--entries` argument is a 13-character `0`/`1` string aligned with `ENTRY_ACTIONS` order.

`crates/wasm/src/lib.rs` exposes:

- `class Solver` — `new(bytes)` deserializes a `scores.bin`. Methods: `recommend(state, dice, roll)` returns `{ value, keepers?, entries? }` where keeper/entry rows carry both `ev` (overall) and `turn_ev` (this turn), plus `best_entry` on keepers that have all 5 dice kept (= "stop rolling and score here"). `stateValue(state)` and `thisTurnEv(state, dice, roll)` are thin passthroughs.
- Free functions `entryScore(state, dice, entry_idx) -> u8` and `achievableScores(entry_idx) -> Uint8Array` for UI scoring / validation without needing a `Solver` instance.

## Web app

Svelte 4 + TypeScript + Vite. `web/src/App.svelte` is the whole UI: a scorecard on the left (score inputs with per-category validation against `achievableScores`, Yahtzee-bonus counter clamped to legal range) and dice + recommendation panel on the right (clickable SVG dice, roll 1/2/3 segmented control, a unified ranked choice list where each row says `keep [dice]` or `score in <Category>`). The top recommendation is visualised by either shading the kept dice or highlighting the target scorecard row, whichever matches the top choice kind. Joker-eligible lower rows link out to the Wikipedia joker-rule section.
