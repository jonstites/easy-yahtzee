# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Build: `cargo build` (release: `cargo build --release`)
- Test: `cargo test` — note `Cargo.toml` sets `[profile.test] opt-level = 3` because `Scores::new()` is used in tests and is slow at `opt-level = 0` (minutes to hours).
- Run a single test: `cargo test test_expected_value -- --nocapture`
- CLI: `cargo run --release -- --help`. Generating scores from scratch takes a long time, so the typical flow is `cargo run --release -- -o scores.ytz` once, then `-i scores.ytz ...` for queries.

## Architecture

The crate is a Rust library (`src/lib.rs`) plus a thin CLI (`src/main.rs`). The library solves Yahtzee optimally via backward-induction dynamic programming.

### State space and indexing

A `State` is `(entries: EntryAction bitflags, yahtzee_bonus_eligible: bool, upper_score_remaining: u8)`. States are packed into a `usize` via `From<State>`/`From<usize>`, giving `NUM_STATES = 2^13 * 2 * 64 = 1_048_576` slots. Only `NUM_VALID_STATES = 536_448` are reachable; `Scores::set_valid_states` computes reachability by BFS from the default state.

`EntryAction` is a `bitflags` type where each of the 13 scoring categories is a single bit; `ENTRY_ACTIONS` is the canonical ordered array. `DiceCounts([u8; 6])` represents a multiset of dice; there are `NUM_DICE_COMBINATIONS = 252` full 5-dice combinations and `NUM_KEEPERS = 462` possible kept subsets (0..=5 dice).

### Precomputed tables (`mod math`, wrapped in `lazy_static!`)

- `DICE_AND_ENTRY_SCORES` — `Array2<u8>` of raw score for each `(entry, dice_idx)`.
- `DICE_TO_ALLOWED_KEEPERS` — `Array2<f32>` indicator matrix: which keeper subsets can be chosen from each full dice roll.
- `KEEPERS_TO_DICE_PROBABILITIES` — `Array2<f32>` transition probabilities from a kept subset to each resulting full roll after re-rolling.
- `YAHTZEE_DICE` — maps dice index to `Some(upper EntryAction)` iff all 5 dice are equal (used for joker rule + bonus eligibility).

These tables let a single-state evaluation reduce to ndarray matrix/vector ops.

### Solver (`Scores`)

`Scores::new()` allocates `state_scores: Array1<f32>` of length `NUM_STATES` and fills it level-by-level from 13 filled entries down to 0 (`set_scores`). Each level is parallelized with `rayon` (`par_iter_mut`); correctness of the parallelism depends on processing strictly higher levels first, since a state only depends on states with more entries filled.

`Scores::values(state)` computes an `ExpectedValues` by walking the three-roll decision tree in reverse:

1. `entry_actions[action, dice]` = immediate score of taking `action` on `dice` + stored EV of the resulting child state (via `State::score_and_child`, which handles upper bonus and the Yahtzee joker rule).
2. `third_dice[dice] = max_action entry_actions[action, dice]` (best entry to pick on the final roll).
3. Alternate between keepers and dice: `second_keepers = KEEPERS_TO_DICE_PROBABILITIES · third_dice`; `second_dice[d] = max over allowed keepers of DICE_TO_ALLOWED_KEEPERS[d] * second_keepers`; repeat to get `first_keepers` and `first_dice`.
4. Overall EV = initial-roll distribution (row 0 of `KEEPERS_TO_DICE_PROBABILITIES`) · `first_dice`.

The expected value from the default state is ~254.5896 (asserted in `test_expected_value`).

### Serialization and CLI

`Scores` derives `Serialize`/`Deserialize` and is persisted with `bincode`. The CLI (`src/main.rs`) either regenerates scores (`-o`) or loads them (`-i`), then for `--roll 1|2|3` prints ranked keepers/entries by EV. The `--entries` argument is a 13-character `0`/`1` string aligned with `ENTRY_ACTIONS` order.
