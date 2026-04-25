# web/CLAUDE.md

Guidance for working in `web/`. The repo-level `CLAUDE.md` covers the workspace layout and the Rust crates; this file adds the frontend-specific bits.

## Module map

`src/`:
- `App.svelte` — single top-level component. Owns reactive state (dice, kept mask, roll counter, undo history, recommendation result) and event handlers. Glue only — game/scoring logic lives in the modules below.
- `Scorecard.svelte` — left-pane scorecard. Pure props-in / `bind:`-out. No wasm calls.
- `HelpModal.svelte` — keyboard-shortcut help.
- `constants.ts` — `ENTRY_LABELS` (canonical scorecard order, indices 0..12), `PIP_POS`, named indices `YAHTZEE_IDX`, `SMALL_STRAIGHT_IDX`.
- `scoring.ts` — pure dice/category helpers (`keepMaskFor`, `scoreContributionMask`, `cycleFace`, `isYahtzee`, `randomDice`, `formatAllowed`, `parseScoreInput`, `maxYahtzeeBonuses`). Each one has unit tests.
- `recommendation.ts` — wire types (`KeeperRec`, `EntryRec`, `Recommendation`), the UI's discriminated `Choice` union (`'reroll' | 'keep' | 'score'`), and `buildChoices(rec, limit)` which merges keeper rows + entry rows by overall EV. **Wire types must mirror `crates/wasm/src/lib.rs`.** Change one, change the other.

Tests:
- `src/*.test.ts` — vitest unit tests (`pnpm test`).
- `e2e/app.spec.ts` — Playwright E2E tests (`pnpm e2e`).

## Important invariants (don't quietly break these)

- **Entry indices are canonical.** `EntryRec.entry: u8` in 0..12, mapped via `ENTRY_LABELS[idx]`. There is *no* string-name round-trip with the wasm.
- **The all-5 keeper is excluded by the wasm.** "Keep all 5 then re-roll some later" is strictly dominated by "re-roll those now". The user's "score these dice immediately" option comes from the `entries` field on rolls 1/2 (populated with the score-now EV from `entries_with_turn_ev`), not from a synthetic 5-die keeper. Don't reintroduce one.
- **`Recommendation.entries` is populated on every roll.** On rolls 1/2 it represents "skip remaining rolls and score now"; on roll 3 it's the only option set. `buildChoices` merges keepers + entries by `ev` and slices to `limit`.
- **`turn_ev` ≠ `ev`.** `turn_ev` is points scored *this turn alone* (incl. +35/+100 bonuses triggered this turn). `ev` is full remaining-game EV. Don't conflate them in display or math.

## Commands

| Task | Command |
|---|---|
| Dev server | `pnpm dev` (port 5173 default) |
| Type-check | `pnpm check` |
| Unit tests | `pnpm test` / `pnpm test:watch` |
| E2E tests | `pnpm e2e` / `pnpm e2e:ui` (interactive) |
| Production build | `pnpm build` |

First-time Playwright setup needs `pnpm exec playwright install chromium` (downloads ~110 MB of browser binary into `~/.cache/ms-playwright/`).

## Things that need rebuilding

- **After touching `crates/core` or `crates/wasm`:** rebuild the wasm pkg with `cd ../crates/wasm && wasm-pack build --target web --out-dir pkg`. The dev server hot-reloads.
- **After touching `Scores` serialization (in `crates/core`):** regenerate `static/scores.bin.br`: from the repo root, `cargo run --release --bin build-cache`.

## Vitest / Playwright file globs

Vitest's default glob picks up both `*.test.ts` and `*.spec.ts`. `vitest.config.ts` confines it to `src/**` so the Playwright specs in `e2e/` aren't picked up by the wrong runner. If you rename or move test directories, update both configs.

## Dev-server quirks

- `vite.config.ts` registers a middleware that intercepts `/scores.bin` requests and streams `static/scores.bin.br` with `Content-Encoding: br`. Don't put a real `scores.bin` at the URL — the brotli middleware short-circuits.
- `publicDir` is `static/`, not the Vite default `public/`.

## Conventions

- Game logic that's pure goes in `scoring.ts` (or `recommendation.ts` if it's about EV/choice plumbing). If you find yourself writing a non-trivial pure function inline in `App.svelte`, lift it.
- Tests live next to the module they cover (`scoring.ts` ↔ `scoring.test.ts`).
- E2E tests use roles + class selectors, not nth-of-type CSS chains. Counting `circle` elements inside a `.die` SVG is the standard way to read a die's face value (pip count == face).
