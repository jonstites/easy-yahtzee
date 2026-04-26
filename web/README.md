# easy-yahtzee — web

Svelte + Vite frontend for the Yahtzee solver. The actual DP solver is a Rust workspace one level up (`crates/core`); this app loads its precomputed value tables (`scores.bin`) through a thin `wasm-bindgen` wrapper (`crates/wasm`) and asks for keep / score recommendations as you play.

## Develop

Prerequisites:

1. Rust toolchain + [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/).
2. Node + [pnpm](https://pnpm.io/).
3. Build the wasm package once:

   ```sh
   cd ../crates/wasm && wasm-pack build --target web --out-dir pkg
   ```

   Re-run this whenever you touch `crates/core` or `crates/wasm`.

4. Generate the solver cache (one-time, slow — minutes on a fast machine):

   ```sh
   # from the workspace root
   cargo run --release --bin build-cache
   ```

   This writes `web/static/scores.bin.br` (a brotli-compressed bincode of the `Scores` table). The dev server serves it at `/scores.bin` with the right `Content-Encoding`.

Then in `web/`:

```sh
pnpm install
pnpm dev          # vite dev server on http://localhost:5173
```

## Tests

```sh
pnpm test         # vitest unit tests
pnpm test:watch   # vitest in watch mode
pnpm e2e          # Playwright E2E tests (headless chromium)
pnpm e2e:ui       # Playwright in interactive UI mode
pnpm check        # svelte-check type pass
```

First-time Playwright setup:

```sh
pnpm exec playwright install chromium
```

## Build

```sh
pnpm build        # outputs to dist/
```

The production build expects `static/scores.bin.br` to exist; copy or fetch it as part of your deploy step alongside `dist/`.

## Layout

```
src/
  App.svelte            top-level UI; orchestration only
  Scorecard.svelte      scorecard with per-row validation
  HelpModal.svelte      keyboard-shortcut overlay
  constants.ts          entry labels, pip positions, named indices
  scoring.ts            pure dice/category helpers (tested)
  recommendation.ts     wire types + Choice union + buildChoices (tested)
  *.test.ts             vitest unit tests
e2e/
  app.spec.ts           Playwright E2E tests
playwright.config.ts    chromium-only, auto-spawns the dev server
vitest.config.ts        confines vitest to src/**
```

## Keyboard

| Key | Action |
|---|---|
| Space | Re-roll dice not marked "keep" |
| Enter | Apply the top recommendation |
| R | Randomize all dice (for querying the solver about a real-world roll) |
| U | Undo |
| ? | Show / hide keyboard help |
| Esc | Close help |

Click a die to advance its face; right-click or scroll to go backward — handy for entering a specific real-world roll.
