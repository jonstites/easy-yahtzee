import { defineConfig, devices } from '@playwright/test';

// E2E config. Spawns the Vite dev server automatically and runs tests against
// it in headless Chromium. We deliberately stick to a single browser — these
// tests guard application logic (button wiring, undo, full-game playthrough),
// not browser-specific rendering, so cross-browser coverage isn't worth the
// runtime cost on a personal project.
//
// Prerequisites for the dev server (same as `pnpm dev`):
//   - crates/wasm/pkg/ built (run `wasm-pack build --target web --out-dir pkg`
//     in crates/wasm if missing)
//   - web/static/scores.bin.br present (the brotli-compressed solver tables)
export default defineConfig({
  testDir: './e2e',
  // Forbid `.only` in CI so a stray focused test can't silently skip the rest.
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'pnpm dev --port 5173 --strictPort',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
