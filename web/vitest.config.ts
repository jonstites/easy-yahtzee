import { defineConfig, mergeConfig } from 'vitest/config';
import viteConfig from './vite.config';

// Vitest config layered on top of vite.config.ts. We only override the test
// section: confine vitest to `src/**` so it doesn't try to run the Playwright
// E2E specs in `e2e/` (which use a different test runner).
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      include: ['src/**/*.{test,spec}.{js,ts}'],
    },
  }),
);
