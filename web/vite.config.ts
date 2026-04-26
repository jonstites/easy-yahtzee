import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import type { Connect, Plugin } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

function serveBrotliScores(): Plugin {
  // Read the canonical brotli blob out of the CLI crate. Same artifact the
  // CLI binary embeds via `include_bytes!`, so the web and CLI never drift.
  const resolved = (root: string) => path.resolve(root, '../crates/cli/data/scores.bin.br');
  const handler = (root: string): Connect.SimpleHandleFunction => (req, res) => {
    const file = resolved(root);
    try {
      const stat = fs.statSync(file);
      res.setHeader('Content-Type', 'application/octet-stream');
      res.setHeader('Content-Encoding', 'br');
      res.setHeader('Content-Length', String(stat.size));
      fs.createReadStream(file).pipe(res);
    } catch {
      res.statusCode = 404;
      res.end(
        `scores.bin.br not found — run \`cargo run -p yahtzee-cli --release -- build --output crates/cli/data/scores.bin --brotli\``,
      );
    }
  };
  return {
    name: 'serve-brotli-scores',
    configureServer(server) {
      server.middlewares.use('/scores.bin', handler(server.config.root));
    },
    configurePreviewServer(server) {
      server.middlewares.use('/scores.bin', handler(server.config.root));
    },
  };
}

export default defineConfig({
  plugins: [svelte(), serveBrotliScores()],
  publicDir: 'static',
  server: {
    fs: {
      allow: ['..'],
    },
  },
});
