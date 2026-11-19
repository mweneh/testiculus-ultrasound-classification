import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const fromHere = (relative) => fileURLToPath(new URL(relative, import.meta.url));

// The API used during `npm run dev`; the production build is served by FastAPI.
const apiTarget = process.env.TESTICULUS_API ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Frozen, tracked experimental records bundled at build time so the
      // interface never retypes reported metrics.
      '@records': fromHere('../reports'),
      '@report': fromHere('../report'),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    fs: { allow: [fromHere('.'), fromHere('../reports'), fromHere('../report')] },
    proxy: Object.fromEntries(
      ['/health', '/predict', '/explain', '/docs', '/openapi.json'].map((route) => [route, apiTarget]),
    ),
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    // Keep every asset a same-origin file so the strict CSP needs no data: fonts.
    assetsInlineLimit: 0,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    css: false,
    restoreMocks: true,
  },
});
