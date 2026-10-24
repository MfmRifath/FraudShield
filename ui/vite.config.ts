import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The UI always calls the API under /api: Vite proxies it in dev, nginx in the container.
const api = process.env.FRAUDSHIELD_API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: api, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
  build: { outDir: "dist", sourcemap: false },
  test: { environment: "jsdom", globals: true, setupFiles: ["src/test/setup.ts"] },
});
