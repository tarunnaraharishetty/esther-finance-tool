import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Vite dev-server config.
//
// We proxy /api → http://localhost:8000 so the frontend can use
// relative paths in dev *and* in eventual same-origin production
// hosting without code changes. The backend separately enables
// permissive CORS, so a direct cross-origin call also works — the
// proxy is just for ergonomics.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  build: {
    // Recharts and cmdk are heavy enough to deserve their own
    // long-lived caches. React is split out so updates to app code
    // don't bust the framework chunk.
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          recharts: ["recharts"],
          cmdk: ["cmdk"],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
