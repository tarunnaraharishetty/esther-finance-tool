import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite dev-server config.
//
// We proxy /api → http://localhost:8000 so the frontend can use
// relative paths in dev *and* in eventual same-origin production
// hosting without code changes. The backend separately enables
// permissive CORS, so a direct cross-origin call also works — the
// proxy is just for ergonomics.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        // EventSource keeps the connection open; Vite's proxy needs
        // ws=false (default) and no automatic close, which it does.
      },
    },
  },
});
