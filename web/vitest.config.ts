/// <reference types="vitest" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Vitest config — kept separate from `vite.config.ts` so the dev
// server's proxy and the test runner's jsdom environment don't
// share configuration accidentally.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true, // expect/describe/it without imports
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
