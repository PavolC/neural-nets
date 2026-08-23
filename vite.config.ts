import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base is relative so the static build works from any subpath (e.g. GitHub Pages).
export default defineConfig({
  base: "./",
  plugins: [react()],
  // Honor an assigned port (e.g. from the Claude Code preview harness) so two
  // sessions can run dev servers side by side; falls back to Vite's default.
  server: process.env.PORT ? { port: Number(process.env.PORT) } : undefined,
});
