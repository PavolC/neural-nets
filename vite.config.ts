import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base is relative so the static build works from any subpath (e.g. GitHub Pages).
export default defineConfig({
  base: "./",
  plugins: [react()],
  // The dev server always answers on 5174, so a bookmarked or hand-typed
  // localhost:5174 is never wrong. strictPort makes a busy port a startup
  // error instead of a silent hop to 5175: a drifting port is how you end up
  // reloading an address that nothing is serving. PORT still wins, so the
  // preview harness can assign one and two sessions can run side by side.
  server: { port: Number(process.env.PORT) || 5174, strictPort: true },
});
