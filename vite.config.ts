import { readFileSync } from "node:fs";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// The built site carries the license text, because it carries the code the
// license is about. src/python/reference_network.py is adapted from Nielsen's
// MIT-licensed network.py and is inlined into the Pyodide worker at build
// time, and every solution.py is imported ?raw into the bundle, so a deploy
// distributes MIT-derived code. MIT asks that "the above copyright notice and
// this permission notice" travel with it, and until this plugin existed they
// did not: dist/ held the code and none of the notice.
//
// Copying at build time rather than keeping a second copy in public/ is what
// stops the two from drifting, which is the usual way this gets quietly
// broken again.
function licenseInBuild(): Plugin {
  return {
    name: "license-in-build",
    apply: "build",
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "LICENSE.txt",
        source: readFileSync(new URL("./LICENSE", import.meta.url), "utf8"),
      });
    },
  };
}

// base is relative so the static build works from any subpath (e.g. GitHub Pages).
export default defineConfig({
  base: "./",
  plugins: [react(), licenseInBuild()],
  // The dev server always answers on 5174, so a bookmarked or hand-typed
  // localhost:5174 is never wrong. strictPort makes a busy port a startup
  // error instead of a silent hop to 5175: a drifting port is how you end up
  // reloading an address that nothing is serving. PORT still wins, so the
  // preview harness can assign one and two sessions can run side by side.
  server: { port: Number(process.env.PORT) || 5174, strictPort: true },
});
