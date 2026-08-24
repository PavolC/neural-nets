import { useEffect, useState } from "react";
import { TrainingDemo } from "./m0/TrainingDemo";
import { MODULES, isModuleUnlocked } from "./modules/NN";
import { subscribeProgress } from "./state/progress";

const DEMO_TAB = "demo";

// The active tab lives in the URL hash (#m1, #demo) so a reload or a shared
// link lands on the same module. Locked or unknown hashes fall back to the
// first module.
function tabFromHash(): string {
  const id = window.location.hash.slice(1);
  if (id === DEMO_TAB) return DEMO_TAB;
  const idx = MODULES.findIndex((m) => m.id === id);
  if (idx >= 0 && isModuleUnlocked(idx)) return id;
  return MODULES[0].id;
}

export default function App() {
  const [tab, setTab] = useState<string>(tabFromHash);
  const selectTab = (id: string) => {
    window.location.hash = id;
    setTab(id);
  };
  useEffect(() => {
    const onHashChange = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  // Gating depends on exercise completion; re-render when it changes.
  const [, setProgressTick] = useState(0);
  useEffect(() => subscribeProgress(() => setProgressTick((t) => t + 1)), []);

  return (
    <div className="app">
      <header>
        <h1>Grokking Nets</h1>
        <p className="tagline">
          An interactive course on neural networks: read a little, play with live
          visualizations, and implement the real thing in Python, right here in your
          browser.
        </p>
        <nav className="tabs">
          {MODULES.map((m, i) => {
            const unlocked = isModuleUnlocked(i);
            return (
              <button
                key={m.id}
                className={`tab ${tab === m.id ? "tab-active" : ""}`}
                disabled={!unlocked}
                title={unlocked ? undefined : "Pass the previous module's exercise to unlock"}
                onClick={() => selectTab(m.id)}
              >
                {unlocked ? m.navLabel : `🔒 ${m.navLabel}`}
              </button>
            );
          })}
          <button
            className={`tab ${tab === DEMO_TAB ? "tab-active" : ""}`}
            onClick={() => selectTab(DEMO_TAB)}
          >
            Training demo
          </button>
        </nav>
      </header>
      <main>
        {/* Everything stays mounted so tab switches never lose editor or
            visualization state. */}
        {MODULES.map((m, i) => {
          const next = MODULES[i + 1];
          const nextUnlocked = next !== undefined && isModuleUnlocked(i + 1);
          return (
            <div key={m.id} hidden={tab !== m.id}>
              <m.Component />
              {next && (
                <div className="next-module">
                  <button
                    disabled={!nextUnlocked}
                    title={nextUnlocked ? undefined : "Pass this module's exercise first"}
                    onClick={() => {
                      selectTab(next.id);
                      window.scrollTo(0, 0);
                    }}
                  >
                    {nextUnlocked
                      ? `Continue to ${next.navLabel} →`
                      : `🔒 Pass this module's exercise to continue to ${next.navLabel}`}
                  </button>
                </div>
              )}
            </div>
          );
        })}
        <div hidden={tab !== DEMO_TAB}>
          <TrainingDemo />
        </div>
      </main>
      <footer>
        <p>
          Adapted from Michael A. Nielsen,{" "}
          <a href="http://neuralnetworksanddeeplearning.com/">
            <em>Neural Networks and Deep Learning</em>
          </a>
          , Determination Press, 2015, licensed{" "}
          <a href="https://creativecommons.org/licenses/by-nc/3.0/">CC BY-NC 3.0</a>.
          This derivative work is non-commercial and inherits the same license.
        </p>
      </footer>
    </div>
  );
}
