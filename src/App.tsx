import { useEffect, useState } from "react";
import { TrainingDemo } from "./m0/TrainingDemo";
import { MODULES, isModuleUnlocked } from "./modules/NN";
import { subscribeProgress } from "./state/progress";

const DEMO_TAB = "demo";

export default function App() {
  const [tab, setTab] = useState<string>(MODULES[0].id);
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
                onClick={() => setTab(m.id)}
              >
                {unlocked ? m.navLabel : `🔒 ${m.navLabel}`}
              </button>
            );
          })}
          <button
            className={`tab ${tab === DEMO_TAB ? "tab-active" : ""}`}
            onClick={() => setTab(DEMO_TAB)}
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
                      setTab(next.id);
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
