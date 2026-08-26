import { useEffect, useRef, useState } from "react";
import { TrainingDemo } from "./m0/TrainingDemo";
import { MODULES } from "./modules/NN";

const DEMO_TAB = "demo";

// The active tab lives in the URL hash (#m1, #demo) so a reload or a shared
// link lands on the same module. Every module is always reachable; only an
// unknown hash falls back to the first module.
function tabFromHash(): string {
  const id = window.location.hash.slice(1);
  if (id === DEMO_TAB) return DEMO_TAB;
  return MODULES.some((m) => m.id === id) ? id : MODULES[0].id;
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

  // An unknown hash resolves to module 1; rewrite it so the URL never claims
  // to be somewhere the reader is not.
  useEffect(() => {
    if (window.location.hash.slice(1) !== tab) window.location.hash = tab;
  }, [tab]);

  // Modules are stacked in one document and hidden rather than unmounted, so
  // without this a tab switch keeps the previous module's scroll offset and the
  // browser clamps it into the new one: leaving Module 4 halfway through used to
  // land the reader on Module 2's closing recap. Start every module at its top,
  // which is what the Continue button already did.
  const panels = useRef<Record<string, HTMLDivElement | null>>({});
  const mounted = useRef(false);

  useEffect(() => {
    // Not on the first render: a deep link should not fight the browser's own
    // restoration, and nothing should steal focus on arrival.
    if (!mounted.current) {
      mounted.current = true;
      return;
    }
    window.scrollTo(0, 0);
    // Keyboard and screen-reader readers would otherwise be left focused on a
    // button whose panel just went away, which drops focus to the document
    // start. Land them on the heading of what they just opened instead.
    const heading = panels.current[tab]?.querySelector<HTMLElement>("h2");
    if (heading) {
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
    }
  }, [tab]);

  return (
    <div className="app">
      <header>
        <h1>Grokking Nets</h1>
        <p className="tagline">
          An interactive course on neural networks: read a little, play with live
          visualizations, and implement the real thing in Python, right here in your
          browser.
        </p>
        <nav className="tabs" aria-label="Course modules">
          {MODULES.map((m) => (
            <button
              key={m.id}
              className={`tab ${tab === m.id ? "tab-active" : ""}`}
              aria-current={tab === m.id ? "page" : undefined}
              onClick={() => selectTab(m.id)}
            >
              {m.navLabel}
            </button>
          ))}
          <button
            className={`tab ${tab === DEMO_TAB ? "tab-active" : ""}`}
            aria-current={tab === DEMO_TAB ? "page" : undefined}
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
          return (
            <div
              key={m.id}
              hidden={tab !== m.id}
              ref={(el) => {
                panels.current[m.id] = el;
              }}
            >
              <m.Component />
              {next && (
                <div className="next-module">
                  <button onClick={() => selectTab(next.id)}>
                    Continue to {next.navLabel} →
                  </button>
                </div>
              )}
            </div>
          );
        })}
        <div
          hidden={tab !== DEMO_TAB}
          ref={(el) => {
            panels.current[DEMO_TAB] = el;
          }}
        >
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
