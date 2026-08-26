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
  // The tab this effect last acted on. A boolean "have I mounted yet" flag does
  // not survive StrictMode, which runs the effect, cleans up, and runs it again:
  // the first pass sets the flag and the second pass sails through and scrolls
  // and steals focus on arrival. Comparing tab values is idempotent.
  const actedOn = useRef(tab);

  useEffect(() => {
    // First render, or a re-run for a tab we already handled: leave the page
    // alone. A deep link should not fight the browser's own restoration.
    if (actedOn.current === tab) return;
    actedOn.current = tab;
    window.scrollTo(0, 0);
    // Rescue focus only when the switch is what took it away. The tab strip
    // lives in the header, outside the panels, so a tab click hides nothing the
    // reader was focused on; the one control that does is "Continue to Module
    // N", which sits inside the panel being hidden. Hence the test: is the
    // focused element inside a subtree we just hid? Asking whether
    // document.activeElement is <body> answers a different question, and
    // answers it wrongly twice over. Too early, because React has hidden the
    // panel but the browser has not blurred the button in it yet; too eager,
    // because nothing focused is also the resting state of a freshly loaded
    // page. Both readings put a focus ring around the module title.
    const active = document.activeElement as HTMLElement | null;
    const lostFocus =
      !!active &&
      active !== document.body &&
      (!active.isConnected || active.closest("[hidden]") !== null);
    if (!lostFocus) return;
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
