import { useState } from "react";
import { TrainingDemo } from "./m0/TrainingDemo";
import { ExercisePage } from "./components/ExercisePage";
import { feedforwardExercise } from "./exercises/feedforward";

const TABS = [
  { id: "exercise", label: "Exercise: feedforward" },
  { id: "m0", label: "Training demo" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<TabId>("exercise");

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
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab ${tab === t.id ? "tab-active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {/* Both stay mounted so switching tabs never loses editor or chart state. */}
        <div hidden={tab !== "exercise"}>
          <ExercisePage exercise={feedforwardExercise} />
        </div>
        <div hidden={tab !== "m0"}>
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
