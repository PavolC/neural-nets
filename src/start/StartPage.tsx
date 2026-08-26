import { useEffect, useRef, useState } from "react";
import { TrainingDemo } from "../m0/TrainingDemo";
import { EXERCISES } from "../exercises/registry";
import {
  exportProgress,
  importProgress,
  loadCompleted,
  resetAll,
  resetExercise,
  subscribeProgress,
} from "../state/progress";

// The course's front door, and the only page that talks about the course
// rather than about neural networks: what it is, how the machinery works,
// what the eight modules cover, the training run that shows where it ends up,
// and what this browser has stored. Reachable at #start, which is where a
// bare link lands.

interface Outline {
  id: string;
  title: string;
  covers: string;
}

const OUTLINE: Outline[] = [
  {
    id: "m1",
    title: "From neurons to networks",
    covers:
      "One neuron as a weighted decision, why one straight cut cannot answer every question, and the sigmoid as a step with a slope.",
  },
  {
    id: "m2",
    title: "Feedforward",
    covers:
      "A layer as one matrix multiplication, shape discipline, and pretrained weights reading real digits through your own code.",
  },
  {
    id: "m3",
    title: "Learning as descent",
    covers:
      "Cost as a landscape, slopes measured by nudging, step size, mini-batches, and what measuring 11,935 slopes that way costs.",
  },
  {
    id: "m4",
    title: "Backpropagation, the idea",
    covers:
      "Where a gradient comes from: factors along a chain, blame flowing backward, and the four equations, stepped through one number at a time.",
  },
  {
    id: "m5",
    title: "Backpropagation, for real",
    covers:
      "The four equations as fifteen lines of NumPy, checked entry by entry against numerically measured gradients, then training the digit reader.",
  },
  {
    id: "m6",
    title: "Universality (an interlude)",
    covers:
      "Why a big enough hidden layer can imitate any curve, built by hand out of sigmoid pairs. No code, no gate.",
  },
  {
    id: "m7",
    title: "Making it actually work",
    covers:
      "Three one-line changes, each measured: the cross-entropy cost, a scaled starting draw, and weight decay against overfitting.",
  },
  {
    id: "m8",
    title: "Why deep is hard (and what came next)",
    covers:
      "What breaks when the same network goes deeper, why it follows from BP2, and where convolutions, ReLU and embedding spaces fit.",
  },
];

const FILE_NAME = "grokking-nets-progress.json";

export function StartPage({ onGoTo }: { onGoTo: (moduleId: string) => void }) {
  const [, bump] = useState(0);
  const [note, setNote] = useState<string | null>(null);
  const [noteError, setNoteError] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => subscribeProgress(() => bump((n) => n + 1)), []);

  const say = (text: string, isError = false) => {
    setNote(text);
    setNoteError(isError);
  };

  const passed = EXERCISES.filter((e) => loadCompleted(e.id));

  const download = () => {
    const blob = new Blob([exportProgress()], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = FILE_NAME;
    a.click();
    URL.revokeObjectURL(url);
    say(`Saved ${FILE_NAME}. Load it in another browser to carry your work over.`);
  };

  const load = (file: File) => {
    file
      .text()
      .then((text) => {
        const written = importProgress(text);
        say(`Loaded ${written} saved ${written === 1 ? "item" : "items"} from ${file.name}.`);
      })
      .catch((err: Error) => say(`Nothing was loaded: ${err.message}.`, true));
  };

  const forgetAll = () => {
    if (
      !window.confirm(
        "Forget everything this browser has stored for the course: the code in every editor, every revealed hint, and every passed mark. The course text is unaffected. This cannot be undone.",
      )
    )
      return;
    resetAll();
    say("Cleared. Every exercise is back to its skeleton.");
  };

  return (
    <article className="module start-page">
      <h2>Start here</h2>
      <p>
        This is a course on neural networks that you finish by building one. Each
        module is a few short readings interleaved with figures you can drag, and
        most of them end with Python you write in the page. Your code is checked by
        tests, kept, and used by every module after it: by Module 5 it is training a
        network that reads handwritten digits, and by Module 8 it is the thing being
        measured when depth stops working.
      </p>
      <p>
        The sequence follows Michael Nielsen's <em>Neural Networks and Deep
        Learning</em>, with the explanations rewritten around the interactive parts.
        It assumes you can read Python and have seen high-school algebra. Vectors,
        matrices, dot products and slopes are all built up here from arithmetic, in
        that order, when they are first needed.
      </p>
      <div className="start-cta">
        <button onClick={() => onGoTo("m1")}>Begin Module 1 →</button>
        {passed.length > 0 && (
          <span className="start-cta-note">
            {passed.length} of {EXERCISES.length} exercises passed in this browser.
          </span>
        )}
      </div>

      <h3 id="start-how">How the machinery works</h3>
      <ul className="start-facts">
        <li>
          <b>Python really runs here.</b> An in-page editor sends your code to
          CPython 3.14 compiled to WebAssembly (Pyodide), with NumPy, in a Web
          Worker so a training run never freezes the page. The first run downloads
          about 10 MB of runtime and the browser caches it after that.
        </li>
        <li>
          <b>Nothing to install, and nothing to sign up for.</b> There is no server
          and no account. The only thing that leaves your machine is the runtime
          download.
        </li>
        <li>
          <b>Your work lives in this browser.</b> Editor contents, revealed hints
          and passed marks are stored in this browser's local storage, per browser
          and per device. The progress section below is how you move or clear them.
        </li>
        <li>
          <b>Nothing is locked.</b> Every module is open from the tabs at any time,
          in any order. What passing an exercise unlocks is the panel that trains
          with your own code, since there is nothing to run until the code exists.
        </li>
        <li>
          <b>Each exercise shows its work.</b> The tests are readable in the page,
          hints come in three stages that you choose to open, and the reference
          solution is one of them.
        </li>
        <li>
          <b>Module 5 is the summit, and nothing after it depends on your version.</b>{" "}
          A training panel needs the exercises in its own module, plus Module 3's
          sgd, which drives all of them. Module 5's backprop is the exception:
          Modules 7 and 8 run on the course's own copy, so an exercise you never
          finish costs you that module's panel and nothing later.
        </li>
      </ul>

      <h3 id="start-modules">The eight modules</h3>
      <ol className="start-outline">
        {OUTLINE.map((m, i) => {
          const here = EXERCISES.filter((e) => e.module === m.id);
          return (
            <li key={m.id}>
              <button className="start-outline-title" onClick={() => onGoTo(m.id)}>
                {i + 1}. {m.title}
              </button>
              <p>{m.covers}</p>
              {here.length > 0 && (
                <p className="start-outline-writes">
                  You write:{" "}
                  {here.map((e, k) => (
                    <span key={e.id}>
                      {k > 0 && ", "}
                      <code>{e.builds.split(":")[0]}</code>
                      {loadCompleted(e.id) && <span className="start-tick"> passed</span>}
                    </span>
                  ))}
                </p>
              )}
            </li>
          );
        })}
      </ol>

      <TrainingDemo />

      <h3 id="start-progress">What this browser has stored</h3>
      <p>
        {passed.length} of {EXERCISES.length} exercises are passing here.{" "}
        {passed.length === 0
          ? "Nothing is saved yet."
          : "Resetting one puts back its skeleton and clears its hints and its passed mark."}
      </p>
      <ul className="start-progress">
        {EXERCISES.map((e) => {
          const done = loadCompleted(e.id);
          return (
            <li key={e.id} className={done ? "start-done" : ""}>
              <span className="start-progress-mark" aria-hidden="true">
                {done ? "✓" : "·"}
              </span>
              <span className="start-progress-name">
                <button className="start-progress-link" onClick={() => onGoTo(e.module)}>
                  {e.title}
                </button>{" "}
                <span className="start-progress-where">
                  Module {e.module.slice(1)}
                  <span className="sr-only">{done ? ", passed" : ", not passed yet"}</span>
                </span>
                <span className="start-progress-builds">{e.builds}</span>
              </span>
              <button
                className="button-secondary"
                onClick={() => {
                  if (
                    window.confirm(
                      `Reset "${e.title}"? Its editor goes back to the skeleton, and its hints and passed mark are cleared.`,
                    )
                  ) {
                    resetExercise(e.id);
                    say(`Reset ${e.title}.`);
                  }
                }}
              >
                Reset
              </button>
            </li>
          );
        })}
      </ul>
      <div className="start-storage">
        <button className="button-secondary" onClick={download}>
          Save my progress to a file
        </button>
        <button className="button-secondary" onClick={() => fileRef.current?.click()}>
          Load progress from a file
        </button>
        <button className="button-secondary" onClick={forgetAll}>
          Forget everything
        </button>
        <input
          ref={fileRef}
          className="sr-only"
          type="file"
          accept="application/json,.json"
          onChange={(ev) => {
            const file = ev.target.files?.[0];
            ev.target.value = ""; // so the same file can be picked twice
            if (file) load(file);
          }}
        />
      </div>
      <p className={`status-fixed ${noteError ? "demo-status-error" : "demo-status"}`} role="status">
        {note ?? ""}
      </p>
      <p className="start-storage-note">
        A saved file holds the code in every editor, which hints you opened, and which
        exercises passed. Loading one replaces the exercises it names and leaves the
        rest alone.
      </p>
    </article>
  );
}
