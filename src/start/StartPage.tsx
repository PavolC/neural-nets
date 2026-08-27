import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { TrainingDemo } from "../m0/TrainingDemo";
import { M } from "../components/Math";
import { EXERCISES } from "../exercises/registry";
import {
  codeReady,
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

/** The lookup the course's own rule asks for: modules are written assuming
 * weeks pass between them, and every symbol below is defined once, thousands
 * of words before a reader comes back wanting it. In the order they are met.
 */
const NOTATION: { id: string; symbol: ReactNode; means: string; from: string }[] = [
  { id: "z", symbol: <M tex="z" />, means: "a neuron's evidence: its inputs times its weights, plus its bias", from: "Module 1" },
  { id: "a", symbol: <M tex="a" />, means: "a neuron's answer, the squash applied to its evidence", from: "Module 1" },
  { id: "sigma", symbol: <M tex="\sigma(z)" />, means: "the sigmoid, which squashes any number into 0 to 1", from: "Module 1" },
  { id: "wb", symbol: <M tex="w,\; b" />, means: "a weight (one per wire) and a bias (one per neuron)", from: "Module 1" },
  { id: "col", symbol: <code>(n, 1)</code>, means: "a column: n rows, 1 column. Every activation in this course is one", from: "Module 1" },
  { id: "flat", symbol: <code>(n,)</code>, means: "a flat array, neither row nor column. In this course, a bug", from: "Module 1" },
  { id: "dot", symbol: <code>6.</code>, means: "a number with a decimal point, so NumPy stores fractions not integers", from: "Module 1" },
  { id: "dotprod", symbol: <M tex="w \cdot x" />, means: "the dot product: multiply each pair, add the results", from: "Module 1" },
  { id: "mat", symbol: <code>(m, n)</code>, means: "a matrix: m rows, n columns. The rows name the receiving layer first", from: "Module 2" },
  { id: "W", symbol: <M tex="W" />, means: "a layer's weight matrix: row j is neuron j's incoming wires", from: "Module 2" },
  { id: "at", symbol: <code>@</code>, means: "matrix multiplication in NumPy. A plain * stays elementwise", from: "Module 2" },
  { id: "C", symbol: <M tex="C" />, means: "the cost: one number scoring the whole network on the data", from: "Module 3" },
  { id: "eta", symbol: <M tex="\eta" />, means: "eta, the learning rate: how far each step moves", from: "Module 3" },
  { id: "nabla", symbol: <M tex="\nabla C" />, means: "nabla C, the gradient: the whole list of slopes, one per parameter", from: "Module 3" },
  { id: "nablacode", symbol: <code>nabla_w, nabla_b</code>, means: "that same list in code, split into the weights' and biases' halves", from: "Module 3" },
  { id: "epoch", symbol: <code>epoch</code>, means: "one full pass through the training data", from: "Module 3" },
  { id: "batch", symbol: <code>mini-batch</code>, means: "the handful of examples one step is scored on", from: "Module 3" },
  { id: "sigprime", symbol: <M tex="\sigma'(z)" />, means: "sigma-prime: the sigmoid's steepness at z, equal to a(1 - a)", from: "Module 4" },
  { id: "delta", symbol: <M tex="\delta" />, means: "delta, a neuron's blame: how much the cost cares about its evidence", from: "Module 4" },
  { id: "odot", symbol: <M tex="\odot" />, means: "multiply matching entries, NumPy's plain *. No adding", from: "Module 4" },
  { id: "sup", symbol: <M tex="w^2,\; a^3" />, means: "a superscript is the layer number, never a power", from: "Module 4" },
  { id: "T", symbol: <M tex="(w)^T" />, means: "transpose: the same wires regrouped by sender instead of receiver", from: "Module 4" },
  { id: "partial", symbol: <M tex="\partial C / \partial w" />, means: "one parameter's slope, read as a single name", from: "Module 4" },
  { id: "receipts", symbol: <code>zs, activations</code>, means: "the receipts: every evidence and answer the forward pass computed", from: "Module 5" },
  { id: "neg", symbol: <code>a[-1]</code>, means: "the last entry of a list; a[-2] the one before it", from: "Module 5" },
  { id: "onehot", symbol: <code>one-hot</code>, means: "a column that is 1 in the right answer's slot and 0 everywhere else", from: "Module 5" },
  { id: "zeroslike", symbol: <code>np.zeros_like(w)</code>, means: "an array of zeros shaped exactly like w", from: "Module 5" },
  { id: "ln", symbol: <M tex="\ln a" />, means: "the natural logarithm, np.log: the power of e that gives a", from: "Module 7" },
  { id: "lmbda", symbol: <M tex="\lambda" />, means: "lambda, the regularization strength. Spelled lmbda in code", from: "Module 7" },
  { id: "nin", symbol: <M tex="n_{\text{in}}" />, means: "how many inputs feed a layer: the count the starting draw divides by", from: "Module 7" },
  { id: "heldout", symbol: <code>held-out</code>, means: "data never trained on, which every score in the course is measured on", from: "Module 7" },
  { id: "hyper", symbol: <code>hyperparameter</code>, means: "a number you choose rather than one descent finds", from: "Module 7" },
  { id: "relu", symbol: <code>ReLU</code>, means: "max(0, z): a squash with no ceiling, read as ray-loo", from: "Module 8" },
  { id: "argmax", symbol: <code>np.argmax(A, axis=0)</code>, means: "which row holds the largest value, down each column", from: "Module 9" },
  { id: "standardize", symbol: <code>standardize</code>, means: "shift and scale a feature to sit near 0 and about 1 wide", from: "Module 10" },
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
      </p>

      <h3 id="start-maths">About the mathematics</h3>
      <p>
        It assumes you can read Python and have seen high-school algebra: what a
        graph is, what a slope is, and how to rearrange <M tex="y = mx + c" />.
        That is the floor, and nothing below it is assumed.
      </p>
      <p>
        Everything else gets built here, in the order the story needs it and never
        before. The dot product arrives in Module 1 after multiplying and adding by
        hand. Matrix times column arrives in Module 2 after doing that twice. A
        slope arrives in Module 3 by nudging a knob and measuring what the cost
        does. The chain rule arrives in Module 4 as a chain of posted exchange
        rates, which is the one genuinely new idea in the course and gets a module
        to itself with no code in it. NumPy is taught the same way: one tool at a
        time, at the moment it is first needed.
      </p>
      <p>
        You will not meet the words derivative or calculus until Module 9, where a
        table translates what you have been doing into what everyone else calls it.
        That is deliberate. The names are worth having at the end, when there is
        something for them to name, and they are an obstacle at the start.
      </p>
      <div className="start-cta">
        <button onClick={() => onGoTo("m1")}>Begin Module 1 →</button>
        {passed.length > 0 && (
          <span className="start-cta-note">
            {passed.length} of {EXERCISES.length} exercises passed in this browser.
          </span>
        )}
      </div>

      <details className="notation">
        <summary>Notation and NumPy reference: every symbol, and where it was introduced</summary>
        <p className="notation-note">
          The course defines each of these once, at the moment it first matters.
          This is the lookup for when weeks have passed since then.
        </p>
        <div className="table-scroll scroll-x" tabIndex={0}>
          <table className="truth-table">
            <thead>
              <tr>
                <th>you will see</th>
                <th>it means</th>
                <th>from</th>
              </tr>
            </thead>
            <tbody>
              {NOTATION.map((row) => (
                <tr key={row.id}>
                  <td className="notation-symbol">{row.symbol}</td>
                  <td>{row.means}</td>
                  <td>{row.from}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

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

      <h3 id="start-modules">The ten modules</h3>
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
                {done && !codeReady(e.id) && (
                  <span className="start-progress-warn">
                    Passed, but its code is not stored here, so panels that run it stay
                    locked.
                  </span>
                )}
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
