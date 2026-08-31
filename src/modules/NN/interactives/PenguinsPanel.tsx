import { useEffect, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { prepareExercise } from "../../../exercises/prepare";
import { trainExercise } from "../../../exercises/train";
import { EpochChart, type EpochSeries } from "./EpochChart";
import { lockedBy, speakList } from "./lockedBy";

// Chapter 10's payoff: the learner's own preparation and their own training
// loop, on a file that arrives the way data actually does. Two switches,
// because the chapter's two claims are both comparisons: scaling against not,
// and four measurements against two.

const EPOCHS = 30;
const HIDDEN = 8;

const SNIPPET = `
import json, types
import numpy as np
import course

_a = json.loads(_args_json)

# Their whole file, once. _prep and _prog are the same module: their
# standardize, one_hot and split sit below their train in it, and train calls
# the pieces above it. So "your train from Chapter 9 does the rest" is now
# literally what happens.
#
# The one_hot collision is harmless and worth naming: the worker's loader has
# a one_hot(y, num_classes=10) for MNIST, and the learner's Chapter 10
# one_hot(values, levels) is a different function with the same name. They
# never meet, because the file execs into its own module and this panel builds
# Y itself. Do not inject the worker's globals into that namespace.
_prep = types.ModuleType("your_code")
exec(compile(_a["code"], "your_code.py", "exec"), _prep.__dict__)
_prog = _prep

with open("/penguins.json", "rb") as _f:
    columns, rows = load_penguins(_f.read())

ALL_FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
ISLANDS = ["Biscoe", "Dream", "Torgersen"]
SEXES = ["female", "male"]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]

_index = {name: i for i, name in enumerate(columns)}
_features = ALL_FOUR if _a["features"] == "all" else ["bill_depth_mm", "body_mass_g"]

# A row with no measurements cannot be fed to anything, so it goes. A row with
# no sex keeps its place: one_hot gives it a column of zeros.
_usable = [r for r in rows if all(r[_index[f]] is not None for f in ALL_FOUR)]

_numbers = np.array([[float(r[_index[f]]) for f in _features] for r in _usable]).T
_train_idx, _val_idx, _test_idx = _prep.split(len(_usable), np.random.default_rng(0), 0.2, 0.2)

if _a["scale"]:
    _, _mean, _spread = _prep.standardize(_numbers[:, _train_idx])
    _numbers, _, _ = _prep.standardize(_numbers, _mean, _spread)

_island = _prep.one_hot([r[_index["island"]] for r in _usable], ISLANDS)
_sex = _prep.one_hot([r[_index["sex"]] for r in _usable], SEXES)
X = np.vstack([_numbers, _island, _sex])

_y = np.array([SPECIES.index(r[_index["species"]]) for r in _usable])
Y = np.zeros((len(SPECIES), len(_usable)))
Y[_y, np.arange(len(_usable))] = 1.0

_counts = np.bincount(_y[_train_idx], minlength=len(SPECIES))
_major = int(np.argmax(_counts))
_baseline = float((_y[_test_idx] == _major).mean())

_scored = getattr(_prog, "accuracy", None)
_epoch = 0

def _reporting_accuracy(weights, biases, Xa, ya):
    global _epoch
    value = _scored(weights, biases, Xa, ya)
    _epoch += 1
    _js_report(json.dumps({"epoch": _epoch, "accuracy": float(value)}))
    return value

if _scored is not None:
    _prog.accuracy = _reporting_accuracy

_weights, _biases, _history = _prog.train(
    [X.shape[0], ${HIDDEN}, len(SPECIES)],
    X[:, _train_idx], Y[:, _train_idx], X[:, _val_idx], _y[_val_idx],
    ${EPOCHS}, 0.5, 0.0, 10, np.random.default_rng(1))

# Scored here through the course's own feedforward, so the report does not
# depend on the function that produced the history.
_out = course.feedforward(_weights, _biases, X[:, _test_idx])
_guesses = np.argmax(_out, axis=0)
_truth = _y[_test_idx]

_per_class = []
for _i, _name in enumerate(SPECIES):
    _mask = _truth == _i
    _per_class.append({"species": _name, "total": int(_mask.sum()),
                       "right": int((_guesses[_mask] == _i).sum())})

_confusions = {}
for _t, _g in zip(_truth.tolist(), _guesses.tolist()):
    if _t != _g:
        _key = SPECIES[_t] + " read as " + SPECIES[_g]
        _confusions[_key] = _confusions.get(_key, 0) + 1

json.dumps({
    "rows": len(rows), "usable": len(_usable), "inputs": int(X.shape[0]),
    "train": len(_train_idx), "val": len(_val_idx), "test_rows": len(_test_idx),
    "baseline": _baseline, "majority": SPECIES[_major],
    "validation": float(_history[-1]), "first_epoch": float(_history[0]),
    "test": float((_guesses == _truth).mean()),
    "per_class": _per_class,
    "confusions": sorted(_confusions.items(), key=lambda kv: -kv[1]),
})
`;

interface Summary {
  rows: number;
  usable: number;
  inputs: number;
  train: number;
  val: number;
  test_rows: number;
  /** Accuracy on the held-back rows, scored once at the end. */
  test: number;
  baseline: number;
  majority: string;
  validation: number;
  first_epoch: number;
  per_class: { species: string; total: number; right: number }[];
  confusions: [string, number][];
}

const needed = () => codeReady(prepareExercise.id) && codeReady(trainExercise.id);

export function PenguinsPanel() {
  const [unlocked, setUnlocked] = useState(needed);
  const [scale, setScale] = useState(true);
  const [features, setFeatures] = useState<"all" | "two">("all");
  const [ran, setRan] = useState<{ scale: boolean; features: string } | null>(null);
  const [points, setPoints] = useState<number[]>([]);
  const [status, setStatus] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => subscribeProgress(() => setUnlocked(needed())), []);

  const run = () => {
    // One projection: the file through the preparation section holds their
    // program too, since Chapter 9 comes before Chapter 10.
    const code = loadCode(prepareExercise.id);
    if (!code) return;
    setRunning(true);
    setRan({ scale, features });
    setPoints([]);
    setSummary(null);
    setError(null);
    setStatus("Starting...");
    sendRequest(
      {
        type: "runPython",
        code: SNIPPET,
        args: { code, scale, features },
        dataUrl: assetUrl("data/penguins.json.gz"),
      },
      (msg) => {
        if (msg.type === "status") setStatus(msg.text);
        if (msg.type === "report") {
          const r = msg.payload as { epoch: number; accuracy: number };
          setPoints((prev) => [...prev, r.accuracy]);
          setStatus(`epoch ${r.epoch}/${EPOCHS}, ${(r.accuracy * 100).toFixed(1)}% of the validation penguins`);
        }
        if (msg.type === "pythonDone") {
          setSummary(msg.result as Summary);
          setRunning(false);
          setStatus("");
        }
        if (msg.type === "cancelled") {
          setRunning(false);
          setStatus("Stopped.");
        }
        if (msg.type === "error") {
          setError(msg.message);
          setRunning(false);
          setStatus("");
        }
      },
    );
  };

  if (!unlocked) {
    const missing = speakList(
      lockedBy([prepareExercise.id, trainExercise.id], {
        [prepareExercise.id]: "this chapter's exercise",
      }),
    );
    return (
      <p className="payoff-locked">
        This run prepares the file with your own code and trains it with your own
        loop, nothing borrowed, so it needs {missing}.
      </p>
    );
  }

  const series: EpochSeries[] = points.length
    ? [{ key: "val", label: "validation accuracy", cls: "m7-line-b", values: points }]
    : [];

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : summary ? "Run it again" : "Prepare and train"}
        </button>
        {running && (
          <button className="button-secondary" onClick={terminateWorker}>
            Stop
          </button>
        )}
        <span className={`demo-status status-fixed ${error ? "demo-status-error" : ""}`}>
          {error ?? status}
        </span>
      </div>
      <div className="interactive-controls">
        <fieldset className="m7-choice">
          <legend>the measurements</legend>
          <button
            className={`chip ${scale ? "chip-active" : ""}`}
            onClick={() => setScale(true)}
            disabled={running}
          >
            scaled
          </button>
          <button
            className={`chip ${!scale ? "chip-active" : ""}`}
            onClick={() => setScale(false)}
            disabled={running}
          >
            as they come
          </button>
        </fieldset>
        <fieldset className="m7-choice">
          <legend>which ones</legend>
          <button
            className={`chip ${features === "all" ? "chip-active" : ""}`}
            onClick={() => setFeatures("all")}
            disabled={running}
          >
            all four
          </button>
          <button
            className={`chip ${features === "two" ? "chip-active" : ""}`}
            onClick={() => setFeatures("two")}
            disabled={running}
          >
            bill depth and mass
          </button>
        </fieldset>
      </div>
      {series.length > 0 && (
        <EpochChart
          series={series}
          epochs={EPOCHS}
          yMin={0}
          yMax={1}
          yTicks={[
            { at: 0, label: "0%" },
            { at: 0.25, label: "25%" },
            { at: 0.5, label: "50%" },
            { at: 0.75, label: "75%" },
            { at: 1, label: "100%" },
          ]}
          yLabel="validation accuracy"
          xLabel="epoch (one pass through the training penguins)"
          ariaLabel="Validation accuracy per epoch while training on the penguin measurements."
        />
      )}
      {summary && ran && (
        <div className="interactive-status">
          <p>
            {summary.usable} penguins with all four measurements out of{" "}
            {summary.rows} in the file, cut into {summary.train} for training,{" "}
            {summary.val} for validation and {summary.test_rows} held back. {summary.inputs} input rows: the{" "}
            {ran.features === "all" ? "four measurements" : "two measurements"} plus
            three islands and two sexes.
          </p>
          <p>
            Answering <b>{summary.majority}</b> every time, the commonest species in
            the training rows, would read{" "}
            <b>{(summary.baseline * 100).toFixed(1)}%</b> of the held-back penguins.
            This network, with the measurements{" "}
            {ran.scale ? "scaled" : "left as they come"}, reads{" "}
            <b>{(summary.test * 100).toFixed(1)}%</b>.
            {!ran.scale && summary.test <= summary.baseline + 0.005 && (
              <>
                {" "}
                Those two numbers are the same, which is the finding: it has learned
                which species is commonest and nothing else.
              </>
            )}
          </p>
          <div className="table-scroll scroll-x" tabIndex={0}>
            <table className="truth-table">
              <thead>
                <tr>
                  <th>species</th>
                  {summary.per_class.map((c) => (
                    <th key={c.species}>{c.species}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>read correctly</td>
                  {summary.per_class.map((c) => (
                    <td key={c.species}>
                      {c.right} of {c.total}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          {summary.confusions.length > 0 && (
            <p>
              What it got wrong:{" "}
              {summary.confusions.map(([label, count], i) => (
                <span key={label}>
                  {i > 0 && "; "}
                  {count} {label}
                </span>
              ))}
              .
            </p>
          )}
        </div>
      )}
    </div>
  );
}
