import { useEffect, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { crossEntropyExercise } from "../../../exercises/cross-entropy";
import { smartInitExercise } from "../../../exercises/smart-init";
import { l2Exercise } from "../../../exercises/l2";
import { EpochChart, type EpochSeries } from "./EpochChart";
import { lockedBy, speakList } from "./lockedBy";

// Chapter 7, third cycle: the 1,000-image slice, trained through the learner's
// l2_step. One button runs the whole experiment: both starting points, each
// at lambda 0 (which is exactly their Chapter 3 update), 1 and 5. The panel
// used to offer the settings as chips and train one pair per press, and the
// comparison the section makes (decay against none, from each start) lived in
// the reader's memory; now every line and every row is on screen at once.

const EPOCHS = 80;

// The fixed experiment, in chart and table order: the divided start's
// three runs, then the undivided start's. Hue carries the start, dash pattern carries the lambda.
const RUNS: { start: "yours" | "plain"; lmbda: number }[] = [
  { start: "yours", lmbda: 0 },
  { start: "yours", lmbda: 1 },
  { start: "yours", lmbda: 5 },
  { start: "plain", lmbda: 0 },
  { start: "plain", lmbda: 1 },
  { start: "plain", lmbda: 5 },
];

const runKey = (r: { start: string; lmbda: number }) => `${r.start}|${r.lmbda}`;
const startName = (v: "yours" | "plain") =>
  v === "yours" ? "the divided start" : "the undivided start";
const lambdaName = (l: number) => (l === 0 ? "no regularization" : `lambda ${l}`);
const lineCls = (r: { start: string; lmbda: number }) =>
  (r.start === "yours" ? "m7-line-a" : "m7-line-b") +
  (r.lmbda === 1 ? " m7-line-dashed" : r.lmbda === 5 ? " m7-line-dotted" : "");

// The loop is their sgd's loop written out, so its update line can be theirs
// from this chapter's exercise. Shuffle order matches the sgd contract:
// rng.permutation, then consecutive slices of batch_size. Each run draws its
// start and its shuffle from fresh fixed seeds, so the six runs are the same
// runs however they are grouped, and the chapter's quoted numbers hold.
const SNIPPET = `
import json, time, types
import numpy as np

_a = json.loads(_args_json)

# The learner's file, once, up to and including their decaying step. The
# epoch loop below stays written out here rather than calling their sgd,
# deliberately: this panel varies lambda and the chapter quotes its numbers.
_lib = types.ModuleType("your_code")
exec(compile(_a["code"], "your_code.py", "exec"), _lib.__dict__)
feedforward = _lib.feedforward
batch_gradient = _lib.batch_gradient

with open("/mnist_subset.bin", "rb") as _f:
    X_full, y_full, X_test, y_test = load_mnist_subset(_f.read())
_slice = ${1000}
X_train, y_train = X_full[:, :_slice], y_full[:_slice]
Y_train, Y_test = one_hot(y_train), one_hot(y_test)
_n = X_train.shape[1]

def _start(which):
    if which == "yours":
        return _lib.init_network([784, 30, 10], np.random.default_rng(8))
    r = np.random.default_rng(8)
    return ([r.standard_normal((30, 784)), r.standard_normal((10, 30))],
            [r.standard_normal((30, 1)), r.standard_normal((10, 1))])

def _accuracy(w, b, X, y):
    return float((np.argmax(feedforward(w, b, X), axis=0) == y).mean())

_out = {}
for _i, _cfg in enumerate(_a["runs"]):
    _key = f"{_cfg['start']}|{_cfg['lmbda']}"
    _lmbda = float(_cfg["lmbda"])
    weights, biases = _start(_cfg["start"])
    _rng = np.random.default_rng(2)
    _t0 = time.time()
    for _e in range(1, ${EPOCHS} + 1):
        _order = _rng.permutation(_n)
        for _k in range(0, _n, 10):
            _batch = _order[_k:_k + 10]
            _nw, _nb = batch_gradient(weights, biases, X_train[:, _batch],
                                      Y_train[:, _batch], _lib.cross_entropy_delta)
            weights, biases = _lib.l2_step(weights, biases, _nw, _nb, 0.5, _lmbda, _n)
        _js_report(json.dumps({
            "key": _key, "start": _cfg["start"], "lmbda": _lmbda,
            "run_index": _i, "epoch": _e,
            "train_accuracy": _accuracy(weights, biases, X_train, y_train),
            "test_accuracy": _accuracy(weights, biases, X_test, y_test),
            "test_cost": float(_lib.cross_entropy_cost(weights, biases, X_test, Y_test)),
            "elapsed": time.time() - _t0,
        }))
    _out[_key] = {
        "start": _cfg["start"],
        "lmbda": _lmbda,
        "train_accuracy": _accuracy(weights, biases, X_train, y_train),
        "test_accuracy": _accuracy(weights, biases, X_test, y_test),
        "test_cost": float(_lib.cross_entropy_cost(weights, biases, X_test, Y_test)),
        "weight_size": float(sum(float((w ** 2).sum()) for w in weights)),
        "seconds": time.time() - _t0,
    }

json.dumps({"runs": _out, "n_train": int(_n), "n_test": int(X_test.shape[1])})
`;

interface Report {
  key: string;
  start: "yours" | "plain";
  lmbda: number;
  run_index: number;
  epoch: number;
  train_accuracy: number;
  test_accuracy: number;
  test_cost: number;
  elapsed: number;
}

interface RunSummary {
  start: "yours" | "plain";
  lmbda: number;
  train_accuracy: number;
  test_accuracy: number;
  test_cost: number;
  weight_size: number;
  seconds: number;
}

interface Summary {
  runs: Record<string, RunSummary>;
  n_train: number;
  n_test: number;
}

const needed = () =>
  codeReady(crossEntropyExercise.id) &&
  codeReady(smartInitExercise.id) &&
  codeReady(l2Exercise.id);

export function RegularizePanel() {
  const [unlocked, setUnlocked] = useState(needed);
  const [view, setView] = useState<"accuracy" | "cost">("accuracy");
  const [points, setPoints] = useState<Report[]>([]);
  const [status, setStatus] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => subscribeProgress(() => setUnlocked(needed())), []);

  const run = () => {
    // One projection: the file through the decaying step already holds their
    // cross-entropy work and their better starting point.
    const code = loadCode(l2Exercise.id);
    if (!code) return;
    setRunning(true);
    setPoints([]);
    setSummary(null);
    setError(null);
    setStatus("Starting...");
    sendRequest(
      {
        type: "runPython",
        code: SNIPPET,
        args: { code, runs: RUNS },
        dataUrl: assetUrl("data/mnist_subset.bin.gz"),
      },
      (msg) => {
        if (msg.type === "status") setStatus(msg.text);
        if (msg.type === "report") {
          const r = msg.payload as Report;
          setPoints((prev) => [...prev, r]);
          setStatus(
            `run ${r.run_index + 1} of ${RUNS.length} (${startName(r.start)}, ` +
              `${lambdaName(r.lmbda)}): epoch ${r.epoch}/${EPOCHS}, ` +
              `${(r.test_accuracy * 100).toFixed(1)}% of the held-out digits, ` +
              `${r.elapsed.toFixed(0)}s`,
          );
        }
        if (msg.type === "pythonDone") {
          setSummary(msg.result as Summary);
          setRunning(false);
          setStatus("");
        }
        if (msg.type === "cancelled") {
          setRunning(false);
          setStatus("Stopped.");
          return;
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
      lockedBy([crossEntropyExercise.id, smartInitExercise.id, l2Exercise.id], {
        [crossEntropyExercise.id]: "the cross-entropy exercise",
        [smartInitExercise.id]: "the starting-point exercise",
        [l2Exercise.id]: "the decaying step above",
      }),
    );
    return (
      <p className="payoff-locked">
        All six runs go through your l2_step, with nothing borrowed, so this needs {missing}.
      </p>
    );
  }

  const pick = (key: string) => points.filter((p) => p.key === key);
  const series = (of: (p: Report) => number): EpochSeries[] =>
    RUNS.map((r) => ({
      key: runKey(r),
      label: `${startName(r.start)}, ${lambdaName(r.lmbda)}`,
      cls: lineCls(r),
      values: pick(runKey(r)).map(of),
    })).filter((s) => s.values.length > 0);
  const accuracySeries = series((p) => p.test_accuracy);
  const costSeries = series((p) => p.test_cost);
  const costMax = Math.max(2, ...costSeries.flatMap((s) => s.values));

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : summary ? "Run all six again" : "Train all six runs"}
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
          <legend>show</legend>
          <button
            className={`chip ${view === "accuracy" ? "chip-active" : ""}`}
            onClick={() => setView("accuracy")}
          >
            accuracy on held-out digits
          </button>
          <button
            className={`chip ${view === "cost" ? "chip-active" : ""}`}
            onClick={() => setView("cost")}
          >
            cost on held-out digits
          </button>
        </fieldset>
      </div>
      {view === "accuracy" && accuracySeries.length > 0 && (
        <EpochChart
          series={accuracySeries}
          epochs={EPOCHS}
          yMin={0.4}
          yMax={1}
          yTicks={[
            { at: 0.4, label: "40%" },
            { at: 0.6, label: "60%" },
            { at: 0.8, label: "80%" },
            { at: 1, label: "100%" },
          ]}
          yLabel="accuracy on the held-out digits"
          xLabel="epoch (one full pass through the 1,000 training images)"
          ariaLabel="Held-out accuracy per epoch for six runs: both starting points, each with no regularization, lambda 1 and lambda 5."
        />
      )}
      {view === "cost" && costSeries.length > 0 && (
        <EpochChart
          series={costSeries}
          epochs={EPOCHS}
          yMax={costMax}
          yTicks={[
            { at: 0, label: "0" },
            { at: costMax / 2, label: (costMax / 2).toFixed(1) },
            { at: costMax, label: costMax.toFixed(1) },
          ]}
          yLabel="cross-entropy cost on the held-out digits"
          xLabel="epoch (one full pass through the 1,000 training images)"
          ariaLabel="Cross-entropy cost on the held-out digits per epoch for six runs: both starting points, each with no regularization, lambda 1 and lambda 5."
        />
      )}
      {summary && (
        <div className="interactive-status">
          <div className="table-scroll scroll-x" tabIndex={0}>
            <table className="truth-table">
              <thead>
                <tr>
                  <th>after {EPOCHS} epochs</th>
                  <th>the {summary.n_train.toLocaleString()} it trained on</th>
                  <th>the {summary.n_test.toLocaleString()} held out</th>
                  <th>held-out cost</th>
                  <th>total of every weight squared</th>
                </tr>
              </thead>
              <tbody>
                {RUNS.map((r) => {
                  const s = summary.runs[runKey(r)];
                  if (!s) return null;
                  return (
                    <tr key={runKey(r)}>
                      <td>{`${startName(r.start)}, ${lambdaName(r.lmbda)}`}</td>
                      <td>{(s.train_accuracy * 100).toFixed(1)}%</td>
                      <td>{(s.test_accuracy * 100).toFixed(1)}%</td>
                      <td>{s.test_cost.toFixed(2)}</td>
                      <td>{Math.round(s.weight_size).toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
