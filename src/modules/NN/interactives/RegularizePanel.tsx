import { useEffect, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { crossEntropyExercise } from "../../../exercises/cross-entropy";
import { smartInitExercise } from "../../../exercises/smart-init";
import { l2Exercise } from "../../../exercises/l2";
import { EpochChart, type EpochSeries } from "./EpochChart";

// Module 7, third cycle: the 1,000-image slice, trained twice through the
// learner's l2_step, once with lmbda = 0 (which is exactly their Module 3
// update) and once with the lmbda they pick. The starting point is a control,
// because the two starts give this comparison two different answers.

const EPOCHS = 80;
const LAMBDAS = [0.5, 1, 2, 5];

// The loop is their sgd's loop written out, so its update line can be theirs
// from this module's exercise. Shuffle order matches the sgd contract:
// rng.permutation, then consecutive slices of batch_size.
const SNIPPET = `
import json, time, types
import numpy as np

_a = json.loads(_args_json)

# The learner's file, once, up to and including their decaying step. The
# epoch loop below stays written out here rather than calling their sgd,
# deliberately: this panel varies lambda and the module quotes its numbers.
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

def _start():
    if _a["start"] == "yours":
        return _lib.init_network([784, 30, 10], np.random.default_rng(8))
    r = np.random.default_rng(8)
    return ([r.standard_normal((30, 784)), r.standard_normal((10, 30))],
            [r.standard_normal((30, 1)), r.standard_normal((10, 1))])

def _accuracy(w, b, X, y):
    return float((np.argmax(feedforward(w, b, X), axis=0) == y).mean())

_out = {}
for _key, _lmbda in (("plain", 0.0), ("l2", float(_a["lmbda"]))):
    weights, biases = _start()
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
            "key": _key, "lmbda": _lmbda, "epoch": _e,
            "train_accuracy": _accuracy(weights, biases, X_train, y_train),
            "test_accuracy": _accuracy(weights, biases, X_test, y_test),
            "test_cost": float(_lib.cross_entropy_cost(weights, biases, X_test, Y_test)),
            "elapsed": time.time() - _t0,
        }))
    _out[_key] = {
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
  lmbda: number;
  epoch: number;
  train_accuracy: number;
  test_accuracy: number;
  test_cost: number;
  elapsed: number;
}

interface RunSummary {
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
  const [start, setStart] = useState<"yours" | "plain">("yours");
  const [lmbda, setLmbda] = useState(1);
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
        args: { code, start, lmbda },
        dataUrl: assetUrl("data/mnist_subset.bin.gz"),
      },
      (msg) => {
        if (msg.type === "status") setStatus(msg.text);
        if (msg.type === "report") {
          const r = msg.payload as Report;
          setPoints((prev) => [...prev, r]);
          setStatus(
            `${r.lmbda === 0 ? "no regularization" : `lambda ${r.lmbda}`}: epoch ` +
              `${r.epoch}/${EPOCHS}, ${(r.train_accuracy * 100).toFixed(1)}% of the ` +
              `training images and ${(r.test_accuracy * 100).toFixed(1)}% of the held-out ` +
              `ones, ${r.elapsed.toFixed(0)}s`,
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
    const missing = [
      codeReady(crossEntropyExercise.id) ? null : "the cross-entropy exercise",
      codeReady(smartInitExercise.id) ? null : "the starting-point exercise",
      codeReady(l2Exercise.id) ? null : "the decaying step above",
    ].filter(Boolean);
    return (
      <p className="payoff-locked">
        Both runs go through your l2_step, so this needs {missing.join(", ")} passed first.
      </p>
    );
  }

  const pick = (key: string) => points.filter((p) => p.key === key);
  // The lambda on the chart is the one the drawn run used, not the one now
  // selected: picking a different chip before rerunning must not relabel a
  // finished run.
  const drawn = pick("l2")[0]?.lmbda ?? lmbda;
  const accuracySeries: EpochSeries[] = [
    { key: "plain-train", label: "no regularization, the training images", cls: "m7-line-a", values: pick("plain").map((p) => p.train_accuracy) },
    { key: "plain-test", label: "no regularization, the held-out images", cls: "m7-line-a", dashed: true, values: pick("plain").map((p) => p.test_accuracy) },
    { key: "l2-train", label: `lambda ${drawn}, the training images`, cls: "m7-line-b", values: pick("l2").map((p) => p.train_accuracy) },
    { key: "l2-test", label: `lambda ${drawn}, the held-out images`, cls: "m7-line-b", dashed: true, values: pick("l2").map((p) => p.test_accuracy) },
  ].filter((s) => s.values.length > 0);
  const costSeries: EpochSeries[] = [
    { key: "plain-cost", label: "no regularization", cls: "m7-line-a", values: pick("plain").map((p) => p.test_cost) },
    { key: "l2-cost", label: `lambda ${drawn}`, cls: "m7-line-b", values: pick("l2").map((p) => p.test_cost) },
  ].filter((s) => s.values.length > 0);
  const costMax = Math.max(2, ...costSeries.flatMap((s) => s.values));

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : summary ? "Run both again" : "Train both ways"}
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
          <legend>starting point</legend>
          <button
            className={`chip ${start === "yours" ? "chip-active" : ""}`}
            onClick={() => setStart("yours")}
            disabled={running}
          >
            your start
          </button>
          <button
            className={`chip ${start === "plain" ? "chip-active" : ""}`}
            onClick={() => setStart("plain")}
            disabled={running}
          >
            Module 5's start
          </button>
        </fieldset>
        <fieldset className="m7-choice">
          <legend>lambda for the second run</legend>
          {LAMBDAS.map((v) => (
            <button
              key={v}
              className={`chip ${lmbda === v ? "chip-active" : ""}`}
              onClick={() => setLmbda(v)}
              disabled={running}
            >
              {v}
            </button>
          ))}
        </fieldset>
        <fieldset className="m7-choice">
          <legend>show</legend>
          <button
            className={`chip ${view === "accuracy" ? "chip-active" : ""}`}
            onClick={() => setView("accuracy")}
          >
            accuracy
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
          yLabel="accuracy"
          xLabel="epoch (one full pass through the 1,000 training images)"
          ariaLabel="Accuracy per epoch on the training images and on the held-out images, for a run with no regularization and a run with the chosen lambda."
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
          ariaLabel="Cross-entropy cost on the held-out digits per epoch, for a run with no regularization and a run with the chosen lambda."
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
                {[
                  ["no regularization", summary.runs["plain"]],
                  [`lambda ${summary.runs["l2"].lmbda}`, summary.runs["l2"]],
                ].map(([label, r]) => {
                  const s = r as RunSummary;
                  return (
                    <tr key={label as string}>
                      <td>{label as string}</td>
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
