import { useEffect, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { crossEntropyExercise } from "../../../exercises/cross-entropy";
import { smartInitExercise } from "../../../exercises/smart-init";
import { sgdExercise } from "../../../exercises/sgd";
import { EpochChart, type EpochSeries } from "./EpochChart";
import { lockedBy, speakList } from "./lockedBy";

// Module 8's payoff run: the same digit reader at two depths under each
// squash, trained by the learner's own sgd from their own starting point, so
// the collapse the bars predicted shows up as epochs that go nowhere, with
// the repair drawn beside it. One button runs the whole experiment: the panel
// used to offer the squash as chips and train one pair per press, and the
// four-way comparison the section makes lived in the reader's memory.

const EPOCHS = 15;
const DEEP = 4;
// Each squash gets the step size that suits it, the same way each cost did in
// Module 7. Found by trying, on the shallow network, at fifteen epochs.
const ETA: Record<string, number> = { sigmoid: 0.5, relu: 0.05 };

// The fixed experiment, in chart and table order: the sigmoid pair, then the
// ReLU pair. Hue carries the squash, dash pattern carries the depth. For the
// sigmoid runs the gradient is the learner's own backprop; for the ReLU runs
// it is the course's two-line variant of it, written into the snippet.
const RUNS: { activation: "sigmoid" | "relu"; hidden: number; eta: number }[] = [
  { activation: "sigmoid", hidden: 1, eta: ETA.sigmoid },
  { activation: "sigmoid", hidden: DEEP, eta: ETA.sigmoid },
  { activation: "relu", hidden: 1, eta: ETA.relu },
  { activation: "relu", hidden: DEEP, eta: ETA.relu },
];

const runKey = (r: { activation: string; hidden: number }) => `${r.activation}|${r.hidden}`;
const squashName = (a: string) => (a === "relu" ? "ReLU" : "sigmoid");
const depthName = (h: number) => (h === 1 ? "1 hidden layer" : `${h} hidden layers`);
const runName = (r: { activation: string; hidden: number }) =>
  `${squashName(r.activation)}, ${depthName(r.hidden)}`;
const lineCls = (r: { activation: string; hidden: number }) =>
  (r.activation === "sigmoid" ? "m7-line-a" : "m7-line-b") +
  (r.hidden === 1 ? " m7-line-dashed" : "");

// Each run draws its start and its shuffle from fresh fixed seeds, so the
// four runs are the same runs however they are grouped, and the numbers the
// module quotes from bench_depth.py hold.
const SNIPPET = `
import json, time, types
import numpy as np

_a = json.loads(_args_json)

# The learner's file, once, up to and including their better starting point.
# For the sigmoid runs, every gradient below is computed by their own
# backprop, through the adapter written for them in Module 5.
_lib = types.ModuleType("your_code")
exec(compile(_a["code"], "your_code.py", "exec"), _lib.__dict__)
feedforward = _lib.feedforward
sigmoid = _lib.sigmoid

with open("/mnist_subset.bin", "rb") as _f:
    X_train, y_train, X_test, y_test = load_mnist_subset(_f.read())
Y_train = one_hot(y_train)

def _relu_backprop(weights, biases, x, y):
    """The same four equations with ReLU in the hidden layers, for one example.

    Identical to the backprop in their file except for the squash and its
    slope, and written out here because theirs has sigmoid_prime in it: BP2
    multiplies by 1 where a hidden neuron's evidence is positive and by 0
    where it is not. The output layer stays a sigmoid, since the
    cross-entropy blame a - y is written for one."""
    L = len(weights)
    a = x
    activations = [x]
    zs = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        z = w @ a + b
        zs.append(z)
        a = sigmoid(z) if i == L - 1 else np.maximum(0.0, z)
        activations.append(a)
    nabla_w = [np.zeros_like(w) for w in weights]
    nabla_b = [np.zeros_like(b) for b in biases]
    delta = activations[-1] - y                       # BP1, cross-entropy
    nabla_b[-1] = delta
    nabla_w[-1] = delta @ activations[-2].T
    for l in range(2, L + 1):
        delta = (weights[-l + 1].T @ delta) * (zs[-l] > 0)   # BP2, ReLU's slope
        nabla_b[-l] = delta
        nabla_w[-l] = delta @ activations[-l - 1].T
    return nabla_w, nabla_b

def _relu_feedforward(weights, biases, X):
    a = X
    L = len(weights)
    for i, (w, b) in enumerate(zip(weights, biases)):
        z = w @ a + b
        a = sigmoid(z) if i == L - 1 else np.maximum(0.0, z)
    return a

def _grad(weights, biases, X, Y):
    if _act == "relu":
        m = X.shape[1]
        nw = [np.zeros_like(w) for w in weights]
        nb = [np.zeros_like(b) for b in biases]
        for k in range(m):
            dw, db = _relu_backprop(weights, biases, X[:, k:k + 1], Y[:, k:k + 1])
            nw = [t + d for t, d in zip(nw, dw)]
            nb = [t + d for t, d in zip(nb, db)]
        return [t / m for t in nw], [t / m for t in nb]
    return _lib.batch_gradient(weights, biases, X, Y, _lib.cross_entropy_delta)

# Their sgd, walking on whichever gradient the run's squash calls for: _grad
# reads _act, which the loop below sets per run.
_lib.gradient = _grad

def _layer_speeds(weights, biases, n=100):
    """Every layer's ||dC/db|| before the first step, from the same gradient."""
    nw, nb = _grad(weights, biases, X_train[:, :n], Y_train[:, :n])
    return [float(np.linalg.norm(g)) for g in nb]

_out = {}
for _i, _cfg in enumerate(_a["runs"]):
    _act = _cfg["activation"]
    _eta = float(_cfg["eta"])
    _hidden = int(_cfg["hidden"])
    _key = _act + "|" + str(_hidden)
    _sizes = [784] + [30] * _hidden + [10]
    _predict = _relu_feedforward if _act == "relu" else feedforward
    weights, biases = _lib.init_network(_sizes, np.random.default_rng(8))
    _js_report(json.dumps({"kind": "start", "key": _key, "run_index": _i,
                           "activation": _act, "hidden": _hidden,
                           "sizes": _sizes, "speeds": _layer_speeds(weights, biases)}))
    _rng = np.random.default_rng(2)
    _t0 = time.time()
    for _e in range(1, ${EPOCHS} + 1):
        weights, biases = _lib.sgd(weights, biases, X_train, Y_train, _eta, 1, 10, _rng)
        _acc = float((np.argmax(_predict(weights, biases, X_test), axis=0) == y_test).mean())
        _js_report(json.dumps({"kind": "epoch", "key": _key, "run_index": _i,
                               "activation": _act, "hidden": _hidden,
                               "epoch": _e, "accuracy": _acc,
                               "elapsed": time.time() - _t0}))
    _out[_key] = {"activation": _act, "hidden": _hidden, "accuracy": _acc,
                  "seconds": time.time() - _t0,
                  "speeds_end": _layer_speeds(weights, biases)}

json.dumps({"runs": _out, "n_test": int(X_test.shape[1])})
`;

interface StartReport {
  kind: "start";
  key: string;
  run_index: number;
  activation: string;
  hidden: number;
  sizes: number[];
  speeds: number[];
}

interface EpochReport {
  kind: "epoch";
  key: string;
  run_index: number;
  activation: string;
  hidden: number;
  epoch: number;
  accuracy: number;
  elapsed: number;
}

interface RunSummary {
  activation: string;
  hidden: number;
  accuracy: number;
  seconds: number;
  speeds_end: number[];
}

interface Summary {
  runs: Record<string, RunSummary>;
  n_test: number;
}

const needed = () =>
  codeReady(crossEntropyExercise.id) &&
  codeReady(smartInitExercise.id) &&
  codeReady(sgdExercise.id);

export function DepthTrainPanel() {
  const [unlocked, setUnlocked] = useState(needed);
  const [starts, setStarts] = useState<StartReport[]>([]);
  const [points, setPoints] = useState<EpochReport[]>([]);
  const [status, setStatus] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => subscribeProgress(() => setUnlocked(needed())), []);

  const run = () => {
    // One projection: the file through the starting-point section already
    // holds their sgd, their backprop and their cross-entropy blame.
    const code = loadCode(smartInitExercise.id);
    if (!code) return;
    setRunning(true);
    setStarts([]);
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
          const r = msg.payload as StartReport | EpochReport;
          const name = `run ${r.run_index + 1} of ${RUNS.length} (${runName(r)})`;
          if (r.kind === "start") {
            setStarts((prev) => [...prev, r]);
            setStatus(`${name}: measuring every layer before the first step...`);
          } else {
            setPoints((prev) => [...prev, r]);
            setStatus(
              `${name}: epoch ${r.epoch}/${EPOCHS}, ` +
                `${(r.accuracy * 100).toFixed(1)}% of the test digits, ${r.elapsed.toFixed(1)}s`,
            );
          }
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
      lockedBy([crossEntropyExercise.id, smartInitExercise.id, sgdExercise.id]),
    );
    return (
      <p className="payoff-locked">
        This experiment needs {missing}: it builds all four networks with your
        init_network and trains them with your sgd, with nothing borrowed.
      </p>
    );
  }

  const series: EpochSeries[] = RUNS.map((r) => ({
    key: runKey(r),
    label: runName(r),
    cls: lineCls(r),
    values: points.filter((p) => p.key === runKey(r)).map((p) => p.accuracy),
  })).filter((s) => s.values.length > 0);

  // The pre-run table shows the deep networks only: the shallow starts have
  // two layers to report, and the imbalance the section is about needs five.
  const deepStarts = starts.filter((s) => s.hidden === DEEP);
  const ratio = (speeds: number[]) => speeds[speeds.length - 1] / speeds[0];
  const fmtRatio = (x: number) => (x < 10 ? x.toFixed(1) : x.toFixed(0));
  const fmtSpeed = (s: number) => (s < 0.01 ? s.toExponential(2) : s.toFixed(4));
  const sigDeep = summary?.runs[`sigmoid|${DEEP}`];
  const sigDeepStart = starts.find((s) => s.key === `sigmoid|${DEEP}`);
  const reluDeepStart = starts.find((s) => s.key === `relu|${DEEP}`);

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : summary ? "Run all four again" : "Train all four runs"}
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
        <p className="m8-eta-note">
          {EPOCHS} epochs each, 5,000 images, mini-batches of 10; step size{" "}
          {ETA.sigmoid} for the sigmoid runs and {ETA.relu} for ReLU
        </p>
      </div>
      {deepStarts.length > 0 && (
        <div className="table-scroll scroll-x" tabIndex={0}>
          <table className="truth-table">
            <thead>
              <tr>
                <th>{deepStarts[0].sizes.join("-")}, before a single step</th>
                {deepStarts[0].speeds.map((_, i) => (
                  <th key={i}>layer {i + 2}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {deepStarts.map((s) => (
                <tr key={s.key}>
                  <td>
                    {s.activation === "relu"
                      ? "ReLU, from the course's gradient"
                      : "sigmoid, from your gradient"}
                  </td>
                  {s.speeds.map((v, i) => (
                    <td key={i}>{fmtSpeed(v)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
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
          yLabel="test accuracy"
          xLabel="epoch (one full pass through the 5,000 training images)"
          ariaLabel="Test accuracy per epoch for four runs: one hidden layer and four hidden layers, under the sigmoid and under ReLU."
        />
      )}
      {summary && (
        <div className="interactive-status">
          {sigDeep && sigDeepStart && (
            <p>
              The deep sigmoid network's layer speeds have levelled out by the
              end: its first hidden layer finishes at{" "}
              {fmtSpeed(sigDeep.speeds_end[0])} against{" "}
              {fmtSpeed(sigDeep.speeds_end[sigDeep.speeds_end.length - 1])} at the
              output, where the table above the chart started them{" "}
              <b>{fmtRatio(ratio(sigDeepStart.speeds))}</b> times apart. The
              imbalance is a condition of the starting point, and the epochs that
              run spends working out of it are what the flat start of its line is
              showing.
              {reluDeepStart && (
                <>
                  {" "}
                  The ReLU start is {fmtRatio(ratio(reluDeepStart.speeds))} times
                  apart, and its deep line has no such flat start to work out of.
                </>
              )}
            </p>
          )}
          <div className="table-scroll scroll-x" tabIndex={0}>
            <table className="truth-table">
              <thead>
                <tr>
                  <th>after {EPOCHS} epochs</th>
                  <th>step size</th>
                  <th>the {summary.n_test.toLocaleString()} held out</th>
                  <th>first hidden layer's learning speed at the end</th>
                </tr>
              </thead>
              <tbody>
                {RUNS.map((r) => {
                  const s = summary.runs[runKey(r)];
                  if (!s) return null;
                  return (
                    <tr key={runKey(r)}>
                      <td>{runName(r)}</td>
                      <td>{r.eta}</td>
                      <td>{(s.accuracy * 100).toFixed(1)}%</td>
                      <td>{fmtSpeed(s.speeds_end[0])}</td>
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
