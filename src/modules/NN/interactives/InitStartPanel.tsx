import { useEffect, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { crossEntropyExercise } from "../../../exercises/cross-entropy";
import { smartInitExercise } from "../../../exercises/smart-init";
import { sgdExercise } from "../../../exercises/sgd";
import { EpochChart, type EpochSeries } from "./EpochChart";
import { lockedBy, speakList } from "./lockedBy";

// Module 7, second cycle: the same digit reader, the same cross-entropy cost,
// the same sgd, the same random draws, started twice. Once at Module 5's
// scale and once at yours, divided by the square root of each layer's input
// count. This is the module's payoff run.

const EPOCHS = 15;

const SNIPPET = `
import json, time, types
import numpy as np

_a = json.loads(_args_json)

# The learner's file, once, up to and including their better starting point.
# Their sgd, their backprop, their cross-entropy blame and their init_network
# all come out of this one exec.
_lib = types.ModuleType("your_code")
exec(compile(_a["code"], "your_code.py", "exec"), _lib.__dict__)
feedforward = _lib.feedforward

with open("/mnist_subset.bin", "rb") as _f:
    X_train, y_train, X_test, y_test = load_mnist_subset(_f.read())
Y_train = one_hot(y_train)

def _plain_start():
    r = np.random.default_rng(8)
    w = [r.standard_normal((30, 784)), r.standard_normal((10, 30))]
    b = [r.standard_normal((30, 1)), r.standard_normal((10, 1))]
    return w, b

def _your_start():
    return _lib.init_network([784, 30, 10], np.random.default_rng(8))

# Their sgd, with their cross-entropy blame behind it. The gradient is the
# same for both runs; only the starting parameters differ.
_lib.gradient = lambda w, b, X, Y: _lib.batch_gradient(
    w, b, X, Y, _lib.cross_entropy_delta)

def _saturation(w, b):
    """Share of hidden neurons flatter than 0.01, over the training images."""
    _a2 = 1.0 / (1.0 + np.exp(-(w[0] @ X_train + b[0])))
    _steep = _a2 * (1.0 - _a2)
    return float((_steep < 0.01).mean()), float(np.median(_steep))

_out = {}
for _key, _label, _make in (("plain", "Module 5's start", _plain_start),
                            ("yours", "your start", _your_start)):
    weights, biases = _make()
    _flat, _median = _saturation(weights, biases)
    _js_report(json.dumps({"kind": "start", "key": _key, "label": _label,
                           "flat_share": _flat, "median_steepness": _median}))
    _rng = np.random.default_rng(2)
    _t0 = time.time()
    for _e in range(1, ${EPOCHS} + 1):
        weights, biases = _lib.sgd(weights, biases, X_train, Y_train, 0.5, 1, 10, _rng)
        _acc = float((np.argmax(feedforward(weights, biases, X_test), axis=0) == y_test).mean())
        _js_report(json.dumps({"kind": "epoch", "key": _key, "label": _label,
                               "epoch": _e, "accuracy": _acc,
                               "elapsed": time.time() - _t0}))
    _flat_end, _median_end = _saturation(weights, biases)
    _out[_key] = {"accuracy": _acc, "seconds": time.time() - _t0,
                  "cost": float(_lib.cross_entropy_cost(weights, biases, X_train, Y_train)),
                  "flat_share_end": _flat_end}

json.dumps({"runs": _out, "n_test": int(X_test.shape[1])})
`;

interface StartReport {
  kind: "start";
  key: string;
  label: string;
  flat_share: number;
  median_steepness: number;
}

interface EpochReport {
  kind: "epoch";
  key: string;
  label: string;
  epoch: number;
  accuracy: number;
  elapsed: number;
}

interface Summary {
  runs: Record<string, { accuracy: number; seconds: number; cost: number; flat_share_end: number }>;
  n_test: number;
}

const LINES: Record<string, { label: string; cls: string; dashed?: boolean }> = {
  plain: { label: "Module 5's start (weights drawn at spread 1)", cls: "m7-line-a", dashed: true },
  yours: { label: "your start (each weight divided by the square root of its layer's inputs)", cls: "m7-line-b" },
};

const needed = () =>
  codeReady(crossEntropyExercise.id) &&
  codeReady(smartInitExercise.id) &&
  codeReady(sgdExercise.id);

export function InitStartPanel() {
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
        args: { code },
        dataUrl: assetUrl("data/mnist_subset.bin.gz"),
      },
      (msg) => {
        if (msg.type === "status") setStatus(msg.text);
        if (msg.type === "report") {
          const r = msg.payload as StartReport | EpochReport;
          if (r.kind === "start") {
            setStarts((prev) => [...prev, r]);
            setStatus(`${r.label}: measuring the hidden layer before training...`);
          } else {
            setPoints((prev) => [...prev, r]);
            setStatus(
              `${r.label}: epoch ${r.epoch}/${EPOCHS}, ` +
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
      lockedBy([crossEntropyExercise.id, smartInitExercise.id, sgdExercise.id], {
        [crossEntropyExercise.id]: "the cross-entropy exercise",
        [smartInitExercise.id]: "the starting-point exercise above",
      }),
    );
    return (
      <p className="payoff-locked">
        This run needs {missing}: it builds the network with your init_network and
        trains it with your cost inside your sgd, with nothing borrowed.
      </p>
    );
  }

  const series: EpochSeries[] = Object.keys(LINES)
    .map((key) => ({
      key,
      label: LINES[key].label,
      cls: LINES[key].cls,
      dashed: LINES[key].dashed,
      values: points.filter((p) => p.key === key).map((p) => p.accuracy),
    }))
    .filter((s) => s.values.length > 0);

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : summary ? "Run both again" : "Train from both starts"}
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
      {starts.length > 0 && (
        <div className="table-scroll scroll-x" tabIndex={0}>
          <table className="truth-table">
            <thead>
              <tr>
                <th>before a single step</th>
                <th>median hidden squash slope</th>
                <th>share of hidden neurons flatter than 0.01</th>
              </tr>
            </thead>
            <tbody>
              {starts.map((s) => (
                <tr key={s.key}>
                  <td>{s.label}</td>
                  <td>{s.median_steepness.toFixed(4)}</td>
                  <td>{(s.flat_share * 100).toFixed(1)}%</td>
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
          yMin={0.5}
          yMax={1}
          yTicks={[
            { at: 0.5, label: "50%" },
            { at: 0.75, label: "75%" },
            { at: 1, label: "100%" },
          ]}
          yLabel="test accuracy"
          xLabel="epoch (one full pass through the 5,000 training images)"
          ariaLabel="Test accuracy per epoch from two starting points, the same random numbers with and without the division by the square root of the input count."
        />
      )}
      {summary && (
        <div className="interactive-status">
          <p>
            After {EPOCHS} epochs, out of {summary.n_test.toLocaleString()} held-out
            digits: Module 5's start reads{" "}
            <b>{(summary.runs["plain"].accuracy * 100).toFixed(1)}%</b>, your start reads{" "}
            <b>{(summary.runs["yours"].accuracy * 100).toFixed(1)}%</b>. Same wiring, same
            cost, same sgd, same random numbers, and{" "}
            {Math.abs(
              (summary.runs["yours"].accuracy - summary.runs["plain"].accuracy) * 100,
            ).toFixed(1)}{" "}
            points between them.
          </p>
          <p>
            Saturation is not gone by the end, and it is not meant to be:{" "}
            {(summary.runs["yours"].flat_share_end * 100).toFixed(1)}% of your run's hidden
            neurons are flatter than 0.01 after training, against{" "}
            {(summary.runs["plain"].flat_share_end * 100).toFixed(1)}% for the other run. A
            hidden neuron that has learned to answer one kind of stroke should be decisive
            about it. What the division fixed is saturation before anything has been
            learned, where the flatness is an accident of the draw.
          </p>
        </div>
      )}
    </div>
  );
}
