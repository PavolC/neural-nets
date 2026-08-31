import { useEffect, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { crossEntropyExercise } from "../../../exercises/cross-entropy";
import { sgdExercise } from "../../../exercises/sgd";
import { EpochChart, type EpochSeries } from "./EpochChart";
import { lockedBy, speakList } from "./lockedBy";

// Chapter 7, first cycle: the digit reader from Chapter 5, trained three ways
// from the same starting parameters. Two costs at the same step size, and
// then the quadratic cost again at the step size Chapter 5 tuned for it.
// Everything except the output blame and eta is held fixed.

const EPOCHS = 8;

// Same seeds as Chapter 5's training panel (init 8, sgd 2, mini-batches of
// 10), so run C reproduces that chapter's first eight epochs exactly.
const SNIPPET = `
import json, time, types
import numpy as np
from course import quadratic_output_delta

_a = json.loads(_args_json)

# The learner's file, once, up to and including their cross-entropy work.
# It carries their sgd from Chapter 3 and the adapter from Chapter 5 too, so
# every name below except the old BP1 is theirs.
_lib = types.ModuleType("your_code")
exec(compile(_a["code"], "your_code.py", "exec"), _lib.__dict__)
feedforward = _lib.feedforward
quadratic_cost = _lib.quadratic_cost

with open("/mnist_subset.bin", "rb") as _f:
    X_train, y_train, X_test, y_test = load_mnist_subset(_f.read())
Y_train = one_hot(y_train)

def _start():
    r = np.random.default_rng(8)
    w = [r.standard_normal((30, 784)), r.standard_normal((10, 30))]
    b = [r.standard_normal((30, 1)), r.standard_normal((10, 1))]
    return w, b

# Their sgd_step asks for gradient(...). Point that name at the adapter with
# one BP1 or the other, and their loop is untouched: only what answers "which
# way is downhill" changes. Rebinding after the exec is enough, because Python
# looks a global up when the call happens, and it means the file is read once
# rather than once per run.
def _use(output_delta):
    _lib.gradient = lambda w, b, X, Y: _lib.batch_gradient(w, b, X, Y, output_delta)

_runs = [
    ("quad-matched", "the quadratic cost, η = 0.5", quadratic_output_delta, 0.5, quadratic_cost),
    ("cross", "your cross-entropy cost, η = 0.5", _lib.cross_entropy_delta, 0.5, _lib.cross_entropy_cost),
    ("quad-tuned", "the quadratic cost, η = 3.0", quadratic_output_delta, 3.0, quadratic_cost),
]

_out = {}
for _key, _label, _delta, _eta, _cost in _runs:
    _use(_delta)
    weights, biases = _start()
    _rng = np.random.default_rng(2)
    _t0 = time.time()
    for _e in range(1, ${EPOCHS} + 1):
        weights, biases = _lib.sgd(weights, biases, X_train, Y_train, _eta, 1, 10, _rng)
        _acc = float((np.argmax(feedforward(weights, biases, X_test), axis=0) == y_test).mean())
        _js_report(json.dumps({"key": _key, "label": _label, "epoch": _e,
                               "accuracy": _acc, "elapsed": time.time() - _t0}))
    _out[_key] = {"accuracy": _acc,
                  "cost": float(_cost(weights, biases, X_train, Y_train)),
                  "seconds": time.time() - _t0}

json.dumps({"runs": _out, "n_test": int(X_test.shape[1])})
`;

interface Report {
  key: string;
  label: string;
  epoch: number;
  accuracy: number;
  elapsed: number;
}

interface Summary {
  runs: Record<string, { accuracy: number; cost: number; seconds: number }>;
  n_test: number;
}

const LINES: Record<string, { label: string; cls: string; dashed?: boolean }> = {
  "quad-matched": { label: "the quadratic cost, step size 0.5", cls: "m7-line-a" },
  cross: { label: "your cross-entropy cost, step size 0.5", cls: "m7-line-b" },
  "quad-tuned": { label: "the quadratic cost, step size 3.0 (Chapter 5's run)", cls: "m7-line-a", dashed: true },
};

const needed = () =>
  codeReady(crossEntropyExercise.id) && codeReady(sgdExercise.id);

export function CostSwapPanel() {
  const [unlocked, setUnlocked] = useState(needed);
  const [points, setPoints] = useState<Report[]>([]);
  const [status, setStatus] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => subscribeProgress(() => setUnlocked(needed())), []);

  const run = () => {
    // One projection: the file through the cross-entropy section already
    // holds their Chapter 3 sgd and Chapter 5 backprop.
    const code = loadCode(crossEntropyExercise.id);
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
        args: { code },
        dataUrl: assetUrl("data/mnist_subset.bin.gz"),
      },
      (msg) => {
        if (msg.type === "status") setStatus(msg.text);
        if (msg.type === "report") {
          const r = msg.payload as Report;
          setPoints((prev) => [...prev, r]);
          setStatus(
            `${r.label}: epoch ${r.epoch}/${EPOCHS}, ` +
              `${(r.accuracy * 100).toFixed(1)}% of the test digits, ${r.elapsed.toFixed(1)}s`,
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
      lockedBy([crossEntropyExercise.id, sgdExercise.id], {
        [crossEntropyExercise.id]: "the cross-entropy exercise above",
      }),
    );
    return (
      <p className="payoff-locked">
        These runs use your cost inside your sgd, with nothing borrowed, so they
        need {missing}.
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
          {running ? "Training..." : summary ? "Run the three again" : "Train it three ways"}
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
      {series.length > 0 && (
        <EpochChart
          series={series}
          epochs={EPOCHS}
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
          ariaLabel="Test accuracy per epoch for three runs of the same network: the quadratic cost at step size 0.5, the cross-entropy cost at step size 0.5, and the quadratic cost at step size 3.0."
        />
      )}
      {summary && (
        <div className="interactive-status">
          <p>
            After {EPOCHS} epochs, out of {summary.n_test.toLocaleString()} held-out digits:
          </p>
          <ul className="m7-result-list">
            <li>
              quadratic cost at step size 0.5:{" "}
              <b>{(summary.runs["quad-matched"].accuracy * 100).toFixed(1)}%</b>
            </li>
            <li>
              your cross-entropy cost at step size 0.5:{" "}
              <b>{(summary.runs["cross"].accuracy * 100).toFixed(1)}%</b>
            </li>
            <li>
              quadratic cost at step size 3.0:{" "}
              <b>{(summary.runs["quad-tuned"].accuracy * 100).toFixed(1)}%</b>
            </li>
          </ul>
          <p>
            Each run took about{" "}
            {(
              (summary.runs["cross"].seconds +
                summary.runs["quad-matched"].seconds +
                summary.runs["quad-tuned"].seconds) /
              3
            ).toFixed(0)}{" "}
            seconds. The two final training costs are not comparable numbers (the runs are
            scored by different costs), so the accuracy column is the comparison.
          </p>
        </div>
      )}
    </div>
  );
}
