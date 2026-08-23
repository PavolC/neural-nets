import { useEffect, useState } from "react";
import { sendRequest } from "../../../runtime/workerClient";
import { loadCode, loadCompleted, subscribeProgress } from "../../../state/progress";
import { sgdExercise } from "../../../exercises/sgd";

// Module 3 payoff: run the learner's own sgd on a toy 2D dataset, live.
// It works, and it is visibly slow; the numbers reported here set up the
// case for backpropagation.

const EPOCHS = 200;

// The dataset, the network init, and the rng are all fixed seeds, so every
// run of a correct implementation reproduces the same curve.
const SNIPPET = `
import json, time, types
import numpy as np
from course import quadratic_loss, feedforward
_a = json.loads(_args_json)
_mod = types.ModuleType("sgd_submission")
exec(compile(_a["code"], "your_code.py", "exec"), _mod.__dict__)

_data_rng = np.random.default_rng(0)
_corners = [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
_pts, _labels = [], []
for _cx, _cy, _lab in _corners:
    _pts.append(_data_rng.normal([_cx, _cy], 0.15, size=(10, 2)))
    _labels.extend([_lab] * 10)
X = np.vstack(_pts).T
Y = np.array(_labels, dtype=float).reshape(1, -1)

_init = np.random.default_rng(1)
weights = [_init.standard_normal((8, 2)), _init.standard_normal((1, 8))]
biases = [_init.standard_normal((8, 1)), _init.standard_normal((1, 1))]

_epochs = ${EPOCHS}
_bs = 10
_sgd_rng = np.random.default_rng(2)
_t0 = time.time()
for _e in range(1, _epochs + 1):
    weights, biases = _mod.sgd(weights, biases, X, Y, 5.0, 1, _bs, _sgd_rng)
    _js_report(json.dumps({
        "epoch": _e, "epochs": _epochs,
        "loss": float(quadratic_loss(weights, biases, X, Y)),
        "elapsed": time.time() - _t0,
    }))
_out = feedforward(weights, biases, X)
_acc = float(((_out >= 0.5).astype(int) == Y.astype(int)).mean())
_params = sum(int(w.size) for w in weights) + sum(int(b.size) for b in biases)
_steps = -(-X.shape[1] // _bs) * _epochs
json.dumps({
    "final_loss": float(quadratic_loss(weights, biases, X, Y)),
    "accuracy": _acc,
    "seconds": time.time() - _t0,
    "params": _params,
    "steps": int(_steps),
    "forward_passes": int(_steps * _params * 2),
})
`;

interface SgdResult {
  final_loss: number;
  accuracy: number;
  seconds: number;
  params: number;
  steps: number;
  forward_passes: number;
}

export function SgdLivePanel() {
  const [unlocked, setUnlocked] = useState(() => loadCompleted(sgdExercise.id));
  const [losses, setLosses] = useState<number[]>([]);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<SgdResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => subscribeProgress(() => setUnlocked(loadCompleted(sgdExercise.id))), []);

  const run = () => {
    const code = loadCode(sgdExercise.id);
    if (!code) return;
    setRunning(true);
    setLosses([]);
    setResult(null);
    setError(null);
    setStatus("Starting...");
    sendRequest({ type: "runPython", code: SNIPPET, args: { code } }, (msg) => {
      if (msg.type === "status") setStatus(msg.text);
      if (msg.type === "report") {
        const r = msg.payload as { epoch: number; epochs: number; loss: number; elapsed: number };
        setLosses((prev) => [...prev, r.loss]);
        setStatus(`Epoch ${r.epoch}/${r.epochs}, loss ${r.loss.toFixed(4)}, ${r.elapsed.toFixed(1)}s`);
      }
      if (msg.type === "pythonDone") {
        setResult(msg.result as SgdResult);
        setRunning(false);
        setStatus("");
      }
      if (msg.type === "error") {
        setError(msg.message);
        setRunning(false);
        setStatus("");
      }
    });
  };

  if (!unlocked) {
    return (
      <p className="payoff-locked">
        Locked: pass the SGD exercise above, then train a real (tiny) network with your
        own code here.
      </p>
    );
  }

  const maxLoss = Math.max(...losses, 0.001);

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : result ? "Train again" : "Train a tiny network with your sgd"}
        </button>
        <span className={error ? "demo-status demo-status-error" : "demo-status"}>
          {error ?? status}
        </span>
      </div>
      {losses.length > 0 && (
        <svg viewBox="0 0 440 120" className="interactive-svg sparkline">
          <polyline
            points={losses
              .map((l, i) => `${12 + (i / Math.max(EPOCHS - 1, 1)) * 416},${108 - (l / maxLoss) * 96}`)
              .join(" ")}
            className="traj traj-sgd"
          />
          <text x={12} y={14} className="chart-tick">loss per epoch (your sgd)</text>
        </svg>
      )}
      {result && (
        <p className="interactive-status">
          Done: loss {result.final_loss.toFixed(4)}, {Math.round(result.accuracy * 100)}%
          of the toy points classified correctly, in {result.seconds.toFixed(1)} seconds.
          The bill: {result.steps} steps, and every step estimated {result.params} partial
          derivatives by nudging each parameter twice, which cost{" "}
          {result.forward_passes.toLocaleString()} forward passes in total. This tiny
          network has {result.params} parameters; the digit reader from Module 2 has
          11,935, roughly 360 times this bill on every single step. That is why
          Module 5 exists.
        </p>
      )}
    </div>
  );
}
