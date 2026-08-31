import { useEffect, useRef, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { backpropExercise } from "../../../exercises/backprop";
import { sgdExercise } from "../../../exercises/sgd";
import { lockedBy, speakList } from "./lockedBy";

// Chapter 5 payoff: train the 784-30-10 digit reader on the bundled MNIST
// subset, driven by the learner's own backprop plugged into the learner's
// own Chapter 3 sgd. Before training, one mini-batch is priced both ways
// (backprop vs nudge-and-measure) for the wall-clock comparison.

const EPOCHS = 15;

// Seeds are fixed so every correct implementation reproduces the same run:
// init rng 8, sgd rng 2, eta 3.0, mini-batches of 10 (verified to reach
// about 89% test accuracy; see the chapter prose).
const SNIPPET = `
import json, time, types
import numpy as np

_a = json.loads(_args_json)

# The learner's file, once, up to and including their backprop and the
# adapter that arrives with it. Everything below is a name out of it.
_lib = types.ModuleType("your_code")
exec(compile(_a["code"], "your_code.py", "exec"), _lib.__dict__)
feedforward = _lib.feedforward
quadratic_cost = _lib.quadratic_cost

# The swap this chapter is about: their sgd_step asks for gradient(...) and
# gets the nudge-and-measure one written for them in Chapter 3. Rebinding the
# name in their own chapter after it has been read, rather than before, is
# enough, because Python looks a global up when the call happens. Their sgd
# runs unmodified and now walks on their backprop.
def _bp_gradient(weights, biases, X, Y):
    return _lib.batch_gradient(weights, biases, X, Y)

_lib.gradient = _bp_gradient

with open("/mnist_subset.bin", "rb") as _f:
    _buf = _f.read()
X_train, y_train, X_test, y_test = load_mnist_subset(_buf)
Y_train = one_hot(y_train)

_init = np.random.default_rng(8)
weights = [_init.standard_normal((30, 784)), _init.standard_normal((10, 30))]
biases = [_init.standard_normal((30, 1)), _init.standard_normal((10, 1))]
_n_params = sum(int(w.size) for w in weights) + sum(int(b.size) for b in biases)

# Price one training step both ways on the same mini-batch of 10.
_Xb, _Yb = X_train[:, :10], Y_train[:, :10]
_t0 = time.time()
_bp_gradient(weights, biases, _Xb, _Yb)
_bp_step = time.time() - _t0
# Nudge-and-measure, timed on the 10 output biases only: every slope costs
# the same two cost evaluations, so scaling by params/10 is exact in count.
_eps = 1e-5
_t0 = time.time()
for _j in range(10):
    _orig = biases[-1][_j, 0]
    biases[-1][_j, 0] = _orig + _eps
    quadratic_cost(weights, biases, _Xb, _Yb)
    biases[-1][_j, 0] = _orig - _eps
    quadratic_cost(weights, biases, _Xb, _Yb)
    biases[-1][_j, 0] = _orig
_nudge_step = (time.time() - _t0) / 10.0 * _n_params
_js_report(json.dumps({"kind": "duel", "params": _n_params,
                       "bp_step": _bp_step, "nudge_step": _nudge_step}))

_epochs = ${EPOCHS}
_rng = np.random.default_rng(2)
_t0 = time.time()
_acc = 0.0
for _e in range(1, _epochs + 1):
    weights, biases = _lib.sgd(weights, biases, X_train, Y_train, 3.0, 1, 10, _rng)
    _preds = np.argmax(feedforward(weights, biases, X_test), axis=0)
    _acc = float((_preds == y_test).mean())
    _js_report(json.dumps({"kind": "epoch", "epoch": _e, "epochs": _epochs,
        "cost": float(quadratic_cost(weights, biases, X_train, Y_train)),
        "accuracy": _acc, "elapsed": time.time() - _t0}))
_seconds = time.time() - _t0
_steps = -(-X_train.shape[1] // 10) * _epochs

# Where it is wrong, which the accuracy alone cannot say. Per digit, and then
# the mistakes it was most sure about: a confident error is the interesting
# kind, and the ones it nearly got right teach nothing.
_out = feedforward(weights, biases, X_test)
_guesses = np.argmax(_out, axis=0)
_per_digit = [{"digit": int(_d),
               "right": int(((_guesses == y_test) & (y_test == _d)).sum()),
               "total": int((y_test == _d).sum())}
              for _d in range(10)]

_wrong = np.flatnonzero(_guesses != y_test)
_confidence = _out[_guesses[_wrong], _wrong]
_worst = _wrong[np.argsort(-_confidence)][:8]
_mistakes = [{"truth": int(y_test[_k]), "guess": int(_guesses[_k]),
              "confidence": float(_out[_guesses[_k], _k]),
              "pixels": [int(round(_v * 255)) for _v in X_test[:, _k]]}
             for _k in _worst]

json.dumps({
    "per_digit": _per_digit,
    "mistakes": _mistakes,
    "n_wrong": int(_wrong.size),
    "accuracy": _acc,
    "seconds": _seconds,
    "params": _n_params,
    "steps": int(_steps),
    "n_train": int(X_train.shape[1]),
    "n_test": int(X_test.shape[1]),
    "bp_step": _bp_step,
    "nudge_step": _nudge_step,
    "nudge_total_hours": _nudge_step * _steps / 3600.0,
})
`;

interface DuelReport {
  kind: "duel";
  params: number;
  bp_step: number;
  nudge_step: number;
}

interface EpochReport {
  kind: "epoch";
  epoch: number;
  epochs: number;
  cost: number;
  accuracy: number;
  elapsed: number;
}

interface Mistake {
  truth: number;
  guess: number;
  confidence: number;
  /** 784 ink levels, 0 to 255, in the row-by-row order Chapter 2 unrolled. */
  pixels: number[];
}

interface TrainResult {
  per_digit: { digit: number; right: number; total: number }[];
  mistakes: Mistake[];
  n_wrong: number;
  accuracy: number;
  seconds: number;
  params: number;
  steps: number;
  n_train: number;
  n_test: number;
  bp_step: number;
  nudge_step: number;
  nudge_total_hours: number;
}

const bothDone = () => codeReady(backpropExercise.id) && codeReady(sgdExercise.id);

export function BackpropTrainPanel() {
  const [unlocked, setUnlocked] = useState(bothDone);
  const [duel, setDuel] = useState<DuelReport | null>(null);
  const [points, setPoints] = useState<EpochReport[]>([]);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<TrainResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => subscribeProgress(() => setUnlocked(bothDone())), []);

  const run = () => {
    // One projection: the file through backprop already contains their sgd,
    // because sgd is Chapter 3 and backprop is Chapter 5.
    const code = loadCode(backpropExercise.id);
    if (!code) return;
    setRunning(true);
    setDuel(null);
    setPoints([]);
    setResult(null);
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
          const r = msg.payload as DuelReport | EpochReport;
          if (r.kind === "duel") {
            setDuel(r);
            setStatus("Training...");
          } else {
            setPoints((prev) => [...prev, r]);
            setStatus(
              `Epoch ${r.epoch}/${r.epochs}: cost ${r.cost.toFixed(4)}, ` +
                `test accuracy ${(r.accuracy * 100).toFixed(1)}%, ${r.elapsed.toFixed(1)}s`,
            );
          }
        }
        if (msg.type === "pythonDone") {
          setResult(msg.result as TrainResult);
          setRunning(false);
          setStatus("");
        }
        // Stop can also come from another panel's Stop button: terminateWorker
        // resolves every pending request, so without this the button here would
        // stay stuck on its running label.
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
      lockedBy([backpropExercise.id, sgdExercise.id], {
        [backpropExercise.id]: "the backprop exercise above",
      }),
    );
    return (
      <p className="payoff-locked">
        This run is your own backprop driving your own sgd, with nothing borrowed,
        so it needs {missing}. Come back here once the tests are green; nothing in
        Chapters 6 to 8 waits on this run, so reading on is fine.
      </p>
    );
  }

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : result ? "Train again" : "Train the digit reader"}
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
      {duel && (
        <p className="interactive-status">
          The price check, one mini-batch of 10 images, all {duel.params.toLocaleString()}{" "}
          slopes: your backprop took {formatSeconds(duel.bp_step)}. Nudge-and-measure,
          timed on 10 of the knobs and scaled to all of them (every slope costs the same
          two rescores), would take {formatSeconds(duel.nudge_step)} for the same step:{" "}
          {Math.round(duel.nudge_step / Math.max(duel.bp_step, 1e-9)).toLocaleString()}{" "}
          times slower.
        </p>
      )}
      {points.length > 0 && <TrainChart points={points} totalEpochs={EPOCHS} />}
      {result && (
        <p className="interactive-status">
          Done: {(result.accuracy * 100).toFixed(1)}% of the {result.n_test.toLocaleString()}{" "}
          held-out digits read correctly, after {result.steps.toLocaleString()} descent
          steps ({(result.n_train / 10).toLocaleString()} mini-batches per epoch, {EPOCHS}{" "}
          epochs) in {result.seconds.toFixed(1)} seconds. The same run on nudge-measured
          gradients, at the step price measured above, works out to about{" "}
          {formatHours(result.nudge_total_hours)}. Same sgd, same cost, same data; the
          only change is where the slopes come from.
        </p>
      )}
      {result && <Mistakes result={result} />}
    </div>
  );
}

/** What the accuracy does not say: which digits it misses, and what its most
 * confident errors look like. The images are the test digits themselves,
 * sent back as ink levels and drawn the way Chapter 2 drew them. */
function Mistakes({ result }: { result: TrainResult }) {
  const worst = result.per_digit.reduce((a, b) =>
    b.right / b.total < a.right / a.total ? b : a,
  );
  return (
    <div className="mistakes">
      <h5>Where the misses are</h5>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>digit</th>
              {result.per_digit.map((d) => (
                <th key={d.digit}>{d.digit}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>read correctly</td>
              {result.per_digit.map((d) => (
                <td key={d.digit}>
                  {d.right}/{d.total}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="interactive-status">
        {result.n_wrong} of the {result.n_test.toLocaleString()} held-out digits are
        read wrong. The hardest is {worst.digit}, at {worst.right} of {worst.total}.
        Below are the eight the network was most confident about while being wrong;
        the number under each is what it answered, and how sure it was.
      </p>
      <div className="mistake-strip">
        {result.mistakes.map((m, i) => (
          <figure key={i} className="mistake">
            <MistakeDigit pixels={m.pixels} truth={m.truth} guess={m.guess} />
            <figcaption>
              said <b>{m.guess}</b>, was {m.truth}
              <span className="mistake-confidence">
                {/* One decimal: these sit between 98 and 99.9, and rounding to
                    whole percent prints a 100% that is not true. */}
                {(m.confidence * 100).toFixed(1)}% sure
              </span>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

function MistakeDigit({
  pixels,
  truth,
  guess,
}: {
  pixels: number[];
  truth: number;
  guess: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = 255 - pixels[i];
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);
  return (
    <canvas
      ref={ref}
      width={28}
      height={28}
      className="mistake-canvas"
      role="img"
      aria-label={`A handwritten ${truth} that the network read as a ${guess}.`}
    />
  );
}

function formatSeconds(s: number): string {
  if (s < 0.1) return `${(s * 1000).toFixed(0)} milliseconds`;
  if (s < 60) return `${s.toFixed(1)} seconds`;
  return `${(s / 60).toFixed(1)} minutes`;
}

function formatHours(h: number): string {
  if (h < 1) return `${Math.round(h * 60)} minutes`;
  return `${h.toFixed(h < 10 ? 1 : 0)} hours`;
}

// Live chart: training cost (left axis) and test accuracy (right axis) per
// epoch. Same visual language as the Milestone 0 chart, with the course's
// settled vocabulary (cost, not loss).
const W = 640;
const H = 260;
const PAD = { top: 16, right: 56, bottom: 36, left: 56 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

function TrainChart({ points, totalEpochs }: { points: EpochReport[]; totalEpochs: number }) {
  const maxCost = Math.max(...points.map((p) => p.cost), 0.001);
  const x = (epoch: number) => PAD.left + ((epoch - 1) / Math.max(totalEpochs - 1, 1)) * PLOT_W;
  const yCost = (c: number) => PAD.top + (1 - c / maxCost) * PLOT_H;
  const yAcc = (a: number) => PAD.top + (1 - a) * PLOT_H;
  const path = (ys: number[]) =>
    points.map((p, i) => `${i === 0 ? "M" : "L"}${x(p.epoch).toFixed(1)},${ys[i].toFixed(1)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="metrics-chart"
      role="img"
      aria-label="Training cost falling and test accuracy rising per epoch"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={PAD.left} x2={W - PAD.right} y1={yAcc(t)} y2={yAcc(t)} className="chart-grid" />
          <text x={W - PAD.right + 8} y={yAcc(t) + 4} className="chart-tick chart-tick-acc">
            {Math.round(t * 100)}%
          </text>
          <text x={PAD.left - 8} y={yAcc(t) + 4} className="chart-tick chart-tick-loss" textAnchor="end">
            {(maxCost * t).toFixed(2)}
          </text>
        </g>
      ))}
      {points.map((p) => (
        <text key={p.epoch} x={x(p.epoch)} y={H - PAD.bottom + 18} className="chart-tick" textAnchor="middle">
          {p.epoch}
        </text>
      ))}
      <text x={W / 2} y={H - 6} className="chart-axis-label" textAnchor="middle">
        epoch (one full pass through the 5,000 training images)
      </text>
      <path d={path(points.map((p) => yCost(p.cost)))} className="chart-line chart-line-loss" />
      <path d={path(points.map((p) => yAcc(p.accuracy)))} className="chart-line chart-line-acc" />
      {points.map((p) => (
        <circle key={p.epoch} cx={x(p.epoch)} cy={yAcc(p.accuracy)} r={3} className="chart-dot-acc" />
      ))}
      <g className="chart-legend">
        <rect x={PAD.left + 8} y={PAD.top + 4} width={12} height={3} className="chart-swatch-loss" />
        <text x={PAD.left + 26} y={PAD.top + 9} className="chart-tick">
          training cost
        </text>
        <rect x={PAD.left + 118} y={PAD.top + 4} width={12} height={3} className="chart-swatch-acc" />
        <text x={PAD.left + 136} y={PAD.top + 9} className="chart-tick">
          test accuracy
        </text>
      </g>
    </svg>
  );
}
