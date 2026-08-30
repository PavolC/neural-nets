import { useEffect, useRef, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { crossEntropyExercise } from "../../../exercises/cross-entropy";
import { smartInitExercise } from "../../../exercises/smart-init";
import { sgdExercise } from "../../../exercises/sgd";
import { EpochChart, type EpochSeries } from "./EpochChart";
import { lockedBy, speakList } from "./lockedBy";

// Module 8's payoff run: the same digit reader at two depths, trained by the
// learner's own sgd from their own starting point, so the collapse the bars
// predicted shows up as epochs that go nowhere. The squash is a choice,
// because the ReLU comparison is the whole point of the section it sits in;
// with ReLU chosen, the gradient is the course's (their backprop has the
// sigmoid built into BP2), and the sgd is still theirs.

const EPOCHS = 15;
const DEEP = 4;
// Each squash gets the step size that suits it, the same way each cost did in
// Module 7. Found by trying, on the shallow network, at fifteen epochs.
const ETA: Record<string, number> = { sigmoid: 0.5, relu: 0.05 };

const SNIPPET = `
import json, time, types
import numpy as np

_a = json.loads(_args_json)
_act = _a["activation"]
_eta = _a["eta"]

# The learner's file, once, up to and including their better starting point.
# With the sigmoid chosen, every gradient below is computed by their own
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

def _one_example(weights, biases, x, y):
    if _act == "relu":
        return _relu_backprop(weights, biases, x, y)
    return _lib.backprop(weights, biases, x, y, _lib.cross_entropy_delta)

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

_predict = _relu_feedforward if _act == "relu" else feedforward

# Their sgd, walking on whichever gradient the squash calls for.
_lib.gradient = _grad

def _layer_speeds(weights, biases, n=100):
    """Every layer's ||dC/db|| before the first step, from the same gradient."""
    nw, nb = _grad(weights, biases, X_train[:, :n], Y_train[:, :n])
    return [float(np.linalg.norm(g)) for g in nb]

_out = {}
for _key, _hidden in (("shallow", 1), ("deep", ${DEEP})):
    _sizes = [784] + [30] * _hidden + [10]
    _label = "1 hidden layer" if _hidden == 1 else str(_hidden) + " hidden layers"
    weights, biases = _lib.init_network(_sizes, np.random.default_rng(8))
    _js_report(json.dumps({"kind": "start", "key": _key, "label": _label,
                           "sizes": _sizes, "speeds": _layer_speeds(weights, biases)}))
    _rng = np.random.default_rng(2)
    _t0 = time.time()
    for _e in range(1, ${EPOCHS} + 1):
        weights, biases = _lib.sgd(weights, biases, X_train, Y_train, _eta, 1, 10, _rng)
        _acc = float((np.argmax(_predict(weights, biases, X_test), axis=0) == y_test).mean())
        _js_report(json.dumps({"kind": "epoch", "key": _key, "label": _label,
                               "epoch": _e, "accuracy": _acc,
                               "elapsed": time.time() - _t0}))
    _out[_key] = {"accuracy": _acc, "seconds": time.time() - _t0,
                  "speeds_end": _layer_speeds(weights, biases)}

json.dumps({"runs": _out, "n_test": int(X_test.shape[1])})
`;

interface StartReport {
  kind: "start";
  key: string;
  label: string;
  sizes: number[];
  speeds: number[];
}

interface EpochReport {
  kind: "epoch";
  key: string;
  label: string;
  epoch: number;
  accuracy: number;
  elapsed: number;
}

interface RunSummary {
  accuracy: number;
  seconds: number;
  speeds_end: number[];
}

interface Summary {
  runs: Record<string, RunSummary>;
  n_test: number;
}

const LINES: Record<string, { label: string; cls: string; dashed?: boolean }> = {
  shallow: { label: "1 hidden layer (Module 7's network)", cls: "m7-line-a", dashed: true },
  deep: { label: `${DEEP} hidden layers of 30`, cls: "m7-line-b" },
};

const needed = () =>
  codeReady(crossEntropyExercise.id) &&
  codeReady(smartInitExercise.id) &&
  codeReady(sgdExercise.id);

export function DepthTrainPanel() {
  const [unlocked, setUnlocked] = useState(needed);
  const [activation, setActivation] = useState("sigmoid");
  // What the drawn chart was actually trained with. The chips select the NEXT
  // run, and nothing retrains on a click (a run is fifteen epochs at each
  // depth), so the panel has to say which squash the lines on screen came
  // from and which is merely selected.
  const [drawnWith, setDrawnWith] = useState<string | null>(null);
  // Every squash trained this session, one row per (squash, depth), a rerun
  // replacing its own two rows. The chart draws the latest run; the table is
  // where runs meet, because the comparison the section makes (four hidden
  // layers against one, under each squash) is across runs, and a summary
  // that only described the last run erased one side of it. The squash is
  // the whole key: the step size is a function of it, never chosen on its
  // own.
  const [history, setHistory] = useState<
    { activation: string; depth: "shallow" | "deep"; s: RunSummary }[]
  >([]);
  // Everything a finished run drew, keyed by its squash. Same seed and same
  // settings give byte-identical training, so a squash trained once this
  // session is never trained again: selecting it back redraws the chart, the
  // speed table and the summary with no run at all. Unlike the weight-decay
  // panel there is no partial case (both depths belong to one squash), so
  // the snippet never needs to be asked for less than both, and stays the
  // exact path check_panels verifies. The refs mirror what setStarts and
  // setPoints receive, because the completion callback cannot read fresh
  // state through its own closure. A stopped run caches nothing, so partial
  // lines can never be replayed as finished ones.
  const runCache = useRef(
    new Map<string, { starts: StartReport[]; points: EpochReport[]; summary: Summary }>(),
  );
  const startsRef = useRef<StartReport[]>([]);
  const reportsRef = useRef<EpochReport[]>([]);
  const [starts, setStarts] = useState<StartReport[]>([]);
  const [points, setPoints] = useState<EpochReport[]>([]);
  const [status, setStatus] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => subscribeProgress(() => setUnlocked(needed())), []);

  const squashName = (a: string) =>
    a === "relu" ? "ReLU (the course's)" : "sigmoid (your backprop)";

  const run = () => {
    // One projection: the file through the starting-point section already
    // holds their sgd, their backprop and their cross-entropy blame.
    const code = loadCode(smartInitExercise.id);
    if (!code) return;
    setError(null);
    setDrawnWith(activation);
    const cached = runCache.current.get(activation);
    if (cached) {
      // This squash already trained this session: redraw, train nothing.
      setStarts(cached.starts);
      setPoints(cached.points);
      setSummary(cached.summary);
      return;
    }
    setRunning(true);
    setStarts([]);
    setPoints([]);
    startsRef.current = [];
    reportsRef.current = [];
    setSummary(null);
    setStatus("Starting...");
    sendRequest(
      {
        type: "runPython",
        code: SNIPPET,
        args: { code, activation, eta: ETA[activation] },
        dataUrl: assetUrl("data/mnist_subset.bin.gz"),
      },
      (msg) => {
        if (msg.type === "status") setStatus(msg.text);
        if (msg.type === "report") {
          const r = msg.payload as StartReport | EpochReport;
          if (r.kind === "start") {
            startsRef.current.push(r);
            setStarts((prev) => [...prev, r]);
            setStatus(`${r.label}: measuring every layer before the first step...`);
          } else {
            reportsRef.current.push(r);
            setPoints((prev) => [...prev, r]);
            setStatus(
              `${r.label}: epoch ${r.epoch}/${EPOCHS}, ` +
                `${(r.accuracy * 100).toFixed(1)}% of the test digits, ${r.elapsed.toFixed(1)}s`,
            );
          }
        }
        if (msg.type === "pythonDone") {
          const done = msg.result as Summary;
          setSummary(done);
          runCache.current.set(activation, {
            starts: startsRef.current,
            points: reportsRef.current,
            summary: done,
          });
          setHistory((prev) => {
            const next = prev.filter((r) => r.activation !== activation);
            next.push({ activation, depth: "shallow", s: done.runs["shallow"] });
            next.push({ activation, depth: "deep", s: done.runs["deep"] });
            next.sort((a, b) =>
              a.activation !== b.activation
                ? a.activation === "sigmoid"
                  ? -1
                  : 1
                : a.depth === "shallow"
                  ? -1
                  : 1,
            );
            return next;
          });
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
        This run needs {missing}: it builds both networks with your init_network
        and trains them with your sgd, with nothing borrowed.
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

  const deepStart = starts.find((s) => s.key === "deep");
  const endSpeeds = summary?.runs["deep"].speeds_end ?? [];
  const startRatio = deepStart
    ? deepStart.speeds[deepStart.speeds.length - 1] / deepStart.speeds[0]
    : 0;

  const runLabel = runCache.current.has(activation)
    ? "Redraw this pair"
    : summary
      ? "Run both again"
      : "Train both depths";

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : runLabel}
        </button>
        {running && (
          <button className="button-secondary" onClick={terminateWorker}>
            Stop
          </button>
        )}
        <span className={`demo-status status-fixed ${error ? "demo-status-error" : ""}`}>
          {error ??
            (running || !drawnWith
              ? status
              : drawnWith === activation
                ? `drawn with ${squashName(drawnWith)}`
                : `drawn with ${squashName(drawnWith)}; the chips pick the next run, so press ${runLabel}`)}
        </span>
      </div>
      <div className="interactive-controls">
        <fieldset className="m7-choice">
          <legend>what the hidden neurons squash with</legend>
          <button
            className={`chip ${activation === "sigmoid" ? "chip-active" : ""}`}
            onClick={() => setActivation("sigmoid")}
            disabled={running}
          >
            sigmoid (your backprop)
          </button>
          <button
            className={`chip ${activation === "relu" ? "chip-active" : ""}`}
            onClick={() => setActivation("relu")}
            disabled={running}
          >
            ReLU (the course's)
          </button>
        </fieldset>
        <p className="m8-eta-note">
          step size {ETA[activation]}, {EPOCHS} epochs, 5,000 images, mini-batches of 10
        </p>
      </div>
      {deepStart && (
        <div className="table-scroll scroll-x" tabIndex={0}>
          <table className="truth-table">
            <thead>
              <tr>
                <th>{deepStart.sizes.join("-")}, before a single step</th>
                {deepStart.speeds.map((_, i) => (
                  <th key={i}>layer {i + 2}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>learning speed, from your gradient</td>
                {deepStart.speeds.map((s, i) => (
                  <td key={i}>{s < 0.01 ? s.toExponential(2) : s.toFixed(4)}</td>
                ))}
              </tr>
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
          ariaLabel="Test accuracy per epoch for a network with one hidden layer and one with four, trained identically."
        />
      )}
      {summary && (
        <div className="interactive-status">
          <p>
            After {EPOCHS} epochs, out of {summary.n_test.toLocaleString()} held-out
            digits: one hidden layer reads{" "}
            <b>{(summary.runs["shallow"].accuracy * 100).toFixed(1)}%</b>, {DEEP} hidden
            layers read <b>{(summary.runs["deep"].accuracy * 100).toFixed(1)}%</b>. Same
            images, same shuffle, same step size, same sgd, {DEEP - 1} extra layers of 30
            neurons and{" "}
            {drawnWith === "relu"
              ? "a squash with no ceiling on its slope"
              : "a squash whose slope never exceeds 0.25"}
            .
          </p>
          <p>
            The deep network's layer speeds have levelled out by the end. Its first
            hidden layer finishes at {endSpeeds[0].toExponential(2)} against{" "}
            {endSpeeds[endSpeeds.length - 1].toExponential(2)} at the output, where
            the table above the chart started them{" "}
            <b>{startRatio < 10 ? startRatio.toFixed(1) : startRatio.toFixed(0)}</b>{" "}
            times apart. The imbalance is a condition of the starting point, and the
            epochs the network spends working out of it are what the left-hand end of
            the chart is showing.
          </p>
          {history.length > 0 && (
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
                  {history.map((row) => {
                    const onChart = drawnWith !== null && row.activation === drawnWith;
                    const label =
                      `${squashName(row.activation)}, ` +
                      (row.depth === "shallow" ? "1 hidden layer" : `${DEEP} hidden layers`) +
                      (onChart ? " (on the chart)" : "");
                    return (
                      <tr key={`${row.activation}-${row.depth}`}>
                        <td>{label}</td>
                        <td>{ETA[row.activation]}</td>
                        <td>{(row.s.accuracy * 100).toFixed(1)}%</td>
                        <td>
                          {row.s.speeds_end[0] < 0.01
                            ? row.s.speeds_end[0].toExponential(2)
                            : row.s.speeds_end[0].toFixed(4)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
