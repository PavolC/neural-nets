import { useEffect, useState } from "react";
import { assetUrl } from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { crossEntropyExercise } from "../../../exercises/cross-entropy";
import { smartInitExercise } from "../../../exercises/smart-init";
import { l2Exercise } from "../../../exercises/l2";
import { trainExercise } from "../../../exercises/train";
import { lockedBy, speakList } from "./lockedBy";
import { EpochChart, type EpochSeries } from "./EpochChart";

// The capstone run: the learner's own loop, on the digit reader, with their
// Module 7 functions patched into `course` first so that everything the loop
// reaches for is theirs. The one exception is backprop, which stays the
// course's copy carrying their Module 5 algorithm with BP1 lifted into an
// argument, exactly as Module 7 handed it back.
//
// Progress streams by wrapping their own accuracy function: train calls it
// once per epoch by contract, so the wrapper is the per-epoch hook. A loop
// that scores some other way simply reports nothing until it returns.

const EPOCHS = 15;
const ETAS = [0.1, 0.5, 3.0];

const SNIPPET = `
import json, time, types
import numpy as np
import course

_a = json.loads(_args_json)

# Their whole file, once. Nothing is patched over anything: init_network,
# l2_step, cross_entropy_delta, batch_gradient, backprop and feedforward are
# all defined above train in the same file, so their loop calls their code by
# reading down the page.
_prog = types.ModuleType("your_code")
exec(compile(_a["code"], "your_code.py", "exec"), _prog.__dict__)

with open("/mnist_subset.bin", "rb") as _f:
    X_train, y_train, X_test, y_test = load_mnist_subset(_f.read())
Y_train = one_hot(y_train)

_t0 = time.time()
_epoch = 0
_scored = getattr(_prog, "accuracy", None)

def _reporting_accuracy(weights, biases, X, y):
    """Their accuracy, with a progress report on the way out."""
    global _epoch
    value = _scored(weights, biases, X, y)
    _epoch += 1
    _js_report(json.dumps({"epoch": _epoch, "accuracy": float(value),
                           "elapsed": time.time() - _t0}))
    return value

if _scored is not None:
    _prog.accuracy = _reporting_accuracy

_weights, _biases, _history = _prog.train(
    [784, 30, 10], X_train, Y_train, X_test, y_test,
    ${EPOCHS}, _a["eta"], _a["lmbda"], 10, np.random.default_rng(8))

# Scored again here, through the course's own forward pass, so the number the
# panel prints does not depend on the accuracy function that produced the
# history: a check that shares its yardstick with the thing checked is not one.
_final = float((np.argmax(course.feedforward(_weights, _biases, X_test), axis=0) == y_test).mean())
_weight_size = float(sum((w ** 2).sum() for w in _weights))

json.dumps({"history": [float(h) for h in _history], "final": _final,
            "seconds": time.time() - _t0, "n_test": int(X_test.shape[1]),
            "weight_size": _weight_size})
`;

interface EpochReport {
  epoch: number;
  accuracy: number;
  elapsed: number;
}

interface Summary {
  history: number[];
  final: number;
  seconds: number;
  n_test: number;
  weight_size: number;
}

const needed = () =>
  codeReady(trainExercise.id) &&
  codeReady(smartInitExercise.id) &&
  codeReady(crossEntropyExercise.id) &&
  codeReady(l2Exercise.id);

export function FullTrainPanel() {
  const [unlocked, setUnlocked] = useState(needed);
  const [eta, setEta] = useState(0.5);
  const [ranWith, setRanWith] = useState(0.5);
  const [points, setPoints] = useState<EpochReport[]>([]);
  const [status, setStatus] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => subscribeProgress(() => setUnlocked(needed())), []);

  const run = () => {
    // One projection: the file through their program holds every piece it
    // calls, in the order they wrote them.
    const code = loadCode(trainExercise.id);
    if (!code) return;
    setRunning(true);
    setRanWith(eta);
    setPoints([]);
    setSummary(null);
    setError(null);
    setStatus("Starting...");
    sendRequest(
      {
        type: "runPython",
        code: SNIPPET,
        args: { code, eta, lmbda: 1.0 },
        dataUrl: assetUrl("data/mnist_subset.bin.gz"),
      },
      (msg) => {
        if (msg.type === "status") setStatus(msg.text);
        if (msg.type === "report") {
          const r = msg.payload as EpochReport;
          setPoints((prev) => [...prev, r]);
          setStatus(
            `epoch ${r.epoch}/${EPOCHS}, ${(r.accuracy * 100).toFixed(1)}% of the ` +
              `held-out digits, ${r.elapsed.toFixed(1)}s`,
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
      lockedBy(
        [trainExercise.id, smartInitExercise.id, crossEntropyExercise.id, l2Exercise.id],
        { [trainExercise.id]: "this module's exercise" },
      ),
    );
    return (
      <p className="payoff-locked">
        This run is your whole program, so it needs {missing}. Every function it
        calls is one you wrote, and nothing is borrowed.
      </p>
    );
  }

  const series: EpochSeries[] = points.length
    ? [
        {
          key: "yours",
          label: `your train(), step size ${ranWith}`,
          cls: "m7-line-b",
          values: points.map((p) => p.accuracy),
        },
      ]
    : [];

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button onClick={run} disabled={running}>
          {running ? "Training..." : summary ? "Run it again" : "Run my program"}
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
          <legend>step size</legend>
          {ETAS.map((v) => (
            <button
              key={v}
              className={`chip ${eta === v ? "chip-active" : ""}`}
              onClick={() => setEta(v)}
              disabled={running}
            >
              {v}
            </button>
          ))}
        </fieldset>
        <p className="m8-eta-note">
          784-30-10, {EPOCHS} epochs, 5,000 images, mini-batches of 10, lambda 1
        </p>
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
          yLabel="test accuracy"
          xLabel="epoch (one full pass through the 5,000 training images)"
          ariaLabel="Test accuracy per epoch for the digit reader, trained by the learner's own program."
        />
      )}
      {summary && (
        <div className="interactive-status">
          <p>
            {EPOCHS} epochs in {summary.seconds.toFixed(1)} seconds, and{" "}
            <b>{(summary.final * 100).toFixed(1)}%</b> of{" "}
            {summary.n_test.toLocaleString()} held-out digits read correctly, at a
            step size of {ranWith}. Nothing in that run came from the course except
            the images and the four equations' bookkeeping: the network was drawn
            by your init_network, its gradients averaged from your backprop's
            algorithm under your cross-entropy blame, every step taken by your
            l2_step, the epochs and mini-batches walked by your train, and the
            score above computed by your accuracy.
          </p>
        </div>
      )}
    </div>
  );
}
