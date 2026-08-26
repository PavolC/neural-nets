import { useState } from "react";
import { MetricsChart } from "./MetricsChart";
import type { EpochMetrics, TrainResult, WorkerResponse } from "../runtime/messages";
import { sendRequest, terminateWorker } from "../runtime/workerClient";

// Milestone 0 defaults: 784-30-10 sigmoid net, quadratic cost, plain SGD.
// Measured envelope: about 4.5s for 30 epochs, 88.6% test accuracy (README).
const TRAIN_PARAMS = {
  epochs: 30,
  hidden: 30,
  miniBatchSize: 10,
  eta: 3.0,
  seed: 1,
};

type Phase = "idle" | "running" | "done" | "error";

export function TrainingDemo() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [status, setStatus] = useState("Ready. Training runs entirely in your browser.");
  const [log, setLog] = useState<string[]>([]);
  const [points, setPoints] = useState<EpochMetrics[]>([]);
  const [result, setResult] = useState<TrainResult | null>(null);

  const start = () => {
    setPhase("running");
    setPoints([]);
    setResult(null);
    setLog([]);
    // Resolve against the page URL on the main thread: inside the worker,
    // relative URLs would resolve against the worker script's location.
    const dataUrl = new URL(
      `${import.meta.env.BASE_URL}data/mnist_subset.bin.gz`,
      window.location.href,
    ).href;
    sendRequest({ type: "train", dataUrl, ...TRAIN_PARAMS }, (msg: WorkerResponse) => {
      switch (msg.type) {
        case "status":
          setStatus(msg.text);
          break;
        case "log":
          setLog((prev) => [...prev.slice(-49), msg.text]);
          break;
        case "epoch":
          setPoints((prev) => [...prev, msg]);
          setStatus(
            `Training: epoch ${msg.epoch}/${msg.epochs}, ` +
              `loss ${msg.loss.toFixed(4)}, test accuracy ${(msg.accuracy * 100).toFixed(1)}%, ` +
              `${msg.elapsed.toFixed(1)}s elapsed`,
          );
          break;
        case "trainDone":
          setResult(msg.result);
          setPhase("done");
          setStatus("Done.");
          break;
        case "cancelled":
          setPhase("idle");
          // The button returns to its idle label, so do not name a button that
          // is no longer on screen.
          setStatus("Stopped.");
          break;
        case "error":
          setPhase("error");
          setStatus(`Something went wrong: ${msg.message}`);
          break;
        default:
          break;
      }
    });
  };

  return (
    <section className="demo">
      <h2>See where this ends up: a trained digit reader</h2>
      <p>
        This is the finished thing, running the course's own reference code rather than
        yours: a 784-30-10 sigmoid network trained on 5,000 MNIST digits (1,000 held out
        for testing) with plain stochastic gradient descent. By Module 5 you will have
        written the parts that make it work. Python and NumPy run in a Web Worker via
        Pyodide, so the page stays responsive while it trains, and loss and test accuracy
        stream in live after every epoch.
      </p>
      <div className="demo-controls">
        <button onClick={start} disabled={phase === "running"}>
          {phase === "running" ? "Training..." : phase === "idle" ? "Load Python and train" : "Train again"}
        </button>
        {phase === "running" && (
          <button className="button-secondary" onClick={terminateWorker}>
            Stop
          </button>
        )}
        <span className="demo-params">
          {TRAIN_PARAMS.epochs} epochs, mini-batch {TRAIN_PARAMS.miniBatchSize}, learning
          rate {TRAIN_PARAMS.eta}, seed {TRAIN_PARAMS.seed}
        </span>
      </div>
      <p className={`demo-status demo-status-${phase}`} role="status" aria-live="polite">
        {status}
      </p>
      {points.length > 0 && <MetricsChart points={points} totalEpochs={TRAIN_PARAMS.epochs} />}
      {result && (
        <p className="demo-result">
          Final test accuracy <strong>{(result.final_accuracy * 100).toFixed(1)}%</strong> on{" "}
          {result.n_test.toLocaleString()} held-out digits, trained on{" "}
          {result.n_train.toLocaleString()} examples in{" "}
          <strong>{result.train_seconds.toFixed(1)} seconds</strong>.
        </p>
      )}
      {log.length > 0 && (
        <details className="demo-log" open={phase === "error"}>
          <summary>Runtime log</summary>
          <pre>{log.join("\n")}</pre>
        </details>
      )}
    </section>
  );
}
