import { useEffect, useRef, useState } from "react";
import {
  fetchMnistTest,
  fetchPretrainedWeights,
  type MnistTestSubset,
  type PretrainedWeights,
} from "../../../runtime/assets";
import { sendRequest, terminateWorker } from "../../../runtime/workerClient";
import { codeReady, loadCode, subscribeProgress } from "../../../state/progress";
import { feedforwardExercise } from "../../../exercises/feedforward";
import { divergingColor, drawMnistDigit } from "./utils";
import { useInViewOnce } from "../../../components/useInViewOnce";

// Module 2 interactive: a 784-15-10 network, rendered compactly. Hovering a
// hidden neuron shows its 784 incoming weights as a 28x28 image patch.
// Once the learner's feedforward passes its tests, their code runs on real
// test digits and its activations light up the diagram.

const N_DIGITS = 10; // thumbnails shown
const SVG_W = 460;
const SVG_H = 420;
const HIDDEN_X = 190;
const OUTPUT_X = 380;
const INPUT_X = 40;

interface DigitResult {
  hidden: number[];
  output: number[];
  pred: number;
}

// Python snippet: run the learner's saved feedforward on the digit batch.
// Hidden activations come from calling their function on the one-layer
// sub-network, so every number on screen is produced by their code.
const PAYOFF_SNIPPET = `
import json, types
import numpy as np
_a = json.loads(_args_json)
_mod = types.ModuleType("payoff_submission")
exec(compile(_a["code"], "your_code.py", "exec"), _mod.__dict__)
_W = [np.array(w) for w in _a["weights"]]
_B = [np.array(b).reshape(-1, 1) for b in _a["biases"]]
_results = []
for _row in _a["digits"]:
    _x = np.array(_row, dtype=float).reshape(-1, 1) / 255.0
    _hidden = _mod.feedforward(_W[:1], _B[:1], _x)
    _out = _mod.feedforward(_W, _B, _x)
    _results.append({
        "hidden": [float(v) for v in np.ravel(_hidden)],
        "output": [float(v) for v in np.ravel(_out)],
        "pred": int(np.argmax(np.ravel(_out))),
    })
json.dumps(_results)
`;

function drawPatch(canvas: HTMLCanvasElement, row: number[]): void {
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(28, 28);
  const max = Math.max(...row.map(Math.abs)) || 1;
  for (let i = 0; i < 784; i++) {
    const m = divergingColor(row[i], max).match(/\d+/g)!.map(Number);
    img.data[i * 4] = m[0];
    img.data[i * 4 + 1] = m[1];
    img.data[i * 4 + 2] = m[2];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

const hiddenY = (j: number) => 34 + j * 25;
const outputY = (j: number) => 60 + j * 33;

export function NetworkDiagram() {
  const [weights, setWeights] = useState<PretrainedWeights | null>(null);
  const [mnist, setMnist] = useState<MnistTestSubset | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [selected, setSelected] = useState(0);
  const [results, setResults] = useState<DigitResult[] | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);
  const [unlocked, setUnlocked] = useState(() => codeReady(feedforwardExercise.id));

  const patchRef = useRef<HTMLCanvasElement>(null);
  const digitRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const bigDigitRef = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const inView = useInViewOnce(hostRef);

  useEffect(() => {
    if (!inView) {
      // Still subscribe: the unlock state matters before the assets do.
      return subscribeProgress(() => setUnlocked(codeReady(feedforwardExercise.id)));
    }
    Promise.all([fetchPretrainedWeights(), fetchMnistTest()])
      .then(([w, m]) => {
        setWeights(w);
        setMnist(m);
      })
      .catch((err) => setLoadError(String(err)));
    return subscribeProgress(() => setUnlocked(codeReady(feedforwardExercise.id)));
  }, [inView]);

  useEffect(() => {
    if (!mnist) return;
    digitRefs.current.forEach((canvas, i) => canvas && drawMnistDigit(canvas, mnist.images, i));
  }, [mnist]);

  useEffect(() => {
    if (mnist && bigDigitRef.current) drawMnistDigit(bigDigitRef.current, mnist.images, selected);
  }, [mnist, selected]);

  useEffect(() => {
    if (weights && hovered !== null && patchRef.current) {
      drawPatch(patchRef.current, weights.weights[0][hovered]);
    }
  }, [weights, hovered]);

  const runPayoff = () => {
    if (!weights || !mnist) return;
    const code = loadCode(feedforwardExercise.id);
    if (!code) return;
    setRunning(true);
    setRunError(null);
    const digits = Array.from({ length: N_DIGITS }, (_, i) =>
      Array.from(mnist.images.subarray(i * 784, (i + 1) * 784)),
    );
    sendRequest(
      {
        type: "runPython",
        code: PAYOFF_SNIPPET,
        args: { code, weights: weights.weights, biases: weights.biases, digits },
      },
      (msg) => {
        if (msg.type === "pythonDone") {
          setResults(msg.result as DigitResult[]);
          setRunning(false);
        } else if (msg.type === "cancelled") {
          // Another panel's Stop button also cancels this request.
          setRunning(false);
        } else if (msg.type === "error") {
          setRunError(msg.message);
          setRunning(false);
        }
      },
    );
  };

  if (loadError)
    return (
      <div className="interactive" ref={hostRef}>
        <p className="demo-status demo-status-error">Could not load assets: {loadError}</p>
      </div>
    );
  if (!weights || !mnist)
    return (
      <div className="interactive" ref={hostRef}>
        <p className="demo-status">Loading network and digits...</p>
      </div>
    );

  const current = results?.[selected] ?? null;
  const maxW1 = Math.max(...weights.weights[1].flat().map(Math.abs));

  return (
    <div className="interactive" ref={hostRef}>
      <div className="netdiag">
        <svg
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          className="netdiag-svg"
          role="img"
          aria-label="The 784-15-10 network as a diagram: one box for the 784 inputs, fifteen hidden neurons in a column, and ten output neurons for the digits, wired left to right. Pointing at a neuron shows its incoming weights as a picture and its numbers in the panel beside the diagram."
        >
          <rect x={INPUT_X - 24} y={SVG_H / 2 - 24} width={48} height={48} className="netdiag-input" rx={4} />
          <text x={INPUT_X} y={SVG_H / 2 + 40} textAnchor="middle" className="netdiag-label">
            784 inputs
          </text>
          <text x={HIDDEN_X} y={16} textAnchor="middle" className="netdiag-label">
            15 hidden
          </text>
          <text x={OUTPUT_X} y={16} textAnchor="middle" className="netdiag-label">
            10 outputs
          </text>
          {Array.from({ length: 15 }, (_, j) => (
            <line
              key={`ih${j}`}
              x1={INPUT_X + 24} y1={SVG_H / 2} x2={HIDDEN_X - 10} y2={hiddenY(j)}
              className="netdiag-edge"
            />
          ))}
          {weights.weights[1].map((row, out) =>
            row.map((w, hid) => (
              <line
                key={`ho${out}-${hid}`}
                x1={HIDDEN_X + 10} y1={hiddenY(hid)} x2={OUTPUT_X - 12} y2={outputY(out)}
                stroke={divergingColor(w, maxW1)}
                strokeWidth={0.5 + (Math.abs(w) / maxW1) * 1.5}
                opacity={0.6}
              />
            )),
          )}
          {Array.from({ length: 15 }, (_, j) => (
            <g key={`h${j}`}>
              <circle
                cx={HIDDEN_X} cy={hiddenY(j)} r={9}
                className={`netdiag-neuron ${hovered === j ? "neuron-hover" : ""}`}
                fill={current ? `rgba(11, 110, 79, ${current.hidden[j]})` : "#e8e8e2"}
              />
              {/* Hover alone put this weight image out of reach of every touch
                  screen and every keyboard. Tap or Enter pins it; a second
                  press lets go. The transparent disc is the hit area: 9px is
                  smaller than a fingertip. */}
              <circle
                cx={HIDDEN_X} cy={hiddenY(j)} r={16}
                className="netdiag-hit"
                role="button"
                tabIndex={0}
                aria-pressed={hovered === j}
                aria-label={`Hidden neuron ${j + 1} of 15: show the pixels it looks at`}
                onPointerEnter={(e) => {
                  if (e.pointerType === "mouse") setHovered(j);
                }}
                onPointerLeave={(e) => {
                  if (e.pointerType === "mouse" && pinned !== j) setHovered(null);
                }}
                onClick={() => {
                  const next = pinned === j ? null : j;
                  setPinned(next);
                  setHovered(next);
                }}
                onKeyDown={(e) => {
                  if (e.key !== "Enter" && e.key !== " ") return;
                  e.preventDefault();
                  const next = pinned === j ? null : j;
                  setPinned(next);
                  setHovered(next);
                }}
              />
            </g>
          ))}
          {Array.from({ length: 10 }, (_, j) => (
            <g key={`o${j}`}>
              <circle
                cx={OUTPUT_X} cy={outputY(j)} r={11}
                className={`netdiag-neuron ${current?.pred === j ? "neuron-pred" : ""}`}
                fill={current ? `rgba(11, 110, 79, ${current.output[j]})` : "#e8e8e2"}
              />
              <text x={OUTPUT_X + 22} y={outputY(j) + 4} className="netdiag-label">
                {j}
                {current?.pred === j ? " ←" : ""}
              </text>
            </g>
          ))}
        </svg>
        <div className="netdiag-side">
          {hovered !== null ? (
            <div className="patch-panel">
              <canvas ref={patchRef} width={28} height={28} className="patch-canvas" />
              <p>
                Hidden neuron {hovered + 1}'s 784 incoming weights, arranged as the 28x28
                image they scan. Red pulls the neuron up when that pixel is bright, blue
                pushes it down. Trained neurons look for strokes, curves, and blobs.
              </p>
            </div>
          ) : (
            <p className="patch-hint">Hover a hidden neuron (middle column) to see what it looks for.</p>
          )}
        </div>
      </div>

      <div className="payoff">
        {!unlocked ? (
          <p className="payoff-locked">
            This panel runs your own feedforward, so it needs the exercise below
            passed first. Come back here once its tests are green.
          </p>
        ) : (
          <>
            <div className="interactive-controls">
              <button onClick={runPayoff} disabled={running}>
                {running ? "Running..." : results ? "Run again" : "Run your feedforward on real digits"}
              </button>
              {running && (
                <button className="button-secondary" onClick={terminateWorker}>
                  Stop
                </button>
              )}
              {runError && <span className="demo-status demo-status-error">{runError}</span>}
            </div>
            <div className="digit-strip">
              {Array.from({ length: N_DIGITS }, (_, i) => (
                <button
                  key={i}
                  className={`digit-thumb ${selected === i ? "digit-selected" : ""}`}
                  aria-pressed={selected === i}
                  aria-label={
                    results
                      ? `Digit ${i + 1}: a ${mnist.labels[i]}, read as ${results[i].pred}`
                      : `Digit ${i + 1}: a ${mnist.labels[i]}`
                  }
                  onClick={() => setSelected(i)}
                >
                  <canvas
                    ref={(el) => {
                      digitRefs.current[i] = el;
                    }}
                    width={28} height={28}
                  />
                  {results && (
                    <span className={results[i].pred === mnist.labels[i] ? "pred-ok" : "pred-bad"}>
                      <span aria-hidden="true">
                        {results[i].pred === mnist.labels[i] ? "\u2713" : "\u2717"}
                      </span>
                      {results[i].pred}
                    </span>
                  )}
                </button>
              ))}
            </div>
            <div className="digit-detail">
              <canvas ref={bigDigitRef} width={28} height={28} className="digit-big" />
              {current ? (
                <p>
                  Your feedforward says <strong>{current.pred}</strong> (confidence{" "}
                  {current.output[current.pred].toFixed(2)}); the true label is{" "}
                  <strong>{mnist.labels[selected]}</strong>. The diagram above is showing
                  your code's activations for this digit: darker green means closer to 1.
                </p>
              ) : (
                <p>
                  Run your feedforward, then click any digit: the diagram lights up with
                  the activations your code computes for it.
                </p>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
