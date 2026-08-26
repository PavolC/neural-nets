import { useEffect, useMemo, useRef, useState } from "react";
import { scale, sigmoid } from "./utils";

// Module 7's opening demo: one neuron, one input pinned at 1, right answer 0.
// Descent under the two costs, side by side. Everything here is arithmetic on
// two numbers, so the whole trajectory is computed at once and the button only
// animates how much of it is on screen.

const STEPS = 500;

interface Run {
  answers: number[]; // the neuron's answer after each step, answers[0] = start
  blames: number[]; // the blame the cost assigns at each step
}

/** Descent on one neuron, x = 1 and y = 0, so z = w + b and the slope of
 * both knobs is the same number: the blame. */
function descend(w0: number, b0: number, eta: number, crossEntropy: boolean): Run {
  let w = w0;
  let b = b0;
  const answers: number[] = [];
  const blames: number[] = [];
  for (let step = 0; step <= STEPS; step++) {
    const a = sigmoid(w + b);
    const blame = crossEntropy ? a : a * a * (1 - a);
    answers.push(a);
    blames.push(blame);
    w -= eta * blame;
    b -= eta * blame;
  }
  return { answers, blames };
}

const PRESETS = [
  { label: "a gentle start", w: 0.6, b: 0.9 },
  { label: "a saturated start", w: 2.0, b: 2.0 },
];

const W = 620;
const H = 250;
const PAD = { top: 26, right: 20, bottom: 42, left: 52 };

export function SlowNeuron() {
  const [w, setW] = useState(2.0);
  const [b, setB] = useState(2.0);
  const [eta, setEta] = useState(0.15);
  const [shown, setShown] = useState(STEPS);

  const quad = useMemo(() => descend(w, b, eta, false), [w, b, eta]);
  const cross = useMemo(() => descend(w, b, eta, true), [w, b, eta]);

  // Play: walk `shown` from 0 to STEPS over about two seconds. A ref holds the
  // frame id so a second press restarts rather than running two animations.
  const frame = useRef<number | null>(null);
  const stop = () => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
  };
  useEffect(() => stop, []);
  const play = () => {
    stop();
    setShown(0);
    let step = 0;
    const tick = () => {
      step = Math.min(STEPS, step + 4);
      setShown(step);
      if (step < STEPS) frame.current = requestAnimationFrame(tick);
      else frame.current = null;
    };
    frame.current = requestAnimationFrame(tick);
  };

  // Changing a knob invalidates whatever is drawn, so show the full run again.
  const setKnob = (setter: (v: number) => void) => (v: number) => {
    stop();
    setter(v);
    setShown(STEPS);
  };

  const px = (step: number) => scale(step, 0, STEPS, PAD.left, W - PAD.right);
  const py = (a: number) => scale(a, 0, 1, H - PAD.bottom, PAD.top);
  const path = (run: Run) =>
    run.answers
      .slice(0, shown + 1)
      .map((a, i) => `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(a).toFixed(1)}`)
      .join(" ");

  const start = sigmoid(w + b);
  const rows = [
    { label: "the quadratic cost", run: quad, cls: "slow-line-quad" },
    { label: "the cross-entropy cost", run: cross, cls: "slow-line-cross" },
  ];

  return (
    <div className="interactive">
      <div className="interactive-controls">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            className={`chip ${w === p.w && b === p.b ? "chip-active" : ""}`}
            onClick={() => {
              stop();
              setW(p.w);
              setB(p.b);
              setShown(STEPS);
            }}
          >
            {p.label}
          </button>
        ))}
        <button onClick={play}>Play the training run</button>
      </div>
      <div className="interactive-controls">
        <label className="slider-row">
          <span>weight w</span>
          <input
            type="range"
            min={-1}
            max={4}
            step={0.1}
            value={w}
            onChange={(e) => setKnob(setW)(Number(e.target.value))}
          />
          <code>{w.toFixed(1)}</code>
        </label>
        <label className="slider-row">
          <span>bias b</span>
          <input
            type="range"
            min={-1}
            max={4}
            step={0.1}
            value={b}
            onChange={(e) => setKnob(setB)(Number(e.target.value))}
          />
          <code>{b.toFixed(1)}</code>
        </label>
        <label className="slider-row">
          <span>step size η</span>
          <input
            type="range"
            min={0.05}
            max={0.6}
            step={0.05}
            value={eta}
            onChange={(e) => setKnob(setEta)(Number(e.target.value))}
          />
          <code>{eta.toFixed(2)}</code>
        </label>
        <span className="interactive-status status-fixed">
          Evidence z = w + b = {(w + b).toFixed(1)}, so it starts by answering{" "}
          {start.toFixed(4)}. The right answer is 0. Showing {shown} steps.
        </span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="metrics-chart"
        role="img"
        aria-label={
          `The neuron's answer over ${STEPS} descent steps, one line per cost. ` +
          `From ${start.toFixed(3)} the cross-entropy line falls to ` +
          `${cross.answers[STEPS].toFixed(3)} and the quadratic line to ` +
          `${quad.answers[STEPS].toFixed(3)}.`
        }
      >
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={W - PAD.right} y1={py(v)} y2={py(v)} className="chart-grid" />
            <text x={PAD.left - 8} y={py(v) + 4} className="chart-tick" textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        <text x={PAD.left - 8} y={PAD.top - 14} className="chart-tick">
          answer
        </text>
        {[0, 100, 200, 300, 400, 500].map((s) => (
          <text key={s} x={px(s)} y={H - PAD.bottom + 18} className="chart-tick" textAnchor="middle">
            {s}
          </text>
        ))}
        <text x={(W + PAD.left) / 2} y={H - 8} className="chart-axis-label" textAnchor="middle">
          descent steps (every step nudges both knobs against their slope)
        </text>
        <path d={path(quad)} className="chart-line slow-line-quad" />
        <path d={path(cross)} className="chart-line slow-line-cross" />
      </svg>
      {/* The curves start at the top left, exactly where an in-chart legend
          would sit, so the key goes underneath as HTML. */}
      <ul className="m7-legend">
        <li>
          <span className="m7-swatch slow-line-quad" aria-hidden="true" />
          under the quadratic cost
        </li>
        <li>
          <span className="m7-swatch slow-line-cross" aria-hidden="true" />
          under the cross-entropy cost
        </li>
      </ul>

      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>under</th>
              <th>blame right now</th>
              <th>answer after {shown} steps</th>
              <th>steps to get below 0.1</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const reached = r.run.answers.findIndex((a) => a <= 0.1);
              return (
                <tr key={r.label}>
                  <td>{r.label}</td>
                  <td>{r.run.blames[0].toFixed(4)}</td>
                  <td>{r.run.answers[shown].toFixed(4)}</td>
                  <td>{reached === -1 ? `more than ${STEPS}` : reached}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
