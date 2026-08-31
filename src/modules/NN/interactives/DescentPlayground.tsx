import { useEffect, useRef, useState } from "react";
import { scale } from "./utils";

// Chapter 3 interactives (a): gradient descent you can steer. 1D shows the
// step rule and learning-rate overshoot; 2D shows zigzag on an elongated
// valley. Both are pure UI: the real thing gets implemented in the exercise.

function useRunner(step: () => boolean) {
  // Repeatedly call step() every 130ms until it returns false.
  const timer = useRef<number | null>(null);
  const stop = () => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
  };
  const start = () => {
    stop();
    timer.current = window.setInterval(() => {
      if (!step()) stop();
    }, 130);
  };
  useEffect(() => stop, []);
  return { start, stop };
}

// ---------- 1D: f(w) = w^2 ----------

const W1 = 420;
const H1 = 260;
const DOM = 2.4;
const x1 = (w: number) => scale(w, -DOM, DOM, 36, W1 - 12);
const y1 = (f: number) => scale(f, 0, DOM * DOM, H1 - 30, 14);

export function Descent1D() {
  const [eta, setEta] = useState(0.1);
  const [trail, setTrail] = useState<number[]>([-2]);
  const stepsLeft = useRef(0);

  const w = trail[trail.length - 1];
  const diverged = Math.abs(w) > DOM;

  const doStep = () => {
    setTrail((prev) => {
      const cur = prev[prev.length - 1];
      if (Math.abs(cur) > DOM || Math.abs(cur) < 1e-4) return prev;
      return [...prev, cur - eta * 2 * cur];
    });
  };
  const runner = useRunner(() => {
    doStep();
    stepsLeft.current -= 1;
    return stepsLeft.current > 0;
  });

  const curve = Array.from({ length: 97 }, (_, i) => {
    const wv = -DOM + (i / 96) * 2 * DOM;
    return `${i === 0 ? "M" : "L"}${x1(wv).toFixed(1)},${y1(wv * wv).toFixed(1)}`;
  }).join(" ");

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button className="button-secondary" onClick={doStep}>Step</button>
        <button
          className="button-secondary"
          onClick={() => {
            stepsLeft.current = 25;
            runner.start();
          }}
        >
          Run 25 steps
        </button>
        <button
          className="button-secondary"
          onClick={() => {
            runner.stop();
            setTrail([-2]);
          }}
        >
          Reset
        </button>
        <label className="slider-row">
          <span>learning rate η</span>
          <input
            type="range" min={0.05} max={1.15} step={0.05}
            value={eta}
            onChange={(e) => setEta(Number(e.target.value))}
          />
          <code>{eta.toFixed(2)}</code>
        </label>
      </div>
      <svg
        viewBox={`0 0 ${W1} ${H1}`}
        className="interactive-svg"
        role="img"
        aria-label="Cost as a valley over one knob w, with the steps taken so far marked along it. The step size and the current position are set by the controls above and printed beside them."
      >
        <path d={curve} className="curve" />
        <line x1={36} y1={H1 - 30} x2={W1 - 12} y2={H1 - 30} className="axis-line" />
        <text x={W1 - 16} y={H1 - 10} className="chart-tick" textAnchor="end">w</text>
        <text x={40} y={18} className="chart-tick">cost</text>
        {trail.map((tw, i) =>
          Math.abs(tw) <= DOM ? (
            <circle
              key={i}
              cx={x1(tw)} cy={y1(tw * tw)}
              r={i === trail.length - 1 ? 7 : 3.5}
              className={i === trail.length - 1 ? "ball" : "ball-trail"}
            />
          ) : null,
        )}
      </svg>
      <p className="interactive-status status-fixed">
        {diverged
          ? "Diverged: each step overshoots the bottom by more than it descends. Lower η and reset."
          : eta > 0.5
            ? `w = ${w.toFixed(4)}. Overshooting: the ball jumps across the valley every step.`
            : `w = ${w.toFixed(4)}, cost ${(w * w).toFixed(4)}. Steps shrink as the slope flattens.`}
      </p>
    </div>
  );
}

// ---------- 2D: f(w) = (w1^2 + 8 w2^2) / 2 ----------

const W2 = 440;
const H2 = 240;
const DX = 3.4;
const DY = 1.7;
const px = (v: number) => scale(v, -DX, DX, 12, W2 - 12);
const py = (v: number) => scale(v, -DY, DY, H2 - 12, 12);
const LEVELS = [0.25, 0.75, 1.5, 3, 5];

export function Descent2D() {
  const [eta, setEta] = useState(0.16);
  const [start, setStart] = useState<[number, number]>([-2.9, 1.25]);
  const [trail, setTrail] = useState<[number, number][]>([[-2.9, 1.25]]);
  const stepsLeft = useRef(0);
  const svgRef = useRef<SVGSVGElement>(null);

  const doStep = () => {
    setTrail((prev) => {
      const [a, b] = prev[prev.length - 1];
      if (Math.abs(a) > 2 * DX || Math.abs(b) > 2 * DY) return prev;
      return [...prev, [a - eta * a, b - eta * 8 * b]];
    });
  };
  const runner = useRunner(() => {
    doStep();
    stepsLeft.current -= 1;
    return stepsLeft.current > 0;
  });

  const placeStart = (e: React.MouseEvent) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const a = scale(((e.clientX - rect.left) / rect.width) * W2, 12, W2 - 12, -DX, DX);
    const b = scale(((e.clientY - rect.top) / rect.height) * H2, H2 - 12, 12, -DY, DY);
    runner.stop();
    setStart([a, b]);
    setTrail([[a, b]]);
  };

  const cur = trail[trail.length - 1];
  const cost = (cur[0] * cur[0] + 8 * cur[1] * cur[1]) / 2;

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button className="button-secondary" onClick={doStep}>Step</button>
        <button
          className="button-secondary"
          onClick={() => {
            stepsLeft.current = 30;
            runner.start();
          }}
        >
          Run 30 steps
        </button>
        <button
          className="button-secondary"
          onClick={() => {
            runner.stop();
            setTrail([start]);
          }}
        >
          Reset
        </button>
        <label className="slider-row">
          <span>η</span>
          <input
            type="range" min={0.02} max={0.26} step={0.02}
            value={eta}
            onChange={(e) => setEta(Number(e.target.value))}
          />
          <code>{eta.toFixed(2)}</code>
        </label>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W2} ${H2}`}
        className="interactive-svg"
        onClick={placeStart}
        role="img"
        aria-label="A cost surface over two knobs, drawn as contour rings around its lowest point, with the path the steps have taken from the chosen start. Clicking the chart moves the start; the buttons above step and reset it."
      >
        {LEVELS.map((c) => (
          <ellipse
            key={c}
            cx={px(0)} cy={py(0)}
            rx={(Math.sqrt(2 * c) / (2 * DX)) * (W2 - 24)}
            ry={(Math.sqrt(c / 4) / (2 * DY)) * (H2 - 24)}
            className="contour"
          />
        ))}
        <circle cx={px(0)} cy={py(0)} r={3} className="minimum-dot" />
        <text x={W2 - 14} y={H2 - 4} className="chart-tick" textAnchor="end">w₁</text>
        <text x={6} y={14} className="chart-tick">w₂</text>
        <polyline
          points={trail.map(([a, b]) => `${px(a).toFixed(1)},${py(b).toFixed(1)}`).join(" ")}
          className="traj traj-batch"
        />
        <circle cx={px(cur[0])} cy={py(cur[1])} r={6} className="ball" />
      </svg>
      <p className="interactive-status">
        cost {cost.toFixed(4)}. Click anywhere to move the start. The valley is 8 times
        steeper up-down than left-right: raise η and watch the path zigzag across the
        steep direction while barely moving along the flat one.
      </p>
    </div>
  );
}
