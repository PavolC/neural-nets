import { useEffect, useMemo, useRef, useState } from "react";
import { mulberry32, scale } from "./utils";

// Chapter 3 interactive (b): full-batch gradient descent vs SGD on the same
// cost surface. The surface is the average of 12 per-example bowls; the
// batch step uses the average gradient, the SGD step uses one example's.

const W = 440;
const H = 240;
const DX = 3.4;
const DY = 1.7;
const px = (v: number) => scale(v, -DX, DX, 12, W - 12);
const py = (v: number) => scale(v, -DY, DY, H - 12, 12);
const LEVELS = [0.25, 0.75, 1.5, 3, 5];
const START: [number, number] = [-2.9, 1.25];
const ETA = 0.1;
const N_SAMPLES = 12;

function makeSamples(): [number, number][] {
  const rand = mulberry32(20260823);
  const raw = Array.from({ length: N_SAMPLES }, () => [
    (rand() - 0.5) * 2.4,
    (rand() - 0.5) * 1.0,
  ]);
  // Center so the average bowl bottoms out exactly at the origin.
  const ma = raw.reduce((s, p) => s + p[0], 0) / N_SAMPLES;
  const mb = raw.reduce((s, p) => s + p[1], 0) / N_SAMPLES;
  return raw.map(([a, b]) => [a - ma, b - mb]);
}

export function BatchVsSgd() {
  const samples = useMemo(makeSamples, []);
  const [batchTrail, setBatchTrail] = useState<[number, number][]>([START]);
  const [sgdTrail, setSgdTrail] = useState<[number, number][]>([START]);
  const order = useRef(0);
  const timer = useRef<number | null>(null);
  const stepsLeft = useRef(0);

  const stopRun = () => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
  };
  useEffect(() => stopRun, []);

  const doStep = () => {
    setBatchTrail((prev) => {
      const [a, b] = prev[prev.length - 1];
      return [...prev, [a - ETA * a, b - ETA * 8 * b]];
    });
    setSgdTrail((prev) => {
      const [a, b] = prev[prev.length - 1];
      const [sa, sb] = samples[order.current % N_SAMPLES];
      order.current += 1;
      return [...prev, [a - ETA * (a - sa), b - ETA * 8 * (b - sb)]];
    });
  };

  const run = (n: number) => {
    stopRun();
    stepsLeft.current = n;
    timer.current = window.setInterval(() => {
      doStep();
      stepsLeft.current -= 1;
      if (stepsLeft.current <= 0) stopRun();
    }, 110);
  };

  const reset = () => {
    stopRun();
    order.current = 0;
    setBatchTrail([START]);
    setSgdTrail([START]);
  };

  const steps = batchTrail.length - 1;

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <button className="button-secondary" onClick={doStep}>Step both</button>
        <button className="button-secondary" onClick={() => run(40)}>Run 40 steps</button>
        <button className="button-secondary" onClick={reset}>Reset</button>
        <span className="interactive-status">
          {steps} steps. Examples looked at: batch {steps * N_SAMPLES}, SGD {steps}.
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="interactive-svg"
        role="img"
        aria-label="A cost surface drawn as contour rings around its lowest point, with two descent paths from the same start: the full-batch path, which steps straight downhill, and the mini-batch path, which zigzags around it."
      >
        {LEVELS.map((c) => (
          <ellipse
            key={c}
            cx={px(0)} cy={py(0)}
            rx={(Math.sqrt(2 * c) / (2 * DX)) * (W - 24)}
            ry={(Math.sqrt(c / 4) / (2 * DY)) * (H - 24)}
            className="contour"
          />
        ))}
        <circle cx={px(0)} cy={py(0)} r={3} className="minimum-dot" />
        <text x={W - 14} y={H - 4} className="chart-tick" textAnchor="end">w₁</text>
        <text x={6} y={56} className="chart-tick">w₂</text>
        <polyline
          points={batchTrail.map(([a, b]) => `${px(a).toFixed(1)},${py(b).toFixed(1)}`).join(" ")}
          className="traj traj-batch"
        />
        <polyline
          points={sgdTrail.map(([a, b]) => `${px(a).toFixed(1)},${py(b).toFixed(1)}`).join(" ")}
          className="traj traj-sgd"
        />
        <circle
          cx={px(batchTrail[batchTrail.length - 1][0])}
          cy={py(batchTrail[batchTrail.length - 1][1])}
          r={6} className="ball"
        />
        <circle
          cx={px(sgdTrail[sgdTrail.length - 1][0])}
          cy={py(sgdTrail[sgdTrail.length - 1][1])}
          r={6} className="ball-sgd"
        />
        <g className="chart-legend">
          <rect x={20} y={16} width={14} height={3} className="swatch-batch" />
          <text x={40} y={22} className="chart-tick">full batch (all 12 examples per step)</text>
          <rect x={20} y={32} width={14} height={3} className="swatch-sgd" />
          <text x={40} y={38} className="chart-tick">SGD (1 example per step)</text>
        </g>
      </svg>
      <p className="interactive-status">
        The green path is smooth because every step averages all 12 examples. The orange
        path wobbles because each step trusts a single example, but each of its steps
        costs one twelfth as much: per example looked at, the wobble wins.
      </p>
    </div>
  );
}
