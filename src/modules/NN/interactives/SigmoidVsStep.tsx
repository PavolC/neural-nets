import { useState } from "react";
import { scale, sigmoid } from "./utils";

// Module 1 interactive: why sigmoid. The perceptron's hard threshold and
// the sigmoid drawn over the same evidence axis; a slider moves z and both
// outputs respond, one in jumps, one smoothly.

const W = 460;
const H = 240;
const ZMIN = -8;
const ZMAX = 8;
const px = (z: number) => scale(z, ZMIN, ZMAX, 40, W - 16);
const py = (out: number) => scale(out, -0.08, 1.12, H - 34, 12);

export function SigmoidVsStep() {
  const [z, setZ] = useState(-3);

  const sigOut = sigmoid(z);
  const stepOut = z > 0 ? 1 : 0;

  const sigPath = Array.from({ length: 129 }, (_, i) => {
    const zv = ZMIN + (i / 128) * (ZMAX - ZMIN);
    return `${i === 0 ? "M" : "L"}${px(zv).toFixed(1)},${py(sigmoid(zv)).toFixed(1)}`;
  }).join(" ");

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <label className="slider-row slider-row-wide">
          <span>evidence z</span>
          <input
            type="range" min={ZMIN} max={ZMAX} step={0.25}
            value={z}
            onChange={(e) => setZ(Number(e.target.value))}
          />
          <code>{z.toFixed(2)}</code>
        </label>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="interactive-svg">
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={40} x2={W - 16} y1={py(v)} y2={py(v)} className="chart-grid" />
            <text x={34} y={py(v) + 4} className="chart-tick" textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        <line x1={px(0)} x2={px(0)} y1={py(-0.08)} y2={py(1.12)} className="axis-line" />
        <text x={px(0) + 4} y={H - 8} className="chart-tick">z = 0</text>

        {/* perceptron: hard threshold */}
        <line x1={px(ZMIN)} x2={px(0)} y1={py(0)} y2={py(0)} className="step-curve" />
        <line x1={px(0)} x2={px(0)} y1={py(0)} y2={py(1)} className="step-curve step-jump" />
        <line x1={px(0)} x2={px(ZMAX)} y1={py(1)} y2={py(1)} className="step-curve" />

        {/* sigmoid */}
        <path d={sigPath} className="sigmoid-curve" />

        {/* current z marker and both outputs */}
        <line x1={px(z)} x2={px(z)} y1={py(-0.08)} y2={py(1.12)} className="z-marker" />
        <circle cx={px(z)} cy={py(stepOut)} r={6} className="dot-step" />
        <circle cx={px(z)} cy={py(sigOut)} r={6} className="dot-sigmoid" />

        <g className="chart-legend">
          <rect x={52} y={16} width={14} height={3} className="swatch-step" />
          <text x={72} y={22} className="chart-tick">perceptron (hard threshold)</text>
          <rect x={52} y={32} width={14} height={3} className="swatch-sigmoid" />
          <text x={72} y={38} className="chart-tick">sigmoid</text>
        </g>
      </svg>
      <p className="interactive-status status-fixed">
        {`At z = ${z.toFixed(2)}: the perceptron says ${stepOut}, the sigmoid says ${sigOut.toFixed(2)}. `}
        {Math.abs(z) > 2.5
          ? "Far from the boundary they agree; nudges barely matter to either."
          : "Near the boundary, every nudge to z moves the sigmoid a little; the perceptron ignores nudges completely, then flips all at once."}
      </p>
    </div>
  );
}
