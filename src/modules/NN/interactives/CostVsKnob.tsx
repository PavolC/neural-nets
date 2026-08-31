import { useState } from "react";
import { scale } from "./utils";

// Chapter 3 interactive: the true cost of the Chapter 1 XOR network (at its
// slider-start values) along ONE chosen knob, the other eight frozen. Makes
// "a knob's slope, measured where you stand" a picture instead of a claim.
// The numbers must match the prose: cost 0.0876 at the start, slope -0.044
// on the output bias.

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

type Knob = "w11" | "w12" | "w21" | "w22" | "b1" | "b2" | "v1" | "v2" | "c";

const START: Record<Knob, number> = {
  w11: 2, w12: 2, w21: 2, w22: 2, b1: -1, b2: -3, v1: 4, v2: -4, c: -2,
};

// One group per neuron, in the same order as Chapter 1's slider panel and
// the chapter's gradient list: h1, h2, then the output neuron.
const GROUPS: { name: string; short: string; knobs: { key: Knob; label: string }[] }[] = [
  {
    name: "hidden neuron h1",
    short: "h1",
    knobs: [
      { key: "w11", label: "weight from x1" },
      { key: "w12", label: "weight from x2" },
      { key: "b1", label: "bias" },
    ],
  },
  {
    name: "hidden neuron h2",
    short: "h2",
    knobs: [
      { key: "w21", label: "weight from x1" },
      { key: "w22", label: "weight from x2" },
      { key: "b2", label: "bias" },
    ],
  },
  {
    name: "output neuron",
    short: "output",
    knobs: [
      { key: "v1", label: "weight from h1" },
      { key: "v2", label: "weight from h2" },
      { key: "c", label: "bias" },
    ],
  },
];

const FULL_NAME: Record<Knob, string> = Object.fromEntries(
  GROUPS.flatMap((g) => g.knobs.map((k) => [k.key, `${g.short}: ${k.label}`])),
) as Record<Knob, string>;

const CORNERS: [number, number, number][] = [
  [0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0],
];

function costWith(knob: Knob, value: number): number {
  const p = { ...START, [knob]: value };
  let sum = 0;
  for (const [x1, x2, y] of CORNERS) {
    const h1 = sigmoid(p.w11 * x1 + p.w12 * x2 + p.b1);
    const h2 = sigmoid(p.w21 * x1 + p.w22 * x2 + p.b2);
    const out = sigmoid(p.v1 * h1 + p.v2 * h2 + p.c);
    sum += (y - out) * (y - out);
  }
  return sum / (2 * CORNERS.length);
}

const W = 440;
const H = 240;
const HALF_SPAN = 3; // knob axis runs start-3 .. start+3
const Y_MAX = 0.25; // fixed across knobs, so flat knobs look flat

export function CostVsKnob() {
  const [knob, setKnob] = useState<Knob>("c");
  const [value, setValue] = useState(START.c);

  const start = START[knob];
  const lo = start - HALF_SPAN;
  const hi = start + HALF_SPAN;
  const px = (v: number) => scale(v, lo, hi, 44, W - 12);
  const py = (cost: number) => scale(cost, 0, Y_MAX, H - 28, 12);

  const pickKnob = (k: Knob) => {
    setKnob(k);
    setValue(START[k]); // the previous knob goes back to its start value
  };

  const cost = costWith(knob, value);
  const eps = 0.01;
  const slope = (costWith(knob, value + eps) - costWith(knob, value - eps)) / (2 * eps);

  const curve = Array.from({ length: 97 }, (_, i) => {
    const v = lo + (i / 96) * (hi - lo);
    return `${i === 0 ? "M" : "L"}${px(v).toFixed(1)},${py(costWith(knob, v)).toFixed(1)}`;
  }).join(" ");

  // Short tangent segment through the ball, tilted by the measured slope.
  const dv = 0.6;
  const tangent = {
    x1: px(value - dv), y1: py(cost - slope * dv),
    x2: px(value + dv), y2: py(cost + slope * dv),
  };

  const flat = Math.abs(slope) < 0.015;
  const direction = slope < 0 ? "up" : "down";

  return (
    <div className="interactive">
      <div className="interactive-controls knob-picker">
        {GROUPS.map((g) => (
          <div key={g.short} className="knob-group">
            <span className="knob-group-label">{g.name}</span>
            {g.knobs.map(({ key, label }) => (
              <button
                key={key}
                className={`button-secondary chip ${key === knob ? "chip-active" : ""}`}
                onClick={() => pickKnob(key)}
              >
                {label}
              </button>
            ))}
          </div>
        ))}
      </div>
      <div className="interactive-controls">
        <label className="slider-row">
          <span>{FULL_NAME[knob]}</span>
          <input
            type="range" min={lo} max={hi} step={0.01}
            value={value}
            onChange={(e) => setValue(Number(e.target.value))}
          />
          <code>{value.toFixed(2)}</code>
        </label>
        <button className="button-secondary" onClick={() => setValue(start)}>
          Back to start
        </button>
        <span className="interactive-status status-fixed">
          {`cost ${cost.toFixed(4)}, slope here ≈ ${slope.toFixed(3)}: `}
          {flat
            ? "nearly flat, this knob barely matters right now."
            : `push this knob ${direction} to lower the cost.`}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="interactive-svg"
        role="img"
        aria-label="Cost on the upright axis against one knob's value across, drawn as a curve, with the knob's current position marked. The numbers are repeated in the readout beside the chart."
      >
        <line x1={44} y1={H - 28} x2={W - 12} y2={H - 28} className="axis-line" />
        <line x1={44} y1={12} x2={44} y2={H - 28} className="axis-line" />
        <text x={40} y={py(0) + 4} className="chart-tick" textAnchor="end">0</text>
        <text x={40} y={py(0.2) + 4} className="chart-tick" textAnchor="end">0.2</text>
        <text x={12} y={12} className="chart-tick">cost</text>
        <text x={W - 14} y={H - 8} className="chart-tick" textAnchor="end">knob value</text>
        <text x={px(start)} y={H - 8} className="chart-tick" textAnchor="middle">
          {start.toFixed(0)} (start)
        </text>
        <path d={curve} className="curve" />
        <line {...tangent} className="tangent" />
        <circle cx={px(start)} cy={py(costWith(knob, start))} r={3.5} className="minimum-dot" />
        <circle cx={px(value)} cy={py(cost)} r={6.5} className="ball" />
      </svg>
    </div>
  );
}
