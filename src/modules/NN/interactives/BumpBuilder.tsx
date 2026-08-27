import { useState } from "react";
import { scale, sigmoid } from "./utils";

// Module 6 interactive (a): one bump out of two sigmoid neurons. A large
// weight turns each neuron into a step, and opposite weights on the two
// outgoing wires leave a flat-topped bump between them. The readout keeps the
// two neurons' actual weights and biases on screen, because the point of
// the interactive is that the picture and those numbers are one object.

const W = 620;
const H = 300;
const YSPAN = 10.8;
const px = (x: number) => scale(x, 0, 1, 54, W - 16);
const py = (v: number) => scale(v, -YSPAN, YSPAN, H - 34, 14);
const STEEPNESS = [10, 20, 40, 100, 200, 400];

function curve(f: (x: number) => number): string {
  return Array.from({ length: 241 }, (_, i) => {
    const x = i / 240;
    return `${i === 0 ? "M" : "L"}${px(x).toFixed(1)},${py(f(x)).toFixed(1)}`;
  }).join(" ");
}

export function BumpBuilder() {
  const [wi, setWi] = useState(5); // weight 400: the step regime the module builds in
  const [s1, setS1] = useState(0.4);
  const [s2, setS2] = useState(0.6);
  const [h, setH] = useState(6);

  const w = STEEPNESS[wi];
  const up = (x: number) => h * sigmoid(w * (x - s1));
  const down = (x: number) => -h * sigmoid(w * (x - s2));
  const total = (x: number) => up(x) + down(x);
  const switchover = 6 / w;
  const peak = total((s1 + s2) / 2);

  const flatTop = switchover <= Math.abs(s2 - s1) / 2;
  const status =
    h === 0
      ? "Height 0: both outgoing wires carry weight 0, so the total is flat wherever the steps sit."
      : Math.abs(s2 - s1) < 0.001
        ? "The two steps sit on the same spot, so they cancel exactly: the total is flat zero everywhere."
        : s2 > s1
          ? flatTop
            ? `A bump ${peak.toFixed(2)} tall between ${s1.toFixed(2)} and ${s2.toFixed(2)}, flat on top and zero on both sides: both edges are steps.`
            : `A bump between ${s1.toFixed(2)} and ${s2.toFixed(2)}, peaking at ${peak.toFixed(2)} instead of the ${h.toFixed(1)} you set: the two switchovers meet in the middle before either finishes. Raise the steepness for a flat top.`
          : `The steps are in the other order, so the bump points down: a dip ${(-peak).toFixed(2)} deep between ${s2.toFixed(2)} and ${s1.toFixed(2)}.`;

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <label className="slider-row">
          <span>steepness w</span>
          <input
            type="range" min={0} max={STEEPNESS.length - 1} step={1}
            value={wi}
            onChange={(e) => setWi(Number(e.target.value))}
          />
          <code>{w}</code>
        </label>
        <label className="slider-row">
          <span>left step at</span>
          <input
            type="range" min={0} max={1} step={0.05}
            value={s1}
            onChange={(e) => setS1(Number(e.target.value))}
          />
          <code>{s1.toFixed(2)}</code>
        </label>
        <label className="slider-row">
          <span>right step at</span>
          <input
            type="range" min={0} max={1} step={0.05}
            value={s2}
            onChange={(e) => setS2(Number(e.target.value))}
          />
          <code>{s2.toFixed(2)}</code>
        </label>
        <label className="slider-row">
          <span>height h</span>
          <input
            type="range" min={0} max={10} step={0.5}
            value={h}
            onChange={(e) => setH(Number(e.target.value))}
          />
          <code>{h.toFixed(1)}</code>
        </label>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="interactive-svg m6-chart"
        role="img"
        aria-label={`Two sigmoid neurons with steepness ${w}, stepping at ${s1.toFixed(2)} and ${s2.toFixed(2)}, their outgoing wires carrying plus and minus ${h.toFixed(1)}. ${status}`}
      >
        {[-10, -5, 0, 5, 10].map((v) => (
          <g key={v}>
            <line
              x1={px(0)} x2={px(1)} y1={py(v)} y2={py(v)}
              className={v === 0 ? "bump-zero" : "curve-grid"}
            />
            <text x={px(0) - 8} y={py(v) + 4} className="chart-tick" textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((x) => (
          <text key={x} x={px(x)} y={H - 20} className="chart-tick" textAnchor="middle">
            {x}
          </text>
        ))}
        <text x={px(0)} y={H - 5} className="chart-tick" textAnchor="start">freezing</text>
        <text x={px(1)} y={H - 5} className="chart-tick" textAnchor="end">scorching</text>

        <path d={curve(up)} className="bump-part" />
        <path d={curve(down)} className="bump-part" />
        <path d={curve(total)} className="curve-net" />

        <g>
          <line x1={64} y1={26} x2={86} y2={26} className="bump-part" />
          <text x={92} y={30} className="chart-tick">
            each neuron's report, times the weight on its outgoing wire (+h above, −h below)
          </text>
          <line x1={64} y1={44} x2={86} y2={44} className="curve-net" />
          <text x={92} y={48} className="chart-tick">the total: what the network answers</text>
        </g>
      </svg>

      <ul className="bump-readout">
        <li>
          left neuron: weight <code>{w}</code>, bias <code>{(-w * s1).toFixed(1)}</code>, so it
          switches on at <code>{s1.toFixed(2)}</code>
        </li>
        <li>
          right neuron: weight <code>{w}</code>, bias <code>{(-w * s2).toFixed(1)}</code>, so it
          switches on at <code>{s2.toFixed(2)}</code>
        </li>
        <li>
          the outgoing wires: <code>+{h.toFixed(1)}</code> from the left neuron,{" "}
          <code>{(-h).toFixed(1)}</code> from the right
        </li>
        <li>
          each switchover is <code>{switchover.toFixed(3)}</code> of the dial wide (six divided by
          the weight)
        </li>
      </ul>
      <p className="interactive-status status-fixed">{status}</p>
    </div>
  );
}
