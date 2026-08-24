import { useMemo, useState } from "react";
import { Eq } from "../../../components/Math";
import { divergingColor } from "./utils";
import {
  START,
  cloneNet,
  compute,
  fmt,
  getParam,
  getSlope,
  paramLabel,
  setParam,
  type Net,
  type ParamRef,
  type Trace,
} from "./backpropNet";

// Module 4 centerpiece: a 2-3-1 network stepped through one forward pass and
// one backward pass. Each step reveals the numbers it computes and shows the
// equation being applied; the learner can select any of the 13 parameters
// (click a wire or a neuron circle) and nudge it, watching every revealed
// number recompute live.

interface Step {
  title: string;
  tex: string;
  gloss: (t: Trace) => string;
}

const STEPS: Step[] = [
  {
    title: "The setup",
    tex: "x = \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix}, \\qquad y = 1",
    gloss: () =>
      "Good weather (x₁ = 1), no friend (x₂ = 0); the contrarian goes, so the target is y = 1. " +
      "Press Next to watch the network answer, then keep going to watch the blame flow back.",
  },
  {
    title: "Forward: the hidden layer",
    tex: "z^2 = w^2 x + b^2, \\qquad a^2 = \\sigma(z^2)",
    gloss: (t) =>
      `Three multiply-adds at once (Module 2's one line), then sigmoid on each entry: ` +
      `h₁: z = ${fmt(t.z2[0])} → a = ${fmt(t.a2[0])}; ` +
      `h₂: z = ${fmt(t.z2[1])} → a = ${fmt(t.a2[1])}; ` +
      `h₃: z = ${fmt(t.z2[2])} → a = ${fmt(t.a2[2])}.`,
  },
  {
    title: "Forward: the output layer",
    tex: "z^3 = w^3 a^2 + b^3, \\qquad a^3 = \\sigma(z^3)",
    gloss: (t) =>
      `The output neuron reads the three hidden confidences: z³ = ${fmt(t.z3[0])}, ` +
      `so a³ = ${fmt(t.a3[0])}. The right answer is 1, so the gap a³ − y is ${fmt(t.a3[0] - 1)}.`,
  },
  {
    title: "The cost",
    tex: "C = \\tfrac{1}{2}\\,(y - a^3)^2",
    gloss: (t) =>
      `C = ½ × (${fmt(1 - t.a3[0])})² = ${fmt(t.cost, 4)}. One number to shrink; ` +
      `every one of the 13 knobs is about to be judged by how it moves this.`,
  },
  {
    title: "BP1: blame starts at the output",
    tex: "\\delta^3 = (a^3 - y) \\odot \\sigma'(z^3)",
    gloss: (t) =>
      `Gap ${fmt(t.a3[0] - 1)} times σ′(z³) = ${fmt(t.a3[0] * (1 - t.a3[0]))} gives ` +
      `δ³ = ${fmt(t.d3[0])} (the circled dot means multiply matching entries; each side here has ` +
      `just one). Negative blame reads: raising this neuron's evidence would lower the cost.`,
  },
  {
    title: "BP2: blame flows backward",
    tex: "\\delta^2 = \\big((w^3)^T \\delta^3\\big) \\odot \\sigma'(z^2)",
    gloss: (t) =>
      `Each hidden neuron receives δ³ scaled by its outgoing wire, then multiplies by its own σ′: ` +
      `δ = ${fmt(t.d2[0])}, ${fmt(t.d2[1])}, ${fmt(t.d2[2])}. The raised T (transpose) flips w³ so ` +
      `the same wires read backward. Watch h₂: its wire is negative, so blame arrives sign-flipped.`,
  },
  {
    title: "BP3 and BP4: every slope, read off",
    tex: "\\frac{\\partial C}{\\partial b^l} = \\delta^l, \\qquad \\frac{\\partial C}{\\partial w^l} = \\delta^l (a^{l-1})^T",
    gloss: () =>
      "All 13 slopes at once (∂C/∂b is read as one name: the slope of C per nudge of b, Module 3's " +
      "nudge-and-divide number). A bias's slope is its neuron's blame; a wire's slope is the " +
      "receiver's blame times the activation the wire carried. Select any knob (click a wire or " +
      "circle) to read its slope below, then drag its slider and watch every number react.",
  },
];

// Reveal thresholds: the first step at which each quantity is shown.
const REVEAL = { hidden: 1, output: 2, cost: 3, d3: 4, d2: 5, slopes: 6 };

// Diagram geometry (SVG user units).
const IN_POS = [
  { x: 85, y: 117 },
  { x: 85, y: 227 },
];
const HID_POS = [
  { x: 315, y: 55 },
  { x: 315, y: 172 },
  { x: 315, y: 289 },
];
const OUT_POS = { x: 545, y: 172 };
const R = 27;

const sameRef = (a: ParamRef | null, b: ParamRef): boolean =>
  a !== null &&
  a.kind === b.kind &&
  a.j === b.j &&
  ((a.kind !== "W2" && a.kind !== "W3") || a.k === (b as { k: number }).k);

export function BackpropStepper() {
  const [net, setNet] = useState<Net>(() => cloneNet(START));
  const [step, setStep] = useState(0);
  const [sel, setSel] = useState<ParamRef | null>(null);

  const trace = useMemo(() => compute(net), [net]);
  const stepDef = STEPS[step];

  const show = {
    hidden: step >= REVEAL.hidden,
    output: step >= REVEAL.output,
    cost: step >= REVEAL.cost,
    d3: step >= REVEAL.d3,
    d2: step >= REVEAL.d2,
    slopes: step >= REVEAL.slopes,
  };

  // One wire: colored by weight, labeled with its value, clickable to select.
  const edge = (
    from: { x: number; y: number },
    to: { x: number; y: number },
    weight: number,
    ref: ParamRef,
    labelT: number,
  ) => {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const len = Math.hypot(dx, dy);
    const ux = dx / len;
    const uy = dy / len;
    const x1 = from.x + ux * R;
    const y1 = from.y + uy * R;
    const x2 = to.x - ux * R;
    const y2 = to.y - uy * R;
    const lx = x1 + (x2 - x1) * labelT - uy * 11;
    const ly = y1 + (y2 - y1) * labelT + ux * 11;
    const selected = sameRef(sel, ref);
    const key = `${ref.kind}-${ref.j}-${"k" in ref ? ref.k : ""}`;
    return (
      <g key={key}>
        {selected && <line x1={x1} y1={y1} x2={x2} y2={y2} className="bp-edge-halo" />}
        <line
          x1={x1} y1={y1} x2={x2} y2={y2}
          className="bp-edge"
          stroke={divergingColor(weight, 5)}
          strokeWidth={1.2 + Math.min(3.6, Math.abs(weight) * 0.7)}
        />
        <line
          x1={x1} y1={y1} x2={x2} y2={y2}
          className="bp-edge-hit"
          onClick={() => setSel(ref)}
        />
        <text x={lx} y={ly} className="bp-wlabel" textAnchor="middle">
          {fmt(weight, 1)}
        </text>
      </g>
    );
  };

  // One neuron: circle fills with its activation once revealed; bias under it,
  // blame under that. Clicking the circle selects the bias.
  const neuron = (
    pos: { x: number; y: number },
    name: string,
    bias: number,
    biasRef: ParamRef,
    a: number | null,
    delta: number | null,
  ) => {
    const selected = sameRef(sel, biasRef);
    const fill = a === null ? "#ffffff" : `rgba(11, 110, 79, ${0.12 + a * 0.68})`;
    return (
      <g key={name} onClick={() => setSel(biasRef)} className="bp-neuron-group">
        <text x={pos.x} y={pos.y - R - 9} textAnchor="middle" className="bp-name">
          {name}
        </text>
        <circle
          cx={pos.x} cy={pos.y} r={R}
          className={`bp-neuron ${selected ? "bp-node-selected" : ""}`}
          style={{ fill }}
        />
        <text
          x={pos.x} y={pos.y + 5} textAnchor="middle"
          className={`bp-avalue ${a !== null && a > 0.55 ? "bp-avalue-dark" : ""}`}
        >
          {a === null ? "" : fmt(a)}
        </text>
        <text x={pos.x} y={pos.y + R + 17} textAnchor="middle" className="bp-bias">
          b = {fmt(bias, 1)}
        </text>
        <text x={pos.x} y={pos.y + R + 33} textAnchor="middle" className="bp-delta">
          {delta === null ? "" : `δ = ${fmt(delta)}`}
        </text>
      </g>
    );
  };

  const selValue = sel === null ? null : getParam(net, sel);
  const selSlope = sel === null ? null : getSlope(trace, sel);

  return (
    <div className="interactive bp-stepper">
      <div className="interactive-controls">
        <button
          className="button-secondary"
          disabled={step === 0}
          onClick={() => setStep((s) => Math.max(0, s - 1))}
        >
          ◀ Back
        </button>
        <button
          className="button-secondary bp-next"
          disabled={step === STEPS.length - 1}
          onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))}
        >
          Next ▶
        </button>
        <button
          className="button-secondary"
          onClick={() => {
            setNet(cloneNet(START));
          }}
        >
          Reset weights
        </button>
        <span className="bp-step-count">
          {step === 0 ? "Start" : `Step ${step} of ${STEPS.length - 1}`}
        </span>
      </div>

      <div className="bp-eqcard">
        <p className="bp-step-title">{stepDef.title}</p>
        <Eq tex={stepDef.tex} gloss={stepDef.gloss(trace)} />
      </div>

      <svg
        viewBox="0 0 672 362"
        className="interactive-svg"
        role="img"
        aria-label="A 2-3-1 network: two inputs feed three hidden neurons, which feed one output neuron. Wires and circles are clickable."
      >
        {/* wires first, under the nodes */}
        {/* Labels sit close to the sources (t = 0.22), where the fan of wires
            is still spread out; in the middle the wires cross and labels
            collide. */}
        {HID_POS.map((h, j) =>
          IN_POS.map((inp, k) =>
            edge(inp, h, net.W2[j][k], { kind: "W2", j, k }, 0.22),
          ),
        )}
        {HID_POS.map((h, k) =>
          edge(h, OUT_POS, net.W3[0][k], { kind: "W3", j: 0, k }, 0.4),
        )}

        {/* inputs: plain numbers, not neurons */}
        {IN_POS.map((pos, i) => (
          <g key={`in${i}`}>
            <text x={pos.x} y={pos.y - R - 9} textAnchor="middle" className="bp-name">
              {i === 0 ? "x₁ (weather)" : "x₂ (friend)"}
            </text>
            <circle cx={pos.x} cy={pos.y} r={R} className="bp-input" />
            <text x={pos.x} y={pos.y + 5} textAnchor="middle" className="bp-avalue">
              {i === 0 ? "1" : "0"}
            </text>
          </g>
        ))}

        {/* hidden neurons */}
        {HID_POS.map((pos, j) =>
          neuron(
            pos,
            ["h₁", "h₂", "h₃"][j],
            net.b2[j],
            { kind: "b2", j },
            show.hidden ? trace.a2[j] : null,
            show.d2 ? trace.d2[j] : null,
          ),
        )}

        {/* output neuron */}
        {neuron(
          OUT_POS,
          "out",
          net.b3[0],
          { kind: "b3", j: 0 },
          show.output ? trace.a3[0] : null,
          show.d3 ? trace.d3[0] : null,
        )}

        {/* target and cost, to the right of the output */}
        <text x={OUT_POS.x + R + 14} y={OUT_POS.y - 6} className="bp-name">
          y = 1
        </text>
        <text x={OUT_POS.x + R + 14} y={OUT_POS.y + 14} className="bp-cost">
          {show.cost ? `C = ${fmt(trace.cost, 4)}` : ""}
        </text>
      </svg>

      <div className="bp-lower">
        <div className="bp-table-scroll">
        <table className="bp-table">
          <thead>
            <tr>
              <th>neuron</th>
              <th>evidence z</th>
              <th>activation a</th>
              <th>blame δ</th>
            </tr>
          </thead>
          <tbody>
            {[0, 1, 2].map((j) => (
              <tr key={j}>
                <td>{["h₁", "h₂", "h₃"][j]}</td>
                <td>{show.hidden ? fmt(trace.z2[j]) : "·"}</td>
                <td>{show.hidden ? fmt(trace.a2[j]) : "·"}</td>
                <td className="bp-delta-cell">{show.d2 ? fmt(trace.d2[j]) : "·"}</td>
              </tr>
            ))}
            <tr>
              <td>out</td>
              <td>{show.output ? fmt(trace.z3[0]) : "·"}</td>
              <td>{show.output ? fmt(trace.a3[0]) : "·"}</td>
              <td className="bp-delta-cell">{show.d3 ? fmt(trace.d3[0]) : "·"}</td>
            </tr>
            <tr>
              <td colSpan={4} className="bp-cost-row">
                cost C = {show.cost ? fmt(trace.cost, 4) : "·"}
              </td>
            </tr>
          </tbody>
        </table>
        </div>

        <div className="bp-param">
          {sel === null || selValue === null ? (
            <p className="bp-param-hint">
              Click any wire or neuron circle in the diagram to select one of the
              13 knobs; a slider will appear here to nudge it.
            </p>
          ) : (
            <>
              <p className="bp-param-name">{paramLabel(sel)}</p>
              <label className="slider-row">
                <span>value</span>
                <input
                  type="range" min={-8} max={8} step={0.1}
                  value={selValue}
                  onChange={(e) => setNet(setParam(net, sel, Number(e.target.value)))}
                />
                <code>{fmt(selValue, 1)}</code>
              </label>
              <p className="bp-param-slope">
                {show.slopes
                  ? `slope of the cost for this knob: ${fmt(selSlope!, 4)}`
                  : "its slope appears at the last step"}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
