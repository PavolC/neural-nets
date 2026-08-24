import { useEffect, useMemo, useRef, useState } from "react";
import { sigmoid } from "./utils";

// Module 1 interactive (b): a 2-2-1 sigmoid network on the XOR points,
// shown as three stacked sections, one per stage of the pipeline. Each
// section holds its chart AND the sliders that own it: phase 1 = input
// space with the hidden layer's cuts and report-tints (hidden sliders),
// phase 2 = hidden space with the output's line and ramp (output sliders),
// combined = the network's verdict plus the scoreboard.

interface Params {
  w11: number; w12: number; b1: number; // hidden neuron 1
  w21: number; w22: number; b2: number; // hidden neuron 2
  v1: number; v2: number; c: number; // output neuron
}

const START: Params = { w11: 2, w12: 2, b1: -1, w21: 2, w22: 2, b2: -3, v1: 4, v2: -4, c: -2 };
const SOLUTION: Params = { w11: 6, w12: 6, b1: -3, w21: 6, w22: 6, b2: -9, v1: 8, v2: -8, c: -4 };

const XOR_POINTS: { p: [number, number]; label: number }[] = [
  { p: [0, 0], label: 0 },
  { p: [1, 1], label: 0 },
  { p: [0, 1], label: 1 },
  { p: [1, 0], label: 1 },
];

const RES = 120; // heatmap resolution
const DOMAIN = [-0.45, 1.45]; // input space
const HDOMAIN = [-0.08, 1.08]; // hidden space (sigmoid outputs live in 0..1)

function hiddenOutputs(params: Params, x1: number, x2: number): [number, number] {
  return [
    sigmoid(params.w11 * x1 + params.w12 * x2 + params.b1),
    sigmoid(params.w21 * x1 + params.w22 * x2 + params.b2),
  ];
}

function outputFromHidden(params: Params, h1: number, h2: number): number {
  return sigmoid(params.v1 * h1 + params.v2 * h2 + params.c);
}

// Verdict shading: 0 = warm sand (class 0), 1 = green (class 1).
function shadeVerdict(img: ImageData, i: number, out: number): void {
  img.data[i] = Math.round(232 - out * 190);
  img.data[i + 1] = Math.round(219 - out * 60);
  img.data[i + 2] = Math.round(191 - out * 90);
  img.data[i + 3] = 255;
}

// Report tint: white pulled toward purple by h1's yes and toward teal by
// h2's yes (the hidden neurons' identity colors; both give a blue blend).
function shadeReports(img: ImageData, i: number, h1: number, h2: number): void {
  const k = 0.42;
  img.data[i] = Math.max(0, Math.round(250 - h1 * k * (250 - 106) - h2 * k * (250 - 27)));
  img.data[i + 1] = Math.max(0, Math.round(250 - h1 * k * (250 - 81) - h2 * k * (250 - 122)));
  img.data[i + 2] = Math.max(0, Math.round(248 - h1 * k * (248 - 163) - h2 * k * (248 - 153)));
  img.data[i + 3] = 255;
}

// Points where w1*u + w2*v + b = 0, clipped to a domain: a neuron's line.
function boundary(
  w1: number, w2: number, b: number,
  dom: number[],
): [number, number][] | null {
  if (Math.abs(w2) < 1e-9 && Math.abs(w1) < 1e-9) return null;
  if (Math.abs(w2) >= Math.abs(w1)) {
    return dom.map((u) => [u, (-b - w1 * u) / w2]) as [number, number][];
  }
  return dom.map((v) => [(-b - w2 * v) / w1, v]) as [number, number][];
}

const SLIDER_GROUPS: {
  title: string;
  swatch: "h1" | "h2" | "out";
  hint: string;
  keys: (keyof Params)[];
}[] = [
  { title: "Hidden neuron 1", swatch: "h1", hint: "the solid purple line", keys: ["w11", "w12", "b1"] },
  { title: "Hidden neuron 2", swatch: "h2", hint: "the dashed teal line", keys: ["w21", "w22", "b2"] },
  { title: "Output neuron", swatch: "out", hint: "the line and the ramp", keys: ["v1", "v2", "c"] },
];

const LABELS: Record<keyof Params, string> = {
  w11: "weight from x1", w12: "weight from x2", b1: "bias",
  w21: "weight from x1", w22: "weight from x2", b2: "bias",
  v1: "weight from h1", v2: "weight from h2", c: "bias",
};

export function XorNetwork() {
  const [params, setParams] = useState<Params>(START);
  const reportsCanvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const combinedCanvasRef = useRef<HTMLCanvasElement>(null);

  const corners = useMemo(
    () =>
      XOR_POINTS.map(({ p, label }) => {
        const [h1, h2] = hiddenOutputs(params, p[0], p[1]);
        const out = outputFromHidden(params, h1, h2);
        return { p, label, h1, h2, out, ok: (out >= 0.5 ? 1 : 0) === label };
      }),
    [params],
  );
  const solved = corners.every((c) => c.ok);

  // Phase 1 canvas: each input point tinted by its pair of reports.
  // Combined canvas: the whole network's verdict at each input point.
  useEffect(() => {
    const reports = reportsCanvasRef.current;
    const combined = combinedCanvasRef.current;
    if (!reports || !combined) return;
    const rctx = reports.getContext("2d")!;
    const cctx = combined.getContext("2d")!;
    const rimg = rctx.createImageData(RES, RES);
    const cimg = cctx.createImageData(RES, RES);
    for (let row = 0; row < RES; row++) {
      const x2 = DOMAIN[1] - ((row + 0.5) / RES) * (DOMAIN[1] - DOMAIN[0]);
      for (let col = 0; col < RES; col++) {
        const x1 = DOMAIN[0] + ((col + 0.5) / RES) * (DOMAIN[1] - DOMAIN[0]);
        const [h1, h2] = hiddenOutputs(params, x1, x2);
        const i = (row * RES + col) * 4;
        shadeReports(rimg, i, h1, h2);
        shadeVerdict(cimg, i, outputFromHidden(params, h1, h2));
      }
    }
    rctx.putImageData(rimg, 0, 0);
    cctx.putImageData(cimg, 0, 0);
  }, [params]);

  // Phase 2 canvas: the output neuron's verdict at every (h1, h2).
  useEffect(() => {
    const canvas = hiddenCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(RES, RES);
    for (let row = 0; row < RES; row++) {
      const h2 = HDOMAIN[1] - ((row + 0.5) / RES) * (HDOMAIN[1] - HDOMAIN[0]);
      for (let col = 0; col < RES; col++) {
        const h1 = HDOMAIN[0] + ((col + 0.5) / RES) * (HDOMAIN[1] - HDOMAIN[0]);
        shadeVerdict(img, (row * RES + col) * 4, outputFromHidden(params, h1, h2));
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [params]);

  const toPct = (v: number) => ((v - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * 100;
  const toPctH = (v: number) => ((v - HDOMAIN[0]) / (HDOMAIN[1] - HDOMAIN[0])) * 100;

  const hiddenLines = [
    boundary(params.w11, params.w12, params.b1, DOMAIN),
    boundary(params.w21, params.w22, params.b2, DOMAIN),
  ];
  const outputLine = boundary(params.v1, params.v2, params.c, HDOMAIN);

  const inputDots = (withRings: boolean) =>
    corners.map(({ p, label, ok }, i) => (
      <g key={i}>
        <circle
          cx={toPct(p[0])} cy={100 - toPct(p[1])} r={3.2}
          className={label === 1 ? "pt-class1" : "pt-class0"}
        />
        {withRings && !ok && (
          <circle cx={toPct(p[0])} cy={100 - toPct(p[1])} r={4.8} className="pt-wrong" />
        )}
      </g>
    ));

  const hiddenLineElements = hiddenLines.map(
    (ln, i) =>
      ln && (
        <line
          key={i}
          x1={toPct(ln[0][0])} y1={100 - toPct(ln[0][1])}
          x2={toPct(ln[1][0])} y2={100 - toPct(ln[1][1])}
          className={i === 0 ? "hidden-line-1" : "hidden-line-2"}
        />
      ),
  );

  const fieldsetFor = (groupIndex: number) => {
    const group = SLIDER_GROUPS[groupIndex];
    return (
      <fieldset key={group.title}>
        <legend>
          <span className={`line-swatch line-swatch-${group.swatch}`} />
          {group.title} ({group.hint})
        </legend>
        {group.keys.map((key) => (
          <label key={key} className="slider-row">
            <span>{LABELS[key]}</span>
            <input
              type="range" min={-10} max={10} step={0.5}
              value={params[key]}
              onChange={(e) => setParams({ ...params, [key]: Number(e.target.value) })}
            />
            <code>{params[key].toFixed(1)}</code>
          </label>
        ))}
      </fieldset>
    );
  };

  return (
    <div className="interactive xor-stack">
      <section className="xor-phase">
        <h4 className="xor-phase-title">
          Phase 1 · The hidden layer cuts input space (x₁ across, x₂ up)
        </h4>
        <p className="xor-space-label">
          The lines are where each hidden neuron's evidence crosses zero. The tint
          is each point's pair of reports:{" "}
          <span className="h1-color">purple where h₁ says yes</span>,{" "}
          <span className="h2-color">teal where h₂ says yes</span>, a blend where
          both, white where neither. These six sliders own this chart.
        </p>
        <div className="xor-phase-body">
          <div className="xor-chart">
            <div className="xor-plane-stack">
              <canvas ref={reportsCanvasRef} width={RES} height={RES} />
              <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                {hiddenLineElements}
                {inputDots(false)}
              </svg>
            </div>
          </div>
          <div className="xor-phase-controls">
            {fieldsetFor(0)}
            {fieldsetFor(1)}
          </div>
        </div>
      </section>

      <section className="xor-phase">
        <h4 className="xor-phase-title">
          Phase 2 · The output neuron cuts hidden space (
          <span className="h1-color">h₁ across</span>,{" "}
          <span className="h2-color">h₂ up</span>)
        </h4>
        <p className="xor-space-label">
          The hidden layer's reports become coordinates, so each dot sits at its
          own (h₁, h₂); the sliders in phase 1 move the dots. The output neuron
          works only here: its line and its sand-to-green confidence ramp belong
          to these three sliders.
        </p>
        <div className="xor-phase-body">
          <div className="xor-chart">
            <div className="xor-plane-stack">
              <canvas ref={hiddenCanvasRef} width={RES} height={RES} />
              <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                {[0, 1].map((v) => (
                  <g key={v}>
                    <line
                      x1={toPctH(v)} x2={toPctH(v)} y1={toPctH(0) - 2} y2={100 - toPctH(0) + 2}
                      className="hidden-grid"
                    />
                    <line
                      x1={toPctH(0) - 2} x2={100 - toPctH(0) + 2}
                      y1={100 - toPctH(v)} y2={100 - toPctH(v)}
                      className="hidden-grid"
                    />
                  </g>
                ))}
                <line x1={0} x2={100} y1={99} y2={99} className="h1-axis-strip" />
                <line x1={1} x2={1} y1={0} y2={100} className="h2-axis-strip" />
                <text x={97} y={95} textAnchor="end" className="axis-id h1-fill">h₁ →</text>
                <text x={4} y={7} className="axis-id h2-fill">h₂ ↑</text>
                {outputLine && (
                  <line
                    x1={toPctH(outputLine[0][0])} y1={100 - toPctH(outputLine[0][1])}
                    x2={toPctH(outputLine[1][0])} y2={100 - toPctH(outputLine[1][1])}
                    className="output-line"
                  />
                )}
                {corners.map(({ h1, h2, label, ok }, i) => (
                  <g key={i}>
                    <circle
                      cx={toPctH(h1)} cy={100 - toPctH(h2)} r={3.2}
                      className={label === 1 ? "pt-class1" : "pt-class0"}
                    />
                    {!ok && (
                      <circle cx={toPctH(h1)} cy={100 - toPctH(h2)} r={4.8} className="pt-wrong" />
                    )}
                  </g>
                ))}
              </svg>
            </div>
          </div>
          <div className="xor-phase-controls">{fieldsetFor(2)}</div>
        </div>
      </section>

      <section className="xor-phase">
        <h4 className="xor-phase-title">Combined · The network's verdict</h4>
        <p className="xor-space-label">
          Back in input space: every point colored by the whole network, which is
          phase 2's color looked up at that point's reports. Green go, sand stay;
          a dashed ring marks a dot the network still gets wrong.
        </p>
        <div className="xor-phase-body">
          <div className="xor-chart">
            <div className="xor-plane-stack">
              <canvas ref={combinedCanvasRef} width={RES} height={RES} />
              <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                {hiddenLineElements}
                {inputDots(true)}
              </svg>
            </div>
          </div>
          <div className="xor-phase-controls">
            <p className={`interactive-status status-fixed ${solved ? "status-good" : ""}`}>
              {solved
                ? "All four points right: the hidden layer re-described the dots until one line could finish."
                : `${corners.filter((c) => c.ok).length} of 4 points right.`}
            </p>
            <p className="xor-readout">
              {corners.map(({ p, out }, i) => (
                <span key={i} className="xor-corner">
                  ({p[0]},{p[1]}) → {out.toFixed(2)}
                </span>
              ))}
            </p>
            <div className="interactive-controls">
              <button className="button-secondary" onClick={() => setParams(START)}>
                Start over
              </button>
              <button className="button-secondary" onClick={() => setParams(SOLUTION)}>
                Show a solution
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
