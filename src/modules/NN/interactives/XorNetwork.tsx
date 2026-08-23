import { useEffect, useMemo, useRef, useState } from "react";
import { sigmoid } from "./utils";

// Module 1 interactive (b): a fixed 2-2-1 sigmoid network on the XOR
// points. Sliders adjust all nine parameters; a heatmap shows the output
// over the plane and two lines show where each hidden neuron's input
// crosses zero (its personal boundary carving the plane).

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
const DOMAIN = [-0.45, 1.45];
const CANVAS = 320;

function netOutput(params: Params, x1: number, x2: number): number {
  const h1 = sigmoid(params.w11 * x1 + params.w12 * x2 + params.b1);
  const h2 = sigmoid(params.w21 * x1 + params.w22 * x2 + params.b2);
  return sigmoid(params.v1 * h1 + params.v2 * h2 + params.c);
}

const SLIDER_GROUPS: { title: string; keys: (keyof Params)[] }[] = [
  { title: "Hidden neuron 1", keys: ["w11", "w12", "b1"] },
  { title: "Hidden neuron 2", keys: ["w21", "w22", "b2"] },
  { title: "Output neuron", keys: ["v1", "v2", "c"] },
];

const LABELS: Record<keyof Params, string> = {
  w11: "weight from x1", w12: "weight from x2", b1: "bias",
  w21: "weight from x1", w22: "weight from x2", b2: "bias",
  v1: "weight from h1", v2: "weight from h2", c: "bias",
};

export function XorNetwork() {
  const [params, setParams] = useState<Params>(START);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const corners = useMemo(
    () =>
      XOR_POINTS.map(({ p, label }) => {
        const out = netOutput(params, p[0], p[1]);
        return { p, label, out, ok: (out >= 0.5 ? 1 : 0) === label };
      }),
    [params],
  );
  const solved = corners.every((c) => c.ok);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(RES, RES);
    for (let row = 0; row < RES; row++) {
      const x2 = DOMAIN[1] - ((row + 0.5) / RES) * (DOMAIN[1] - DOMAIN[0]);
      for (let col = 0; col < RES; col++) {
        const x1 = DOMAIN[0] + ((col + 0.5) / RES) * (DOMAIN[1] - DOMAIN[0]);
        const out = netOutput(params, x1, x2);
        const i = (row * RES + col) * 4;
        // 0 = warm sand (class 0), 1 = green (class 1)
        img.data[i] = Math.round(232 - out * 190);
        img.data[i + 1] = Math.round(219 - out * 60);
        img.data[i + 2] = Math.round(191 - out * 90);
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [params]);

  const toPct = (v: number) => ((v - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0])) * 100;

  // Hidden-neuron boundary lines: w1*x1 + w2*x2 + b = 0.
  const boundary = (w1: number, w2: number, b: number): [number, number][] | null => {
    if (Math.abs(w2) < 1e-9 && Math.abs(w1) < 1e-9) return null;
    if (Math.abs(w2) >= Math.abs(w1)) {
      return DOMAIN.map((x1) => [x1, (-b - w1 * x1) / w2]) as [number, number][];
    }
    return DOMAIN.map((x2) => [(-b - w2 * x2) / w1, x2]) as [number, number][];
  };
  const lines = [
    boundary(params.w11, params.w12, params.b1),
    boundary(params.w21, params.w22, params.b2),
  ];

  return (
    <div className="interactive">
      <div className="xor-net">
        <div className="xor-plane">
          <div className="xor-plane-stack">
            <canvas ref={canvasRef} width={RES} height={RES} style={{ width: CANVAS, height: CANVAS }} />
            <svg viewBox="0 0 100 100" preserveAspectRatio="none">
              {lines.map(
                (ln, i) =>
                  ln && (
                    <line
                      key={i}
                      x1={toPct(ln[0][0])} y1={100 - toPct(ln[0][1])}
                      x2={toPct(ln[1][0])} y2={100 - toPct(ln[1][1])}
                      className={i === 0 ? "hidden-line-1" : "hidden-line-2"}
                    />
                  ),
              )}
              {corners.map(({ p, label, ok }, i) => (
                <g key={i}>
                  <circle
                    cx={toPct(p[0])} cy={100 - toPct(p[1])} r={3.2}
                    className={label === 1 ? "pt-class1" : "pt-class0"}
                  />
                  {!ok && (
                    <circle cx={toPct(p[0])} cy={100 - toPct(p[1])} r={4.8} className="pt-wrong" />
                  )}
                </g>
              ))}
            </svg>
          </div>
          <p className={`interactive-status ${solved ? "status-good" : ""}`}>
            {solved
              ? "All four points right: two straight cuts, combined, bent the boundary."
              : `${corners.filter((c) => c.ok).length} of 4 points right.`}
          </p>
          <p className="xor-readout">
            {corners.map(({ p, out }, i) => (
              <span key={i} className="xor-corner">
                ({p[0]},{p[1]}) → {out.toFixed(2)}
              </span>
            ))}
          </p>
        </div>
        <div className="xor-sliders">
          {SLIDER_GROUPS.map((group) => (
            <fieldset key={group.title}>
              <legend>{group.title}</legend>
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
          ))}
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
    </div>
  );
}
