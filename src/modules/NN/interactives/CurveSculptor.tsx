import { useMemo, useRef, useState } from "react";
import { scale, sigmoid } from "./utils";

// Module 6 interactive (b): the sculpting playground. The dial runs 0 to 1
// across the chart; the reader sets a height for each slice of it, and each
// slice is one bump, i.e. one pair of hidden sigmoid neurons. The score is
// the area between the target curve and the network's output, shaded so the
// number and the picture are the same thing.

const W = 640;
const H = 336;
const PAD_L = 52;
const PAD_R = 16;
const TOP = 18;
const BASE = H - 48; // where a height of 0 sits
const VMAX = 10.6;
const px = (x: number) => scale(x, 0, 1, PAD_L, W - PAD_R);
const py = (v: number) => scale(v, 0, VMAX, BASE, TOP);

const TN = 161; // target samples, one every 1/160 of the dial
const AG = 641; // grid the area is measured on
const BAR_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24];
const STEEPNESS = [20, 50, 100, 200, 400, 700];
const DEFAULT_BARS = 3; // index: 6 bars
const DEFAULT_STEEP = 4; // index: weight 400

const clampV = (v: number) => Math.max(0, Math.min(10, v));
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

const PRESETS = {
  mine: {
    label: "your curve",
    f: (x: number) => 1.2 + 8.0 * Math.exp(-Math.pow((x - 0.45) / 0.17, 2)),
  },
  friend: {
    label: "your friend's curve",
    f: (x: number) =>
      0.8 +
      6.6 * Math.exp(-Math.pow((x - 0.22) / 0.1, 2)) +
      8.2 * Math.exp(-Math.pow((x - 0.68) / 0.14, 2)),
  },
};
type Kind = keyof typeof PRESETS | "own";

const sampleTarget = (f: (x: number) => number) =>
  Array.from({ length: TN }, (_, j) => clampV(f(j / (TN - 1))));

/** The target between its samples: straight lines, which is what drawing by
 * hand produces anyway. */
function targetAt(t: number[], x: number): number {
  const u = clamp01(x) * (TN - 1);
  const i = Math.min(TN - 2, Math.floor(u));
  return t[i] + (t[i + 1] - t[i]) * (u - i);
}

/** Each bar at the target's average height across its own slice. */
function fitHeights(t: number[], k: number): number[] {
  return Array.from({ length: k }, (_, i) => {
    const S = 40;
    let sum = 0;
    for (let j = 0; j < S; j++) sum += targetAt(t, (i + (j + 0.5) / S) / k);
    return clampV(sum / S);
  });
}

/** The network's answer: one pair of sigmoid neurons per bar, the left one
 * switching on at the bar's left edge with output weight +h, the right one at
 * its right edge with output weight -h. */
function netAt(hs: number[], k: number, w: number, x: number): number {
  let sum = 0;
  for (let i = 0; i < k; i++) {
    if (hs[i] === 0) continue;
    sum += hs[i] * (sigmoid(w * (x - i / k)) - sigmoid(w * (x - (i + 1) / k)));
  }
  return sum;
}

function areaBetween(t: number[], hs: number[], k: number, w: number): number {
  let sum = 0;
  for (let i = 0; i < AG; i++) {
    const x = i / (AG - 1);
    const weight = i === 0 || i === AG - 1 ? 0.5 : 1;
    sum += weight * Math.abs(targetAt(t, x) - netAt(hs, k, w, x));
  }
  return sum / (AG - 1);
}

/** Write value v at index i, filling in every index skipped since the last
 * paint so that a fast drag leaves no gaps. Pure, because React may run a
 * state updater more than once for the same event. */
function paint(
  arr: number[],
  last: { index: number; value: number } | null,
  i: number,
  v: number,
): number[] {
  const next = [...arr];
  const from = last === null ? i : last.index;
  const fromV = last === null ? v : last.value;
  for (let j = Math.min(from, i); j <= Math.max(from, i); j++) {
    const u = from === i ? 1 : (j - from) / (i - from);
    next[j] = clampV(fromV + (v - fromV) * Math.max(0, Math.min(1, u)));
  }
  next[i] = v;
  return next;
}

export function CurveSculptor() {
  const [kind, setKind] = useState<Kind>("mine");
  const [target, setTarget] = useState<number[]>(() => sampleTarget(PRESETS.mine.f));
  const [ki, setKi] = useState(DEFAULT_BARS);
  const [heights, setHeights] = useState<number[]>(() =>
    new Array(BAR_COUNTS[DEFAULT_BARS]).fill(0),
  );
  const [wi, setWi] = useState(DEFAULT_STEEP);
  const [mode, setMode] = useState<"bars" | "target">("bars");

  const bars = BAR_COUNTS[ki];
  const steep = STEEPNESS[wi];

  const svgRef = useRef<SVGSVGElement>(null);
  const painting = useRef(false);
  const pressedAt = useRef<{ x: number; y: number } | null>(null);
  const lastPaint = useRef<{ index: number; value: number } | null>(null);

  const { netPath, targetPath, gapPath, area } = useMemo(() => {
    const netPts = Array.from({ length: 321 }, (_, i) => {
      const x = i / 320;
      return `${i === 0 ? "M" : "L"}${px(x).toFixed(1)},${py(netAt(heights, bars, steep, x)).toFixed(1)}`;
    }).join(" ");
    const tgtPts = target
      .map((v, j) => `${j === 0 ? "M" : "L"}${px(j / (TN - 1)).toFixed(1)},${py(v).toFixed(1)}`)
      .join(" ");
    // The band between the curves as one quad per sample interval: they sit
    // side by side, so a single path fills correctly even where the two
    // curves cross.
    const quads: string[] = [];
    for (let j = 0; j < TN - 1; j++) {
      const xa = j / (TN - 1);
      const xb = (j + 1) / (TN - 1);
      const ta = py(target[j]);
      const tb = py(target[j + 1]);
      const na = py(netAt(heights, bars, steep, xa));
      const nb = py(netAt(heights, bars, steep, xb));
      quads.push(
        `M${px(xa).toFixed(1)},${ta.toFixed(1)}L${px(xb).toFixed(1)},${tb.toFixed(1)}` +
          `L${px(xb).toFixed(1)},${nb.toFixed(1)}L${px(xa).toFixed(1)},${na.toFixed(1)}Z`,
      );
    }
    return {
      netPath: netPts,
      targetPath: tgtPts,
      gapPath: quads.join(""),
      area: areaBetween(target, heights, bars, steep),
    };
  }, [target, heights, bars, steep]);

  const changeBars = (next: number) => {
    const nextBars = BAR_COUNTS[next];
    // Keep the shape that is already sculpted: each new bar starts at the
    // height the old bars had over its middle.
    setHeights((prev) =>
      Array.from({ length: nextBars }, (_, i) => {
        const center = (i + 0.5) / nextBars;
        return prev[Math.min(prev.length - 1, Math.floor(center * prev.length))];
      }),
    );
    setKi(next);
  };

  const pickTarget = (next: Kind) => {
    setKind(next);
    if (next === "own") {
      setMode("target");
    } else {
      setTarget(sampleTarget(PRESETS[next].f));
      setMode("bars");
    }
  };

  const applyAt = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const x = clamp01(
      (((clientX - rect.left) / rect.width) * W - PAD_L) / (W - PAD_R - PAD_L),
    );
    const v = clampV(((((clientY - rect.top) / rect.height) * H - BASE) / (TOP - BASE)) * VMAX);
    const last = lastPaint.current;
    if (mode === "bars") {
      const i = Math.min(bars - 1, Math.floor(x * bars));
      setHeights((prev) => paint(prev, last, i, v));
      lastPaint.current = { index: i, value: v };
    } else {
      const i = Math.round(x * (TN - 1));
      setTarget((prev) => paint(prev, last, i, v));
      lastPaint.current = { index: i, value: v };
      setKind("own");
    }
  };

  // Mouse drags paint continuously. On touch, a drag is the page's scroll
  // gesture everywhere except in draw mode, so a tap (press and release
  // without moving) is what sets a value.
  const onPointerDown = (e: React.PointerEvent) => {
    pressedAt.current = { x: e.clientX, y: e.clientY };
    lastPaint.current = null;
    if (e.pointerType === "mouse" || mode === "target") {
      painting.current = true;
      // Capture keeps a drag that wanders off the plot alive. It throws if the
      // pointer is already gone, which must not take the paint with it.
      try {
        (e.target as Element).setPointerCapture?.(e.pointerId);
      } catch {
        /* no capture, and the drag still works while the pointer is inside */
      }
      applyAt(e.clientX, e.clientY);
    }
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (painting.current) applyAt(e.clientX, e.clientY);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const from = pressedAt.current;
    if (!painting.current && from) {
      const moved = Math.hypot(e.clientX - from.x, e.clientY - from.y);
      if (moved < 10) applyAt(e.clientX, e.clientY);
    }
    painting.current = false;
    pressedAt.current = null;
    lastPaint.current = null;
  };

  const nudgeBar = (i: number, delta: number) =>
    setHeights((prev) => {
      const next = [...prev];
      next[i] = clampV(next[i] + delta);
      return next;
    });

  const targetLabel = kind === "own" ? "your own drawing" : PRESETS[kind].label;

  return (
    <div className="interactive">
      <div className="interactive-controls">
        <span className="sculpt-group-label">Curve to match</span>
        {(["mine", "friend", "own"] as Kind[]).map((id) => (
          <button
            key={id}
            className={`button-secondary chip ${kind === id ? "chip-active" : ""}`}
            aria-pressed={kind === id}
            onClick={() => pickTarget(id)}
          >
            {id === "own" ? "draw your own" : PRESETS[id].label}
          </button>
        ))}
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="interactive-svg m6-chart"
        aria-label={`The curve to match is ${targetLabel}. Your network has ${bars} bars, ${2 * bars} hidden neurons, and the area between the two curves is ${area.toFixed(3)}.`}
      >
        {[0, 2, 4, 6, 8, 10].map((v) => (
          <g key={v}>
            <line x1={px(0)} x2={px(1)} y1={py(v)} y2={py(v)} className="curve-grid" />
            <text x={px(0) - 8} y={py(v) + 4} className="chart-tick" textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        <text x={px(0) - 8} y={TOP + 2} className="chart-tick" textAnchor="end">
          rating
        </text>
        {[0, 0.25, 0.5, 0.75, 1].map((x) => (
          <text key={x} x={px(x)} y={BASE + 20} className="chart-tick" textAnchor="middle">
            {x}
          </text>
        ))}
        <text x={px(0)} y={BASE + 36} className="chart-tick" textAnchor="start">
          freezing
        </text>
        <text x={px(0.5)} y={BASE + 36} className="chart-tick" textAnchor="middle">
          the forecast dial
        </text>
        <text x={px(1)} y={BASE + 36} className="chart-tick" textAnchor="end">
          scorching
        </text>

        <path d={gapPath} className="curve-gap" />

        {/* the bars: each one is a pair of hidden neurons */}
        {heights.map((h, i) => (
          <rect
            key={`bar${i}`}
            x={px(i / bars)}
            y={py(h)}
            width={px((i + 1) / bars) - px(i / bars)}
            height={Math.max(0, BASE - py(h))}
            className="sculpt-bar"
          />
        ))}

        <path d={targetPath} className="curve-target" />
        <path d={netPath} className="curve-net" />
        <line x1={px(0)} x2={px(1)} y1={BASE} y2={BASE} className="axis-line" />

        {/* the draggable top edge of each bar, and its keyboard slider */}
        {heights.map((h, i) => (
          <rect
            key={`cap${i}`}
            x={px(i / bars)}
            y={py(h) - 5}
            width={px((i + 1) / bars) - px(i / bars)}
            height={10}
            className="sculpt-cap drag-handle"
            tabIndex={0}
            role="slider"
            aria-label={`bar ${i + 1} of ${bars}, height`}
            aria-valuemin={0}
            aria-valuemax={10}
            aria-valuenow={Number(h.toFixed(1))}
            onKeyDown={(e) => {
              const step = e.shiftKey ? 1 : 0.5;
              if (e.key === "ArrowUp" || e.key === "ArrowRight") nudgeBar(i, step);
              else if (e.key === "ArrowDown" || e.key === "ArrowLeft") nudgeBar(i, -step);
              else if (e.key === "Home") nudgeBar(i, -10);
              else if (e.key === "End") nudgeBar(i, 10);
              else return;
              e.preventDefault();
            }}
          />
        ))}

        {/* the whole plot is the pointer surface, over everything else */}
        <rect
          x={px(0)}
          y={TOP - 6}
          width={px(1) - px(0)}
          height={BASE - TOP + 6}
          className={`sculpt-surface ${mode === "target" ? "drag-handle" : ""}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        />
      </svg>

      <p className="sculpt-legend">
        Dashed line: the curve to match. Green line: what your network answers. Shaded
        bars: the bump each pair of neurons adds. Red shading: the area between the two
        curves, which is the score.
      </p>
      {/* The changing half of the key gets its own line with a reserved
          height, so switching modes never reflows the chart above it. */}
      <p className="interactive-status status-fixed">
        {mode === "bars"
          ? "Drag inside the chart to set the bars; on a touch screen, a tap sets one bar."
          : "Drag inside the chart to draw the curve you want to match."}
      </p>

      <ul className="sculpt-stats">
        <li>
          <b>{bars}</b> bars
        </li>
        <li>
          <b>{2 * bars}</b> hidden neurons
        </li>
        <li>
          <b>{6 * bars}</b> numbers in the network
        </li>
        <li className="sculpt-score">
          area between the curves: <b>{area.toFixed(3)}</b>
        </li>
      </ul>

      <div className="interactive-controls">
        <label className="slider-row">
          <span>bars</span>
          <input
            type="range" min={0} max={BAR_COUNTS.length - 1} step={1}
            value={ki}
            onChange={(e) => changeBars(Number(e.target.value))}
          />
          <code>{bars}</code>
        </label>
        <label className="slider-row">
          <span>step sharpness</span>
          <input
            type="range" min={0} max={STEEPNESS.length - 1} step={1}
            value={wi}
            onChange={(e) => setWi(Number(e.target.value))}
          />
          <code>{steep}</code>
        </label>
        <button
          className="button-secondary"
          onClick={() => setHeights(fitHeights(target, bars))}
        >
          Fit it for me
        </button>
        <button
          className="button-secondary"
          onClick={() => setHeights(new Array(bars).fill(0))}
        >
          Flatten
        </button>
        <button
          className={`button-secondary chip ${mode === "bars" ? "chip-active" : ""}`}
          aria-pressed={mode === "bars"}
          onClick={() => setMode("bars")}
        >
          drag sets the bars
        </button>
        <button
          className={`button-secondary chip ${mode === "target" ? "chip-active" : ""}`}
          aria-pressed={mode === "target"}
          onClick={() => setMode("target")}
        >
          drag draws the curve
        </button>
      </div>
    </div>
  );
}
