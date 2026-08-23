import { useRef, useState } from "react";
import { scale } from "./utils";

// Module 1 interactive (a): drag a line and try to separate two classes of
// points. AND and OR are separable; XOR is not, which is the point.

type DatasetId = "XOR" | "OR" | "AND";

const DATASETS: Record<DatasetId, { points: [number, number][]; labels: number[] }> = {
  XOR: { points: [[0, 0], [1, 1], [0, 1], [1, 0]], labels: [0, 0, 1, 1] },
  OR: { points: [[0, 0], [0, 1], [1, 0], [1, 1]], labels: [0, 1, 1, 1] },
  AND: { points: [[0, 0], [0, 1], [1, 0], [1, 1]], labels: [0, 0, 0, 1] },
};

const W = 360;
const H = 320;
const DOMAIN = [-0.45, 1.45];

const toPx = (v: number) => scale(v, DOMAIN[0], DOMAIN[1], 30, W - 10);
const toPy = (v: number) => scale(v, DOMAIN[0], DOMAIN[1], H - 30, 10);
const fromPx = (px: number) => scale(px, 30, W - 10, DOMAIN[0], DOMAIN[1]);
const fromPy = (py: number) => scale(py, H - 30, 10, DOMAIN[0], DOMAIN[1]);

export function SeparatingLine() {
  const [dataset, setDataset] = useState<DatasetId>("XOR");
  // The line is defined by two draggable handles.
  const [handles, setHandles] = useState<[number, number][]>([[-0.2, 0.8], [1.2, 0.4]]);
  const dragging = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const { points, labels } = DATASETS[dataset];
  const [h1, h2] = handles;

  // Which side of the line is each point on?
  const side = (p: [number, number]) =>
    Math.sign((h2[0] - h1[0]) * (p[1] - h1[1]) - (h2[1] - h1[1]) * (p[0] - h1[0]));
  const sides = points.map(side);
  // Best assignment of sides to classes (the learner should not have to
  // care which side means which class).
  const correctA = sides.filter((s, i) => (s >= 0 ? 1 : 0) === labels[i]).length;
  const correct = Math.max(correctA, points.length - correctA);
  const solved = correct === points.length;

  const onPointerMove = (e: React.PointerEvent) => {
    if (dragging.current === null || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = fromPx(((e.clientX - rect.left) / rect.width) * W);
    const y = fromPy(((e.clientY - rect.top) / rect.height) * H);
    const clamp = (v: number) => Math.max(DOMAIN[0], Math.min(DOMAIN[1], v));
    setHandles((prev) => {
      const next: [number, number][] = [...prev];
      next[dragging.current!] = [clamp(x), clamp(y)];
      return next;
    });
  };

  // Extend the segment far past both handles so it reads as a full line.
  const dx = h2[0] - h1[0];
  const dy = h2[1] - h1[1];
  const ext = 10;
  const lineA = [h1[0] - dx * ext, h1[1] - dy * ext];
  const lineB = [h2[0] + dx * ext, h2[1] + dy * ext];

  return (
    <div className="interactive">
      <div className="interactive-controls">
        {(Object.keys(DATASETS) as DatasetId[]).map((id) => (
          <button
            key={id}
            className={`button-secondary chip ${dataset === id ? "chip-active" : ""}`}
            onClick={() => setDataset(id)}
          >
            {id}
          </button>
        ))}
        <span className={`interactive-status ${solved ? "status-good" : ""}`}>
          {solved
            ? `${correct} of ${points.length} separated. Solved!`
            : `${correct} of ${points.length} separated. Drag the line handles.`}
        </span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="interactive-svg"
        onPointerMove={onPointerMove}
        onPointerUp={() => (dragging.current = null)}
        onPointerLeave={() => (dragging.current = null)}
      >
        {[0, 1].map((v) => (
          <g key={v} className="axis-guides">
            <line x1={toPx(v)} y1={toPy(DOMAIN[0])} x2={toPx(v)} y2={toPy(DOMAIN[1])} />
            <line x1={toPx(DOMAIN[0])} y1={toPy(v)} x2={toPx(DOMAIN[1])} y2={toPy(v)} />
            <text x={toPx(v) - 4} y={H - 12}>{v}</text>
            <text x={14} y={toPy(v) + 4}>{v}</text>
          </g>
        ))}
        <line
          x1={toPx(lineA[0])} y1={toPy(lineA[1])} x2={toPx(lineB[0])} y2={toPy(lineB[1])}
          className="sep-line"
        />
        {points.map((p, i) => {
          const assignPositiveTo = correctA >= points.length - correctA ? 1 : 0;
          const predicted = sides[i] >= 0 ? assignPositiveTo : 1 - assignPositiveTo;
          const ok = predicted === labels[i];
          return (
            <g key={i}>
              <circle
                cx={toPx(p[0])} cy={toPy(p[1])} r={9}
                className={labels[i] === 1 ? "pt-class1" : "pt-class0"}
              />
              {!ok && <circle cx={toPx(p[0])} cy={toPy(p[1])} r={13} className="pt-wrong" />}
            </g>
          );
        })}
        {handles.map((h, i) => (
          <circle
            key={i}
            cx={toPx(h[0])} cy={toPy(h[1])} r={7}
            className="sep-handle"
            onPointerDown={(e) => {
              dragging.current = i;
              (e.target as Element).setPointerCapture?.(e.pointerId);
            }}
          />
        ))}
      </svg>
    </div>
  );
}
