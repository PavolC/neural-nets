import { scale } from "./utils";

// Static figure: how big the output layer's blame is, plotted against how
// wrong the answer is. Rendered twice in Module 7, first with the quadratic
// curve alone (the problem) and then with the cross-entropy line added (the
// fix), so the same axes carry both halves of the argument.
//
// Right answer y = 0 throughout, so the answer a IS the gap, and the blames
// are a * a * (1 - a) for the quadratic cost and a for cross-entropy.

const W = 520;
const H = 270;
const PAD = { top: 28, right: 96, bottom: 44, left: 52 };

const MARKS = [0.5, 0.9, 0.98];

export function BlameCurves({ showCrossEntropy = false }: { showCrossEntropy?: boolean }) {
  const px = (a: number) => scale(a, 0, 1, PAD.left, W - PAD.right);
  const py = (v: number) => scale(v, 0, 1, H - PAD.bottom, PAD.top);
  const quad = (a: number) => a * a * (1 - a);
  const path = (f: (a: number) => number) =>
    Array.from({ length: 201 }, (_, i) => {
      const a = i / 200;
      return `${i === 0 ? "M" : "L"}${px(a).toFixed(1)},${py(f(a)).toFixed(1)}`;
    }).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="curve-figure blame-figure"
      role="img"
      aria-label={
        "The blame the output layer receives, against how wrong its answer is. " +
        "The quadratic cost's curve rises to a peak of 0.148 at an answer of two " +
        "thirds and then falls back toward zero as the answer approaches 1." +
        (showCrossEntropy
          ? " The cross-entropy line rises straight to 1: the more wrong the answer, the bigger the blame."
          : "")
      }
    >
      {/* the confidently wrong end of the axis */}
      <rect
        x={px(0.9)}
        y={PAD.top}
        width={px(1) - px(0.9)}
        height={H - PAD.bottom - PAD.top}
        className="blame-zone"
      />
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={PAD.left} x2={W - PAD.right} y1={py(v)} y2={py(v)} className="curve-grid" />
          <text x={PAD.left - 8} y={py(v) + 4} className="chart-tick" textAnchor="end">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={PAD.left - 8} y={PAD.top - 14} className="chart-tick">
        blame
      </text>
      <line x1={PAD.left} x2={W - PAD.right} y1={py(0)} y2={py(0)} className="axis-line" />
      {[0, 0.5, 1].map((a) => (
        <text key={a} x={px(a)} y={py(0) + 18} className="chart-tick" textAnchor="middle">
          {a}
        </text>
      ))}
      <text x={(px(0) + px(1)) / 2} y={H - 8} className="chart-axis-label" textAnchor="middle">
        the answer, when the right answer is 0 (so this is also the gap)
      </text>
      {/* Upright at the top of the band this label lands on the cross-entropy
          line, which passes through the band's top corner, so it stands on its
          side in the clear middle instead. */}
      <text
        x={px(0.95)}
        y={(PAD.top + H - PAD.bottom) / 2}
        transform={`rotate(-90 ${px(0.95)} ${(PAD.top + H - PAD.bottom) / 2})`}
        className="chart-tick"
        textAnchor="middle"
      >
        confidently wrong
      </text>

      {showCrossEntropy && (
        <>
          <path d={path((a) => a)} className="chart-line slow-line-cross" />
          <text x={px(1) + 8} y={py(1) + 4} className="chart-tick blame-label-cross">
            cross-entropy
          </text>
        </>
      )}
      <path d={path(quad)} className="chart-line slow-line-quad" />
      <text x={px(1) + 8} y={py(0.03) + 4} className="chart-tick blame-label-quad">
        quadratic
      </text>

      {/* the peak: past this point, being more wrong means learning less */}
      <line
        x1={px(2 / 3)}
        x2={px(2 / 3)}
        y1={py(quad(2 / 3))}
        y2={py(0)}
        className="blame-peak"
      />
      <text x={px(2 / 3)} y={py(quad(2 / 3)) - 8} className="chart-tick" textAnchor="middle">
        peak 0.148
      </text>
      {MARKS.map((a) => (
        <g key={a}>
          <circle cx={px(a)} cy={py(quad(a))} r={3.2} className="blame-dot-quad" />
          {showCrossEntropy && <circle cx={px(a)} cy={py(a)} r={3.2} className="blame-dot-cross" />}
        </g>
      ))}
    </svg>
  );
}
