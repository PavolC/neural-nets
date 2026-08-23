import type { EpochMetrics } from "./messages";

// Live training chart: quadratic loss (left axis) and test accuracy (right
// axis) per epoch. Plain SVG, no chart library (see design doc, section 3).

const W = 640;
const H = 280;
const PAD = { top: 16, right: 56, bottom: 36, left: 56 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

function toPath(xs: number[], ys: number[]): string {
  return xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ");
}

export function MetricsChart({ points, totalEpochs }: { points: EpochMetrics[]; totalEpochs: number }) {
  const maxEpoch = Math.max(totalEpochs, 1);
  const maxLoss = Math.max(...points.map((p) => p.loss), 0.001);

  const x = (epoch: number) => PAD.left + ((epoch - 1) / Math.max(maxEpoch - 1, 1)) * PLOT_W;
  const yLoss = (loss: number) => PAD.top + (1 - loss / maxLoss) * PLOT_H;
  const yAcc = (acc: number) => PAD.top + (1 - acc) * PLOT_H;

  const epochs = points.map((p) => x(p.epoch));
  const lossPath = toPath(epochs, points.map((p) => yLoss(p.loss)));
  const accPath = toPath(epochs, points.map((p) => yAcc(p.accuracy)));
  const accTicks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="metrics-chart" role="img"
         aria-label="Training loss and test accuracy per epoch">
      {accTicks.map((t) => (
        <g key={t}>
          <line x1={PAD.left} x2={W - PAD.right} y1={yAcc(t)} y2={yAcc(t)} className="chart-grid" />
          <text x={W - PAD.right + 8} y={yAcc(t) + 4} className="chart-tick chart-tick-acc">
            {Math.round(t * 100)}%
          </text>
          <text x={PAD.left - 8} y={yAcc(t) + 4} className="chart-tick chart-tick-loss" textAnchor="end">
            {(maxLoss * t).toFixed(2)}
          </text>
        </g>
      ))}
      {points.map((p) => (
        <text key={p.epoch} x={x(p.epoch)} y={H - PAD.bottom + 18} className="chart-tick" textAnchor="middle">
          {p.epoch}
        </text>
      ))}
      <text x={W / 2} y={H - 6} className="chart-axis-label" textAnchor="middle">epoch</text>
      {points.length > 0 && (
        <>
          <path d={lossPath} className="chart-line chart-line-loss" />
          <path d={accPath} className="chart-line chart-line-acc" />
          {points.map((p) => (
            <circle key={p.epoch} cx={x(p.epoch)} cy={yAcc(p.accuracy)} r={3} className="chart-dot-acc" />
          ))}
        </>
      )}
      <g className="chart-legend">
        <rect x={PAD.left + 8} y={PAD.top + 4} width={12} height={3} className="chart-swatch-loss" />
        <text x={PAD.left + 26} y={PAD.top + 9} className="chart-tick">training loss</text>
        <rect x={PAD.left + 118} y={PAD.top + 4} width={12} height={3} className="chart-swatch-acc" />
        <text x={PAD.left + 136} y={PAD.top + 9} className="chart-tick">test accuracy</text>
      </g>
    </svg>
  );
}
