// Shared chart for Chapter 7's training panels: several runs of the same
// network, one line each, plotted against the epoch. Kept separate from the
// panels because three of them draw exactly this picture with different
// series in it. The legend is HTML below the plot rather than text inside it,
// so four entries can wrap instead of colliding with the top of the chart.

export interface EpochSeries {
  key: string;
  label: string;
  /** Stroke class, one of the m7-line-* rules in styles.css. */
  cls: string;
  dashed?: boolean;
  /** One value per epoch, in order, starting at epoch 1. */
  values: number[];
}

const W = 640;
const H = 250;
const PAD = { top: 30, right: 24, bottom: 40, left: 58 };

export function EpochChart({
  series,
  epochs,
  yMin = 0,
  yMax,
  yTicks,
  yLabel,
  xLabel,
  ariaLabel,
}: {
  series: EpochSeries[];
  epochs: number;
  /** Bottom of the value axis. Accuracy runs never come near zero once the
   * network works at all, so those charts start their axis where the data
   * does instead of wasting half the plot. */
  yMin?: number;
  yMax: number;
  /** Tick values in data units, already formatted as they should read. */
  yTicks: { at: number; label: string }[];
  yLabel: string;
  xLabel: string;
  ariaLabel: string;
}) {
  const px = (epoch: number) =>
    PAD.left + ((epoch - 1) / Math.max(epochs - 1, 1)) * (W - PAD.left - PAD.right);
  // Clamped to the axis: a value outside [yMin, yMax] would otherwise draw
  // outside the plot box, and these axes are floored deliberately.
  const py = (v: number) => {
    const t = Math.min(1, Math.max(0, (v - yMin) / (yMax - yMin)));
    return PAD.top + (1 - t) * (H - PAD.top - PAD.bottom);
  };
  const path = (values: number[]) =>
    values
      .map((v, i) => `${i === 0 ? "M" : "L"}${px(i + 1).toFixed(1)},${py(v).toFixed(1)}`)
      .join(" ");

  // Epoch labels thin out as the run gets long, so the axis never crowds.
  const stride = epochs <= 10 ? 1 : epochs <= 20 ? 2 : Math.ceil(epochs / 8);
  const xTicks = [
    1,
    ...Array.from({ length: epochs }, (_, i) => i + 1).filter((e) => e % stride === 0 && e !== 1),
  ];

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} className="metrics-chart" role="img" aria-label={ariaLabel}>
        {yTicks.map((t) => (
          <g key={t.at}>
            <line x1={PAD.left} x2={W - PAD.right} y1={py(t.at)} y2={py(t.at)} className="chart-grid" />
            <text x={PAD.left - 8} y={py(t.at) + 4} className="chart-tick" textAnchor="end">
              {t.label}
            </text>
          </g>
        ))}
        {/* Left-anchored above the plot rather than right-anchored beside the
            top tick: some of these labels are a phrase, and a right-anchored
            phrase runs off the left edge of the chart. */}
        <text x={PAD.left - 8} y={PAD.top - 14} className="chart-tick">
          {yLabel}
        </text>
        {xTicks.map((e) => (
          <text key={e} x={px(e)} y={H - PAD.bottom + 18} className="chart-tick" textAnchor="middle">
            {e}
          </text>
        ))}
        <text x={(W + PAD.left) / 2} y={H - 6} className="chart-axis-label" textAnchor="middle">
          {xLabel}
        </text>
        {series.map((s) => (
          <path
            key={s.key}
            d={path(s.values)}
            className={`chart-line ${s.cls}${s.dashed ? " m7-line-dashed" : ""}`}
          />
        ))}
      </svg>
      <ul className="m7-legend">
        {series.map((s) => (
          <li key={s.key}>
            <span
              className={`m7-swatch ${s.cls}${s.dashed ? " m7-line-dashed" : ""}`}
              aria-hidden="true"
            />
            {s.label}
          </li>
        ))}
      </ul>
    </>
  );
}
