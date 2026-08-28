import { scale } from "./utils";
import { fig } from "../../../components/ModuleBits";

// Static figure: the measured log of one run, the 784-30-10 digit reader
// trained on 1,000 images for 80 epochs (cross-entropy cost, weights scaled
// by 1/sqrt(inputs), step size 0.5, mini-batches of 10, the same seeds the
// panel later in the module uses). Four numbers per epoch, so the split
// between what the network learns and what it knows is visible in one
// picture. The panel reproduces this run and adds the second one.
//
// Rows are [epoch, accuracy on the training images, accuracy on the held-out
// images, cost on the training images, cost on the held-out images].
const LOG: [number, number, number, number, number][] = [
  [1, 0.821, 0.722, 1.3878, 1.6388],
  [2, 0.903, 0.784, 0.89, 1.2722],
  [3, 0.935, 0.833, 0.5927, 1.0034],
  [4, 0.942, 0.841, 0.4995, 0.9589],
  [5, 0.949, 0.84, 0.4273, 0.924],
  [6, 0.963, 0.842, 0.3531, 0.9312],
  [7, 0.979, 0.847, 0.2798, 0.8875],
  [8, 0.983, 0.846, 0.2294, 0.8649],
  [9, 0.992, 0.855, 0.1964, 0.8375],
  [10, 0.991, 0.851, 0.1879, 0.8882],
  [12, 0.994, 0.862, 0.1386, 0.8679],
  [14, 0.996, 0.857, 0.1075, 0.867],
  [16, 0.998, 0.851, 0.0913, 0.894],
  [18, 0.998, 0.86, 0.0743, 0.8687],
  [20, 0.998, 0.855, 0.0647, 0.898],
  [24, 0.999, 0.857, 0.0486, 0.9224],
  [28, 0.999, 0.858, 0.0393, 0.9439],
  [32, 1.0, 0.86, 0.0322, 0.9451],
  [36, 1.0, 0.863, 0.027, 0.9678],
  [40, 1.0, 0.86, 0.0229, 0.9871],
  [44, 1.0, 0.858, 0.02, 1.0016],
  [48, 1.0, 0.865, 0.0176, 1.0183],
  [52, 1.0, 0.86, 0.0159, 1.0298],
  [56, 1.0, 0.863, 0.0144, 1.0388],
  [60, 1.0, 0.865, 0.0132, 1.0484],
  [64, 1.0, 0.864, 0.0122, 1.0654],
  [68, 1.0, 0.862, 0.0113, 1.0705],
  [72, 1.0, 0.865, 0.0105, 1.0775],
  [76, 1.0, 0.864, 0.0099, 1.0881],
  [80, 1.0, 0.863, 0.0093, 1.0931],
];

const W = 640;
const H = 300;
const PAD = { top: 26, right: 58, bottom: 44, left: 54 };
const EPOCHS = 80;
const COST_MAX = 1.8;

export function OverfitFigure() {
  const px = (epoch: number) => scale(epoch, 1, EPOCHS, PAD.left, W - PAD.right);
  const pyAcc = (v: number) => scale(v, 0.6, 1.0, H - PAD.bottom, PAD.top);
  const pyCost = (v: number) => scale(v, 0, COST_MAX, H - PAD.bottom, PAD.top);
  const path = (pick: (row: (typeof LOG)[number]) => number, y: (v: number) => number) =>
    LOG.map((row, i) => `${i === 0 ? "M" : "L"}${px(row[0]).toFixed(1)},${y(pick(row)).toFixed(1)}`)
      .join(" ");

  return (
    <>
    <svg
      {...fig(5, -8, 618, 309)}
      className="curve-figure overfit-figure"
      role="img"
      aria-label="Eighty epochs of training on 1,000 images. Accuracy on the training images climbs from 82 percent to 100 percent and stays there; accuracy on the held-out images climbs to about 86 percent by epoch 10 and then stops. The cost on the training images falls to nearly zero; the cost on the held-out images bottoms out near 0.84 around epoch 9 and then rises to 1.09."
    >
      {[0.6, 0.7, 0.8, 0.9, 1.0].map((v) => (
        <g key={v}>
          <line x1={PAD.left} x2={W - PAD.right} y1={pyAcc(v)} y2={pyAcc(v)} className="curve-grid" />
          <text x={PAD.left - 8} y={pyAcc(v) + 4} className="chart-tick chart-tick-acc" textAnchor="end">
            {Math.round(v * 100)}%
          </text>
        </g>
      ))}
      {[0, 0.6, 1.2, 1.8].map((v) => (
        <text
          key={v}
          x={W - PAD.right + 8}
          y={pyCost(v) + 4}
          className="chart-tick chart-tick-loss"
        >
          {v.toFixed(1)}
        </text>
      ))}
      <text x={PAD.left - 8} y={PAD.top - 14} className="chart-tick chart-tick-acc">
        accuracy
      </text>
      <text x={W - PAD.right + 8} y={PAD.top - 14} className="chart-tick chart-tick-loss">
        cost
      </text>
      {[1, 20, 40, 60, 80].map((e) => (
        <text key={e} x={px(e)} y={H - PAD.bottom + 18} className="chart-tick" textAnchor="middle">
          {e}
        </text>
      ))}
      <text x={(W + PAD.left - PAD.right) / 2} y={H - 8} className="chart-axis-label" textAnchor="middle">
        epoch (one full pass through the 1,000 training images)
      </text>

      <path d={path((r) => r[3], pyCost)} className="chart-line chart-line-loss" />
      <path d={path((r) => r[4], pyCost)} className="chart-line chart-line-loss m7-line-dashed" />
      <path d={path((r) => r[1], pyAcc)} className="chart-line chart-line-acc" />
      <path d={path((r) => r[2], pyAcc)} className="chart-line chart-line-acc m7-line-dashed" />

    </svg>
    {/* Four lines converge and cross in the right-hand third, so naming them
        in place put two labels on top of each other. The key goes below. */}
    <ul className="m7-legend">
      <li>
        <span className="m7-swatch chart-line-acc" aria-hidden="true" />
        accuracy, the 1,000 it trains on
      </li>
      <li>
        <span className="m7-swatch chart-line-acc m7-line-dashed" aria-hidden="true" />
        accuracy, the 1,000 held out
      </li>
      <li>
        <span className="m7-swatch chart-line-loss" aria-hidden="true" />
        cost, the 1,000 it trains on
      </li>
      <li>
        <span className="m7-swatch chart-line-loss m7-line-dashed" aria-hidden="true" />
        cost, the 1,000 held out
      </li>
    </ul>
    </>
  );
}
