import { useEffect, useMemo, useRef, useState } from "react";
import { fetchMnistTest } from "../../../runtime/assets";
import { useInViewOnce } from "../../../components/useInViewOnce";
import { makeBatch, measure, type Activation, type Batch } from "./deepNet";

// Module 8's centerpiece: how fast every layer of a freshly started network is
// learning, before it takes a step. Two mounts of the same component. The
// first (bars and hop factors only) carries the measurement; the second, with
// `full`, opens the squash and the weight size and shows BP2's two factors,
// which is what the sections after it read from.
//
// Everything here is Module 7's start: each weight a standard normal draw
// divided by the square root of its layer's input count, biases at spread 1,
// cross-entropy blame at the output. The arithmetic lives in deepNet.ts.

const HIDDEN_SIZE = 30;
const SAMPLES = 200;
// Picked because its draw lands near the middle of what this start usually
// gives at every depth: with four hidden layers the output learns 567 times
// faster than the first, against a median of about 650 over forty draws.
const DEFAULT_SEED = 191;

const SCALES = [1, 2, 4, 8];

const W = 640;
const H = 320;
// The top pad holds two stacked rows above the plot, the axis label and then
// the hop arrows, and the bottom pad two rows of layer labels. On a phone the
// whole thing renders at about half scale with the label font bumped, so the
// rows need room to stay clear of each other there.
const PAD = { top: 70, right: 18, bottom: 60, left: 66 };
const AXIS_LABEL_Y = 18;
const HOP_LABEL_Y = PAD.top - 26;
const HOP_ARROW_Y = PAD.top - 16;

function decadeLabel(exp: number): string {
  if (exp >= 0) return (10 ** exp).toLocaleString();
  return (10 ** exp).toFixed(-exp);
}

// Four significant figures, with trailing zeros dropped, so a bar reads the
// same as the same number in the module's tables.
/** A ratio, to one decimal below 100 and none above: 4.8, 26.2, 567. */
function fmtRatio(v: number): string {
  return v < 100 ? v.toFixed(1) : v.toFixed(0);
}

function fmtSpeed(v: number): string {
  if (v === 0) return "0";
  const rounded = v.toPrecision(4);
  return rounded.includes("e") ? v.toExponential(2) : String(Number(rounded));
}

export function LayerSpeedBars({ full = false }: { full?: boolean }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const inView = useInViewOnce(hostRef);
  const [batch, setBatch] = useState<Batch | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Four is where the collapse is unmistakable and the bars are still readable.
  const [hidden, setHidden] = useState(4);
  const [activation, setActivation] = useState<Activation>("sigmoid");
  const [weightScale, setWeightScale] = useState(1);
  const [seed, setSeed] = useState(DEFAULT_SEED);

  useEffect(() => {
    if (!inView) return;
    fetchMnistTest()
      .then((m) => setBatch(makeBatch(m.images, m.labels, SAMPLES)))
      .catch((err) => setLoadError(String(err)));
  }, [inView]);

  const result = useMemo(
    () =>
      batch
        ? measure(batch, { hidden, hiddenSize: HIDDEN_SIZE, activation, weightScale, seed })
        : null,
    [batch, hidden, activation, weightScale, seed],
  );

  if (loadError)
    return (
      <div className="interactive" ref={hostRef}>
        <p className="demo-status demo-status-error">Could not load the digits: {loadError}</p>
      </div>
    );
  if (!result)
    return (
      <div className="interactive" ref={hostRef}>
        <p className="demo-status">Loading the digits this is measured on...</p>
      </div>
    );

  const layers = result.layers;
  const speeds = layers.map((l) => l.speed).filter((s) => s > 0);
  const loExp = Math.floor(Math.log10(Math.min(...speeds)));
  const hiExp = Math.ceil(Math.log10(Math.max(...speeds)));
  const decades = [];
  for (let e = loExp; e <= hiExp; e++) decades.push(e);

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const slot = plotW / layers.length;
  const barW = Math.min(58, slot * 0.5);
  const cx = (i: number) => PAD.left + slot * (i + 0.5);
  const py = (v: number) => {
    const t = (Math.log10(Math.max(v, 10 ** loExp)) - loExp) / Math.max(hiExp - loExp, 1);
    return PAD.top + (1 - t) * plotH;
  };
  const baseY = PAD.top + plotH;

  const slowdown = result.slowdown;

  return (
    <div className="interactive" ref={hostRef}>
      <div className="interactive-controls">
        <label className="slider-row m8-depth">
          <span>hidden layers</span>
          <input
            type="range"
            min={1}
            max={5}
            step={1}
            value={hidden}
            onChange={(e) => setHidden(Number(e.target.value))}
          />
          <code>{hidden}</code>
        </label>
        <button className="button-secondary" onClick={() => setSeed((s) => s + 1)}>
          Draw again
        </button>
      </div>
      {full && (
        <div className="interactive-controls">
          <fieldset className="m7-choice">
            <legend>what the hidden neurons squash with</legend>
            {(["sigmoid", "relu"] as Activation[]).map((a) => (
              <button
                key={a}
                className={`chip ${activation === a ? "chip-active" : ""}`}
                onClick={() => setActivation(a)}
              >
                {a === "sigmoid" ? "sigmoid" : "ReLU"}
              </button>
            ))}
          </fieldset>
          <fieldset className="m7-choice">
            <legend>every weight multiplied by</legend>
            {SCALES.map((s) => (
              <button
                key={s}
                className={`chip ${weightScale === s ? "chip-active" : ""}`}
                onClick={() => setWeightScale(s)}
              >
                {s === 1 ? "1 (Module 7's draw)" : `${s}`}
              </button>
            ))}
          </fieldset>
        </div>
      )}
      <p className="m8-key">
        Each bar is one layer's learning speed before the first step: how far that
        layer's biases move in one step, per unit of step size. The scale is by tens,
        so every gridline is a tenth of the one above it and a bar one gridline lower
        is learning ten times slower. The arrows on top are the hop factors, each one
        the bar below it divided by the bar it came from, read right to left because
        that is the direction blame travels.
      </p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="metrics-chart m8-bars"
        role="img"
        aria-label={`Learning speed of every layer of a 784-${HIDDEN_SIZE} network with ${hidden} hidden layers, on a scale by tens, with the factor between neighbouring layers marked above the bars.`}
      >
        {decades.map((e) => (
          <g key={e}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={py(10 ** e)}
              y2={py(10 ** e)}
              className="chart-grid"
            />
            <text x={PAD.left - 8} y={py(10 ** e) + 4} className="chart-tick" textAnchor="end">
              {decadeLabel(e)}
            </text>
          </g>
        ))}
        <text x={PAD.left - 8} y={AXIS_LABEL_Y} className="chart-tick">
          learning speed
        </text>
        {layers.map((l, i) => (
          <g key={l.layer}>
            <rect
              x={cx(i) - barW / 2}
              y={py(l.speed)}
              width={barW}
              height={Math.max(1, baseY - py(l.speed))}
              className={l.isOutput ? "m8-bar-out" : "m8-bar-hidden"}
            />
            <text x={cx(i)} y={py(l.speed) - 6} className="chart-tick" textAnchor="middle">
              {fmtSpeed(l.speed)}
            </text>
            <text x={cx(i)} y={baseY + 17} className="chart-tick" textAnchor="middle">
              layer {l.layer}
            </text>
            <text x={cx(i)} y={baseY + 37} className="chart-tick m8-bar-sub" textAnchor="middle">
              {l.isOutput ? `output, ${l.size}` : `${l.size} neurons`}
            </text>
          </g>
        ))}
        {layers.map((l, i) =>
          l.hop == null ? null : (
            <g key={`hop-${l.layer}`}>
              <path
                d={`M${cx(i + 1) - barW / 4},${HOP_ARROW_Y} L${cx(i) + barW / 4},${HOP_ARROW_Y}`}
                className="m8-hop-arrow"
                markerEnd="url(#m8-arrow)"
              />
              <text
                x={(cx(i) + cx(i + 1)) / 2}
                y={HOP_LABEL_Y}
                className="chart-tick m8-hop-label"
                textAnchor="middle"
              >
                × {l.hop.toFixed(3)}
              </text>
            </g>
          ),
        )}
        <defs>
          <marker
            id="m8-arrow"
            markerWidth="7"
            markerHeight="7"
            refX="6"
            refY="3.5"
            orient="auto"
          >
            <path d="M0,0 L7,3.5 L0,7 z" className="m8-arrow-head" />
          </marker>
        </defs>
      </svg>
      <p className="m8-readout status-fixed">
        {slowdown >= 1 ? (
          <>
            The output layer moves <b>{fmtRatio(slowdown)}</b> times as far as layer 2
            in one step.
          </>
        ) : (
          <>
            Layer 2 moves <b>{fmtRatio(1 / slowdown)}</b> times as far as the output
            layer in one step: the blame grows on its way back.
          </>
        )}
      </p>
      {full && (
        <div className="table-scroll scroll-x" tabIndex={0}>
          <table className="truth-table">
            <thead>
              <tr>
                <th>hop into</th>
                <th>back through the wire ledger</th>
                <th>times the steepness step</th>
                <th>= the hop</th>
                <th>average steepness there</th>
              </tr>
            </thead>
            <tbody>
              {/* Deepest hop first, so the rows run in the direction the
                  backward sweep does, the same order as the arrows above. */}
              {layers
                .filter((l) => l.hop != null)
                .reverse()
                .map((l) => (
                  <tr key={l.layer}>
                    <td>layer {l.layer}</td>
                    <td>{l.weightFactor!.toFixed(3)}</td>
                    <td>{l.steepFactor!.toFixed(3)}</td>
                    <td>{l.hop!.toFixed(3)}</td>
                    <td>{l.meanSteepness!.toFixed(3)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
