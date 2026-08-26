import { mulberry32 } from "./utils";

// Module 8's measurement: how fast each layer of a deep network is learning,
// at the moment training starts. One place, because the bars, the hop
// factors and the prose numbers all have to come from the same arithmetic.
//
// Course conventions in JS: a layer's weight matrix has one row per receiving
// neuron (shape (receiving, sending), Module 2), and a batch of m images is
// the (n, m) matrix stored row-major, so entry (i, k) is at i * m + k.
// Layers are numbered Module 4's way: layer 1 is the input, layer L the
// output, so the first hidden layer of a 784-30-...-10 network is layer 2.

export type Activation = "sigmoid" | "relu";

export interface Matrix {
  rows: number;
  cols: number;
  a: Float64Array;
}

function zeros(rows: number, cols: number): Matrix {
  return { rows, cols, a: new Float64Array(rows * cols) };
}

/** Standard normal draws from mulberry32, by the Box-Muller transform. */
function gaussians(rand: () => number, n: number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 2) {
    // log(0) is -Infinity, so keep u strictly positive.
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** C = A @ B, with A (p, q) and B (q, m). */
function matmul(A: Matrix, B: Matrix): Matrix {
  const out = zeros(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let k = 0; k < A.cols; k++) {
      const aik = A.a[i * A.cols + k];
      if (aik === 0) continue;
      const rowB = k * B.cols;
      const rowC = i * B.cols;
      for (let j = 0; j < B.cols; j++) out.a[rowC + j] += aik * B.a[rowB + j];
    }
  }
  return out;
}

/** C = A^T @ B, with A (p, q) and B (p, m), giving (q, m). */
function matmulT(A: Matrix, B: Matrix): Matrix {
  const out = zeros(A.cols, B.cols);
  for (let i = 0; i < A.rows; i++) {
    const rowB = i * B.cols;
    for (let k = 0; k < A.cols; k++) {
      const aik = A.a[i * A.cols + k];
      if (aik === 0) continue;
      const rowC = k * B.cols;
      for (let j = 0; j < B.cols; j++) out.a[rowC + j] += aik * B.a[rowB + j];
    }
  }
  return out;
}

function norm(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

const squash = {
  sigmoid: (z: number) => 1 / (1 + Math.exp(-z)),
  relu: (z: number) => (z > 0 ? z : 0),
};

const squashSlope = {
  sigmoid: (z: number) => {
    const s = 1 / (1 + Math.exp(-z));
    return s * (1 - s);
  },
  relu: (z: number) => (z > 0 ? 1 : 0),
};

export interface LayerSpeed {
  /** Module 4's numbering: layer 1 is the input, so the first hidden layer is 2. */
  layer: number;
  size: number;
  isOutput: boolean;
  /** ||dC/db|| for this layer: the size of the whole layer's blame. */
  speed: number;
  /** The BP2 hop that made this layer's blame out of the layer above it.
   * Null for the output layer, which gets its blame from BP1 instead. */
  hop: number | null;
  /** The hop's two factors, each measured as what that step does to the size
   * of the blame column, so the two multiply to the hop exactly: passing
   * back through the wire ledger, then scaling by this layer's steepness. */
  weightFactor: number | null;
  steepFactor: number | null;
  /** The average steepness itself, for reading against the sigmoid's 0.25
   * ceiling. Not the factor: deleting entries shortens a column by less than
   * it lowers the average, which is why ReLU's two numbers differ. */
  meanSteepness: number | null;
}

export interface Measurement {
  sizes: number[];
  layers: LayerSpeed[];
  /** How many times slower the first hidden layer learns than the output. */
  slowdown: number;
}

export interface Batch {
  /** (784, m) pixels in [0, 1], row-major. */
  X: Matrix;
  /** (10, m) one-hot right answers. */
  Y: Matrix;
}

/** Build the (784, m) and (10, m) matrices from the bundled MNIST bytes. */
export function makeBatch(images: Uint8Array, labels: Uint8Array, m: number): Batch {
  const X = zeros(784, m);
  const Y = zeros(10, m);
  for (let k = 0; k < m; k++) {
    for (let i = 0; i < 784; i++) X.a[i * m + k] = images[k * 784 + i] / 255;
    Y.a[labels[k] * m + k] = 1;
  }
  return { X, Y };
}

export interface Options {
  /** How many hidden layers, each of hiddenSize neurons. */
  hidden: number;
  hiddenSize: number;
  activation: Activation;
  /** Multiplier on Module 7's divided draw: 1 is that draw exactly. */
  weightScale: number;
  seed: number;
}

/**
 * Measure every layer's learning speed for a freshly initialized network,
 * before it takes a step. The start is Module 7's: each weight a standard
 * normal draw divided by the square root of its layer's input count, each
 * bias a standard normal draw. The output layer is always a sigmoid with the
 * cross-entropy blame a - y, so BP1 contributes nothing of its own to the
 * pattern down the layers.
 */
export function measure(batch: Batch, opts: Options): Measurement {
  const { hidden, hiddenSize, activation, weightScale, seed } = opts;
  const sizes = [784, ...Array(hidden).fill(hiddenSize), 10];
  const rand = mulberry32(seed);
  const L = sizes.length - 1; // number of weight matrices
  const w: Matrix[] = [];
  const b: Matrix[] = [];
  for (let l = 0; l < L; l++) {
    const nIn = sizes[l];
    const nOut = sizes[l + 1];
    const g = gaussians(rand, nOut * nIn);
    const scale = weightScale / Math.sqrt(nIn);
    for (let i = 0; i < g.length; i++) g[i] *= scale;
    w.push({ rows: nOut, cols: nIn, a: g });
    b.push({ rows: nOut, cols: 1, a: gaussians(rand, nOut) });
  }

  const m = batch.X.cols;
  const zs: Matrix[] = [];
  const acts: Matrix[] = [batch.X];
  let a = batch.X;
  for (let l = 0; l < L; l++) {
    const z = matmul(w[l], a);
    for (let i = 0; i < z.rows; i++) {
      const bias = b[l].a[i];
      for (let k = 0; k < m; k++) z.a[i * m + k] += bias;
    }
    zs.push(z);
    // Hidden layers carry the chosen squash; the output layer stays a
    // sigmoid, since the cross-entropy blame is written for one.
    const f = l === L - 1 ? squash.sigmoid : squash[activation];
    const next = zeros(z.rows, m);
    for (let i = 0; i < z.a.length; i++) next.a[i] = f(z.a[i]);
    acts.push(next);
    a = next;
  }

  const slope = squashSlope[activation];
  const layers: LayerSpeed[] = [];

  // A layer's learning speed is the size of its bias gradient, dC/db: square
  // every entry, add, take the square root (Module 3's double bars, used here
  // as a size rather than as a cost). The gradient is the batch average of
  // that layer's blame columns, so the factors below are measured on the
  // averaged column too. Averaging commutes with the wire ledger and not with
  // the steepness, so taking both ratios along the averaged column is what
  // makes the two factors multiply to the hop exactly.
  const meanColumn = (d: Matrix) => {
    const out = new Float64Array(d.rows);
    for (let i = 0; i < d.rows; i++) {
      let s = 0;
      for (let k = 0; k < d.cols; k++) s += d.a[i * d.cols + k];
      out[i] = s / d.cols;
    }
    return out;
  };

  // BP1 under the cross-entropy cost: the output layer's blame is the gap.
  let delta = zeros(sizes[L], m);
  for (let i = 0; i < delta.a.length; i++) delta.a[i] = acts[L].a[i] - batch.Y.a[i];
  layers.push({
    layer: L + 1,
    size: sizes[L],
    isOutput: true,
    speed: norm(meanColumn(delta)),
    hop: null,
    weightFactor: null,
    steepFactor: null,
    meanSteepness: null,
  });

  for (let l = L - 1; l >= 1; l--) {
    const above = layers[layers.length - 1].speed;
    const through = matmulT(w[l], delta); // BP2's first step: back through the ledger
    const throughSize = norm(meanColumn(through));
    const z = zs[l - 1];
    let slopeSum = 0;
    const next = zeros(through.rows, m);
    for (let i = 0; i < z.a.length; i++) {
      const s = slope(z.a[i]);
      slopeSum += s;
      next.a[i] = through.a[i] * s; // BP2's second step: scale by the steepness
    }
    const speed = norm(meanColumn(next));
    layers.push({
      layer: l + 1,
      size: sizes[l],
      isOutput: false,
      speed,
      hop: above === 0 ? 0 : speed / above,
      weightFactor: above === 0 ? 0 : throughSize / above,
      steepFactor: throughSize === 0 ? 0 : speed / throughSize,
      meanSteepness: slopeSum / z.a.length,
    });
    delta = next;
  }

  layers.reverse();
  return {
    sizes,
    layers,
    slowdown: layers[0].speed === 0 ? Infinity : layers[layers.length - 1].speed / layers[0].speed,
  };
}
