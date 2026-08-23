import { sigmoid } from "./utils";

// Module 4's fixed demonstration network: the contrarian's 2-2-1 grown by a
// third hidden neuron (2-3-1), so no two layers share a size and every shape
// stays honest. Both the step-through visualization and the quiz compute
// from this one definition, so their numbers can never drift apart.
//
// JS mirrors of the course's column-vector conventions: a "column" here is a
// plain number[], and a weight matrix is number[][] with one row per
// receiving neuron (shape (receiving layer, sending layer), as in Module 2).

export interface Net {
  W2: number[][]; // (3, 2): weights into the hidden layer
  b2: number[]; // (3,): hidden biases
  W3: number[][]; // (1, 3): weights into the output layer
  b3: number[]; // (1,): output bias
}

/** The fixed training example: good weather, no friend; the contrarian goes. */
export const INPUT = [1, 0];
export const TARGET = 1;

export const START: Net = {
  W2: [
    [3.0, 3.0],
    [2.0, 2.0],
    [-2.0, 1.0],
  ],
  b2: [-1.5, -3.0, 0.5],
  W3: [[4.0, -4.0, 1.0]],
  b3: [-2.0],
};

export function cloneNet(net: Net): Net {
  return {
    W2: net.W2.map((r) => [...r]),
    b2: [...net.b2],
    W3: net.W3.map((r) => [...r]),
    b3: [...net.b3],
  };
}

export interface Trace {
  z2: number[];
  a2: number[];
  z3: number[];
  a3: number[];
  cost: number;
  d3: number[]; // output-layer blame (BP1)
  d2: number[]; // hidden-layer blame (BP2)
  gW3: number[][]; // slope of C for each weight into the output (BP4)
  gb3: number[]; // slope of C for the output bias (BP3)
  gW2: number[][];
  gb2: number[];
}

const sigmoidPrime = (z: number) => sigmoid(z) * (1 - sigmoid(z));

/** One forward pass and one backward pass on the fixed example. */
export function compute(net: Net): Trace {
  const z2 = net.W2.map((row, j) => row[0] * INPUT[0] + row[1] * INPUT[1] + net.b2[j]);
  const a2 = z2.map(sigmoid);
  const z3 = net.W3.map(
    (row, j) => row[0] * a2[0] + row[1] * a2[1] + row[2] * a2[2] + net.b3[j],
  );
  const a3 = z3.map(sigmoid);
  const cost = 0.5 * (TARGET - a3[0]) ** 2;

  // BP1: blame starts at the output.
  const d3 = z3.map((z, j) => (a3[j] - TARGET) * sigmoidPrime(z));
  // BP2: blame flows backward through the same wires, transposed.
  const d2 = z2.map((z, k) => net.W3[0][k] * d3[0] * sigmoidPrime(z));
  // BP3 and BP4: every slope read off the blames.
  const gb3 = [...d3];
  const gW3 = [a2.map((a) => a * d3[0])];
  const gb2 = [...d2];
  const gW2 = d2.map((d) => INPUT.map((x) => x * d));

  return { z2, a2, z3, a3, cost, d3, d2, gW3, gb3, gW2, gb2 };
}

/** Which single parameter of the 13 is selected in the stepper. */
export type ParamRef =
  | { kind: "W2" | "W3"; j: number; k: number }
  | { kind: "b2" | "b3"; j: number };

export function getParam(net: Net, ref: ParamRef): number {
  if (ref.kind === "W2" || ref.kind === "W3") return net[ref.kind][ref.j][ref.k];
  return net[ref.kind][ref.j];
}

export function setParam(net: Net, ref: ParamRef, value: number): Net {
  const next = cloneNet(net);
  if (ref.kind === "W2" || ref.kind === "W3") next[ref.kind][ref.j][ref.k] = value;
  else next[ref.kind][ref.j] = value;
  return next;
}

export function getSlope(trace: Trace, ref: ParamRef): number {
  if (ref.kind === "W2") return trace.gW2[ref.j][ref.k];
  if (ref.kind === "W3") return trace.gW3[ref.j][ref.k];
  if (ref.kind === "b2") return trace.gb2[ref.j];
  return trace.gb3[ref.j];
}

const HIDDEN_NAMES = ["h₁", "h₂", "h₃"];
const INPUT_NAMES = ["x₁", "x₂"];

export function paramLabel(ref: ParamRef): string {
  if (ref.kind === "W2") return `weight on the wire ${INPUT_NAMES[ref.k]} → ${HIDDEN_NAMES[ref.j]}`;
  if (ref.kind === "W3") return `weight on the wire ${HIDDEN_NAMES[ref.k]} → out`;
  if (ref.kind === "b2") return `bias of ${HIDDEN_NAMES[ref.j]}`;
  return "bias of the output neuron";
}

/** Format with a real minus sign (the ASCII hyphen reads badly in prose). */
export function fmt(v: number, decimals = 3): string {
  return v.toFixed(decimals).replace("-", "−");
}
