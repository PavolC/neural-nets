import { sigmoid } from "./utils";

// The arithmetic behind Module 6's two interactives, kept out of the React
// components so that tools/bench_bumps.ts can import the panel's own code
// rather than a copy of it (the same arrangement deepNet.ts has with
// tools/bench_layer_speeds.ts). Every number Module 6 quotes comes from
// here, through one or the other caller.

export const TN = 161; // target samples, one every 1/160 of the dial
export const AG = 641; // grid the area is measured on
export const BAR_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24];
export const STEEPNESS = [20, 50, 100, 200, 400, 700];
export const DEFAULT_BARS = 3; // index: 6 bars
export const DEFAULT_STEEP = 4; // index: weight 400

export const clampV = (v: number) => Math.max(0, Math.min(10, v));
export const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export const PRESETS = {
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
export type PresetKind = keyof typeof PRESETS;

export const sampleTarget = (f: (x: number) => number) =>
  Array.from({ length: TN }, (_, j) => clampV(f(j / (TN - 1))));

/** The target between its samples: straight lines, which is what drawing by
 * hand produces anyway. */
export function targetAt(t: number[], x: number): number {
  const u = clamp01(x) * (TN - 1);
  const i = Math.min(TN - 2, Math.floor(u));
  return t[i] + (t[i + 1] - t[i]) * (u - i);
}

/** Each bar at the target's average height across its own slice. */
export function fitHeights(t: number[], k: number): number[] {
  return Array.from({ length: k }, (_, i) => {
    const S = 40;
    let sum = 0;
    for (let j = 0; j < S; j++) sum += targetAt(t, (i + (j + 0.5) / S) / k);
    return clampV(sum / S);
  });
}

/** The network's answer: one pair of sigmoid neurons per bar, the left one
 * switching on at the bar's left edge, its outgoing wire carrying +h, the
 * right one at its right edge, its outgoing wire carrying -h. */
export function netAt(hs: number[], k: number, w: number, x: number): number {
  let sum = 0;
  for (let i = 0; i < k; i++) {
    if (hs[i] === 0) continue;
    sum += hs[i] * (sigmoid(w * (x - i / k)) - sigmoid(w * (x - (i + 1) / k)));
  }
  return sum;
}

export function areaBetween(t: number[], hs: number[], k: number, w: number): number {
  let sum = 0;
  for (let i = 0; i < AG; i++) {
    const x = i / (AG - 1);
    const weight = i === 0 || i === AG - 1 ? 0.5 : 1;
    sum += weight * Math.abs(targetAt(t, x) - netAt(hs, k, w, x));
  }
  return sum / (AG - 1);
}

/** One bump: two step neurons at s1 and s2, their outgoing wires carrying
 * +h and -h. What BumpBuilder draws, and what a bar of the sculptor is. */
export function bumpAt(w: number, s1: number, s2: number, h: number, x: number): number {
  return h * (sigmoid(w * (x - s1)) - sigmoid(w * (x - s2)));
}
