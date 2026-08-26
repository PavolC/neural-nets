// Shared helpers for the interactive visualizations.

export function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

/** Deterministic 32-bit PRNG (interactives must not depend on Math.random
 * for anything that should replay identically). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Diverging blue-white-red color for a value in [-max, max]:
 * negative = blue, zero = near-white, positive = red. */
export function divergingColor(value: number, max: number): string {
  const t = Math.max(-1, Math.min(1, value / (max || 1)));
  const lerp = (a: number, b: number, u: number) => Math.round(a + (b - a) * u);
  const white = [247, 247, 247];
  const target = t < 0 ? [33, 102, 172] : [178, 24, 43];
  const u = Math.abs(t);
  return `rgb(${lerp(white[0], target[0], u)}, ${lerp(white[1], target[1], u)}, ${lerp(white[2], target[2], u)})`;
}

/** Map a value from [d0, d1] to [r0, r1]. */
export function scale(v: number, d0: number, d1: number, r0: number, r1: number): number {
  return r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

/** Draw one 28x28 MNIST digit (dark ink on white) onto a canvas. */
export function drawMnistDigit(
  canvas: HTMLCanvasElement,
  images: Uint8Array,
  index: number,
): void {
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = 255 - images[index * 784 + i];
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
