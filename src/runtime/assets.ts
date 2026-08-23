// Main-thread asset loading: gzipped JSON and the MNIST subset binary.
// Mirrors the worker's gzip handling: some servers strip the compression
// via Content-Encoding, others serve raw bytes; check the magic.

async function fetchMaybeGz(url: string): Promise<Uint8Array> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to fetch ${url}: HTTP ${resp.status}`);
  let bytes = new Uint8Array(await resp.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return bytes;
}

export function assetUrl(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, window.location.href).href;
}

export interface PretrainedWeights {
  sizes: number[];
  weights: number[][][]; // weights[l][j][k]
  biases: number[][]; // biases[l][j]
  test_accuracy: number;
}

let weightsPromise: Promise<PretrainedWeights> | null = null;

export function fetchPretrainedWeights(): Promise<PretrainedWeights> {
  weightsPromise ??= fetchMaybeGz(assetUrl("data/pretrained_weights.json.gz")).then(
    (bytes) => JSON.parse(new TextDecoder().decode(bytes)) as PretrainedWeights,
  );
  return weightsPromise;
}

export interface MnistTestSubset {
  images: Uint8Array; // n * 784, row-major per image
  labels: Uint8Array; // n
  count: number;
}

let mnistPromise: Promise<MnistTestSubset> | null = null;

/** Test-split images from the bundled subset (format: tools/make_mnist_subset.py). */
export function fetchMnistTest(): Promise<MnistTestSubset> {
  mnistPromise ??= fetchMaybeGz(assetUrl("data/mnist_subset.bin.gz")).then((bytes) => {
    const magic = new TextDecoder().decode(bytes.subarray(0, 4));
    if (magic !== "MNSS") throw new Error(`bad MNIST subset magic: ${magic}`);
    const view = new DataView(bytes.buffer, bytes.byteOffset);
    const nTrain = view.getUint32(5, true);
    const nTest = view.getUint32(9, true);
    const pixels = view.getUint32(13, true) * view.getUint32(17, true);
    const testStart = 21 + nTrain * pixels + nTrain;
    return {
      images: bytes.subarray(testStart, testStart + nTest * pixels),
      labels: bytes.subarray(testStart + nTest * pixels, testStart + nTest * pixels + nTest),
      count: nTest,
    };
  });
  return mnistPromise;
}
