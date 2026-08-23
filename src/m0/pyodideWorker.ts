/// <reference lib="webworker" />
// Milestone 0 worker: runs Pyodide + NumPy off the main thread so training
// never freezes the UI. Loads the runtime once, then handles train requests.

import dataLoaderSource from "../python/data_loader.py?raw";
import referenceNetworkSource from "../python/reference_network.py?raw";
import type { TrainRequest, WorkerMessage } from "./messages";

// Pinned Pyodide version (see CLAUDE.md). Do not bump without re-verifying
// the Milestone 0 training-time envelope.
const PYODIDE_VERSION = "314.0.5";
const PYODIDE_BASE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

const post = (msg: WorkerMessage) => self.postMessage(msg);

// Minimal typing for the parts of the Pyodide API we use.
interface Pyodide {
  loadPackage(name: string): Promise<void>;
  runPythonAsync(code: string): Promise<unknown>;
  globals: { set(name: string, value: unknown): void };
  FS: { writeFile(path: string, data: Uint8Array): void };
}

let pyodidePromise: Promise<Pyodide> | null = null;

function getPyodide(): Promise<Pyodide> {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      post({ type: "status", text: "Downloading Pyodide runtime (about 10 MB, first run only)..." });
      const mod = await import(/* @vite-ignore */ `${PYODIDE_BASE_URL}pyodide.mjs`);
      const pyodide: Pyodide = await mod.loadPyodide({
        indexURL: PYODIDE_BASE_URL,
        stdout: (text: string) => post({ type: "log", text }),
        stderr: (text: string) => post({ type: "log", text }),
      });
      post({ type: "status", text: "Loading NumPy..." });
      await pyodide.loadPackage("numpy");
      const version = await pyodide.runPythonAsync(
        "import sys, numpy; f'Python {sys.version.split()[0]}, NumPy {numpy.__version__}'",
      );
      post({ type: "log", text: `Pyodide ${PYODIDE_VERSION} ready (${version})` });
      await pyodide.runPythonAsync(dataLoaderSource);
      await pyodide.runPythonAsync(referenceNetworkSource);
      return pyodide;
    })();
    pyodidePromise.catch(() => {
      pyodidePromise = null; // allow retry after a failed load
    });
  }
  return pyodidePromise;
}

async function fetchDataset(pyodide: Pyodide, dataUrl: string): Promise<void> {
  post({ type: "status", text: "Fetching MNIST subset..." });
  const resp = await fetch(dataUrl);
  if (!resp.ok) {
    throw new Error(`failed to fetch ${dataUrl}: HTTP ${resp.status}`);
  }
  // Some servers (Vite dev among them) serve .gz files with
  // Content-Encoding: gzip, so the browser has already decompressed the
  // body; others serve the raw bytes. Check the gzip magic and decompress
  // only if still compressed.
  let bytes = new Uint8Array(await resp.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  pyodide.FS.writeFile("/mnist_subset.bin", bytes);
  post({ type: "log", text: `MNIST subset loaded (${(bytes.byteLength / 1e6).toFixed(1)} MB decompressed)` });
}

async function train(msg: TrainRequest): Promise<void> {
  const pyodide = await getPyodide();
  await fetchDataset(pyodide, msg.dataUrl);

  pyodide.globals.set(
    "_js_on_epoch",
    (epoch: number, epochs: number, loss: number, accuracy: number, elapsed: number) =>
      post({ type: "epoch", epoch, epochs, loss, accuracy, elapsed }),
  );

  post({ type: "status", text: "Training..." });
  const params = {
    epochs: msg.epochs,
    hidden: msg.hidden,
    mini_batch_size: msg.miniBatchSize,
    eta: msg.eta,
    seed: msg.seed,
  };
  for (const value of Object.values(params)) {
    if (!Number.isFinite(value)) throw new Error("invalid training parameter");
  }
  const resultJson = (await pyodide.runPythonAsync(`
import json
with open("/mnist_subset.bin", "rb") as _f:
    _mnist_buf = _f.read()
_result = run_m0(
    _mnist_buf,
    epochs=${params.epochs},
    hidden=${params.hidden},
    mini_batch_size=${params.mini_batch_size},
    eta=${params.eta},
    seed=${params.seed},
    on_epoch=_js_on_epoch,
)
json.dumps(_result)
`)) as string;
  post({ type: "done", result: JSON.parse(resultJson) });
}

self.onmessage = (event: MessageEvent<TrainRequest>) => {
  if (event.data.type !== "train") return;
  train(event.data).catch((err) =>
    post({ type: "error", message: err instanceof Error ? err.message : String(err) }),
  );
};
