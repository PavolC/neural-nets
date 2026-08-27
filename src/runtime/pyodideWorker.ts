/// <reference lib="webworker" />
// Pyodide worker: runs Python + NumPy off the main thread. Serves two kinds
// of requests: "train" (the Milestone 0 reference training run) and
// "runTests" (execute learner code against an exercise's test suite).
// Requests are processed sequentially; every response echoes the request id.

import dataLoaderSource from "../python/data_loader.py?raw";
import referenceNetworkSource from "../python/reference_network.py?raw";
import courseHelpersSource from "../python/course_helpers.py?raw";
import harnessSource from "../python/harness.py?raw";
import type { WorkerRequest, WorkerResponse } from "./messages";

// Pinned Pyodide version (see CLAUDE.md). Do not bump without re-verifying
// the Milestone 0 training-time envelope.
const PYODIDE_VERSION = "314.0.5";
const PYODIDE_BASE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// The id of the request currently being processed; the worker is
// single-threaded, so status/log lines always belong to this request.
let currentId = 0;

const post = (msg: WorkerResponse) => self.postMessage(msg);
const status = (text: string) => post({ type: "status", id: currentId, text });

// Until the runtime finishes booting, stdout is loader noise; afterwards it
// is the user's own print() output.
let bootDone = false;
const logStdout = (text: string) =>
  post({ type: "log", id: currentId, source: bootDone ? "stdout" : "runtime", text });
const logRuntime = (text: string) =>
  post({ type: "log", id: currentId, source: "runtime", text });

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
      // Not "first run only": pressing Stop discards the worker, so this runs
      // again on the next attempt. The browser cache is what makes the repeat
      // cheap, and that is what the reader needs told.
      status("Downloading the Python runtime (about 10 MB, then cached)...");
      const mod = await import(/* @vite-ignore */ `${PYODIDE_BASE_URL}pyodide.mjs`);
      const pyodide: Pyodide = await mod.loadPyodide({
        indexURL: PYODIDE_BASE_URL,
        stdout: (text: string) => logStdout(text),
        stderr: (text: string) => logStdout(text),
      });
      status("Loading NumPy...");
      await pyodide.loadPackage("numpy");
      const version = await pyodide.runPythonAsync(
        "import sys, numpy; f'Python {sys.version.split()[0]}, NumPy {numpy.__version__}'",
      );
      logRuntime(`Pyodide ${PYODIDE_VERSION} ready (${version})`);
      // Shared course Python: data loader, reference network, helpers
      // (registered as the `course` module so exercises can import from it),
      // and the exercise test harness.
      await pyodide.runPythonAsync(dataLoaderSource);
      await pyodide.runPythonAsync(referenceNetworkSource);
      pyodide.globals.set("_course_src", courseHelpersSource);
      await pyodide.runPythonAsync(`
import sys, types
_course = types.ModuleType("course")
_course.__file__ = "course_helpers.py"
exec(compile(_course_src, "course_helpers.py", "exec"), _course.__dict__)
sys.modules["course"] = _course
`);
      await pyodide.runPythonAsync(harnessSource);
      bootDone = true;
      return pyodide;
    })();
    pyodidePromise.catch(() => {
      pyodidePromise = null; // allow retry after a failed load
    });
  }
  return pyodidePromise;
}

/** The path a fetched dataset lands on inside Pyodide: the URL's own file
 * name, with the .gz suffix dropped, at the root. So
 * data/mnist_subset.bin.gz becomes /mnist_subset.bin, which is where every
 * snippet in the course reads it from, and data/penguins.json.gz becomes
 * /penguins.json without any of them having to say so. */
function datasetPath(dataUrl: string): string {
  const name = new URL(dataUrl).pathname.split("/").pop() || "dataset";
  return `/${name.replace(/\.gz$/, "")}`;
}

async function fetchDataset(pyodide: Pyodide, dataUrl: string): Promise<void> {
  const path = datasetPath(dataUrl);
  status(`Fetching ${path.slice(1)}...`);
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
  pyodide.FS.writeFile(path, bytes);
  const size =
    bytes.byteLength >= 1e6
      ? `${(bytes.byteLength / 1e6).toFixed(1)} MB`
      : `${Math.round(bytes.byteLength / 1e3)} kB`;
  logRuntime(`${path.slice(1)} loaded (${size} decompressed)`);
}

async function train(msg: Extract<WorkerRequest, { type: "train" }>): Promise<void> {
  const pyodide = await getPyodide();
  await fetchDataset(pyodide, msg.dataUrl);

  pyodide.globals.set(
    "_js_on_epoch",
    (epoch: number, epochs: number, loss: number, accuracy: number, elapsed: number) =>
      post({ type: "epoch", id: msg.id, epoch, epochs, loss, accuracy, elapsed }),
  );

  status("Training...");
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
  post({ type: "trainDone", id: msg.id, result: JSON.parse(resultJson) });
}

async function runTests(msg: Extract<WorkerRequest, { type: "runTests" }>): Promise<void> {
  const pyodide = await getPyodide();
  status("Running tests...");
  pyodide.globals.set("_learner_code", msg.learnerCode);
  pyodide.globals.set("_tests_code", msg.testsCode);
  const resultJson = (await pyodide.runPythonAsync(
    "run_exercise(_learner_code, _tests_code)",
  )) as string;
  post({ type: "testsDone", id: msg.id, result: JSON.parse(resultJson) });
}

async function runPython(msg: Extract<WorkerRequest, { type: "runPython" }>): Promise<void> {
  const pyodide = await getPyodide();
  if (msg.dataUrl) await fetchDataset(pyodide, msg.dataUrl);
  pyodide.globals.set("_args_json", JSON.stringify(msg.args ?? null));
  pyodide.globals.set("_js_report", (payloadJson: string) =>
    post({ type: "report", id: msg.id, payload: JSON.parse(payloadJson) }),
  );
  const resultJson = (await pyodide.runPythonAsync(msg.code)) as string;
  post({ type: "pythonDone", id: msg.id, result: JSON.parse(resultJson) });
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  currentId = msg.id;
  const job =
    msg.type === "train" ? train(msg) : msg.type === "runTests" ? runTests(msg) : runPython(msg);
  job.catch((err) =>
    post({
      type: "error",
      id: msg.id,
      message: err instanceof Error ? err.message : String(err),
    }),
  );
};
