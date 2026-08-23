// Singleton client for the Pyodide worker. The worker is expensive to boot
// (10 MB runtime download), so all components share one instance and
// correlate responses by request id.

import type { WorkerRequest, WorkerResponse } from "./messages";

type Handler = (msg: WorkerResponse) => void;

// Omit that distributes over a union (plain Omit collapses WorkerRequest
// to its common keys, losing the per-variant fields).
type DistributiveOmit<T, K extends keyof T> = T extends unknown ? Omit<T, K> : never;

let worker: Worker | null = null;
const handlers = new Map<number, Handler>();
let nextId = 1;

const FINAL_TYPES = new Set(["trainDone", "testsDone", "error"]);

function ensureWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./pyodideWorker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data;
      const handler = handlers.get(msg.id);
      if (FINAL_TYPES.has(msg.type)) handlers.delete(msg.id);
      handler?.(msg);
    };
  }
  return worker;
}

/** Send a request; onMessage receives every response for it, ending with
 * trainDone/testsDone/error. Returns the request id. */
export function sendRequest(req: DistributiveOmit<WorkerRequest, "id">, onMessage: Handler): number {
  const id = nextId++;
  handlers.set(id, onMessage);
  ensureWorker().postMessage({ ...req, id });
  return id;
}

/** Hard-stop the worker (the only way out of runaway learner code, e.g. an
 * infinite loop). Pending requests get an error response; the next request
 * boots a fresh worker (the runtime re-download is served from HTTP cache). */
export function terminateWorker(): void {
  worker?.terminate();
  worker = null;
  const pending = [...handlers.entries()];
  handlers.clear();
  for (const [id, handler] of pending) {
    handler({ type: "error", id, message: "cancelled: the Python runtime was stopped" });
  }
}
