// Message protocol between the main thread and the Pyodide worker.
// Every request carries an id; every response echoes the id of the request
// it belongs to, so multiple UI components can share one worker.

export interface TrainParams {
  epochs: number;
  hidden: number;
  miniBatchSize: number;
  eta: number;
  seed: number;
}

/** The line ranges the harness needs to say which section a failure came
 * from. Worked out on this side, because the document format has exactly one
 * parser (src/state/workbenchDoc.ts). */
export interface SectionRange {
  id: string;
  label: string;
  kind: string;
  start: number;
  end: number;
}

export interface RunSpec {
  /** The section whose tests are running. */
  target: string;
  sections: SectionRange[];
  /** Names the course lends because the learner has not written them yet. */
  lend: string[];
}

export type WorkerRequest =
  | ({ type: "train"; id: number; dataUrl: string } & TrainParams)
  | { type: "runTests"; id: number; learnerCode: string; testsCode: string }
  // The workbench: one document, one section's tests, and the map that says
  // which lines belong to which section.
  | {
      type: "runDocument";
      id: number;
      document: string;
      testsCode: string;
      spec: RunSpec;
      dataUrl?: string;
    }
  // "Run my code": the whole document, then the scratch pad, in one namespace.
  | {
      type: "runDocumentScratch";
      id: number;
      document: string;
      scratchCode: string;
      spec: RunSpec;
      dataUrl?: string;
    }
  // First-party Python snippets from interactives. The snippet reads its
  // input by json.loads(_args_json), may stream progress via
  // _js_report(json_string), and must evaluate to a JSON string. When
  // dataUrl is set, the MNIST subset is fetched first and written to
  // /mnist_subset.bin before the snippet runs.
  | { type: "runPython"; id: number; code: string; args?: unknown; dataUrl?: string };

export interface EpochMetrics {
  epoch: number;
  epochs: number;
  loss: number;
  accuracy: number;
  elapsed: number;
}

export interface TrainResult {
  final_accuracy: number;
  train_seconds: number;
  epochs: number;
  n_train: number;
  n_test: number;
  hidden: number;
}

export interface TestResultEntry {
  name: string;
  title: string;
  passed: boolean;
  message: string;
  /** The section a crash came from, when it was not this one. */
  section?: string | null;
}

export interface TestRunResult {
  setup_error: {
    message: string;
    line: number | null;
    section?: string | null;
  } | null;
  tests: TestResultEntry[];
  passed: boolean;
  /** Names the course supplied because their section is not written yet. An
   * empty list is the reward: the run was entirely the learner's own code. */
  lent?: string[];
}

export interface ScratchRunResult {
  error: { message: string; line: number | null; label?: string | null } | null;
  lent?: string[];
}

export type WorkerResponse =
  | { type: "status"; id: number; text: string }
  // source "stdout" is the user's own prints; "runtime" is loader noise
  // (Pyodide boot, package downloads, dataset fetches).
  | { type: "log"; id: number; source: "runtime" | "stdout"; text: string }
  | ({ type: "epoch"; id: number } & EpochMetrics)
  | { type: "trainDone"; id: number; result: TrainResult }
  | { type: "testsDone"; id: number; result: TestRunResult }
  | { type: "report"; id: number; payload: unknown }
  | { type: "pythonDone"; id: number; result: unknown }
  // The reader pressed Stop. Distinct from "error" so the UI can return to
  // idle instead of reporting a fault the reader caused on purpose.
  | { type: "cancelled"; id: number }
  | { type: "error"; id: number; message: string };
