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

export type WorkerRequest =
  | ({ type: "train"; id: number; dataUrl: string } & TrainParams)
  | { type: "runTests"; id: number; learnerCode: string; testsCode: string };

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
}

export interface TestRunResult {
  setup_error: { message: string; line: number | null } | null;
  tests: TestResultEntry[];
  passed: boolean;
}

export type WorkerResponse =
  | { type: "status"; id: number; text: string }
  | { type: "log"; id: number; text: string }
  | ({ type: "epoch"; id: number } & EpochMetrics)
  | { type: "trainDone"; id: number; result: TrainResult }
  | { type: "testsDone"; id: number; result: TestRunResult }
  | { type: "error"; id: number; message: string };
