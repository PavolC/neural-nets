// Message protocol between the main thread and the Pyodide training worker.

export interface TrainRequest {
  type: "train";
  dataUrl: string;
  epochs: number;
  hidden: number;
  miniBatchSize: number;
  eta: number;
  seed: number;
}

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

export type WorkerMessage =
  | { type: "status"; text: string }
  | { type: "log"; text: string }
  | ({ type: "epoch" } & EpochMetrics)
  | { type: "done"; result: TrainResult }
  | { type: "error"; message: string };
