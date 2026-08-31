// Fetching one exercise's tests without its module.
//
// When a run fails, the panel checks whether the cause is a section further
// up the file, by running that section's own suite. Those suites live in
// exercise modules that belong to other chapter chunks, so they are imported
// on demand: the happy path never loads any of this, and a blame pass loads
// only the sections it actually reaches.

import type { Exercise } from "./types";

const LOADERS: Readonly<Record<string, () => Promise<Exercise>>> = {
  "sigmoid-neuron": () => import("./sigmoid-neuron").then((m) => m.sigmoidNeuronExercise),
  feedforward: () => import("./feedforward").then((m) => m.feedforwardExercise),
  sgd: () => import("./sgd").then((m) => m.sgdExercise),
  backprop: () => import("./backprop").then((m) => m.backpropExercise),
  "cross-entropy": () => import("./cross-entropy").then((m) => m.crossEntropyExercise),
  "smart-init": () => import("./smart-init").then((m) => m.smartInitExercise),
  l2: () => import("./l2").then((m) => m.l2Exercise),
  train: () => import("./train").then((m) => m.trainExercise),
  prepare: () => import("./prepare").then((m) => m.prepareExercise),
};

const cache = new Map<string, Promise<Exercise>>();

/** One exercise, loaded once. Null for a section that has no tests of its
 * own, which is every section written for the learner. */
export function loadExercise(id: string): Promise<Exercise> | null {
  const loader = LOADERS[id];
  if (!loader) return null;
  let pending = cache.get(id);
  if (!pending) {
    pending = loader();
    cache.set(id, pending);
  }
  return pending;
}
