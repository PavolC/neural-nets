/** One exercise, named without pulling in its Python.
 *
 * The exercise objects themselves carry a skeleton, a test suite and a
 * solution as strings, so importing all seven of them to list their titles
 * would put every line of course Python in the first chunk the reader
 * downloads. This is the list, and nothing else. The ids are the localStorage
 * keys, so they must match the `id` in each exercise's index.ts; the module
 * ids must match src/modules/NN/index.ts, which the check below enforces.
 */
export interface ExerciseRef {
  id: string;
  title: string;
  /** Module id from MODULES, for linking. */
  module: string;
  /** What the learner ends up with, in one line. */
  builds: string;
}

export const EXERCISES: ExerciseRef[] = [
  {
    id: "sigmoid-neuron",
    title: "A sigmoid neuron",
    module: "m1",
    builds: "sigmoid and fire: the squash, and one neuron's weighted sum plus bias",
  },
  {
    id: "feedforward",
    title: "Feedforward",
    module: "m2",
    builds: "feedforward: a whole network, one matrix multiplication per layer",
  },
  {
    id: "sgd",
    title: "Stochastic gradient descent",
    module: "m3",
    builds: "sgd_step and sgd: descent with mini-batches and a shuffle",
  },
  {
    id: "backprop",
    title: "Backpropagation",
    module: "m5",
    builds: "backprop: BP1 to BP4, checked against numerically measured gradients",
  },
  {
    id: "cross-entropy",
    title: "The cross-entropy cost",
    module: "m7",
    builds: "cross_entropy_cost and cross_entropy_delta: a cost whose blame is the gap",
  },
  {
    id: "smart-init",
    title: "A better starting point",
    module: "m7",
    builds: "init_network: weights divided by the square root of their input count",
  },
  {
    id: "l2",
    title: "The decaying update step",
    module: "m7",
    builds: "l2_step: the update rule with weight decay",
  },
];
