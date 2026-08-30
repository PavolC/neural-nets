// The starting text of every section, and nothing else.
//
// The workbench seeds a section the first time the learner reaches it, which
// has to happen without loading that exercise's tests and reference solution,
// so those stay in each exercise's own index.ts and its own chunk. This file
// is the skeletons alone (about 17 KB of comments and stubs) plus the two
// sections written for the learner, which have one body rather than two.

import sigmoidNeuron from "./sigmoid-neuron/skeleton.py?raw";
import feedforward from "./feedforward/skeleton.py?raw";
import sgd from "./sgd/skeleton.py?raw";
import backprop from "./backprop/skeleton.py?raw";
import crossEntropy from "./cross-entropy/skeleton.py?raw";
import smartInit from "./smart-init/skeleton.py?raw";
import l2 from "./l2/skeleton.py?raw";
import train from "./train/skeleton.py?raw";
import prepare from "./prepare/skeleton.py?raw";
import givenCost from "./given/cost.py?raw";
import givenBatch from "./given/batch.py?raw";

/** Section id to the body it starts life with. */
export const SECTION_BODIES: Readonly<Record<string, string>> = {
  "sigmoid-neuron": sigmoidNeuron,
  feedforward,
  "given-cost": givenCost,
  sgd,
  backprop,
  "given-batch": givenBatch,
  "cross-entropy": crossEntropy,
  "smart-init": smartInit,
  l2,
  train,
  prepare,
};

export function startingBody(id: string): string {
  return (SECTION_BODIES[id] ?? "").replace(/\s+$/, "");
}
