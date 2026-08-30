// What a locked payoff panel says it is waiting on.
//
// A panel's gate is codeReady over everything its projection runs, so the
// list can name a section from a much earlier module: a learner who opens
// Module 3 first passes sgd with feedforward on loan, and the Module 3 panel
// then waits on Module 2. One phrase map for the whole course, so every
// panel names a section the same way; a panel overrides an entry only for a
// section on its own page ("the exercise above"). The fallback is the
// section table's label, so a section this map does not know is still named
// rather than dropped.

import { notReadyFor } from "../../../state/progress";

const PHRASE: Record<string, string> = {
  "sigmoid-neuron": "Module 1's sigmoid exercise",
  feedforward: "Module 2's feedforward exercise",
  "given-cost": "the section written for you in Module 3, back in your file",
  sgd: "Module 3's sgd exercise",
  backprop: "Module 5's backprop exercise",
  "given-batch": "the section written for you in Module 5, back in your file",
  "cross-entropy": "Module 7's cross-entropy exercise",
  "smart-init": "Module 7's starting-point exercise",
  l2: "Module 7's decaying-step exercise",
  train: "Module 9's program",
  prepare: "Module 10's preparation exercise",
};

/** The not-ready sections these exercises run on, each named the way the
 * course speaks about it, in course order. Empty means the panel may run. */
export function lockedBy(
  exerciseIds: readonly string[],
  overrides: Record<string, string> = {},
): string[] {
  return notReadyFor(exerciseIds).map((s) => overrides[s.id] ?? PHRASE[s.id] ?? s.label);
}

/** "a", "a and b", "a, b and c": the list as a locked note speaks it. */
export function speakList(items: readonly string[]): string {
  if (items.length <= 2) return items.join(" and ");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}
