import type { ComponentType } from "react";
import { loadCompleted } from "../../state/progress";
import { Module1 } from "./module1";
import { Module2 } from "./module2";
import { Module3 } from "./module3";
import { Module4 } from "./module4";
import { Module5 } from "./module5";

export interface ModuleDef {
  id: string;
  navLabel: string;
  /** Exercises that must pass before the NEXT module unlocks. */
  exerciseIds: string[];
  Component: ComponentType;
}

export const MODULES: ModuleDef[] = [
  { id: "m1", navLabel: "1 · Neurons", exerciseIds: ["sigmoid-neuron"], Component: Module1 },
  { id: "m2", navLabel: "2 · Feedforward", exerciseIds: ["feedforward"], Component: Module2 },
  { id: "m3", navLabel: "3 · Descent", exerciseIds: ["sgd"], Component: Module3 },
  // Module 4 deliberately has no exercise (a quiz instead), so it gates
  // nothing beyond what Module 3's exercise already gates.
  { id: "m4", navLabel: "4 · Backprop", exerciseIds: [], Component: Module4 },
  { id: "m5", navLabel: "5 · Training", exerciseIds: ["backprop"], Component: Module5 },
];

/** A module unlocks when every exercise of every earlier module has passed.
 * Navigation back to anything already unlocked is always allowed. */
export function isModuleUnlocked(index: number): boolean {
  return MODULES.slice(0, index).every((m) => m.exerciseIds.every(loadCompleted));
}
