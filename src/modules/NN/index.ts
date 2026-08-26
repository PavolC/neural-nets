import type { ComponentType } from "react";
import { Module1 } from "./module1";
import { Module2 } from "./module2";
import { Module3 } from "./module3";
import { Module4 } from "./module4";
import { Module5 } from "./module5";
import { Module6 } from "./module6";

export interface ModuleDef {
  id: string;
  navLabel: string;
  Component: ComponentType;
}

// Every module is reachable from the start: the reader decides the order, and
// the nav never withholds anything. (Exercise completion still gates the
// payoff panels that run the reader's own code, because those need the code to
// exist; that check lives in the panels themselves.)
export const MODULES: ModuleDef[] = [
  { id: "m1", navLabel: "1 · Neurons", Component: Module1 },
  { id: "m2", navLabel: "2 · Feedforward", Component: Module2 },
  { id: "m3", navLabel: "3 · Descent", Component: Module3 },
  { id: "m4", navLabel: "4 · Backprop", Component: Module4 },
  { id: "m5", navLabel: "5 · Training", Component: Module5 },
  { id: "m6", navLabel: "6 · Universality", Component: Module6 },
];
