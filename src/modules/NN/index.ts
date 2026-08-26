import { lazy, type ComponentType } from "react";
import { Module1 } from "./module1";

export interface ModuleDef {
  id: string;
  navLabel: string;
  Component: ComponentType;
  /** Start fetching this module's chunk without rendering it. App calls this on
   * idle for every module, so a tab switch never waits on a download. */
  preload?: () => void;
}

// Module 1 is what a first visit lands on, so it ships in the main chunk. The
// rest load separately: every module used to be parsed and rendered before
// anything painted, and modules 2 to 6 are the larger half of that work. Each
// one still stays mounted once visited, so editor and visualization state
// survives tab switches exactly as before.
function deferred(load: () => Promise<{ default: ComponentType }>) {
  return { Component: lazy(load), preload: () => void load() };
}

export const MODULES: ModuleDef[] = [
  { id: "m1", navLabel: "1 · Neurons", Component: Module1 },
  {
    id: "m2",
    navLabel: "2 · Feedforward",
    ...deferred(() => import("./module2").then((m) => ({ default: m.Module2 }))),
  },
  {
    id: "m3",
    navLabel: "3 · Descent",
    ...deferred(() => import("./module3").then((m) => ({ default: m.Module3 }))),
  },
  {
    id: "m4",
    navLabel: "4 · Backprop",
    ...deferred(() => import("./module4").then((m) => ({ default: m.Module4 }))),
  },
  {
    id: "m5",
    navLabel: "5 · Training",
    ...deferred(() => import("./module5").then((m) => ({ default: m.Module5 }))),
  },
  {
    id: "m6",
    navLabel: "6 · Universality",
    ...deferred(() => import("./module6").then((m) => ({ default: m.Module6 }))),
  },
  {
    id: "m7",
    navLabel: "7 · Making it work",
    ...deferred(() => import("./module7").then((m) => ({ default: m.Module7 }))),
  },
  {
    id: "m8",
    navLabel: "8 · Depth",
    ...deferred(() => import("./module8").then((m) => ({ default: m.Module8 }))),
  },
];
