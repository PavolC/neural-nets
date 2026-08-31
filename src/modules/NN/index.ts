import { lazy, type ComponentType } from "react";
import { Module1 } from "./module1";

export interface ModuleDef {
  id: string;
  navLabel: string;
  Component: ComponentType;
  /** Start fetching this chapter's chunk without rendering it. App calls this on
   * idle for every chapter, so a tab switch never waits on a download. */
  preload?: () => void;
}

// Chapter 1 is what a first visit lands on, so it ships in the main chunk. The
// rest load separately: every chapter used to be parsed and rendered before
// anything painted, and chapters 2 to 6 are the larger half of that work. Each
// one still stays mounted once visited, so editor and visualization state
// survives tab switches exactly as before.
function deferred(load: () => Promise<{ default: ComponentType }>) {
  return { Component: lazy(load), preload: () => void load() };
}

export const MODULES: ModuleDef[] = [
  { id: "c1", navLabel: "1 · Neurons", Component: Module1 },
  {
    id: "c2",
    navLabel: "2 · Feedforward",
    ...deferred(() => import("./module2").then((m) => ({ default: m.Module2 }))),
  },
  {
    id: "c3",
    navLabel: "3 · Descent",
    ...deferred(() => import("./module3").then((m) => ({ default: m.Module3 }))),
  },
  {
    id: "c4",
    navLabel: "4 · Backprop",
    ...deferred(() => import("./module4").then((m) => ({ default: m.Module4 }))),
  },
  {
    id: "c5",
    navLabel: "5 · Training",
    ...deferred(() => import("./module5").then((m) => ({ default: m.Module5 }))),
  },
  {
    id: "c6",
    navLabel: "6 · Universality",
    ...deferred(() => import("./module6").then((m) => ({ default: m.Module6 }))),
  },
  {
    id: "c7",
    navLabel: "7 · Making it work",
    ...deferred(() => import("./module7").then((m) => ({ default: m.Module7 }))),
  },
  {
    id: "c8",
    navLabel: "8 · Depth",
    ...deferred(() => import("./module8").then((m) => ({ default: m.Module8 }))),
  },
  {
    id: "c9",
    navLabel: "9 · Assembly",
    ...deferred(() => import("./module9").then((m) => ({ default: m.Module9 }))),
  },
  {
    id: "c10",
    navLabel: "10 · Your own data",
    ...deferred(() => import("./module10").then((m) => ({ default: m.Module10 }))),
  },
];
