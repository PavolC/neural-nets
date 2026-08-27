/**
 * Reproduce every number Module 6 quotes, from the interactives' own code.
 *
 * Module 6 is the one module whose numbers come from a browser panel rather
 * than from Pyodide, so this bench imports the panel's arithmetic directly
 * (src/modules/NN/interactives/bumpMath.ts, which CurveSculptor and
 * BumpBuilder both call) instead of reimplementing it. A number that moves
 * here has moved in the module.
 *
 * Each section prints the prose sentence it backs, so a figure that has
 * drifted is visible without holding the module open beside it.
 *
 * Run it with the repo's own TypeScript compiler, no new dependency:
 *
 *     npm run bench:bumps
 */

import {
  BAR_COUNTS,
  DEFAULT_STEEP,
  PRESETS,
  STEEPNESS,
  areaBetween,
  bumpAt,
  fitHeights,
  sampleTarget,
} from "../src/modules/NN/interactives/bumpMath";

const DEFAULT_W = STEEPNESS[DEFAULT_STEEP]; // 400, where the playground starts

function section(title: string, claim: string): void {
  console.log("\n" + "=".repeat(78));
  console.log(title);
  console.log("prose: " + claim);
  console.log("-".repeat(78));
}

/** The module's own target: the reader's rating of an outdoor concert. */
const mine = sampleTarget(PRESETS.mine.f);
const friend = sampleTarget(PRESETS.friend.f);

// ---------------------------------------------------------------- 1
section(
  "1. What accuracy costs (the m6-price table)",
  "bars 2/4/8/16/24 score 2.311, 1.030, 0.445, 0.204 and 0.118, at two neurons " +
    "and six numbers per bar",
);
for (const k of [2, 4, 8, 16, 24]) {
  const area = areaBetween(mine, fitHeights(mine, k), k, DEFAULT_W);
  console.log(
    `  ${String(k).padStart(2)} bars   ${2 * k} hidden neurons   ${6 * k} numbers   ` +
      `area ${area.toFixed(3)}`,
  );
}

// ---------------------------------------------------------------- 2
section(
  "2. Every doubling cuts the area by slightly more than half",
  "each doubling of the bars cuts the area by slightly more than half",
);
let prev = 0;
for (const k of [2, 4, 8, 16]) {
  const area = areaBetween(mine, fitHeights(mine, k), k, DEFAULT_W);
  console.log(
    `  ${String(k).padStart(2)} bars   area ${area.toFixed(3)}` +
      (prev ? `   ${(prev / area).toFixed(2)} times better than half the bars` : ""),
  );
  prev = area;
}

// ---------------------------------------------------------------- 3
section(
  "3. The sharpness experiment",
  "at 16 bars, lowering the sharpness from 400 to 100 drops the area from 0.204 to 0.110",
);
{
  const hs = fitHeights(mine, 16);
  for (const w of STEEPNESS) {
    const area = areaBetween(mine, hs, 16, w);
    const mark = w === 400 || w === 100 ? "  <- quoted" : "";
    console.log(`  16 bars, sharpness ${String(w).padStart(3)}   area ${area.toFixed(3)}${mark}`);
  }
}

// ---------------------------------------------------------------- 4
section(
  "4. Why not a thousand bars? (the price, not diminishing returns)",
  "the fit keeps improving past the slider's 24 bars, so the answer is the bill: " +
    "1,000 bars is 2,000 hidden neurons and 6,000 numbers for one curve of one input, " +
    "against the digit reader's 23,860 for 784 inputs and ten answers",
);
for (const k of [24, 48, 96, 250, 1000]) {
  const area = areaBetween(mine, fitHeights(mine, k), k, DEFAULT_W);
  console.log(
    `  ${String(k).padStart(4)} bars   ${String(2 * k).padStart(4)} hidden neurons   ` +
      `${String(6 * k).padStart(4)} numbers   area ${area.toFixed(4)}`,
  );
}

// ---------------------------------------------------------------- 5
section(
  "5. One bump, built by hand (the m6-bump table and the by-hand bias)",
  "two step neurons at 0.40 and 0.60 with weight 400 and output wires +6 and -6 give " +
    "a bump 6.00 tall with biases -160 and -240; b = -ws puts a step at 0.75 with bias -300; " +
    "at steepness 40 the same pair peaks at 5.78 instead of 6",
);
for (const w of [40, 100, 400]) {
  const peak = bumpAt(w, 0.4, 0.6, 6, 0.5);
  console.log(
    `  weight ${String(w).padStart(3)}   biases ${(-w * 0.4).toFixed(0)} and ` +
      `${(-w * 0.6).toFixed(0)}   switchover ${(6 / w).toFixed(3)} of the dial   ` +
      `peak ${peak.toFixed(2)}`,
  );
}
for (const s of [0.4, 0.6, 0.75]) {
  console.log(`  a step at ${s.toFixed(2)} with weight 400 needs bias ${(-400 * s).toFixed(0)}`);
}

// ---------------------------------------------------------------- 6
section(
  "6. The playground's opening state and the friend's curve",
  "context rather than quoted prose: the flat start the reader first sees, and the " +
    "second preset's fits",
);
console.log(
  `  6 bars, all heights 0, sharpness ${DEFAULT_W}   area ` +
    `${areaBetween(mine, new Array(6).fill(0), 6, DEFAULT_W).toFixed(3)}`,
);
for (const k of BAR_COUNTS) {
  const area = areaBetween(friend, fitHeights(friend, k), k, DEFAULT_W);
  console.log(`  friend's curve, ${String(k).padStart(2)} bars   area ${area.toFixed(3)}`);
}
console.log();

// ---------------------------------------------------------------- 7
section(
  "7. Module 1's XOR network, read as a bump (the m6-bump recognition table)",
  "two hidden neurons with weights 6 and 6 and biases -3 and -9, wires out +8 and -8 " +
    "and an output bias of -4, report 0.05/0.00, 0.95/0.05 and 1.00/0.95 at the totals " +
    "0, 1 and 2, so the evidence into the output runs -3.6, +3.2, -3.6; the bump reaches " +
    "about 7.2 rather than 8 because a weight of 6 makes each switchover 6/6 = 1 wide " +
    "while the two steps sit 1 apart",
);
// The same bumpAt the playground draws with, at Module 1's arguments: height 8,
// steepness 6, steps at 0.5 and 1.5 along the total x1 + x2. That the identity
// checks out through the panel's own function is the point of the beat.
const XOR = { w: 6, s1: 0.5, s2: 1.5, h: 8, outBias: -4 };
const report = (bias: number, total: number) => 1 / (1 + Math.exp(-(XOR.w * total + bias)));
for (const total of [0, 1, 2]) {
  const first = report(-3, total);
  const second = report(-9, total);
  const bump = bumpAt(XOR.w, XOR.s1, XOR.s2, XOR.h, total);
  const evidence = bump + XOR.outBias;
  console.log(
    `  total ${total}   first ${first.toFixed(2)}   second ${second.toFixed(2)}   ` +
      `bump ${bump.toFixed(3)}   evidence ${evidence >= 0 ? "+" : ""}${evidence.toFixed(1)}   ` +
      `squashed ${(1 / (1 + Math.exp(-evidence))).toFixed(3)}`,
  );
}
console.log(
  `  switchover 6/${XOR.w} = ${(6 / XOR.w).toFixed(2)} wide, steps ` +
    `${(XOR.s2 - XOR.s1).toFixed(2)} apart, so no flat top: peak ` +
    `${bumpAt(XOR.w, XOR.s1, XOR.s2, XOR.h, (XOR.s1 + XOR.s2) / 2).toFixed(2)} of ${XOR.h}`,
);
console.log();
