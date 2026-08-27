/**
 * Reproduce every layer-speed number Module 8 quotes, from the panel's own code.
 *
 * Module 8's tables come from two engines, and this is the second of the two
 * benches that regenerate them (see tools/bench_depth.py for the first):
 *
 *   * tools/bench_depth.py mirrors DepthTrainPanel, the Pyodide run: accuracy
 *     tables, the step-size sweep, dead units, ratios measured during training.
 *   * tools/bench_layer_speeds.ts (this file) mirrors LayerSpeedBars by
 *     importing the panel's own measure() from deepNet.ts. Everything in the
 *     module's learning-speed and hop tables is quoted from here.
 *
 * The two engines draw their weights from different generators (NumPy's PCG64
 * against the panel's mulberry32), so their numbers are not meant to agree.
 * They agree on the pattern, which is the module's whole claim; a number from
 * one engine must never be quoted as if it came from the other.
 *
 * Run it with the repo's own TypeScript compiler, no new dependency:
 *
 *     npm run bench:speeds
 */

import { makeBatch, measure, type Activation, type Batch } from "../src/modules/NN/interactives/deepNet";
import { mulberry32 } from "../src/modules/NN/interactives/utils";

// Node's own modules, reached through require so the repo needs no @types/node
// (the app's tsconfig deliberately declares only vite/client). npm run
// bench:speeds compiles this file to CommonJS, where require is what runs.
declare const require: (id: string) => {
	gunzipSync(buf: unknown): { buffer: ArrayBuffer; byteOffset: number; length: number };
	readFileSync(path: string): unknown;
};
const { gunzipSync } = require("node:zlib");
const { readFileSync } = require("node:fs");

// LayerSpeedBars' constants. Keep these in step with the component.
const HIDDEN_SIZE = 30;
const SAMPLES = 200;
const DEFAULT_SEED = 191;

// Run from the repo root (npm run bench:speeds does).
const DATA = "public/data/mnist_subset.bin.gz";

/** Parse the test split out of the bundled subset, exactly as assets.ts does. */
function loadBatch(): Batch {
	const raw = gunzipSync(readFileSync(DATA));
	const bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.length);
	const magic = new TextDecoder().decode(bytes.subarray(0, 4));
	if (magic !== "MNSS") throw new Error(`bad MNIST subset magic: ${magic}`);
	const view = new DataView(bytes.buffer, bytes.byteOffset);
	const nTrain = view.getUint32(5, true);
	const nTest = view.getUint32(9, true);
	const pixels = view.getUint32(13, true) * view.getUint32(17, true);
	const testStart = 21 + nTrain * pixels + nTrain;
	const images = bytes.subarray(testStart, testStart + nTest * pixels);
	const labels = bytes.subarray(testStart + nTest * pixels, testStart + nTest * pixels + nTest);
	return makeBatch(images, labels, SAMPLES);
}

const sig = (v: number, digits = 4) => Number(v.toPrecision(digits));
const ratio = (v: number) => (v < 100 ? v.toFixed(1) : v.toFixed(0));

function section(title: string, claim: string) {
	console.log();
	console.log("=".repeat(78));
	console.log(title);
	console.log("prose:", claim);
	console.log("-".repeat(78));
}

function speeds(batch: Batch, hidden: number, activation: Activation, weightScale: number, seed = DEFAULT_SEED) {
	return measure(batch, { hidden, hiddenSize: HIDDEN_SIZE, activation, weightScale, seed });
}

function benchStaircase(batch: Batch) {
	section(
		"1. Every layer's learning speed, 784-30-30-30-30-10, before the first step",
		"0.002539 / 0.0124 / 0.06668 / 0.2652 / 1.439, so the output moves 567 times as far as layer 2",
	);
	const m = speeds(batch, 4, "sigmoid", 1);
	for (const l of m.layers) {
		console.log(
			`  layer ${l.layer}${l.isOutput ? " (output)" : ""}: speed ${sig(l.speed)}` +
				(l.hop === null ? "" : `, hop ${sig(l.hop, 3)}`),
		);
	}
	console.log(`  output ahead of layer 2 by ${ratio(m.slowdown)}x`);
}

function benchDepthLadder(batch: Batch) {
	section(
		"2. The gap grows by a constant factor per hidden layer (sigmoid)",
		"1 to 5 hidden layers put the output ahead of the first by 4.8, 26.2, 117, 567 and 3,002 times, against 5^depth = 5, 25, 125, 625, 3125",
	);
	for (let hidden = 1; hidden <= 5; hidden++) {
		const m = speeds(batch, hidden, "sigmoid", 1);
		console.log(`  ${hidden} hidden: ${ratio(m.slowdown)}x   (5^${hidden} = ${5 ** hidden})`);
	}
}

function benchHops(batch: Batch) {
	section(
		"3. BP2's two factors per hop, 4 hidden layers, sigmoid",
		"ledger 0.878 / 1.159 / 0.906 / 0.962, steepness 0.210 / 0.217 / 0.205 / 0.213, hops 0.184 / 0.251 / 0.186 / 0.205",
	);
	const m = speeds(batch, 4, "sigmoid", 1);
	const back = [...m.layers].reverse();
	for (let i = 1; i < back.length; i++) {
		const l = back[i];
		const from = back[i - 1];
		console.log(
			`  layer ${from.layer} to ${l.layer}: from ${sig(from.speed)}` +
				`, ledger ${sig(l.weightFactor ?? 0, 3)}, steepness ${sig(l.steepFactor ?? 0, 3)}` +
				`, hop ${sig(l.hop ?? 0, 3)}, lands on ${sig(l.speed)}`,
		);
	}
}

function benchLedgerSpread() {
	section(
		"4. A divided 30-by-30 ledger sends a column back at about its own length",
		"averaged over 200 draws the length comes back at 0.99 of what went in, with the middle 90 percent of those draws between 0.80 and 1.19",
	);
	// The same draw the panel uses (a standard normal divided by sqrt of the
	// input count), applied to one random unit column, measuring |W^T d| / |d|.
	const n = HIDDEN_SIZE;
	const rand = mulberry32(DEFAULT_SEED);
	const gauss = () => {
		const u = Math.max(rand(), 1e-12);
		const v = rand();
		return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
	};
	const factors: number[] = [];
	for (let t = 0; t < 200; t++) {
		const W: number[][] = Array.from({ length: n }, () =>
			Array.from({ length: n }, () => gauss() / Math.sqrt(n)),
		);
		const d = Array.from({ length: n }, () => gauss());
		const dNorm = Math.hypot(...d);
		const out = Array.from({ length: n }, (_, k) => {
			let s = 0;
			for (let i = 0; i < n; i++) s += W[i][k] * d[i];
			return s;
		});
		factors.push(Math.hypot(...out) / dNorm);
	}
	factors.sort((a, b) => a - b);
	const mean = factors.reduce((a, b) => a + b, 0) / factors.length;
	console.log(`  mean ${mean.toFixed(3)}, min ${factors[0].toFixed(2)}, max ${factors[199].toFixed(2)}`);
	console.log(`  middle 90%: ${factors[10].toFixed(2)} to ${factors[189].toFixed(2)}`);
	console.log("  (the middle 90% is what the module quotes: the extremes of one");
	console.log("   200-draw stream move by a lot from run to run, the quantiles do not)");
}

function benchRelu(batch: Batch) {
	section(
		"5. ReLU's hops, 4 hidden layers",
		"hops 0.634, 0.571, 0.900 and 0.771, and the output ends up 4.0 times ahead of layer 2",
	);
	const m = speeds(batch, 4, "relu", 1);
	const hops = [...m.layers]
		.reverse()
		.slice(1)
		.map((l) => sig(l.hop ?? 0, 3));
	console.log(`  hops (output-most first): ${hops.join(", ")}`);
	console.log(`  output ahead of layer 2 by ${ratio(m.slowdown)}x`);
}

function benchScale(batch: Batch) {
	section(
		"6. The hop is a product, and the weight multiplier walks it (4 hidden layers)",
		"sigmoid x1/2/4/8: ledger 0.962/1.914/3.630/7.220, steepness 0.213/0.206/0.170/0.126, " +
			"mean steepness 0.20/0.18/0.14/0.08, hop into layer 2 0.205/0.394/0.618/0.908, slowdown 567/60.4/15.0/4.2; " +
			"relu x1/2/4: hop into layer 2 0.634/1.456/2.990, layer 2 speed 0.35/9.74/136.32, output speed 1.40/1.73/1.52",
	);
	for (const activation of ["sigmoid", "relu"] as Activation[]) {
		const scales = activation === "sigmoid" ? [1, 2, 4, 8] : [1, 2, 4];
		for (const scale of scales) {
			const m = speeds(batch, 4, activation, scale);
			const first = m.layers[0]; // layer 2, the hop into it
			const out = m.layers[m.layers.length - 1];
			// The module quotes the steepness averaged over ALL the hidden layers,
			// not layer 2's alone: it is being read against the sigmoid's 0.25
			// ceiling, which is a fact about the squash rather than about one layer.
			const hiddenSteep = m.layers.filter((l) => !l.isOutput).map((l) => l.meanSteepness ?? 0);
			const meanHiddenSteep = hiddenSteep.reduce((a, b) => a + b, 0) / hiddenSteep.length;
			console.log(
				`  ${activation} x${scale}: ledger ${sig(first.weightFactor ?? 0, 3)}` +
					`, steepness ${sig(first.steepFactor ?? 0, 3)}` +
					`, mean steepness over the hidden layers ${meanHiddenSteep.toFixed(2)}` +
					` (layer 2 alone ${(first.meanSteepness ?? 0).toFixed(2)})` +
					`, hop ${sig(first.hop ?? 0, 3)}` +
					`, layer 2 speed ${sig(first.speed, 3)}, output speed ${sig(out.speed, 3)}` +
					`, layer 2 moves ${sig(first.speed / out.speed, 3)} times as far as the output` +
					` (output ahead by ${ratio(m.slowdown)}x)`,
			);
		}
	}
}

function benchSeedSpread(batch: Batch) {
	section(
		"7. Is the default seed representative? (sigmoid, 4 hidden layers)",
		"seed 191 was picked because its draw lands near the middle: 567x against a median of about 650 over forty draws",
	);
	const slowdowns: number[] = [];
	for (let seed = 1; seed <= 40; seed++) slowdowns.push(speeds(batch, 4, "sigmoid", 1, seed).slowdown);
	slowdowns.sort((a, b) => a - b);
	const median = (slowdowns[19] + slowdowns[20]) / 2;
	console.log(`  over 40 draws: median ${ratio(median)}x, min ${ratio(slowdowns[0])}x, max ${ratio(slowdowns[39])}x`);
	console.log(`  seed ${DEFAULT_SEED} (the panel's default): ${ratio(speeds(batch, 4, "sigmoid", 1).slowdown)}x`);
}

const batch = loadBatch();
console.log(`${SAMPLES} test images, hidden layers of ${HIDDEN_SIZE}, default seed ${DEFAULT_SEED}`);
benchStaircase(batch);
benchDepthLadder(batch);
benchHops(batch);
benchLedgerSpread();
benchRelu(batch);
benchScale(batch);
benchSeedSpread(batch);
console.log();
