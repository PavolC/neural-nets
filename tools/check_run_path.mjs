// Drive the whole run path in a real browser, with a stub Pyodide.
//
// Everything else in tools/ checks the Python or the document format. Nothing
// checked what the panel does with a verdict once it has one: the message
// protocol, the run state, the results in each of their shapes, the borrowed
// names line, the output stream, and the three ways to start a run. Booting the
// real Pyodide for that is 10 MB and several seconds per case, and it cannot
// run at all where the CDN is unreachable. The worker needs exactly four
// methods from Pyodide, so this serves a module with those four and answers
// with a canned verdict.
//
// Needs a Chromium and a dev server:
//
//     npm run dev &
//     npx playwright@latest install chromium     # once
//     node tools/check_run_path.mjs
//
// Not in `npm run check`, and not in CI: it wants a browser and a running
// server, which is the same bargain tools/make_og_image.sh already makes.

// Imported dynamically so a missing Playwright is a sentence rather than a
// stack trace: it is not a dependency of this repo and should not become one
// for a check that already needs a browser and a running server.
let chromium;
try {
	({ chromium } = await import("playwright"));
} catch {
	console.error(
		"This check needs Playwright, which this repo does not depend on:\n" +
			"    npm i --no-save playwright && npx playwright install chromium\n" +
			"and a dev server on http://localhost:5174 (npm run dev).",
	);
	process.exit(2);
}

// Where the browser is. Left to Playwright unless something points elsewhere,
// which is what a sandbox with a preinstalled Chromium does.
const LAUNCH = process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {};
const ORIGIN = process.env.COURSE_ORIGIN ?? "http://localhost:5174";
const MOD = process.platform === "darwin" ? "Meta" : "Control";

const fakeModule = (verdict) => `
export async function loadPyodide(opts) {
  return {
    loadPackage: async () => {},
    FS: { writeFile: () => {} },
    globals: { set: () => {} },
    runPythonAsync: async (code) => {
      if (code.includes("sys.version")) return "Python 3.14.0, NumPy 2.4.6 (stub)";
      if (code.startsWith("run_document(")) {
        opts.stdout("a line the learner printed");
        return ${JSON.stringify(JSON.stringify(verdict))};
      }
      if (code.startsWith("run_document_scratch(")) {
        opts.stdout("scratch ran");
        return JSON.stringify({ error: null, lent: [] });
      }
      return null;
    },
  };
}
`;

const pass = [
	"sigmoid(0) is exactly one half",
	"sigmoid works elementwise on arrays",
	"sigmoid squashes everything into (0, 1)",
	"fire returns a plain number, not an array",
	"fire computes sigmoid(w . x + b)",
	"the bias shifts the neuron's output",
].map((title, i) => ({ name: `tp${i}`, title, passed: true, message: "", section: null }));

const VERDICTS = {
	"everything passes": { setup_error: null, passed: true, lent: [], tests: pass },
	"two fail, four pass, two names borrowed": {
		setup_error: null,
		passed: false,
		lent: ["feedforward", "init_network"],
		tests: [
			...pass.slice(0, 4),
			{
				name: "tf1",
				title: "fire computes sigmoid(w . x + b)",
				passed: false,
				section: null,
				message: "expected 0.7311 for w=[1,1], b=0, x=[1,0], got 0.5. Check that the bias is added before the squash.",
			},
			{ name: "tf2", title: "the bias shifts the neuron's output", passed: false, section: null, message: "raising b by 1 should raise the output; it fell." },
		],
	},
	"the file does not run": {
		setup_error: { message: "SyntaxError: invalid syntax", line: 34, section: "sigmoid-neuron" },
		passed: false,
		lent: [],
		tests: [],
	},
	"the failure is upstream": {
		setup_error: null,
		passed: false,
		lent: [],
		tests: [
			{
				name: "tx",
				title: "feedforward's two-layer values",
				passed: false,
				section: "sigmoid-neuron",
				message: "your code raised ValueError at line 118 of your file, which is inside your Chapter 1, A sigmoid neuron section. That section is where the error is, not this one.",
			},
		],
	},
};

let failures = 0;

for (const [label, verdict] of Object.entries(VERDICTS)) {
	const b = await chromium.launch(LAUNCH);
	const ctx = await b.newContext({ viewport: { width: 1600, height: 950 } });
	await ctx.route("**/pyodide/**/pyodide.mjs", (route) => route.fulfill({ status: 200, contentType: "text/javascript", body: fakeModule(verdict) }));
	const p = await ctx.newPage();
	const errors = [];
	p.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));

	await p.goto(ORIGIN + "/#m1", { waitUntil: "networkidle" });
	await p.locator(".wb-handle").click();
	await p.waitForSelector(".cm-content", { timeout: 20000 });
	// The notebook chord, from inside the editor, rather than the button: if it
	// works the whole path works, and the button is the same call.
	if (label === "everything passes") {
		await p.locator(".cm-content").click();
		await p.keyboard.press(`${MOD}+Enter`);
	} else {
		await p.locator(".wb-head button", { hasText: "Run tests" }).click();
	}
	await p.waitForSelector(".test-results", { timeout: 30000 }).catch(() => {});
	await p.waitForTimeout(1200);

	const s = await p.evaluate(() => ({
		summary: document.querySelector(".test-summary")?.textContent?.trim() ?? null,
		setupError: document.querySelector(".test-fail strong")?.textContent?.trim() ?? null,
		failCards: document.querySelectorAll(".test-fail").length,
		folded: document.querySelector(".test-passed > summary")?.textContent?.trim() ?? null,
		// The hints and the test code live beside the code now, not in the page.
		hintsInPanel: document.querySelectorAll(".wb-hints button").length > 0,
		testCodeInPanel: !!document.querySelector(".wb-tests"),
		backToCode: !!document.querySelector(".wb-back"),
		railGone: !document.querySelector(".wb-rail"),
		foldedOpen: document.querySelector(".test-passed")?.hasAttribute("open") ?? null,
		// Only when something really was borrowed: "run entirely on your own
		// code" is the ordinary case, and a line saying nothing happened is noise
		// under every passing run.
		lent: document.querySelector(".wb-lent")?.textContent?.trim() ?? null,
		// Whatever the run is saying, on its own line under the head. In the head
		// it was the sixth thing across the row and rendered as "R...".
		status: document.querySelector(".wb-status")?.textContent?.trim() ?? null,
		goTo: document.querySelector(".wb-goto")?.textContent?.trim() ?? null,
		resultsPx: Math.round(document.querySelector(".test-results")?.getBoundingClientRect().height ?? 0),
		output: document.querySelector(".output-panel pre")?.textContent?.trim().slice(0, 30) ?? null,
		// The bar is panel chrome, outside the one scroll region, so it is on
		// screen however far down the file the reader is.
		runAlwaysVisible: !document.querySelector(".wb-flow").contains(document.querySelector(".wb-run")),
		// Visible ones only: a closed <details> still reports a scrollHeight in
		// Chromium, which counted blocks nobody can see. The panel is meant to be
		// the flow plus the deliberately capped output pane, and nothing else.
		scrollRegions: [...document.querySelectorAll(".wb *")]
			.filter((e) => {
				const cs = getComputedStyle(e);
				return e.offsetParent !== null && (cs.overflowY === "auto" || cs.overflowY === "scroll") && e.scrollHeight > e.clientHeight + 2;
			})
			.map((e) => e.className.toString() || e.tagName),
		editorHasItsOwnScroller: (() => {
			const s = document.querySelector(".wb-editor .cm-scroller");
			return !!s && s.scrollHeight > s.clientHeight + 2;
		})(),
	}));
	console.log(`\n${label}`);
	for (const [k, v] of Object.entries(s)) if (v !== null) console.log(`  ${k}: ${JSON.stringify(v)}`);
	const borrowed = verdict.lent.length > 0;
	if (label === "everything passes" && s.summary !== "All tests passed.") {
		failures++;
		console.log("  Mod-Enter did not render the passing verdict");
	}
	if (borrowed !== (s.lent !== null)) {
		failures++;
		console.log(`  the borrowed line is wrong: lent ${JSON.stringify(verdict.lent)}, line ${JSON.stringify(s.lent)}`);
	}
	if (errors.length) {
		failures++;
		console.log("  ERRORS:", errors.slice(0, 3));
	}
	await b.close();
}

// The other two ways to start a run, which are the notebook affordances.
{
	const b = await chromium.launch(LAUNCH);
	const ctx = await b.newContext({ viewport: { width: 1600, height: 950 } });
	await ctx.route("**/pyodide/**/pyodide.mjs", (route) => route.fulfill({ status: 200, contentType: "text/javascript", body: fakeModule(VERDICTS["everything passes"]) }));
	const p = await ctx.newPage();
	// Seed two sections, so there is a marker belonging to a section other than
	// the current one. With only one in the file there is nothing to prove.
	for (const hash of ["#m1", "#m2"]) {
		await p.goto(ORIGIN + "/" + hash, { waitUntil: "networkidle" });
		const launcher = p.locator("main > div:not([hidden]) .exercise-launcher-open").first();
		await launcher.scrollIntoViewIfNeeded();
		await launcher.click();
		await p.waitForSelector(".cm-content", { timeout: 20000 });
		await p.waitForTimeout(500);
	}

	console.log("\nShift-Enter runs the scratch pad");
	await p.locator(".wb-editor .cm-content").click();
	await p.keyboard.press("Shift+Enter");
	await p.waitForTimeout(2500);
	console.log("  output:", JSON.stringify((await p.locator(".output-panel pre").textContent())?.trim().slice(0, 20)));
	console.log("  no test results, as it should be:", (await p.locator(".test-results").count()) === 0);

	console.log("\nthe section picker scrolls the one flow to that section");
	await p.evaluate(() => {
		document.querySelector(".wb-flow").scrollTop = 99999;
	});
	await p.waitForTimeout(300);
	const deep = await p.evaluate(() => Math.round(document.querySelector(".wb-flow").scrollTop));
	await p.locator(".wb-sections > summary").click();
	await p.locator(".wb-section-item", { hasText: "A sigmoid neuron" }).click();
	await p.waitForTimeout(600);
	const back = await p.evaluate(() => Math.round(document.querySelector(".wb-flow").scrollTop));
	console.log(`  scrolled ${deep} -> ${back}, moved: ${back < deep}`);
	if (!(back < deep)) failures++;

	console.log("\nthe gutter marker runs its own section, not the current one");
	// The picker owns the section name; .wb-target is the run status now.
	const target = () =>
		p
			.locator(".wb-sections-label")
			.textContent()
			.then((s) => s.trim());
	// Point the panel at Chapter 2 first, so clicking Chapter 1's marker has
	// something to prove. The chip above already moved it to Chapter 1.
	await p.locator(".wb-sections > summary").click();
	await p.locator(".wb-section-item", { hasText: "Feedforward" }).click();
	await p.waitForTimeout(400);
	const before = await target();
	// The panel is one scroll now, so this is the thing that moves.
	await p.evaluate(() => {
		document.querySelector(".wb-flow").scrollTop = 0;
	});
	await p.waitForTimeout(300);
	const marker = p.locator(".cm-run-gutter .cm-run-section:visible").first();
	console.log("  markers in view:", await p.locator(".cm-run-gutter .cm-run-section:visible").count());
	await marker.click();
	await p.waitForTimeout(2500);
	const after = await target();
	console.log("  target before:", JSON.stringify(before));
	console.log("  target after :", JSON.stringify(after));
	console.log("  the marker carried its own section:", after !== before);
	if (after === before) failures++;

	// CodeMirror paints the selection in a layer BEHIND the content, so any
	// background the theme puts on a line has to be an alpha tint. The two
	// course tints, --accent-wash and --accent-panel, mix with the page ground
	// and are opaque, and the section highlight covers every line of the
	// section the reader is working in: the selection was invisible exactly
	// where they type.
	console.log("\nthe selection reads on every ground the editor paints");
	await p.locator(".wb-editor .cm-content").click();
	await p.keyboard.press("Control+Home");
	for (let i = 0; i < 6; i++) await p.keyboard.press("Shift+ArrowDown");
	await p.waitForTimeout(300);
	const sel = await p.evaluate(() => {
		const alpha = (c) => {
			const m = c.match(/[\d.]+\s*\)\s*$/);
			return /rgba|\/\s*[\d.]+\s*\)/.test(c) && m ? parseFloat(m[0]) : c === "rgba(0, 0, 0, 0)" ? 0 : 1;
		};
		const opaque = [];
		for (const cls of ["cm-activeLine", "cm-section-here"]) {
			const line = document.querySelector(".wb-editor .cm-line." + cls);
			if (line && alpha(getComputedStyle(line).backgroundColor) >= 1) opaque.push(cls);
		}
		const boxes = [...document.querySelectorAll(".wb-editor .cm-selectionBackground")];
		return { boxes: boxes.length, bg: boxes[0] ? getComputedStyle(boxes[0]).backgroundColor : null, opaque };
	});
	console.log("  selection boxes:", sel.boxes, "painted", JSON.stringify(sel.bg));
	console.log("  line backgrounds that would hide it:", JSON.stringify(sel.opaque));
	if (!sel.boxes || sel.opaque.length) failures++;
	// CodeMirror's own light theme is #d7d4f0, which wins on specificity unless
	// the theme spells out the whole child chain the base rule uses.
	if (sel.bg && /215,\s*212,\s*240/.test(sel.bg)) {
		failures++;
		console.log("  the selection is CodeMirror's lavender, not the course accent");
	}
	await b.close();
}

// The stateful paths around a run: a result becomes historical as soon as
// its document changes, progress imports merge rather than erase local-only
// sections, and resizing a user-sized dock updates both layouts and ARIA.
{
	const b = await chromium.launch(LAUNCH);
	const ctx = await b.newContext({ viewport: { width: 1600, height: 950 } });
	await ctx.route("**/pyodide/**/pyodide.mjs", (route) => route.fulfill({ status: 200, contentType: "text/javascript", body: fakeModule(VERDICTS["everything passes"]) }));
	const p = await ctx.newPage();
	const errors = [];
	p.on("pageerror", (e) => errors.push(e.message));

	await p.goto(ORIGIN + "/#m1", { waitUntil: "networkidle" });
	await p.locator(".wb-handle").click();
	await p.waitForSelector(".cm-content", { timeout: 20000 });
	await p.locator(".wb-run").click();
	await p.waitForSelector(".test-summary-pass", { timeout: 30000 });
	const importedDocument = await p.evaluate(() => localStorage.getItem("gn:v1:code:workbench"));

	console.log("\nan edit makes the last verdict historical");
	await p.locator(".wb-editor .cm-content").click();
	await p.keyboard.press("Control+End");
	await p.keyboard.type("\n# changed after the passing run");
	await p.waitForSelector(".test-stale-note");
	const stale = await p.evaluate(() => ({
		section: document.querySelector(".wb-section-current .wb-section-state")?.textContent?.trim(),
		note: document.querySelector(".test-stale-note")?.textContent?.trim(),
	}));
	console.log("  section state:", JSON.stringify(stale.section));
	console.log("  verdict note :", JSON.stringify(stale.note));
	if (stale.section !== "changed since" || !stale.note?.startsWith("From an earlier run")) failures++;

	await p.goto(ORIGIN + "/#m2", { waitUntil: "networkidle" });
	const launcher = p.locator("main > div:not([hidden]) .exercise-launcher-open").first();
	await launcher.scrollIntoViewIfNeeded();
	await launcher.click();
	await p.waitForFunction(() => localStorage.getItem("gn:v1:code:workbench")?.includes("[section:feedforward]"));
	await p.locator(".wb-editor .cm-content").click();
	await p.keyboard.press("Control+End");
	await p.keyboard.type("\n# receiver-only Chapter 2 work");
	await p.waitForFunction(() => localStorage.getItem("gn:v1:code:workbench")?.includes("receiver-only Chapter 2 work"));

	console.log("\na current progress file merges without losing local sections");
	const progress = {
		format: "nets-course-progress-v1",
		saved: new Date().toISOString(),
		entries: {
			"code:workbench": importedDocument,
			"code:sigmoid-neuron": importedDocument,
		},
	};
	await p.locator("input[type=file]").setInputFiles({
		name: "module-1-only-progress.json",
		mimeType: "application/json",
		buffer: Buffer.from(JSON.stringify(progress)),
	});
	await p.waitForFunction(() => document.querySelector(".start-storage + [role=status]")?.textContent?.includes("Loaded"));
	const merged = await p.evaluate(() => {
		const stored = localStorage.getItem("gn:v1:code:workbench") ?? "";
		const undo = localStorage.getItem("gn:v1:code:undo-workbench") ?? "";
		return {
			module2: stored.includes("[section:feedforward]"),
			localText: stored.includes("receiver-only Chapter 2 work"),
			undo: undo.includes("receiver-only Chapter 2 work"),
			imported: !stored.includes("changed after the passing run"),
		};
	});
	console.log("  stored:", JSON.stringify(merged));
	if (!merged.module2 || !merged.localText || !merged.undo || !merged.imported) failures++;

	// Type once after the import. If CodeMirror kept its pre-import document,
	// this write puts the changed Chapter 1 text back into storage.
	await p.locator(".wb-editor .cm-content").click();
	await p.keyboard.press("Control+End");
	await p.keyboard.type("\n# after import");
	await p.waitForFunction(() => localStorage.getItem("gn:v1:code:workbench")?.includes("# after import"));
	const editorFresh = await p.evaluate(() => !localStorage.getItem("gn:v1:code:workbench")?.includes("changed after the passing run"));
	console.log("  editor followed the import:", editorFresh);
	if (!editorFresh) failures++;

	console.log("\na resized dock updates its narrow layout and separator range");
	await p.setViewportSize({ width: 1800, height: 950 });
	await p.evaluate(() => {
		localStorage.setItem("gn:ui:dock", "1");
		localStorage.setItem("gn:ui:dock-w", "800");
	});
	await p.reload({ waitUntil: "networkidle" });
	await p.waitForSelector(".wb:not([hidden])");
	await p.setViewportSize({ width: 1650, height: 950 });
	await p.waitForFunction(() => document.querySelector(".shell")?.dataset.narrow === "1");
	const resized = await p.evaluate(() => {
		const app = document.querySelector(".app");
		const style = getComputedStyle(app);
		return {
			max: document.querySelector(".wb-grip")?.getAttribute("aria-valuemax"),
			content: parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--col-content")),
			expectedContent: app.getBoundingClientRect().width - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
			tabs: getComputedStyle(document.querySelector(".tabs")).display,
			picker: getComputedStyle(document.querySelector(".module-picker")).display,
		};
	});
	console.log("  resized:", JSON.stringify(resized));
	if (resized.max !== "898" || Math.abs(resized.content - resized.expectedContent) > 0.5 || resized.tabs !== "none" || resized.picker === "none") failures++;

	console.log("\nrange controls meet the phone touch floor");
	await p.setViewportSize({ width: 390, height: 844 });
	await p.goto(ORIGIN + "/#m1", { waitUntil: "networkidle" });
	const ranges = await p.evaluate(() => [...document.querySelectorAll("main > div:not([hidden]) input[type=range]")].map((el) => Math.round(el.getBoundingClientRect().height)));
	console.log("  range heights:", JSON.stringify([...new Set(ranges)]));
	if (!ranges.length || ranges.some((height) => height < 44)) failures++;

	if (errors.length) {
		failures++;
		console.log("  ERRORS:", errors.slice(0, 3));
	}
	await b.close();
}

if (failures) {
	console.log(`\n${failures} case(s) failed.`);
	process.exit(1);
}
console.log("\nthe browser paths hold: runs, saved state, responsive dock and touch controls work.");
