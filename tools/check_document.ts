// The document format's own invariants, run against the real code.
//
// src/state/workbenchDoc.ts is what the app parses and splices the learner's
// file with, and check_exercises.py can only test the documents this repo
// assembles, not the ones the editor produces. The difference is where the
// bug was: replacing a section put its text back with the trailing whitespace
// trimmed, so the next section's marker landed at the end of the previous
// section's last line, where the anchored regex cannot see it, and that
// section silently vanished into its neighbour. Nothing in the app would have
// reported it: the parse succeeds, it just finds one section fewer.
//
// Run with `npm run check:doc`, which compiles this and the module it tests to
// CommonJS in .bench/ and runs it in Node. That works because workbenchDoc.ts
// imports nothing a bundler has to resolve.

// Declared rather than pulled in with @types/node: this repo has no Node
// types and the other benches do the same, so the checker adds no dependency.
declare const console: { log(...args: unknown[]): void };
declare const process: { exit(code: number): never };

import {
  JOIN,
  SECTIONS,
  SECTION_ORDER,
  assemble,
  closure,
  givensFor,
  hashBody,
  parseDocument,
  projection,
  upsertSection,
} from "../src/state/workbenchDoc";

const PRELUDE = '"""My library."""\n\nimport numpy as np';

let failures = 0;

function ok(condition: boolean, what: string, detail = ""): void {
  if (condition) {
    console.log(`  OK   ${what}`);
  } else {
    failures++;
    console.log(`  FAIL ${what}${detail ? `\n         ${detail}` : ""}`);
  }
}

function body(id: string): string {
  return `def ${id.replace(/-/g, "_")}_one():\n    return "${id}"\n\n\ndef ${id.replace(/-/g, "_")}_two():\n    return 2`;
}

function fullDocument(): string {
  return assemble(new Map(SECTION_ORDER.map((id) => [id, body(id)])), PRELUDE);
}

console.log("--- the assembled document ---");
{
  const doc = parseDocument(fullDocument());
  ok(doc.sections.length === SECTIONS.length,
     `every section is found (${doc.sections.length} of ${SECTIONS.length})`);
  ok(doc.sections.map((s) => s.id).join(",") === SECTION_ORDER.join(","),
     "in course order");
  ok(doc.problems.length === 0, "with no problems reported",
     doc.problems.map((p) => p.message).join(" | "));
  ok(doc.sections.every((s) => s.body === body(s.id)),
     "and every body comes back byte for byte");
}

console.log("\n--- replacing a section keeps every other one ---");
for (const id of SECTION_ORDER) {
  const spliced = upsertSection(fullDocument(), id, "def replaced():\n    return 1");
  const doc = parseDocument(spliced.text);
  const found = doc.sections.map((s) => s.id).join(",");
  ok(found === SECTION_ORDER.join(",") && doc.problems.length === 0,
     `replacing ${id} leaves all ${SECTIONS.length} sections intact`,
     found === SECTION_ORDER.join(",") ? "" : `found ${found}`);
  ok(doc.byId.get(id)?.body === "def replaced():\n    return 1",
     `  and ${id} holds exactly the new text`);
}

console.log("\n--- repeated splices do not drift ---");
{
  let text = fullDocument();
  for (let round = 0; round < 5; round++) {
    for (const id of SECTION_ORDER) text = upsertSection(text, id, body(id)).text;
  }
  ok(text === fullDocument(),
     "five rounds of replacing every section with its own text is a no-op",
     `${text.length} chars against ${fullDocument().length}`);
}

console.log("\n--- inserting a section that is not there yet ---");
for (const id of SECTION_ORDER) {
  const others = SECTION_ORDER.filter((x) => x !== id);
  const partial = assemble(new Map(others.map((x) => [x, body(x)])), PRELUDE);
  const doc = parseDocument(upsertSection(partial, id, body(id)).text);
  ok(doc.sections.map((s) => s.id).join(",") === SECTION_ORDER.join(","),
     `${id} is inserted in its own place`,
     doc.sections.map((s) => s.id).join(","));
}

console.log("\n--- inserting into an empty file ---");
{
  let text = PRELUDE + "\n";
  // The order a learner meets them in, which is not the order they sit in.
  for (const id of SECTION_ORDER.filter((x) => x !== "given-cost" && x !== "given-batch")) {
    for (const given of givensFor(id)) text = upsertSection(text, given, body(given)).text;
    text = upsertSection(text, id, body(id)).text;
  }
  const doc = parseDocument(text);
  ok(doc.sections.map((s) => s.id).join(",") === SECTION_ORDER.join(","),
     "seeding one section at a time builds the file in course order",
     doc.sections.map((s) => s.id).join(","));
  ok(doc.problems.length === 0, "with no problems reported",
     doc.problems.map((p) => p.message).join(" | "));
}

console.log("\n--- a mangled file degrades rather than breaks ---");
{
  const full = fullDocument();
  const cases: [string, string, (t: string) => boolean][] = [
    [
      "a deleted marker merges its section into the one above",
      full.replace(/^#.*\[section:backprop\].*$\n/m, ""),
      (t) => {
        const d = parseDocument(t);
        return !d.byId.has("backprop") && d.sections.length === SECTIONS.length - 1;
      },
    ],
    [
      "an unknown id is kept verbatim and reported",
      full.replace("[section:backprop]", "[section:backprap]"),
      (t) => parseDocument(t).problems.some((p) => p.kind === "unknown-section"),
    ],
    [
      "a duplicated section is reported, first one wins",
      full + JOIN + SECTIONS[0].marker + "\n\ndef again():\n    return 1",
      (t) => {
        const d = parseDocument(t);
        return (
          d.problems.some((p) => p.kind === "duplicate-section") &&
          d.byId.get(SECTIONS[0].id)?.body === body(SECTIONS[0].id)
        );
      },
    ],
    [
      "a lost numpy import is reported",
      full.replace("import numpy as np", "# import numpy as np"),
      (t) => parseDocument(t).problems.some((p) => p.kind === "missing-prelude"),
    ],
    [
      "out of order is a note, not a break",
      assemble(new Map([...SECTION_ORDER].reverse().map((id) => [id, body(id)])), PRELUDE),
      (t) => {
        const d = parseDocument(t);
        return d.sections.length === SECTIONS.length && d.problems.every((p) => p.kind === "out-of-order");
      },
    ],
    ["an empty file parses", "", (t) => parseDocument(t).sections.length === 0],
    ["a file of only markers parses", SECTIONS.map((s) => s.marker).join("\n"),
     (t) => parseDocument(t).sections.length === SECTIONS.length],
  ];
  for (const [what, text, predicate] of cases) {
    let passed = false;
    try {
      passed = predicate(text);
    } catch (err) {
      ok(false, what, `threw: ${(err as Error).message}`);
      continue;
    }
    ok(passed, what);
  }
}

console.log("\n--- projections ---");
{
  const full = fullDocument();
  for (const id of SECTION_ORDER) {
    if (SECTIONS.find((s) => s.id === id)?.kind === "given") continue;
    const text = projection(full, id);
    const found = parseDocument(text).sections.map((s) => s.id);
    const rank = Math.max(
      SECTION_ORDER.indexOf(id),
      ...givensFor(id).map((g) => SECTION_ORDER.indexOf(g)),
    );
    ok(found.join(",") === SECTION_ORDER.slice(0, rank + 1).join(","),
       `the projection through ${id} stops there`, found.join(","));
    ok(closure(id).size > 0 && [...closure(id)].every((need) => found.includes(need)),
       `  and carries everything ${id} calls into`);
  }
}

console.log("\n--- the body hash ---");
{
  ok(hashBody("abc") === hashBody("abc  \n"), "trailing whitespace does not change it");
  ok(hashBody("abc") !== hashBody("abd"), "one character does");
}

console.log();
if (failures) {
  console.log(`${failures} failure(s) in the document format.`);
  process.exit(1);
}
console.log("the document format holds: splices keep every section, repeated splices do not drift, and a mangled file degrades rather than breaks.");
