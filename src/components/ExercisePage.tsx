import { lazy, Suspense, useEffect, useRef, useState } from "react";
import type { Exercise } from "../exercises/types";
import type { TestRunResult, WorkerResponse } from "../runtime/messages";
import { sendRequest, terminateWorker } from "../runtime/workerClient";
import {
  loadCode,
  loadCompleted,
  loadRevealStage,
  resetExercise,
  saveCode,
  saveCompleted,
  saveRevealStage,
} from "../state/progress";
import type { CodeEditorHandle } from "./CodeEditor";
import { useInViewOnce } from "./useInViewOnce";

// CodeMirror is the largest dependency in the app, 147 KB gzipped against the
// 155 KB of everything else. It is needed only at an exercise, which sits
// thousands of pixels into a module, so it loads in its own chunk and that chunk
// is not requested until the reader is within 600px of the editor. Splitting it
// out alone was not enough: Module 1 renders on first paint, exercise included,
// so the lazy import fired immediately and the download was never actually
// deferred. The in-view gate below is what defers it.
const CodeEditor = lazy(() =>
  import("./CodeEditor").then((m) => ({ default: m.CodeEditor })),
);

const REVEAL_LABELS = ["Show hint 1", "Show hint 2", "Show the solution"];

// Runs the editor contents alone (no tests) via the harness, so the learner
// can print() and experiment. Prints stream back as stdout log messages.
const SCRATCH_SNIPPET = `
import json
run_scratch(json.loads(_args_json)["code"])
`;

interface ScratchError {
  message: string;
  line: number | null;
}

/** Which of the two run buttons is in flight. Sharing one boolean meant
 * pressing "Run my code" relabelled "Run tests" to "Running...". */
type RunKind = "tests" | "scratch";

export function ExercisePage({ exercise }: { exercise: Exercise }) {
  const editorRef = useRef<CodeEditorHandle | null>(null);
  const [busy, setBusy] = useState<RunKind | null>(null);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<TestRunResult | null>(null);
  // Results survive the next run instead of being cleared: unmounting them
  // collapsed the page by well over a thousand pixels and put it back
  // afterwards. They are marked stale rather than removed.
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState(false);
  const [output, setOutput] = useState<string[]>([]);
  const [trimmed, setTrimmed] = useState(false);
  const [scratchError, setScratchError] = useState<ScratchError | null>(null);
  const [ranOnce, setRanOnce] = useState(false);
  const [reveal, setReveal] = useState(() => loadRevealStage(exercise.id));
  const [completed, setCompleted] = useState(() => loadCompleted(exercise.id));
  const [fullscreen, setFullscreen] = useState(false);
  const [editorReady, setEditorReady] = useState(false);
  const workbenchRef = useRef<HTMLDivElement>(null);
  const editorNeeded = useInViewOnce(workbenchRef);
  const running = busy !== null;
  // Nothing to run until the editor chunk has landed and handed over its handle.
  const canRun = editorReady && !running;

  // Fullscreen: Escape exits, and the page behind must not scroll.
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    // A hash change unmounts nothing but does hide this module, and the
    // overlay would go with it while body overflow stayed hidden, leaving a
    // page that cannot be scrolled until a reload.
    const onHashChange = () => setFullscreen(false);
    window.addEventListener("keydown", onKey);
    window.addEventListener("hashchange", onHashChange);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("hashchange", onHashChange);
      document.body.style.overflow = "";
    };
  }, [fullscreen]);

  const outputRef = useRef<HTMLPreElement>(null);
  // Follow the tail while a run streams, unless the reader has scrolled up to
  // look at something.
  useEffect(() => {
    const el = outputRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [output]);

  const initialDoc = loadCode(exercise.id) ?? exercise.skeleton;

  const beginRun = (kind: RunKind) => {
    setBusy(kind);
    setRanOnce(true);
    setError(null);
    setCancelled(false);
    setScratchError(null);
    setOutput([]);
    setTrimmed(false);
    setStatus("Starting...");
  };

  const collectCommon = (msg: WorkerResponse): boolean => {
    switch (msg.type) {
      case "status":
        setStatus(msg.text);
        return true;
      case "log":
        if (msg.source === "stdout")
          setOutput((prev) => {
            if (prev.length >= 200) setTrimmed(true);
            return [...prev.slice(-199), msg.text];
          });
        return true;
      case "cancelled":
        setCancelled(true);
        setBusy(null);
        setStatus("");
        return true;
      case "error":
        setError(msg.message);
        setBusy(null);
        setStatus("");
        return true;
      default:
        return false;
    }
  };

  const runTests = () => {
    if (!editorRef.current) return;
    beginRun("tests");
    setStale(true);
    sendRequest(
      {
        type: "runTests",
        learnerCode: editorRef.current.getDoc(),
        testsCode: exercise.tests,
      },
      (msg: WorkerResponse) => {
        if (collectCommon(msg)) return;
        if (msg.type === "testsDone") {
          setResult(msg.result);
          setStale(false);
          setBusy(null);
          setStatus("");
          if (msg.result.passed) {
            setCompleted(true);
            saveCompleted(exercise.id);
          }
        }
      },
    );
  };

  const runScratch = () => {
    if (!editorRef.current) return;
    beginRun("scratch");
    // The verdict on screen was reached by code that may since have changed.
    if (result) setStale(true);
    sendRequest(
      {
        type: "runPython",
        code: SCRATCH_SNIPPET,
        args: { code: editorRef.current.getDoc() },
      },
      (msg: WorkerResponse) => {
        if (collectCommon(msg)) return;
        if (msg.type === "pythonDone") {
          const r = msg.result as { error: ScratchError | null };
          setScratchError(r.error);
          setBusy(null);
          setStatus("");
        }
      },
    );
  };

  const resetToSkeleton = () => {
    if (
      !window.confirm(
        "Replace your code with the original skeleton? Your edits, your revealed hints and this exercise's passed mark will be lost. Other modules are unaffected.",
      )
    )
      return;
    resetExercise(exercise.id);
    editorRef.current?.setDoc(exercise.skeleton);
    setResult(null);
    setStale(false);
    setError(null);
    setCancelled(false);
    setScratchError(null);
    setOutput([]);
    setRanOnce(false);
    setReveal(0);
    setCompleted(false);
  };

  const revealNext = () => {
    const next = Math.min(reveal + 1, 3);
    setReveal(next);
    saveRevealStage(exercise.id, next);
  };

  const copySolution = () => {
    if (!window.confirm("Replace the code in the editor with the reference solution?")) return;
    editorRef.current?.setDoc(exercise.solution);
    saveCode(exercise.id, exercise.solution);
  };

  // One announcement per state change, from a region that is always mounted:
  // a live region inserted in the same tick as its text is announced
  // unreliably. Without this, a screen reader hears nothing at all between
  // pressing Run tests and finding the results by hand, every single run.
  const liveMessage = error
    ? `Run failed. ${error}`
    : cancelled
      ? "Stopped."
      : busy
        ? status || "Running..."
        : result && !stale
          ? result.setup_error
            ? "Your code did not run. The reason is above the results."
            : result.passed
              ? "All tests passed."
              : `${result.tests.filter((t) => t.passed).length} of ${result.tests.length} tests passed.`
          : "";

  return (
    <section className="exercise">
      <h4 className="exercise-title">
        <span>Exercise: {exercise.title}</span>
        {completed && (
          <span className="badge-done" title="All tests passed">
            passed
          </span>
        )}
      </h4>
      {exercise.prompt.map((part, i) =>
        typeof part === "string" ? (
          <p key={i}>{part}</p>
        ) : (
          <PlaySnippet
            key={i}
            code={part.code}
            onAppend={() => {
              const editor = editorRef.current;
              if (!editor) return;
              const doc = editor.getDoc();
              editor.setDoc(`${doc.replace(/\s+$/, "")}\n\n\n${part.code}\n`);
              saveCode(exercise.id, editor.getDoc());
            }}
          />
        ),
      )}

      <div
        className={fullscreen ? "workbench workbench-fullscreen" : "workbench"}
        aria-busy={running}
        ref={workbenchRef}
      >
        {/* The placeholder carries the editor's own height, so the page geometry
            is the same before and after it arrives. */}
        {editorNeeded ? (
          <Suspense fallback={<EditorPlaceholder />}>
            <CodeEditor
              initialDoc={initialDoc}
              onChange={(doc) => saveCode(exercise.id, doc)}
              handleRef={editorRef}
              onReady={setEditorReady}
            />
          </Suspense>
        ) : (
          <EditorPlaceholder />
        )}

        <p className="exercise-tip">
          Two ways to run. "Run my code" just executes what is in the editor, so you
          can call your functions, print(...) values, and experiment. "Run tests"
          checks your work. Everything you print appears in the Output panel below,
          as it happens. The editor is resizable (drag its bottom edge); Tab indents,
          and Escape then Tab moves keyboard focus out. Your code is saved in this
          browser only.
        </p>

        <div className="exercise-controls">
          <button onClick={runTests} disabled={!canRun}>
            {busy === "tests" ? "Running..." : "Run tests"}
          </button>
          <button className="button-secondary" onClick={runScratch} disabled={!canRun}>
            {busy === "scratch" ? "Running..." : "Run my code"}
          </button>
          {running && (
            <button className="button-secondary" onClick={terminateWorker}>
              Stop
            </button>
          )}
          <button className="button-secondary" onClick={resetToSkeleton} disabled={!canRun}>
            Reset to skeleton
          </button>
          <button
            className="button-secondary workbench-expand"
            onClick={() => setFullscreen((f) => !f)}
          >
            {fullscreen ? "Exit fullscreen (Esc)" : "Fullscreen"}
          </button>
        </div>
        <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
          {liveMessage}
        </p>
        {status && <p className="demo-status">{status}</p>}
        {cancelled && <p className="demo-status">Stopped. Press Run tests to try again.</p>}
        {error && <p className="demo-status demo-status-error">Something went wrong: {error}</p>}

        {/* Mounted from the first run onward, including while a run is in
            flight: prints stream in during the run, which is when the reader
            wants them, and the panel no longer appears and disappears
            underneath the page on every click. */}
        {ranOnce && (
          <div className="output-panel">
            <h5>Output</h5>
            <pre ref={outputRef}>
              {trimmed && "(earlier output trimmed to the last 200 lines)\n"}
              {output.length
                ? output.join("\n")
                : running
                  ? "(waiting for output...)"
                  : "(your code printed nothing; add print(...) anywhere to inspect values)"}
            </pre>
            {scratchError && (
              <p className="demo-status demo-status-error">
                Your code stopped with {scratchError.message}
                {scratchError.line !== null && ` (line ${scratchError.line})`}.
              </p>
            )}
          </div>
        )}

        {result && (
          <TestResults result={result} flagship={exercise.flagship} stale={stale} />
        )}
      </div>

      <details className="demo-log tests-viewer">
        <summary>See exactly what the tests check (the test code)</summary>
        <pre>{exercise.tests}</pre>
      </details>

      <div className="hints">
        {reveal > 0 && (
          <div className="hint">
            <h5>Hint 1</h5>
            <p>{exercise.hints[0]}</p>
          </div>
        )}
        {reveal > 1 && (
          <div className="hint">
            <h5>Hint 2</h5>
            <pre className="hint-pre">{exercise.hints[1]}</pre>
          </div>
        )}
        {reveal > 2 && (
          <div className="hint">
            <h5>Reference solution</h5>
            <pre className="hint-pre">{exercise.solution}</pre>
            <button className="button-secondary" onClick={copySolution} disabled={!canRun}>
              Copy solution into editor
            </button>
          </div>
        )}
        {reveal < 3 && (
          <button className="button-secondary button-hint" onClick={revealNext}>
            {REVEAL_LABELS[reveal]}
          </button>
        )}
      </div>
    </section>
  );
}

function EditorPlaceholder() {
  return <div className="code-editor code-editor-loading">Loading the editor...</div>;
}

function PlaySnippet({ code, onAppend }: { code: string; onAppend: () => void }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(code).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className="play-snippet">
      <pre>{code}</pre>
      <div className="play-snippet-buttons">
        <button className="button-secondary" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
        <button className="button-secondary" onClick={onAppend}>
          Append to my code
        </button>
      </div>
    </div>
  );
}

/** Failures that stopped for the same reason, collected under one card. The
 * untouched skeleton fails every test with the identical NotImplementedError,
 * which used to render as six full-size copies of one sentence. */
function groupFailures(tests: TestRunResult["tests"]) {
  const groups: { message: string; titles: string[] }[] = [];
  for (const t of tests) {
    if (t.passed) continue;
    const existing = groups.find((g) => g.message === t.message);
    if (existing) existing.titles.push(t.title);
    else groups.push({ message: t.message, titles: [t.title] });
  }
  return groups;
}

function TestResults({
  result,
  flagship,
  stale,
}: {
  result: TestRunResult;
  flagship?: { test: string; note: string };
  stale: boolean;
}) {
  if (result.setup_error) {
    const { message, line } = result.setup_error;
    return (
      <div className={stale ? "test-results test-results-stale" : "test-results"}>
        <div className="test-result test-fail">
          <span className="test-mark" aria-hidden="true">
            ✗
          </span>
          <div>
            <strong>Your code did not run</strong>
            <p>
              {message}
              {line !== null && ` (line ${line})`}. Fix this before the tests can start.
            </p>
          </div>
        </div>
      </div>
    );
  }
  const passedCount = result.tests.filter((t) => t.passed).length;
  const failures = groupFailures(result.tests);
  const nothingWritten =
    failures.length > 0 &&
    passedCount === 0 &&
    failures.every((g) => g.message.includes("NotImplementedError"));
  const flagshipPassed =
    flagship && result.tests.some((t) => t.name === flagship.test && t.passed);
  return (
    <div className={stale ? "test-results test-results-stale" : "test-results"}>
      {stale && (
        <p className="test-stale-note">
          From an earlier run. Press Run tests to check the code as it stands now.
        </p>
      )}
      <p className={result.passed ? "test-summary test-summary-pass" : "test-summary"}>
        {result.passed
          ? "All tests passed."
          : `${passedCount} of ${result.tests.length} tests passed.`}
      </p>
      {nothingWritten && (
        <p className="test-orient">
          Nothing is implemented yet, so every test stopped at the first function it
          called. Work down the list below: the earlier functions are the ones the
          later tests need.
        </p>
      )}
      {flagshipPassed && <p className="flagship-banner">{flagship.note}</p>}
      {failures.map((g) => (
        <div key={g.message} className="test-result test-fail">
          <span className="test-mark" aria-hidden="true">
            ✗
          </span>
          <div>
            <strong>
              <span className="sr-only">Failed: </span>
              {g.titles[0]}
            </strong>
            {g.titles.length > 1 && (
              <p className="test-also">
                and {g.titles.length - 1} more for the same reason: {g.titles.slice(1).join("; ")}
              </p>
            )}
            <p>{g.message}</p>
          </div>
        </div>
      ))}
      {result.tests
        .filter((t) => t.passed)
        .map((t) => (
          <div key={t.name} className="test-result test-pass">
            <span className="test-mark" aria-hidden="true">
              ✓
            </span>
            <div>
              <strong>
                <span className="sr-only">Passed: </span>
                {t.title}
              </strong>
            </div>
          </div>
        ))}
    </div>
  );
}
