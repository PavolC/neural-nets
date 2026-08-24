import { useEffect, useRef, useState } from "react";
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
import { CodeEditor, type CodeEditorHandle } from "./CodeEditor";

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

export function ExercisePage({ exercise }: { exercise: Exercise }) {
  const editorRef = useRef<CodeEditorHandle | null>(null);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<TestRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<string[]>([]);
  const [scratchError, setScratchError] = useState<ScratchError | null>(null);
  const [ranOnce, setRanOnce] = useState(false);
  const [reveal, setReveal] = useState(() => loadRevealStage(exercise.id));
  const [completed, setCompleted] = useState(() => loadCompleted(exercise.id));
  const [fullscreen, setFullscreen] = useState(false);

  // Fullscreen: Escape exits, and the page behind must not scroll.
  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [fullscreen]);

  const initialDoc = loadCode(exercise.id) ?? exercise.skeleton;

  const beginRun = () => {
    setRunning(true);
    setRanOnce(true);
    setError(null);
    setScratchError(null);
    setOutput([]);
    setStatus("Starting...");
  };

  const collectCommon = (msg: WorkerResponse): boolean => {
    switch (msg.type) {
      case "status":
        setStatus(msg.text);
        return true;
      case "log":
        if (msg.source === "stdout") setOutput((prev) => [...prev.slice(-199), msg.text]);
        return true;
      case "error":
        setError(msg.message);
        setRunning(false);
        setStatus("");
        return true;
      default:
        return false;
    }
  };

  const runTests = () => {
    if (!editorRef.current) return;
    beginRun();
    setResult(null);
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
          setRunning(false);
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
    beginRun();
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
          setRunning(false);
          setStatus("");
        }
      },
    );
  };

  const resetToSkeleton = () => {
    if (!window.confirm("Replace your code with the original skeleton? Your edits and progress on this exercise will be lost.")) return;
    resetExercise(exercise.id);
    editorRef.current?.setDoc(exercise.skeleton);
    setResult(null);
    setError(null);
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

  return (
    <section className="exercise">
      <h2>
        Exercise: {exercise.title}
        {completed && <span className="badge-done" title="All tests passed">passed</span>}
      </h2>
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

      <div className={fullscreen ? "workbench workbench-fullscreen" : "workbench"}>
        <CodeEditor
          initialDoc={initialDoc}
          onChange={(doc) => saveCode(exercise.id, doc)}
          handleRef={editorRef}
        />

        <p className="exercise-tip">
          Two ways to run. "Run my code" just executes what is in the editor, so you
          can call your functions, print(...) values, and experiment. "Run tests"
          checks your work. Everything you print appears in the Output panel below.
          The editor is resizable (drag its bottom edge); Tab indents, and Escape
          then Tab moves keyboard focus out.
        </p>

        <div className="exercise-controls">
          <button onClick={runTests} disabled={running}>
            {running ? "Running..." : "Run tests"}
          </button>
          <button className="button-secondary" onClick={runScratch} disabled={running}>
            Run my code
          </button>
          {running && (
            <button className="button-secondary" onClick={terminateWorker}>
              Stop
            </button>
          )}
          <button className="button-secondary" onClick={resetToSkeleton} disabled={running}>
            Reset to skeleton
          </button>
          <button
            className="button-secondary workbench-expand"
            onClick={() => setFullscreen((f) => !f)}
          >
            {fullscreen ? "Exit fullscreen (Esc)" : "Fullscreen"}
          </button>
        </div>
        {status && <p className="demo-status">{status}</p>}
        {error && <p className="demo-status demo-status-error">Something went wrong: {error}</p>}

        {ranOnce && !running && (
          <div className="output-panel">
            <h3>Output</h3>
            <pre>
              {output.length
                ? output.join("\n")
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

        {result && <TestResults result={result} flagship={exercise.flagship} />}
      </div>

      <details className="demo-log tests-viewer">
        <summary>See exactly what the tests check (the test code)</summary>
        <pre>{exercise.tests}</pre>
      </details>

      <div className="hints">
        {reveal > 0 && (
          <div className="hint">
            <h3>Hint 1</h3>
            <p>{exercise.hints[0]}</p>
          </div>
        )}
        {reveal > 1 && (
          <div className="hint">
            <h3>Hint 2</h3>
            <pre className="hint-pre">{exercise.hints[1]}</pre>
          </div>
        )}
        {reveal > 2 && (
          <div className="hint">
            <h3>Reference solution</h3>
            <pre className="hint-pre">{exercise.solution}</pre>
            <button className="button-secondary" onClick={copySolution} disabled={running}>
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

function TestResults({
  result,
  flagship,
}: {
  result: TestRunResult;
  flagship?: { test: string; note: string };
}) {
  if (result.setup_error) {
    const { message, line } = result.setup_error;
    return (
      <div className="test-results">
        <div className="test-result test-fail">
          <span className="test-mark">✗</span>
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
  const flagshipPassed =
    flagship && result.tests.some((t) => t.name === flagship.test && t.passed);
  return (
    <div className="test-results">
      <p className={result.passed ? "test-summary test-summary-pass" : "test-summary"}>
        {result.passed
          ? "All tests passed."
          : `${passedCount} of ${result.tests.length} tests passed.`}
      </p>
      {flagshipPassed && <p className="flagship-banner">{flagship.note}</p>}
      {result.tests.map((t) => (
        <div key={t.name} className={`test-result ${t.passed ? "test-pass" : "test-fail"}`}>
          <span className="test-mark">{t.passed ? "✓" : "✗"}</span>
          <div>
            <strong>{t.title}</strong>
            {!t.passed && <p>{t.message}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}
