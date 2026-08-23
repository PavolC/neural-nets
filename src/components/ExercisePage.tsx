import { useRef, useState } from "react";
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

export function ExercisePage({ exercise }: { exercise: Exercise }) {
  const editorRef = useRef<CodeEditorHandle | null>(null);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<TestRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reveal, setReveal] = useState(() => loadRevealStage(exercise.id));
  const [completed, setCompleted] = useState(() => loadCompleted(exercise.id));

  const initialDoc = loadCode(exercise.id) ?? exercise.skeleton;

  const runTests = () => {
    if (!editorRef.current) return;
    setRunning(true);
    setResult(null);
    setError(null);
    setStatus("Starting...");
    sendRequest(
      {
        type: "runTests",
        learnerCode: editorRef.current.getDoc(),
        testsCode: exercise.tests,
      },
      (msg: WorkerResponse) => {
        switch (msg.type) {
          case "status":
            setStatus(msg.text);
            break;
          case "testsDone":
            setResult(msg.result);
            setRunning(false);
            setStatus("");
            if (msg.result.passed) {
              setCompleted(true);
              saveCompleted(exercise.id);
            }
            break;
          case "error":
            setError(msg.message);
            setRunning(false);
            setStatus("");
            break;
          default:
            break;
        }
      },
    );
  };

  const cancel = () => {
    terminateWorker();
  };

  const resetToSkeleton = () => {
    if (!window.confirm("Replace your code with the original skeleton? Your edits and progress on this exercise will be lost.")) return;
    resetExercise(exercise.id);
    editorRef.current?.setDoc(exercise.skeleton);
    setResult(null);
    setError(null);
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
      {exercise.prompt.map((para, i) => (
        <p key={i}>{para}</p>
      ))}

      <CodeEditor
        initialDoc={initialDoc}
        onChange={(doc) => saveCode(exercise.id, doc)}
        handleRef={editorRef}
      />

      <div className="exercise-controls">
        <button onClick={runTests} disabled={running}>
          {running ? "Running..." : "Run tests"}
        </button>
        {running && (
          <button className="button-secondary" onClick={cancel}>
            Stop
          </button>
        )}
        <button className="button-secondary" onClick={resetToSkeleton} disabled={running}>
          Reset to skeleton
        </button>
      </div>
      {status && <p className="demo-status">{status}</p>}
      {error && <p className="demo-status demo-status-error">Something went wrong: {error}</p>}

      {result && <TestResults result={result} />}

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

function TestResults({ result }: { result: TestRunResult }) {
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
  return (
    <div className="test-results">
      <p className={result.passed ? "test-summary test-summary-pass" : "test-summary"}>
        {result.passed
          ? "All tests passed."
          : `${passedCount} of ${result.tests.length} tests passed.`}
      </p>
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
