// What an exercise looks like in the module page now that the editor lives in
// the panel: the prompt, a button that opens the section, the runnable
// snippets, the test code, and the hints.
//
// Nothing is re-parented. The prompt paragraphs stay direct children of
// .exercise, so the stylesheet's measure rules keep matching them and nine
// exercises do not silently widen from the prose measure to the full column.

import { useEffect, useState } from "react";
import type { Exercise } from "../exercises/types";
import { loadRevealStage, saveRevealStage, subscribeProgress } from "../state/progress";
import { putSection, sectionState } from "../state/workbench";
import { SECTION_BY_ID } from "../state/workbenchDoc";
import { useWorkbench } from "./WorkbenchProvider";

const REVEAL_LABELS = ["Show hint 1", "Show hint 2", "Show the solution"];

export function ExerciseCard({ exercise }: { exercise: Exercise }) {
  const wb = useWorkbench();
  const [reveal, setReveal] = useState(() => loadRevealStage(exercise.id));
  const [tick, setTick] = useState(0);
  useEffect(() => subscribeProgress(() => setTick((t) => t + 1)), []);

  // wb.revision moves when the document does, tick when a pass mark does.
  const state = sectionState(exercise.id);
  void tick;
  void wb.revision;

  const def = SECTION_BY_ID.get(exercise.id);
  const result = wb.resultFor(exercise.id);
  const isOpen = wb.current === exercise.id && wb.dockState !== "closed";

  const revealNext = () => {
    const next = Math.min(reveal + 1, 3);
    setReveal(next);
    saveRevealStage(exercise.id, next);
  };

  const putSolution = () => {
    if (
      !window.confirm(
        "Replace this section of your file with the reference solution? One Undo brings back what is there now.",
      )
    )
      return;
    const outcome = putSection(exercise.id, exercise.solution);
    if (!outcome.ok && outcome.reason) window.alert(outcome.reason);
    wb.reveal(exercise.id);
  };

  const summary =
    state === "passing"
      ? "passing"
      : state === "stale"
        ? "passed, changed since"
        : result
          ? `${result.tests.filter((t) => t.passed).length} of ${result.tests.length} tests passing`
          : state === "missing"
            ? "not started"
            : "written, not passing yet";

  return (
    <section className="exercise">
      <h4 className="exercise-title">
        <span>Exercise: {exercise.title}</span>
        {state === "passing" && (
          <span className="badge-done" title="All tests passed">
            passed
          </span>
        )}
      </h4>

      {exercise.prompt.map((part, i) =>
        typeof part === "string" ? (
          <p key={i}>{part}</p>
        ) : (
          <PlaySnippet key={i} code={part.code} onSend={() => wb.sendToScratch(part.code)} />
        ),
      )}

      <p className="exercise-launcher">
        <button className="exercise-launcher-open" onClick={() => wb.reveal(exercise.id)}>
          {isOpen ? "Show this section in the workbench" : "Open this section in the workbench"}
        </button>
        <span className="exercise-launcher-state">
          {def ? `${def.label}: ${summary}` : summary}
        </span>
      </p>

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
            <button className="button-secondary" onClick={putSolution}>
              Put this solution in my file
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

function PlaySnippet({ code, onSend }: { code: string; onSend: () => void }) {
  const [copied, setCopied] = useState(false);
  const [sent, setSent] = useState(false);
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
        <button
          className="button-secondary"
          onClick={() => {
            onSend();
            setSent(true);
            window.setTimeout(() => setSent(false), 1500);
          }}
        >
          {sent ? "Sent" : "Send to the scratch pad"}
        </button>
      </div>
    </div>
  );
}
