// The panel: the learner's whole file, the run controls, and what a run says.
//
// Docked to the right of the reading column above 1360px, so the prose stays
// on screen while they type, and a full-screen sheet below that. Always
// mounted, hidden with CSS, so a tab switch never costs the editor's state
// and one editor serves the whole course.

import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadExercise } from "../exercises/loaders";
import type { ScratchRunResult, TestRunResult } from "../runtime/messages";
import nielsenNotice from "../python/nielsen_notice.txt?raw";
import { resetExercise } from "../state/progress";
import { loadUiNumber, saveUi } from "../state/ui";
import {
  canUndo,
  currentDoc,
  downloadText,
  editedGivens,
  loadDocument,
  loadScratch,
  sectionState,
  subscribeDocument,
  undoLastSplice,
  type SectionState,
} from "../state/workbench";
import { SECTIONS, SECTION_BY_ID, type SectionDef } from "../state/workbenchDoc";
import type { CodeEditorHandle } from "./CodeEditor";
import { TestResults } from "./TestResults";
import type { DockState, RunKind, UpstreamBlame } from "./WorkbenchProvider";

// CodeMirror is the largest dependency in the app, 147 KB gzipped against the
// 155 KB of everything else, and it is needed only once the panel is opened.
// It used to be deferred by an in-view gate on one exercise; a panel has no
// in-view moment, so the first open is the gate instead.
const CodeEditor = lazy(() => import("./CodeEditor").then((m) => ({ default: m.CodeEditor })));

const SPLIT_KEY = "split";

interface Props {
  dockState: DockState;
  current: string | null;
  revision: number;
  busy: RunKind | null;
  status: string;
  output: string[];
  trimmed: boolean;
  error: string | null;
  cancelled: boolean;
  ranOnce: boolean;
  scratchError: ScratchRunResult["error"];
  result: TestRunResult | undefined;
  stale: boolean;
  lent: string[] | null;
  blame: UpstreamBlame | null;
  blaming: boolean;
  editorReady: boolean;
  revealRequest: { id: string; at: number } | null;
  onEditorReady(ready: boolean): void;
  onDocumentChange(text: string): void;
  onScratchChange(text: string): void;
  onCaret(pos: number): void;
  onSelectSection(id: string): void;
  onRunTests(): void;
  onRunScratch(): void;
  onStop(): void;
  onClose(): void;
  onChanged(): void;
}

const STATE_LABEL: Record<SectionState, string> = {
  missing: "not in your file yet",
  written: "written, not passing yet",
  passing: "passing",
  stale: "passed, changed since",
};

export function Workbench(props: Props) {
  const {
    dockState,
    current,
    revision,
    busy,
    status,
    output,
    trimmed,
    error,
    cancelled,
    ranOnce,
    scratchError,
    result,
    stale,
    lent,
    blame,
    blaming,
    editorReady,
    revealRequest,
  } = props;

  const editorRef = useRef<CodeEditorHandle | null>(null);
  const scratchRef = useRef<CodeEditorHandle | null>(null);
  const panelRef = useRef<HTMLElement>(null);
  const outputRef = useRef<HTMLPreElement>(null);
  const [scratchOpen, setScratchOpen] = useState(false);
  const [splitPercent, setSplitPercent] = useState(() => loadUiNumber(SPLIT_KEY, 60));
  const [spliceNote, setSpliceNote] = useState<string | null>(null);
  const [flagship, setFlagship] = useState<{ test: string; note: string } | undefined>();
  // The editor chunk is fetched on the first open and never unmounted after.
  const [everOpened, setEverOpened] = useState(dockState !== "closed");

  const open = dockState !== "closed";
  useEffect(() => {
    if (open) setEverOpened(true);
  }, [open]);

  const doc = useMemo(() => currentDoc(), [revision]);
  const def = current ? SECTION_BY_ID.get(current) : undefined;

  // Follow an import or a splice that happened somewhere else. Without this
  // the always-mounted editor keeps the text it had and the next keystroke
  // writes it back over what was just loaded.
  useEffect(
    () =>
      subscribeDocument((source) => {
        if (source === "editor") return;
        editorRef.current?.setDoc(loadDocument());
        props.onChanged();
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  // Scroll to whichever section was asked for, once the editor exists.
  useEffect(() => {
    if (!revealRequest || !editorReady) return;
    const section = currentDoc().byId.get(revealRequest.id);
    if (!section) return;
    editorRef.current?.setDoc(loadDocument());
    editorRef.current?.reveal(section.from, section.to);
  }, [revealRequest, editorReady]);

  // Mark the current section whenever the caret moves it.
  useEffect(() => {
    if (!editorReady || !current) return;
    const section = currentDoc().byId.get(current);
    if (section) editorRef.current?.markSection(section.from, section.to);
  }, [current, editorReady, revision]);

  // A CodeMirror laid out inside a hidden box has stale geometry.
  useEffect(() => {
    if (open) editorRef.current?.remeasure();
  }, [open, splitPercent]);

  // The sheet is modal, so focus goes into it and comes back out again.
  const returnFocus = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (open) {
      returnFocus.current = document.activeElement as HTMLElement | null;
      // Not the editor: landing a keyboard reader inside a text area with no
      // announcement is worse than landing them on the panel's own heading.
      panelRef.current?.querySelector<HTMLElement>(".wb-title")?.focus();
    } else if (returnFocus.current?.isConnected) {
      returnFocus.current.focus();
      returnFocus.current = null;
    }
  }, [open]);

  // Follow the tail while a run streams, unless the reader has scrolled up.
  useEffect(() => {
    const el = outputRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [output]);

  // The flagship banner belongs to one exercise, so it arrives with it.
  useEffect(() => {
    if (!current) return;
    let live = true;
    loadExercise(current)?.then((ex) => {
      if (live) setFlagship(ex.flagship);
    });
    return () => {
      live = false;
    };
  }, [current]);

  const running = busy !== null;
  const canRun = editorReady && !running;

  const download = useCallback(() => {
    const blob = new Blob([downloadText(nielsenNotice)], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "nn.py";
    link.click();
    URL.revokeObjectURL(url);
  }, []);

  const resetCurrent = useCallback(() => {
    if (!def) return;
    if (
      !window.confirm(
        `Put the ${def.label} section back to how it started? Your text in that section, its hints and its passed mark are lost. The rest of your file is untouched, and one Undo brings it back.`,
      )
    )
      return;
    const result = resetExercise(def.id);
    setSpliceNote(result.ok ? null : (result.reason ?? null));
    props.onChanged();
  }, [def, props]);

  const undo = useCallback(() => {
    if (undoLastSplice()) {
      setSpliceNote(null);
      props.onChanged();
    }
  }, [props]);

  // Dragging the divider between the editor and the output.
  const dragSplit = useCallback((startEvent: React.PointerEvent) => {
    const host = (startEvent.currentTarget as HTMLElement).parentElement;
    if (!host) return;
    startEvent.preventDefault();
    const box = host.getBoundingClientRect();
    const move = (e: PointerEvent) => {
      const pct = ((e.clientY - box.top) / box.height) * 100;
      setSplitPercent(Math.max(25, Math.min(85, pct)));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      setSplitPercent((pct) => {
        saveUi(SPLIT_KEY, String(Math.round(pct)));
        return pct;
      });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }, []);

  const problems = doc.problems.filter((p) => p.kind !== "out-of-order");
  const notes = doc.problems.filter((p) => p.kind === "out-of-order");
  const touchedGivens = useMemo(() => editedGivens(), [revision]);

  const liveMessage = error
    ? `Run failed. ${error}`
    : cancelled
      ? "Stopped."
      : blaming
        ? "Checking the sections above this one."
        : busy
          ? status || "Running..."
          : result && !stale
            ? result.setup_error
              ? "Your file did not run. The reason is above the results."
              : result.passed
                ? "All tests passed."
                : `${result.tests.filter((t) => t.passed).length} of ${result.tests.length} tests passed.`
            : "";

  return (
    <aside
      ref={panelRef}
      className="wb"
      data-dock={dockState}
      hidden={!open}
      aria-label="Your library"
      role={dockState === "sheet" ? "dialog" : undefined}
      aria-modal={dockState === "sheet" ? true : undefined}
      aria-busy={running}
    >
      <div className="wb-head">
        <h2 className="wb-title" tabIndex={-1}>
          Your library <code>nn.py</code>
        </h2>
        <div className="wb-head-buttons">
          <button className="button-secondary wb-download" onClick={download}>
            Download
          </button>
          <button className="button-secondary wb-close" onClick={props.onClose}>
            Close
          </button>
        </div>
      </div>

      <div className="wb-rail scroll-x" role="list" aria-label="Sections of your file">
        {SECTIONS.map((section) => (
          <RailChip
            key={section.id}
            section={section}
            state={sectionState(section.id)}
            currentId={current}
            onSelect={props.onSelectSection}
          />
        ))}
      </div>

      {(problems.length > 0 || spliceNote || touchedGivens.length > 0) && (
        <div className="wb-repair">
          {problems.map((p) => (
            <p key={`${p.kind}-${p.line}`}>{p.message}</p>
          ))}
          {touchedGivens.map((s) => (
            <p key={s.id}>
              Your {s.label} section has been changed. That stretch was written for you and the
              tests above and below it are pinned to what it does, so a failure there will look
              like a failure in your own code.
            </p>
          ))}
          {spliceNote && <p>{spliceNote}</p>}
          {canUndo() && (
            <button className="button-secondary" onClick={undo}>
              Undo the last change the course made
            </button>
          )}
        </div>
      )}

      <div className="wb-body" style={{ "--wb-split": `${splitPercent}%` } as React.CSSProperties}>
        <div className="wb-editor-slot">
          {everOpened ? (
            <Suspense fallback={<EditorPlaceholder />}>
              <CodeEditor
                className="code-editor wb-editor"
                initialDoc={loadDocument()}
                onChange={props.onDocumentChange}
                onSelection={props.onCaret}
                handleRef={editorRef}
                onReady={props.onEditorReady}
              />
            </Suspense>
          ) : (
            <EditorPlaceholder />
          )}
        </div>
        <div
          className="wb-splitter"
          role="separator"
          aria-orientation="horizontal"
          aria-label="Space given to the editor"
          aria-valuenow={Math.round(splitPercent)}
          aria-valuemin={25}
          aria-valuemax={85}
          tabIndex={0}
          onPointerDown={dragSplit}
          onKeyDown={(e) => {
            const step = e.key === "ArrowUp" ? -4 : e.key === "ArrowDown" ? 4 : 0;
            if (!step) return;
            e.preventDefault();
            setSplitPercent((pct) => {
              const next = Math.max(25, Math.min(85, pct + step));
              saveUi(SPLIT_KEY, String(Math.round(next)));
              return next;
            });
          }}
        />
        <div className="wb-lower">
          <div className="wb-controls">
            <button onClick={props.onRunTests} disabled={!canRun || !current}>
              {busy === "tests" ? "Running..." : "Run tests"}
            </button>
            <button className="button-secondary" onClick={props.onRunScratch} disabled={!canRun}>
              {busy === "scratch" ? "Running..." : "Run my code"}
            </button>
            {running && (
              <button className="button-secondary" onClick={props.onStop}>
                Stop
              </button>
            )}
            <details className="wb-more">
              <summary>More</summary>
              <div className="wb-more-menu">
                <button className="button-secondary" onClick={resetCurrent} disabled={!def}>
                  Reset this section
                </button>
                <button className="button-secondary" onClick={undo} disabled={!canUndo()}>
                  Undo that
                </button>
              </div>
            </details>
          </div>
          <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">
            {liveMessage}
          </p>
          <p className="wb-where status-fixed">
            {def ? (
              <>
                Run tests checks <b>{def.label}</b>.
              </>
            ) : (
              "Put the caret in a section, or pick one above, to choose what Run tests checks."
            )}
          </p>
          {status && <p className="demo-status">{status}</p>}
          {blaming && <p className="demo-status">Checking the sections above this one...</p>}
          {cancelled && <p className="demo-status">Stopped. Press Run tests to try again.</p>}
          {error && <p className="demo-status demo-status-error">Something went wrong: {error}</p>}
          {notes.map((p) => (
            <p key={p.line} className="demo-status">
              {p.message}
            </p>
          ))}

          <details
            className="wb-scratch"
            open={scratchOpen}
            onToggle={(e) => setScratchOpen((e.currentTarget as HTMLDetailsElement).open)}
          >
            <summary>Scratch pad</summary>
            <p className="wb-scratch-note">
              Anything here runs after your library, with every name in it available, and is never
              part of it. This is where a "Run my code" experiment goes.
            </p>
            {scratchOpen && (
              <Suspense fallback={<EditorPlaceholder />}>
                <CodeEditor
                  className="code-editor wb-scratch-editor"
                  initialDoc={loadScratch()}
                  onChange={props.onScratchChange}
                  handleRef={scratchRef}
                />
              </Suspense>
            )}
          </details>

          {blame && (
            <div className="wb-blame">
              <strong>Before this section.</strong>
              <p>
                Your {blame.section.label} section is failing {blame.failing} of {blame.total} of its
                own tests, and the tests below call it. Fixing it there is likely to fix these.
              </p>
              {blame.firstMessage && <p className="wb-blame-message">{blame.firstMessage}</p>}
              <button
                className="button-secondary"
                onClick={() => props.onSelectSection(blame.section.id)}
              >
                Go to {blame.section.label}
              </button>
            </div>
          )}

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
                  {scratchError.line !== null && ` (line ${scratchError.line})`}
                  {scratchError.label ? `, in the ${scratchError.label}` : ""}.
                </p>
              )}
            </div>
          )}

          {lent !== null && !running && (
            <p className="wb-lent">
              {lent.length === 0
                ? "Run entirely on your own code."
                : `Run with the course's ${listNames(lent)}. Your own versions run here once those sections are written.`}
            </p>
          )}

          {result && (
            <TestResults
              result={result}
              flagship={flagship}
              stale={stale}
              onGoToSection={props.onSelectSection}
            />
          )}

          <details className="wb-tip">
            <summary>How the two Run buttons differ</summary>
            <p>
              <b>Run tests</b> runs your whole file and then checks the section named above.{" "}
              <b>Run my code</b> runs your whole file and then the scratch pad, so you can call your
              functions and print(...) values. Everything you print appears in the Output panel, as
              it happens. Tab indents, and Escape then Tab moves keyboard focus out. Your file is
              saved in this browser only. A symbol you have forgotten is in the notation reference
              on the Start page.
            </p>
          </details>
        </div>
      </div>
    </aside>
  );
}

function listNames(names: string[]): string {
  const code = names.map((n) => `${n}`);
  if (code.length === 1) return code[0];
  return `${code.slice(0, -1).join(", ")} and ${code[code.length - 1]}`;
}

function RailChip({
  section,
  state,
  currentId,
  onSelect,
}: {
  section: SectionDef;
  state: SectionState;
  currentId: string | null;
  onSelect(id: string): void;
}) {
  const isCurrent = currentId === section.id;
  return (
    <button
      role="listitem"
      className={`wb-chip wb-chip-${state} ${isCurrent ? "wb-chip-current" : ""}`}
      aria-current={isCurrent ? "true" : undefined}
      onClick={() => onSelect(section.id)}
      title={`${section.label}: ${STATE_LABEL[state]}`}
    >
      <span className="wb-chip-mark" aria-hidden="true">
        {state === "passing" ? "✓" : state === "stale" ? "!" : state === "missing" ? "+" : "·"}
      </span>
      {section.label.replace(/^Module (\d+), /, "$1. ")}
      <span className="sr-only">, {STATE_LABEL[state]}</span>
    </button>
  );
}

function EditorPlaceholder() {
  return <div className="code-editor code-editor-loading">Loading the editor...</div>;
}
