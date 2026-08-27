import { useEffect, useRef } from "react";
import { basicSetup, EditorView } from "codemirror";
import { keymap } from "@codemirror/view";
import { indentWithTab } from "@codemirror/commands";
import { HighlightStyle, indentUnit, syntaxHighlighting } from "@codemirror/language";
import { python } from "@codemirror/lang-python";
import { Prec } from "@codemirror/state";
import { tags } from "@lezer/highlight";

/* The editor's own look, which used to be CodeMirror's stock light theme:
   the one surface in the course nobody had chosen. Everything below is a
   brand token, so the editor moves with the page and a sibling course that
   changes its accent gets a matching editor without touching this file.

   Only chrome here. The token colours are the highlight style below. */
const chrome = EditorView.theme(
  {
    "&": {
      backgroundColor: "var(--surface-card)",
      color: "var(--ink)",
      fontSize: "0.85rem",
    },
    ".cm-content": {
      fontFamily: "var(--font-mono)",
      caretColor: "var(--accent)",
      padding: "0.6rem 0",
    },
    ".cm-gutters": {
      backgroundColor: "var(--surface-sunken)",
      color: "var(--muted)",
      border: "none",
      borderRight: "1px solid var(--rule)",
      fontFamily: "var(--font-mono)",
    },
    ".cm-lineNumbers .cm-gutterElement": { padding: "0 0.55rem 0 0.9rem" },
    ".cm-activeLine": { backgroundColor: "var(--accent-wash)" },
    ".cm-activeLineGutter": {
      backgroundColor: "var(--accent-wash)",
      color: "var(--ink)",
    },
    "&.cm-focused .cm-cursor": { borderLeftColor: "var(--accent)", borderLeftWidth: "2px" },
    // Both spellings: the focused editor uses ::selection, the unfocused one
    // paints its own layer, and leaving either out means a selection that
    // vanishes the moment the reader clicks the Run button.
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection": {
      backgroundColor: "var(--accent-panel)",
    },
    ".cm-matchingBracket, .cm-nonmatchingBracket": {
      backgroundColor: "var(--accent-panel)",
      outline: "none",
    },
    ".cm-foldPlaceholder": {
      backgroundColor: "var(--surface-sunken)",
      border: "1px solid var(--rule)",
      color: "var(--muted)",
    },
  },
  { dark: false },
);

/* The token colours, taken from the brand's accent family rather than picked.
   Every hue in that family sits at one OKLCH lightness chosen so it clears
   6:1 against the page ground, which is what makes this palette legible by
   construction instead of by eye. Comments are the page's muted ink, so they
   recede the way a comment should. */
const highlight = HighlightStyle.define([
  { tag: tags.comment, color: "var(--muted)", fontStyle: "italic" },
  { tag: [tags.keyword, tags.controlKeyword, tags.moduleKeyword], color: "var(--hue-violet)" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--hue-moss)" },
  { tag: [tags.number, tags.bool, tags.null, tags.atom], color: "var(--hue-oxide)" },
  { tag: [tags.definition(tags.variableName), tags.function(tags.variableName)], color: "var(--hue-blue)" },
  { tag: [tags.className, tags.typeName], color: "var(--hue-teal)" },
  { tag: tags.operator, color: "var(--hue-plum)" },
  { tag: tags.self, color: "var(--hue-violet)", fontStyle: "italic" },
  { tag: tags.invalid, color: "var(--loss)" },
]);

export interface CodeEditorHandle {
  getDoc(): string;
  setDoc(doc: string): void;
}

// Thin CodeMirror 6 wrapper. Uncontrolled: the editor owns its document;
// the parent reads/writes through the handle and gets onChange callbacks.
export function CodeEditor({
  initialDoc,
  onChange,
  handleRef,
  onReady,
}: {
  initialDoc: string;
  onChange: (doc: string) => void;
  handleRef: React.MutableRefObject<CodeEditorHandle | null>;
  /** Called once the view exists and the handle is usable. The parent loads
   * this component lazily, so until then there is no document to run. */
  onReady?: (ready: boolean) => void;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  useEffect(() => {
    const view = new EditorView({
      doc: initialDoc,
      parent: hostRef.current!,
      extensions: [
        basicSetup,
        // After basicSetup, and at raised precedence: basicSetup ships
        // CodeMirror's own highlight style, and the facet applies the first
        // matching rule it finds rather than the last one added.
        chrome,
        Prec.high(syntaxHighlighting(highlight)),
        python(),
        indentUnit.of("    "), // Python convention: 4 spaces
        // Tab indents (Shift-Tab dedents). CodeMirror leaves this off by
        // default for keyboard accessibility; Escape then Tab moves focus.
        keymap.of([indentWithTab]),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) onChangeRef.current(update.state.doc.toString());
        }),
      ],
    });
    handleRef.current = {
      getDoc: () => view.state.doc.toString(),
      setDoc: (doc: string) => {
        view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: doc } });
      },
    };
    onReadyRef.current?.(true);
    return () => {
      handleRef.current = null;
      onReadyRef.current?.(false);
      view.destroy();
    };
    // Mount once; initialDoc changes after mount are applied via the handle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div className="code-editor" ref={hostRef} />;
}
