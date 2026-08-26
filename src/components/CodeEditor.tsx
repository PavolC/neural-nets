import { useEffect, useRef } from "react";
import { basicSetup, EditorView } from "codemirror";
import { keymap } from "@codemirror/view";
import { indentWithTab } from "@codemirror/commands";
import { indentUnit } from "@codemirror/language";
import { python } from "@codemirror/lang-python";

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
