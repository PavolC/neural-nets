import { useEffect, useRef } from "react";
import { basicSetup, EditorView } from "codemirror";
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
}: {
  initialDoc: string;
  onChange: (doc: string) => void;
  handleRef: React.MutableRefObject<CodeEditorHandle | null>;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    const view = new EditorView({
      doc: initialDoc,
      parent: hostRef.current!,
      extensions: [
        basicSetup,
        python(),
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
    return () => {
      handleRef.current = null;
      view.destroy();
    };
    // Mount once; initialDoc changes after mount are applied via the handle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div className="code-editor" ref={hostRef} />;
}
