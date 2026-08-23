/** A prompt is paragraphs of plain text, with optional runnable code blocks
 * (rendered copyable, with an append-to-editor button). */
export type PromptPart = string | { code: string };

export interface Exercise {
  id: string;
  title: string;
  /** Short prose beats shown above the editor. */
  prompt: PromptPart[];
  skeleton: string;
  tests: string;
  solution: string;
  /** Three-stage reveal: [conceptual nudge, pseudocode/structure]. Stage 3
   * is the full solution above. */
  hints: [string, string];
}
