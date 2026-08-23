export interface Exercise {
  id: string;
  title: string;
  /** Short prose beat shown above the editor. Plain text paragraphs. */
  prompt: string[];
  skeleton: string;
  tests: string;
  solution: string;
  /** Three-stage reveal: [conceptual nudge, pseudocode/structure]. Stage 3
   * is the full solution above. */
  hints: [string, string];
}
