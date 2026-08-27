// Progress persistence (localStorage, versioned key prefix). Stores per
// exercise: the learner's editor code, the hint reveal stage (0 to 3), and
// a completion flag. No accounts, no backend (see design doc).

// Not renamed with the series. Every learner's saved code, revealed hints and
// passed marks live under this prefix in their own browser, and changing it
// orphans all of it silently: they would open the course to empty editors with
// no way to get their work back. The name it is short for is history.
const PREFIX = "gn:v1:";

function get(key: string): string | null {
  try {
    return localStorage.getItem(PREFIX + key);
  } catch {
    return null; // storage disabled: the course still works, nothing persists
  }
}

function set(key: string, value: string): void {
  try {
    localStorage.setItem(PREFIX + key, value);
  } catch {
    // ignore: see above
  }
}

function remove(key: string): void {
  try {
    localStorage.removeItem(PREFIX + key);
  } catch {
    // ignore
  }
}

export function loadCode(exerciseId: string): string | null {
  return get(`code:${exerciseId}`);
}

export function saveCode(exerciseId: string, code: string): void {
  set(`code:${exerciseId}`, code);
}

/** 0 = nothing revealed, 1 = hint 1, 2 = hint 2, 3 = full solution. */
export function loadRevealStage(exerciseId: string): number {
  const raw = get(`reveal:${exerciseId}`);
  const n = raw === null ? 0 : parseInt(raw, 10);
  return Number.isInteger(n) && n >= 0 && n <= 3 ? n : 0;
}

export function saveRevealStage(exerciseId: string, stage: number): void {
  set(`reveal:${exerciseId}`, String(stage));
}

export function loadCompleted(exerciseId: string): boolean {
  return get(`done:${exerciseId}`) === "1";
}

export function saveCompleted(exerciseId: string): void {
  set(`done:${exerciseId}`, "1");
  emitProgress();
}

// Completion unlocks the payoff panels that run the learner's own code (and
// the start page's progress list), so interested components can subscribe to
// changes. It has never gated navigation: every module is reachable always.
const PROGRESS_EVENT = "gn:progress";

function emitProgress(): void {
  window.dispatchEvent(new CustomEvent(PROGRESS_EVENT));
}

export function subscribeProgress(fn: () => void): () => void {
  window.addEventListener(PROGRESS_EVENT, fn);
  return () => window.removeEventListener(PROGRESS_EVENT, fn);
}

/** Passed, and the code that passed is still stored.
 *
 * The panels that train with the learner's own code need both, and the two
 * can come apart: a progress file that carries the passed marks without the
 * editor contents, or storage cleared halfway. Gating on the pass alone gave
 * those panels an unlocked button that quietly did nothing, because every one
 * of them returns early when the code turns out to be missing.
 */
export function codeReady(exerciseId: string): boolean {
  return loadCompleted(exerciseId) && loadCode(exerciseId) !== null;
}

export function resetExercise(exerciseId: string): void {
  remove(`code:${exerciseId}`);
  remove(`reveal:${exerciseId}`);
  remove(`done:${exerciseId}`);
  emitProgress();
}

/** Every stored key, without the prefix. Empty when storage is unavailable. */
function keys(): string[] {
  try {
    const out: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(PREFIX)) out.push(key.slice(PREFIX.length));
    }
    return out;
  } catch {
    return [];
  }
}

/** Forget everything: all saved code, hints and passes, for every exercise. */
export function resetAll(): void {
  for (const key of keys()) remove(key);
  emitProgress();
}

export interface ProgressFile {
  format: string;
  saved: string;
  entries: Record<string, string>;
}

// What a new export writes. The old spelling carried the series name, which
// has since changed, so it is no longer a good tag; but files exported under it
// are on people's disks, and importProgress accepts both. A tag naming the
// course rather than the series cannot go stale the same way again.
const FORMAT = "nets-course-progress-v1";
const ACCEPTED_FORMATS = [FORMAT, "grokking-nets-progress-v1"];

/** The whole of this browser's progress, as a JSON string to keep or move. */
export function exportProgress(): string {
  const entries: Record<string, string> = {};
  for (const key of keys()) {
    const value = get(key);
    if (value !== null) entries[key] = value;
  }
  const file: ProgressFile = {
    format: FORMAT,
    saved: new Date().toISOString(),
    entries,
  };
  return JSON.stringify(file, null, 2);
}

/** Load an exported file over the current progress.
 *
 * Additive by key: an exercise present in the file replaces the one in this
 * browser, and one absent from the file is left alone. Returns how many keys
 * were written, or throws with a message worth showing the reader.
 */
export function importProgress(text: string): number {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("that file is not JSON, so it is not a progress file this course wrote");
  }
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error("that file does not hold a progress record");
  }
  const file = parsed as Partial<ProgressFile>;
  if (typeof file.format !== "string" || !ACCEPTED_FORMATS.includes(file.format)) {
    throw new Error(
      `that file says its format is ${JSON.stringify(file.format ?? "missing")}, and this ` +
        `course reads ${ACCEPTED_FORMATS.join(" or ")}`,
    );
  }
  if (typeof file.entries !== "object" || file.entries === null) {
    throw new Error("that progress file has no entries");
  }
  let written = 0;
  for (const [key, value] of Object.entries(file.entries)) {
    // Only the three shapes this course writes, so a hand-edited file cannot
    // fill the browser's storage with anything else under our prefix.
    if (!/^(code|reveal|done):[a-z0-9-]+$/.test(key)) continue;
    if (typeof value !== "string") continue;
    set(key, value);
    written++;
  }
  if (written === 0) throw new Error("that progress file holds nothing this course can read");
  emitProgress();
  return written;
}
