// Progress persistence (localStorage, versioned key prefix). Stores per
// exercise: the learner's editor code, the hint reveal stage (0 to 3), and
// a completion flag. No accounts, no backend (see design doc).

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
}

export function resetExercise(exerciseId: string): void {
  remove(`code:${exerciseId}`);
  remove(`reveal:${exerciseId}`);
  remove(`done:${exerciseId}`);
}
