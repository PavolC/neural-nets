/**
 * The series identity, and this course's slot in it.
 *
 * This is the one file in the brand folder a course edits. Nothing else here
 * knows the subject, so a course that changes its name, its hue and its glyph
 * below is fully rebranded. The other change is one line in brand.css, marked
 * COURSE ACCENT.
 *
 * Everything marked FILL has to be closed before the first chapter ships.
 */

/**
 * A course's mark, drawn once and used in three places: the monogram tile in
 * the masthead, the smaller tile in the footer, and the favicon in index.html.
 * One path so the three can never drift apart; tools/check_brand.py asserts
 * that the favicon still carries this exact path.
 *
 * Pick something the course itself draws. The first course used the sigmoid
 * curve, which is the first figure in its chapter 1 and the shape every unit in
 * the course is built from. A glyph that means nothing is worse than a letter.
 */
export interface Glyph {
  viewBox: string;
  /** A single stroked path. No fills, so the tile works at 16px. */
  d: string;
  strokeWidth: number;
}

export interface CourseRef {
  id: string;
  /** The subject alone: the series name is prepended for the full title. */
  subject: string;
  /** Where it is published, or null for the course you are reading. */
  url: string | null;
}

export const SERIES = {
  name: "Grokking",
  /** Uppercased into the masthead beside the wordmark. Keep it to four words. */
  note: "build-it-yourself courses",
  /** One sentence, in the footer. */
  what: "A series of courses you finish by building the thing they are about.",
  /** An index page listing every course, once one exists. Null leaves the
   *  wordmark as plain text rather than as a link to a 404. */
  homeUrl: null as string | null,
  /** Every course in the series, in the order they were written. The footer
   *  turns this into sibling links, which is what makes a reader who lands on
   *  one course discover the rest. Add a row here when a new course ships,
   *  in every course's copy of this file. */
  courses: [
    { id: "nets", subject: "Nets", url: "https://pavolc.github.io/neural-nets/" },
    // FILL: this course, with url: null.
  ] as CourseRef[],
};

export const COURSE = {
  /** FILL: matches this course's id in SERIES.courses above. */
  id: "FILL",
  /** FILL: the series name plus this reads as the full title. One word is best. */
  subject: "FILL",
  /** FILL: one sentence, in the masthead under the title. What the reader does
   *  here, not what the topic is. */
  tagline: "FILL",
  /** FILL: see the Glyph docstring above. */
  glyph: {
    viewBox: "0 0 32 32",
    d: "M5 24C12 24 15 19 16 16C17 13 20 8 27 8",
    strokeWidth: 3,
  } as Glyph,
};

/** The full title. One place, so the document title, the masthead's accessible
 *  name and the footer cannot disagree. */
export const COURSE_TITLE = `${SERIES.name} ${COURSE.subject}`;
