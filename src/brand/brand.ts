/**
 * The series identity, and this course's slot in it.
 *
 * A sibling course copies the whole brand folder and edits this file. Nothing
 * else in the folder knows the subject, so a course that changes its name, its
 * hue and its glyph here is fully rebranded.
 */

/**
 * A course's mark, drawn once and used in three places: the monogram tile in
 * the masthead, the smaller tile in the footer, and the favicon in index.html.
 * One path so the three can never drift apart; tools/check_brand.py asserts
 * that the favicon still carries this exact path.
 *
 * Pick something the course itself draws. Nets uses the sigmoid curve, which
 * is the first figure in Module 1 and the shape every neuron in the course is
 * built from. A glyph that means nothing is worse than a letter.
 */
export interface Glyph {
  viewBox: string;
  /** A single stroked path. No fills, so the tile works at 16px. */
  d: string;
  strokeWidth: number;
}

export interface CourseRef {
  id: string;
  /** The subject alone. The series name is an imprint, not a prefix. */
  subject: string;
  /** Where it is published, or null for the course you are reading. */
  url: string | null;
}

export const SERIES = {
  /** An imprint rather than a prefix: the courses are "Nets" and "Ciphers",
   *  published under this name, not "Moving Parts Nets". That is why the
   *  masthead puts the wordmark above the title instead of in front of it, and
   *  why COURSE_TITLE below reads subject first. */
  name: "Moving Parts",
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
  courses: [{ id: "nets", subject: "Nets", url: null }] as CourseRef[],
};

export const COURSE = {
  id: "nets",
  subject: "Nets",
  tagline:
    "An interactive course on neural networks: read a little, play with live " +
    "visualizations, and implement the real thing in Python, right here in your browser.",
  /** The sigmoid curve: flat, steepest in the middle, flat. Two cubics
   *  meeting at the tile's centre, each horizontal at its outer end and
   *  sharing the same slope where they join, which is the one property that
   *  makes the shape read as a sigmoid rather than as a squiggle. */
  glyph: {
    viewBox: "0 0 32 32",
    d: "M5 24C12 24 15 19 16 16C17 13 20 8 27 8",
    strokeWidth: 3,
  } as Glyph,
};

/** The document title. Subject first, because that is the word a reader needs
 *  when eight tabs are open, then the series as the imprint.
 *
 *  This is the canonical spelling, and index.html has to repeat it as a
 *  literal, because a tab needs its title before any JavaScript runs.
 *  tools/check_brand.py asserts that the two agree, which is what stops a
 *  rename from reaching the masthead and leaving the tab behind. */
export const COURSE_TITLE = `${COURSE.subject} · ${SERIES.name}`;
