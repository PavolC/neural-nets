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
 * Pick something the course itself draws. This course uses the sigmoid curve, which
 * is the first figure in Module 1 and the shape every neuron in the course is
 * built from. A glyph that means nothing is worse than a letter.
 */
export interface Glyph {
  viewBox: string;
  /** A single stroked path. No fills, so the tile works at 16px. */
  d: string;
  strokeWidth: number;
}

export const SERIES = {
  /** An imprint rather than a prefix: the courses are "Neural Networks" and
   *  "Ciphers", published under this name, not "Moving Parts Neural Networks".
   *  That is why the masthead puts the wordmark above the title instead of in
   *  front of it, and why COURSE_TITLE below reads subject first. */
  name: "Moving Parts",
  /** Uppercased into the masthead beside the wordmark. Keep it to four words. */
  note: "build-it-yourself courses",
  /** One sentence, in the footer. */
  what: "A series of courses you finish by building the thing they are about.",
  /** The series index, which is the only place that knows what else exists.
   *
   *  A course links UP to it and never across to a sibling. The obvious design
   *  was the other way round, with each course carrying the list and linking
   *  to its siblings, and it is a trap: shipping the fourth course would mean
   *  editing and redeploying four repositories, and any one of them forgotten
   *  shows a stale list forever. This is the same hand-maintained-list failure
   *  that once let the front page claim ten modules over a list of eight,
   *  multiplied by the number of courses. Linking up means shipping a course
   *  edits exactly one repository, and nothing anywhere else can go stale.
   *
   *  Set once when a course is created, then never touched. It is series-level
   *  rather than course-level: every course in the series points at the same
   *  index, so this line is identical in all of them and a sibling copies it
   *  unchanged. Null would leave the wordmark as plain text rather than a link
   *  to a 404, which is what it was until the index went live. */
  homeUrl: "https://pavolc.github.io/moving-parts/" as string | null,
};

export const COURSE = {
  /** The slug, not the display name. It is the stem of the progress file's
   *  name and the series index's key for this course, so it stays put when the
   *  subject above is reworded. */
  id: "nets",
  subject: "Neural Networks",
  /** Under the title, which now carries the topic itself, so this says what the
   *  reader does here rather than repeating the subject back at them. */
  tagline:
    "Read a little, play with live visualizations, and implement feedforward, " +
    "gradient descent and backpropagation in Python, right here in your browser.",
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
