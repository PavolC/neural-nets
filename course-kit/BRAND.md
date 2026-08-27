# The series brand

A visual identity meant to carry across several courses on unrelated topics, published as
separate GitHub Pages sites, so a reader who lands on one recognizes the others as siblings.

Design constraint: plain and readable, with identity. Not a design system, and no
dependency a course has to install. The whole thing is five files and about 480 lines.

---

## What makes it a set

Five things, and only these five, have to hold across courses.

**1. The accent family.** Nine hues at one OKLCH lightness (0.478) and one chroma (0.0975),
the values of the first course's green, held fixed while the hue turns 36 degrees at a time.
Same lightness and chroma means siblings read as a set: only the hue tells them apart, so no
course looks louder than its neighbour, and none can pick an accent that is off-brand by
being darker or more saturated.

Contrast is a consequence rather than a hope. Every hue lands between 6.1:1 and 6.9:1
against the page ground and between 6.2:1 and 6.9:1 under white ink, so any one of them
works as text, as a rule, and as a button fill. AA wants 4.5:1.

| token | hex | on the ground | under white ink |
|---|---|---|---|
| `--hue-green` | `#0b6e4f` | 6.14 | 6.25 |
| `--hue-teal` | `#016a70` | 6.25 | 6.37 |
| `--hue-blue` | `#12648d` | 6.38 | 6.50 |
| `--hue-indigo` | `#4b5894` | 6.60 | 6.72 |
| `--hue-violet` | `#6d4d87` | 6.73 | 6.86 |
| `--hue-plum` | `#83456a` | 6.82 | 6.94 |
| `--hue-crimson` | `#8c4445` | 6.81 | 6.94 |
| `--hue-oxide` | `#864d1e` | 6.66 | 6.79 |
| `--hue-moss` | `#4c6726` | 6.30 | 6.41 |

`tools/brand_palette.py` regenerates the family and prints these ratios; `--check` fails if
`brand.css` has drifted from what it computes. One stop on the circle is deliberately
skipped: the hue between oxide and moss comes out an olive that reads as a mistake rather
than as a choice.

Washes are mixed from whatever the accent is, so a course that changes one line gets a
matching set instead of hand-picked near-whites that now clash:

```css
--accent-wash:  color-mix(in oklab, var(--accent) 6%,  var(--bg));  /* tinted paper */
--accent-panel: color-mix(in oklab, var(--accent) 14%, var(--bg));  /* a visible panel */
--accent-rule:  color-mix(in oklab, var(--accent) 30%, var(--bg));  /* a visible rule */
```

**2. The rule across the top.** Three pixels of the course's own hue, at the very top of
every page. It is on `body` rather than fixed, so it never fights a sticky bar or a
fullscreen editor for the top three pixels.

**3. The type pairing.** A serif for everything read in sentences, a sans for everything
that is chrome. The contrast between them is what tells a reader that a piece of text is
machinery rather than prose, and it costs nothing:

```css
--font-prose:   Georgia, "Times New Roman", serif;
--font-ui:      system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
--font-display: var(--font-ui);
--font-mono:    ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
```

Georgia is a choice, not a fallback: the most consistently available serif with a real
italic, and a large enough x-height to survive the 19px root at the sizes figures need.
Nothing is downloaded, which keeps a promise the courses make about what leaves the
reader's machine. If a course ever ships a display face, it replaces `--font-display`
alone and nothing else moves.

The label idiom is the sans doing its most visible job: small, uppercase, letterspaced.
It marks the series wordmark, the aside and nav labels, the passed badge, and every
"this is a kind of thing" marker.

```css
--label-size: 0.72rem;  --label-tracking: 0.09em;  --label-weight: 600;
```

**4. The reading measure.** Prose runs to `--measure` (34rem, about 69 characters at a 19px
root); figures, tables, panels and the editor keep the whole column. Display equations take
`--measure-wide` (42rem) because they centre themselves and cannot re-wrap. The page then
reads as a text column with wider illustrations beside it, which is the single change that
does most for how designed it looks.

Apply it with direct-child selectors on the reading column, never on `p` globally: in a
stylesheet with two dozen interactives in it, a paragraph inside a panel or a control row
must not inherit a prose measure.

**5. The lockup.** A monogram tile, the series wordmark, and the course subject on the
line below:

```
[▮]  MOVING PARTS  |  BUILD-IT-YOURSELF COURSES
Neural Networks
An interactive course on neural networks: read a little, ...
```

**The series name is an imprint, not a prefix.** The courses are "Neural Networks" and
"Ciphers", published under one name, so the wordmark sits above the heading rather than in
front of it, and a sibling reads the same two lines with only the heading changed. Pick a
series name that works this way. A name that has to be prefixed stops being grammatical
the moment the subject is a plural or a noun phrase ("Grokking Neural Networks" is a
phrase, not a title), and it puts a word in front of the heading that every page in the
series already carries.

The heading needs no hidden prefix, because the wordmark right above it is text: a screen
reader reaches the series name first and then the course. The document title reverses them
("Neural Networks · Moving Parts"), since the subject is the word that has to survive being
one tab of eight.

The monogram is the course's glyph in white on a rounded tile of its accent. **Pick
something the course itself draws.** The first course uses the sigmoid curve, which is the
first figure in its chapter 1 and the shape every unit in the course is built from. A glyph
that means nothing is worse than a letter.

## Three details that carry more than they look like

**A section title wears a short accent rule above it.** 26 by 2 pixels, in the course's
hue. On a long single-scroll page it is the only thing that says a new section started, and
because it is drawn from the accent it is the one piece of structural furniture a sibling
course inherits already recoloured.

**Hover is declared per variant, never on `button`.** A bare `button:hover` outranks
`.tab:hover` and every other variant that sets its own transparent background, and fills
them solid. The first course shipped that bug and had to revert the whole hover pass. The
primary treatment is reached as `button:not([class])`: every variant carries a class, so
the selector cannot reach one.

**The nav strip is one panning row below 720px.** Eleven tabs wrap to six rows on a phone
and take the entire first screen before the course starts. One row that pans, with the
active tab scrolled to centre and an edge fade that says there is more, is the same
information in a tenth of the height.

## The code editor

If the course has an in-page editor, theme it: a stock editor theme is the most visible
surface on the page that nobody chose. Two halves, both in the editor component rather
than the stylesheet, because the editor generates its own class names:

- **Chrome** from the surfaces and the accent: the card surface behind the code, the sunken
  surface behind the gutter, the accent as the caret, the 6 percent wash as the active line,
  the 14 percent panel as the selection.
- **Token colours from the accent family.** Do not hand-pick syntax colours. Every hue in
  the family sits at the one lightness that clears 6:1 on the page ground, so taking
  keywords from violet, strings from moss, numbers from oxide and definitions from blue
  gives a syntax theme that is legible by construction and unmistakably the same brand.

## The files

| file | lines | what a course changes |
|---|---|---|
| `brand.css` | ~300 | one line: which hue `--accent` points at |
| `brand.ts` | ~67 | all of it: name, note, glyph, sibling list |
| `Monogram.tsx` | ~26 | nothing |
| `Masthead.tsx` | ~44 | nothing |
| `SeriesFooter.tsx` | ~43 | nothing |

`brand.css` is loaded first, so a course can still override anything in it:

```css
@import "./brand/brand.css";
:root { /* the course's own tokens */ }
```

The footer's legal text arrives as children rather than from `brand.ts`, because every
course carries different obligations and a shared component holding them would end up
either wrong or empty.

## Wiring it into a course

1. Copy `brand/` to `src/brand/`.
2. Edit `brand.ts`: the series name, the note, the course's subject and tagline, and the
   glyph path. Add the course to `SERIES.courses`.
3. Edit the one `--accent` line in `brand.css` to an unused hue.
4. `@import "./brand/brand.css";` at the top of the course stylesheet, and delete whatever
   it already had for `h1`, the tagline, the nav strip and the footer, so the two do not
   fight over specificity.
5. Render `<Masthead nav={...} />` and `<SeriesFooter>{attribution}</SeriesFooter>`.
6. Put the same glyph in `index.html`'s inline favicon, tile filled with the accent and the
   path stroked white, and set `theme-color` to the accent.
7. Copy `tools/check_brand.py` and `tools/brand_palette.py` and run both.

Step 6 is the one that goes stale, which is what `check_brand.py` is for: the favicon is
the only copy of the mark that no component can generate, because a tab needs its icon
before any JavaScript runs.

## When a new course ships: link up, never across

**A course links up to the series index. It does not list its siblings.**

The obvious design is the other way round: each course carries the list of courses and
links across to the others. It is a trap. Shipping the fourth course then means editing and
redeploying four repositories, and any one of them forgotten shows a stale list forever.
That is the hand-maintained-list failure in its purest form, multiplied by the number of
courses, and the first course in this series shipped a smaller version of the same bug: a
front page claiming ten modules over a list of eight, because the list was written before
two of them existed.

So: `SERIES.homeUrl` is set once when a course is created and never touched again, and the
index is the one thing that knows what exists. Shipping a course edits exactly one
repository. Nothing anywhere else can go stale, because nothing anywhere else knows.

The index is a single static page, one card per course, each card in that course's own hue.
It wants no build step: with a handful of courses, a hand-maintained list in the one place
that is allowed to have one is correct.

`homeUrl` stays `null` until the index is actually published, which leaves the wordmark as
plain text rather than as a link to a 404.

## The social card

A course with no `og:image` is invisible in every place a link is pasted: Slack, Discord,
LinkedIn, X, iMessage and Bluesky all render it as a line of grey text, which is the
weakest possible showing for work whose whole argument is that it is worth looking at.
The card is the one piece of the identity that is seen by people who have not arrived yet.

Course one draws it as a rendered HTML page (`tools/og_card.html`, screenshotted to
`public/og-image.png` by `tools/make_og_image.sh`) rather than as a drawn image, for the
same reason the rest of the identity is tokens: the card is then made of the accent, the
monogram path and the type roles, and a rebrand reaches it. Copy both files, change the
subject line and the three chips, and run the script.

Two things a sibling course must get right, because both fail silently:

- **Every URL in the card's meta tags is absolute.** `base: "./"` makes the build
  subpath-safe for the browser, and does nothing for a crawler, which has no page to
  resolve `./og-image.png` against. The deployed origin is a literal in `index.html`,
  alongside the favicon and the theme colour, and for the same reason.
- **The image is 1200x630.** That is the slot both `summary_large_image` and Open Graph
  render. Anything else is letterboxed or cropped, usually through the title.

`check_brand.py` carries the check: the four URLs name one origin, the file they name
exists in `public/`, and it is the size the tags declare.

## Deliberately not here

- **Dark mode.** Every colour is already a token, so a dark palette is a later drop-in
  rather than a rewrite. It was left out because the interactives carry dozens of hand-tuned
  SVG palettes that would each need a second reading, and that is a project rather than a
  pass.
- **A social card image.** `og:title` and `og:description` are set; an `og:image` needs a
  raster asset per course and the platforms that consume it do not reliably take SVG.
- **A type scale sweep.** The tokens name the scale the chrome uses. Rewriting every
  `font-size` in an existing course's stylesheet is churn with no visible return; new
  courses should use the tokens from the start.
