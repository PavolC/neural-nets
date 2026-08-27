# The series brand

A visual identity meant to carry across several courses on unrelated topics, published as
separate GitHub Pages sites, so a reader who lands on one recognizes the others as siblings.

Design constraint: plain and readable, with identity. Not a design system, and no
dependency a course has to install. The whole thing is five files and about 480 lines.

---

## What makes it a set

Four things, and only these four, have to hold across courses.

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

**4. The lockup.** A monogram tile, the series wordmark, and the course subject on the
line below:

```
[▮]  GROKKING  |  BUILD-IT-YOURSELF COURSES
Nets
An interactive course on neural networks: read a little, ...
```

The name reads down the page, so a sibling reads `GROKKING / Ciphers` with nothing else
moved. The heading carries the whole name for a screen reader (the series half is
`sr-only`), because a heading that announces "Nets" alone leaves a listener without the
series it belongs to.

The monogram is the course's glyph in white on a rounded tile of its accent. **Pick
something the course itself draws.** Nets uses the sigmoid curve, which is the first figure
in chapter 1 and the shape every unit in the course is built from. A glyph that means
nothing is worse than a letter.

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

## When a new course ships

Add its row to `SERIES.courses` **in every course's copy of `brand.ts`**, and redeploy them.
That is what turns a reader who found one course into a reader who knows there are others.
Set `SERIES.homeUrl` once an index page exists; until then it stays `null` and the wordmark
is plain text rather than a link to a 404.

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
