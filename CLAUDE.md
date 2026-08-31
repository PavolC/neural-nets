# CLAUDE.md

## What this project is

"Neural Networks", the first course in the **Moving Parts** series: a self-contained,
browser-based interactive course that teaches neural
networks from first principles. The learner reads short original explanations, manipulates
live visualizations, and writes real Python (NumPy) in an in-browser editor. Code is
validated by automated tests (including numerical gradient checking) and trains a real
network on an MNIST subset. No installs, no accounts, no backend; progress lives in
localStorage. Full design: `nn-course-design-doc.md`.

Goals, in priority order:

1. The primary learner achieves deep understanding by implementing feedforward, SGD, and
   backpropagation himself.
2. The artifact is fully self-sufficient: a colleague can open a link and complete the
   course with no book, no setup, no author involvement.
3. Demoable in under two minutes: open link, show a live visualization, show training.

Non-goals: not a production ML tool (small networks, subsampled data, clarity over
performance); not a full port of Nielsen's book (chapters 5-6 covered conceptually only);
no accounts, no backend; analytics off by default (opt-in counting via GoatCounter,
editor contents never sent; see `src/analytics.ts`).

## Hard rules

- **Never write solution logic into skeleton files.** Solutions live only in
  `solution.py`. Skeletons contain stubs, docstrings, and shape contracts only.
  The invariant is now about the assembled file, not one skeleton: **an untouched
  workbench must implement nothing.** Concatenating the nine untouched skeletons
  in their pre-workbench state passed 19 of 52 tests, because each opened with a
  line like `from course import sigmoid` and the harness execs the document
  before the tests import from it, so the last binding won retroactively for
  every suite. Running the suites cannot detect that; only the mutation check in
  `tools/check_exercises.py` (assertion G) can, which is why it exists and why
  its docstring says so.
- **A reader-facing unit is a chapter, while shared code still says module.** The
  series settled on one word, and it is the kit's rule rather than this course's
  (`course-kit/CLAUDE.md` and `METHOD.md` in `PavolC/moving-parts`): "Chapter N" in
  headings, navigation and pickers, with `cN-` section ids. The stylesheet and the
  shared components keep the `module` vocabulary they were written with (`.module > p`,
  `ModuleBits`, `ModuleToc`, `module-picker`, `MODULES`, the `module` field in
  `registry.ts` and `sections.json`, and `src/modules/NN/module1.tsx`), which is what
  makes them liftable into a sibling course. A reader never sees any of it, so renaming
  the code buys nothing and breaks that lift.
  This course predated the word and shipped ten modules, addressed `#m1` to `#m10`. Those
  addresses are in bookmarks and in links already sent, so `tabFromHash` in `App.tsx`
  **aliases** them to `#c1` to `#c10` and rewrites the bar, rather than swapping them.
  Do not delete that alias: a straight swap is what the rename was written to avoid.
  Two things stay stale on purpose, because both live inside a learner's saved file and
  rewriting a body is the one edit that could destroy work. `refreshMarkers` in
  `state/workbench.ts` fixes the marker lines and the prelude's indented example, which
  are course-authored and carry the word; the section bodies keep whatever they were
  saved with, so a file written before the rename still has "Chapter 3" in the picker
  above "Module 3" inside a docstring the course supplied. New readers get neither.
- **Attribution and license requirements must never be removed.** The app footer, README,
  and LICENSE carry the CC BY-NC 3.0 attribution to Michael A. Nielsen's _Neural Networks
  and Deep Learning_ (Determination Press, 2015). The **course content** follows its
  source and is non-commercial.
  **State that scope precisely, and do not widen it.** All three places used to say the
  project inherits CC BY-NC, which is broader than the facts and broader than the licence
  requires. CC BY-NC 3.0 has no ShareAlike clause, so matching it is a choice about the
  adapted content rather than an obligation that spreads; and the exercise harness, the
  Pyodide worker, the brand layer and `tools/` contain none of Nielsen's
  material, so nothing reaches them. Saying otherwise gives away rights over the one
  asset that transfers to every future course in the series.
  The split is now made rather than pending: the software is **MIT**
  (`LICENSE-MIT`), the course content stays
  CC BY-NC 3.0, and the series name and glyph are granted by neither. Chosen over
  Apache-2.0 because the repository already carries MIT code and all seven dependencies
  are MIT, so one licence covers every line and there is no compatibility surface; the
  patent grant Apache adds buys nothing on a test harness. Chosen over copyleft because
  reuse is the point.
  Nielsen's reference code is **MIT**, which requires its copyright and permission notice
  to travel with the code. That code is inlined into the Pyodide worker at build time, so
  `vite.config.ts` emits `LICENSE.txt` into `dist/` and the footer links it. A build that
  ships the code without the notice is a licence defect, not a formatting one.
- **No em dashes in any user-facing prose.** Use commas, colons, or parentheses.
- **`src/brand/` is this course's copy of a layer that is now upstream.** The canonical
  copy is the kit's, in the series repository; this folder is downstream of it. So a change
  to the four shared files starts there and flows down, rather than being made here and
  mirrored: made here it is a fork, and nothing in either repository can see it, because
  the byte-equality guard went with the kit. What still catches a drifted palette is that
  both sides measure the same OKLCH arithmetic, `brand_palette.py --check` here and the
  series' own copy of it there. A change that belongs to this course alone belongs in
  `src/styles.css`, which loads after the brand layer and can override any of it. Two
  things a course does own inside the layer: `brand.ts` and the one `--accent` line.

## Global shape conventions (all Python must comply)

- Activations and inputs are **column vectors**: an input is shape `(784, 1)`, never
  `(784,)`. Batches stack samples as columns: `X` is `(784, m)`.
- `w[l]` (weights into layer l+1) has shape `(size of layer l+1, size of layer l)`, so a
  layer computes `sigmoid(w @ a + b)`. Biases are column vectors `(n, 1)`.
- Labels are one-hot column vectors `(10, 1)` for training, integer class ids for
  evaluation.
- These match Nielsen's conventions. Notation drift between chapters is a bug.

## Exercise tests

- All exercise tests must be **deterministic**: fixed seeds everywhere
  (`np.random.default_rng(seed)`), no wall-clock-dependent assertions.
- Tests must run inside Pyodide: pure Python + NumPy only, no packages beyond NumPy, no
  filesystem or network access at test time.
- Test failure messages are teaching content. They must say what was expected, what was
  received, and point at the likely misconception (for example: "expected shape (30, 1),
  got (30,): remember activations are column vectors").

## Pinned versions

- **Pyodide: 314.0.5** (CPython 3.14), loaded from
  `https://cdn.jsdelivr.net/pyodide/v314.0.5/full/`. The version string lives in
  `src/runtime/pyodideWorker.ts` (`PYODIDE_VERSION`). Do not bump without re-verifying
  the training-time envelope from Milestone 0.
- **React: 19.2.8**, **Vite: 8.2.2**, **CodeMirror: 6.0.2**,
  **@codemirror/lang-python: 6.2.1**, **KaTeX: 0.18.4** (all pinned exactly in
  `package.json`).

## Decisions

- **Chapter content is authored in TSX, not MDX** (decided in Milestone 1): prose beats
  are short and always interleaved with components, so MDX would add a dependency
  without saving friction. Exercise prompts and hints live in each exercise's
  `index.ts`.
- **The exercises are one growing file, not nine** (the workbench; decided after
  the course shipped, on the primary learner's ask for something that felt
  cohesive). `src/exercises/sections.json` is the section table both the app and
  the tools read: eleven sections in course order, the exact marker line, what
  each provides, what it requires, which given sections arrive with it, and which
  names its own tests examine directly. `src/state/workbenchDoc.ts` owns the
  format and holds the only marker regex; `tools/workbench.py` reads that regex
  out of it rather than restating it. Markers are metadata for editing and
  reporting, never for running: every run, download and panel takes the whole
  string, so a mangled marker degrades features and cannot break execution or
  lose code. Two sections are **given** (`given-cost`, `given-batch`), marked
  "written for you", which is what lets the file import NumPy and nothing else.
- **Exercise test contract**: tests import the learner's code via
  `from submission import ...`, which is now the whole document exec'd as one
  chapter. A section the learner has not touched is **lent**: the course sets its
  own copy of that section's names onto `submission` for the run, and the panel
  says what it borrowed. Never a name the target section owns, and never one the
  target's own tests examine (`checks` in the section table; Chapter 7's seam test
  is the only case). Test functions are named `test_*`, run in definition order,
  and fail by raising `AssertionError` with a teaching message; the first
  docstring line is the test's display title. Test fixtures are hardcoded
  literals (never regenerated at test time from random streams), so results cannot
  drift between NumPy versions. An `AssertionError` carries no line number, so a
  wrong upstream function cannot be located from the failure itself: on a failing
  run the panel runs the upstream sections' own suites, earliest first, and stops
  at the first that fails. That is sound because a section that breaks its
  dependants always breaks its own suite too.
- **Pretrained weights are gzipped JSON** (`pretrained_weights.json.gz`), not npz:
  the Chapter 2 diagram reads the weights in JS (weight-image patches, edge colors)
  and Python reads the same file for the payoff run, and JS has no npz parser.
  Regenerate with `tools/pretrain_weights.py` (needs NumPy).
- **Interactive Python snippets** (payoff runs, live toy training) go through the
  worker's `runPython` request: the snippet reads input via
  `json.loads(_args_json)`, may stream progress with `_js_report(json_string)`, and
  must evaluate to a JSON string. First-party snippets only.
- **Chapter 10 teaches applying the network to data of the learner's own**
  (added after the teaching review, and a change to the design doc's original
  non-goals): a second bundled dataset (Palmer penguins, CC0) that arrives the
  way data actually does, with words, holes, unequal classes and measurements
  245 times apart in scale. Its exercise is `standardize`, `one_hot` and
  `split`, and its panel shows the two things a practitioner needs to see:
  unscaled the network scores exactly the majority-class baseline, and a
  73.5 percent score hides a class it never predicts. Numbers regenerated by
  `python3 tools/bench_penguins.py`.
- **Chapter 9 assembles the program, and Chapter 10 ends the course.** Chapter 9 is
  where the learner writes `train` and `accuracy`, the course's only assessment of
  assembly, and the payoff it cashes out is the download: before `train` exists the
  file is fifteen definitions that nothing calls, and after it the same file trains
  a network from a Python prompt with NumPy as its only import. `m9-program` says
  that, because no page said it before and the panel's own button is the only place
  the file's name appears. Chapter 9 also names the two jobs nobody does by hand
  (automatic differentiation, modern optimizers) and hands off to Chapter 10.
  **The where-to-go-next list belongs to the last page in the course, whichever
  that is.** It ended Chapter 8, then Chapter 9, and Chapters 9 and 10 were both
  written while it sat in the middle: Chapter 9 closed with a reading list and then
  had to open that list by saying "Chapter 10 is still ahead of you", which is a page
  admitting it is in the wrong place. A reader who reaches an exit door takes it, and
  the primary learner reported skimming both chapters. It is now `m10-next`, last
  before the data credit, and the `.course-next-list` class is named for the course
  rather than for a chapter so that the next move costs nothing.
  Neither chapter follows a Nielsen chapter, so neither has a `<Recap>` or a
  "Go deeper" link; everything else about a chapter page applies to both. Both open
  on a number rather than on a summary of what came before, which is the same rule
  Chapter 6 was rewritten to: Chapter 9 on the file that does nothing, Chapter 10 on
  the 42.6 percent that is exactly the majority-class baseline.
- **`course_helpers.py` is now only what gets lent.** Its seven reference copies
  (`sigmoid`, `feedforward`, `sigmoid_prime`, `backprop`, `cross_entropy_delta`,
  `init_network`, `l2_step`) exist so a reader who opens Chapter 9 first gets a run
  rather than a `NameError`. Nothing imports from it any more: no skeleton, no
  solution, and no panel. Three test suites still reach into it from inside a test
  body, deliberately and with a comment saying why, because a correctness
  guarantee whose oracle shares the code under test is not a guarantee: Chapter 3's
  downhill checks, Chapter 5's gradient check and Chapter 7's. The prompts' play
  snippets also import from it by name (`from course import gradient`) for any
  name owned by a section their exercise does not require, because the scratch
  pad lends only the current section's requires closure and a snippet has to run
  for a reader who skipped a chapter.
- **A payoff panel waits for everything its projection runs, not only its own
  page's exercises.** A pass can be earned with borrowed names (open Chapter 3
  first and sgd goes green with `feedforward` on loan), but the panels exec the
  projection raw, with no lending, so a gate that asks only "did sgd pass?"
  hands the panel a file with no `feedforward` in it and the run dies on a bare
  `AttributeError`. `codeReady` therefore walks the requires closure plus the
  given sections that arrive with it (`withGivens`, mirrored by `with_givens`
  in `tools/workbench.py`), and a locked note names what is actually missing
  through one shared phrase map (`interactives/lockedBy.ts`) instead of a
  hand-kept list per panel that could claim to wait on a finished exercise.
- **A payoff panel that varies settings runs its whole grid on one press**
  (decided when the primary learner asked plainly: run all the runs, plot it
  all, drop the buttons). The weight-decay panel is the 2-by-3 grid its
  section argues from (two starts, lambda 0, 1 and 5) and the depth panel is
  a 2-by-2 (one hidden layer against four, under each squash); both used to
  offer the settings as chips and train one pair per press, which left the
  reader assembling the comparison from memory. Three same-day patches (a
  drawn-with status, a table accumulating one row per setting trained, a
  cache so nothing retrains) were explaining that design rather than fixing
  it: the chips were the problem, not the feedback around them. One button
  now trains every run, streaming each line as it finishes; hue carries one
  axis and dash pattern the other; the table lists every row; the only chips
  left switch what the chart shows, never what the next press trains. Each
  run draws its start and its shuffle from fresh fixed seeds, so the runs are
  byte-identical to the runs the old panels trained and every number the
  chapters quote from the benches holds. The snippets take a runs list, and
  check_panels passes each panel one mixed list that walks both of its code
  paths. The accepted cost is that a press is minutes rather than seconds,
  with Stop the way out.
- **Chapter 7 asks the learner to edit their own Chapter 5 backprop** (the BP1
  seam). It is the first time the course asks anyone to change working code, and a
  single file is the only design where that is a two-line edit rather than an
  impossibility. The prompt ships the two lines; there is deliberately no button
  that splices them, because a splice into a function the learner has written
  themselves is the one edit that could destroy work. The adapter written for them
  in Chapter 5 calls `backprop` with four arguments until a replacement BP1 is
  actually handed over, so nothing needs the edit before Chapter 7, and
  `test_backprop_takes_the_blame_argument` in the cross-entropy suite names the
  section and the lines when it is missing. `src/exercises/backprop/seam.py` is the
  course's copy of the post-edit state, used only by the checker to prove the edit
  keeps every Chapter 5 test green.

- **The visual identity is a shared series layer, not this course's stylesheet**
  (added after the course was finished, when a second course became likely). `src/brand/`
  holds an accent family of nine hues at one OKLCH lightness and chroma, four named type
  roles, and the masthead, tab strip, link and footer chrome. A sibling course copies four
  of the five files unchanged and edits `brand.ts` plus one `--accent` line. The
  alternative was leaving each course to style itself, which is what makes a set of pages
  look unrelated; the cost is that the shared files have two copies in two repositories,
  with the kit's upstream and this one downstream of it, and nothing in either can prove
  they still agree. See "Visual identity" below, and `course-kit/BRAND.md` in the series
  repository.
- **The method is extracted rather than described** (same change), **and it has since
  moved out of this repository.** The kit is this file with the neural networks taken out,
  plus the process that produced it (`METHOD.md`), the fifteen learner failures behind its
  rules (`CASEBOOK.md`), the design doc template, the brand, and six slash commands. It is
  meant to be dropped into an empty repo. Rejected: a second generic application scaffold,
  which would be a guess about a topic that does not exist yet. What the kit says instead
  is which of this repo's ~2,900 topic-free lines to crib, and from where.
  It lived here while it was still being refined by building this course, because every
  refinement was discovered by writing a chapter rather than by thinking about the kit, and
  moving it out earlier would have turned each of those into a pair of cross-repository
  changes. With a second course starting it belongs above both, so it is now at
  `course-kit/` in the series repository (`PavolC/moving-parts`), with its own licence
  travelling inside it. Two things this repo lost in the move and should not pretend to
  still have: the byte-equality guard on the brand files, and a local copy to edit when a
  chapter teaches the kit something new. The second one is now a pull request against the
  series.
- **The front page's chapter outline is rendered from the chapter registry, not from its own
  list** (same change). Chapters 9 and 10 were added after the outline was written and were
  missing from it, under a heading that said ten, so the only page showing the whole course
  showed eight of it. `COVERS` in `StartPage.tsx` is now a lookup keyed by chapter id, read
  through `MODULES`, and the heading counts the registry. A chapter with no entry still
  appears under its nav label, so the worst a gap can cost is a missing sentence.
  `tools/check_exercises.py` checks the same seam from the other side: every exercise's
  chapter id has to exist in the registry.

## Chapter authoring playbook (learned from the primary learner, follow it)

The primary learner has high-school algebra, Python but no NumPy before Chapter 1,
and no vectors, dot products, matrices, or calculus. Chapters 1-2 were rewritten to
this floor after direct feedback; every later chapter must be written to it too.

- **Numbers before notation.** Compute a concrete example by hand first, then name
  the operation and its shorthand. The dot product appeared only after multiplying
  and adding by hand; matrix-times-column only after doing multiply-and-add twice;
  gradients only after nudge-one-knob-and-measure. Define every symbol at first
  use (the norm bars, e, argmax-as-"most confident").
- **One continuous world.** The running story is the concert decision: x1 weather,
  x2 friend, go/stay, green/gold dots, and the three personalities (easygoing OR,
  picky AND, contrarian XOR). New material connects explicitly to prior artifacts
  (the truth table, the 2-2-1 network, the 11,935-number tally) instead of opening
  fresh abstractions.
- **Draw every conceptual jump.** Never ask the reader to imagine a picture; put
  the figure in the page (the concert plot, the 2-2-1 wiring diagram, the
  sigmoid-vs-step slider, the pixels-are-numbers digit). Before any interactive,
  the prose gives a how-to-read key (what the shading, lines, and colors mean);
  captions state counts and label what is and is not a neuron.
- **Tally explicitly.** Count parameters and costs in the text: nine numbers in the
  XOR net, 11,935 in 784-15-10, two cost evaluations per parameter per step. The
  tallies are load-bearing for motivation (why Chapter 3, why Chapter 5).
- **Succeed before failing.** The learner solves OR and AND before meeting XOR;
  "press Show a solution, break it one slider at a time, then rebuild" is a valid
  on-ramp for a fiddly interactive.
- **Diagram widths are standardized, two families, and every viewBox is tight
  to its ink.** A figure's viewBox comes from `fig(x, y, w, h)` in ModuleBits,
  which also hands `w` to CSS as `--fig-units`; every figure width in the
  stylesheet is derived from that, so no rule repeats a number that lives in
  the markup. Box-and-arrow diagrams (chain-ripple / chain-net / bp-diagram /
  ripple-slice) render at one scale for the whole course, `--fig-scale`,
  calibrated so the widest of them fills the column exactly (Chapter 4's ripple
  log, 817 units across 873px): a diagram with fewer boxes in it is narrower,
  never differently drawn. Plot-family figures (concert-plot, tiny-net,
  shapes-diagram, curve-figure, m8-conv, m8-spaces) render at natural scale,
  one unit to one pixel, capped at their own width. Below 720px the box family
  stops shrinking at `--fig-scale-min` times its units and pans inside
  `.figure-scroll`.
  Padding a viewBox to center a narrower layout, which is what this rule used
  to say, is what made five diagrams claim up to 37 percent more column than
  they drew in: the ink was centered, but the figure reserved the width of the
  widest diagram in the course and started shrinking as if it were that wide,
  and a phone panned across the empty margins. Tight viewBoxes cost nothing on
  a wide screen and draw 35 figures larger at 700px, up to 44 percent.
  Measure a candidate with `svg.getBBox()` against its viewBox: the two should
  differ by 8 units a side. New diagrams join one of the two families.
- **A figure or an equation drawn on a card takes the card's ground, not the
  page's.** `--fig-ground` is the surface a figure sits on; the sideways-scroll
  covers and the SVG label halos read it, so anything with a background of its
  own (`.interactive`, `.module-aside`, `.bp-eqcard`, the cards) declares it to
  match. Two traps. A custom property that references another is substituted
  where it is **declared**, so `--x: var(--fig-ground)` at `:root` freezes the
  page's ground and inherits that down; read `var(--fig-ground)` on the element
  that paints. And an invalid `var()` makes the whole declaration invalid at
  computed-value time, so a `width` built that way silently becomes `auto`,
  which for an SVG with a viewBox is 100 percent: that scaled every plot-family
  figure to twice its drawn size, and only a before-and-after measurement of
  drawn width caught it.
- **A label on a wire needs clearance, not just a halo.** The halo (`paint-order:
stroke`) covers the line beside each glyph and not in the gaps between them,
  so a line running through a label still reads through it: Chapter 4's fork
  diagram showed "w = =4:0". Put the label clear of the line (the offset moves
  the text's _baseline_, and its digits stand about 9 units above that, so a
  label below a line needs roughly 15 units and one above it needs 6), and draw
  every label in a pass after every line, or a later wire paints over an
  earlier label. Where wires cross there is no position that clears them all,
  and the halo has to carry it: give it the width of the thickest wire.
- **A box drawn around a label has to fit the label with room to spare.**
  `tools`-style checks live in the session, not the repo, but the measurement is
  cheap: compare each `<text>`'s bbox against the `<rect>` its centre falls in
  and require 4 units of room. Five box captions were 1 to 3 units wider than
  their own boxes, and a caption that merely fits still reads as clipped.
- **A score gets a breakdown.** Accuracy alone hides which class a network
  fails at, and the course teaches the habit twice: Chapter 5's training panel
  shows per-digit counts and the eight mistakes the network was surest about
  (drawn from the test images it sent back), and Chapter 10 shows a 73.5 percent
  score that never once answers the minority class. A new panel that reports one
  number should report the breakdown beside it.
- **Interactives must not jump.** Fixed-basis flex columns (wrap depends on window
  width, never content), reserved heights for changing status text (.status-fixed),
  statuses on their own full-width line inside control rows, range inputs allowed
  to shrink (min-width: 0). Live readouts tie manipulation to meaning (the dragged
  line shows its w1, w2, b).
- **Exercises are visible.** The Output panel shows everything printed (worker logs
  are tagged runtime vs stdout); "Run my code" executes the editor without tests;
  the test code is viewable in a collapsible; prompts include a concrete
  "Run my code" experiment that ties back to an earlier chapter's numbers,
  shipped as a code block with Copy and Append-to-my-code buttons (a prompt
  entry of `{ code: "..." }`), never as code woven into a prose sentence.
- **Notation down to the punctuation.** Anything that could read as a typo is
  notation to explain at first use: the trailing comma in shape `(n,)`, the bare
  dot in `1.`, the `@` operator, slice colons. Python and NumPy idioms count as
  much as math symbols do. Also flag order reversals explicitly: weight shapes
  name the receiving layer first, so data flowing 2 -> 1 gives a (1, 2) matrix;
  teach the check "the inner numbers must touch" ((1, 2) @ (2, 1) -> (1, 1)).
  A symbol also needs its spoken name, not just its meaning, whenever code
  borrows that name: the course said the gradient is written with an
  upside-down delta but never that the symbol is read "nabla", so the
  `nabla_w` / `nabla_b` in every later prompt and skeleton looked like an
  unrelated word. Name the glyph where it first appears, then bridge to the
  code names at their own first use.
- **Quoted measurements must come from the code the reader runs, and the bench
  that produces them is committed.** `tools/bench_depth.py` and
  `npm run bench:speeds` regenerate every number Chapters 7 and 8 quote, each
  section printing the prose sentence it backs; extend one of them rather than
  writing a throwaway. A chapter's numbers can come from two different engines
  (the Pyodide panels use NumPy's PCG64, the layer-speed panel its own
  mulberry32), and they are not meant to agree: never quote a number from one
  engine for a measurement the reader makes with the other, and say which panel
  a table came from when both are on the page. Prefer statistics that hold
  still: the mean and the middle 90 percent of 200 draws reproduce, the extremes
  of that same stream do not. The bench has to match the browser exactly, not
  just mathematically. Two traps, both found while writing Chapter 8: `init_network` draws every weight and then every bias,
  so a bench that interleaves them builds a different network from the same
  seed; and `batch_gradient` sums per-example gradients in a loop, so a
  vectorized bench rounds differently. The second one matters because a deep
  sigmoid network amplifies it: two mathematically identical runs agree for
  about seven epochs and then drift a point or two apart. Where a run is still
  moving at the end, quote an average over the last several epochs (Chapter 7's
  regularization section and Chapter 8's depth tables both do) rather than a
  single epoch's score that will not reproduce.
- **No unexplained constants.** Every number in the prose is either derived in
  front of the reader, quoted from an earlier chapter, or explicitly labeled a
  free design choice with its trade-off (hidden layer size: more detectors vs
  slower training). An unexplained "15" or an unestablished "the nine numbers"
  is a bug.
- **Arithmetic goes in display math.** Any tally with more than two terms becomes
  an Eq with `\underbrace` labels (the 11,935 parameter count), never a prose
  sentence of times-and-plus.
- **One inference per sentence.** "Reshape its row and you get a picture, red
  pushes it up" compressed three steps and lost the learner. Unpack chains:
  each weight belongs to one pixel; therefore the weights can be drawn as an
  image; and the colors mean excite / suppress / ignore.
- **Say the implicit connections.** If the course knows a fact the learner will
  wonder about, state it where they will wonder: the XOR playground's dots ARE
  the contrarian's dataset; "Show a solution" sets exactly the table's numbers,
  with h1 as the at-least-one detector.
- **Recognize, do not rebuild.** The primary learner asked how the course gets
  from "a neuron is a straight line trying to divide data up" to "pairs of them
  with sigmoids making bars", and the answer was that Chapter 6 had been
  rebuilding three things he already owned. The switchover at -b/w is Chapter 1's
  decision boundary with one input removed (the line has nowhere to run, so it
  shrinks to a point). The bump is Chapter 1's XOR network with the squash taken
  off: two hidden neurons stepping at 0.5 and 1.5 along the total x1 + x2, wires
  out carrying +8 and -8, low then high then low. The tower's thresholding
  neuron is Chapter 1's output neuron doing what its bias of -4 did. A chapter
  that re-derives an artifact the learner built by hand says so, with the
  numbers restated, because a reader who is not told sees new machinery and asks
  why the ground moved. Watch for silent axis swaps in the same way: Chapter 1
  plots input space with a second input up the page, Chapter 6 plots one input
  with the output up, and the identical-looking square means two different
  things.
- **A callback earns prose only if it removes work or carries the argument.**
  Removing work means the reader has nothing new to learn because it is the same
  object (the three identities above). Carrying the argument means the page's
  conclusion depends on it: Chapter 6's bars, towers and boxes are one lookup
  table growing in dimension, which is why the box network fits every training
  image and reads nothing new, so "the weights now hold a lookup table" belongs
  in the bars section and pays off at the boxes. A callback that only says
  "remember this from before" costs attention and returns nothing; cut it. Two
  callbacks to the same earlier figure inside one section are clutter. Density
  check: callbacks per paragraph across Chapters 2 to 8 runs 0.4 to 1.2, so a
  chapter below that band is under-connected rather than at risk of pointing
  backwards too often.
- **One anatomy, stated everywhere ownership comes up.** Weights live on wires
  (exactly one per wire into a neuron: the wire is where the multiplication
  happens); biases live in neurons (the adding, the bias, and the squash are
  the neuron's own). The matrix W is the wire ledger: row = one neuron's
  incoming wires, column = one input's outgoing wires, transpose = regroup by
  sender. A weight plays two roles depending on what wiggles: booth for
  changes passing through, knob when nudged itself. The learner asked "why is
  the weight on the wire and not the neuron?"; any prose that files weights
  with neurons (counts, matrix rows) must reconcile against this anatomy in
  place.
- **A hand-built network gets its wiring drawn, and its parts named the way
  Chapters 1 to 5 name them.** Chapter 6 built its bump out of "two steps" and
  gave "the output neuron the weights +6 and -6", with no wiring diagram in
  the section; the primary learner read it and asked "is this a single neuron
  per layer? what are we talking about here?". Two bugs behind one question.
  The prose never said that the two steps are two neurons side by side in one
  hidden layer, so the shape had to be inferred from a plot of curves: a
  chapter that places numbers by hand shows the network those numbers live in,
  in Chapter 1's tiny-net idiom (gray input circles, green-bordered neurons,
  one weight per arrow, a two-line caption under each column, biases beside
  the neuron that holds them). And "output weight" was a coined term the
  course never establishes, filing weights with a neuron; the established
  phrasing is the wire's ("the wire out of h1 carries +6"). Before coining a
  noun, check what the earlier chapters already call the thing, and check the
  reverse too: Chapter 6 had "band" meaning a step's switchover in one section
  and a strip of the concert plane in another.
- **Derive at the size of the step, and suspect a missing picture when a
  count will not land.** Two findings from one Chapter 8 exchange. The
  fencepost 24 (a 5-wide window on a 28-wide image) did need deriving, but in
  one clause ("one further and the 5-wide window would run past the image's
  edge"); the column-by-column walkthrough written first was read back as
  overexplained. And the twenty windows stayed opaque through two prose
  patches because the learner's real question was architectural ("how many
  layers are we talking about", are twenty windows twenty layers?), which no
  sentence about the count could settle: it took drawing the whole network
  (ConvNetFigure: image, window layer, pooling layer, the tail from Chapter 5,
  a bracket marking the only new part). When a number that names a component
  keeps not landing, the missing thing is usually the assembly it belongs to,
  drawn.
- **Mind vocabulary collisions.** "Line" means the decision boundary in this
  course; never reuse it for an equation or a row of code. Prefer short
  sentences over connective-heavy ones; wordiness reads as weirdness. A word
  must also not appear before the section that defines it: Chapter 6's bump
  payoff said "outside its own slice" one section ahead of "Slice the dial into
  equal pieces", so it leaned on a word the reader did not have yet.
- **Interactives carry the algorithm; equations recap it.** Chapter 4's first
  draft put nine equations and 1,200 words before its centerpiece stepper, and
  the learner reported it "over my head". The fix that worked: teach the one
  genuinely new concept in prose with concrete numbers, reach the interactive by
  the midpoint, and present the formal equations after it as "what you just
  watched, written down". Recap means instantiated: open by saying the job is
  recognition, not derivation; write the equations with the interactive's
  concrete layer numbers (general indices wait until the implementation loop
  needs them); pair each equation with the number the learner already computed
  (a receipts table); and give direction-of-use and sign-reading their own
  explicit beat, since neither is visible in the symbols. Notation is taught just-in-time, one symbol at the
  moment it first appears (an equation gloss or a step card is a fine place),
  never as a several-notations dump paragraph.
- **Log first, explain second.** For any multi-stage numeric process, show the
  full log of concrete values (before, after, change) as one figure or table
  FIRST, then explain each stage as a rule that predicts the next logged number
  from the previous one. Claims-first prose ("nudge w and z moves by x times
  that amount") floats past this learner; prediction-against-a-log lands. Run
  checks in the multiplying direction (factor times change = next change), not
  the dividing direction.
- **Departures wear the Aside box.** Anything that pauses the main thread (a
  borrowed analogy, a scope note, a why-digression like Chapter 2's
  loop-vs-matrix) goes in the shared <Aside> component from ModuleBits, never
  in a long parenthetical; the shaded box tells the reader the lesson pauses
  and resumes. Models: Chapter 3's nudge-size trade-off, Chapter 4's currency
  chain, Chapter 1's how-special-is-this-toy notes.
- **A borrowed mini-world is allowed when the home story has no carrier.** The
  concert world has no natural chain-of-conversions, so Chapter 4 teaches
  factors as posted exchange rates in a two-booth currency chain (a raise:
  euros -> dollars -> pesos), proven with the primary learner live, then maps
  back explicitly (raise = nudge, pesos = cost, factor = posted rate,
  through-rate = slope). Keep such a world to one beat plus one figure, map
  every element back by name, and let its vocabulary (booth, posted rate,
  through-rate) run through the section it serves.
- **Demonstrate on numbers where the mechanism is visible.** Chapter 4 first
  proved "the change comes out multiplied by the partner" on the hop whose
  partner was 1.0, and the learner read it as "changing by 0.01 changes it by
  0.01, so what?". A worked example whose value makes the key effect a no-op
  (multiply by 1, add 0, gap of 0) teaches that nothing happened. Lead with an
  instance where the effect shows (the times-2 wire), then explain the no-op
  value as the special case it is.
- **A chapter about a property, not a technique, must be motivated by the
  learner's own number and cashed out on the learner's own artifact.** Chapter
  6's first draft opened with "the question is what a network of these neurons
  can express", posed a curve-fitting problem in a new world (continuous input,
  a 0-to-10 rating, an unsquashed output, no training), and put its only
  connection to the course in the closing beat. The learner stopped five
  sections in: "we're just talking about curves..... why??". Chapters 1-5 each
  deliver a thing the learner then owns, so a chapter that delivers a fact has
  to work harder, in this order: open with a number the learner produced
  (Chapter 5's 89 percent), name the competing explanations for it, say which
  one this page settles and which it leaves standing, and then, at the end,
  spend the result back on the same artifact (the 7.8 million-neuron box
  network against the trained 30). Anything that looks like a departure
  (one input instead of 784, hand-placed weights instead of trained ones, no
  exercise) gets named as a deliberate shrink in the opener, with the reason,
  plus the sentence that the small case is a sub-case and not a detour: hold
  783 pixels still, turn one, and the outputs trace curves.
- **Three sentence shapes to keep out of chapter prose, all found by the
  primary learner in Chapter 6 ("dense, backwards, vague, clippy").** First,
  clefts and abstract-first openers: "What no part of the argument provides
  is...", "Nothing in that argument stops anywhere", "The claim is about
  relationships". They front a placeholder and delay the content, which reads
  as backwards. Give every sentence a concrete subject and a verb, in that
  order. Second, short pronoun aphorisms, especially as paragraph openers:
  Chapter 6's limits section had four consecutive paragraphs beginning "It
  covers...", "It promises...", "It says can, not will", with the antecedent
  five paragraphs away. Name the subject each time, even at the cost of a
  longer sentence. Third, meta-narration of the exposition ("this section
  makes X, the next turns it into Y", "now the question this page opened
  with", "shrink it to one input"): describe the thing, not the plan for
  describing it. The measurable signature, comparing against Chapters 1 to 5:
  cleft openers at or under 5 percent of sentences, pronoun aphorisms at or
  under 5 percent, median sentence length 19 to 23 words. Cutting words is
  not the fix when prose reads badly; Chapter 6 got clippy precisely because
  two rounds of cutting turned full sentences into telegraphese. Rejoin
  clauses instead.
- **State what a section buys before proving it.** Chapter 4's factor
  paragraphs read to the learner as "simple math, overexplained" until the
  stakes came first: every factor is computable from the first run alone, so
  the nudge and its second run become unnecessary. When a stretch of prose
  verifies something, open with the one-sentence payoff the verification
  earns, and close by cashing it out; checks without stated stakes read as
  arithmetic for its own sake.
- **Teach kinds, not instances.** When several facts repeat one pattern (Chapter
  4's five ripple factors), position-by-position derivations read as N separate
  proofs and lose the learner even when each line checks out. Instead: name the
  pattern once, framed in a concept already taught ("every factor is a slope,
  Chapter 3's kind of number"), sort the instances into their few kinds, and give
  each kind one plain-words why plus an extreme case (x = 0: the nudge dies
  here). Color-code the kinds in the figure so the picture carries the grouping.
- **Multi-factor arithmetic never sits inline.** Chains like 0.235 x 2.0 x
  (-0.108) inside a sentence are the densest thing on a page. Put them in
  display math with \underbrace labels naming each factor, or in a small table
  (knob, value, read off as). This is the "arithmetic goes in display math"
  rule applied to products, and it makes pages FEEL lighter even when the
  equation count rises.
- **Interactives carry their own keys; prose then shrinks.** Put each chart's
  how-to-read key on the interactive itself (section titles, one-line legends at
  the point of use). The prose before it keeps only what sections cannot say:
  the connection to earlier material and the single carrying idea. Prose that
  narrates an interactive's own labels is duplication to delete.
- **Assume weeks pass between chapters.** Never lean on a bare name from an
  earlier chapter (fire, the contrarian, the nine numbers) as a load-bearing
  reference; either restate the thing in a few plain words or drop the
  callback entirely. Callbacks are seasoning, not structure.
- Housekeeping: learners paste rendered math as doubled text ("x1x1", "WW").
  That is a copy artifact of KaTeX's accessibility markup, not a rendering bug;
  do not chase it.
- **Every new symbol goes in the notation reference.** The start page carries a
  folded lookup (`NOTATION` in `src/start/StartPage.tsx`): symbol, one line of
  meaning, the field's name for it (`also`, rendered as a muted line under the
  meaning rather than as a fourth column, which would pan the whole lookup at the
  measure), and the chapter that introduced it, in the order a reader meets them.
  The course's own rule is that weeks pass between chapters, and a symbol defined
  once four thousand words ago is not defined for that reader. Introducing
  notation in a chapter means adding a row there in the same change.
- **A coined word hands over to the field's word in the chapter that earned it, not
  at the end of the course.** Every chapter from 1 to 8 carries a naming note: one
  short paragraph, at the first use of the thing, saying what everyone else calls
  it, after which both words are in play. The note is deliberately not labelled as
  one, and does not wear an `<Aside>`. Seven of them opened with "this chapter's
  naming note is" in the first draft, which is both meta-narration of the exposition
  and a formula a reader learns to skip after the second one; and a shaded box says
  the lesson pauses here, which for the one paragraph whose whole job is to put a
  word into the reader's working vocabulary is the wrong signal. Each note leads with
  its content instead. The squash becomes the activation
  function and the line becomes the decision boundary (Chapter 1), `a` becomes an
  activation and `W` a weight matrix (Chapter 2), a knob becomes a parameter, the
  cost the loss and one knob's slope a partial derivative (Chapter 3), a factor
  becomes a local derivative and blame the error (Chapter 4), receipts become the
  cached forward pass (Chapter 5), the divided start becomes Xavier initialization
  and the decay factor L2 regularization (Chapter 7), learning speed becomes the
  gradient norm and a hop under 1 the vanishing gradient problem (Chapter 8).
  The primary learner read the course and reported that the terms arrive as "a big
  brain dump of all of these terms that I now need to swap in my head". The dump
  was a twenty-row table in Chapter 9, at the point of lowest energy in the course
  and after the last exercise, asking for the highest-effort operation in it:
  re-index twenty concepts learned under other names. Worse, the course was already
  bilingual and never said so. The equation glosses used the field's words while the
  prose beside them used the coined ones ("the weight matrix" in Chapter 2's gloss
  for `a' = sigma(Wa+b)`, "the layer's wire ledger" twenty lines below it), and
  before Chapter 9 the two phrases appeared seven times each. So the field's words
  were arriving unlabelled either way; the handover only declares an equivalence
  the reader was already being shown.
  Three tiers, and the tier decides how much prose changes. **Switch**: the coined
  word was a stand-in for a name on page one of everything, so the field's word
  becomes primary in the formal registers (equation glosses, `<Recap>` items,
  `AfterThis`, exercise prompts) while the prose keeps the plain word wherever it is
  carrying the intuition. **Run both**: the plain word is why the idea is
  comprehensible, so it stays primary and the field's word rides along in equations
  and code (blame and the error, cost and loss). **Local only**: scaffolding for one
  beat with no counterpart anywhere (booths, posted rates, the through-rate, the
  dial, bars and towers and boxes, the hop). Chapter 9's table is now the inverted
  one, six rows long: these are ours, do not go looking for them. Do not replace a
  coined word everywhere downstream of its handover; two rounds of that is what made
  Chapter 6 read as telegraphese, and `knob` alone has 135 uses.
- **Skeleton docstrings freeze into saved code.** The editor persists the
  learner's copy, so improving a skeleton never reaches anyone who already
  opened the exercise. Anything essential to understanding the contract must
  (also) live in the prompt, which always renders current; the skeleton
  docstring is a convenience copy, not the canonical explanation.

## Content voice rules

- Prose beats are short (150-400 words), one idea each, written to sit beside an
  interactive or editor.
- Plain, direct, second person, no hype. Define every symbol at first use.
- Math rendered with KaTeX; every equation gets a one-sentence plain-language gloss
  immediately after it.
- Each chapter opens with "What you'll be able to do after this" (2-3 items) and closes
  with a recap and a "Go deeper" link to the corresponding Nielsen chapter.
- Chapter bodies are broken into sections with <SectionHeader id title /> markers
  (ids unique across chapters, prefixed "c4-"), and each chapter mounts <ModuleToc />
  once after its AfterThis block: a floating on-this-page nav in the right gutter
  (self-hides on narrow screens) that discovers that chapter's headers from the DOM
  and scrollspies them. Section titles are short noun phrases, 5-8 sections per
  chapter.

### Register: plotted, narrator muted (adopted after Chapter 4; applies to all chapters)

Chapters are plotted like a story (setup, tally, payoff, callbacks): keep that. What
must go is the audible narrator, the voice that sells, promises, and points at its
own storytelling. Concretely:

- Motivate with numbers, not verdicts. Let 23,870 be the drama. A judgment is allowed
  once, in plain words ("training it this way stops being realistic"), never as a
  punchline ("the bill is fatal") and never twice for rhythm.
- No promises about the reader's future experience ("it will be short", "you will
  beat it later", "should take you a few seconds").
- No stage directions that command attention or feelings ("watch the clock, and
  count", "remember this feeling", "feel this rule"). Directing perception at content
  is fine ("watch how the steps shrink as the ground flattens").
- Never narrate the course's own storytelling ("and that is foreshadowing"). Devices
  stay, labels go: a callback works without being announced.
- No flattery or possession theatrics ("Your sgd is real").
- No moralized vocabulary for algorithms ("the honest way", "SGD cheats").
- Replace aphorisms with their literal content. "Everything else is bookkeeping"
  became "the rest is arranging the multiplications so no factor is computed twice",
  which teaches more. Punchlines compress by discarding information.
- Test for a borderline sentence: could it appear unchanged in a careful colleague's
  explanation email? "The bill is fatal" fails; "stops being realistic" passes.
- Exception: a chapter's final beat may carry one slightly hot sentence. Endings do
  gating work in a self-paced course, and they are the two-minute-demo moments.

## Visual identity

The series brand layer (`src/brand/`, documented in the series kit's `BRAND.md`) owns the accent
family, the four type roles, the label idiom, the masthead, the tab strip, the links and
the footer. Rules that follow from it:

- **Anything derived from the accent is a token, never a literal.** Alpha tints are
  `color-mix(in srgb, var(--accent) N%, transparent)`, which is what `rgba()` was doing;
  surface tints are `--accent-wash` (6 percent), `--accent-panel` (14 percent) and
  `--accent-rule` (30 percent), mixed in oklab. Neutral near-whites may stay literal,
  because a sibling course changing its hue must not repaint them.
- **The accent family is computed, not picked.** Nine hues at one OKLCH lightness and
  chroma, so no course can be louder than its neighbours and none can fail contrast.
  `python3 tools/brand_palette.py --check` fails if `brand.css` has drifted from what the
  arithmetic gives.
- **Faces are named, not spelled.** `var(--font-prose)` for anything read in sentences,
  `var(--font-ui)` for chrome, `var(--font-display)` for the wordmark and labels,
  `var(--font-mono)` for code. A literal font stack in `styles.css` is a bug: it makes
  `--font-ui` a decoration rather than a switch.
- **The series mark and the course mark have separate jobs.** `Monogram.tsx` draws the
  three-band Moving Parts mark beside the series name in the masthead and footer. The
  sigmoid belongs to Neural Networks, so `COURSE.glyph` identifies the course in
  `index.html`'s favicon and `tools/og_card.html`; the series index repeats it on the
  course card. A mark and the name it identifies always form one horizontal lockup;
  descriptors and taglines sit outside that pair. `theme-color` is also a literal because
  browser chrome needs it before JavaScript runs. `python3 tools/check_brand.py` checks
  both marks and the literals it can reach from this repository.
- **The social card is a rendered page, and its URLs are absolute.** A share is unfurled
  by a crawler that has no page to resolve a relative path against, so `og:image`,
  `og:url`, `twitter:image` and the canonical link all spell out the deployed origin,
  which is the one place `base: "./"` cannot save. `tools/og_card.html` is the card's
  source, built from the same accent, course glyph and type roles as the site, so a rebrand
  reaches it; `bash tools/make_og_image.sh` redraws it and `check_brand.py` fails if the
  four URLs disagree, if the image named is missing from `public/`, or if it is not the
  1200x630 the tags declare. A course that ships without this is not unstyled, it is
  invisible: every share of it renders as a line of grey text.
- Data hues (`--acc`, `--loss`, `--data-*`) are not brand colours and do not follow the
  accent. They name quantities in charts, and a course that changes its accent must not
  silently repaint its charts.
- **One axis, and everything is centred on it.** A line of prose runs to `--measure`
  (34rem, about 69 characters at the 19px root) and is centred in the column, not
  left-aligned in it. Left-aligned was the first attempt and it was wrong: every block
  wider than the prose (an equation, a figure, a caption, a table) hung off to its right
  and the page read as lopsided. Centred, the prose, the equations and the figures share
  one centre line, so an equation sits dead centre over the paragraph that introduces it.
  Headings, captions and the masthead are in the set too; left out, they started at the
  column's left edge while the prose under them started 113px further in, which reads as a
  mistake rather than as a wider header.
- **The column does not shift.** It is centred at every width. It used to sit 100px left of
  centre above 1200px to open a gutter for the on-this-page nav, and the cost was that
  every page looked mis-centred whether it had that nav or not: the front page never does,
  so a first visit landed on one that did. The nav now needs 1400px, where a centred column
  leaves room beside it, and below that the sticky section bar takes over as it already did.
- **The selectors that apply the measure are all direct-child of the reading column**
  (`.module > p`, `.exercise > p`), which is what makes it safe in a stylesheet with
  twenty-five interactives: a paragraph inside a panel is not a child of the article and
  never sees it. A rule further down the file that sets a `margin` shorthand resets the
  horizontal `auto`, so those write `margin: X auto Y` themselves.
- **Two things keep the column's full width, and only two:** the figures and panels
  themselves, and the rules under the tab strip and above the footer. The tab strip starts
  on the text's axis and runs on to the column's right edge, because the left edge is where
  alignment is read and eleven tabs inset on both sides wrap to three rows instead of two.
  It wraps to two rows down to 881px and pans as one row below that. The breakpoint is
  880 rather than the 854 where the third row actually appears, and the 26px is headroom:
  the labels are set in `system-ui`, a different face on every platform, and a sans a few
  percent wider moves that boundary up into the band. Panning early costs one row;
  getting it wrong the other way puts a three-row wall above the course. Shrinking the
  labels instead does not reach (at 0.72rem with tight gaps a 390px phone still takes four
  rows, 167px of navigation). Where it pans it has no scrollbar of its own: a thin thumb
  lands a second grey bar right under the active tab's accent rule and reads as a
  mis-drawn underline, so the edge fades carry the affordance alone, the way a native
  phone tab strip does.
- **The shared scroll affordance must leave nothing behind on a container that has
  nothing to scroll.** `.scroll-x` is the two-signal treatment (a thin scrollbar and edge
  shadows) built the usual way: covers painted with the content, shadows painted with the
  box, so the covers sit exactly on the shadows at rest and slide off once an edge is
  scrolled past. Two details are load-bearing and both were found by pixel-differencing a
  render against the same render with the shadow layers made transparent. The cover holds
  its surface colour flat across the shadow's whole width before it starts to fade, and
  the shadow itself starts transparent and reaches full strength 4px in rather than at the
  edge, because on a 2x screen a box whose edges land on a half CSS pixel rounds the two
  layers to different device columns and leaves one column of shadow showing. That column
  was 29 levels darker than the page and read as a grey border down the side of every
  equation at 707px; it is 2 now. When touching this rule, re-run the difference test
  rather than eyeballing a screenshot: a downscaled composite invents lines that are not
  in the file, and this session chased two of them.
- **Two widths on a page, and only two, and the rule is about the ink, not the boxes.**
  The eye judges alignment by ragged left text edges, so everything that has one puts
  that edge on the prose axis at `--measure`: prose, headings, captions, the cards, the
  exercise's h4, the display equations, the blocks that hold code (the play snippets,
  the torch listings, Chapter 1's five-ideas block), and the start page's rows of
  buttons. The full column belongs only to ink that centres or to boxes that show their
  own extent: figures, tables, interactive panels, the editor. A centred figure reads as
  a picture sized to its content however wide it is; code and buttons cannot centre
  their ink, so classed as illustrations they started it at the column edge, 100 to
  113px left of every paragraph, which is the lopsidedness the centred axis was chosen
  to prevent. The boxes all obeyed the two widths while that was true, which is why the
  rule is stated in ink: a wide box whose ink cannot centre is misfiled however correct
  its container measures. Equations used to have a third width in between,
  `--measure-wide` at 42rem, and three widths read as an accident rather than as a rule.
  A card's box is `--measure` plus its padding AND its borders, the 3px accent rule
  included, so the lines inside it land on the axis at the measure like every other
  line; the notation lookup and the code blocks carry their own paddings the same way.
- **An equation that does not fit at the measure is split, not widened.** 45 of the
  course's 51 already fit; the seven that did not (`m2`'s 11,935 tally, `m3`'s cost and
  gradient, `m4`'s deltas and its new-z2 difference, `m7`'s blame chain and its
  `(w+h)^2` expansion) are now `aligned` or `gathered` over two or three lines, which is
  what a typesetter does with a long `\underbrace` tally anyway. Nothing scrolls above a
  665px viewport, where the column itself drops below the widest equation's 617px;
  below that, display math cannot re-wrap and scrolling is the only option left. Measure
  a candidate with `.eq-scroll { width: 1px }` and read `scrollWidth`: that is the
  natural width, and the box to beat is 646px.
- **Below 880px the tab strip is replaced by a picker, not panned.** Eleven tabs
  wrap to two rows down to 881px; below that the strip cannot show them without
  panning, and dragging a row sideways is the worst way to offer a list of
  eleven things (the active tab is centred, so a phone shows three of them and
  the reader has to discover the rest). `.module-picker` is the same list folded
  into one line, in the same idiom as the course's own section bar, and neither
  navigation pans at any width. Wrapping instead was measured and does not
  reach: a 390px phone takes six rows, 306px of navigation, and four rows even
  at 0.72rem with tight gaps.
- **The masthead compresses on inner pages at phone width.** `Masthead` takes a
  `compact` flag (App passes it for every page but the start page) and below
  560px that drops the tagline and sizes the title down to a running head. The
  tagline summarizes the course's concrete arc, which belongs on the front
  door; on page seven it costs four lines. Measured on Chapter 2 at 390x844, the whole
  first-screenful chrome was 329px and the first line of prose sat at 892px, off
  the bottom of the screen: it is 459px now, and no chapter is past 600px.
- **The chapter opener folds on a phone.** `AfterThis` is a `<details>`, open
  everywhere except below 560px, where three items ran to between 250 and 390px
  and Chapter 8's card alone was 46 percent of the first screenful. Folded, the
  chapter still opens by naming what it will teach. The state is read once at
  mount, not watched, so a reader who opens it does not have it shut again by a
  rotation. A `<summary>` is `display: list-item`, and `display: flex` on it
  (which is a tempting way to centre it in a 44px touch target) removes the
  disclosure marker and leaves a heading nobody would think to tap: use padding.
- **In-page jumps are instant, and that is correctness, not taste.** A browser's
  smooth scroll animates toward the offset it computed when it started, and
  these pages mount their panels as they come into view: measured on a 390px
  screen, the document grew 971px in flight and the heading landed 189px above
  the viewport, repeatably, while the same jump at 1440px was exact. Section
  headings also need `scroll-margin-top` clearing the sticky section bar (68px
  below 1400px, where the bar exists); at 24px the heading landed behind the bar
  it was chosen from.
- **Touch targets are 44px below 880px**, the band where the picker takes over.
  The worst of it was the range inputs: an unstyled one is 16px tall and
  twenty-two of them drive the interactives. `min-height` on the input grows its
  box and the browser keeps the track centred, so the affordance looks the same
  and the whole 44px box is live.
- **The exercise's Run buttons sit directly under the editor on a phone.** The
  tip explaining them is twelve lines there, so in source order it put 330px of
  reading inside the course's tightest loop: edit, run, read the output. The
  three that move take negative flex orders, so the statuses and the output
  panel keep the default 0 and follow in source order; a catch-all `:not()` rule
  was tried and lost to its own specificity, which put the editor last.
- **A course links up to the series index, never across to a sibling.** `SERIES.homeUrl`
  is set once and never touched; the index is the only thing that knows what else exists.
  There is deliberately no list of courses in `brand.ts`: carrying one would mean editing
  and redeploying every course's repository on each new course, which is the same
  hand-maintained-list failure as the front page's chapter outline, once per repository.
- **A section title carries a short accent rule above it.** It is the one piece of
  structural furniture a sibling course inherits already recoloured, and the start page's
  plain h3s wear it too, so a reader cannot tell which sections are scrollspied.
- **Hover is declared per variant, never on `button`.** A bare `button:hover` has higher
  specificity than `.tab:hover` or `.module-toc-item:hover`, so it fills every variant that
  sets its own transparent background: that is a bug this course has already shipped once.
  The primary treatment is reached as `button:not([class])`, which cannot match a variant,
  because every variant here carries a class.
- **The panel is one scroll, not two.** It used to be an editor with its own
  scroller above a results pane with its own scroller: two keyholes, two
  scrollbars, and neither half able to use the other's space. Now the chrome
  (head, section rail, controls) is fixed at the top and everything below it
  flows down a single column, code then output then verdict, the way a page
  does. The editor has no height of its own (`height: auto`, and
  `overflow: visible` on `.cm-scroller`), so it grows to its content and the
  panel is what scrolls. CodeMirror still virtualizes against the scroll
  ancestor: measured on the full 507-line file, 49 of those lines are in the
  DOM at a time and the whole 14,000px scrolls in under a second.
  Two consequences. The controls are panel chrome rather than something inside
  the scroll, so they are on screen however deep in the file the reader is; and
  a finished run scrolls its results into view, because the answer would
  otherwise land at the bottom of a column thousands of pixels long. Nothing
  else in the panel is allowed to move the reader.
- **One chrome row, one Run button, and a passing check on one line.** The
  panel's chrome was three rows: a title with Download and Close, a rail of
  section chips, and a controls row. It is one now, 54px against 151: the
  file's name, the one button the loop uses, a picker naming the section that
  button is pointed at, the run's status, and a More disclosure holding
  Download, Reset and Undo.
  **The second run button moved onto the scratch pad**, which is the thing it
  actually runs. "Run my code" sat beside "Run tests" as if the two were a
  choice to make every time, when one is the loop and the other is an aside;
  the name also implied the other button did not run your code. They are not
  merged, and should not be: the tests are the expensive path (Chapter 5's
  gradient check nudges 54 parameters twice), so making every print-a-value
  experiment pay for them would be a real regression. A cell owning its own run
  button is the notebook habit worth copying. Sending a snippet from a prompt
  opens the scratch pad and scrolls to it, because code sent somewhere the
  reader cannot see was not sent.
  **What the run is saying gets a line of its own**, under the head rather than
  in it. In the head it was the sixth thing across the row and about 50px wide,
  so "Running tests..." rendered as "R...". The strip is 31px and is there only
  during a run, so the chrome at rest is still one row, and the code moves by
  that much when a run starts, which is the smallest price that leaves the
  message readable.
  Passing checks fold into "N checks passed": six of them at full size pushed
  the failures and the output most of a screen down, and a receipt is not a
  finding. Failures never fold. The borrowed-names line appears only when
  something really was borrowed: "run entirely on your own code" is the
  ordinary case, and a line saying nothing happened is noise under every
  passing run. Output is one pool, headed with the section the
  run was for, because one run execs the whole file and pretending otherwise
  would be a lie about what just happened.
- **The workbench borrows three things from a notebook, and not the fourth.**
  Mod-Enter runs the tests and Shift-Enter runs the scratch pad, because that is
  what anybody who has used a notebook reaches for first; both need
  `Prec.highest`, since `basicSetup` binds Mod-Enter to "insert a blank line"
  and silently swallowed it. Every section line carries a run marker in the
  gutter, so the control for "run this piece" sits next to the piece. The
  picker carries each section's state, which is what a notebook's execution
  count is for. What is deliberately not borrowed is independent cells: this is one
  file that runs top to bottom and downloads as an `nn.py`, and per-cell state
  would cost exactly the thing the whole design is for.
- **The section bar and the panel head share one height, `--bar-h`.** Both sit
  at the top of the screen when the panel is docked, so their bottom rules have
  to land on one line. They were 53.4px and 57.1px, and four pixels apart reads
  as a misalignment rather than as two different things. The in-page jump
  offset is keyed to the same token, so the offset that clears the bar cannot
  drift from the bar.
- **The section list is a picker, not a rail.** Eleven chips in a row that
  pans showed four of them at the dock's own width, which is a keyhole onto the
  course rather than a picture of it, and it cost 50px of every screen to do
  that. The same eleven, with the same state marks, now open from the head: all
  of them at once, and no permanent height. Three details are load-bearing. The
  menu is anchored to the panel rather than to its own summary, because the
  summary sits after the file's name and the Run button, so a menu hung off its
  left edge ran 163px past the panel's right edge at the narrow dock. A flex
  item cannot shrink below its content without `min-width: 0`, so without it on
  `.wb-section-name` the longest label (38 characters) pushes the state clean
  off the menu. And a given section says whether it is in the file and nothing
  else, because it has no tests and "not passing yet" would be a lie about it.
  Below 600px of panel the picker takes a row of its own, asked of the panel
  with a container query and not of the window, because the reader sets that
  width by dragging. Everything else in the head is fixed width and comes to
  418px with the gaps and the padding, so under 600 there is nothing left to
  name a section in: measured, the picker was a 39px box at the narrowest dock
  and a 20px one on a 390px phone, showing one letter of the name, which is the
  one thing in that row a reader needs to read. The threshold counts Stop,
  which is only on screen during a run, so the head cannot change height under
  a reader mid-run.
  The chrome that remains is `flex: 0 0 auto`, the head and the repair strip:
  left shrinkable, in a column whose editor wants every pixel, Firefox
  collapsed the old rail to the height of its own scrollbar with nothing
  visible in it, and Chromium did not, which is why the rule is written down
  rather than trusted.
- **The hints and the test code live in the panel, beside the code they
  describe.** They sat in the chapter page under the prompt, which is where they
  were written before there was a panel: reference material for while you are
  coding, a scroll away from the code. The prompt stays in the reading column
  at the measure, because it is course prose and is read, and several hundred
  words at panel width beside prose at 646px is the "three widths read as an
  accident" failure. Nothing is re-parented in the move: the prompt paragraphs
  stay direct children of `.exercise`, so the stylesheet's measure rules keep
  matching them and nine exercises do not silently widen. The output panel's
  heading carries **Back to the code**, because one scroll means the code is
  above whatever a run just produced, sometimes thousands of pixels above.
- **The workbench is the one thing allowed to move the column, and only because
  the reader moved it.** `.shell` holds the reading column away from a fixed panel
  with `padding-right: var(--dock-w)`. Padding, not a grid track: a `1fr` track
  takes a min-content floor from the widest table in the course, so `minmax(0,1fr)`
  would be mandatory, and a grid item stretches by default, which silently defeats
  `position: sticky` on the section bar inside it. The dock opens at exactly the
  width left over beside a full-width column, so a first open moves nothing;
  dragging past that narrows the column, down to 752px, which is the widest thing
  in the measure set. Closed by default on a first visit, so the front page a
  stranger lands on is byte for byte the page it always was.
  Two consequences are load-bearing. `--fig-scale` stops being a literal: the
  shell publishes `--col-content` and `--fig-scale` on the root, computed as the
  measured content box over 817 units, which is the same calibration the comment
  always claimed and is now true at more than one width. And three rules that
  asked the window how wide it is (the gutter nav, the sticky bar, the offset that
  clears it) move onto one `data-toc` predicate that asks how much room is left
  beside the column, keyed together so they cannot disagree; the tab strip's fold
  to the picker gets the same treatment through `data-narrow`, overriding
  `brand.css` from `styles.css` rather than editing the shared layer.
  Below 1360px there is no dock, only a modal sheet. That is half of what the
  panel is for, denied on small screens, and it is stated rather than dressed up:
  no width under 1360 fits prose at the measure and code at a readable column
  side by side.
- **The editor is themed from brand tokens, in `CodeEditor.tsx`.** Its chrome comes from
  the surfaces and the accent; its token colours come from the accent family, so syntax
  highlighting is legible by construction (every hue in that family clears 6:1 on the page
  ground) rather than by eye. Do not hand-pick a syntax colour: take one of the nine.
  **A background painted on a line is an alpha tint, never `--accent-wash` or
  `--accent-panel`.** Those two mix with the page ground, so they are opaque, and
  CodeMirror draws the selection in a layer behind the content: an opaque line
  background hides it completely. With the section highlight covering every line of
  the section the reader is working in, the selection was invisible exactly where
  they type. The selection rule also spells out the whole child chain CodeMirror's
  own base theme uses (`&.cm-focused > .cm-scroller > .cm-selectionLayer
.cm-selectionBackground`), because a shorter selector loses to it on specificity
  and the selection comes out in CodeMirror's default lavender.
  `tools/check_run_path.mjs` checks both from the browser.
- **KaTeX keeps its own face, and that is a choice.** Computer Modern is what mathematics
  is set in, and matching it to the prose serif would make the notation harder to read, not
  more consistent. It is the one type system on the page that is deliberately not the
  brand's.

Two things deliberately absent, so a future session does not treat them as oversights:
there is no dark mode (every colour is a token, so it is a later drop-in rather than a
rewrite, but the interactives carry dozens of hand-tuned SVG palettes that each need a
second reading), and the type scale is named for the chrome only. Rewriting every
`font-size` in this stylesheet is churn with no visible return; a new course uses the
tokens from the start.

## Repo layout

```
/                    README (what/why/how to run/license/attribution)
/CLAUDE.md           this file
/src/                React app
/src/brand/          the series brand layer, shared with sibling courses
/src/start/          the front door: what the course is, the outline, stored progress
/src/modules/NN/     one folder per chapter: content, interactives/
/src/exercises/      per exercise: skeleton.py, tests.py, solution.py, index.ts (prompt, hints)
/src/exercises/sections.json  the section table, read by the app AND by tools/
/src/exercises/given/         the two sections written for the learner
/src/python/         shared Python: harness, course helpers, gradient checker, data loader
/src/runtime/        Pyodide Web Worker, message protocol, shared worker client
/src/components/     shared UI: CodeEditor (CodeMirror), ExercisePage
/src/state/          the workbench document, its format, and progress persistence
/src/m0/             Milestone 0 training demo UI
/public/data/        mnist_subset.bin.gz, pretrained_weights.json.gz, penguins.json.gz
/tools/              build-time scripts (MNIST preprocessing, weight pretraining)
```

## Commands

- `npm run dev`: start the dev server.
- `npm run build`: static production build (deployable to any static host).
- `python3 tools/make_mnist_subset.py`: regenerate `public/data/mnist_subset.bin.gz`
  (pure stdlib, downloads MNIST from a public mirror, deterministic output).
- `npm run check`: the five checkers, which is what CI runs before it builds.
- `npm run check:doc`: the document format's own invariants, run against
  `src/state/workbenchDoc.ts` itself in Node. `check_exercises.py` can only test
  the documents this repo assembles; this tests the ones the editor produces.
  The difference is where a bug was: replacing a section put its text back
  trimmed, so the next marker landed at the end of the previous section's last
  line, where the anchored regex cannot see it, and that section vanished into
  its neighbour with no error anywhere. `workbenchDoc.ts` therefore imports
  nothing a bundler has to resolve, which is why the prelude lives in
  `workbench.ts` and `assemble()` takes it as an argument.
- `python3 tools/check_exercises.py`: the workbench, under twelve lettered
  assertions (the file's docstring lists them). It assembles the document, checks
  it compiles at every prefix a learner can reach, that no section rebinds a name
  an earlier one owns, that the solved document passes all 54 tests and the
  untouched one passes none, that lending cannot make an unwritten exercise pass,
  and that markers round-trip. Assertion G is the mutation check and is the one
  that looks redundant and is not: sabotage a provider, require its consumer's
  suite to notice. `--quick` skips it. Needs NumPy.
- `python3 tools/check_panels.py`: every payoff panel's Python, lifted out of its
  `.tsx` and run against the assembled document with the worker's globals in
  place. Nothing else checks that these run, and several chapters quote their
  numbers. `--fast` caps every loop at two epochs. Needs NumPy.
- `node tools/check_run_path.mjs`: what the panel does with a verdict once it
  has one, driven in a real browser against a stub Pyodide (the worker needs
  exactly four methods from it). Covers the message protocol, the run state,
  every shape of result, the borrowed-names line, the output stream, all three
  ways to start a run, and that a selection in the editor is visible on every
  ground the theme paints. Not in `npm run check` and not in CI: it wants a
  browser and a dev server, which is the bargain `make_og_image.sh` already
  makes.
- `python3 tools/make_penguins.py`: regenerate `public/data/penguins.json.gz`
  (Chapter 10's dataset; stdlib only, downloads from the palmerpenguins repo,
  deterministic output, written RAW because preparing it is the exercise).
- `python3 tools/bench_penguins.py`: re-measure every number Chapter 10 quotes,
  on the panel's own code path (needs NumPy).
- `python3 tools/bench_depth.py`: re-measure every Python-side number Chapters 7
  and 8 quote, on the browser's own code path (needs NumPy; `--quick` for 5
  epochs, `--only <section>` for one section). Each section prints the prose
  sentence it backs, so a number that has drifted is visible without holding
  the chapter open beside it. Every section but `grid` mirrors a panel the reader
  can run; `grid` is Chapter 7's sixteen-cell hyperparameter table, which no panel
  trains, so the bench is the only place those numbers come from.
- `npm run bench:speeds`: the same for the layer-speed and hop tables, by
  importing `deepNet.ts` (the panel's own arithmetic) into Node. No new
  dependency: it compiles through `tools/tsconfig.bench.json` into `.bench/`.
- `python3 tools/check_brand.py`: the course's mark and hue agree everywhere they
  appear (the two components, the favicon, the theme colour, the social card and its
  source, and the kit's copy of the shared brand files), the title agrees in four places,
  and the social tags name one origin and an image that exists at the size they declare.
  Stdlib only.
- `bash tools/make_og_image.sh`: redraw `public/og-image.png` from `tools/og_card.html`.
  The card is a rendered HTML page rather than a drawn image so it is made of the same
  tokens as the site; `check_brand.py` fails if its accent or monogram drift. Needs a
  Chromium.
- `python3 tools/brand_palette.py`: print the accent family and every contrast ratio in
  it; `--check` fails if `src/brand/brand.css` has drifted from what the OKLCH arithmetic
  computes. Stdlib only.
- `npm run bench:bumps`: every number Chapter 6 quotes (the bars-to-area table,
  the sharpness experiment, the bump's biases and peak), by importing
  `bumpMath.ts`, which is the arithmetic CurveSculptor and BumpBuilder both
  run. Chapter 6's figures come from a browser panel rather than Pyodide, so
  this is the bench that keeps them honest.
