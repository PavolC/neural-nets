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
no accounts, no backend, no analytics.

## Hard rules

- **Never write solution logic into skeleton files.** Solutions live only in
  `solution.py`. Skeletons contain stubs, docstrings, and shape contracts only.
- **Attribution and license requirements must never be removed.** The app footer, README,
  and LICENSE carry the CC BY-NC 3.0 attribution to Michael A. Nielsen's *Neural Networks
  and Deep Learning* (Determination Press, 2015). The derivative work inherits
  CC BY-NC 3.0 and cannot be commercialized.
- **No em dashes in any user-facing prose.** Use commas, colons, or parentheses.
- **`src/brand/` is shared with every other course in the series.** Four of its five
  files are copied unchanged between courses, and `tools/check_brand.py` asserts that
  `course-kit/brand/` still matches them. A change that belongs to this course belongs in
  `src/styles.css`, which loads after the brand layer and can override any of it. Two
  things a course does own inside the layer: `brand.ts` and the one `--accent` line.

## Global shape conventions (all Python must comply)

- Activations and inputs are **column vectors**: an input is shape `(784, 1)`, never
  `(784,)`. Batches stack samples as columns: `X` is `(784, m)`.
- `w[l]` (weights into layer l+1) has shape `(size of layer l+1, size of layer l)`, so a
  layer computes `sigmoid(w @ a + b)`. Biases are column vectors `(n, 1)`.
- Labels are one-hot column vectors `(10, 1)` for training, integer class ids for
  evaluation.
- These match Nielsen's conventions. Notation drift between modules is a bug.

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

- **Module content is authored in TSX, not MDX** (decided in Milestone 1): prose beats
  are short and always interleaved with components, so MDX would add a dependency
  without saving friction. Exercise prompts and hints live in each exercise's
  `index.ts`.
- **Exercise test contract**: tests import the learner's code via
  `from submission import ...`; helpers the learner built in earlier modules are
  provided via `from course import ...` (defined in `src/python/course_helpers.py`),
  so skeletons never contain prior solutions. Test functions are named `test_*`, run
  in definition order, and fail by raising `AssertionError` with a teaching message;
  the first docstring line is the test's display title. Test fixtures are hardcoded
  literals (never regenerated at test time from random streams), so results cannot
  drift between NumPy versions.
- **Pretrained weights are gzipped JSON** (`pretrained_weights.json.gz`), not npz:
  the Module 2 diagram reads the weights in JS (weight-image patches, edge colors)
  and Python reads the same file for the payoff run, and JS has no npz parser.
  Regenerate with `tools/pretrain_weights.py` (needs NumPy).
- **Interactive Python snippets** (payoff runs, live toy training) go through the
  worker's `runPython` request: the snippet reads input via
  `json.loads(_args_json)`, may stream progress with `_js_report(json_string)`, and
  must evaluate to a JSON string. First-party snippets only.
- **Module 10 teaches applying the network to data of the learner's own**
  (added after the teaching review, and a change to the design doc's original
  non-goals): a second bundled dataset (Palmer penguins, CC0) that arrives the
  way data actually does, with words, holes, unequal classes and measurements
  245 times apart in scale. Its exercise is `standardize`, `one_hot` and
  `split`, and its panel shows the two things a practitioner needs to see:
  unscaled the network scores exactly the majority-class baseline, and a
  73.5 percent score hides a class it never predicts. Numbers regenerated by
  `python3 tools/bench_penguins.py`.
- **Module 9 is a closing page, not a chapter** (added after the teaching
  review): it assembles the program the learner has been writing in pieces
  (`train` and `accuracy`, the course's only assessment of assembly), translates
  the course's invented vocabulary into the field's, names what the course did
  not teach (automatic differentiation, modern optimizers, data preparation,
  error analysis), and carries the where-to-go-next list that used to end Module
  8. Neither it nor Module 10 follows a Nielsen chapter, so neither has a `<Recap>`
  or a "Go deeper" link; everything else about a module page applies to both.
- **`course_helpers.py` carries Module 7's three functions too** (`init_network`,
  `l2_step`, `cross_entropy_delta`), so Module 9's capstone can import the
  learner's earlier work the way every exercise does. Module 9's panel then
  patches the learner's own saved versions over them before running their loop,
  so the run really is theirs.
- **`course.backprop` carries a seam for BP1** (added in Module 7): its signature is
  `backprop(weights, biases, x, y, output_delta=None)`, the learner's Module 5
  algorithm with the output layer's blame lifted into an argument (default
  `quadratic_output_delta`, which reproduces Module 5 exactly). This is what keeps
  Module 7's three exercises one-line diffs instead of rewrites; `batch_gradient`
  beside it is the per-example-average adapter Module 5's panel used inline. Module 7's
  panels swap the delta rather than the algorithm, and say so in the prose.

- **The visual identity is a shared series layer, not this course's stylesheet**
  (added after the course was finished, when a second course became likely). `src/brand/`
  holds an accent family of nine hues at one OKLCH lightness and chroma, four named type
  roles, and the masthead, tab strip, link and footer chrome. A sibling course copies four
  of the five files unchanged and edits `brand.ts` plus one `--accent` line. The
  alternative was leaving each course to style itself, which is what makes a set of pages
  look unrelated; the cost is that a change to the shared files has to be made in
  `course-kit/brand/` too, which `tools/check_brand.py` enforces. See "Visual identity"
  below and `course-kit/BRAND.md`.
- **The method is extracted rather than described** (same change). `course-kit/` is this
  file with the neural networks taken out, plus the process that produced it
  (`METHOD.md`), the fifteen learner failures behind its rules (`CASEBOOK.md`), the design
  doc template, the brand, and six slash commands. It is meant to be dropped into an empty
  repo. Rejected: a second generic application scaffold, which would be a guess about a
  topic that does not exist yet. What the kit says instead is which of this repo's ~2,900
  topic-free lines to crib, and from where.
- **The front page's module outline is rendered from the module registry, not from its own
  list** (same change). Modules 9 and 10 were added after the outline was written and were
  missing from it, under a heading that said ten, so the only page showing the whole course
  showed eight of it. `COVERS` in `StartPage.tsx` is now a lookup keyed by module id, read
  through `MODULES`, and the heading counts the registry. A module with no entry still
  appears under its nav label, so the worst a gap can cost is a missing sentence.
  `tools/check_exercises.py` checks the same seam from the other side: every exercise's
  module id has to exist in the registry.

## Module authoring playbook (learned from the primary learner, follow it)

The primary learner has high-school algebra, Python but no NumPy before Module 1,
and no vectors, dot products, matrices, or calculus. Modules 1-2 were rewritten to
this floor after direct feedback; every later module must be written to it too.

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
  tallies are load-bearing for motivation (why Module 3, why Module 5).
- **Succeed before failing.** The learner solves OR and AND before meeting XOR;
  "press Show a solution, break it one slider at a time, then rebuild" is a valid
  on-ramp for a fiddly interactive.
- **Diagram widths are standardized, two families.** Box-and-arrow diagrams
  (chain-ripple / chain-net classes, the stepper's diagram, Module 5's
  receipts) share one 812-unit-wide viewBox rendered at full column width:
  pad a narrower layout by lowering the viewBox min-x to center it, never by
  scaling. Plot-family figures (concert-plot, tiny-net, shapes-diagram)
  render at exactly natural scale, capped at their viewBox width and
  centered. On sub-720px screens the box family keeps min-width 620px and
  pans inside .figure-scroll. New diagrams join one of the two families.
- **A score gets a breakdown.** Accuracy alone hides which class a network
  fails at, and the course teaches the habit twice: Module 5's training panel
  shows per-digit counts and the eight mistakes the network was surest about
  (drawn from the test images it sent back), and Module 10 shows a 73.5 percent
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
  "Run my code" experiment that ties back to an earlier module's numbers,
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
  `npm run bench:speeds` regenerate every number Modules 7 and 8 quote, each
  section printing the prose sentence it backs; extend one of them rather than
  writing a throwaway. A module's numbers can come from two different engines
  (the Pyodide panels use NumPy's PCG64, the layer-speed panel its own
  mulberry32), and they are not meant to agree: never quote a number from one
  engine for a measurement the reader makes with the other, and say which panel
  a table came from when both are on the page. Prefer statistics that hold
  still: the mean and the middle 90 percent of 200 draws reproduce, the extremes
  of that same stream do not. The bench has to match the browser exactly, not
  just mathematically. Two traps, both found while writing Module 8: `init_network` draws every weight and then every bias,
  so a bench that interleaves them builds a different network from the same
  seed; and `batch_gradient` sums per-example gradients in a loop, so a
  vectorized bench rounds differently. The second one matters because a deep
  sigmoid network amplifies it: two mathematically identical runs agree for
  about seven epochs and then drift a point or two apart. Where a run is still
  moving at the end, quote an average over the last several epochs (Module 7's
  regularization section and Module 8's depth tables both do) rather than a
  single epoch's score that will not reproduce.
- **No unexplained constants.** Every number in the prose is either derived in
  front of the reader, quoted from an earlier module, or explicitly labeled a
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
  with sigmoids making bars", and the answer was that Module 6 had been
  rebuilding three things he already owned. The switchover at -b/w is Module 1's
  decision boundary with one input removed (the line has nowhere to run, so it
  shrinks to a point). The bump is Module 1's XOR network with the squash taken
  off: two hidden neurons stepping at 0.5 and 1.5 along the total x1 + x2, wires
  out carrying +8 and -8, low then high then low. The tower's thresholding
  neuron is Module 1's output neuron doing what its bias of -4 did. A module
  that re-derives an artifact the learner built by hand says so, with the
  numbers restated, because a reader who is not told sees new machinery and asks
  why the ground moved. Watch for silent axis swaps in the same way: Module 1
  plots input space with a second input up the page, Module 6 plots one input
  with the output up, and the identical-looking square means two different
  things.
- **A callback earns prose only if it removes work or carries the argument.**
  Removing work means the reader has nothing new to learn because it is the same
  object (the three identities above). Carrying the argument means the page's
  conclusion depends on it: Module 6's bars, towers and boxes are one lookup
  table growing in dimension, which is why the box network fits every training
  image and reads nothing new, so "the weights now hold a lookup table" belongs
  in the bars section and pays off at the boxes. A callback that only says
  "remember this from before" costs attention and returns nothing; cut it. Two
  callbacks to the same earlier figure inside one section are clutter. Density
  check: callbacks per paragraph across Modules 2 to 8 runs 0.4 to 1.2, so a
  module below that band is under-connected rather than at risk of pointing
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
  Modules 1 to 5 name them.** Module 6 built its bump out of "two steps" and
  gave "the output neuron the weights +6 and -6", with no wiring diagram in
  the section; the primary learner read it and asked "is this a single neuron
  per layer? what are we talking about here?". Two bugs behind one question.
  The prose never said that the two steps are two neurons side by side in one
  hidden layer, so the shape had to be inferred from a plot of curves: a
  module that places numbers by hand shows the network those numbers live in,
  in Module 1's tiny-net idiom (gray input circles, green-bordered neurons,
  one weight per arrow, a two-line caption under each column, biases beside
  the neuron that holds them). And "output weight" was a coined term the
  course never establishes, filing weights with a neuron; the established
  phrasing is the wire's ("the wire out of h1 carries +6"). Before coining a
  noun, check what the earlier modules already call the thing, and check the
  reverse too: Module 6 had "band" meaning a step's switchover in one section
  and a strip of the concert plane in another.
- **Mind vocabulary collisions.** "Line" means the decision boundary in this
  course; never reuse it for an equation or a row of code. Prefer short
  sentences over connective-heavy ones; wordiness reads as weirdness. A word
  must also not appear before the section that defines it: Module 6's bump
  payoff said "outside its own slice" one section ahead of "Slice the dial into
  equal pieces", so it leaned on a word the reader did not have yet.
- **Interactives carry the algorithm; equations recap it.** Module 4's first
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
  borrowed analogy, a scope note, a why-digression like Module 2's
  loop-vs-matrix) goes in the shared <Aside> component from ModuleBits, never
  in a long parenthetical; the shaded box tells the reader the lesson pauses
  and resumes. Models: Module 3's nudge-size trade-off, Module 4's currency
  chain, Module 1's how-special-is-this-toy notes.
- **A borrowed mini-world is allowed when the home story has no carrier.** The
  concert world has no natural chain-of-conversions, so Module 4 teaches
  factors as posted exchange rates in a two-booth currency chain (a raise:
  euros -> dollars -> pesos), proven with the primary learner live, then maps
  back explicitly (raise = nudge, pesos = cost, factor = posted rate,
  through-rate = slope). Keep such a world to one beat plus one figure, map
  every element back by name, and let its vocabulary (booth, posted rate,
  through-rate) run through the section it serves.
- **Demonstrate on numbers where the mechanism is visible.** Module 4 first
  proved "the change comes out multiplied by the partner" on the hop whose
  partner was 1.0, and the learner read it as "changing by 0.01 changes it by
  0.01, so what?". A worked example whose value makes the key effect a no-op
  (multiply by 1, add 0, gap of 0) teaches that nothing happened. Lead with an
  instance where the effect shows (the times-2 wire), then explain the no-op
  value as the special case it is.
- **A module about a property, not a technique, must be motivated by the
  learner's own number and cashed out on the learner's own artifact.** Module
  6's first draft opened with "the question is what a network of these neurons
  can express", posed a curve-fitting problem in a new world (continuous input,
  a 0-to-10 rating, an unsquashed output, no training), and put its only
  connection to the course in the closing beat. The learner stopped five
  sections in: "we're just talking about curves..... why??". Modules 1-5 each
  deliver a thing the learner then owns, so a module that delivers a fact has
  to work harder, in this order: open with a number the learner produced
  (Module 5's 89 percent), name the competing explanations for it, say which
  one this page settles and which it leaves standing, and then, at the end,
  spend the result back on the same artifact (the 7.8 million-neuron box
  network against the trained 30). Anything that looks like a departure
  (one input instead of 784, hand-placed weights instead of trained ones, no
  exercise) gets named as a deliberate shrink in the opener, with the reason,
  plus the sentence that the small case is a sub-case and not a detour: hold
  783 pixels still, turn one, and the outputs trace curves.
- **Three sentence shapes to keep out of module prose, all found by the
  primary learner in Module 6 ("dense, backwards, vague, clippy").** First,
  clefts and abstract-first openers: "What no part of the argument provides
  is...", "Nothing in that argument stops anywhere", "The claim is about
  relationships". They front a placeholder and delay the content, which reads
  as backwards. Give every sentence a concrete subject and a verb, in that
  order. Second, short pronoun aphorisms, especially as paragraph openers:
  Module 6's limits section had four consecutive paragraphs beginning "It
  covers...", "It promises...", "It says can, not will", with the antecedent
  five paragraphs away. Name the subject each time, even at the cost of a
  longer sentence. Third, meta-narration of the exposition ("this section
  makes X, the next turns it into Y", "now the question this page opened
  with", "shrink it to one input"): describe the thing, not the plan for
  describing it. The measurable signature, comparing against Modules 1 to 5:
  cleft openers at or under 5 percent of sentences, pronoun aphorisms at or
  under 5 percent, median sentence length 19 to 23 words. Cutting words is
  not the fix when prose reads badly; Module 6 got clippy precisely because
  two rounds of cutting turned full sentences into telegraphese. Rejoin
  clauses instead.
- **State what a section buys before proving it.** Module 4's factor
  paragraphs read to the learner as "simple math, overexplained" until the
  stakes came first: every factor is computable from the first run alone, so
  the nudge and its second run become unnecessary. When a stretch of prose
  verifies something, open with the one-sentence payoff the verification
  earns, and close by cashing it out; checks without stated stakes read as
  arithmetic for its own sake.
- **Teach kinds, not instances.** When several facts repeat one pattern (Module
  4's five ripple factors), position-by-position derivations read as N separate
  proofs and lose the learner even when each line checks out. Instead: name the
  pattern once, framed in a concept already taught ("every factor is a slope,
  Module 3's kind of number"), sort the instances into their few kinds, and give
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
- **Assume weeks pass between modules.** Never lean on a bare name from an
  earlier module (fire, the contrarian, the nine numbers) as a load-bearing
  reference; either restate the thing in a few plain words or drop the
  callback entirely. Callbacks are seasoning, not structure.
- Housekeeping: learners paste rendered math as doubled text ("x1x1", "WW").
  That is a copy artifact of KaTeX's accessibility markup, not a rendering bug;
  do not chase it.
- **Every new symbol goes in the notation reference.** The start page carries a
  folded lookup (`NOTATION` in `src/start/StartPage.tsx`): symbol, one line of
  meaning, and the module that introduced it, in the order a reader meets them.
  The course's own rule is that weeks pass between modules, and a symbol defined
  once four thousand words ago is not defined for that reader. Introducing
  notation in a module means adding a row there in the same change.
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
- Each module opens with "What you'll be able to do after this" (2-3 items) and closes
  with a recap and a "Go deeper" link to the corresponding Nielsen chapter.
- Module bodies are broken into sections with <SectionHeader id title /> markers
  (ids unique across modules, prefixed "m4-"), and each module mounts <ModuleToc />
  once after its AfterThis block: a floating on-this-page nav in the right gutter
  (self-hides on narrow screens) that discovers that module's headers from the DOM
  and scrollspies them. Section titles are short noun phrases, 5-8 sections per
  module.

### Register: plotted, narrator muted (adopted after Module 4; applies to all modules)

Modules are plotted like a story (setup, tally, payoff, callbacks): keep that. What
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
- Exception: a module's final beat may carry one slightly hot sentence. Endings do
  gating work in a self-paced course, and they are the two-minute-demo moments.

## Visual identity

The series brand layer (`src/brand/`, documented in `course-kit/BRAND.md`) owns the accent
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
- **The mark exists in six places and one of them is a literal.** The masthead monogram
  and the footer monogram render from `COURSE.glyph`, but `index.html`'s favicon and
  `theme-color` have to be literals, because a tab needs its icon before any JavaScript
  runs. `python3 tools/check_brand.py` is what keeps them equal.
- Data hues (`--acc`, `--loss`, `--data-*`) are not brand colours and do not follow the
  accent. They name quantities in charts, and a course that changes its accent must not
  silently repaint its charts.
- **Prose takes `--measure`, everything else takes the column.** A line of prose runs to
  34rem (about 69 characters at the 19px root); figures, tables, panels and the editor keep
  the full 920px. The selectors that apply it are all direct-child of the reading column
  (`.module > p`, `.exercise > p`), which is what makes it safe in a stylesheet with
  twenty-five interactives: a paragraph inside a panel is not a child of the article and
  never sees it. Display equations take `--measure-wide` (42rem) because they centre
  themselves and cannot re-wrap; two of the course's 51 equations still scroll inside their
  own affordance, both of them wide `\underbrace` tallies.
- **A course links up to the series index, never across to a sibling.** `SERIES.homeUrl`
  is set once and never touched; the index is the only thing that knows what else exists.
  There is deliberately no list of courses in `brand.ts`: carrying one would mean editing
  and redeploying every course's repository on each new course, which is the same
  hand-maintained-list failure as the front page's module outline, once per repository.
- **A section title carries a short accent rule above it.** It is the one piece of
  structural furniture a sibling course inherits already recoloured, and the start page's
  plain h3s wear it too, so a reader cannot tell which sections are scrollspied.
- **Hover is declared per variant, never on `button`.** A bare `button:hover` has higher
  specificity than `.tab:hover` or `.module-toc-item:hover`, so it fills every variant that
  sets its own transparent background: that is a bug this course has already shipped once.
  The primary treatment is reached as `button:not([class])`, which cannot match a variant,
  because every variant here carries a class.
- **The editor is themed from brand tokens, in `CodeEditor.tsx`.** Its chrome comes from
  the surfaces and the accent; its token colours come from the accent family, so syntax
  highlighting is legible by construction (every hue in that family clears 6:1 on the page
  ground) rather than by eye. Do not hand-pick a syntax colour: take one of the nine.
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
/src/modules/NN/     one folder per module: content, interactives/
/src/exercises/      per exercise: skeleton.py, tests.py, solution.py, index.ts (prompt, hints)
/src/python/         shared Python: harness, course helpers, gradient checker, data loader
/src/runtime/        Pyodide Web Worker, message protocol, shared worker client
/src/components/     shared UI: CodeEditor (CodeMirror), ExercisePage
/src/state/          localStorage progress persistence (gn:v1: key prefix)
/src/m0/             Milestone 0 training demo UI
/public/data/        mnist_subset.bin.gz, pretrained_weights.json.gz, penguins.json.gz
/tools/              build-time scripts (MNIST preprocessing, weight pretraining)
/course-kit/         this course's method with the neural networks taken out
```

## Commands

- `npm run dev`: start the dev server.
- `npm run build`: static production build (deployable to any static host).
- `python3 tools/make_mnist_subset.py`: regenerate `public/data/mnist_subset.bin.gz`
  (pure stdlib, downloads MNIST from a public mirror, deterministic output).
- `python3 tools/check_exercises.py`: run every exercise's tests against its
  reference solution (all must pass) and against its skeleton (all must fail,
  with the skeleton's own NotImplementedError). Same harness the app runs, so
  it gates a change to any test, skeleton or solution. Needs NumPy.
- `python3 tools/make_penguins.py`: regenerate `public/data/penguins.json.gz`
  (Module 10's dataset; stdlib only, downloads from the palmerpenguins repo,
  deterministic output, written RAW because preparing it is the exercise).
- `python3 tools/bench_penguins.py`: re-measure every number Module 10 quotes,
  on the panel's own code path (needs NumPy).
- `python3 tools/bench_depth.py`: re-measure every Python-side number Modules 7
  and 8 quote, on the browser's own code path (needs NumPy; `--quick` for 5
  epochs, `--only <section>` for one section). Each section prints the prose
  sentence it backs, so a number that has drifted is visible without holding
  the module open beside it.
- `npm run bench:speeds`: the same for the layer-speed and hop tables, by
  importing `deepNet.ts` (the panel's own arithmetic) into Node. No new
  dependency: it compiles through `tools/tsconfig.bench.json` into `.bench/`.
- `python3 tools/check_brand.py`: the course's mark and hue agree in all six places
  they appear (the two components, the favicon, the theme colour, and the kit's copy of
  the shared brand files). Stdlib only.
- `python3 tools/brand_palette.py`: print the accent family and every contrast ratio in
  it; `--check` fails if `src/brand/brand.css` has drifted from what the OKLCH arithmetic
  computes. Stdlib only.
- `npm run bench:bumps`: every number Module 6 quotes (the bars-to-area table,
  the sharpness experiment, the bump's biases and peak), by importing
  `bumpMath.ts`, which is the arithmetic CurveSculptor and BumpBuilder both
  run. Module 6's figures come from a browser panel rather than Pyodide, so
  this is the bench that keeps them honest.
