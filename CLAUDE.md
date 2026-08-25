# CLAUDE.md

## What this project is

"Grokking Nets": a self-contained, browser-based interactive course that teaches neural
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
- **Mind vocabulary collisions.** "Line" means the decision boundary in this
  course; never reuse it for an equation or a row of code. Prefer short
  sentences over connective-heavy ones; wordiness reads as weirdness.
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

## Repo layout

```
/                    README (what/why/how to run/license/attribution)
/CLAUDE.md           this file
/src/                React app
/src/modules/NN/     one folder per module: content, interactives/
/src/exercises/      per exercise: skeleton.py, tests.py, solution.py, index.ts (prompt, hints)
/src/python/         shared Python: harness, course helpers, gradient checker, data loader
/src/runtime/        Pyodide Web Worker, message protocol, shared worker client
/src/components/     shared UI: CodeEditor (CodeMirror), ExercisePage
/src/state/          localStorage progress persistence (gn:v1: key prefix)
/src/m0/             Milestone 0 training demo UI
/public/data/        mnist_subset.bin.gz, pretrained_weights.json.gz
/tools/              build-time scripts (MNIST preprocessing, weight pretraining)
```

## Commands

- `npm run dev`: start the dev server.
- `npm run build`: static production build (deployable to any static host).
- `python3 tools/make_mnist_subset.py`: regenerate `public/data/mnist_subset.bin.gz`
  (pure stdlib, downloads MNIST from a public mirror, deterministic output).
