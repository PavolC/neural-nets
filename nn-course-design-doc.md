# Design Doc: Interactive Neural Networks Course (working title: "Grokking Nets")

## 1. What this is

A self-contained, browser-based interactive course that teaches neural networks from first principles through hands-on implementation. The learner reads short original explanations, manipulates live visualizations, and writes real Python (NumPy) in an in-browser editor. Their code is validated by automated tests, including numerical gradient checking, and used to train a real network on MNIST digits, all without installing anything.

Pedagogical sequence adapted from Michael A. Nielsen, *Neural Networks and Deep Learning*, Determination Press, 2015 (CC BY-NC 3.0). Reference implementations adapted from his MIT-licensed code repo (github.com/mnielsen/neural-networks-and-deep-learning).

### Goals, in priority order
1. The primary learner (Pavol) achieves deep understanding by implementing feedforward, SGD, and backpropagation himself.
2. The artifact is fully self-sufficient: a colleague can open a link and complete the course with no book, no setup, no author involvement.
3. It is demoable in under two minutes: open link, show a live visualization, show a network training.

### Non-goals
- Not a production ML tool. Small networks, subsampled data, clarity over performance.
- Not a full port of Nielsen's book. Chapters 5 and 6 are covered conceptually, not implemented (see Module 8).
- No accounts, no backend, no analytics. Progress lives in localStorage.

## 2. Licensing and attribution

- The book is CC BY-NC 3.0: adaptation permitted with attribution, non-commercial only. This project is a free internal educational tool, which qualifies.
- Required: a persistent attribution notice in the app footer and README: "Adapted from Michael A. Nielsen, Neural Networks and Deep Learning, Determination Press, 2015, licensed CC BY-NC 3.0."
- The derivative work inherits CC BY-NC 3.0. State this in the repo LICENSE/README so downstream colleagues know they cannot commercialize it.
- Prose policy: write original explanations structured for interactivity. Verbatim borrowing is legally fine with attribution, but keep it rare and deliberate (only where Nielsen's phrasing is clearly the best version). Each module links to the corresponding book chapter as the optional "go deeper" path.
- Code policy: Nielsen's network.py (MIT) may be adapted freely for reference solutions.

## 3. Architecture

### Stack
- **Frontend:** React + Vite, single-page app, static build. Deployable to GitHub Pages / Netlify / any static host, or run locally with one command.
- **Python runtime:** Pyodide (WebAssembly CPython) with NumPy, loaded lazily on first exercise. Pin the Pyodide version in CLAUDE.md.
- **Editor:** CodeMirror 6 with Python mode. One editor pane per exercise, pre-filled with skeleton code.
- **Execution model:** Run Pyodide in a Web Worker so training loops never freeze the UI. Main thread sends code + test suite to worker; worker streams back stdout, test results, and training metrics (per-epoch loss/accuracy) for live charting.
- **Visualizations:** SVG/Canvas in React. No heavy chart library needed; D3 acceptable for scales/paths if useful. Keep each interactive a self-contained component.
- **State:** localStorage for module progress, saved editor code per exercise, and completion flags. "Reset module" and "Reset all" controls. Export/import progress as JSON (nice-to-have).

### Data
- Bundle a preprocessed MNIST subset as a static asset: 5,000 training images + 1,000 test images, stored as a compressed binary (e.g., uint8 pixels + labels, gzipped, roughly 4 MB). A build-time script (Python, in `tools/`) produces this file from the canonical MNIST source.
- In-app loader converts to NumPy arrays inside Pyodide. Target: a 784-30-10 network reaches roughly 90% test accuracy in under 30 seconds of in-browser training. Verify this early (see Milestone 0).

### Exercise/test contract (uniform across modules)
Each exercise defines:
- `skeleton.py`: function stubs with docstrings stating exact signatures, shapes, and return types. Shape conventions fixed globally: inputs are column vectors, weights `w[l]` has shape (size of layer l+1, size of layer l), consistent with Nielsen.
- `tests.py`: pure-Python/NumPy assertions run inside Pyodide. Deterministic (fixed seeds). Each test has a human-readable failure message that teaches ("expected shape (30, 1), got (30,)... remember activations are column vectors").
- `solution.py`: reference implementation. Hidden behind a three-stage reveal in the UI: Hint 1 (conceptual nudge), Hint 2 (pseudocode/structure), Full solution. Reveals are per-exercise and recorded in progress state, no shame mechanics, just friction.
- Gating: a module's "next" unlock requires its tests passing, but free navigation to any previously seen content is always allowed.

The gradient checker (Module 5) is the flagship test: compares analytic gradients from the learner's backprop against central-difference numerical gradients on a tiny fixed network, elementwise, with tolerance 1e-7 relative. This is the single strongest correctness guarantee in the course and should be prominently celebrated in the UI when it passes.

## 4. Modules

Conventions per module: opens with "What you'll be able to do after this" (2-3 items), body interleaves short prose beats (150-400 words each) with interactives and exercises, closes with a recap and a "Go deeper" link to the corresponding Nielsen chapter.

### Module 1: From neurons to networks
- **Covers:** Perceptrons, weights/bias intuition, why a single linear unit cannot solve XOR, sigmoid neurons as smooth perceptrons, network anatomy (layers, notation).
- **Interactives:** (a) Draggable decision boundary over 2D points; learner tries and fails to separate XOR. (b) A fixed 2-2-1 network solving XOR with adjustable weights via sliders, showing the two hidden units carving the plane.
- **Exercise:** Implement `sigmoid(z)` and a single neuron `fire(w, b, x)`. Trivial by design: its real job is onboarding the editor/test workflow.
- **Book link:** Chapter 1 (first half).

### Module 2: Feedforward
- **Covers:** Layers as matrix multiplication, shape discipline, why vectorization matters.
- **Interactive:** Live network diagram (784-15-10, rendered compactly) where hovering any neuron shows its incoming weights as an image patch; learner's own `feedforward` drives the display once implemented.
- **Exercise:** Implement `feedforward(weights, biases, x)` returning output activations. Tests check shapes, values against fixtures, and correct layer ordering.
- **Payoff moment:** Load pre-trained weights (bundled asset), run the learner's feedforward on real MNIST digits, show predictions. Their code, recognizing digits, before any training theory.
- **Book link:** Chapter 1 (network section).

### Module 3: Learning as descent
- **Covers:** Cost functions (quadratic first), gradient descent intuition, learning rate, stochastic mini-batches.
- **Interactives:** (a) 1D and 2D cost-surface playground: pick a start point, set learning rate, step; overshoot and oscillation visible. (b) Mini-batch intuition: full-batch vs. SGD trajectories on the same surface.
- **Exercise:** Implement `sgd_step` using a provided **numerical** gradient function (finite differences), and a mini-batch loop. Train a tiny network on a toy 2D dataset live. It works but is visibly slow; the module ends by quantifying why (one gradient estimate costs one forward pass per parameter) to set up backprop.
- **Book link:** Chapter 1 (learning section).

### Module 4: Backpropagation, the idea
- **Covers:** Error signal delta, the four fundamental equations (BP1-BP4), built up one at a time with meaning-first framing (this is a place where borrowing Nielsen's framing verbatim, with attribution, is justified).
- **Interactive:** The centerpiece visualization: a 2-3-1 network with every activation, weighted input, weight, and delta inspectable. Step controls: forward pass step-by-step, then backward pass step-by-step, numbers updating live, with the currently-applied equation highlighted. Learner can nudge any weight and watch downstream/upstream values change.
- **Exercise:** None (deliberately). A short interactive quiz instead: predict which delta grows/shrinks when a given weight changes, answered by manipulating the visualization.
- **Book link:** Chapter 2.

### Module 5: Backpropagation, for real
- **Covers:** Implementing BP1-BP4 in NumPy.
- **Exercise (the summit):** Implement `backprop(weights, biases, x, y)` returning gradient lists. Test ladder: (1) shape checks, (2) fixed-fixture value checks on a tiny net, (3) the numerical gradient check, (4) integration: plug into the Module 3 SGD loop, train 784-30-10 on the MNIST subset, live loss/accuracy chart, target >= 88% test accuracy.
- **UI:** Prominent success state when the gradient check passes. Show a wall-clock comparison against Module 3's numerical-gradient training to make the speedup visceral.
- **Book link:** Chapter 2.

### Module 6 (Interlude): Universality
- **Covers:** Why networks can approximate any continuous function, via the bump-construction argument.
- **Interactive:** Function-sculpting playground: learner is given a target 1D curve and builds an approximation by adding sigmoid pairs (bumps), adjusting position/height. Score = area between curves. Pure play, no code, no gate.
- **Book link:** Chapter 4. (Note the reorder: Nielsen's Chapter 3 maps to our Module 7.)

### Module 7: Making it actually work
- **Covers:** Learning slowdown and cross-entropy cost, L2 regularization and overfitting, smarter weight initialization (1/sqrt(n)). Softmax mentioned, not required.
- **Structure:** Three short cycles, each: problem demo -> small code change to the learner's own network from Module 5 -> before/after training comparison chart. Overfitting demo uses a deliberately small training slice (1,000 images) so the train/test gap is dramatic.
- **Exercises:** `cross_entropy_delta`, `l2_update` (modify the weight update rule), `smart_init`. Each is a focused diff, not a rewrite.
- **Book link:** Chapter 3.

### Module 8: Why deep is hard (and what came next)
- **Covers:** Vanishing/unstable gradients, why they follow directly from BP2 (chained sigmoid derivatives), and a conceptual bridge: convolutions as weight sharing, ReLU, and one paragraph honestly connecting this 2015-era foundation to modern practice.
- **Interactive:** Per-layer gradient magnitude bars on the learner's own network, with a layer-count slider (2 to 6 hidden layers): watch early-layer gradients collapse as depth grows. Toggle sigmoid vs. ReLU to see the mitigation.
- **Exercise:** None mandatory. Optional: swap sigmoid for ReLU in their network and compare training on 3 hidden layers.
- **Covers conceptually, does not implement:** convnets (Nielsen Ch. 6). From-scratch convolutions in Pyodide are slow and add little conceptual return; say this openly in the module and link the chapter.
- **Closing note to include (learner-requested): embedding spaces and LLMs.** Build the
  bridge from the course's own artifacts, in this order: input space (the concert plane;
  MNIST's 784 axes) -> hidden activations as a learned re-description (the XOR network's
  (h1, h2) space, where classes that were not line-separable become separable) -> word
  embeddings (a word starts as a one-hot column, one slot per vocabulary word; multiplying
  by a learned matrix selects that word's row, and training by the same SGD drags
  similarly-used words to nearby points, so distance and direction acquire meaning) ->
  LLMs as the same substrate (columns, matrices, cost, gradient descent) at billions of
  parameters. Name attention as the genuinely new ingredient and leave it unexplained.
  One line to keep: input space is where the data arrives; embedding space is where the
  network chooses to think, and it learns that choice by the descent the learner
  implemented in Module 3.
- **Ends with:** A closing page: what the learner built, in their own code, and a curated "where to go next" list (Nielsen Ch. 5-6, then modern resources).

## 5. Content conventions

- Prose beats are short (150-400 words), one idea each, written to sit beside an interactive or editor, not before a wall of them.
- Voice: plain, direct, second person, no hype. Define every symbol at first use. Math rendered with KaTeX; every equation gets a one-sentence plain-language gloss immediately after it.
- Every module's notation must match the global shape conventions exactly. Notation drift between modules is a bug.
- Test failure messages are part of the teaching content and get the same writing care as prose.
- Attribution footer on every page; per-module "Go deeper" links to the book.

## 6. Repo layout

```
/                    README (what/why/how to run/license/attribution)
/CLAUDE.md           conventions for Claude Code sessions (below)
/src/                React app
/src/modules/NN/     one folder per module: content.mdx (or .tsx), interactives/
/src/exercises/      skeleton.py, tests.py, solution.py per exercise
/src/python/         shared Python: harness, gradient checker, data loader
/public/data/        mnist_subset.bin.gz, pretrained_weights.npz
/tools/              build-time scripts (MNIST preprocessing, weight pretraining)
```

## 7. CLAUDE.md conventions (seed content)

- Project purpose and the goals/non-goals above, condensed.
- Never write solution logic into skeleton files; solutions live only in solution.py.
- Global shape conventions (column vectors; w[l] shape rule) restated; all Python must comply.
- All exercise tests must be deterministic (fixed seeds) and runnable inside Pyodide (no packages beyond NumPy).
- Pinned versions: Pyodide, React, CodeMirror.
- Content voice rules from Section 5, condensed.
- No em dashes in any user-facing prose.
- Attribution/license requirements (Section 2) must never be removed.

## 8. Build milestones

- **M0, feasibility spike (do first):** Pyodide + NumPy in a Web Worker, load the MNIST subset, train 784-30-10 with a reference implementation, stream metrics to a live chart. Confirms the performance envelope before any course content exists. If training exceeds ~60s, shrink the dataset or hidden layer and record the decision.
- **M1, exercise pipeline:** Editor -> worker -> tests -> results UI, with the three-stage hint reveal and localStorage persistence. Build using Module 2's exercise as the guinea pig.
- **M2, spine:** Modules 1-3 complete (content + interactives + exercises).
- **M3, summit:** Modules 4-5, including the step-through visualization and gradient checker. This is the highest-effort, highest-value milestone.
- **M4, tail:** Modules 6-8.
- **M5, handoff polish:** README, demo path (a "tour" landing page), progress export, cross-browser check (Chrome/Firefox/Safari), colleague dry-run.

Each milestone should end in a deployable state. Deploy early (GitHub Pages) so the demo link exists from M1 onward.

## 9. Open questions (all since resolved)

This section is kept as written; the answers are recorded here rather than by editing the questions away.

- MDX vs. TSX for module content authoring (decide in M1 based on friction). **TSX.** Prose beats are short and always interleaved with components, so MDX would have added a dependency without saving friction.
- Whether Module 5's big training run should offer a "use reference backprop" fallback so a stuck learner can still experience Modules 7-8 (recommended: yes, clearly labeled). **Yes, and it fell out of the architecture rather than being built.** Modules 7 and 8's panels take their gradient from `course.backprop`, so no panel after Module 5 requires the learner's own version; only Module 5's own payoff run does. Module 5's locked panel and the start page both say so.
- Dark mode. (No.) **Still no.**

## 10. What the build actually produced

Milestones 0 through 5 are complete. Departures from the plan above, all deliberate:

- **Module 8 grew a payoff run and a second engine.** The design doc called for per-layer gradient bars with a depth slider and a sigmoid/ReLU toggle; that is `LayerSpeedBars` (TypeScript, so the bars move as fast as the slider does), and beside it `DepthTrainPanel` trains one hidden layer against four with the learner's own code in Pyodide. Two engines means two sets of numbers from two random generators: `tools/bench_layer_speeds.ts` and `tools/bench_depth.py` regenerate them, and CLAUDE.md records that a number from one must never be quoted for the other.
- **The optional ReLU exercise was not built.** Module 8 swaps the squash inside its own panel instead, which gets the comparison without asking the learner to rewrite `backprop` after the course has already taken its summit.
- **A tenth module was added, and with it a fourth goal.** The design doc's
  non-goals said "not a production ML tool", and nothing in Modules 1 to 9 shows a
  learner how to point the network at a problem of their own: every dataset in the
  course arrives scaled, numeric, complete, labelled and split. Module 10 is that
  goal made explicit. It bundles a second dataset (Palmer penguins, CC0) chosen for
  its defects rather than despite them, and teaches preparation, baselines, per-class
  evaluation, picking a first network, a diagnostic table, and a read-along PyTorch
  listing beside the parts the learner wrote. The non-goal still holds for the
  artifact; what changed is that the learner is now expected to leave able to apply
  this, not only to have understood it.
- **A ninth module was added** (not in the design doc), after a teaching review of
  the finished course found that the learner owned seven functions but had never
  assembled them: every training run was started by a course-written panel. Module 9
  is the loop, plus the glossary from this course's words to the field's and the
  where-to-go-next list moved out of Module 8. The review also added the validation
  split to Module 7, which had been tuning hyperparameters against the same held-out
  thousand it reported.
- **A start page was added** (not in the design doc): what the course is, how the machinery works, the eight modules, the training demo that used to sit on a tab of its own, and the stored progress with save, load and reset. It is where a bare link lands.
- **Gating stayed softer than section 3 proposed.** Nothing locks a module; what an exercise gates is the panel that runs the learner's code, since there is nothing to run until the code exists.
