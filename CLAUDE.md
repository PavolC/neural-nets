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
  **@codemirror/lang-python: 6.2.1** (all pinned exactly in `package.json`).

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
  the first docstring line is the test's display title.

## Content voice rules

- Prose beats are short (150-400 words), one idea each, written to sit beside an
  interactive or editor.
- Plain, direct, second person, no hype. Define every symbol at first use.
- Math rendered with KaTeX; every equation gets a one-sentence plain-language gloss
  immediately after it.
- Each module opens with "What you'll be able to do after this" (2-3 items) and closes
  with a recap and a "Go deeper" link to the corresponding Nielsen chapter.

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
/public/data/        mnist_subset.bin.gz, pretrained_weights.npz
/tools/              build-time scripts (MNIST preprocessing, weight pretraining)
```

## Commands

- `npm run dev`: start the dev server.
- `npm run build`: static production build (deployable to any static host).
- `python3 tools/make_mnist_subset.py`: regenerate `public/data/mnist_subset.bin.gz`
  (pure stdlib, downloads MNIST from a public mirror, deterministic output).
