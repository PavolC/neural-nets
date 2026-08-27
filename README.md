# Neural Networks

A course in the **Moving Parts** series.

**[Open the course](https://pavolc.github.io/neural-nets/)**

A self-contained, browser-based course that teaches neural networks by having you build
one. You read short explanations, manipulate live visualizations, and write real Python
(NumPy) in an in-browser editor. Your code is checked by automated tests, including a
numerical gradient check, and then trains a network that reads handwritten digits.

Nothing to install and no account: Python runs in the page through Pyodide (CPython
compiled to WebAssembly), in a Web Worker so a training run never freezes the tab. The
only thing that leaves your machine is the runtime download, about 10 MB, on the first
run.

The sequence follows Michael Nielsen's *Neural Networks and Deep Learning*, with the
explanations rewritten around the interactive parts. It assumes Python and high-school
algebra; vectors, matrices, dot products and slopes are built up from arithmetic where
they are first needed.

## Run it

```
npm install
npm run dev
```

Then open the printed URL (http://localhost:5174). `npm run build` produces a static
site in `dist/` that any static host will serve.

## What you build

Ten modules, each a few readings interleaved with figures, most ending in an exercise
whose output every later module uses.

| # | Module | You write |
|---|--------|-----------|
| 1 | From neurons to networks | `sigmoid`, `fire` |
| 2 | Feedforward | `feedforward` |
| 3 | Learning as descent | `sgd_step`, `sgd` |
| 4 | Backpropagation, the idea | (a step-through visualization and a quiz) |
| 5 | Backpropagation, for real | `backprop` |
| 6 | Universality (an interlude) | (a curve-sculpting playground) |
| 7 | Making it actually work | `cross_entropy_cost`, `cross_entropy_delta`, `init_network`, `l2_step` |
| 8 | Why deep is hard | (a depth and squash comparison) |
| 9 | Assembling the program | `train`, `accuracy` |
| 10 | Your own problem | `standardize`, `one_hot`, `split` |

Module 5 is the summit: the learner's `backprop` is checked entry by entry against
central-difference numerical gradients on a fixed 3-5-4-2 network, to one part in ten
million, and then trains 784-30-10 to about 89% of a held-out thousand in a few seconds.
Module 7's three one-line changes take that to about 92%. Module 8 takes the same
network deeper and measures what breaks: four hidden layers of 30 score 12.6% after a
full epoch, which is exactly the share of the held-out digits that are 1s.

Module 9 is the capstone: the training loop itself, which the course's panels had been
running around the learner's functions until then, plus a glossary translating this
course's vocabulary into the field's and an honest list of what it did not teach.
Module 10 turns the whole thing on data that arrives the way data does: a second
bundled dataset with words, holes, unequal classes and measurements on scales 245 times
apart, where leaving out one preprocessing step drops the network to the majority-class
baseline and a 73.5% score turns out to contain a species it never once predicts.

Nothing is locked. Every module is reachable at any time; what an exercise gates is the
panel that trains with your own code. Nothing after Module 5 depends on your version of
`backprop` either, so a learner who never finishes it still gets every module after it in
full.

Progress (editor contents, revealed hints, passed marks) lives in this browser's local
storage. The start page can save it to a file and load it back, which is how you move it
between browsers.

## Publishing it

`.github/workflows/deploy.yml` builds and publishes to GitHub Pages on every push to
`main`, and it is on: seven runs so far, the first failed and the six since have succeeded,
so the course is live at
[pavolc.github.io/neural-nets](https://pavolc.github.io/neural-nets/). The build is
subpath-safe (`base: "./"`), so a project page under `user.github.io/repo/` needs no
configuration beyond setting **Settings > Pages > Source** to **GitHub Actions**, which is
a one-time manual step for the reason the workflow file records.

## Repo layout

```
/src/                React app
/src/brand/          the series brand layer: accent family, masthead, series footer
/src/start/          the start page: what the course is, the outline, stored progress
/src/modules/NN/     one file per course module, plus interactives/
/src/exercises/      per exercise: skeleton.py, tests.py, solution.py, index.ts
/src/python/         shared Python: course helpers, data loader, reference network, harness
/src/runtime/        Pyodide Web Worker, message protocol, shared worker client
/src/components/     shared UI: CodeEditor (CodeMirror), ExercisePage, KaTeX wrappers
/src/state/          localStorage progress persistence
/src/m0/             the training demo shown on the start page
/public/data/        mnist_subset.bin.gz, pretrained_weights.json.gz, penguins.json.gz
/tools/              build-time scripts, the benches, and the consistency checkers
/course-kit/         the portable version of all of this, for a course on another topic
```

`CLAUDE.md` holds the working conventions: shape conventions all Python obeys, the
exercise and test contract, the module authoring playbook, and the voice rules.
`nn-course-design-doc.md` is the original design. `course-kit/` is the transferable part,
extracted after this course was finished: the rules with the neural networks taken out, the
process that produced them, the incidents behind them, and the shared visual identity, for
building a course like this one on a different topic.

## Regenerating the data and checking the numbers

The three data files are committed, so these are only needed to change them:

```
python3 tools/make_mnist_subset.py     # public/data/mnist_subset.bin.gz, stdlib only
python3 tools/pretrain_weights.py      # public/data/pretrained_weights.json.gz, needs NumPy
python3 tools/make_penguins.py         # public/data/penguins.json.gz, stdlib only
```

Every measurement quoted in Modules 7 and 8 is regenerated by a bench that runs the same
code path the browser does, each section printing the prose sentence it backs. The
exercises have a checker that runs the app's own harness outside the browser:

```
python3 tools/check_exercises.py       # 52 tests: solutions pass, skeletons fail
python3 tools/bench_depth.py           # Modules 7 and 8's Pyodide numbers (needs NumPy)
python3 tools/bench_penguins.py        # Module 10's numbers (needs NumPy)
npm run bench:speeds                   # the layer-speed panel's numbers
npm run bench:bumps                    # Module 6's numbers
python3 tools/check_brand.py           # the mark agrees in all six places it appears
python3 tools/brand_palette.py --check # the accent family matches what OKLCH computes
```

## Measured envelope

Chrome, Pyodide 314.0.5, 2026-08-23: 30 epochs of mini-batch SGD on the bundled subset
(5,000 training and 1,000 test images) complete in about 4.5 seconds and reach 88.6% test
accuracy, deterministic at seed 1. That is well inside the 60-second budget the design
doc set, so no dataset or architecture shrinking was needed. Accuracy plateaus just under
90% because of the subsampled data and the quadratic cost; Module 7 reaches about 92%
once the cross-entropy cost is paired with weights scaled by 1/sqrt(inputs).

Firefox and Safari have not been run against this build. The web APIs it depends on that
are not ancient are `DecompressionStream` (Safari 16.4+, Firefox 113+) for the gzipped
data files and `IntersectionObserver` for deferring the editor; `requestIdleCallback` and
`navigator.clipboard` are both used behind a fallback. So both should work at those
versions and up, but that is inference, not a test.

## License and attribution

Adapted from Michael A. Nielsen, *Neural Networks and Deep Learning*, Determination
Press, 2015, licensed [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/).

This project is a free educational tool. As a derivative of the book it inherits
**CC BY-NC 3.0: it may not be used commercially.** See [LICENSE](LICENSE).

Reference implementations are adapted from Nielsen's MIT-licensed code at
[github.com/mnielsen/neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning).

Module 10's dataset is the Palmer Station Antarctica LTER penguin survey collected by
Dr Kristen Gorman, published as
[palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) by Allison Horst,
Alison Hill and Kristen Gorman, and released CC0 (public domain).
