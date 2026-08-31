# Neural Networks

A course in the **Moving Parts** series.

**[Open the course](https://pavolc.github.io/neural-nets/)**

Build a neural network from its smallest parts, then teach it to recognize handwritten
digits. This self-contained, browser-based course pairs short explanations and live visualizations with real Python
(NumPy) in a workbench that docks beside the reading, so the explanation stays on screen
while you type. Every exercise adds a section to one growing file, checked by automated
tests including a numerical gradient check, and by the end that file is a small neural
network library you wrote, trains a network that reads handwritten digits, and downloads
as an `nn.py` that runs anywhere NumPy is installed.

Nothing to install and no account: Python runs in the page through Pyodide (CPython
compiled to WebAssembly), in a Web Worker so a training run never freezes the tab. Your
code and your progress never leave the browser, and the only download is the runtime
itself, about 10 MB, on the first run.

The sequence follows Michael Nielsen's _Neural Networks and Deep Learning_, with the
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

Ten chapters, each a few readings interleaved with figures, most ending in an exercise
whose output every later chapter uses.

| #   | Chapter                     | You write                                                              |
| --- | --------------------------- | ---------------------------------------------------------------------- |
| 1   | From neurons to networks    | `sigmoid`, `fire`                                                      |
| 2   | Feedforward                 | `feedforward`                                                          |
| 3   | Learning as descent         | `sgd_step`, `sgd`                                                      |
| 4   | Backpropagation, the idea   | (a step-through visualization and a quiz)                              |
| 5   | Backpropagation, for real   | `backprop`                                                             |
| 6   | Universality (an interlude) | (a curve-sculpting playground)                                         |
| 7   | Making it actually work     | `cross_entropy_cost`, `cross_entropy_delta`, `init_network`, `l2_step` |
| 8   | Why deep is hard            | (a depth and squash comparison)                                        |
| 9   | Assembling the program      | `train`, `accuracy`                                                    |
| 10  | Your own problem            | `standardize`, `one_hot`, `split`                                      |

Chapter 5 is the summit: the learner's `backprop` is checked entry by entry against
central-difference numerical gradients on a fixed 3-5-4-2 network, to one part in ten
million, and then trains 784-30-10 to about 89% of a held-out thousand in a few seconds.
Chapter 7's three one-line changes take that to about 92%. Chapter 8 takes the same
network deeper and measures what breaks: four hidden layers of 30 score 12.6% after a
full epoch, which is exactly the share of the held-out digits that are 1s.

Chapter 9 is the capstone: the training loop itself, which the course's panels had been
running around the learner's functions until then, and what the file becomes once that
loop is in it, which is a program that trains a network from a Python prompt with NumPy
as its only import. Chapter 10 turns the whole thing on data that arrives the way data
does: a second bundled dataset with words, holes, unequal classes and measurements on
scales 245 times apart, where leaving out one preprocessing step drops the network to the
majority-class baseline and a 73.5% score turns out to contain a species it never once
predicts. It is the last page, so it carries the where-to-go-next list.

Every chapter names the field's word for its own ideas as it goes, in a short naming note
at the first use: the squash is an activation function, a knob is a parameter, blame is
the error. There is no glossary at the end, because a vocabulary handed over twenty rows
at a time on the last page is a memorization task rather than a translation.

Nothing is locked. Every chapter is reachable at any time; what an exercise gates is the
panel that trains with your own code. Later chapters genuinely run on earlier ones: Chapter
9's program calls the backprop from Chapter 5, which calls the sigmoid from Chapter 1. A
section that has not been written yet is filled in from the course's own copy for the run,
and the panel names what it borrowed, so a learner who opens Chapter 9 first still gets a
run rather than a `NameError`.

Progress (your file, revealed hints, passed marks) lives in this browser's local storage.
The start page can save it to a file and load it back, which is how you move it between
browsers. A learner who started before the exercises became one file has theirs merged in
on first load, with the originals kept and restorable.

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
/src/modules/NN/     one file per course chapter, plus interactives/
/src/exercises/      per exercise: skeleton.py, tests.py, solution.py, index.ts
/src/python/         shared Python: course helpers, data loader, reference network, harness
/src/runtime/        Pyodide Web Worker, message protocol, shared worker client
/src/components/     shared UI: CodeEditor (CodeMirror), ExercisePage, KaTeX wrappers
/src/state/          localStorage progress persistence
/src/m0/             the training demo shown on the start page
/public/data/        mnist_subset.bin.gz, pretrained_weights.json.gz, penguins.json.gz
/tools/              build-time scripts, the benches, and the consistency checkers
```

`CLAUDE.md` holds the working conventions: shape conventions all Python obeys, the
exercise and test contract, the chapter authoring playbook, and the voice rules.
`nn-course-design-doc.md` is the original design. The transferable part of all of this was
extracted after the course was finished and now lives in the series repository, at
[course-kit/](https://github.com/PavolC/moving-parts/tree/main/course-kit): the rules with
the neural networks taken out, the process that produced them, the incidents behind them,
and the shared visual identity, for
building a course like this one on a different topic.

## Regenerating the data and checking the numbers

The three data files are committed, so these are only needed to change them:

```
python3 tools/make_mnist_subset.py     # public/data/mnist_subset.bin.gz, stdlib only
python3 tools/pretrain_weights.py      # public/data/pretrained_weights.json.gz, needs NumPy
python3 tools/make_penguins.py         # public/data/penguins.json.gz, stdlib only
```

Every measurement quoted in Chapters 7 and 8 is regenerated by a bench that runs the same
code path the browser does, each section printing the prose sentence it backs. The
exercises have a checker that runs the app's own harness outside the browser:

```
npm run check                         # document, exercise, panel and brand checks
npm run check:doc                     # the document format: splices, projections
python3 tools/check_exercises.py       # the workbench: 54 tests, twelve assertions
python3 tools/check_panels.py --fast   # every payoff panel runs on the learner's file
node tools/check_run_path.mjs          # browser run and saved-state paths (needs Playwright + Chromium)
python3 tools/bench_depth.py           # Chapters 7 and 8's Pyodide numbers (needs NumPy)
python3 tools/bench_penguins.py        # Chapter 10's numbers (needs NumPy)
npm run bench:speeds                   # the layer-speed panel's numbers
npm run bench:bumps                    # Chapter 6's numbers
python3 tools/check_brand.py           # the mark, the title and the social card agree
bash tools/make_og_image.sh            # redraw public/og-image.png (needs a Chromium)
python3 tools/brand_palette.py --check # the accent family matches what OKLCH computes
```

## Measured envelope

Chrome, Pyodide 314.0.5, 2026-08-23: 30 epochs of mini-batch SGD on the bundled subset
(5,000 training and 1,000 test images) complete in about 4.5 seconds and reach 88.6% test
accuracy, deterministic at seed 1. That is well inside the 60-second budget the design
doc set, so no dataset or architecture shrinking was needed. Accuracy plateaus just under
90% because of the subsampled data and the quadratic cost; Chapter 7 reaches about 92%
once the cross-entropy cost is paired with weights scaled by 1/sqrt(inputs).

The course has been opened in Firefox and Safari and it runs; that is a smoke test rather
than a sweep, so an interactive somewhere may still misbehave. The web APIs it depends on
that are not ancient are `DecompressionStream` (Safari 16.4+, Firefox 113+) for the gzipped
data files and `ResizeObserver` for the on-this-page nav; `requestIdleCallback` and
`navigator.clipboard` are both used behind a fallback. So both should work at those
versions and up, but that is inference, not a test.

## Counting readers

The course ships with no analytics and sets no cookie, so out of the box a deploy tells
its author nothing about who reached it. That was the right default while the course had
one reader who could simply be asked, and it is the wrong one for a course meant to be
found: "two hundred people opened Chapter 1 and stopped" and "nobody arrived" are opposite
problems with opposite fixes, and an uninstrumented build cannot tell them apart.

`src/analytics.ts` will load [GoatCounter](https://www.goatcounter.com/) if, and only if,
`VITE_GOATCOUNTER` names a site at build time:

```
VITE_GOATCOUNTER=https://yourcode.goatcounter.com/count npm run build
```

With the variable unset the chapter compiles away and the bundle contains no reference to
it, which is verifiable: `grep -r gc.zgo.at dist/` finds nothing. With it set, a page view
is counted. It never sends your file, the stored progress, or anything a
learner typed; it sets no cookie, stores no personal data, skips the dev server, and
honours `navigator.doNotTrack`. GoatCounter is free for non-commercial use, which this is.

## License and attribution

Adapted from Michael A. Nielsen, _Neural Networks and Deep Learning_, Determination
Press, 2015, licensed [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/).

The **course content** (the chapter prose, the pedagogical sequence and the interactive
figures under `src/modules/`) follows its source: **CC BY-NC 3.0, so it may not be used
commercially.**

The **application around it** is a separate matter. The exercise harness, the Pyodide
worker, the brand layer and `tools/` are original work containing none of Nielsen's
material, so the CC BY-NC grant does not reach them. They are **MIT**
([LICENSE-MIT](LICENSE-MIT)). The method kit has moved to the series repository and carries
its own licence. Take the harness, take the worker, take the brand layer, and build
something else with them.

Not granted by either: the name _Moving Parts_, its three-band mark, and the sigmoid
glyph that marks this course. The MIT licence covers the code that draws them, not the
marks themselves. Copy `src/brand/`, change the accent and the glyph in `brand.ts`,
replace the three bands in `Monogram.tsx`, and publish under your own name; that is what
the layer is for. Full terms in [LICENSE](LICENSE).

Reference implementations are adapted from Nielsen's MIT-licensed code at
[github.com/mnielsen/neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning).

Chapter 10's dataset is the Palmer Station Antarctica LTER penguin survey collected by
Dr Kristen Gorman, published as
[palmerpenguins](https://allisonhorst.github.io/palmerpenguins/) by Allison Horst,
Alison Hill and Kristen Gorman, and released CC0 (public domain).
