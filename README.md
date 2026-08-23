# Grokking Nets (working title)

A self-contained, browser-based interactive course that teaches neural networks from
first principles. You read short explanations, manipulate live visualizations, and write
real Python (NumPy) in an in-browser editor. Your code is checked by automated tests,
including numerical gradient checking, and trains a real network on MNIST digits. Nothing
to install: Python runs in your browser via Pyodide (WebAssembly), in a Web Worker so the
page never freezes.

## Status

**Milestone 2 (Modules 1-3) is complete.** The course spine is live: Module 1 (neurons,
sigmoid, why XOR needs a hidden layer) with a draggable separating-line playground and
a 2-2-1 XOR network on sliders; Module 2 (feedforward) with a live 784-15-10 network
diagram, hoverable weight-image patches, and a payoff panel where the learner's own
feedforward classifies real MNIST digits with pretrained weights; Module 3 (gradient
descent) with 1D/2D descent playgrounds, a batch-vs-SGD race, an SGD exercise, and a
live toy-network training run that quantifies the cost of numerical gradients. Math is
rendered with KaTeX, each equation glossed in plain language. Modules unlock in order
as their exercises pass; navigation back is always free.

**Milestone 1 (exercise pipeline) is complete.** The app has a working exercise loop:
a CodeMirror editor pre-filled with skeleton Python, a Run tests button that executes
the learner's code against a deterministic NumPy test suite inside Pyodide, per-test
pass/fail results whose failure messages teach (shapes, layer order, common
misconceptions), a three-stage hint reveal (conceptual nudge, pseudocode, full
solution), and localStorage persistence of code, hints, and completion. Module 2's
feedforward exercise is the first working example.

**Milestone 0 (feasibility spike) is complete.** The app loads Pyodide + NumPy in a Web
Worker, loads a bundled MNIST subset (5,000 training / 1,000 test images), trains a
784-30-10 sigmoid network with a reference implementation, and streams per-epoch loss and
test accuracy to a live chart. Course modules come next (see
`nn-course-design-doc.md`, section 8).

Measured envelope (2026-08-23, Chrome, Pyodide 314.0.5): 30 epochs of mini-batch SGD
complete in about 4.5 seconds and reach 88.6% test accuracy, deterministic at seed 1.
That is far inside the 60-second budget from the design doc, so no dataset or
architecture shrinking was needed. Accuracy plateaus just under 90% because of the
5,000-example subset and the quadratic cost; this meets the course's own Module 5 gate
(at least 88%) and improves later with cross-entropy (Module 7).

## How to run

```
npm install
npm run dev
```

Then open the printed URL (the first training run downloads the Pyodide runtime, about
10 MB, so it needs network access). `npm run build` produces a static build in `dist/`
deployable to GitHub Pages, Netlify, or any static host.

### Regenerating the MNIST subset

`public/data/mnist_subset.bin.gz` is committed, so this is only needed to change the
subset. Requires network access, pure Python stdlib only:

```
python3 tools/make_mnist_subset.py
```

`public/data/pretrained_weights.json.gz` (the Module 2 payoff network) is also
committed; regenerate with `python3 tools/pretrain_weights.py` (needs NumPy).

## Repo layout

```
/src/                React app
/src/modules/NN/     one folder per course module (content + interactives)
/src/exercises/      skeleton.py, tests.py, solution.py per exercise
/src/python/         shared Python: data loader, reference network, harness
/src/m0/             Milestone 0: Pyodide worker + training demo
/public/data/        mnist_subset.bin.gz
/tools/              build-time scripts (MNIST preprocessing)
```

## License and attribution

Adapted from Michael A. Nielsen, *Neural Networks and Deep Learning*, Determination
Press, 2015, licensed [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/).

This project is a free educational tool. As a derivative of the book it inherits
**CC BY-NC 3.0: it may not be used commercially.** See [LICENSE](LICENSE).

Reference implementations are adapted from Nielsen's MIT-licensed code at
[github.com/mnielsen/neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning).
