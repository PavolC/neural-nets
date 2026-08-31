"""Reproduce every Python-side number Chapters 5, 7 and 8 quote, on the browser's code path.

Chapter 8's prose reports measurements from two engines, and this script is the
first of the two benches that regenerate them:

  * tools/bench_depth.py (this file) mirrors DepthTrainPanel's snippet: the
    learner's own init_network, sgd and cross-entropy blame, the course's
    backprop, and the bundled MNIST subset. It produces the accuracy tables,
    the step-size sweep, the ReLU comparison, the dead-unit counts and the
    learning-speed ratios measured during training. It also runs Chapter 7's
    weight decay, which is RegularizePanel's six runs on the 1,000-image
    slice, and Chapter 7's hyperparameter grid, which has no panel behind it.
  * tools/bench_layer_speeds.ts mirrors LayerSpeedBars: the same measurement
    written in TypeScript, which is what the chapter's layer-speed tables and
    hop factors are quoted from.

Two traps make a "mathematically equivalent" bench print different numbers than
the browser (both were found while writing Chapter 8, and both are why this file
exists rather than a fresh reimplementation):

  1. init_network draws EVERY weight matrix and THEN every bias vector, so a
     bench that interleaves the two draws builds a different network from the
     same seed. This script imports the exercise's own solution instead of
     restating it.
  2. batch_gradient sums per-example gradients in a Python loop, so a
     vectorized bench rounds differently. A deep sigmoid network amplifies
     that: two mathematically identical runs agree for about seven epochs and
     then drift a point or two apart.

Needs NumPy (the only dependency outside the standard library):

    python3 -m venv .venv && .venv/bin/pip install numpy
    .venv/bin/python tools/bench_depth.py            # the default set
    .venv/bin/python tools/bench_depth.py --quick    # 5 epochs, for a smoke test
    .venv/bin/python tools/bench_depth.py --only depth relu

Every section prints the prose sentence it backs, so a number that has drifted
is visible without holding the chapter open beside it.
"""

import argparse
import gzip
import pathlib
import sys
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import workbench as wb  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / "src" / "python"
EX = ROOT / "src" / "exercises"
DATA = ROOT / "public" / "data" / "mnist_subset.bin.gz"

# The panel's constants (src/modules/NN/interactives/DepthTrainPanel.tsx).
EPOCHS = 15
HIDDEN_SIZE = 30
BATCH = 10
INIT_SEED = 8
SHUFFLE_SEED = 2
SPEED_SAMPLES = 100
ETA = {"sigmoid": 0.5, "relu": 0.05}

# RegularizePanel's constants (src/modules/NN/interactives/RegularizePanel.tsx).
L2_EPOCHS = 80
L2_SLICE = 1000
L2_ETA = 0.5
L2_TAIL = 20        # the window Chapter 7 averages its two readings over
# The panel pins its shuffle at SHUFFLE_SEED and offers no way to change it, so
# the two further shuffles Chapter 7 quotes are this bench's choice rather than
# the panel's. They are the next two seeds, fixed here before anything was run,
# so the pair cannot have been picked after seeing what it printed.
L2_EXTRA_SHUFFLES = (3, 4)

# Chapter 7's hyperparameter grid: two costs by two starting points, each at
# four step sizes, on the same 5,000 images and the same fifteen epochs as the
# rest of the chapter. Nothing in the app runs this one, so this is the only
# place its sixteen numbers come from.
GRID_ETAS = (0.5, 1.0, 3.0, 6.0)
GRID_SETUPS = (("quadratic", "undivided"), ("cross-entropy", "undivided"),
               ("quadratic", "divided"), ("cross-entropy", "divided"))


def load_module(path, name):
    """Execute a course .py file as a module, the way the worker does."""
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(path.read_text(), path.name, "exec"), mod.__dict__)
    return mod


def library(through="prepare"):
    """The learner's file, assembled from the reference solutions.

    The panels exec one document and read every name out of it, so this does
    too: benching separately-loaded solution files would stop being a bench of
    the code the reader runs the moment the two shapes drift apart.
    """
    rank = wb.SECTIONS.index(wb.BY_ID[through])
    ids = [s["id"] for s in wb.SECTIONS[: rank + 1]]
    kinds = {"backprop": "seam"} if wb.needs_seam(ids) else {}
    document = wb.assemble(ids, "solution", kinds)
    mod = types.ModuleType("your_code")
    mod.__file__ = "your_code.py"
    exec(compile(document, "your_code.py", "exec"), mod.__dict__)
    return mod


def bootstrap():
    """Register `course`, the data loader, and the learner's assembled file."""
    course = load_module(PY / "course_helpers.py", "course")
    sys.modules["course"] = course
    loader = load_module(PY / "data_loader.py", "data_loader")
    lib = library()
    return course, loader, lib, lib


def load_sgd(lib, grad_fn):
    """Point the file's own gradient at the one under test, then hand it back.

    Python looks a global up when the call happens, so rebinding after the
    file has been read is enough, and it is what the panels do now. Getting
    this wrong silently benches the nudge-and-measure gradient.
    """
    lib.gradient = grad_fn
    return lib


def relu(z):
    return np.maximum(0.0, z)


def make_engine(lib, ce, activation, cost="cross-entropy"):
    """The panel's two code paths, returned as (per_example, batch, predict).

    With the sigmoid the gradient is the learner's own backprop, reached
    through the adapter written for them in Chapter 5. With ReLU it is the same
    four equations with one line changed, written out here and in the panel,
    because a learner's BP2 has sigmoid_prime in it.

    cost picks which BP1 the adapter hands down. Quadratic is the adapter's own
    default, backprop called with four arguments, which is the shape Chapter 5
    runs in; cross-entropy passes the replacement through, which is the shape
    everything from Chapter 7 on runs in.
    """
    sigmoid = lib.sigmoid

    def relu_backprop(weights, biases, x, y):
        L = len(weights)
        a = x
        activations = [x]
        zs = []
        for i, (w, b) in enumerate(zip(weights, biases)):
            z = w @ a + b
            zs.append(z)
            a = sigmoid(z) if i == L - 1 else relu(z)
            activations.append(a)
        nabla_w = [np.zeros_like(w) for w in weights]
        nabla_b = [np.zeros_like(b) for b in biases]
        delta = activations[-1] - y                    # BP1, cross-entropy
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T
        for l in range(2, L + 1):
            delta = (weights[-l + 1].T @ delta) * (zs[-l] > 0)   # BP2, ReLU
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l - 1].T
        return nabla_w, nabla_b

    def relu_feedforward(weights, biases, X):
        L = len(weights)
        a = X
        for i, (w, b) in enumerate(zip(weights, biases)):
            z = w @ a + b
            a = sigmoid(z) if i == L - 1 else relu(z)
        return a

    def relu_grad(weights, biases, X, Y):
        m = X.shape[1]
        nw = [np.zeros_like(w) for w in weights]
        nb = [np.zeros_like(b) for b in biases]
        for k in range(m):
            dw, db = relu_backprop(weights, biases, X[:, k:k + 1], Y[:, k:k + 1])
            nw = [t + d for t, d in zip(nw, dw)]
            nb = [t + d for t, d in zip(nb, db)]
        return [t / m for t in nw], [t / m for t in nb]

    if activation == "relu":
        return relu_backprop, relu_grad, relu_feedforward

    delta = None if cost == "quadratic" else ce.cross_entropy_delta

    def sigmoid_grad(weights, biases, X, Y):
        return lib.batch_gradient(weights, biases, X, Y, delta)

    return None, sigmoid_grad, lib.feedforward


def layer_speeds(grad, weights, biases, X, Y, n=SPEED_SAMPLES):
    """Every layer's ||dC/db|| on the panel's first n training images."""
    _, nabla_b = grad(weights, biases, X[:, :n], Y[:, :n])
    return [float(np.linalg.norm(g)) for g in nabla_b]


class Bench:
    def __init__(self, epochs, quick=False):
        self.epochs = epochs
        self.quick = quick
        self.course, loader, self.ce, self.init = bootstrap()
        self.lib = self.ce
        with gzip.open(DATA, "rb") as f:
            raw = f.read()
        self.X_train, self.y_train, self.X_test, self.y_test = loader.load_mnist_subset(raw)
        self.Y_train = loader.one_hot(self.y_train)
        self.Y_test = loader.one_hot(self.y_test)

    def start(self, sizes, which="divided", seed=INIT_SEED):
        """One of the two starting points Chapter 7 compares.

        The divided start is the learner's own init_network. The undivided one
        is that same draw with the division left out, so for a 784-30-10
        network the two scales 28 and sqrt(30) are simply never applied. The
        panels build it by drawing it rather than by multiplying init_network's
        weights back up by those two numbers, and so does this: dividing by 28
        and multiplying by 28 is not the identity in floating point, and a
        start a few last bits away from the panel's is not the panel's run.
        Every weight and then every bias, which is init_network's own order, so
        both starts take the same numbers off the same generator.
        """
        rng = np.random.default_rng(seed)
        if which == "divided":
            return self.init.init_network(sizes, rng)
        weights = [rng.standard_normal((sizes[i + 1], sizes[i]))
                   for i in range(len(sizes) - 1)]
        biases = [rng.standard_normal((sizes[i + 1], 1))
                  for i in range(len(sizes) - 1)]
        return weights, biases

    def run(self, hidden, activation="sigmoid", eta=None, epochs=None,
            init_seed=INIT_SEED, cost="cross-entropy", start="divided", watch=()):
        """One training run, reporting per-epoch test accuracy.

        watch: epochs after which to record every layer's learning speed.
        """
        eta = ETA[activation] if eta is None else eta
        epochs = self.epochs if epochs is None else epochs
        _, grad, predict = make_engine(self.lib, self.ce, activation, cost)
        sgd = load_sgd(self.lib, grad)
        sizes = [784] + [HIDDEN_SIZE] * hidden + [10]
        weights, biases = self.start(sizes, start, init_seed)
        speeds = {0: layer_speeds(grad, weights, biases, self.X_train, self.Y_train)}
        rng = np.random.default_rng(SHUFFLE_SEED)
        accs = []
        for epoch in range(1, epochs + 1):
            weights, biases = sgd.sgd(weights, biases, self.X_train, self.Y_train,
                                      eta, 1, BATCH, rng)
            out = predict(weights, biases, self.X_test)
            accs.append(float((np.argmax(out, axis=0) == self.y_test).mean()))
            if epoch in watch:
                speeds[epoch] = layer_speeds(grad, weights, biases, self.X_train, self.Y_train)
        return {"acc": accs, "speeds": speeds, "weights": weights, "biases": biases,
                "predict": predict, "sizes": sizes}


def pct(x):
    return f"{100 * x:.1f}%"


def tail_mean(accs, k=5):
    return float(np.mean(accs[-k:]))


def section(title, claim):
    print()
    print("=" * 78)
    print(title)
    print("prose:", claim)
    print("-" * 78)


def bench_depth(b):
    section(
        "1. Depth costs accuracy (sigmoid, eta 0.5)",
        "epoch 1: 85.3 / 81.8 / 41.2 / 12.6 | epoch 5: 91.2 / 89.8 / 87.6 / 61.1 | "
        "epochs 11-15: 91.6 / 92.0 / 89.2 / 86.5",
    )
    rows = {}
    for hidden in (1, 2, 3, 4):
        r = b.run(hidden)
        rows[hidden] = r["acc"]
        print(f"  {hidden} hidden: epoch 1 {pct(r['acc'][0])}, epoch 5 {pct(r['acc'][4])}, "
              f"last 5 {pct(tail_mean(r['acc']))}, per-epoch "
              + " ".join(pct(a) for a in r["acc"]))
    return rows


def bench_first_epoch(b):
    section(
        "2. What the 4-layer network answers after one epoch",
        "answers 1 for all 1,000 images; its ten outputs sit between 0.054 and 0.179, "
        "and no single image's highest output is more than 0.13 above its lowest; "
        "12.6% is exactly the share of the held-out digits that are 1s",
    )
    r = b.run(4, epochs=1)
    out = r["predict"](r["weights"], r["biases"], b.X_test)
    preds = np.argmax(out, axis=0)
    counts = np.bincount(preds, minlength=10)
    ones_share = float((b.y_test == 1).mean())
    train_counts = np.bincount(b.y_train, minlength=10)
    per_image_spread = float((out.max(axis=0) - out.min(axis=0)).max())
    print(f"  predictions by class: {counts.tolist()}")
    print(f"  all ten outputs, over all images, span {out.min():.3f} to {out.max():.3f}")
    print(f"  widest spread within a single image: {per_image_spread:.3f}")
    print(f"  test accuracy {pct(r['acc'][0])}, share of test digits that are 1s {pct(ones_share)}")
    print(f"  commonest training digit: {int(np.argmax(train_counts))} "
          f"({train_counts.max()} of {b.y_train.size})")


def bench_eta(b):
    section(
        "3. Step size does not rescue depth (sigmoid, 4 hidden layers)",
        "0.5: 12.6 then 86.5 | 1.0: 23.2 then 84.0 | 2.0: 24.2 then 74.4 | 3.0: 21.0 then 65.4",
    )
    for eta in (0.5, 1.0, 2.0, 3.0):
        r = b.run(4, eta=eta)
        print(f"  eta {eta}: epoch 1 {pct(r['acc'][0])}, last 5 {pct(tail_mean(r['acc']))}")


def bench_ratios(b):
    section(
        "4. The imbalance is a property of the start (sigmoid, 4 hidden layers)",
        "output ahead of layer 2 by 520x before the first step, 178x after 1 epoch, "
        "1.9x after 5, 1.2x after 15",
    )
    r = b.run(4, watch=(1, 5, EPOCHS))
    for epoch in sorted(r["speeds"]):
        s = r["speeds"][epoch]
        when = "before the first step" if epoch == 0 else f"after epoch {epoch}"
        print(f"  {when:>22}: ratio {s[-1] / s[0]:.1f}x   "
              + " ".join(f"{v:.3g}" for v in s))


def bench_relu(b):
    section(
        "5. ReLU flattens the depth cost (eta 0.05)",
        "epoch 1: 83.5 / 83.7 / 83.0 / 63.6 | epoch 5: 89.8 / 90.1 / 89.8 / 90.3 | "
        "epochs 11-15: 91.0 / 91.6 / 90.8 / 90.9 (spans 0.8 of a point)",
    )
    last = {}
    for hidden in (1, 2, 3, 4):
        r = b.run(hidden, activation="relu")
        last[hidden] = tail_mean(r["acc"])
        print(f"  {hidden} hidden: epoch 1 {pct(r['acc'][0])}, epoch 5 {pct(r['acc'][4])}, "
              f"last 5 {pct(last[hidden])}")
    span = 100 * (max(last.values()) - min(last.values()))
    print(f"  span across depths, last 5 epochs: {span:.1f} points")


def bench_dead(b):
    section(
        "6. Dead units at too large a step (ReLU, 4 hidden layers)",
        "at eta 0.5, 28 of the 30 neurons in layer 4 answer 0 on every training image "
        "after 1 epoch and all 30 by epoch 3, and the run sits between 9 and 13 percent; "
        "at 0.05 the same layer has 3 silent neurons after one epoch and 1 after three",
    )
    _, _, predict = make_engine(b.lib, b.ce, "relu")

    def silent_fractions(weights, biases):
        """Per hidden layer: the share of neurons that answer 0 on all 5,000 images.

        Layers are numbered Chapter 4's way, so the four hidden layers of a
        784-30-30-30-30-10 network are layers 2 to 5 and the output is layer 6.
        """
        out = {}
        a = b.X_train
        for i in range(len(weights) - 1):
            a = relu(weights[i] @ a + biases[i])
            out[i + 2] = float((a.max(axis=1) == 0).mean())
        return out

    for eta in (0.5, 0.05):
        for epochs in (1, 3):
            r = b.run(4, activation="relu", eta=eta, epochs=epochs)
            silent = silent_fractions(r["weights"], r["biases"])
            per_layer = ", ".join(f"layer {k} {pct(v)}" for k, v in silent.items())
            print(f"  eta {eta}, after {epochs} epoch(s): silent on every image: {per_layer}"
                  f"; test accuracy {pct(r['acc'][-1])}")
    r = b.run(4, activation="relu", eta=0.5)
    print(f"  eta 0.5, full run: accuracy spans {pct(min(r['acc']))} to {pct(max(r['acc']))}")


def bench_mistakes(b):
    section(
        "0. Chapter 5's run, and where its misses are",
        "108 of the thousand are read wrong; 1 comes back at 122 of 126 and 0 at "
        "82 of 85, while 8 manages 68 of 89; of the eight most confident mistakes, "
        "three are 3s read as 5 and two are 4s read as 9, all above 98 percent",
    )
    # Chapter 5's panel exactly: the learner's backprop inside their sgd, the
    # quadratic cost, eta 3.0, the undivided draw, init seed 8, shuffle seed 2.
    # The adapter written for the learner in Chapter 5, which is what the panel
    # calls: one backprop per column, slopes averaged, summed in that order.
    def grad(w, bs, X, Y):
        return b.lib.batch_gradient(w, bs, X, Y)

    sgd = load_sgd(b.lib, grad)
    r = np.random.default_rng(INIT_SEED)
    weights = [r.standard_normal((30, 784)), r.standard_normal((10, 30))]
    biases = [r.standard_normal((30, 1)), r.standard_normal((10, 1))]
    rng = np.random.default_rng(SHUFFLE_SEED)
    for _ in range(b.epochs):
        weights, biases = sgd.sgd(weights, biases, b.X_train, b.Y_train, 3.0, 1, BATCH, rng)

    out = b.course.feedforward(weights, biases, b.X_test)
    guesses = np.argmax(out, axis=0)
    print(f"  accuracy {pct(float((guesses == b.y_test).mean()))}, "
          f"{int((guesses != b.y_test).sum())} of {b.y_test.size} read wrong")
    for d in range(10):
        mask = b.y_test == d
        print(f"    {d}: {int(((guesses == b.y_test) & mask).sum())} of {int(mask.sum())}")
    wrong = np.flatnonzero(guesses != b.y_test)
    confidence = out[guesses[wrong], wrong]
    worst = wrong[np.argsort(-confidence)][:8]
    print("  the eight it was surest about, and wrong: "
          + ", ".join(f"{int(b.y_test[k])} read as {int(guesses[k])} "
                      f"({out[guesses[k], k]:.3f})" for k in worst))


def bench_module7(b):
    section(
        "7. Chapter 7's numbers that Chapter 8 opens by quoting",
        "the digit reader stands at 92.1 percent; the hidden layer's median squash slope at "
        "the divided start is 0.22",
    )
    r = b.run(1)
    print(f"  1 hidden layer, {b.epochs} epochs: final {pct(r['acc'][-1])}, "
          f"last 5 {pct(tail_mean(r['acc']))}")
    # InitStartPanel measures the two starts over ALL the training images, not
    # over the 100 the layer-speed table uses, so this mirrors that.
    rng = np.random.default_rng(INIT_SEED)
    plain = ([rng.standard_normal((30, 784)), rng.standard_normal((10, 30))],
             [rng.standard_normal((30, 1)), rng.standard_normal((10, 1))])
    divided = b.init.init_network([784, HIDDEN_SIZE, 10], np.random.default_rng(INIT_SEED))
    for label, (weights, biases) in (("the undivided start", plain), ("the divided start", divided)):
        z = weights[0] @ b.X_train + biases[0]
        a = b.course.sigmoid(z)
        steep = a * (1.0 - a)
        print(f"  {label}: typical |z| {float(np.abs(z).mean()):.2f}, "
              f"median squash slope {float(np.median(steep)):.4f}, "
              f"share flatter than 0.01 {pct(float((steep < 0.01).mean()))}")
        if label == "the undivided start":
            w = weights[0]
            print(f"    the draw itself: {pct(float((w < 0).mean()))} negative, "
                  f"typical size {float(np.abs(w).mean()):.1f}, "
                  f"{pct(float((np.abs(w) > 3).mean()))} past 3")
        # The distance histogram the chapter's first table quotes.
        d = np.abs(z)
        shares = " / ".join(
            f"{float(((d >= lo) & (d < hi)).mean()) * 100:.1f}%"
            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)))
        print(f"    distance bins 0-1 / 1-2 / 2-4 / 4-8 / 8+: {shares}")


def bench_regularize(b):
    section(
        "8. Chapter 7's weight decay, on the 1,000-image slice",
        "from the divided start, lambda 1 averages 85.7 over the last twenty epochs "
        "against 86.4 without it, where the epoch-80 rows read 84.7 and 86.3; two "
        "other shuffles of the unregularized run give 86.0 and 85.4; decay ends the "
        "held-out cost at 0.86 instead of 1.09 and the total of the squared weights "
        "at 655 instead of 1,926; the undivided start begins at 23,538 of squared "
        "weight and stays there without decay, 25,528 and 77.8 percent, against 679 "
        "and 85.1 percent with it",
    )
    # RegularizePanel's loop, written out the way the panel writes it rather
    # than called through sgd: the panel varies lambda, so the update line has
    # to be the learner's l2_step. The shuffle follows the sgd contract,
    # rng.permutation and then consecutive slices of ten, and each run draws
    # its start and its shuffle from fresh generators.
    epochs = 8 if b.quick else L2_EPOCHS
    tail = min(L2_TAIL, epochs)
    n = L2_SLICE
    X, y, Y = b.X_train[:, :n], b.y_train[:n], b.Y_train[:, :n]
    lib, sizes = b.lib, [784, HIDDEN_SIZE, 10]

    def squared(weights):
        return sum(float((w ** 2).sum()) for w in weights)

    def train(start, lmbda, shuffle_seed=SHUFFLE_SEED):
        weights, biases = b.start(sizes, start)
        rng = np.random.default_rng(shuffle_seed)
        accs = []
        for _ in range(epochs):
            order = rng.permutation(n)
            for k in range(0, n, BATCH):
                batch = order[k:k + BATCH]
                nabla_w, nabla_b = lib.batch_gradient(
                    weights, biases, X[:, batch], Y[:, batch], lib.cross_entropy_delta)
                weights, biases = lib.l2_step(
                    weights, biases, nabla_w, nabla_b, L2_ETA, lmbda, n)
            out = lib.feedforward(weights, biases, b.X_test)
            accs.append(float((np.argmax(out, axis=0) == b.y_test).mean()))
        return accs, weights, biases

    for start in ("divided", "undivided"):
        opening, _ = b.start(sizes, start)
        print(f"  the {start} start, {squared(opening):,.0f} of squared weight before a step:")
        for lmbda in (0.0, 1.0, 5.0):
            accs, weights, biases = train(start, lmbda)
            hits = np.argmax(lib.feedforward(weights, biases, X), axis=0) == y
            print(f"    lambda {lmbda:g}: last {tail} epochs {pct(tail_mean(accs, tail))}, "
                  f"epoch {epochs} {pct(accs[-1])}, on the {n:,} it trained on "
                  f"{pct(float(hits.mean()))}, held-out cost "
                  f"{lib.cross_entropy_cost(weights, biases, b.X_test, b.Y_test):.2f}, "
                  f"squared weights {squared(weights):,.0f}")
    for seed in L2_EXTRA_SHUFFLES:
        accs, _, _ = train("divided", 0.0, seed)
        print(f"  the divided start with no decay, shuffle seed {seed}: "
              f"last {tail} epochs {pct(tail_mean(accs, tail))}, "
              f"epoch {epochs} {pct(accs[-1])}")


def bench_grid(b):
    section(
        "9. Chapter 7's hyperparameter grid: two costs, two starts, four step sizes",
        "at eta 0.5 / 1.0 / 3.0 / 6.0 the quadratic cost from the undivided start "
        "reads 70.8 / 79.4 / 89.2 / 89.9, cross-entropy from the undivided start "
        "87.8 / 89.3 / 87.6 / 80.9, the quadratic cost from the divided start "
        "90.0 / 90.3 / 91.0 / 91.3, and cross-entropy from the divided start "
        "92.1 / 92.3 / 87.7 / 69.4; the best of each row runs 89.9, 89.3, 91.3, 92.3, "
        "and two of the four rows span nineteen points or more",
    )
    # Two of the sixteen cells are covered elsewhere in this file and should
    # print the same numbers here: the quadratic cost from the undivided start
    # at 3.0 is Chapter 5's own run (bench_mistakes, 89.2), and cross-entropy
    # from the divided start at 0.5 is the 92.1 Chapter 8 opens by quoting
    # (bench_module7). The table quotes the epoch-15 score, so that is what
    # `best` reads; the last-five average is beside it because one epoch of
    # one run is the statistic Chapter 8 says wobbles by a point or two.
    for cost, start in GRID_SETUPS:
        finals = []
        cells = []
        for eta in GRID_ETAS:
            r = b.run(1, eta=eta, cost=cost, start=start)
            finals.append(r["acc"][-1])
            cells.append(f"eta {eta}: {pct(r['acc'][-1])} (last 5 {pct(tail_mean(r['acc']))})")
        print(f"  {cost} cost, the {start} start: " + " | ".join(cells))
        print(f"    best {pct(max(finals))} at eta {GRID_ETAS[int(np.argmax(finals))]}, "
              f"worst {pct(min(finals))}, spanning "
              f"{100 * (max(finals) - min(finals)):.1f} points")


def bench_capstone(b):
    section(
        "10. Chapter 9's panel: the learner's own loop on the digit reader",
        "at 0.5 the last five epochs average 90.4; at 0.1 the first epoch reads 77.2 "
        "against 0.5's 85.5, and by the end the two are level; at 3.0 the last five "
        "average 86.6, and consecutive passes read 80.9, then 89.3, then 85.1",
    )
    # FullTrainPanel's exact configuration: ONE generator seeded at 8 draws the
    # network and then shuffles every epoch, lambda is 1, and the loop is the
    # capstone exercise's own.
    prog = b.lib
    for eta in (0.1, 0.5, 3.0):
        _, _, history = prog.train(
            [784, HIDDEN_SIZE, 10], b.X_train, b.Y_train, b.X_test, b.y_test,
            b.epochs, eta, 1.0, BATCH, np.random.default_rng(INIT_SEED))
        print(f"  eta {eta}: epoch 1 {pct(history[0])}, final {pct(history[-1])}, "
              f"last 5 {pct(tail_mean(history))}, per-epoch "
              + " ".join(pct(a) for a in history))


SECTIONS = {
    "mistakes": bench_mistakes,
    "depth": bench_depth,
    "first-epoch": bench_first_epoch,
    "eta": bench_eta,
    "ratios": bench_ratios,
    "relu": bench_relu,
    "dead": bench_dead,
    "module7": bench_module7,
    "regularize": bench_regularize,
    "grid": bench_grid,
    "capstone": bench_capstone,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="5 epochs instead of 15, and 8 instead of 80 for the weight decay")
    ap.add_argument("--only", nargs="+", choices=sorted(SECTIONS),
                    help="run only these sections")
    args = ap.parse_args()
    b = Bench(epochs=5 if args.quick else EPOCHS, quick=args.quick)
    print(f"NumPy {np.__version__}; {b.X_train.shape[1]} training and "
          f"{b.X_test.shape[1]} test images; {b.epochs} epochs; "
          f"init seed {INIT_SEED}, shuffle seed {SHUFFLE_SEED}")
    for name in (args.only or list(SECTIONS)):
        SECTIONS[name](b)
    print()


if __name__ == "__main__":
    main()
