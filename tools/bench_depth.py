"""Reproduce every Python-side number Modules 5, 7 and 8 quote, on the browser's code path.

Module 8's prose reports measurements from two engines, and this script is the
first of the two benches that regenerate them:

  * tools/bench_depth.py (this file) mirrors DepthTrainPanel's snippet: the
    learner's own init_network, sgd and cross-entropy blame, the course's
    backprop, and the bundled MNIST subset. It produces the accuracy tables,
    the step-size sweep, the ReLU comparison, the dead-unit counts and the
    learning-speed ratios measured during training.
  * tools/bench_layer_speeds.ts mirrors LayerSpeedBars: the same measurement
    written in TypeScript, which is what the module's layer-speed tables and
    hop factors are quoted from.

Two traps make a "mathematically equivalent" bench print different numbers than
the browser (both were found while writing Module 8, and both are why this file
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
is visible without holding the module open beside it.
"""

import argparse
import gzip
import pathlib
import sys
import types

import numpy as np

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


def load_module(path, name):
    """Execute a course .py file as a module, the way the worker does."""
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(path.read_text(), path.name, "exec"), mod.__dict__)
    return mod


def bootstrap():
    """Register `course` and load the three exercise solutions the panel uses.

    Order matters: the sgd solution does `from course import gradient`, which
    binds the function at import time, so `course.gradient` has to be pointed
    at the gradient under test BEFORE sgd is loaded. The panel does exactly
    this, and getting it wrong silently benches the numerical gradient.
    """
    course = load_module(PY / "course_helpers.py", "course")
    sys.modules["course"] = course
    loader = load_module(PY / "data_loader.py", "data_loader")
    ce = load_module(EX / "cross-entropy" / "solution.py", "your_cost")
    init = load_module(EX / "smart-init" / "solution.py", "your_init")
    return course, loader, ce, init


def load_sgd(course, grad_fn):
    """Load the learner's sgd on top of a chosen gradient (see bootstrap)."""
    course.gradient = grad_fn
    return load_module(EX / "sgd" / "solution.py", "your_sgd")


def relu(z):
    return np.maximum(0.0, z)


def make_engine(course, ce, activation):
    """The panel's two code paths, returned as (per_example, batch, predict).

    With the sigmoid the gradient is course.backprop with the cross-entropy
    blame, which is the learner's Module 5 algorithm. With ReLU it is the same
    four equations with one line changed, because a learner's BP2 has
    sigmoid_prime written into it.
    """
    sigmoid = course.sigmoid

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

    def sigmoid_grad(weights, biases, X, Y):
        return course.batch_gradient(weights, biases, X, Y, ce.cross_entropy_delta)

    return None, sigmoid_grad, course.feedforward


def layer_speeds(grad, weights, biases, X, Y, n=SPEED_SAMPLES):
    """Every layer's ||dC/db|| on the panel's first n training images."""
    _, nabla_b = grad(weights, biases, X[:, :n], Y[:, :n])
    return [float(np.linalg.norm(g)) for g in nabla_b]


class Bench:
    def __init__(self, epochs):
        self.epochs = epochs
        self.course, loader, self.ce, self.init = bootstrap()
        with gzip.open(DATA, "rb") as f:
            raw = f.read()
        self.X_train, self.y_train, self.X_test, self.y_test = loader.load_mnist_subset(raw)
        self.Y_train = loader.one_hot(self.y_train)

    def run(self, hidden, activation="sigmoid", eta=None, epochs=None,
            init_seed=INIT_SEED, weight_scale=1.0, watch=()):
        """One training run, reporting per-epoch test accuracy.

        watch: epochs after which to record every layer's learning speed.
        """
        eta = ETA[activation] if eta is None else eta
        epochs = self.epochs if epochs is None else epochs
        _, grad, predict = make_engine(self.course, self.ce, activation)
        sgd = load_sgd(self.course, grad)
        sizes = [784] + [HIDDEN_SIZE] * hidden + [10]
        weights, biases = self.init.init_network(sizes, np.random.default_rng(init_seed))
        if weight_scale != 1.0:
            weights = [w * weight_scale for w in weights]
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
        "at eta 0.5, 97% of layer 4's neurons answer 0 on every training image after "
        "1 epoch and all of them by epoch 3, and the run sits between 9 and 13 percent; "
        "at 0.05 the same layer is 17% silent after 1 epoch and 7% after 3",
    )
    _, _, predict = make_engine(b.course, b.ce, "relu")

    def silent_fractions(weights, biases):
        """Per hidden layer: the share of neurons that answer 0 on all 5,000 images.

        Layers are numbered Module 4's way, so the four hidden layers of a
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
        "0. Module 5's run, and where its misses are",
        "108 of the thousand are read wrong; 1 comes back at 122 of 126 and 0 at "
        "82 of 85, while 8 manages 68 of 89; of the eight most confident mistakes, "
        "three are 3s read as 5 and two are 4s read as 9, all above 98 percent",
    )
    # Module 5's panel exactly: the learner's backprop inside their sgd, the
    # quadratic cost, eta 3.0, the undivided draw, init seed 8, shuffle seed 2.
    bp = load_module(EX / "backprop" / "solution.py", "your_backprop")

    def grad(w, bs, X, Y):
        m = X.shape[1]
        nw = [np.zeros_like(x) for x in w]
        nb = [np.zeros_like(x) for x in bs]
        for k in range(m):
            dw, db = bp.backprop(w, bs, X[:, k:k + 1], Y[:, k:k + 1])
            nw = [t + d for t, d in zip(nw, dw)]
            nb = [t + d for t, d in zip(nb, db)]
        return [t / m for t in nw], [t / m for t in nb]

    sgd = load_sgd(b.course, grad)
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
        "7. Module 7's numbers that Module 8 opens by quoting",
        "the digit reader stands at 92.1 percent; the median hidden-neuron steepness at "
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
    for label, (weights, biases) in (("Module 5's start", plain), ("the divided start", divided)):
        z = weights[0] @ b.X_train + biases[0]
        a = b.course.sigmoid(z)
        steep = a * (1.0 - a)
        print(f"  {label}: typical |z| {float(np.abs(z).mean()):.2f}, "
              f"median steepness {float(np.median(steep)):.4f}, "
              f"share flatter than 0.01 {pct(float((steep < 0.01).mean()))}")


def bench_capstone(b):
    section(
        "8. Module 9's panel: the learner's own loop on the digit reader",
        "at 0.5 it lands near 90 percent; at 0.1 it is still climbing at the end; "
        "at 3.0 it is worse than either",
    )
    # FullTrainPanel's exact configuration: ONE generator seeded at 8 draws the
    # network and then shuffles every epoch, lambda is 1, and the loop is the
    # capstone exercise's own.
    prog = load_module(EX / "train" / "solution.py", "your_program")
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
    "capstone": bench_capstone,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="5 epochs instead of 15")
    ap.add_argument("--only", nargs="+", choices=sorted(SECTIONS),
                    help="run only these sections")
    args = ap.parse_args()
    b = Bench(epochs=5 if args.quick else EPOCHS)
    print(f"NumPy {np.__version__}; {b.X_train.shape[1]} training and "
          f"{b.X_test.shape[1]} test images; {b.epochs} epochs; "
          f"init seed {INIT_SEED}, shuffle seed {SHUFFLE_SEED}")
    for name in (args.only or list(SECTIONS)):
        SECTIONS[name](b)
    print()


if __name__ == "__main__":
    main()
