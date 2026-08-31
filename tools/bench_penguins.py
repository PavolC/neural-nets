"""Reproduce every number Module 10 quotes, on the panel's own code path.

Module 10's claims are all about data preparation, so this bench runs the same
pipeline PenguinsPanel runs: the exercise's own standardize, one_hot and split,
the learner's train and accuracy from Module 9, and the bundled penguin file
exactly as it ships. Each section prints the prose sentence it backs.

Needs NumPy:

    python3 -m venv .venv && .venv/bin/pip install numpy
    .venv/bin/python tools/bench_penguins.py
"""

import collections
import gzip
import pathlib
import sys
import types

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import workbench as wb  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "public" / "data" / "penguins.json.gz"

# The panel's constants (src/modules/NN/interactives/PenguinsPanel.tsx).
HIDDEN = 8
EPOCHS = 30
ETA = 0.5
LMBDA = 0.0
BATCH = 10
SPLIT_SEED = 0
TRAIN_SEED = 1
VAL_SHARE = 0.2
TEST_SHARE = 0.2

ISLANDS = ["Biscoe", "Dream", "Torgersen"]
SEXES = ["female", "male"]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
ALL_FOUR = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
JUST_TWO = ["bill_depth_mm", "body_mass_g"]


def load_module(path, name):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(path.read_text(), path.name, "exec"), mod.__dict__)
    return mod


def bootstrap():
    """The panel's own code path: one file, exec'd once.

    prep and prog are the same module, because standardize, one_hot and split
    sit below train in the learner's file and train calls the pieces above it.
    Loading them as two separate modules, which is what this did before the
    workbench, benched a wiring the panel no longer has.
    """
    course = load_module(ROOT / "src" / "python" / "course_helpers.py", "course")
    sys.modules["course"] = course
    loader = load_module(ROOT / "src" / "python" / "data_loader.py", "data_loader")
    document = wb.assemble([s["id"] for s in wb.SECTIONS], "solution",
                           {"backprop": "seam"})
    lib = types.ModuleType("your_code")
    lib.__file__ = "your_code.py"
    exec(compile(document, "your_code.py", "exec"), lib.__dict__)
    with gzip.open(DATA, "rb") as f:
        columns, rows = loader.load_penguins(f.read())
    return course, lib, lib, columns, rows


def build(prep, columns, rows, features, scale):
    """The panel's preparation, start to finish."""
    index = {name: i for i, name in enumerate(columns)}
    usable = [r for r in rows if all(r[index[f]] is not None for f in ALL_FOUR)]

    numbers = np.array([[float(r[index[f]]) for f in features] for r in usable]).T
    train_idx, val_idx, test_idx = prep.split(
        len(usable), np.random.default_rng(SPLIT_SEED), VAL_SHARE, TEST_SHARE
    )
    if scale:
        # Measured on the training columns only, then applied to all of them.
        _, mean, spread = prep.standardize(numbers[:, train_idx])
        numbers, _, _ = prep.standardize(numbers, mean, spread)

    # Categorical rows are already 0 and 1, so they are stacked after the
    # scaling rather than through it.
    island = prep.one_hot([r[index["island"]] for r in usable], ISLANDS)
    sex = prep.one_hot([r[index["sex"]] for r in usable], SEXES)
    X = np.vstack([numbers, island, sex])

    y = np.array([SPECIES.index(r[index["species"]]) for r in usable])
    Y = np.zeros((len(SPECIES), len(usable)))
    Y[y, np.arange(len(usable))] = 1.0
    return X, Y, y, train_idx, val_idx, test_idx, usable


def run(prog, X, Y, y, train_idx, val_idx, test_idx):
    weights, biases, history = prog.train(
        [X.shape[0], HIDDEN, len(SPECIES)],
        X[:, train_idx], Y[:, train_idx], X[:, val_idx], y[val_idx],
        EPOCHS, ETA, LMBDA, BATCH, np.random.default_rng(TRAIN_SEED),
    )
    test = prog.accuracy(weights, biases, X[:, test_idx], y[test_idx])
    return weights, biases, history, test


def pct(x):
    return f"{100 * x:.1f}%"


def section(title, claim):
    print()
    print("=" * 78)
    print(title)
    print("prose:", claim)
    print("-" * 78)


def main():
    course, prep, prog, columns, rows = bootstrap()
    index = {name: i for i, name in enumerate(columns)}
    usable = [r for r in rows if all(r[index[f]] is not None for f in ALL_FOUR)]

    section(
        "1. The file as it ships",
        "344 rows, 2 with no measurements at all, 11 with no sex; the four "
        "measurements sit on scales 245 times apart",
    )
    holes = collections.Counter()
    for r in rows:
        for c in columns:
            if r[index[c]] is None:
                holes[c] += 1
    print(f"  {len(rows)} rows, {len(usable)} with all four measurements")
    print(f"  holes: {dict(holes)}")
    print(f"  species: {dict(collections.Counter(r[index['species']] for r in rows))}")
    for f in ALL_FOUR:
        vals = np.array([float(r[index[f]]) for r in usable])
        print(f"  {f:18} mean {vals.mean():8.2f}  spread {vals.std():7.2f}")
    means = [np.mean([float(r[index[f]]) for r in usable]) for f in ALL_FOUR]
    print(f"  largest mean over smallest: {max(means) / min(means):.0f} times")

    section(
        "2. Scaling is the whole difference",
        "unscaled the network lands on the majority-class baseline and learns "
        "nothing else; scaled it reads every held-out penguin",
    )
    for scale in (False, True):
        X, Y, y, tr, va, te = build(prep, columns, rows, ALL_FOUR, scale)[:6]
        counts = collections.Counter(y[tr].tolist())
        major = max(counts, key=counts.get)
        baseline = float((y[te] == major).mean())
        _, _, history, test = run(prog, X, Y, y, tr, va, te)
        label = "scaled" if scale else "as it comes"
        print(f"  {label:12} epoch 1 {pct(history[0])}, validation {pct(history[-1])}, "
              f"test {pct(test)}  (baseline {pct(baseline)}, "
              f"{SPECIES[major]} is the commonest of {len(tr)} training rows)")

    section(
        "3. Fewer features, and errors worth looking at",
        "with only bill depth and body mass the score falls to 73.5 percent, "
        "and the mistakes are Adelie and Chinstrap",
    )
    X, Y, y, tr, va, te = build(prep, columns, rows, JUST_TWO, True)[:6]
    weights, biases, history, test = run(prog, X, Y, y, tr, va, te)
    print(f"  two measurements: validation {pct(history[-1])}, test {pct(test)}")
    guesses = np.argmax(course.feedforward(weights, biases, X[:, te]), axis=0)
    truth = y[te]
    for i, name in enumerate(SPECIES):
        mask = truth == i
        if mask.any():
            print(f"    {name:10} {int((guesses[mask] == i).sum())}/{int(mask.sum())} right")
    confusions = collections.Counter(
        (SPECIES[t], SPECIES[g]) for t, g in zip(truth, guesses) if t != g
    )
    for (was, said), count in confusions.most_common():
        print(f"    {count} {was} read as {said}")

    section(
        "4. The split has to be shuffled",
        "the file is sorted by species, so an unshuffled cut trains on two "
        "species and tests on a third",
    )
    order = [r[index["species"]] for r in usable]
    n = len(order)
    # The exercise's own cut, not an approximation of it: round(n * share) for
    # each held-back set, taken from the END, training keeps the rest. int(0.6
    # * n) put the boundaries one row off and quoted a Gentoo into the test set
    # that the reader's own split never sends there.
    n_val = round(n * VAL_SHARE)
    n_test = round(n * TEST_SHARE)
    a = n - n_val - n_test
    b = a + n_val
    print(f"  first 10 rows of the file: {order[:10]}")
    print(f"  an unshuffled 60/20/20 cut would give:")
    print(f"    train {dict(collections.Counter(order[:a]))}")
    print(f"    val   {dict(collections.Counter(order[a:b]))}")
    print(f"    test  {dict(collections.Counter(order[b:]))}")
    print()


if __name__ == "__main__":
    main()
