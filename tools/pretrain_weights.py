#!/usr/bin/env python3
"""Pretrain the Module 2 payoff network and write public/data/pretrained_weights.json.gz.

Trains a 784-15-10 sigmoid network (15 hidden so the Module 2 diagram stays
compact, per the design doc) on the bundled MNIST subset with the reference
implementation, then saves weights as gzipped JSON so both the JS
visualizations (weight-image patches) and Python (running the learner's
feedforward) can read them without an npz parser.

Requires NumPy (unlike make_mnist_subset.py). Deterministic: fixed seed.

    python3 tools/pretrain_weights.py
"""

import gzip
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "public" / "data" / "mnist_subset.bin.gz"
OUT = ROOT / "public" / "data" / "pretrained_weights.json.gz"

SIZES = [784, 15, 10]
EPOCHS = 30
ETA = 3.0
MINI_BATCH = 10
SEED = 1


def load_course_python():
    """Exec the shared course Python files (they are Pyodide-first, but run
    anywhere NumPy exists)."""
    ns = {}
    for name in ["data_loader.py", "reference_network.py"]:
        exec((ROOT / "src" / "python" / name).read_text(), ns)
    return ns


def main() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("this script needs NumPy: pip install numpy (or use a venv)")

    ns = load_course_python()
    buf = gzip.decompress(DATA.read_bytes())
    X_train, y_train, X_test, y_test = ns["load_mnist_subset"](buf)
    Y_train = ns["one_hot"](y_train)

    net = ns["Network"](SIZES, seed=SEED)
    accuracy = net.sgd(
        X_train, Y_train, EPOCHS, MINI_BATCH, ETA,
        X_test=X_test, y_test=y_test,
        on_epoch=lambda e, n, loss, acc, t: print(
            f"epoch {e}/{n}: loss {loss:.4f}, test accuracy {acc:.3f}"),
    )

    payload = {
        "sizes": SIZES,
        "weights": [[[round(float(v), 6) for v in row] for row in w] for w in net.weights],
        "biases": [[round(float(v), 6) for v in b.ravel()] for b in net.biases],
        "test_accuracy": round(float(accuracy), 4),
        "seed": SEED,
        "epochs": EPOCHS,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    with open(OUT, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0) as gz:
            gz.write(raw)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, "
          f"test accuracy {accuracy:.3f})")


if __name__ == "__main__":
    main()
