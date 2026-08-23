#!/usr/bin/env python3
"""Build public/data/mnist_subset.bin.gz from the canonical MNIST dataset.

Pure Python stdlib (no NumPy required). Downloads the four MNIST IDX files
from a public mirror (cached in tools/.cache/), takes the first N_TRAIN
training and N_TEST test examples, and writes a single gzipped binary asset.

Output format (all integers little-endian, then gzipped as a whole):

    magic    4 bytes   b"MNSS"
    version  1 byte    0x01
    n_train  uint32
    n_test   uint32
    rows     uint32    (28)
    cols     uint32    (28)
    train_images  n_train * rows * cols  uint8  (row-major per image)
    train_labels  n_train                uint8
    test_images   n_test * rows * cols   uint8
    test_labels   n_test                 uint8

The subset is the first N examples of each split, which is deterministic and
roughly class-balanced. Written with gzip mtime=0 so output is reproducible.
"""

import gzip
import struct
import sys
import urllib.request
from pathlib import Path

N_TRAIN = 5000
N_TEST = 1000

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tools" / ".cache"
OUT = ROOT / "public" / "data" / "mnist_subset.bin.gz"


def fetch(filename: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / filename
    if cached.exists():
        return cached.read_bytes()
    last_err = None
    for mirror in MIRRORS:
        url = mirror + filename
        try:
            print(f"downloading {url} ...")
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            cached.write_bytes(data)
            return data
        except Exception as err:  # try next mirror
            last_err = err
            print(f"  failed: {err}", file=sys.stderr)
    raise RuntimeError(f"could not download {filename}") from last_err


def parse_idx_images(raw: bytes, n: int) -> bytes:
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051, f"bad image magic {magic}"
    assert count >= n, f"only {count} images, need {n}"
    assert (rows, cols) == (28, 28), f"unexpected size {rows}x{cols}"
    return raw[16 : 16 + n * rows * cols]


def parse_idx_labels(raw: bytes, n: int) -> bytes:
    magic, count = struct.unpack(">II", raw[:8])
    assert magic == 2049, f"bad label magic {magic}"
    assert count >= n, f"only {count} labels, need {n}"
    return raw[8 : 8 + n]


def main() -> None:
    train_images = parse_idx_images(gzip.decompress(fetch(FILES["train_images"])), N_TRAIN)
    train_labels = parse_idx_labels(gzip.decompress(fetch(FILES["train_labels"])), N_TRAIN)
    test_images = parse_idx_images(gzip.decompress(fetch(FILES["test_images"])), N_TEST)
    test_labels = parse_idx_labels(gzip.decompress(fetch(FILES["test_labels"])), N_TEST)

    header = b"MNSS" + struct.pack("<BIIII", 1, N_TRAIN, N_TEST, 28, 28)
    payload = header + train_images + train_labels + test_images + test_labels

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0) as gz:
            gz.write(payload)

    counts = [0] * 10
    for b in train_labels:
        counts[b] += 1
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    print(f"train label distribution: {counts}")


if __name__ == "__main__":
    main()
