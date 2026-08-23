"""Load the bundled MNIST subset (public/data/mnist_subset.bin.gz) into NumPy arrays.

The binary format is documented in tools/make_mnist_subset.py. The buffer passed
in is already gunzipped (the worker decompresses it with DecompressionStream).

Shape conventions (global, see CLAUDE.md): samples are columns, so image matrices
are (784, n) with float32 pixels in [0, 1]; labels are (n,) uint8 class ids.
"""

import struct

import numpy as np


def load_mnist_subset(buf):
    """Parse the decompressed asset. Returns (X_train, y_train, X_test, y_test)."""
    data = bytes(buf)
    magic = data[:4]
    if magic != b"MNSS":
        raise ValueError(f"bad magic {magic!r}, expected b'MNSS'")
    version, n_train, n_test, rows, cols = struct.unpack("<BIIII", data[4:21])
    if version != 1:
        raise ValueError(f"unsupported format version {version}")
    pixels = rows * cols

    offset = 21
    def take(count):
        nonlocal offset
        out = np.frombuffer(data, dtype=np.uint8, count=count, offset=offset)
        offset += count
        return out

    X_train = take(n_train * pixels).reshape(n_train, pixels).T.astype(np.float32) / 255.0
    y_train = take(n_train).copy()
    X_test = take(n_test * pixels).reshape(n_test, pixels).T.astype(np.float32) / 255.0
    y_test = take(n_test).copy()
    return X_train, y_train, X_test, y_test


def one_hot(y, num_classes=10):
    """(n,) class ids -> (num_classes, n) one-hot float32 columns."""
    Y = np.zeros((num_classes, y.size), dtype=np.float32)
    Y[y, np.arange(y.size)] = 1.0
    return Y
