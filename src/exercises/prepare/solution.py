# Reference solution. Nothing here is borrowed: standardizing, one-hot
# encoding and splitting are the same three moves in every framework's data
# tooling, written out.

import numpy as np


def standardize(X, mean=None, spread=None):
    """Centre and scale each feature; reuse a given mean and spread if handed one."""
    if mean is None:
        mean = X.mean(axis=1, keepdims=True)
    if spread is None:
        spread = X.std(axis=1, keepdims=True)
    # A feature that never varies would divide by zero; leaving it at its
    # centred value (all zeros) says the same thing without the NaN.
    safe = np.where(spread == 0, 1.0, spread)
    return (X - mean) / safe, mean, spread


def one_hot(values, levels):
    """Turn a list of category labels into rows of 0.0 and 1.0."""
    out = np.zeros((len(levels), len(values)))
    position = {level: i for i, level in enumerate(levels)}
    for column, value in enumerate(values):
        if value in position:
            out[position[value], column] = 1.0
    return out


def split(n, rng, val_share, test_share):
    """Shuffle 0..n-1 once and cut it into training, validation and test."""
    order = rng.permutation(n)
    n_val = round(n * val_share)
    n_test = round(n * test_share)
    n_train = n - n_val - n_test
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]
