# Reference solution. The cross-entropy cost is from Michael Nielsen's
# network2.py (MIT license, github.com/mnielsen/neural-networks-and-deep-learning);
# he plugs the 0 * log(0) hole with np.nan_to_num instead of clipping.

import numpy as np
from course import feedforward


def cross_entropy_cost(weights, biases, X, Y):
    """Mean cross-entropy cost of the batch (X, Y), as a float."""
    A = feedforward(weights, biases, X)
    A = np.clip(A, 1e-12, 1.0 - 1e-12)
    per_entry = -(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))
    m = X.shape[1]
    return float(per_entry.sum()) / m


def cross_entropy_delta(a, y, z):
    """The output layer's blame under the cross-entropy cost."""
    return a - y
