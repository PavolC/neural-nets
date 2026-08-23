# Reference solution, adapted from Michael Nielsen's network.py (MIT license,
# github.com/mnielsen/neural-networks-and-deep-learning).

import numpy as np
from course import quadratic_cost, gradient


def sgd_step(weights, biases, X, Y, eta):
    """One gradient descent step on the batch (X, Y)."""
    nabla_w, nabla_b = gradient(weights, biases, X, Y)
    new_weights = [w - eta * nw for w, nw in zip(weights, nabla_w)]
    new_biases = [b - eta * nb for b, nb in zip(biases, nabla_b)]
    return new_weights, new_biases


def sgd(weights, biases, X, Y, eta, epochs, batch_size, rng):
    """Mini-batch SGD over the whole dataset."""
    n = X.shape[1]
    for _ in range(epochs):
        idx = rng.permutation(n)
        for k in range(0, n, batch_size):
            batch = idx[k : k + batch_size]
            weights, biases = sgd_step(weights, biases, X[:, batch], Y[:, batch], eta)
    return weights, biases
