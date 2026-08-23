# Reference solution, adapted from Michael Nielsen's network.py (MIT license,
# github.com/mnielsen/neural-networks-and-deep-learning).

import numpy as np


def sigmoid(z):
    """Return 1 / (1 + e^(-z)), elementwise."""
    return 1.0 / (1.0 + np.exp(-z))


def fire(w, b, x):
    """Return this neuron's output for input x, as a plain float."""
    z = (w * x).sum() + b
    return float(sigmoid(z))
