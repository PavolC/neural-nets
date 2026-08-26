# Reference solution. The 1/sqrt(n) scaling is from Michael Nielsen's
# network2.py (MIT license, github.com/mnielsen/neural-networks-and-deep-learning).

import numpy as np


def init_network(sizes, rng):
    """Weights scaled by 1/sqrt(inputs), biases standard normal."""
    weights = [
        rng.standard_normal((sizes[i + 1], sizes[i])) / np.sqrt(sizes[i])
        for i in range(len(sizes) - 1)
    ]
    biases = [rng.standard_normal((sizes[i + 1], 1)) for i in range(len(sizes) - 1)]
    return weights, biases
