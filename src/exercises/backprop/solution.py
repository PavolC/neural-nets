# Reference solution, adapted from Michael Nielsen's network.py (MIT license,
# github.com/mnielsen/neural-networks-and-deep-learning).

import numpy as np
from course import sigmoid


def sigmoid_prime(z):
    """Sigmoid's steepness at z: sigmoid(z) * (1 - sigmoid(z)), elementwise."""
    s = sigmoid(z)
    return s * (1.0 - s)


def backprop(weights, biases, x, y):
    """Exact slopes of one example's quadratic cost, via BP1 to BP4."""
    # Forward pass, keeping receipts: every z and every activation.
    a = x
    activations = [x]
    zs = []
    for w, b in zip(weights, biases):
        z = w @ a + b
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)

    nabla_w = [np.zeros_like(w) for w in weights]
    nabla_b = [np.zeros_like(b) for b in biases]

    # BP1: the output layer's blame.
    delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta                       # BP3
    nabla_w[-1] = delta @ activations[-2].T   # BP4

    # BP2: carry the blame backward, one layer at a time.
    for l in range(2, len(weights) + 1):
        delta = (weights[-l + 1].T @ delta) * sigmoid_prime(zs[-l])
        nabla_b[-l] = delta                        # BP3
        nabla_w[-l] = delta @ activations[-l - 1].T  # BP4

    return nabla_w, nabla_b
