# Reference solution, adapted from Michael Nielsen's network2.py (MIT license,
# github.com/mnielsen/neural-networks-and-deep-learning).

import numpy as np


def l2_step(weights, biases, nabla_w, nabla_b, eta, lmbda, n):
    """One descent step with the weights decayed toward zero."""
    decay = 1.0 - eta * lmbda / n
    new_weights = [decay * w - eta * nw for w, nw in zip(weights, nabla_w)]
    new_biases = [b - eta * nb for b, nb in zip(biases, nabla_b)]
    return new_weights, new_biases
