"""Helpers the course provides to exercises via `from course import ...`.

These are functions the learner has already built in an earlier module (or
that the course supplies as given). The harness registers this file as the
`course` module inside Pyodide, so skeletons never contain solution logic
for the current exercise (see CLAUDE.md hard rules).

Shape conventions (see CLAUDE.md): activations are column vectors, batches
stack samples as columns, weights[l] has shape (size of layer l+1, size of
layer l).
"""

import numpy as np


def sigmoid(z):
    """The sigmoid function 1 / (1 + exp(-z)), applied elementwise.

    Built by the learner in Module 1.
    """
    return 1.0 / (1.0 + np.exp(-z))


def feedforward(weights, biases, x):
    """Return the network's output activation for input x.

    Built by the learner in Module 2. Works on a single column vector
    (n, 1) or a batch (n, m).
    """
    a = x
    for w, b in zip(weights, biases):
        a = sigmoid(w @ a + b)
    return a


def quadratic_loss(weights, biases, X, Y):
    """Mean quadratic cost over a batch: 0.5 * sum((a - y)^2) / m.

    X is (n_in, m), Y is (n_out, m). Returns a float.
    """
    out = feedforward(weights, biases, X)
    m = X.shape[1]
    return 0.5 * float(((out - Y) ** 2).sum()) / m


def numerical_gradient(loss_fn, weights, biases, eps=1e-5):
    """Estimate gradients by central finite differences.

    loss_fn(weights, biases) must return a float. Returns (nabla_w,
    nabla_b): lists of arrays with the same shapes as weights and biases,
    where each entry is the estimated partial derivative of the loss with
    respect to that parameter.

    This is the expensive way to get gradients: two loss evaluations (two
    full forward passes) per parameter. Module 5 replaces it with
    backpropagation.
    """
    def grad_of(params, index):
        p = params[index]
        g = np.zeros_like(p, dtype=float)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            original = p[i]
            p[i] = original + eps
            up = loss_fn(weights, biases)
            p[i] = original - eps
            down = loss_fn(weights, biases)
            p[i] = original
            g[i] = (up - down) / (2 * eps)
        return g

    nabla_w = [grad_of(weights, l) for l in range(len(weights))]
    nabla_b = [grad_of(biases, l) for l in range(len(biases))]
    return nabla_w, nabla_b
