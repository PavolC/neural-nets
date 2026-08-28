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


def quadratic_cost(weights, biases, X, Y):
    """Mean quadratic cost over a batch: 0.5 * sum((a - y)^2) / m.

    X is (n_in, m), Y is (n_out, m). Returns a float.
    """
    out = feedforward(weights, biases, X)
    m = X.shape[1]
    return 0.5 * float(((out - Y) ** 2).sum()) / m


# Old name for quadratic_cost, kept so learner code saved before the
# rename keeps running. New material must say "cost", never "loss".
quadratic_loss = quadratic_cost


def gradient(weights, biases, X, Y, eps=1e-5):
    """Slopes of the quadratic cost on the batch (X, Y), one per parameter.

    Takes the same inputs as quadratic_cost and returns (nabla_w,
    nabla_b): lists of arrays shaped like weights and biases, where each
    entry is that parameter's slope. Measured numerically, the hand way:
    nudge the parameter by eps, rescore the batch, divide (two cost
    evaluations per parameter; Module 5 replaces this with
    backpropagation).
    """
    def cost_fn(ws, bs):
        return quadratic_cost(ws, bs, X, Y)

    return numerical_gradient(cost_fn, weights, biases, eps)


def numerical_gradient(cost_fn, weights, biases, eps=1e-5):
    """Estimate gradients by central finite differences.

    cost_fn(weights, biases) must return a float. Returns (nabla_w,
    nabla_b): lists of arrays with the same shapes as weights and biases,
    where each entry is the estimated partial derivative of the cost with
    respect to that parameter.

    This is the expensive way to get gradients: two cost evaluations (two
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
            up = cost_fn(weights, biases)
            p[i] = original - eps
            down = cost_fn(weights, biases)
            p[i] = original
            g[i] = (up - down) / (2 * eps)
        return g

    nabla_w = [grad_of(weights, l) for l in range(len(weights))]
    nabla_b = [grad_of(biases, l) for l in range(len(biases))]
    return nabla_w, nabla_b


def sigmoid_prime(z):
    """Sigmoid's steepness at z: sigmoid(z) * (1 - sigmoid(z)), elementwise.

    Built by the learner in Module 5.
    """
    s = sigmoid(z)
    return s * (1.0 - s)


def quadratic_output_delta(a, y, z):
    """BP1 under the quadratic cost: the gap times the output's steepness.

    The output layer's blame, exactly as the learner wrote it in Module 5's
    backprop. Provided as a function so it can be swapped (see backprop's
    output_delta argument).
    """
    return (a - y) * sigmoid_prime(z)


def backprop(weights, biases, x, y, output_delta=None):
    """Every parameter's slope for ONE example, via BP1 to BP4.

    The learner's Module 5 algorithm with BP1 lifted into an argument:
    output_delta(a, y, z) supplies the output layer's blame. It defaults to
    quadratic_output_delta, which reproduces Module 5's function exactly;
    Module 7 passes a different one. Everything after BP1 is untouched,
    which is the point.

    x is one input column (n_in, 1), y its right answer (n_out, 1).
    Returns (nabla_w, nabla_b), lists shaped like weights and biases.

    Adapted from Michael Nielsen's network.py (MIT license).
    """
    if output_delta is None:
        output_delta = quadratic_output_delta

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

    delta = output_delta(activations[-1], y, zs[-1])   # BP1
    nabla_b[-1] = delta                               # BP3
    nabla_w[-1] = delta @ activations[-2].T            # BP4

    for l in range(2, len(weights) + 1):
        delta = (weights[-l + 1].T @ delta) * sigmoid_prime(zs[-l])  # BP2
        nabla_b[-l] = delta
        nabla_w[-l] = delta @ activations[-l - 1].T

    return nabla_w, nabla_b


def cross_entropy_delta(a, y, z):
    """BP1 under the cross-entropy cost: the gap, with nothing multiplying it.

    Built by the learner in Module 7. z is in the signature because backprop
    passes it; this blame has no use for it.
    """
    return a - y


def init_network(sizes, rng):
    """A fresh network: weights divided by the square root of their inputs.

    Built by the learner in Module 7. sizes runs front to back, so
    [784, 30, 10] is the digit reader. Draw order is part of the contract:
    every weight matrix first, front to back, then every bias column, so
    that a given seed builds the same network here as it does there.

    The 1/sqrt(n) scaling is from Nielsen's network2.py (MIT license).
    """
    weights = [
        rng.standard_normal((sizes[i + 1], sizes[i])) / np.sqrt(sizes[i])
        for i in range(len(sizes) - 1)
    ]
    biases = [rng.standard_normal((sizes[i + 1], 1)) for i in range(len(sizes) - 1)]
    return weights, biases


def l2_step(weights, biases, nabla_w, nabla_b, eta, lmbda, n):
    """One descent step with weight decay.

    Built by the learner in Module 7. Every weight is multiplied by
    (1 - eta * lmbda / n) before the usual step; biases keep Module 3's
    rule, since the regularization term does not mention them. n is the
    size of the whole training set, not of this mini-batch.
    """
    decay = 1.0 - eta * lmbda / n
    new_weights = [decay * w - eta * nw for w, nw in zip(weights, nabla_w)]
    new_biases = [b - eta * nb for b, nb in zip(biases, nabla_b)]
    return new_weights, new_biases


def batch_gradient(weights, biases, X, Y, output_delta=None):
    """A mini-batch's gradient: backprop per column, slopes averaged.

    The adapter Module 5's training panel used, in one place. X is
    (n_in, m), Y is (n_out, m). Returns (nabla_w, nabla_b).
    """
    m = X.shape[1]
    nabla_w = [np.zeros_like(w) for w in weights]
    nabla_b = [np.zeros_like(b) for b in biases]
    for k in range(m):
        dw, db = backprop(weights, biases, X[:, k:k + 1], Y[:, k:k + 1], output_delta)
        nabla_w = [t + d for t, d in zip(nabla_w, dw)]
        nabla_b = [t + d for t, d in zip(nabla_b, db)]
    return [t / m for t in nabla_w], [t / m for t in nabla_b]
