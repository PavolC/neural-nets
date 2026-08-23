"""Reference sigmoid network for the Milestone 0 feasibility spike.

Adapted from Michael Nielsen's network.py (MIT license,
github.com/mnielsen/neural-networks-and-deep-learning), vectorized over
mini-batches (samples as columns) so training is fast enough inside Pyodide.

Shape conventions (global, see CLAUDE.md): activations are column vectors,
a mini-batch X is (784, m), weights[l] is (size of layer l+1, size of layer l).
Deterministic: all randomness comes from a seeded np.random.default_rng.
"""

import time

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


class Network:
    def __init__(self, sizes, seed=1):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.biases = [rng.standard_normal((y, 1)).astype(np.float32) for y in sizes[1:]]
        self.weights = [
            rng.standard_normal((y, x)).astype(np.float32)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]
        self.rng = rng

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def backprop_batch(self, X, Y):
        """Gradients of the quadratic cost for a whole mini-batch at once.

        X is (784, m), Y is (10, m) one-hot. Returns (nabla_w, nabla_b,
        batch_loss) where gradients are already summed over the batch.
        """
        m = X.shape[1]
        activation = X
        activations = [X]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        nabla_b = [None] * len(self.biases)
        nabla_w = [None] * len(self.weights)
        delta = (activations[-1] - Y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)
        nabla_w[-1] = delta @ activations[-2].T
        for l in range(2, len(self.sizes)):
            delta = (self.weights[-l + 1].T @ delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta.sum(axis=1, keepdims=True)
            nabla_w[-l] = delta @ activations[-l - 1].T

        batch_loss = 0.5 * float(((activations[-1] - Y) ** 2).sum()) / m
        return nabla_w, nabla_b, batch_loss

    def evaluate(self, X_test, y_test):
        """Fraction of test inputs classified correctly (argmax of output)."""
        preds = np.argmax(self.feedforward(X_test), axis=0)
        return float((preds == y_test).mean())

    def sgd(self, X_train, Y_train, epochs, mini_batch_size, eta,
            X_test=None, y_test=None, on_epoch=None):
        """Mini-batch stochastic gradient descent with per-epoch reporting.

        on_epoch(epoch, epochs, mean_loss, test_accuracy, elapsed_seconds)
        is called after every epoch. Returns the final test accuracy.
        """
        n = X_train.shape[1]
        start = time.time()
        accuracy = None
        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(n)
            losses = []
            for k in range(0, n, mini_batch_size):
                idx = perm[k : k + mini_batch_size]
                X, Y = X_train[:, idx], Y_train[:, idx]
                m = X.shape[1]
                nabla_w, nabla_b, loss = self.backprop_batch(X, Y)
                self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]
                losses.append(loss)
            if X_test is not None:
                accuracy = self.evaluate(X_test, y_test)
            if on_epoch is not None:
                on_epoch(epoch, epochs, float(np.mean(losses)), accuracy,
                         time.time() - start)
        return accuracy


def run_m0(mnist_buf, epochs=15, hidden=30, mini_batch_size=10, eta=3.0,
           seed=1, on_epoch=None):
    """Milestone 0 entry point: load data, train 784-hidden-10, report metrics.

    Returns a dict with final accuracy and timing for the UI summary.
    """
    X_train, y_train, X_test, y_test = load_mnist_subset(mnist_buf)
    Y_train = one_hot(y_train)
    net = Network([X_train.shape[0], hidden, 10], seed=seed)
    start = time.time()
    accuracy = net.sgd(X_train, Y_train, epochs, mini_batch_size, eta,
                       X_test=X_test, y_test=y_test, on_epoch=on_epoch)
    return {
        "final_accuracy": accuracy,
        "train_seconds": time.time() - start,
        "epochs": epochs,
        "n_train": int(X_train.shape[1]),
        "n_test": int(X_test.shape[1]),
        "hidden": hidden,
    }
