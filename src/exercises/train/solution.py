# Reference solution. The loop is the one Nielsen's network.py runs (MIT
# license, github.com/mnielsen/neural-networks-and-deep-learning), assembled
# here out of the functions the learner wrote across Modules 2 to 7.

import numpy as np
from course import batch_gradient, cross_entropy_delta, feedforward, init_network, l2_step


def accuracy(weights, biases, X, y):
    """Share of the columns of X that the network reads correctly."""
    guesses = np.argmax(feedforward(weights, biases, X), axis=0)
    return float((guesses == y).mean())


def train(sizes, X, Y, X_test, y_test, epochs, eta, lmbda, batch_size, rng):
    """Build a network, train it, and score it after every epoch."""
    weights, biases = init_network(sizes, rng)
    n = X.shape[1]
    history = []
    for _ in range(epochs):
        idx = rng.permutation(n)
        for k in range(0, n, batch_size):
            batch = idx[k : k + batch_size]
            nabla_w, nabla_b = batch_gradient(
                weights, biases, X[:, batch], Y[:, batch], cross_entropy_delta
            )
            weights, biases = l2_step(weights, biases, nabla_w, nabla_b, eta, lmbda, n)
        history.append(accuracy(weights, biases, X_test, y_test))
    return weights, biases, history
