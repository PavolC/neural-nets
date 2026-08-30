# Reference solution, adapted from Michael Nielsen's network.py (MIT license,
# github.com/mnielsen/neural-networks-and-deep-learning).


def feedforward(weights, biases, x):
    """Return the network's output activation for the input column vector x."""
    a = x
    for w, b in zip(weights, biases):
        a = sigmoid(w @ a + b)
    return a
