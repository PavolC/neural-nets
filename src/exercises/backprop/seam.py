# The backprop section as it stands after Chapter 7's edit: BP1 lifted into an
# argument. This file is the course's copy of that end state, used to check
# that the edit Chapter 7 asks for keeps every Chapter 5 test green and leaves
# the default behaviour identical. It is not shown to the learner as a
# solution; Chapter 7's prompt ships the two changed lines.
#
# Reference solution, adapted from Michael Nielsen's network.py (MIT license,
# github.com/mnielsen/neural-networks-and-deep-learning).


def sigmoid_prime(z):
    """The squash's slope at z: sigmoid(z) * (1 - sigmoid(z)), elementwise."""
    s = sigmoid(z)
    return s * (1.0 - s)


def backprop(weights, biases, x, y, output_delta=None):
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

    # BP1: the output layer's blame. Chapter 7 swaps this one line out.
    if output_delta is None:
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    else:
        delta = output_delta(activations[-1], y, zs[-1])
    nabla_b[-1] = delta                       # BP3
    nabla_w[-1] = delta @ activations[-2].T   # BP4

    # BP2: carry the blame backward, one layer at a time.
    for l in range(2, len(weights) + 1):
        delta = (weights[-l + 1].T @ delta) * sigmoid_prime(zs[-l])
        nabla_b[-l] = delta                          # BP3
        nabla_w[-l] = delta @ activations[-l - 1].T  # BP4

    return nabla_w, nabla_b
