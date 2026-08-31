# One function. It builds a network's parameters from scratch, at sizes it is
# told, with the weights scaled down by the square root of the number of
# inputs feeding their layer.
#
# Contract:
# - init_network(sizes, rng): sizes is a list of layer sizes, front to back,
#   so [784, 30, 10] is the digit reader. rng is a NumPy random generator
#   (np.random.default_rng(seed)); every random number must come from it, so
#   the same seed rebuilds the same network. Returns (weights, biases):
#
#   * weights has one entry per layer of wires, len(sizes) - 1 of them.
#     weights[l] has shape (sizes[l+1], sizes[l]), receiving layer named
#     first as always. Its entries are standard normal draws (spread 1,
#     centred on 0) divided by the square root of sizes[l], the number of
#     inputs feeding that layer.
#   * biases[l] has shape (sizes[l+1], 1). Standard normal draws, NOT
#     scaled: a bias reads no inputs, so there is nothing to divide by.
#
#   Draw order matters, because the tests and the panel reproduce exact
#   numbers: draw every weight matrix first, front to back, and then every
#   bias column, front to back. That is the order Chapter 5's training panel
#   used, so the only difference between its start and yours is the scaling.
#
# NumPy tools: rng.standard_normal(shape) draws an array of that shape,
# np.sqrt(n) is the square root of n.


def init_network(sizes, rng):
    """Weights scaled by 1/sqrt(inputs), biases standard normal.

    Returns (weights, biases). Replace the raise below with your
    implementation.
    """
    raise NotImplementedError("implement init_network")
