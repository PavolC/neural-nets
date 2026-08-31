# Tests for the smart-init exercise. Fixture values are hardcoded literals
# from np.random.default_rng, whose stream is fixed across NumPy versions.
# Failure messages are teaching content (CLAUDE.md).

import numpy as np
from submission import init_network


def test_shapes():
    """init_network: one weight matrix and one bias column per layer of wires"""
    weights, biases = init_network([6, 4, 3], np.random.default_rng(4))
    assert len(weights) == 2 and len(biases) == 2, (
        f"a 6-4-3 network has two layers of wires, so expected 2 weight "
        f"matrices and 2 bias columns, got {len(weights)} and {len(biases)}. "
        "Three layer sizes means len(sizes) - 1 layers of wires: the input "
        "layer has no wires coming into it."
    )
    assert weights[0].shape == (4, 6) and weights[1].shape == (3, 4), (
        f"expected weight shapes (4, 6) and (3, 4), got {weights[0].shape} "
        f"and {weights[1].shape}. Receiving layer first, as everywhere in "
        "this course: 6 inputs into 4 neurons is a (4, 6) matrix, one row "
        "per receiving neuron."
    )
    assert biases[0].shape == (4, 1) and biases[1].shape == (3, 1), (
        f"expected bias shapes (4, 1) and (3, 1), got {biases[0].shape} and "
        f"{biases[1].shape}. Biases are column vectors, one entry per "
        "neuron in the receiving layer; a shape of (4,) is the flat array "
        "the course has been avoiding since Chapter 1."
    )


def test_scaling():
    """init_network: each weight is a standard draw divided by sqrt(inputs)"""
    weights, biases = init_network([6, 4, 3], np.random.default_rng(4))
    assert abs(weights[0][0, 0] - (-0.2660926238)) < 1e-9, (
        f"expected weights[0][0, 0] = -0.2660926238, got "
        f"{weights[0][0, 0]:.10f}. From this seed the first standard draw is "
        "-0.6517911526, and the first layer reads 6 inputs, so the entry is "
        "-0.6517911526 / sqrt(6). If you got the raw -0.6518, the division "
        "is missing; if you got -0.1086, you divided by 6 instead of by its "
        "square root."
    )
    assert abs(weights[1][0, 0] - 0.7412855815) < 1e-9, (
        f"expected weights[1][0, 0] = 0.7412855815, got "
        f"{weights[1][0, 0]:.10f}. Each layer divides by the square root of "
        "its OWN input count: sqrt(4) here, not sqrt(6). And this matrix "
        "comes second in the draw order, after the whole first weight "
        "matrix and before any bias."
    )
    expected_b0 = np.array([[0.9591848587], [-0.9798545664],
                            [-0.7977957579], [-0.2033324858]])
    assert np.allclose(biases[0], expected_b0, atol=1e-9), (
        f"expected biases[0] =\n{expected_b0}\ngot\n{biases[0]}\n"
        "Two things this checks. The values are unscaled standard draws: a "
        "bias reads no inputs, so there is nothing to divide by. And they "
        "are the draws that come AFTER both weight matrices, which is the "
        "order the contract fixes: all weights, then all biases."
    )


def test_spread_matches_the_rule():
    """init_network: on a big layer, the spread lands on 1/sqrt(inputs)"""
    weights, biases = init_network([400, 200, 3], np.random.default_rng(1))
    for l, n_in in ((0, 400), (1, 200)):
        target = 1.0 / np.sqrt(n_in)
        got = float(weights[l].std())
        assert abs(got - target) < 0.2 * target, (
            f"layer {l + 1}'s weights have spread {got:.5f}; with {n_in} "
            f"inputs the rule asks for about {target:.5f}. A whole layer of "
            "draws lands near its intended spread, so a number far from "
            "this one means the scaling used the wrong count (the receiving "
            "layer's size instead of the sending layer's) or no scaling at "
            "all."
        )
    bias_spread = float(biases[0].std())
    assert abs(bias_spread - 1.0) < 0.2, (
        f"the biases' spread is {bias_spread:.4f}, and it should be about 1: "
        "biases keep the standard draw. If it came out near 0.05, the "
        "1/sqrt(n) scaling was applied to them too."
    )


def test_same_seed_same_network():
    """init_network: every random number comes from the rng it is handed"""
    first = init_network([6, 4, 3], np.random.default_rng(11))
    second = init_network([6, 4, 3], np.random.default_rng(11))
    for a, b in zip(first[0] + first[1], second[0] + second[1]):
        assert np.array_equal(a, b), (
            "two calls with the same seed built different networks. Every "
            "draw has to come from the rng argument: np.random.standard_normal "
            "or np.random.randn read a global generator instead, which no "
            "seed of yours controls, and then no training run in this chapter "
            "can be compared with another."
        )
    third = init_network([6, 4, 3], np.random.default_rng(12))
    assert not np.array_equal(first[0][0], third[0][0]), (
        "two different seeds built the same weights, so the draws are not "
        "random at all. The scaling changes the size of the numbers, not "
        "where they come from."
    )
