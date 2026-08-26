# Tests for the l2 exercise. Fixture values are chosen so every expected
# number can be checked by hand. Failure messages are teaching content
# (CLAUDE.md).

import numpy as np
from submission import l2_step


def _fixture():
    weights = [np.array([[1.0, -2.0], [0.5, 0.25]])]
    biases = [np.array([[0.4], [-0.6]])]
    nabla_w = [np.array([[0.1, 0.2], [-0.3, 0.05]])]
    nabla_b = [np.array([[0.02], [-0.04]])]
    return weights, biases, nabla_w, nabla_b


def test_weights_decay_and_descend():
    """l2_step: the weights shrink, then step against their slopes"""
    weights, biases, nabla_w, nabla_b = _fixture()
    new_w, _ = l2_step(weights, biases, nabla_w, nabla_b, 0.5, 5.0, 1000)
    expected = np.array([[0.9475, -2.095], [0.64875, 0.224375]])
    assert np.allclose(new_w[0], expected, atol=1e-12), (
        f"expected\n{expected}\ngot\n{new_w[0]}\n"
        "With eta = 0.5, lmbda = 5 and n = 1000 the shrinking factor is "
        "1 - 0.5 * 5 / 1000 = 0.9975. The first weight goes "
        "0.9975 * 1.0 - 0.5 * 0.1 = 0.9475. If you got 0.95, the factor is "
        "missing (that is the plain step); if you got -1.5, the factor was "
        "subtracted rather than multiplied; if you got -1.5025, lmbda went "
        "in raw, without eta and without dividing by n."
    )


def test_biases_are_left_alone():
    """l2_step: biases take the plain step, with no shrinking"""
    weights, biases, nabla_w, nabla_b = _fixture()
    _, new_b = l2_step(weights, biases, nabla_w, nabla_b, 0.5, 5.0, 1000)
    expected = np.array([[0.39], [-0.58]])
    assert np.allclose(new_b[0], expected, atol=1e-12), (
        f"expected\n{expected}\ngot\n{new_b[0]}\n"
        "The bias line is untouched from Module 3: 0.4 - 0.5 * 0.02 = 0.39. "
        "Only weights decay. A bias multiplies no input, so shrinking it "
        "buys none of what this technique is for, and large biases are not "
        "what makes a network fragile."
    )


def test_lmbda_zero_is_the_old_step():
    """l2_step: with lmbda = 0 it is exactly Module 3's update"""
    weights, biases, nabla_w, nabla_b = _fixture()
    new_w, new_b = l2_step(weights, biases, nabla_w, nabla_b, 0.5, 0.0, 1000)
    assert np.allclose(new_w[0], np.array([[0.95, -2.1], [0.65, 0.225]]), atol=1e-12), (
        f"with lmbda = 0 the weights should move by the slope alone, to "
        f"[[0.95, -2.1], [0.65, 0.225]], and yours went to\n{new_w[0]}\n"
        "The factor 1 - eta * 0 / n is exactly 1, so this case has to "
        "reproduce your Module 3 step. Every comparison in the panel below "
        "runs both settings through this one function, so a wrong lmbda = 0 "
        "case would make the baseline wrong too."
    )
    assert np.allclose(new_b[0], np.array([[0.39], [-0.58]]), atol=1e-12), (
        f"the biases moved to {new_b[0].ravel()} instead of [0.39, -0.58]; "
        "lmbda never touched them, so this case must match the last test."
    )


def test_inputs_are_not_modified():
    """l2_step: returns new arrays and leaves its arguments alone"""
    weights, biases, nabla_w, nabla_b = _fixture()
    w_before = weights[0].copy()
    b_before = biases[0].copy()
    new_w, new_b = l2_step(weights, biases, nabla_w, nabla_b, 0.5, 5.0, 1000)
    assert np.array_equal(weights[0], w_before) and np.array_equal(biases[0], b_before), (
        "l2_step changed the arrays it was given. Build new arrays instead: "
        "decay * w - eta * nw already makes one, but w *= decay would "
        "overwrite the caller's parameters, and the panel below holds two "
        "runs' parameters at once."
    )
    assert new_w[0] is not weights[0] and new_b[0] is not biases[0], (
        "the returned arrays are the same objects that came in. The caller "
        "keeps the old ones to compare against, so hand back new arrays, "
        "the way your sgd_step did."
    )


def test_decay_alone_shrinks_geometrically():
    """l2_step: with no gradient, every weight shrinks by the same fraction"""
    weights = [np.array([[1.0]])]
    biases = [np.array([[0.0]])]
    zero_w = [np.zeros((1, 1))]
    zero_b = [np.zeros((1, 1))]
    for _ in range(100):
        weights, biases = l2_step(weights, biases, zero_w, zero_b, 0.5, 5.0, 1000)
    got = float(weights[0][0, 0])
    assert abs(got - 0.7785570396) < 1e-9, (
        f"after 100 steps with no gradient at all, a weight of 1.0 should "
        f"stand at 0.9975 raised to the 100th power, 0.7785570396; yours "
        f"stands at {got:.10f}. Multiplying by the same factor every step is "
        "the whole behaviour this line adds: left alone, a weight decays "
        "toward zero, and it takes the data pulling the other way to keep it "
        "large."
    )
