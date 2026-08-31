# Tests for the sigmoid-neuron exercise. All fixtures are hardcoded
# literals, so results are deterministic. Failure messages are teaching
# content (see CLAUDE.md).

import numpy as np
from submission import sigmoid, fire


def test_sigmoid_at_zero():
    """sigmoid(0) is exactly one half"""
    got = sigmoid(0.0)
    assert np.ndim(got) == 0, (
        f"sigmoid(0.0) should be a single number, got an array of shape "
        f"{np.shape(got)}. Write the formula with np.exp and it will work "
        "for numbers and arrays alike."
    )
    assert abs(float(got) - 0.5) < 1e-12, (
        f"expected sigmoid(0) = 0.5, got {float(got)}. At z = 0, "
        "e^(-z) = 1, so the formula gives 1 / (1 + 1) = 0.5. Check the "
        "sign in the exponent and where the 1s go: 1 / (1 + e^(-z))."
    )


def test_sigmoid_elementwise():
    """sigmoid works elementwise on arrays"""
    z = np.array([[-2.0, -0.5], [0.0, 1.5]])
    expected = np.array([[0.1192029220, 0.3775406688],
                         [0.5000000000, 0.8175744762]])
    got = sigmoid(z)
    assert np.shape(got) == (2, 2), (
        f"expected sigmoid of a (2, 2) array to be a (2, 2) array, got "
        f"shape {np.shape(got)}. Use np.exp (not math.exp) so the formula "
        "applies to every element at once."
    )
    assert np.allclose(got, expected, atol=1e-6), (
        f"wrong values: expected\n{expected}\ngot\n{np.asarray(got)}\n"
        "Check the formula: 1 / (1 + np.exp(-z)). The most common slip is "
        "a lost minus sign, which mirrors the function (large z would give "
        "values near 0 instead of near 1)."
    )


def test_sigmoid_squashes():
    """sigmoid squashes everything into (0, 1)"""
    got_low = float(sigmoid(-10.0))
    got_high = float(sigmoid(10.0))
    assert 0 < got_low < 0.001, (
        f"expected sigmoid(-10) to be almost 0, got {got_low}. Very "
        "negative inputs should be squashed toward 0: that is what makes "
        "the sigmoid a smooth version of a perceptron's hard 0."
    )
    assert 0.999 < got_high < 1, (
        f"expected sigmoid(10) to be almost 1, got {got_high}. Very "
        "positive inputs should be squashed toward 1. If your low and high "
        "values are swapped, the sign of the exponent is flipped."
    )


def test_fire_returns_float():
    """fire returns a plain number, not an array"""
    w = np.array([[0.4], [-0.7], [1.2]])
    x = np.array([[0.5], [-1.0], [0.25]])
    got = fire(w, 0.3, x)
    assert isinstance(got, float), (
        f"expected a plain number, got {type(got).__name__} of shape {np.shape(got)}. "
        "If you used w.T @ x you got a (1, 1) array; a dot product of two "
        "column vectors is a single number, so pull it out with float(...) "
        "or use (w * x).sum()."
    )


def test_fire_values():
    """fire computes sigmoid(w . x + b)"""
    w = np.array([[0.4], [-0.7], [1.2]])
    x = np.array([[0.5], [-1.0], [0.25]])
    got = float(fire(w, 0.3, x))
    assert abs(got - 0.8175744762) < 1e-6, (
        f"expected fire(w, 0.3, x) = 0.8175744762, got {got}. Here "
        "w . x = 1.2 and b = 0.3, so the neuron computes sigmoid(1.5). "
        "Check that you multiply w and x elementwise, sum, add b, and only "
        "then apply sigmoid."
    )


def test_fire_uses_bias():
    """the bias shifts the neuron's output"""
    w = np.array([[0.4], [-0.7], [1.2]])
    x = np.array([[0.5], [-1.0], [0.25]])
    got = float(fire(w, -1.1, x))
    assert abs(got - 0.5249791875) < 1e-6, (
        f"expected fire(w, -1.1, x) = 0.5249791875, got {got}. Same w and "
        "x as before, only b changed, so only the bias can explain a "
        "mismatch: make sure b is added inside the sigmoid, once."
    )
