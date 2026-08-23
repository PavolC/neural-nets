# Tests for the feedforward exercise. All fixtures are hardcoded literals,
# so results are deterministic and independent of NumPy's random streams.
# Failure messages are teaching content (see CLAUDE.md).

import numpy as np
from submission import feedforward


def _describe_shape(got):
    if not isinstance(got, np.ndarray):
        return f"a {type(got).__name__}, not a NumPy array"
    return f"an array of shape {got.shape}"


def test_returns_column_vector():
    """Output is a column vector of the right shape"""
    w = np.array([[0.0012, 0.2987, -0.2741], [-0.8906, -0.4547, -0.9916]])
    b = np.array([[0.0601], [1.3402]])
    x = np.array([[-0.4922], [-0.6205], [0.4898]])
    out = feedforward([w], [b], x)
    assert isinstance(out, np.ndarray), (
        f"expected a NumPy array, got {_describe_shape(out)}. "
        "feedforward should return the final activation array itself."
    )
    assert out.shape == (2, 1), (
        f"expected shape (2, 1), got {out.shape}: this network has 2 output "
        "neurons and activations are column vectors, so the result must be "
        "2 rows by 1 column. If you got (2,), something flattened your "
        "activations; check that x stays (n, 1) through every layer."
    )


def test_single_layer_values():
    """A single layer computes sigmoid(w @ x + b)"""
    w = np.array([[0.0012, 0.2987, -0.2741], [-0.8906, -0.4547, -0.9916]])
    b = np.array([[0.0601], [1.3402]])
    x = np.array([[-0.4922], [-0.6205], [0.4898]])
    expected = np.array([[0.4353420342], [0.8284965074]])
    out = feedforward([w], [b], x)
    assert out.shape == expected.shape, (
        f"expected shape {expected.shape}, got {out.shape} (see the shape test)."
    )
    assert np.allclose(out, expected, atol=1e-6), (
        f"wrong values for a one-layer network: expected\n{expected}\ngot\n{out}\n"
        "One layer is just sigmoid(w @ x + b). Common causes: using * "
        "(elementwise) instead of @ (matrix multiply), forgetting the bias, "
        "or forgetting to apply sigmoid."
    )


def test_two_layer_values():
    """Two layers chain: the output of layer 1 feeds layer 2"""
    w1 = np.array([[0.3047, -1.04, 0.7505, 0.9406],
                   [-1.951, -1.3022, 0.1278, -0.3162],
                   [-0.0168, -0.853, 0.8794, 0.7778]])
    b1 = np.array([[0.066], [1.1272], [0.4675]])
    w2 = np.array([[-0.8593, 0.3688, -0.9589], [0.8785, -0.0499, -0.1849]])
    b2 = np.array([[-0.6809], [1.2225]])
    x = np.array([[-0.1545], [-0.4283], [-0.3521], [0.5323]])
    expected = np.array([[0.1639029911], [0.83672299]])
    out = feedforward([w1, w2], [b1, b2], x)
    assert out.shape == (2, 1), (
        f"expected shape (2, 1), got {out.shape}: this is a 4-3-2 network, so "
        "the output has 2 rows. If your shapes exploded, check that each "
        "layer's output becomes the next layer's input."
    )
    assert np.allclose(out, expected, atol=1e-6), (
        f"wrong values for a 4-3-2 network: expected\n{expected}\ngot\n{out}\n"
        "Each layer must consume the previous layer's activation: "
        "a = sigmoid(w @ a + b), starting from a = x. Also check that you "
        "apply sigmoid after every layer, not only the last one."
    )


def test_layer_order():
    """Layers are applied in order: weights[0] first"""
    v1 = np.array([[2.0409, -2.5557], [0.4181, -0.5678]])
    c1 = np.array([[-0.4526], [-0.2156]])
    v2 = np.array([[-2.02, -0.2319], [-0.8652, 3.323]])
    c2 = np.array([[0.2258], [-0.3526]])
    x = np.array([[-0.2813], [-0.668]])
    expected = np.array([[0.2255170958], [0.6841283347]])
    reversed_order = np.array([[0.6882319116], [0.508853066]])
    out = feedforward([v1, v2], [c1, c2], x)
    assert out.shape == (2, 1), (
        f"expected shape (2, 1), got {out.shape}."
    )
    assert not np.allclose(out, reversed_order, atol=1e-6), (
        "you applied the layers in reverse order: weights[0] connects the "
        "input to the first hidden layer, so it must be applied first. "
        "This slipped past the earlier tests because their layer shapes "
        "happened to make reversed order crash; here both orders are "
        "shape-valid and only the correct one gives the right numbers."
    )
    assert np.allclose(out, expected, atol=1e-6), (
        f"wrong values: expected\n{expected}\ngot\n{out}\n"
        "Layer order looks right, so check the arithmetic of each layer: "
        "a = sigmoid(w @ a + b)."
    )


def test_input_not_modified():
    """feedforward must not modify its inputs"""
    w = np.array([[0.0012, 0.2987, -0.2741], [-0.8906, -0.4547, -0.9916]])
    b = np.array([[0.0601], [1.3402]])
    x = np.array([[-0.4922], [-0.6205], [0.4898]])
    x_copy = x.copy()
    w_copy = w.copy()
    feedforward([w], [b], x)
    assert np.array_equal(x, x_copy) and np.array_equal(w, w_copy), (
        "feedforward changed its input arrays. Avoid in-place operations "
        "(like += or x[...] = ...) on the arguments: later modules reuse "
        "the same weights across many calls and silent mutation will "
        "corrupt training."
    )
