# Tests for the SGD exercise. All fixtures are hardcoded literals computed
# with course.gradient (eps=1e-5), so results are deterministic.
# Failure messages are teaching content (see CLAUDE.md).

import numpy as np
from submission import sgd_step, sgd


def _fixture_net():
    w0 = np.array([[0.5, -0.3], [-0.8, 0.9], [0.1, 0.4]])
    b0 = np.array([[0.1], [-0.2], [0.3]])
    w1 = np.array([[0.7, -0.5, 0.2]])
    b1 = np.array([[-0.1]])
    X = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    Y = np.array([[0.0, 1.0, 1.0, 0.0]])
    return [w0, w1], [b0, b1], X, Y


def test_step_returns_new_lists():
    """sgd_step returns new arrays and leaves its inputs alone"""
    weights, biases, X, Y = _fixture_net()
    w0_before = weights[0].copy()
    b0_before = biases[0].copy()
    out = sgd_step(weights, biases, X, Y, 0.5)
    assert isinstance(out, tuple) and len(out) == 2, (
        "sgd_step must return a pair (new_weights, new_biases), got "
        f"{type(out).__name__}. Return both lists together as a tuple."
    )
    new_w, new_b = out
    assert len(new_w) == 2 and len(new_b) == 2, (
        f"expected 2 weight arrays and 2 bias arrays back, got "
        f"{len(new_w)} and {len(new_b)}: one entry per layer, same "
        "structure as the inputs."
    )
    assert new_w[0].shape == (3, 2) and new_b[0].shape == (3, 1), (
        f"expected shapes (3, 2) and (3, 1) for the first layer, got "
        f"{new_w[0].shape} and {new_b[0].shape}. Each gradient has the "
        "same shape as its parameter, so the update cannot change shapes."
    )
    assert np.array_equal(weights[0], w0_before) and np.array_equal(biases[0], b0_before), (
        "sgd_step modified its input arrays. Build new arrays "
        "(w - eta * grad creates one) instead of updating in place with "
        "-=: the caller may still need the old parameters."
    )
    assert new_w[0] is not weights[0], (
        "sgd_step returned the same array objects it was given. Return "
        "newly computed arrays: new_param = param - eta * gradient."
    )


def test_step_eta_zero():
    """with eta = 0 nothing moves"""
    weights, biases, X, Y = _fixture_net()
    new_w, new_b = sgd_step(weights, biases, X, Y, 0.0)
    assert np.allclose(new_w[0], weights[0]) and np.allclose(new_b[1], biases[1]), (
        "with learning rate 0 the update w - 0 * gradient must return the "
        "parameters unchanged, but something moved. Check that eta "
        "multiplies the gradient, not the parameter."
    )


def test_step_values():
    """one step with eta = 0.5 matches the reference update"""
    weights, biases, X, Y = _fixture_net()
    new_w, new_b = sgd_step(weights, biases, X, Y, 0.5)
    expected_w1 = np.array([[0.6965775837, -0.5018613743, 0.1965692186]])
    expected_b1 = np.array([[-0.1056129477]])
    assert np.allclose(new_w[1], expected_w1, atol=1e-6), (
        f"wrong updated weights: expected\n{expected_w1}\ngot\n{new_w[1]}\n"
        "The update is new_w = w - eta * slope, where the slopes come "
        "from course.gradient(weights, biases, X, Y) on this batch. "
        "A plus instead of a minus walks uphill; a missing eta takes a "
        "full-size step."
    )
    assert np.allclose(new_b[1], expected_b1, atol=1e-6), (
        f"wrong updated bias: expected {expected_b1}, got {new_b[1]}. "
        "Biases follow the same rule as weights: new_b = b - eta * gradient."
    )


def test_step_goes_downhill():
    """a step reduces the cost on its own batch"""
    from course import quadratic_cost
    weights, biases, X, Y = _fixture_net()
    before = quadratic_cost(weights, biases, X, Y)
    new_w, new_b = sgd_step(weights, biases, X, Y, 0.5)
    after = quadratic_cost(new_w, new_b, X, Y)
    assert after < before, (
        f"the cost went from {before:.6f} to {after:.6f}: uphill. "
        "Gradient descent subtracts the gradient. If you added it, every "
        "step makes things worse."
    )


def test_sgd_final_values():
    """the full loop follows the prescribed batch order"""
    from course import quadratic_cost
    weights, biases, X, Y = _fixture_net()
    initial_cost = quadratic_cost(weights, biases, X, Y)
    final_w, final_b = sgd(weights, biases, X, Y, 2.0, 3, 2,
                           np.random.default_rng(0))
    final_cost = quadratic_cost(final_w, final_b, X, Y)
    expected_w1 = np.array([[0.6497401933, -0.5159356158, 0.1556485109]])
    assert final_cost < initial_cost, (
        f"after 3 epochs the cost should have decreased, but it is "
        f"{final_cost:.6f}. Check that sgd actually applies sgd_step to "
        "every mini-batch and carries the updated parameters forward "
        "between batches and between epochs."
    )
    assert np.allclose(final_w[1], expected_w1, atol=1e-6), (
        f"training works but the exact result differs: expected final "
        f"output weights\n{expected_w1}\ngot\n{final_w[1]}\n"
        "The tests prescribe the batch order exactly: one "
        "rng.permutation(n) per epoch, then slices idx[k:k+batch_size] "
        "for k = 0, batch_size, 2*batch_size, ... Any other use of the "
        "rng (or reshuffling per batch) gives different batches."
    )
