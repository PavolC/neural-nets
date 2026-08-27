# Tests for the capstone. The fixtures are built by arithmetic, never from a
# random stream, so they are identical on every NumPy version. The reference
# numbers were produced by the reference solution and pin the loop order.
# Failure messages are teaching content (see CLAUDE.md).

import numpy as np
from submission import accuracy, train


def _bias_only_net():
    """A network that ignores its input: every column gets the same answer.

    One input, three outputs, all weights zero, so the biases alone decide.
    Output 0 has the largest bias, so this network answers 0 every time.
    """
    weights = [np.zeros((3, 1))]
    biases = [np.array([[2.0], [0.0], [-2.0]])]
    return weights, biases


def _reader_net():
    """A network that answers 0 for a positive input and 2 for a negative one."""
    weights = [np.array([[3.0], [0.0], [-3.0]])]
    biases = [np.zeros((3, 1))]
    return weights, biases


def _label(v):
    """Left, middle or right third of the range."""
    return 0 if v < -1.0 else (1 if v < 1.0 else 2)


def _fixture_data():
    """A three-class problem: which third of its range the first input is in.

    Twelve training points spread evenly across the range, four per class,
    and six held-out points at different positions. The second input carries
    nothing, so a working network has to learn to ignore it. Separable, so a
    small network reaches every held-out point.
    """
    x1 = np.linspace(-2.75, 2.75, 12)
    X = np.vstack([x1, np.array([0.2, 0.8] * 6)])
    y = np.array([_label(v) for v in x1])
    Y = np.zeros((3, 12))
    Y[y, np.arange(12)] = 1.0

    t1 = np.linspace(-2.45, 2.45, 6)
    X_test = np.vstack([t1, np.full(6, 0.5)])
    y_test = np.array([_label(v) for v in t1])
    return X, Y, X_test, y_test


def test_accuracy_reads_the_most_confident_output():
    """accuracy counts the most confident output, not a threshold"""
    weights, biases = _bias_only_net()
    X = np.array([[0.0, 1.0, -1.0, 0.5]])
    y = np.array([0, 0, 1, 2])
    got = accuracy(weights, biases, X, y)
    assert isinstance(got, float), (
        f"expected a plain float, got {type(got).__name__}. Wrap the result in "
        "float(...): a NumPy scalar prints the same and behaves differently."
    )
    assert abs(got - 0.5) < 1e-9, (
        f"expected 0.5, got {got}. This network's weights are all zero, so its "
        "answer is the same for every column: output 0, which has the largest "
        "bias. Two of the four right answers are 0, so half are read correctly. "
        "The network's answer is the position of its largest output, which is "
        "np.argmax(..., axis=0) down each column, and y holds class ids (0, 1, "
        "2), not one-hot columns."
    )


def test_accuracy_at_both_ends():
    """accuracy is 1.0 when every column is right and 0.0 when none is"""
    weights, biases = _reader_net()
    X = np.array([[1.0, -1.0]])
    assert accuracy(weights, biases, X, np.array([0, 2])) == 1.0, (
        "this network answers 0 for a positive input and 2 for a negative one, "
        "and both right answers were given, so accuracy must be exactly 1.0."
    )
    assert accuracy(weights, biases, X, np.array([2, 0])) == 0.0, (
        "with both answers wrong, accuracy must be exactly 0.0. If you got 1.0, "
        "check that you compare the network's answer with y rather than "
        "comparing y with itself."
    )


def test_train_returns_the_network_and_its_history():
    """train returns (weights, biases, history) with one score per epoch"""
    X, Y, X_test, y_test = _fixture_data()
    out = train([2, 6, 3], X, Y, X_test, y_test, 3, 1.0, 0.0, 3,
                np.random.default_rng(0))
    assert isinstance(out, tuple) and len(out) == 3, (
        f"expected a triple (weights, biases, history), got "
        f"{type(out).__name__} of length {len(out) if hasattr(out, '__len__') else '?'}. "
        "Return all three together."
    )
    weights, biases, history = out
    assert len(weights) == 2 and len(biases) == 2, (
        f"expected 2 weight arrays and 2 bias arrays for sizes [2, 6, 3], got "
        f"{len(weights)} and {len(biases)}: one per layer, and a 3-layer "
        "network has 2 layers of wires."
    )
    assert weights[0].shape == (6, 2) and weights[1].shape == (3, 6), (
        f"expected shapes (6, 2) and (3, 6), got {weights[0].shape} and "
        f"{weights[1].shape}. init_network builds these from sizes; if they are "
        "transposed, check that you passed sizes through unchanged."
    )
    assert len(history) == 3, (
        f"expected 3 entries in history, one per epoch, got {len(history)}. "
        "Score the held-out data once after each full epoch, not after each "
        "mini-batch."
    )
    assert all(0.0 <= h <= 1.0 for h in history), (
        f"history should hold accuracies between 0 and 1, got {history}. Append "
        "what your accuracy function returns, not a count or a cost."
    )


def test_train_learns():
    """training improves the held-out score"""
    X, Y, X_test, y_test = _fixture_data()
    _, _, history = train([2, 6, 3], X, Y, X_test, y_test, 20, 1.0, 0.0, 3,
                          np.random.default_rng(0))
    assert history[-1] >= 0.8, (
        f"after 20 epochs the network reads {history[-1]:.0%} of the held-out "
        f"points; this problem is separable and should reach at least 80%. The "
        f"whole history is {history}. If it never moves, the updated parameters "
        "are probably not being carried forward: every l2_step returns new "
        "arrays, and the next batch has to use them."
    )
    assert history[-1] >= history[0], (
        f"the score went down over the run, from {history[0]:.0%} to "
        f"{history[-1]:.0%}. Check that l2_step receives the gradient in the "
        "order it expects, (nabla_w, nabla_b), and that eta is not being "
        "applied twice."
    )


def test_train_is_repeatable():
    """the same seed trains the same network"""
    X, Y, X_test, y_test = _fixture_data()
    first = train([2, 6, 3], X, Y, X_test, y_test, 4, 1.0, 0.0, 3,
                  np.random.default_rng(7))
    second = train([2, 6, 3], X, Y, X_test, y_test, 4, 1.0, 0.0, 3,
                   np.random.default_rng(7))
    assert np.allclose(first[0][1], second[0][1]) and first[2] == second[2], (
        "two runs from the same seed produced different networks. Every random "
        "number must come from the rng argument: the draw inside init_network "
        "and the shuffle at the top of each epoch, and nothing else. "
        "np.random.permutation reads a global generator no seed of yours "
        "controls."
    )


def test_train_follows_the_prescribed_order():
    """the loop draws, shuffles and steps in the prescribed order"""
    X, Y, X_test, y_test = _fixture_data()
    weights, _, history = train([2, 6, 3], X, Y, X_test, y_test, 4, 1.0, 0.0, 3,
                                np.random.default_rng(0))
    expected_w1 = np.array(
        [[-1.4498611758, -0.6356552713, 0.4960253773, -1.2786443413, 0.9016864788, 0.9531190470],
         [-0.5381754944, -0.2121928664, -0.3815151510, 0.0687655409, -0.5369533911, -0.1135646919],
         [1.1141601227, 0.6166165680, -1.3349822269, 0.2962892957, -1.3362486024, -1.1295394431]]
    )
    expected_history = [2 / 3, 5 / 6, 2 / 3, 2 / 3]
    assert np.allclose(history, expected_history), (
        f"expected the history {[round(h, 3) for h in expected_history]}, got "
        f"{[round(h, 3) for h in history]}. The order is "
        "prescribed: draw the network from rng first, then for each epoch one "
        "rng.permutation(n), then slices idx[k:k+batch_size] taken front to "
        "back, one l2_step per slice, and the held-out score once at the end of "
        "the epoch. Any other use of the rng gives different batches."
    )
    assert np.allclose(weights[1], expected_w1, atol=1e-6), (
        f"the history matches but the weights differ: expected\n{expected_w1}\n"
        f"got\n{weights[1]}\nThe batches are right, so check what happens "
        "inside one: the gradient comes from batch_gradient on that batch's "
        "columns with cross_entropy_delta, and l2_step consumes it with the "
        "size of the WHOLE training set as n, not the size of the batch."
    )


def test_train_passes_lmbda_through():
    """weight decay reaches the update"""
    X, Y, X_test, y_test = _fixture_data()
    loose, _, _ = train([2, 6, 3], X, Y, X_test, y_test, 6, 1.0, 0.0, 3,
                        np.random.default_rng(3))
    tight, _, _ = train([2, 6, 3], X, Y, X_test, y_test, 6, 1.0, 8.0, 3,
                        np.random.default_rng(3))
    loose_size = sum(float((w ** 2).sum()) for w in loose)
    tight_size = sum(float((w ** 2).sum()) for w in tight)
    assert tight_size < loose_size, (
        f"with lmbda = 8 the weights total {tight_size:.4f} squared, against "
        f"{loose_size:.4f} with lmbda = 0; decay should have made them smaller. "
        "Pass lmbda through to l2_step rather than leaving it out of the call."
    )
