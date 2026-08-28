# Tests for the backprop exercise. All fixture values are hardcoded literals,
# verified against central-difference numerical gradients when they were
# generated. Failure messages are teaching content (see CLAUDE.md).

import numpy as np
from submission import sigmoid_prime, backprop


def _fixture_net():
    # The same 2-3-1 network as the Module 3 sgd tests, with one example:
    # the concert corner (1, 0), right answer 1.
    w0 = np.array([[0.5, -0.3], [-0.8, 0.9], [0.1, 0.4]])
    b0 = np.array([[0.1], [-0.2], [0.3]])
    w1 = np.array([[0.7, -0.5, 0.2]])
    b1 = np.array([[-0.1]])
    x = np.array([[1.0], [0.0]])
    y = np.array([[1.0]])
    return [w0, w1], [b0, b1], x, y


def test_sigmoid_prime():
    """sigmoid_prime: the steepness formula"""
    v = float(sigmoid_prime(np.array(0.0)))
    assert abs(v - 0.25) < 1e-9, (
        f"sigmoid_prime(0) should be 0.25, got {v:.6f}. At z = 0 the "
        "confidence is 0.5, and the steepness is confidence times doubt: "
        "0.5 * 0.5 = 0.25. If you got 0.5, you returned sigmoid(z) itself; "
        "the formula is sigmoid(z) * (1 - sigmoid(z))."
    )
    z = np.array([[0.0], [2.0], [-2.0]])
    out = sigmoid_prime(z)
    assert out.shape == (3, 1), (
        f"expected shape (3, 1) back for a (3, 1) input, got {out.shape}: "
        "sigmoid_prime works elementwise, like sigmoid, so shapes pass "
        "through unchanged."
    )
    expected = np.array([[0.25], [0.1049935854], [0.1049935854]])
    assert np.allclose(out, expected, atol=1e-9), (
        f"expected\n{expected}\ngot\n{out}\n"
        "sigmoid_prime(2) and sigmoid_prime(-2) are both 0.10499...: the "
        "curve is equally steep either side of the fence. If your values "
        "differ, check that BOTH factors use sigmoid(z), not z itself: "
        "the formula is sigmoid(z) * (1 - sigmoid(z))."
    )


def test_backprop_shapes():
    """backprop returns slope arrays shaped like the parameters"""
    weights, biases, x, y = _fixture_net()
    w0_before = weights[0].copy()
    b0_before = biases[0].copy()
    out = backprop(weights, biases, x, y)
    assert isinstance(out, tuple) and len(out) == 2, (
        f"backprop must return a pair (nabla_w, nabla_b), got "
        f"{type(out).__name__}. Weights' slopes first, then biases', the "
        "same order the parameters travel everywhere in this course."
    )
    nabla_w, nabla_b = out
    assert len(nabla_w) == 2 and len(nabla_b) == 2, (
        f"expected 2 weight-slope arrays and 2 bias-slope arrays (one per "
        f"layer), got {len(nabla_w)} and {len(nabla_b)}. Every layer gets "
        "slopes, including the ones your backward loop visits."
    )
    assert nabla_w[1].shape == (1, 3) and nabla_b[1].shape == (1, 1), (
        f"expected output-layer shapes (1, 3) and (1, 1), got "
        f"{nabla_w[1].shape} and {nabla_b[1].shape}. BP4 builds nabla_w as "
        "delta @ activations.T: here (1, 1) @ (1, 3) gives (1, 3), exactly "
        "the shape of weights[1]. If yours came out (3, 1), the product is "
        "backwards or a transpose is missing."
    )
    assert nabla_w[0].shape == (3, 2) and nabla_b[0].shape == (3, 1), (
        f"expected hidden-layer shapes (3, 2) and (3, 1), got "
        f"{nabla_w[0].shape} and {nabla_b[0].shape}. The hidden layer's "
        "delta is (3, 1) and the activations feeding it are the input x, "
        "so BP4 is (3, 1) @ (1, 2). A gradient always has the exact shape "
        "of its parameter: one slope per knob."
    )
    assert np.array_equal(weights[0], w0_before) and np.array_equal(biases[0], b0_before), (
        "backprop modified the weights or biases it was given. It only "
        "reads them: the caller (your sgd) decides how to move the "
        "parameters, using the slopes you return."
    )


def test_backprop_output_layer():
    """the output layer: BP1 gives the blame, BP3 and BP4 read off slopes"""
    weights, biases, x, y = _fixture_net()
    nabla_w, nabla_b = backprop(weights, biases, x, y)
    expected_b1 = np.array([[-0.1012158735]])
    assert np.allclose(nabla_b[1], expected_b1, atol=1e-8), (
        f"expected nabla_b[1] = {expected_b1}, got {nabla_b[1]}. BP1 has "
        "exactly two factors: the gap (a - y) and the steepness "
        "sigmoid_prime(z), both at the output layer, multiplied "
        "elementwise. Common wrong answers here: -0.4165 means the "
        "steepness factor is missing; +0.1012 means you wrote y - a "
        "(Module 4's flip: slopes use output minus right answer); "
        "-0.0957 means you fed sigmoid_prime the activation a instead of "
        "the evidence z."
    )
    expected_w1 = np.array([[-0.065350667, -0.0272211409, -0.0605966945]])
    assert np.allclose(nabla_w[1], expected_w1, atol=1e-8), (
        f"expected nabla_w[1] =\n{expected_w1}\ngot\n{nabla_w[1]}\n"
        "BP4 at the output layer: delta @ activations[-2].T, where "
        "activations[-2] is the HIDDEN layer's column (0.6457, 0.2689, "
        "0.5987), the values those wires actually carried. Each weight's "
        "slope is the blame times its own wire's activation."
    )


def test_backprop_hidden_layer():
    """the hidden layer: BP2 carries the blame back through the wires"""
    weights, biases, x, y = _fixture_net()
    nabla_w, nabla_b = backprop(weights, biases, x, y)
    expected_b0 = np.array([[-0.0162096177], [0.0099501243], [-0.0048636402]])
    assert np.allclose(nabla_b[0], expected_b0, atol=1e-8), (
        f"expected nabla_b[0] =\n{expected_b0}\ngot\n{nabla_b[0]}\n"
        "BP2: the hidden delta is (weights[1].T @ delta) * "
        "sigmoid_prime(zs[0]): collect the blame through the transposed "
        "wires, then scale by THIS layer's steepness. If you got "
        "(-0.0172, 0.0123, -0.0049), you reused the output layer's z in "
        "sigmoid_prime; each layer's steepness is measured at its own "
        "stored z. Note the middle entry is positive: it arrives through "
        "the negative wire -0.5, which flips the blame's sign."
    )
    expected_w0 = np.array([[-0.0162096177, 0.0], [0.0099501243, 0.0], [-0.0048636402, 0.0]])
    assert np.allclose(nabla_w[0], expected_w0, atol=1e-8), (
        f"expected nabla_w[0] =\n{expected_w0}\ngot\n{nabla_w[0]}\n"
        "BP4 at the hidden layer: delta @ activations[-3].T, and "
        "activations[-3] here is the input column x itself. The whole "
        "second column must be exactly zero: those weights multiply "
        "x2 = 0, Module 4's dead-nudge case (a knob whose input is zero "
        "changes nothing, so its slope is zero)."
    )


def test_gradient_check():
    """the gradient check: your slopes against nudge-and-measure, all 54"""
    # The oracle is deliberately the course's forward pass and the
    # course's nudge-and-measure, not the copies in your file. This is
    # the course's strongest promise about your code, and a guarantee
    # whose yardstick shares the code under test is not a guarantee.
    from course import feedforward, numerical_gradient
    # A fixed 3-5-4-2 network: two hidden layers, so the backward loop has
    # to take more than one step, and no two layers the same size, so every
    # transposed product keeps its shapes honest.
    weights = [
        np.array([[-0.8, -1.32, -0.25],
                  [0.42, 1.14, 0.11],
                  [-0.55, -0.78, 0.75],
                  [1.63, 0.27, -1.23],
                  [-0.96, 1.6, 0.2]]),
        np.array([[-1.73, -0.08, -1.16, -0.63, -0.49],
                  [-0.71, 0.55, -0.06, -0.59, 0.41],
                  [0.83, -1.64, -0.26, -0.98, -0.17],
                  [-1.29, 0.02, -0.04, -0.3, -1.05]]),
        np.array([[-0.4, -1.09, -1.36, 0.22],
                  [-1.11, 1.17, 0.72, -2.0]]),
    ]
    biases = [
        np.array([[0.27], [-1.1], [0.03], [0.04], [-1.99]]),
        np.array([[-0.23], [-0.26], [0.96], [-1.18]]),
        np.array([[0.74], [-1.1]]),
    ]
    x = np.array([[-0.33], [-0.84], [1.45]])
    y = np.array([[1.0], [0.0]])

    nabla_w, nabla_b = backprop(weights, biases, x, y)

    def cost_fn(ws, bs):
        out = feedforward(ws, bs, x)
        return 0.5 * float(((out - y) ** 2).sum())

    num_w, num_b = numerical_gradient(cost_fn, weights, biases, eps=1e-4)

    names = ["weights layer 1", "weights layer 2", "weights layer 3",
             "biases layer 1", "biases layer 2", "biases layer 3"]
    worst = 0.0
    worst_report = ""
    for name, yours, measured in zip(names, nabla_w + nabla_b, num_w + num_b):
        rel = np.abs(yours - measured) / np.maximum(np.abs(yours) + np.abs(measured), 1e-8)
        i = np.unravel_index(int(np.argmax(rel)), rel.shape)
        if rel[i] > worst:
            worst = float(rel[i])
            worst_report = (
                f"worst disagreement: {name}, entry {tuple(int(k) for k in i)}: "
                f"nudge-and-measure says {measured[i]:.10f}, your backprop "
                f"says {yours[i]:.10f}"
            )
    assert worst < 1e-7, (
        f"the gradient check failed: {worst_report} (relative discrepancy "
        f"{worst:.2e}; the bar is 1e-7, and a correct backprop lands near "
        "1e-9). Both methods measure the same slopes, so a gap this size "
        "is a formula bug, not rounding. If only layers 1 and 2 disagree "
        "while layer 3 matches, your backward loop is off by one: check "
        "that BP2 transposes the NEXT layer's weights and that "
        "sigmoid_prime gets THIS layer's stored z."
    )


def test_backprop_trains():
    """plugged into descent, your slopes retrain the XOR network"""
    from course import feedforward
    # Module 1's slider network at its starting position (cost 0.0876),
    # trained by full-batch descent on the four corners: each step averages
    # your per-example slopes, exactly what the training panel below does.
    weights = [np.array([[2.0, 2.0], [2.0, 2.0]]), np.array([[4.0, -4.0]])]
    biases = [np.array([[-1.0], [-3.0]]), np.array([[-2.0]])]
    X = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    Y = np.array([[0.0, 1.0, 1.0, 0.0]])
    for _ in range(300):
        acc_w = [np.zeros_like(w) for w in weights]
        acc_b = [np.zeros_like(b) for b in biases]
        for k in range(4):
            nw, nb = backprop(weights, biases, X[:, k:k + 1], Y[:, k:k + 1])
            acc_w = [t + d for t, d in zip(acc_w, nw)]
            acc_b = [t + d for t, d in zip(acc_b, nb)]
        weights = [w - 2.0 * t / 4 for w, t in zip(weights, acc_w)]
        biases = [b - 2.0 * t / 4 for b, t in zip(biases, acc_b)]
    out = feedforward(weights, biases, X)
    cost = 0.5 * float(((out - Y) ** 2).sum()) / 4
    assert cost < 0.01, (
        f"after 300 descent steps driven by your slopes, the cost is "
        f"{cost:.4f}; it should fall from 0.0876 to about 0.008. Your "
        "gradients passed the value checks, so if this fails, something "
        "in backprop depends on state it should not (for example, "
        "modifying the weights or biases in place)."
    )
    assert np.array_equal((out >= 0.5).astype(int), Y.astype(int)), (
        f"the cost fell to {cost:.4f} but the four answers {out.round(2)} "
        "do not lean 0, 1, 1, 0. With a correct backprop this training "
        "run ends with every corner on the right side of 0.5."
    )
