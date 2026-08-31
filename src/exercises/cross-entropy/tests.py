# Tests for the cross-entropy exercise. Fixture values are hardcoded
# literals, verified against central-difference numerical gradients when
# they were generated. Failure messages are teaching content (CLAUDE.md).

import inspect

import numpy as np
from submission import cross_entropy_cost, cross_entropy_delta


def _fixture_net():
    # The same 2-3-1 network as Module 5's tests, so its numbers can be
    # compared with the ones that module printed.
    w0 = np.array([[0.5, -0.3], [-0.8, 0.9], [0.1, 0.4]])
    b0 = np.array([[0.1], [-0.2], [0.3]])
    w1 = np.array([[0.7, -0.5, 0.2]])
    b1 = np.array([[-0.1]])
    return [w0, w1], [b0, b1]


def test_cost_one_example():
    """cross_entropy_cost: one example, one output neuron"""
    weights, biases = _fixture_net()
    x = np.array([[1.0], [0.0]])
    y = np.array([[1.0]])
    got = cross_entropy_cost(weights, biases, x, y)
    assert isinstance(got, float), (
        f"cross_entropy_cost must return a plain float, got "
        f"{type(got).__name__}. Wrap the total in float(...), the way the "
        "course's quadratic_cost does; a (1, 1) array will not chart."
    )
    assert abs(got - 0.5386824035) < 1e-8, (
        f"expected 0.5386824035, got {got:.10f}. This network answers "
        "0.5835 on the corner (1, 0) and the right answer is 1, so the "
        "bill is -ln(0.5835) = 0.5387: only the y = 1 term survives, "
        "since the other one is multiplied by 1 - y = 0. If you got "
        "0.0868, that is the quadratic cost, half the squared gap. If you "
        "got 0.8763, you paid -ln(1 - a) instead: that is the bill for a "
        "right answer of 0."
    )


def test_cost_batch():
    """cross_entropy_cost: averaged over a batch of columns"""
    weights, biases = _fixture_net()
    X = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    Y = np.array([[0.0, 1.0, 1.0, 0.0]])
    got = cross_entropy_cost(weights, biases, X, Y)
    assert abs(got - 0.6997436605) < 1e-8, (
        f"expected 0.6997436605, got {got:.10f}. The four corners cost "
        "0.7749, 0.5387, 0.6858 and 0.7996 (in column order), which sum "
        "to 2.7990. Divide by the number of columns, 4. If you got "
        "2.7990, the division by m is missing; if you got 0.1749, you "
        "divided by 2m as the quadratic cost does (that cost's half has "
        "no counterpart here)."
    )


def test_cost_stays_finite():
    """cross_entropy_cost: a perfectly confident network still costs a number"""
    # A weight of 50 saturates the output: sigmoid(50) is exactly 1.0 in
    # floating point, so the unclipped formula meets log(0).
    weights = [np.array([[50.0], [50.0]])]
    biases = [np.array([[0.0], [0.0]])]
    X = np.array([[1.0]])

    right = cross_entropy_cost(weights, biases, X, np.array([[1.0], [1.0]]))
    assert np.isfinite(right) and right < 1e-6, (
        f"expected about 0 for a network that answers 1.0 when both right "
        f"answers are 1.0, got {right}. A nan here means the 0 * log(0) "
        "hole is still open: the y = 0 term is 0 * log(1 - 1) = "
        "0 * (-inf), which is nan, and nan plus anything is nan. Clip the "
        "answers into (0, 1) before the logarithms."
    )

    wrong = cross_entropy_cost(weights, biases, X, np.array([[0.0], [0.0]]))
    assert np.isfinite(wrong) and wrong > 10.0, (
        f"expected a large but finite cost for a network that answers 1.0 "
        f"when both right answers are 0.0, got {wrong}. Confidently wrong "
        "is what this cost charges heavily for; clipping is what keeps the "
        "charge finite rather than infinite."
    )


def test_delta_values():
    """cross_entropy_delta: the blame is the gap itself"""
    a = np.array([[0.98], [0.02]])
    y = np.array([[0.0], [1.0]])
    z = np.array([[3.8918202981], [-3.8918202981]])
    out = cross_entropy_delta(a, y, z)
    assert np.asarray(out).shape == (2, 1), (
        f"expected shape (2, 1) back, got {np.asarray(out).shape}: one "
        "blame per output neuron, the same column shape BP1 always "
        "returned."
    )
    expected = np.array([[0.98], [-0.98]])
    assert np.allclose(out, expected, atol=1e-12), (
        f"expected\n{expected}\ngot\n{out}\n"
        "Both neurons are as wrong as a sigmoid can be, and both blames "
        "come out at size 0.98, the gap. If you got 0.0192, the "
        "sigmoid_prime factor is still there: that factor is exactly what "
        "this cost was chosen to cancel. If the signs are swapped, the "
        "gap is output minus right answer, as it has been since Module 4."
    )


def test_delta_ignores_z():
    """cross_entropy_delta: the answer does not depend on z"""
    a = np.array([[0.7], [0.3]])
    y = np.array([[1.0], [0.0]])
    honest = cross_entropy_delta(a, y, np.array([[0.8472978604], [-0.8472978604]]))
    absurd = cross_entropy_delta(a, y, np.array([[500.0], [-500.0]]))
    assert np.allclose(honest, absurd, atol=1e-12), (
        f"the same a and y gave {np.asarray(honest).ravel()} with the real "
        f"z and {np.asarray(absurd).ravel()} with a nonsense z. Under this "
        "cost the output layer's blame depends on the gap alone; z is in "
        "the signature only because backprop hands it over. A function "
        "that still reads it is still slowing down on saturated neurons."
    )


def test_cost_and_delta_agree():
    """the gradient check: your delta really is your cost's slope, all 54"""
    # The course's backprop and the course's nudge-and-measure, deliberately:
    # a guarantee whose yardstick shares the code under test is not a
    # guarantee, and the code under test here is your delta.
    from course import backprop, numerical_gradient
    # Module 5's gradient-check network, a 3-5-4-2 with two hidden layers.
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

    nabla_w, nabla_b = backprop(weights, biases, x, y, cross_entropy_delta)

    def cost_fn(ws, bs):
        return cross_entropy_cost(ws, bs, x, y)

    num_w, num_b = numerical_gradient(cost_fn, weights, biases, eps=1e-4)

    names = ["nabla_w[0]", "nabla_w[1]", "nabla_w[2]",
             "nabla_b[0]", "nabla_b[1]", "nabla_b[2]"]
    worst = 0.0
    report = ""
    for name, yours, measured in zip(names, nabla_w + nabla_b, num_w + num_b):
        rel = np.abs(yours - measured) / np.maximum(np.abs(yours) + np.abs(measured), 1e-8)
        i = np.unravel_index(int(np.argmax(rel)), rel.shape)
        if rel[i] > worst:
            worst = float(rel[i])
            report = (
                f"worst disagreement: {name}, entry {tuple(int(k) for k in i)}: "
                f"nudging your cost says {measured[i]:.10f}, backprop driven "
                f"by your delta says {yours[i]:.10f}"
            )
    assert worst < 1e-7, (
        f"your two functions disagree: {report} (relative discrepancy "
        f"{worst:.2e}; the bar is 1e-7, and a matching pair lands near "
        "1e-9). This is the check that ties them together: it nudges every "
        "parameter, rescores with YOUR cost, and compares against the "
        "slopes your delta produces through backprop. A gap this size "
        "means the delta is not the slope of the cost you wrote. If every "
        "one of your slopes is a fixed multiple of the measured one, the "
        "cost formula has a stray factor (a half, or a division by the "
        "number of output entries)."
    )


def test_backprop_takes_the_blame_argument():
    """your backprop accepts a swapped-in BP1 and uses it"""
    import submission
    backprop = getattr(submission, "backprop", None)
    assert backprop is not None, (
        "your file has no backprop yet. This exercise changes the one you "
        "wrote in Module 5, so write that first."
    )
    params = list(inspect.signature(backprop).parameters)
    assert len(params) >= 5, (
        f"your backprop takes {len(params)} arguments, {params}, so there is "
        "nowhere to hand it a different BP1. Module 7's prompt shows the "
        "two lines: add output_delta=None to the signature, and make the "
        "BP1 line use it when it is not None. Leave it out and your "
        "function does exactly what it did in Module 5. Until you make "
        "that edit, everything that swaps the cost has no way in."
    )
    last = params[4]
    assert last == "output_delta", (
        f"your backprop's fifth argument is called {last!r}. The adapter "
        "written for you passes it by position, so any name runs, but "
        "output_delta is the name the rest of the course uses and the one "
        "the prompts will keep saying."
    )
    weights, biases = _fixture_net()
    x = np.array([[0.6], [-0.2]])
    y = np.array([[1.0]])
    plain = backprop(weights, biases, x, y)
    defaulted = backprop(weights, biases, x, y, None)
    for a, b in zip(plain[0] + plain[1], defaulted[0] + defaulted[1]):
        assert np.allclose(a, b), (
            "your backprop gives different slopes with the argument left "
            "out than with it passed as None. Those two have to mean the "
            "same thing: no replacement was supplied, so use the BP1 you "
            "already had."
        )

    # The yardstick is the course's nudge-and-measure against YOUR cost, not
    # another copy of BP1, for the same reason as the gradient check above:
    # a signature can carry output_delta while the BP1 line ignores it, and
    # nothing that shares that line can tell the two apart.
    from course import numerical_gradient
    swapped_w, swapped_b = backprop(weights, biases, x, y, cross_entropy_delta)

    def cost_fn(ws, bs):
        return cross_entropy_cost(ws, bs, x, y)

    num_w, num_b = numerical_gradient(cost_fn, weights, biases, eps=1e-4)
    names = ["nabla_w[0]", "nabla_w[1]",
             "nabla_b[0]", "nabla_b[1]"]
    worst = 0.0
    report = ""
    for name, yours, measured in zip(names, swapped_w + swapped_b,
                                     num_w + num_b):
        rel = np.abs(yours - measured) / np.maximum(
            np.abs(yours) + np.abs(measured), 1e-8)
        i = np.unravel_index(int(np.argmax(rel)), rel.shape)
        if rel[i] > worst:
            worst = float(rel[i])
            report = (
                f"worst disagreement: {name}, entry "
                f"{tuple(int(k) for k in i)}: nudging your cost says "
                f"{measured[i]:.10f}, your backprop handed "
                f"cross_entropy_delta says {yours[i]:.10f}"
            )
    assert worst < 1e-7, (
        f"the replacement BP1 is not reaching your slopes: {report} "
        f"(relative discrepancy {worst:.2e}; the bar is 1e-7, and a "
        "matching pair lands near 1e-9). The likely cause: the signature "
        "now carries output_delta and the BP1 line still computes the "
        "quadratic blame, the gap times the squash slope. When "
        "output_delta is not None the blame IS "
        "output_delta(activations[-1], y, zs[-1]), with nothing multiplied "
        "onto it, because cancelling that squash slope is what this cost "
        "is for."
    )
