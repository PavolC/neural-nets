# Module 4's four equations as code. Two functions; the first is a single
# formula.
#
# Already in this file, above you: sigmoid(z), from Module 1, elementwise,
# any shape.
#
# Contract:
# - sigmoid_prime(z): sigmoid's steepness at z, sigmoid(z) * (1 - sigmoid(z)),
#   elementwise: any shape in, the same shape out.
# - backprop(weights, biases, x, y): the exact slopes of ONE example's
#   quadratic cost, C = 0.5 * sum of squared gaps between the output and y.
#   x is one input column, shape (n_in, 1); y is its right answer as a
#   one-hot column, shape (n_out, 1). Returns (nabla_w, nabla_b): lists of
#   arrays shaped exactly like weights and biases, one slope per parameter.
#   Read the parameters only; never modify them.
#
# The plan (the four equations, in the order the code meets them):
# 1. Forward pass, keeping receipts: your feedforward from Module 2, but
#    append every z to a list zs and every activation to a list activations
#    (which starts as [x]).
# 2. BP1: the output layer's delta is (a - y) * sigmoid_prime(z), using
#    that layer's stored z. Elementwise products throughout: plain *.
# 3. BP3 and BP4: that layer's slopes, read off delta. nabla_b is delta
#    itself; nabla_w is delta @ (previous layer's activations).T.
# 4. BP2: walk backward one layer at a time. Each earlier layer's delta is
#    (next layer's weights, transposed) @ delta, times sigmoid_prime of
#    THIS layer's stored z. Read off BP3 and BP4 at every stop.


def sigmoid_prime(z):
    """Sigmoid's steepness at z: sigmoid(z) * (1 - sigmoid(z)), elementwise.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement sigmoid_prime")


def backprop(weights, biases, x, y):
    """Exact slopes of one example's quadratic cost, via BP1 to BP4.

    Returns (nabla_w, nabla_b), shaped like (weights, biases).
    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement backprop")
