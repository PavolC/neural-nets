# Two small functions. Neither one is an algorithm: the first is a formula
# over a batch, the second is a subtraction.
#
# Before either of them, one edit to code you already wrote. See the prompt
# on the page: your backprop from Chapter 5 needs to accept a swapped-in BP1.
#
# Already in this file, above you: feedforward, from Chapter 2. Works on one
# column or a whole batch of columns.
#
# Contract:
# - cross_entropy_cost(weights, biases, X, Y): the mean cross-entropy cost of
#   the batch, one float. X is (n_in, m), Y is (n_out, m): samples are
#   columns, as always. Run the network on X to get its answers A, then for
#   every entry pay
#
#       -( y * ln(a)  +  (1 - y) * ln(1 - a) )
#
#   add those payments up inside each column (one column is one example's
#   bill), and average over the m columns. In NumPy the natural logarithm is
#   np.log: log base e, not base 10.
#
#   One hole to plug first. A perfectly confident network makes a exactly 1.0
#   or exactly 0.0, np.log of 0.0 is minus infinity, and one of the two terms
#   is multiplied by a zero, so 0 * (-inf) evaluates to nan and poisons the
#   whole average. Pull the answers a hair inside the open interval before
#   taking any logarithm:
#
#       A = np.clip(A, 1e-12, 1.0 - 1e-12)
#
# - cross_entropy_delta(a, y, z): the output layer's blame under that cost,
#   the replacement for BP1. a, y and z are the output layer's columns, all
#   (n_out, 1); return the same shape. z is in the signature because BP1's
#   old version needed it. This version does not.


def cross_entropy_cost(weights, biases, X, Y):
    """Mean cross-entropy cost of the batch (X, Y), as a float.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement cross_entropy_cost")


def cross_entropy_delta(a, y, z):
    """The output layer's blame under the cross-entropy cost.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement cross_entropy_delta")
