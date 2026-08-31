# One function, one changed line. Your sgd_step from Chapter 3 moved every
# parameter against its slope; this one shrinks every weight a little first.
#
# Contract:
# - l2_step(weights, biases, nabla_w, nabla_b, eta, lmbda, n): one descent
#   step, with the gradient handed in rather than measured here. eta is the
#   learning rate, lmbda (spelled without the a, since lambda is a Python
#   keyword) is the regularization strength, and n is the number of examples
#   in the whole training set, not in this mini-batch.
#
#   Weights:  w  <-  (1 - eta * lmbda / n) * w  -  eta * nabla_w
#   Biases:   b  <-  b - eta * nabla_b
#
#   Return (new_weights, new_biases) as NEW lists of NEW arrays; do not
#   modify the inputs. With lmbda = 0 the weight line has to reduce exactly
#   to Chapter 3's plain step, since the factor becomes 1.


def l2_step(weights, biases, nabla_w, nabla_b, eta, lmbda, n):
    """One descent step with the weights decayed toward zero.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement l2_step")
