"""Exercise: stochastic gradient descent.

The gradients come for free: course.numerical_gradient estimates them by
finite differences (slow, but correct). Your job is the descent itself:
one update step, then the loop over epochs and mini-batches.

Provided by the course:
- quadratic_loss(weights, biases, X, Y): mean quadratic cost of the batch,
  a float. X is (n_in, m), Y is (n_out, m): samples are columns.
- numerical_gradient(loss_fn, weights, biases): returns (nabla_w, nabla_b),
  lists of arrays shaped like weights and biases. loss_fn must be a
  function of (weights, biases) returning a float.

Contract:
- sgd_step(weights, biases, X, Y, eta): one gradient descent step on the
  batch (X, Y) with learning rate eta. Every parameter moves downhill:
  new_param = param - eta * gradient. Return (new_weights, new_biases) as
  NEW lists of NEW arrays; do not modify the inputs.
- sgd(weights, biases, X, Y, eta, epochs, batch_size, rng): full training
  loop. For each epoch, visit every sample once in random mini-batches,
  in exactly this order (the tests depend on it):

      idx = rng.permutation(n)              # n = number of samples
      for k in range(0, n, batch_size):
          batch = idx[k:k+batch_size]       # column indices into X and Y

  Apply sgd_step to each mini-batch and return the final
  (weights, biases).
"""

import numpy as np
from course import quadratic_loss, numerical_gradient


def sgd_step(weights, biases, X, Y, eta):
    """One gradient descent step on the batch (X, Y).

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement sgd_step")


def sgd(weights, biases, X, Y, eta, epochs, batch_size, rng):
    """Mini-batch SGD over the whole dataset.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement sgd")
