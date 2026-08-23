"""Exercise: feedforward.

Compute a network's output activation for a single input.

Shape contract (course conventions, same everywhere):
- x is a column vector of input activations, shape (n_in, 1). Never (n_in,).
- weights is a list of arrays. weights[l] connects layer l to layer l+1 and
  has shape (size of layer l+1, size of layer l).
- biases is a list of column vectors. biases[l] has shape (size of layer l+1, 1).
- Each layer computes sigmoid(w @ a + b), where a is the previous layer's
  activation. Return the final activation, shape (n_out, 1).

You already built sigmoid in Module 1; the course provides it here.
"""

import numpy as np
from course import sigmoid


def feedforward(weights, biases, x):
    """Return the network's output activation for the input column vector x.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement feedforward")
