"""Exercise: a sigmoid neuron.

Two small functions. sigmoid is the activation function you will use for
the rest of the course. fire is one neuron: it weighs its inputs, adds its
bias, and squashes the result.

Shape contract:
- sigmoid(z): z is a number or a NumPy array of any shape. Return the same
  shape, with the function applied elementwise (NumPy does this for you if
  you write the formula with np.exp).
- fire(w, b, x): w and x are column vectors of shape (n, 1), b is a plain
  float. Return the neuron's output sigmoid(w . x + b) as a plain float,
  where w . x is the dot product (a single number).
"""

import numpy as np


def sigmoid(z):
    """Return 1 / (1 + e^(-z)), elementwise.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement sigmoid")


def fire(w, b, x):
    """Return this neuron's output for input x, as a plain float.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement fire")
