# Compute the network's output for one input column x, shape (n_in, 1).
#
# A network has several layers, so you receive one weight matrix and one bias
# column PER LAYER, packed in two parallel lists. weights[l] and biases[l]
# describe the same layer: the matrix has one row per neuron, the column has
# that layer's one bias per neuron. Concretely, a call with Module 1's XOR
# network looks like this:
#
#     weights = [np.array([[6., 6.],        # hidden layer: 2 neurons,
#                          [6., 6.]]),      #   one row of 2 weights each
#                np.array([[8., -8.]])]     # output layer: 1 neuron, 2 weights
#     biases  = [np.array([[-3.], [-9.]]),  # hidden layer: one bias per neuron
#                np.array([[-4.]])]         # output layer: one bias
#     feedforward(weights, biases, np.array([[1.], [0.]]))
#
# Shape note: shapes are (rows, columns), and rows belong to the RECEIVING
# layer, so data flowing 2 -> 1 gives a (1, 2) matrix. The inner numbers
# must touch: (1, 2) @ (2, 1) -> (1, 1).
#
# Each layer turns the running activation a into sigmoid(w @ a + b). Return
# the final activation, shape (n_out, 1).
#
# sigmoid is the one you wrote in Module 1, higher up this same file. Call it
# by name; there is nothing to import.


def feedforward(weights, biases, x):
    """Return the network's output activation for the input column x.

    Replace the raise below with your implementation.
    """
    raise NotImplementedError("implement feedforward")
