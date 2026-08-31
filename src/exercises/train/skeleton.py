# Two functions. Neither one is a new idea: everything they call is something
# you have already written, and all of it is above you in this file. What is
# new is that nothing is wired for you.
#
# Already in this file, above you:
# - init_network(sizes, rng): Chapter 7. Draws a network, weights divided by
#   the square root of their layer's input count.
# - batch_gradient(weights, biases, X, Y, output_delta): the adapter written
#   for you in Chapter 5. Runs your backprop once per column of the batch and
#   averages the slopes. Returns (nabla_w, nabla_b).
# - cross_entropy_delta(a, y, z): Chapter 7's BP1, the gap alone.
# - l2_step(weights, biases, nabla_w, nabla_b, eta, lmbda, n): Chapter 7's
#   update, with weight decay.
# - feedforward(weights, biases, X): Chapter 2. Works on a batch of columns.
#
# Contract:
#
# - accuracy(weights, biases, X, y): the share of X's columns the network
#   reads correctly, as a float between 0 and 1. X is (n_in, m). y holds
#   integer class ids, shape (m,), NOT one-hot columns: y[k] is the right
#   answer for column k. A network's answer for a column is the row number
#   of its most confident output.
#
# - train(sizes, X, Y, X_test, y_test, epochs, eta, lmbda, batch_size, rng):
#   builds a network of the given sizes and trains it. X is (n_in, n) and Y
#   is (n_out, n) one-hot, the packing every chapter has used; X_test and
#   y_test are the held-out images and their integer labels, for scoring
#   only, never trained on. Returns (weights, biases, history), where
#   history[i] is the accuracy on the held-out data after epoch i + 1, so
#   len(history) == epochs.
#
#   The order is prescribed, because the tests check the exact numbers it
#   produces:
#
#     1. draw the network from rng, with init_network
#     2. for each epoch:
#          a. idx = rng.permutation(n), n being the number of training
#             columns
#          b. walk idx in slices of batch_size, front to back
#          c. for each slice: the batch's gradient, then one l2_step
#          d. after the whole epoch, score the held-out data and append it
#     3. return the trained parameters and the history
#
#   One generator does both jobs, drawing the network and then shuffling, so
#   a single seed fixes the entire run.


def accuracy(weights, biases, X, y):
    """Share of the columns of X that the network reads correctly."""
    raise NotImplementedError("implement accuracy")


def train(sizes, X, Y, X_test, y_test, epochs, eta, lmbda, batch_size, rng):
    """Build a network, train it, and score it after every epoch."""
    raise NotImplementedError("implement train")
