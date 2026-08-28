# Written for you, so this file needs nothing but NumPy. Adapted from Michael
# Nielsen's network.py (MIT license,
# github.com/mnielsen/neural-networks-and-deep-learning).
#
# Your backprop above answers for one example. Training walks mini-batches,
# so something has to call it once per column and average the slopes. This is
# that adapter, and it is what Module 5's training panel runs on your behalf.
#
# The output_delta argument is for later: Module 7 swaps BP1 out, and this is
# what hands the replacement through. Until then it stays None and backprop
# is called the way you wrote it in Module 5, with four arguments.


def batch_gradient(weights, biases, X, Y, output_delta=None):
    """A mini-batch's gradient: backprop per column, slopes averaged.

    X is (n_in, m), Y is (n_out, m). Returns (nabla_w, nabla_b).
    """
    m = X.shape[1]
    nabla_w = [np.zeros_like(w) for w in weights]
    nabla_b = [np.zeros_like(b) for b in biases]
    for k in range(m):
        xk, yk = X[:, k:k + 1], Y[:, k:k + 1]
        if output_delta is None:
            dw, db = backprop(weights, biases, xk, yk)
        else:
            dw, db = backprop(weights, biases, xk, yk, output_delta)
        nabla_w = [t + d for t, d in zip(nabla_w, dw)]
        nabla_b = [t + d for t, d in zip(nabla_b, db)]
    return [t / m for t in nabla_w], [t / m for t in nabla_b]
