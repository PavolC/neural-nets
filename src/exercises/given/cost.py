# Written for you, so this file needs nothing but NumPy. Adapted from Michael
# Nielsen's network.py (MIT license,
# github.com/mnielsen/neural-networks-and-deep-learning).
#
# Chapter 3 needs two things you have not written: the cost of a batch, and
# the slope of that cost for every parameter. Both are here so your sgd_step
# below can call them. The slopes are measured the hand way, by nudging each
# parameter and rescoring, which is correct and slow. Chapter 5 is where you
# write the fast way.


def quadratic_cost(weights, biases, X, Y):
    """Mean quadratic cost over a batch: 0.5 * sum((a - y)^2) / m.

    X is (n_in, m), Y is (n_out, m). Returns a float.
    """
    out = feedforward(weights, biases, X)
    m = X.shape[1]
    return 0.5 * float(((out - Y) ** 2).sum()) / m


def numerical_gradient(cost_fn, weights, biases, eps=1e-5):
    """Estimate every parameter's slope by nudging it to both sides.

    cost_fn(weights, biases) must return a float. Returns (nabla_w,
    nabla_b): lists of arrays with the same shapes as weights and biases,
    where each entry is that parameter's slope.

    Two cost evaluations, so two full forward passes, per parameter.
    """
    def grad_of(params, index):
        p = params[index]
        g = np.zeros_like(p, dtype=float)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            original = p[i]
            p[i] = original + eps
            up = cost_fn(weights, biases)
            p[i] = original - eps
            down = cost_fn(weights, biases)
            p[i] = original
            g[i] = (up - down) / (2 * eps)
        return g

    nabla_w = [grad_of(weights, l) for l in range(len(weights))]
    nabla_b = [grad_of(biases, l) for l in range(len(biases))]
    return nabla_w, nabla_b


def gradient(weights, biases, X, Y, eps=1e-5):
    """Slopes of the quadratic cost on the batch (X, Y), one per parameter."""
    def cost_fn(ws, bs):
        return quadratic_cost(ws, bs, X, Y)

    return numerical_gradient(cost_fn, weights, biases, eps)
