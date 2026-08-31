# Three functions, none of them a network. This is the work that stands
# between a file someone hands you and the (features, examples) matrix
# everything you have written expects.
#
# Contract:
#
# - standardize(X, mean=None, spread=None): X is (n_features, m), one row per
#   feature and one column per example, the packing every module has used.
#   Returns (X_scaled, mean, spread).
#
#   Each FEATURE (each row) is shifted and scaled on its own: subtract that
#   row's mean, divide by that row's spread, so every feature ends up centred
#   near 0 and about 1 wide. mean and spread come back with shape
#   (n_features, 1), so they can be handed to a later call.
#
#   When mean and spread are passed in, use them and do not measure anything:
#   that is how the validation and test columns get the training set's
#   scaling rather than their own. A feature that never varies has a spread
#   of 0; leave those rows alone rather than dividing by zero, and hand back
#   the 0 you measured rather than the 1 you divided by, because a later call
#   has to make the same decision about the same feature.
#
# - one_hot(values, levels): values is a list of m labels (strings), levels is
#   the list of the possible ones, in the order you want the rows. Returns an
#   (len(levels), m) array of 0.0 and 1.0, with a single 1.0 per column in
#   the row of that column's level. A value that is not in levels gets a
#   column of all zeros, which is what an unknown category should say: none
#   of these.
#
# - split(n, rng, val_share, test_share): shuffle the numbers 0 to n-1 with
#   one rng.permutation(n), then cut them into three index arrays and return
#   (train_idx, val_idx, test_idx). The last two hold round(n * val_share)
#   and round(n * test_share) entries, taken from the END of the shuffled
#   order, and training keeps the rest. Every index appears exactly once.
#   Either end holds the same kind of rows after the shuffle, so which end
#   you take is a convention, but it fixes which rows land in which set and
#   the tests check for it.
#
# Nothing above you in this file is needed here. np is all it takes.


def standardize(X, mean=None, spread=None):
    """Centre and scale each feature; reuse a given mean and spread if handed one."""
    raise NotImplementedError("implement standardize")


def one_hot(values, levels):
    """Turn a list of category labels into rows of 0.0 and 1.0."""
    raise NotImplementedError("implement one_hot")


def split(n, rng, val_share, test_share):
    """Shuffle 0..n-1 once and cut it into training, validation and test."""
    raise NotImplementedError("implement split")
