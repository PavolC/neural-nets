# Tests for the data-preparation exercise. Fixtures are hardcoded literals
# chosen so every expected number can be checked by hand.
# Failure messages are teaching content (see CLAUDE.md).

import numpy as np
from submission import one_hot, split, standardize


def _two_features():
    """Two features on very different scales, four examples.

    Row 0 runs 1 to 4 (mean 2.5, spread 1.118); row 1 runs 1000 to 4000
    (mean 2500, spread 1118.03). The point of the exercise in miniature.
    """
    return np.array([[1.0, 2.0, 3.0, 4.0], [1000.0, 2000.0, 3000.0, 4000.0]])


def test_standardize_centres_and_scales_each_feature():
    """each feature ends up centred near 0 and about 1 wide"""
    X = _two_features()
    out = standardize(X)
    assert isinstance(out, tuple) and len(out) == 3, (
        f"expected a triple (X_scaled, mean, spread), got {type(out).__name__}. "
        "The mean and spread have to come back too: the validation and test "
        "columns need the training set's numbers, not their own."
    )
    scaled, mean, spread = out
    assert scaled.shape == (2, 4), (
        f"expected the scaled array to keep the shape (2, 4), got {scaled.shape}. "
        "Scaling changes the values in place, never the layout."
    )
    assert np.allclose(scaled.mean(axis=1), [0.0, 0.0], atol=1e-9), (
        f"each row should average 0 after centring, got {scaled.mean(axis=1)}. "
        "Subtract each row's own mean, with axis=1 and keepdims=True so the "
        "column of means lines up against the rows."
    )
    assert np.allclose(scaled.std(axis=1), [1.0, 1.0], atol=1e-9), (
        f"each row should have a spread of 1 after scaling, got "
        f"{scaled.std(axis=1)}. Divide each row by its own standard deviation "
        "(np.std with axis=1), not by a single number for the whole array: the "
        "two features here differ by a factor of a thousand and have to be "
        "scaled separately."
    )
    assert np.allclose(scaled[0], scaled[1], atol=1e-9), (
        "these two features are the same numbers a thousand times apart, so "
        "after scaling their rows must be identical. That is the whole point: "
        "the units a feature was measured in stop mattering."
    )


def test_standardize_reports_what_it_used():
    """mean and spread come back with the right shape and values"""
    X = _two_features()
    _, mean, spread = standardize(X)
    assert mean.shape == (2, 1) and spread.shape == (2, 1), (
        f"expected mean and spread to be columns of shape (2, 1), got "
        f"{mean.shape} and {spread.shape}. keepdims=True keeps them as columns, "
        "which is what makes X - mean line up."
    )
    assert np.allclose(mean.ravel(), [2.5, 2500.0]), (
        f"expected the means [2.5, 2500.0], got {mean.ravel()}."
    )
    assert np.allclose(spread.ravel(), [1.1180339887, 1118.0339887], atol=1e-6), (
        f"expected the spreads [1.118, 1118.03], got {spread.ravel()}. Use "
        "np.std, which measures the typical distance from the mean."
    )


def test_standardize_reuses_a_given_scaling():
    """passing mean and spread in applies them instead of measuring again"""
    X = _two_features()
    _, mean, spread = standardize(X)
    later = np.array([[5.0, 6.0], [5000.0, 6000.0]])
    scaled, out_mean, out_spread = standardize(later, mean, spread)
    assert np.allclose(out_mean, mean) and np.allclose(out_spread, spread), (
        "when mean and spread are handed in, hand the same ones back: a caller "
        "chains these calls and must not be given the new data's numbers."
    )
    expected = (later - mean) / spread
    assert np.allclose(scaled, expected), (
        f"expected\n{expected}\ngot\n{scaled}\nWhen a mean and spread are "
        "passed, use them and measure nothing. Re-measuring on validation or "
        "test data leaks what those rows look like into the preparation, and "
        "the score stops being honest."
    )
    assert scaled[0, 0] > 2.0, (
        "a value well above the training range should land well above 1 after "
        "scaling. If it came back near 0, the new data was re-centred on its "
        "own mean instead of the training mean."
    )


def test_standardize_survives_a_constant_feature():
    """a feature that never varies does not become NaN"""
    X = np.array([[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]])
    scaled, _, spread = standardize(X)
    assert np.isfinite(scaled).all(), (
        f"the scaled array holds a NaN or an infinity:\n{scaled}\nThe second "
        "feature is 7 in every column, so its spread is 0 and dividing by it "
        "is what produced that. Leave a zero-spread row as it is."
    )
    assert np.allclose(scaled[1], [0.0, 0.0, 0.0]), (
        f"expected the constant feature to end up all zeros, got {scaled[1]}. "
        "Centring alone does that: every value equals the mean."
    )
    assert float(spread[1, 0]) == 0.0, (
        "report the spread you measured, 0, rather than the 1 you divided by: "
        "a later call has to make the same choice on the same feature."
    )


def test_one_hot_builds_one_row_per_level():
    """one_hot puts a single 1.0 in each column, in that level's row"""
    got = one_hot(["Dream", "Biscoe", "Dream"], ["Biscoe", "Dream", "Torgersen"])
    expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert got.shape == (3, 3), (
        f"expected shape (3, 3), three levels by three examples, got {got.shape}. "
        "Levels become rows and examples stay columns, like every other feature."
    )
    assert np.allclose(got, expected), (
        f"expected\n{expected}\ngot\n{got}\nColumn k should hold a single 1.0, "
        "in the row of levels.index(values[k]). Row order follows levels, not "
        "the order the values happen to appear in."
    )


def test_one_hot_says_none_of_these_for_an_unknown_level():
    """a value outside levels becomes a column of zeros"""
    got = one_hot(["male", None, "unlisted"], ["female", "male"])
    assert got.shape == (2, 3), f"expected shape (2, 3), got {got.shape}."
    assert np.allclose(got[:, 1], [0.0, 0.0]) and np.allclose(got[:, 2], [0.0, 0.0]), (
        f"expected columns of zeros for the missing and the unlisted value, got\n{got}\n"
        "Guessing a level would invent data; all zeros says none of these, which "
        "is what a network can act on. Check membership before writing the 1.0 "
        "rather than letting an unknown value raise."
    )


def test_split_covers_everything_exactly_once():
    """the three sets partition 0..n-1, at the asked-for sizes, cut off the end"""
    train, val, test = split(100, np.random.default_rng(0), 0.2, 0.2)
    assert len(val) == 20 and len(test) == 20 and len(train) == 60, (
        f"expected 60 training, 20 validation and 20 test out of 100, got "
        f"{len(train)}, {len(val)} and {len(test)}. Validation and test take "
        "round(n * share) each; training keeps whatever is left."
    )
    everything = np.concatenate([train, val, test])
    assert sorted(everything.tolist()) == list(range(100)), (
        "the three sets together should be exactly 0 to 99, each once. An index "
        "in two sets means a row trained on and scored on, which is the one "
        "mistake this function exists to prevent."
    )
    assert not np.array_equal(train, np.arange(60)), (
        "the training indices came back in their original order, so nothing was "
        "shuffled. Data often arrives sorted by class, and cutting it unshuffled "
        "can hand training one species and test another."
    )
    order = np.random.default_rng(0).permutation(100)
    assert (
        np.array_equal(train, order[:60])
        and np.array_equal(val, order[60:80])
        and np.array_equal(test, order[80:])
    ), (
        "expected validation to be order[60:80], where order is one "
        f"rng.permutation(100): it should begin {order[60:63].tolist()} and it "
        f"begins {np.asarray(val)[:3].tolist()}. Training takes the front of "
        "the shuffled order, and the two held-back sets are cut off the END. "
        "Either end holds the same kind of rows after the shuffle, so which "
        "end you take is a convention, but it fixes which rows land in which "
        "set, and the counts Module 10's panel reports come from this cut."
    )


def test_split_is_repeatable():
    """the same seed cuts the same way"""
    first = split(50, np.random.default_rng(4), 0.2, 0.2)
    second = split(50, np.random.default_rng(4), 0.2, 0.2)
    assert all(np.array_equal(a, b) for a, b in zip(first, second)), (
        "two splits from the same seed differed. Take the order from one "
        "rng.permutation(n) on the generator you were handed; np.random.shuffle "
        "reads a global generator no seed of yours controls."
    )
