import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const prepareExercise: Exercise = {
  id: "prepare",
  title: "Getting your own data ready",
  prompt: [
    "Three functions, and not one of them is a network. This is the work that " +
      "sits between a file somebody hands you and the (features, examples) " +
      "matrix everything you have written expects. MNIST hid all of it: its " +
      "pixels arrived scaled, numeric, complete and already split.",
    "standardize(X, mean=None, spread=None) centres and scales each feature " +
      "on its own: subtract that row's mean, divide by that row's spread. It " +
      "returns (X_scaled, mean, spread), both of them columns of shape " +
      "(n_features, 1). The two NumPy calls you need are X.mean(axis=1, " +
      "keepdims=True) and X.std(axis=1, keepdims=True): axis=1 averages along " +
      "each row, and keepdims=True keeps the answer a column so that X - mean " +
      "lines up row by row.",
    "The default arguments carry the rule that keeps a score honest. When " +
      "mean and spread are handed in, use them and measure nothing, which is " +
      "how the validation and test columns get scaled by the training set's " +
      "numbers rather than their own. Measuring on those rows would let what " +
      "they look like leak into the preparation, and Module 7 already showed " +
      "what happens to a number the network was allowed to peek at. One edge " +
      "case: a feature that never varies has a spread of 0, so leave that row " +
      "alone instead of dividing by it.",
    "one_hot(values, levels) turns a column of category labels into rows: one " +
      "row per level, a single 1.0 per column. Module 5's labels were packed " +
      "this way and the course did the packing; here you do. A value that is " +
      "not in levels, including a missing one, gets a column of all zeros, " +
      "which says none of these rather than inventing a category.",
    "split(n, rng, val_share, test_share) shuffles 0 to n-1 with one " +
      "rng.permutation(n) and cuts it into three index arrays: validation and " +
      "test take round(n * share) from the end, training keeps the rest. The " +
      "shuffle is not decoration. Data usually arrives sorted, and this file " +
      "is sorted by species, so cutting it unshuffled would train on two " +
      "species and test on a third.",
    "Once the tests pass, look at what scaling does to the file you are about " +
      "to use. Append this and press Run my code:",
    {
      code:
        "import numpy as np\n" +
        "\n" +
        "with open('/penguins.json', 'rb') as f:\n" +
        "    columns, rows = load_penguins(f.read())\n" +
        "\n" +
        "measured = [r for r in rows if r[2] is not None]\n" +
        "X = np.array([[r[2], r[3], r[4], r[5]] for r in measured]).T\n" +
        "names = columns[2:6]\n" +
        "\n" +
        "print('before scaling:')\n" +
        "for name, row in zip(names, X):\n" +
        "    print(f'  {name:18} mean {row.mean():8.2f}  spread {row.std():7.2f}')\n" +
        "\n" +
        "Xs, mean, spread = standardize(X)\n" +
        "print('after:')\n" +
        "for name, row in zip(names, Xs):\n" +
        "    print(f'  {name:18} mean {row.mean():8.2f}  spread {row.std():7.2f}')",
    },
    "Body mass arrives averaging 4,201 with a spread of 800, and bill depth " +
      "averaging 17.15 with a spread of 1.97. A weight into the first hidden " +
      "layer meets numbers 245 times apart depending on which row it is on, " +
      "and Module 7's whole argument for dividing the starting weights assumed " +
      "inputs of about the same size. After scaling every row reads 0.00 and " +
      "1.00. The panel below is what that is worth.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "Take them one at a time; none is longer than four lines.\n\n" +
      "standardize: compute mean and spread only when they arrive as None, so " +
      "that a caller who passes them gets them used. Guard the division by " +
      "building a copy of spread with its zeros replaced by 1, and return the " +
      "spread you measured rather than the guarded one.\n\n" +
      "one_hot: start from np.zeros((len(levels), len(values))) and walk the " +
      "values with enumerate, writing a 1.0 only when the value is one of the " +
      "levels.\n\n" +
      "split: one call to rng.permutation(n) gives the order; the rest is " +
      "slicing it in three.",
    "The structure, in pseudocode:\n\n" +
      "    standardize(X, mean=None, spread=None):\n" +
      "        if mean is None:   mean   = X.mean(axis=1, keepdims=True)\n" +
      "        if spread is None: spread = X.std(axis=1, keepdims=True)\n" +
      "        safe = np.where(spread == 0, 1.0, spread)\n" +
      "        return (X - mean) / safe, mean, spread\n\n" +
      "    one_hot(values, levels):\n" +
      "        out = np.zeros((len(levels), len(values)))\n" +
      "        for column, value in enumerate(values):\n" +
      "            if value in levels:\n" +
      "                out[levels.index(value), column] = 1.0\n" +
      "        return out\n\n" +
      "    split(n, rng, val_share, test_share):\n" +
      "        order = rng.permutation(n)\n" +
      "        n_val, n_test = round(n * val_share), round(n * test_share)\n" +
      "        n_train = n - n_val - n_test\n" +
      "        return (order[:n_train],\n" +
      "                order[n_train : n_train + n_val],\n" +
      "                order[n_train + n_val :])",
  ],
};
