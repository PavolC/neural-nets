import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const trainExercise: Exercise = {
  id: "train",
  title: "The whole program",
  prompt: [
    "Every training run so far has been started for you. The panels loaded the " +
      "data, built the network, ran the epochs, cut the mini-batches, called " +
      "your gradient, applied your update and scored the result. Your functions " +
      "did the work inside that loop; the loop itself was the course's. This " +
      "exercise is the loop.",
    "The contract, in two functions. accuracy(weights, biases, X, y) returns " +
      "the share of X's columns the network reads correctly, as a float. Note " +
      "what y is here: integer class ids, shape (m,), one per column, not the " +
      "one-hot columns of Y. A network's answer is the output neuron that is " +
      "most confident, so this is np.argmax(..., axis=0) down each column, " +
      "compared against y. You have never written this: the panels always " +
      "scored for you.",
    "train(sizes, X, Y, X_test, y_test, epochs, eta, lmbda, batch_size, rng) " +
      "builds a network and trains it, returning (weights, biases, history) " +
      "with one held-out score per epoch. Everything it calls is your own work " +
      "from earlier modules, handed back through the course module: " +
      "init_network draws the network, batch_gradient runs your backprop once " +
      "per column of a mini-batch and averages the slopes, cross_entropy_delta " +
      "is the blame it hands to BP1, and l2_step is the update.",
    "The order is prescribed, because the tests check the exact numbers the " +
      "loop produces: draw the network first, then per epoch take one " +
      "rng.permutation(n), walk it in slices of batch_size front to back, take " +
      "one l2_step per slice, and score the held-out data once at the end of " +
      "the epoch. One generator does both jobs, the draw and the shuffles, so " +
      "one seed fixes the whole run. Two traps worth naming: l2_step wants the " +
      "size of the whole training set as n, not the size of the mini-batch; and " +
      "every function here returns new arrays, so the next batch has to be " +
      "given what the last one returned.",
    "The tests climb from accuracy on a network whose answers you can predict " +
      "by hand, to the shapes and the length of the history, to whether the " +
      "thing actually learns, to the exact weights after four epochs.",
    "Once they pass, run the program on a problem of your own before pointing " +
      "it at the digits. This is the same three-band problem the tests use, " +
      "trained for longer. Send it to the scratch pad and press Run my code:",
    {
      code:
        "import numpy as np\n" +
        "\n" +
        "x1 = np.linspace(-2.75, 2.75, 12)\n" +
        "X = np.vstack([x1, np.array([0.2, 0.8] * 6)])\n" +
        "label = lambda v: 0 if v < -1.0 else (1 if v < 1.0 else 2)\n" +
        "y = np.array([label(v) for v in x1])\n" +
        "Y = np.zeros((3, 12))\n" +
        "Y[y, np.arange(12)] = 1.0\n" +
        "\n" +
        "t1 = np.linspace(-2.45, 2.45, 6)\n" +
        "X_test = np.vstack([t1, np.full(6, 0.5)])\n" +
        "y_test = np.array([label(v) for v in t1])\n" +
        "\n" +
        "w, b, history = train([2, 6, 3], X, Y, X_test, y_test,\n" +
        "                      20, 1.0, 0.0, 3, np.random.default_rng(0))\n" +
        'print("held-out score per epoch:")\n' +
        "for epoch, score in enumerate(history, 1):\n" +
        '    print(" ", epoch, round(score, 3))',
    },
    "It starts at two thirds and reaches every one of the six held-out points " +
      "by the twentieth epoch. Then change one thing at a time and rerun: a " +
      "step size of 0.1, or 8.0; a hidden layer of 2 neurons instead of 6; " +
      "lmbda at 8.0. That is the search Module 7 described, and it is now " +
      "yours to run.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "Write accuracy first, and test it before you touch train: it is two " +
      "lines, and every score in the history goes through it. Run the network " +
      "on the whole batch with feedforward, take np.argmax(out, axis=0) to get " +
      "one answer per column, compare with y, and take the mean of the " +
      "resulting True/False array, wrapped in float(...).\n\n" +
      "For train, write the shape of the loop before the contents: an outer " +
      "loop over epochs, a shuffle, an inner loop over the starts of the " +
      "slices, and one append at the end of each epoch. The body of the inner " +
      "loop is two calls, one for the gradient and one for the step.",
    "The structure, in pseudocode:\n\n" +
      "    accuracy(weights, biases, X, y):\n" +
      "        guesses = np.argmax(feedforward(weights, biases, X), axis=0)\n" +
      "        return float((guesses == y).mean())\n\n" +
      "    train(sizes, X, Y, X_test, y_test, epochs, eta, lmbda, batch_size, rng):\n" +
      "        weights, biases = init_network(sizes, rng)\n" +
      "        n = X.shape[1]\n" +
      "        history = []\n" +
      "        repeat epochs times:\n" +
      "            idx = rng.permutation(n)\n" +
      "            for k in range(0, n, batch_size):\n" +
      "                batch = idx[k : k + batch_size]\n" +
      "                nabla_w, nabla_b = batch_gradient(weights, biases,\n" +
      "                                                  X[:, batch], Y[:, batch],\n" +
      "                                                  cross_entropy_delta)\n" +
      "                weights, biases = l2_step(weights, biases, nabla_w, nabla_b,\n" +
      "                                          eta, lmbda, n)\n" +
      "            history.append(accuracy(weights, biases, X_test, y_test))\n" +
      "        return weights, biases, history\n\n" +
      "    X[:, batch] takes those columns of X, in that order: the same\n" +
      "    indexing your Module 3 sgd used to cut its mini-batches.",
  ],
};
