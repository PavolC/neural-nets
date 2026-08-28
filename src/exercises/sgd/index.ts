import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const sgdExercise: Exercise = {
  id: "sgd",
  title: "Stochastic gradient descent",
  prompt: [
    "The module ended with one rule: new parameter = old parameter - " +
      "eta * slope. eta is the learning rate, the η from the module's " +
      "sliders: the small number you choose that sets the step size. This " +
      "exercise is that rule as code, in the two functions waiting in the " +
      "editor.",
    "sgd_step(weights, biases, X, Y, eta) takes one step of descent on " +
      "one mini-batch: X holds that batch's inputs as columns and Y the " +
      "right answers in matching columns, the packing the module just " +
      "showed. Call gradient(weights, biases, X, Y), which is written for you " +
      "in the section just above yours, to get the " +
      "slopes: it takes the same inputs as quadratic_cost, and where " +
      "quadratic_cost returns the score, gradient returns (nabla_w, " +
      "nabla_b). Those two names are the module's ∇C: nabla is how that " +
      "upside-down delta is read, and the pair is the one list of slopes " +
      "split in two, the weights' slopes and the biases' slopes. There is " +
      "one slope array per parameter array, nabla_w[0] shaped like " +
      "weights[0] and so on, and each number in it is the slope of the " +
      "knob in the matching spot. Apply the rule to every weight and " +
      "bias, and return (new_weights, new_biases), built as new lists of " +
      "new arrays with the inputs untouched (the tests check).",
    "sgd(weights, biases, X, Y, eta, epochs, batch_size, rng) loops it " +
      "into training. Careful: here X and Y are the same packing but hold " +
      "the whole dataset, and cutting them into mini-batches is your job. " +
      "Repeat epochs times: shuffle the column order, cut it into " +
      "mini-batches, call your sgd_step on each, feeding each result into " +
      "the next call. Return the final (weights, biases). That is all " +
      "training is.",
    "The shuffling is prescribed, because the tests replay your run and " +
      "compare to six decimal places. Each epoch, idx = rng.permutation(n) " +
      "deals the sample numbers 0 to n-1 into a random order (n = 6 might " +
      "give [3, 0, 5, 1, 4, 2]); the mini-batches are the consecutive " +
      "slices idx[0:batch_size], idx[batch_size:2*batch_size], and so on. " +
      "To pull those samples out, use one new piece of NumPy: an array of " +
      "indices works anywhere a single index does, so X[:, batch] is the " +
      "matrix of just those columns. Use the rng for nothing else.",
    "A satisfying experiment once sgd_step works: let it repair Module 1's " +
      "slider network from its starting position, the one the module scored " +
      "by hand at cost 0.0875. Send this to the scratch pad, then press " +
      "Run my code:",
    {
      code:
        "\n" +
        "weights = [np.array([[2., 2.], [2., 2.]]),   # the sliders' start position\n" +
        "           np.array([[4., -4.]])]\n" +
        "biases = [np.array([[-1.], [-3.]]),\n" +
        "          np.array([[-2.]])]\n" +
        "X = np.array([[0., 1., 0., 1.],              # the four corners, as columns\n" +
        "              [0., 0., 1., 1.]])\n" +
        "Y = np.array([[0., 1., 1., 0.]])             # the contrarian's answers\n" +
        "\n" +
        'print("cost before:", quadratic_cost(weights, biases, X, Y))\n' +
        "for _ in range(100):\n" +
        "    weights, biases = sgd_step(weights, biases, X, Y, 2.0)\n" +
        'print("cost after: ", quadratic_cost(weights, biases, X, Y))\n' +
        'print("the four answers:", feedforward(weights, biases, X))',
    },
    "The cost falls from 0.08758 to about 0.026, and the four answers lean " +
      "toward 0, 1, 1, 0. Your code just did what your hands did with the " +
      "sliders.",
    "The 2.0 is a learning rate, found by trying. Now make it misbehave: " +
      "edit the eta in the block you appended and run again. At 20.0 the " +
      "very first step throws the cost up to 0.165, nearly double where it " +
      "started, and it bounces for a dozen steps before settling, yet by " +
      "step 100 it lands near 0.002, lower than 2.0 managed (print the " +
      "cost inside the loop to watch the ride). Bigger steps gamble, and " +
      "here the gamble pays. At 200.0 it does not: the cost jams near " +
      "0.250 and never comes back. And notice the bowl's numbers do not " +
      "carry over: there, 0.5 already overshot; here 20.0 survives, " +
      "because this landscape is far flatter (you measured its slopes at " +
      "around 0.04). Every landscape has its own safe range.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "For sgd_step: one call to gradient(weights, biases, X, Y) gives " +
      "(nabla_w, nabla_b). Then the update is one list comprehension per " +
      "parameter list, with zip pairing each parameter with its slope " +
      "(the same zip that walked weights and biases in feedforward): " +
      "each new parameter is param - eta * slope. For sgd: two nested " +
      "loops, epochs outside, batches inside, reassigning weights and " +
      "biases with each sgd_step so progress carries forward.",
    "The structure, in pseudocode:\n\n" +
      "    sgd_step(weights, biases, X, Y, eta):\n" +
      "        nabla_w, nabla_b = gradient(weights, biases, X, Y)\n" +
      "        new_weights = [w - eta * nw  for each w, nw pair]\n" +
      "        new_biases  = [b - eta * nb  for each b, nb pair]\n" +
      "        return new_weights, new_biases\n\n" +
      "    sgd(weights, biases, X, Y, eta, epochs, batch_size, rng):\n" +
      "        n = number of columns of X\n" +
      "        repeat epochs times:\n" +
      "            idx = rng.permutation(n)\n" +
      "            for k = 0, batch_size, 2*batch_size, ... up to n:\n" +
      "                batch = idx[k : k + batch_size]\n" +
      "                weights, biases = sgd_step(weights, biases,\n" +
      "                                           X[:, batch], Y[:, batch], eta)\n" +
      "        return weights, biases",
  ],
};
