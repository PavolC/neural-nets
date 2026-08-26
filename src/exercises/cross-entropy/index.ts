import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const crossEntropyExercise: Exercise = {
  id: "cross-entropy",
  title: "The cross-entropy cost",
  prompt: [
    "Two functions, both short. cross_entropy_cost scores a batch, so the " +
      "panels below have something to chart. cross_entropy_delta is the " +
      "replacement for BP1, and it is the whole fix: one subtraction.",
    "The cost, precisely: cross_entropy_cost(weights, biases, X, Y) returns " +
      "one float. X is (n_in, m) and Y is (n_out, m), samples as columns, " +
      "the same pair your sgd has taken since Module 3. Run the network on " +
      "X with the course's feedforward, then charge every entry " +
      "-(y * ln(a) + (1 - y) * ln(1 - a)), add the charges up inside each " +
      "column, and average over the m columns. Two notation points: in " +
      "NumPy the natural logarithm is np.log (log base e, not base 10), " +
      "and there is no bookkeeping half here, because this cost's slope " +
      "comes out clean without one.",
    "Plug the hole before you take any logarithm. A confident network " +
      "answers exactly 1.0 or exactly 0.0 in floating point, np.log(0.0) " +
      "is minus infinity, and the term multiplied by zero then evaluates " +
      "0 * (-inf) as nan, which poisons the average and every chart drawn " +
      "from it. One line prevents it: A = np.clip(A, 1e-12, 1.0 - 1e-12), " +
      "which pulls every answer a hair inside the open interval. The " +
      "clipped edge is also what turns an infinite charge for a " +
      "confidently wrong answer into a large finite one, about 27.6.",
    "The delta, precisely: cross_entropy_delta(a, y, z) takes the output " +
      "layer's three columns, each (n_out, 1), and returns the blame, the " +
      "same shape. z is in the signature because backprop passes it; this " +
      "version has no use for it, and the last test checks that it really " +
      "does not.",
    "The tests climb the usual ladder: the cost on one example, the cost " +
      "over a batch, the cost when the network is perfectly confident " +
      "(where the nan lives), the delta's values, the delta's " +
      "independence from z, and last a gradient check that ties the two " +
      "functions together. That check nudges all 54 parameters of Module " +
      "5's 3-5-4-2 network, rescores with YOUR cost, and compares against " +
      "the slopes YOUR delta produces through backprop. Passing it means " +
      "the delta you wrote really is the slope of the cost you wrote.",
    "Once the tests pass, put the old blame and the new one side by side on " +
      "Module 5's own fixture network. Append this and press Run my code:",
    {
      code:
        "from course import backprop, quadratic_output_delta, quadratic_cost\n" +
        "\n" +
        "weights = [np.array([[0.5, -0.3], [-0.8, 0.9], [0.1, 0.4]]),\n" +
        "           np.array([[0.7, -0.5, 0.2]])]\n" +
        "biases = [np.array([[0.1], [-0.2], [0.3]]), np.array([[-0.1]])]\n" +
        "x = np.array([[1.0], [0.0]])   # the concert corner (1, 0)\n" +
        "y = np.array([[1.0]])          # right answer: go\n" +
        "\n" +
        'print("quadratic cost:    ", quadratic_cost(weights, biases, x, y))\n' +
        'print("cross-entropy cost:", cross_entropy_cost(weights, biases, x, y))\n' +
        "\n" +
        "_, old = backprop(weights, biases, x, y, quadratic_output_delta)\n" +
        "_, new = backprop(weights, biases, x, y, cross_entropy_delta)\n" +
        'print("output blame, quadratic:    ", old[-1][0, 0])\n' +
        'print("output blame, cross-entropy:", new[-1][0, 0])\n' +
        'print("hidden blames, quadratic:    ", old[0].ravel())\n' +
        'print("hidden blames, cross-entropy:", new[0].ravel())',
    },
    "The output blame goes from -0.1012 to -0.4165, four times larger, and " +
      "4.115 is exactly 1 divided by that neuron's steepness 0.2430: the " +
      "factor the new cost cancels. The hidden blames grow by the same " +
      "factor, because BP2 carries whatever arrives from the output layer " +
      "and multiplies by its own numbers, which did not change. And the " +
      "two costs are different numbers for the same network (0.0868 " +
      "against 0.5387), which is worth seeing once: they disagree about " +
      "how bad this answer is, not about which answer is right.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "For the cost: A = feedforward(weights, biases, X) gives every answer " +
      "at once, (n_out, m). Clip it, then build the whole per-entry charge " +
      "as one array with * and np.log, exactly as written in the prompt, " +
      "with no loops. .sum() with no arguments adds every entry of an " +
      "array, which is the summing inside columns and across columns in " +
      "one call, so all that remains is dividing by m = X.shape[1] and " +
      "wrapping it in float(...). For the delta: read the last line of the " +
      "section above this exercise, then write it.",
    "The structure, in pseudocode:\n\n" +
      "    cross_entropy_cost(weights, biases, X, Y):\n" +
      "        A = feedforward(weights, biases, X)\n" +
      "        A = np.clip(A, 1e-12, 1.0 - 1e-12)\n" +
      "        charges = -(Y * np.log(A) + (1 - Y) * np.log(1 - A))\n" +
      "        return float(charges.sum()) / X.shape[1]\n\n" +
      "    cross_entropy_delta(a, y, z):\n" +
      "        return <the gap>\n\n" +
      "    The gap is output minus right answer, the order the course has\n" +
      "    used since Module 4. Nothing multiplies it.",
  ],
};
