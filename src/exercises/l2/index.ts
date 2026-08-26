import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const l2Exercise: Exercise = {
  id: "l2",
  title: "The decaying update step",
  prompt: [
    "The smallest exercise in the course: your Module 3 update step with one " +
      "factor added to one of its two lines.",
    "The contract: l2_step(weights, biases, nabla_w, nabla_b, eta, lmbda, n) " +
      "performs one descent step and returns (new_weights, new_biases). The " +
      "gradient arrives as an argument rather than being measured here, " +
      "since the caller already has it. eta is the learning rate. lmbda is " +
      "the regularization strength, spelled without the a because lambda is " +
      "a Python keyword and cannot be a variable name. n is the size of the " +
      "whole training set, not of this mini-batch: the term came from a cost " +
      "averaged over all n examples, so the same n belongs here whatever the " +
      "batch size.",
    "The weights are the changed line, w becomes " +
      "(1 - eta * lmbda / n) * w - eta * nabla_w. The biases keep Module 3's " +
      "line exactly, b becomes b - eta * nabla_b. Return new lists of new " +
      "arrays and leave the inputs alone, as your sgd_step did: the panel " +
      "below keeps two runs' parameters side by side.",
    "The tests check the arithmetic on numbers small enough to verify by " +
      "hand, that biases are untouched, that lmbda = 0 reproduces your " +
      "Module 3 step exactly (the panel's baseline run goes through this " +
      "same function, so that case has to be right), and that a weight left " +
      "with no gradient at all shrinks by the same fraction every step.",
    "Once the tests pass, watch a weight decay on its own, with the data " +
      "silent. Append this and press Run my code:",
    {
      code:
        "w = [np.array([[1.0]])]\n" +
        "b = [np.array([[0.0]])]\n" +
        "no_slope = [np.zeros((1, 1))]\n" +
        "\n" +
        "eta, lmbda, n = 0.5, 5.0, 1000\n" +
        'print("shrink factor per step:", 1 - eta * lmbda / n)\n' +
        "for step in range(1, 501):\n" +
        "    w, b = l2_step(w, b, no_slope, no_slope, eta, lmbda, n)\n" +
        "    if step in (1, 10, 100, 500):\n" +
        '        print("after", step, "steps:", round(float(w[0][0, 0]), 6))',
    },
    "At lmbda = 5 the factor is 0.9975, so the weight reads 0.9975 after one " +
      "step, 0.97528 after ten, 0.778557 after a hundred and 0.286057 after " +
      "five hundred: multiplying by the same fraction repeatedly, which is a " +
      "curve that keeps halving rather than a line that reaches zero. Five " +
      "hundred steps is five epochs of the run below, whose default lmbda of " +
      "1 makes the factor a gentler 0.9995. Either way the panel's weights do " +
      "not shrink like this, because the gradient is pulling the other way " +
      "the whole time. What the factor sets is how hard a weight has to be " +
      "pulled to stay where it is.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "Start from your Module 3 sgd_step and delete the line that measured the " +
      "gradient, since it now arrives as an argument. That leaves two list " +
      "comprehensions, one per parameter list. The bias one is unchanged. In " +
      "the weight one, compute the factor once before the comprehension, " +
      "give it a name, and multiply w by it where the old line just had w.",
    "The structure, in pseudocode:\n\n" +
      "    l2_step(weights, biases, nabla_w, nabla_b, eta, lmbda, n):\n" +
      "        decay = 1 - eta * lmbda / n\n" +
      "        new_weights = [decay * w - eta * nw\n" +
      "                       for w, nw in zip(weights, nabla_w)]\n" +
      "        new_biases  = [b - eta * nb\n" +
      "                       for b, nb in zip(biases, nabla_b)]\n" +
      "        return new_weights, new_biases\n\n" +
      "    decay is one number and w is an array, so decay * w scales every\n" +
      "    entry, exactly as eta * nw does.",
  ],
};
