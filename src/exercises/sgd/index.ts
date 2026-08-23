import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const sgdExercise: Exercise = {
  id: "sgd",
  title: "Stochastic gradient descent",
  prompt: [
    "You are going to implement learning itself. The course hands you a " +
      "gradient function (course.numerical_gradient, computed by finite " +
      "differences: slow, but correct), and you write the two parts of " +
      "descent: sgd_step, which moves every parameter one step downhill on " +
      "one mini-batch, and sgd, the loop that visits the whole dataset in " +
      "random mini-batches, epoch after epoch.",
    "Two contracts matter. First, sgd_step must return new lists of new " +
      "arrays and leave its inputs untouched. Second, sgd must draw its " +
      "mini-batches in exactly the order given in the skeleton docstring " +
      "(one permutation per epoch, then consecutive slices), because the " +
      "tests reproduce that order to check your result to six decimal places.",
    "A satisfying experiment once sgd_step works: let it repair Module 1's " +
      "slider network from its starting position, the one the module scored " +
      "by hand at cost 0.0875. Append this below your code, then press " +
      "Run my code:",
    {
      code:
        "from course import feedforward\n" +
        "\n" +
        "weights = [np.array([[2., 2.], [2., 2.]]),   # the sliders' start position\n" +
        "           np.array([[4., -4.]])]\n" +
        "biases = [np.array([[-1.], [-3.]]),\n" +
        "          np.array([[-2.]])]\n" +
        "X = np.array([[0., 1., 0., 1.],              # the four corners, as columns\n" +
        "              [0., 0., 1., 1.]])\n" +
        "Y = np.array([[0., 1., 1., 0.]])             # the contrarian's answers\n" +
        "\n" +
        'print("cost before:", quadratic_loss(weights, biases, X, Y))\n' +
        "for _ in range(100):\n" +
        "    weights, biases = sgd_step(weights, biases, X, Y, 2.0)\n" +
        'print("cost after: ", quadratic_loss(weights, biases, X, Y))\n' +
        'print("the four answers:", feedforward(weights, biases, X))',
    },
    "The cost falls from 0.08758 to about 0.026, and the four answers lean " +
      "toward 0, 1, 1, 0. Your code just did what your hands did with the " +
      "sliders.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "For sgd_step: numerical_gradient wants a function of (weights, biases) " +
      "only, but quadratic_loss also needs X and Y. Wrap it: define an " +
      "inner function (or lambda) that closes over this batch's X and Y. " +
      "Then the update is one list comprehension per parameter list: " +
      "each new parameter is param - eta * gradient. For sgd: two nested " +
      "loops, epochs outside, batches inside, reassigning weights and " +
      "biases with each sgd_step so progress carries forward.",
    "The structure, in pseudocode:\n\n" +
      "    sgd_step(weights, biases, X, Y, eta):\n" +
      "        loss_fn = (ws, bs) -> quadratic_loss(ws, bs, X, Y)\n" +
      "        nabla_w, nabla_b = numerical_gradient(loss_fn, weights, biases)\n" +
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
