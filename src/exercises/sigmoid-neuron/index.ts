import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const sigmoidNeuronExercise: Exercise = {
  id: "sigmoid-neuron",
  title: "A sigmoid neuron",
  prompt: [
    "Time to build the two smallest pieces of the whole course. First " +
      "sigmoid(z), the activation function: it takes any number and squashes " +
      "it into the range 0 to 1, smoothly. You will use it in every network " +
      "from here on. Write it with np.exp so it works on whole arrays at " +
      "once, not just single numbers.",
    "Then fire(w, b, x): one neuron, exactly the concert decision from the " +
      "prose. It multiplies each input in x by the matching weight in w, " +
      "sums everything, adds the bias b, and passes the result through your " +
      "sigmoid. Both w and x are columns of numbers, shape (n, 1), like in " +
      "the NumPy notes above; the output is a plain float, so wrap it in " +
      "float(...). This exercise is small on purpose: it is also your first " +
      "trip through the editor, the tests, and the hints, so you know the " +
      "workflow before the real climbing starts.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "For sigmoid, translate the math directly: 1 / (1 + e^(-z)), using " +
      "np.exp for the exponential. NumPy applies the whole expression " +
      "elementwise when z is an array, so one line covers numbers and " +
      "arrays alike. For fire, the weighted sum w . x is: multiply w and x " +
      "elementwise, then sum the results. Add b, then apply your sigmoid. " +
      "The tests want a plain float back, not a 1x1 array.",
    "The structure, in pseudocode:\n\n" +
      "    sigmoid(z):\n" +
      "        return 1 / (1 + np.exp(-z))\n\n" +
      "    fire(w, b, x):\n" +
      "        z = sum of (w * x) + b      # (w * x).sum() + b\n" +
      "        return sigmoid(z) as a plain float   # wrap in float(...)",
  ],
};
