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
    "Once both functions run, point fire at the three-fact concert neuron " +
      "from the top of the chapter: weather weighted 6, friend 2, transport " +
      "1, and the threshold of 5 written as the bias -5. Send this to the " +
      "scratch pad and run it:",
    {
      code:
        "w = np.array([[6.0], [2.0], [1.0]])   # weather, friend, transport\n" +
        "x = np.array([[1.0], [0.0], [0.0]])   # good weather, no friend, bad transport\n" +
        "\n" +
        "print((w * x).sum() - 5.0)            # the evidence z, the chapter's 6 - 5\n" +
        "print(fire(w, -5.0, x))               # the same neuron, squashed",
    },
    "The first print shows 1.0, the z the chapter computed by hand for good " +
      "weather alone. The second shows about 0.731: where the perceptron " +
      "gave a bare yes, the sigmoid neuron gives a confidence. Turn the " +
      "friend on (make x's second entry 1.0) and the evidence rises to 3, " +
      "the confidence to about 0.953.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "For sigmoid, translate the math directly: 1 / (1 + e^(-z)), using " +
      "np.exp for the exponential. NumPy applies the whole expression " +
      "elementwise when z is an array, so the same formula covers numbers " +
      "and arrays alike. For fire, the weighted sum w . x is: multiply w and x " +
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
