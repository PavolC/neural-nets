import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const feedforwardExercise: Exercise = {
  id: "feedforward",
  title: "Feedforward",
  prompt: [
    "One idea carries this whole exercise: the 2D matrix W is nothing new. " +
      "It is several neurons' weight lists, stacked, one row per neuron. " +
      "And computing a neuron is still what it always was: multiply each " +
      "input by its weight, add it all up. W @ x does exactly that for " +
      "every row at once. Run it:",
    {
      code:
        "x = np.array([[1.], [0.]])      # 2 inputs, as a column\n" +
        "W = np.array([[6., 2.],         # row 0: neuron A's two weights\n" +
        "              [1., 5.]])        # row 1: neuron B's two weights\n" +
        "b = np.array([[0.5], [-1.]])    # one bias per neuron\n" +
        "\n" +
        "print(W @ x)                    # each row multiplied-and-added against x\n" +
        "print(W @ x + b)                # -> a column: one number per neuron",
    },
    "So sigmoid(w @ a + b) computes one whole layer: column in, column out. " +
      "A network has several layers, which is why your two arguments are " +
      "lists: one weight matrix and one bias column per layer, in step " +
      "(weights[0] and biases[0] describe the same layer). feedforward is: " +
      "start with a = x; for each pair from zip(weights, biases), replace a " +
      "with sigmoid(w @ a + b); return the final a. Looping over the few " +
      "layers is fine; the loop the module banned was over the neurons " +
      "inside a layer, and @ already does those.",
    "Keep every activation a column, shape (n, 1); the tests check that " +
      "first. sigmoid is provided: from course import sigmoid.",
    "A satisfying way to play before (or after) the tests: rebuild Module 1's " +
      "XOR solution with your own function, and ask it about the corners. " +
      "Append this below your code, then press Run my code:",
    {
      code:
        "weights = [np.array([[6., 6.], [6., 6.]]),   # h1 and h2, stacked\n" +
        "           np.array([[8., -8.]])]            # the output neuron\n" +
        "biases = [np.array([[-3.], [-9.]]),\n" +
        "          np.array([[-4.]])]\n" +
        "\n" +
        "print(feedforward(weights, biases, np.array([[1.], [0.]])))",
    },
    "You should see roughly 0.96: the contrarian says go for good weather " +
      "alone, exactly as Module 1's table promised. Try all four corners.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "Think of the network as a pipeline. Start with a = x. Each layer " +
      "transforms a into sigmoid(w @ a + b) using its own w and b. After the " +
      "last layer, a is the answer. You never need to know how many layers " +
      "there are: just walk the two lists in step, weights[0] with biases[0], " +
      "weights[1] with biases[1], and so on.",
    "The structure, in pseudocode:\n\n" +
      "    a = x\n" +
      "    for each pair (w, b) taken from weights and biases in order:\n" +
      "        a = sigmoid(w @ a + b)\n" +
      "    return a\n\n" +
      "In Python, zip(weights, biases) walks the two lists in step. Use @ for " +
      "matrix multiplication, not *.",
  ],
};
