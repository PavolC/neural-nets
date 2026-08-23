import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const feedforwardExercise: Exercise = {
  id: "feedforward",
  title: "Feedforward",
  prompt: [
    "A network computes its output one layer at a time. The input x is the " +
      "activation of the first layer. Each later layer takes the previous " +
      "layer's activation a and produces a new one: sigmoid(w @ a + b), " +
      "where w is that layer's weight matrix and b its bias vector. The " +
      "activation of the last layer is the network's output.",
    "Your job: implement feedforward(weights, biases, x). The lists weights " +
      "and biases hold one entry per layer, in order from input to output. " +
      "Keep every activation a column vector of shape (n, 1); the tests " +
      "check shapes first because almost every bug here is a shape bug. " +
      "You wrote sigmoid in Module 1, so it is provided: from course import sigmoid.",
    "A satisfying way to play before (or after) the tests: rebuild Module 1's " +
      "XOR solution with your own function. Set " +
      "weights = [np.array([[6., 6.], [6., 6.]]), np.array([[8., -8.]])] and " +
      "biases = [np.array([[-3.], [-9.]]), np.array([[-4.]])], then " +
      "print(feedforward(weights, biases, np.array([[1.], [0.]]))) and use " +
      "Run my code. You should see roughly 0.96: the contrarian says go for " +
      "good weather alone, exactly as Module 1's table promised. Try all four " +
      "corners.",
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
