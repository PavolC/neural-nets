import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const smartInitExercise: Exercise = {
  id: "smart-init",
  title: "A better starting point",
  prompt: [
    "One function, and it builds a network rather than running one. Until " +
      "now the starting parameters arrived from outside: Module 1's sliders, " +
      "Module 2's file, Module 5's panel drawing standard normal numbers. " +
      "This is that last step, written out and scaled.",
    "The contract: init_network(sizes, rng) takes a list of layer sizes " +
      "front to back, so [784, 30, 10] is the digit reader, and a NumPy " +
      "generator to draw from. It returns (weights, biases), the pair every " +
      "function in this course has taken. weights[l] has shape " +
      "(sizes[l+1], sizes[l]), receiving layer first, and its entries are " +
      "standard normal draws divided by the square root of sizes[l], the " +
      "number of inputs feeding that layer. biases[l] has shape " +
      "(sizes[l+1], 1) and keeps the standard draw unscaled, because a bias " +
      "reads no inputs and so has nothing to divide by.",
    "Draw order is part of the contract, as it was for your sgd's shuffle: " +
      "every weight matrix first, front to back, then every bias column, " +
      "front to back. That is the order Module 5's panel used, so with the " +
      "same seed your network gets the same random numbers it did, each one " +
      "divided by its layer's square root. The comparison below is exactly " +
      "that: identical draws, one set scaled.",
    "Two NumPy tools: rng.standard_normal(shape) draws an array of that " +
      "shape from the standard bell, and np.sqrt(n) is the square root of n. " +
      "Take every number from the rng you are handed, never from " +
      "np.random.standard_normal, which reads a generator no seed of yours " +
      "controls.",
    "Once the tests pass, measure the thing this exercise is for: how far " +
      "from zero the hidden layer's evidence lands under each start. The " +
      "inputs here are stand-in digits, 103 pixels lit and the rest dark, " +
      "103 being the bundled images' average. Send this to the scratch pad " +
      "and press Run my code:",
    {
      code:
        "pixels = np.random.default_rng(0)\n" +
        "X = np.zeros((784, 200))                       # 200 stand-in digits\n" +
        "for k in range(200):\n" +
        "    X[pixels.choice(784, size=103, replace=False), k] = 1.0\n" +
        "\n" +
        "plain = np.random.default_rng(8)               # Module 5's start\n" +
        "plain_w = [plain.standard_normal((30, 784)), plain.standard_normal((10, 30))]\n" +
        "plain_b = [plain.standard_normal((30, 1)), plain.standard_normal((10, 1))]\n" +
        "\n" +
        "yours_w, yours_b = init_network([784, 30, 10], np.random.default_rng(8))\n" +
        "\n" +
        "for name, w, b in ((\"Module 5's start\", plain_w, plain_b), (\"your start\", yours_w, yours_b)):\n" +
        "    z = w[0] @ X + b[0]                        # every hidden neuron, every image\n" +
        "    steep = sigmoid(z) * (1 - sigmoid(z))\n" +
        '    print(name, "-> typical distance from zero:", round(float(np.abs(z).mean()), 2),\n' +
        '          " median steepness:", round(float(np.median(steep)), 4),\n' +
        '          " share flatter than 0.01:", round(float((steep < 0.01).mean()), 3))',
    },
    "Module 5's start reports a typical distance of 8.27 and a median " +
      "steepness of 0.0009, with 66 percent of the readings flatter than " +
      "0.01. Yours reports 0.79 and 0.2206, with none at all below 0.01. " +
      "Same random numbers, same images; the only change is the division. " +
      "For the real bundled digits the two distances are 7.43 and 0.78: a " +
      "little smaller, because a real digit's grey edge pixels sit below 1.",
  ],
  skeleton,
  tests,
  solution,
  hints: [
    "One list comprehension per list. For the weights, the entries of sizes " +
      "come in pairs: sizes[i] inputs feeding sizes[i+1] neurons, for i " +
      "running over range(len(sizes) - 1). Build the matrix with " +
      "rng.standard_normal((sizes[i+1], sizes[i])) and divide the whole " +
      "array by np.sqrt(sizes[i]): dividing an array by a number divides " +
      "every entry, so no loop over entries is needed. The biases are the " +
      "same walk without the division, and they must all be drawn after " +
      "all the weights, so build the weights list first.",
    "The structure, in pseudocode:\n\n" +
      "    init_network(sizes, rng):\n" +
      "        pairs = range(len(sizes) - 1)\n" +
      "        weights = [ rng.standard_normal((sizes[i+1], sizes[i]))\n" +
      "                    / np.sqrt(sizes[i])           for i in pairs ]\n" +
      "        biases  = [ rng.standard_normal((sizes[i+1], 1))\n" +
      "                                                  for i in pairs ]\n" +
      "        return weights, biases\n\n" +
      "    Two things the tests check that are easy to get backwards: the\n" +
      "    square root uses sizes[i], the SENDING layer's size (its inputs\n" +
      "    are what pile up), and the two comprehensions must run in this\n" +
      "    order, because they draw from the same stream.",
  ],
};
