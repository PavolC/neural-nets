import type { Exercise } from "../types";
import skeleton from "./skeleton.py?raw";
import tests from "./tests.py?raw";
import solution from "./solution.py?raw";

export const backpropExercise: Exercise = {
  id: "backprop",
  title: "Backpropagation",
  prompt: [
    "Two functions to add to your file. sigmoid_prime is a single formula: " +
      "Module 4's steepness, sigmoid(z) times one minus sigmoid(z), " +
      "elementwise like sigmoid itself. backprop is the algorithm, and " +
      "every statement in it is one of the four equations above.",
    "The contract, precisely: backprop(weights, biases, x, y) handles ONE " +
      "example. x is a single input column, shape (n_in, 1); y is its " +
      "right answer as a one-hot column, shape (n_out, 1). It returns " +
      "(nabla_w, nabla_b), the same pair your sgd already takes: the " +
      "gradient, split into the weights' slopes and the biases' slopes. " +
      "They are lists of arrays shaped exactly like weights " +
      "and biases, each entry the exact slope of this example's cost " +
      "(one half times the sum of squared gaps, no averaging) for that " +
      "parameter. Read the parameters, never change them; moving them is " +
      "your sgd's job.",
    "The code has two halves. The forward half is your feedforward from " +
      "Module 2 with receipts: append each z to a list zs and each " +
      "activation to a list activations (started as [x]) as you go. The " +
      "backward half is BP1 once, then BP2, BP3, BP4 in a loop that " +
      "walks from the last layer toward the first, filling nabla_w and " +
      "nabla_b back to front.",
    "One new NumPy tool for the backward half: np.zeros_like(w) makes a " +
      "zeros array of w's shape. Build nabla_w and nabla_b that way " +
      "before the walk starts, then fill their slots as the loop goes, " +
      "with the negative indices the section above laid out.",
    "The tests climb a ladder: sigmoid_prime's values, your shapes, then " +
      "exact numbers on a small fixed network, then the gradient check, " +
      "the strictest test in this course. It builds a 3-5-4-2 network: " +
      "two hidden layers, so your backward loop has to actually loop, " +
      "with 54 knobs by the usual count (weights 15 + 20 + 8, biases " +
      "5 + 4 + 2). The check measures all 54 slopes the slow Module 3 " +
      "way, nudging every parameter both ways, and compares them to " +
      "yours one by one, demanding agreement to within one part in ten " +
      "million. Passing it means your backpropagation is correct. Last, " +
      "your slopes drive a descent that retrains Module 1's XOR network.",
    "Once the tests pass, point your code at Module 4's by-hand numbers, " +
      "the two-neuron chain. Send this to the scratch pad and run it:",
    {
      code:
        "from course import gradient\n" +
        "\n" +
        "weights = [np.array([[1.0]]), np.array([[2.0]])]   # w1, w2 from Module 4's chain\n" +
        "biases = [np.array([[-0.5]]), np.array([[-1.0]])]  # b1, b2\n" +
        "x = np.array([[1.0]])\n" +
        "y = np.array([[1.0]])\n" +
        "\n" +
        "nabla_w, nabla_b = backprop(weights, biases, x, y)\n" +
        'print("w1 slope:", nabla_w[0][0, 0], "  b1 slope:", nabla_b[0][0, 0])\n' +
        'print("w2 slope:", nabla_w[1][0, 0], "  b2 slope:", nabla_b[1][0, 0])\n' +
        "\n" +
        "nudge_w, nudge_b = gradient(weights, biases, x, y)  # Module 3's slow way\n" +
        'print("w1 slope, nudge-measured:", nudge_w[0][0, 0])',
    },
    "The four slopes should be Module 4's table exactly: -0.0508 for both " +
      "of neuron A's knobs, -0.0673 and -0.1081 for neuron B's. And the " +
      "two w1 printouts agree to about ten decimal places: the nudge method " +
      "approximates, from two reruns per knob, what your code now " +
      "computes exactly, all knobs in one sweep.",
  ],
  skeleton,
  tests,
  solution,
  flagship: {
    test: "test_gradient_check",
    note:
      "The gradient check passed: on a network your code has never seen, " +
      "all 54 of your slopes match nudge-and-measure to within one part " +
      "in ten million. This is the strongest guarantee the course can " +
      "give. Your backpropagation is correct.",
  },
  hints: [
    "Mirror the four equations in order. Forward: loop over zip(weights, " +
      "biases) exactly like feedforward, but keep two lists as you go " +
      "(zs, and activations starting as [x]). Backward: compute delta " +
      "for the output layer with BP1, store its two slope arrays (BP3: " +
      "nabla_b is delta itself; BP4: delta @ activations[-2].T), then " +
      "loop toward the front, each pass replacing delta via BP2 and " +
      "storing that layer's two arrays. For a two-layer network the loop " +
      "body runs once; write it so it would keep going on a deeper one " +
      "(the gradient check's network has three).",
    "The structure, in pseudocode:\n\n" +
      "    backprop(weights, biases, x, y):\n" +
      "        activations = [x];  zs = []\n" +
      "        a = x\n" +
      "        for each layer's (w, b):\n" +
      "            z = w @ a + b;  append z to zs\n" +
      "            a = sigmoid(z); append a to activations\n" +
      "        nabla_w = [np.zeros_like(w) for each w]\n" +
      "        nabla_b = [np.zeros_like(b) for each b]\n" +
      "        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])   # BP1\n" +
      "        nabla_b[-1] = delta                                     # BP3\n" +
      "        nabla_w[-1] = delta @ activations[-2].T                 # BP4\n" +
      "        for l = 2, 3, ... up to the number of layers:\n" +
      "            delta = (weights[-l+1].T @ delta) * sigmoid_prime(zs[-l])   # BP2\n" +
      "            nabla_b[-l] = delta\n" +
      "            nabla_w[-l] = delta @ activations[-l-1].T\n" +
      "        return nabla_w, nabla_b\n\n" +
      "    Reading the loop's indices: -l names the layer being blamed, so\n" +
      "    weights[-l+1], one step less negative, is the layer AFTER it, the\n" +
      "    wires the blame flows back through. At l = 2 that is weights[-1],\n" +
      "    the last weight matrix.\n\n" +
      "    One warning about that letter: this l is not the l of the four\n" +
      "    equations. Here it counts steps back from the end, so l = 2 is\n" +
      "    the second-to-last layer whatever the network's depth; there it\n" +
      "    numbers layers from the front. On a two-layer network the two\n" +
      "    happen to agree, which is exactly why a deeper one catches an\n" +
      "    off-by-one.",
  ],
};
