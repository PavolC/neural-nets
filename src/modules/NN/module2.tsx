import { AfterThis, Figure, Recap } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExercisePage } from "../../components/ExercisePage";
import { feedforwardExercise } from "../../exercises/feedforward";
import { NetworkDiagram } from "./interactives/NetworkDiagram";
import { PixelsInteractive } from "./interactives/PixelsInteractive";

export function Module2() {
  return (
    <article className="module">
      <h2>Module 2: Feedforward</h2>
      <AfterThis
        items={[
          "Turn a digit image into the column of 784 numbers a network actually sees.",
          "Compute a whole layer of neurons with one matrix multiplication.",
          "Run your own feedforward on real MNIST digits and watch it read them.",
        ]}
      />

      <p>
        Module 1 ended with three neurons wired into a network, and one phrase did
        the heavy lifting: <M tex="h_1" /> and <M tex="h_2" /> both read the same two
        inputs. Whenever several neurons sit side by side reading the same inputs,
        they form a layer, and this module is about computing a whole layer at once.
        Start small. Take the concert neuron, weights 6 and 2, and give it a
        colleague with weights 1 and 5. Feed both the input <M tex="x_1 = 1" />,{" "}
        <M tex="x_2 = 0" /> and do the multiply-and-add twice by hand: the first
        neuron's evidence is <M tex="6 \cdot 1 + 2 \cdot 0 = 6" />, the second's is{" "}
        <M tex="1 \cdot 1 + 5 \cdot 0 = 1" />. Nothing new happened; it happened
        twice. Math has a compact notation for exactly this: stack the two weight
        rows into a grid of numbers, called a matrix, and write the input as the
        column you already know:
      </p>
      <Eq
        tex="\begin{bmatrix} 6 & 2 \\ 1 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 6 \cdot 1 + 2 \cdot 0 \\ 1 \cdot 1 + 5 \cdot 0 \end{bmatrix} = \begin{bmatrix} 6 \\ 1 \end{bmatrix}"
        gloss="Each row of the matrix is one neuron's weights. The rule for matrix-times-column is: run each row's multiply-and-add against the column, and write the answers top to bottom. In NumPy this is the @ operator."
      />
      <p>
        Call the matrix <M tex="W" />, stack the biases into a column{" "}
        <M tex="b" />, and a whole layer of neurons becomes one line:
      </p>
      <Eq
        tex="a' = \sigma(W a + b)"
        gloss="The next layer's activations a' come from multiplying the weight matrix by the current activations a, adding the biases, and squashing every entry with sigmoid."
      />
      <p>
        Now apply that line layer after layer. The output column of one layer
        becomes the input column of the next; that is Module 1's "the output neuron
        only hears the reports," now with clean bookkeeping. Running the rule from
        the input column all the way to the output column is called feedforward, and
        it is the entire forward story of a neural network. The rest of this course
        is about choosing what goes inside <M tex="W" /> and <M tex="b" />.
      </p>

      <p>
        The shapes deserve one careful paragraph, because nearly every bug you will
        write in this course is a shape bug. Look back at the worked example: two
        neurons reading two inputs made a 2-by-2 matrix, one row per neuron, one
        column per input. In general a layer of <M tex="m" /> neurons reading{" "}
        <M tex="n" /> inputs has <M tex="W" /> of shape <M tex="(m, n)" />, and
        activations are always columns: shape <M tex="(n, 1)" />, never a flat{" "}
        <M tex="(n,)" />. Then <M tex="Wa" /> is <M tex="(m, 1)" />, the biases are{" "}
        <M tex="(m, 1)" />, and everything adds up cleanly. Why matrices instead of
        a Python loop over neurons? The loop runs in the interpreter, one neuron at
        a time; the matrix product hands the whole layer to fast numerical code in
        one call. The same discipline that keeps shapes honest makes the code
        hundreds of times faster.
      </p>

      <p>
        Time to leave four-dot toy worlds. The real task of this course is reading
        handwritten digits from MNIST, a classic collection of 70,000 of them
        (this course bundles a slice of it). And here is the bridge from fuzzy to
        numeric: a digit image IS already numbers. Each one is a 28-by-28 grid of
        ink levels, 0 for blank paper, 1 for full ink. Read the grid row by row into
        one long column and you get 784 numbers: that column is the network's input,
        exactly like <M tex="(x_1, x_2)" /> was, just taller. Point at the image
        below and see for yourself:
      </p>
      <Figure caption="An image is a grid of numbers, and the grid unrolls into the (784, 1) input column. There is no other magic between a picture and a network.">
        <PixelsInteractive />
      </Figure>

      <p>
        What about the output? Use ten output neurons, one per digit. Each one's
        activation is its confidence, exactly like the sigmoid outputs you watched
        in Module 1, and the network's answer is simply whichever of the ten is most
        confident. So the whole machine is: a 784-tall input column, a hidden layer
        (15 neurons here), and a 10-neuron output layer. Count the numbers the way
        we counted Module 1's nine: the hidden <M tex="W" /> is 15 rows of 784
        weights plus 15 biases, and the output layer is 10 rows of 15 plus 10
        biases. That is 11,935 numbers, and finding good values by hand-fiddling
        sliders is out of the question, which is why Module 3 exists.
      </p>

      <p>
        The network below has already been trained (86% test accuracy; you will
        beat it later). Its weights are not code; they are 11,935 learned numbers,
        and you can look at them. Each hidden neuron has 784 incoming weights, one
        per pixel, so reshape its row of <M tex="W" /> back into 28-by-28 and you
        get a picture of what that neuron looks for: red pixels push it up when
        inked, blue pull it down. Hover the hidden neurons and browse.
      </p>
      <Figure caption="A 784-15-10 network. Edge colors show the sign and strength of hidden-to-output weights. Hover a hidden neuron (middle column) to see its incoming weights as a 28x28 patch. The digit panel below the diagram unlocks when your feedforward passes the tests.">
        <NetworkDiagram />
      </Figure>

      <p>
        Now implement feedforward yourself. Your function receives the lists
        weights and biases (one entry per layer, input side first) and an input
        column, and returns the output column. When the tests pass, go back up to
        the diagram: the button there runs your code on real digits with the
        trained weights, and every activation it lights up will have been computed
        by the function you are about to write.
      </p>

      <ExercisePage exercise={feedforwardExercise} />

      <Recap
        items={[
          "A layer is several neurons reading the same inputs; stack their weight rows into W and the layer is one line: a' = sigmoid(Wa + b).",
          "W has one row per neuron and one column per input, shape (m, n); activations stay columns, and one layer's output column is the next layer's input.",
          "A digit image is already numbers: a 28x28 grid of ink levels, unrolled into a (784, 1) column; ten output confidences, and the answer is the largest.",
          "Your feedforward plus trained weights already reads digits; learning the 11,935 numbers yourself is Modules 3 to 5.",
        ]}
        chapter="Chapter 1 (the architecture of neural networks)"
        href="http://neuralnetworksanddeeplearning.com/chap1.html"
      />
    </article>
  );
}
