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
        A network is just that rule repeated. Each layer takes the previous layer's
        output column as its input, computes <M tex="\sigma(Wa + b)" /> with its own{" "}
        <M tex="W" /> and <M tex="b" />, and passes the result on. (Same idea as
        Module 1, where the output neuron read <M tex="h_1" />'s and{" "}
        <M tex="h_2" />'s reports instead of the raw inputs.) Running an input
        through every layer in order is called feedforward, and it is everything a
        network does when it answers. The rest of the course is about finding good
        numbers to put in the <M tex="W" />'s and <M tex="b" />'s.
      </p>

      <p>
        The shapes deserve one careful paragraph, because nearly every bug you will
        write in this course is a shape bug. NumPy describes every array by its
        shape, written as a pair: <M tex="(m, n)" /> means <M tex="m" /> rows and{" "}
        <M tex="n" /> columns. In the worked example, two neurons reading two
        inputs made <M tex="W" /> of shape <M tex="(2, 2)" />; in general a layer
        of <M tex="m" /> neurons reading <M tex="n" /> inputs has <M tex="W" /> of
        shape <M tex="(m, n)" />, and activations are always columns, shape{" "}
        <M tex="(n, 1)" />. NumPy also allows a third thing we deliberately avoid:
        np.array([1.0, 2.0, 3.0]) has the odd-looking shape (3,), trailing comma and
        all. That comma is real notation (Python's way of writing a one-item
        list-of-shapes), and it marks a flat array: three numbers in a bare line
        that is neither a row nor a column. Flat arrays make matrix arithmetic
        misbehave silently, so the course rule is simple: build columns, and if a
        shape ever prints with a trailing comma, treat it as a bug. With columns
        everywhere, <M tex="Wa" /> is <M tex="(m, 1)" />, the biases are{" "}
        <M tex="(m, 1)" />, and everything adds up cleanly. Here is the whole
        contract as a picture:
      </p>
      <Figure caption="How the shapes lock together. Teal is the input side: W is n wide because a is n tall, one weight per input, and the two must match or the multiply is impossible. Purple is the neuron side: W has m rows, so Wa, b, and a' are all m tall, one entry per neuron. The shaded strip shows one neuron's whole story: its row of W, times all of a, plus its own bias entry, becomes its entry of a'. For Module 2's hidden layer, m = 15 and n = 784.">
        <ShapesDiagram />
      </Figure>
      <p>
        Why matrices instead of a Python loop over neurons? The loop runs in the
        interpreter, one neuron at a time; the matrix product hands the whole layer
        to fast numerical code in one call. The same discipline that keeps shapes
        honest makes the code hundreds of times faster.
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

// Static diagram: the shapes in a' = sigma(Wa + b), drawn as proportioned
// rectangles. Teal marks the n dimension (inputs), purple the m dimension
// (neurons); one row of W is shaded through to the a' entry it produces.
function ShapesDiagram() {
  const ROW_TOP = 107; // the shaded "neuron j" band
  const ROW_H = 30;
  return (
    <svg viewBox="0 0 500 240" className="shapes-diagram" role="img"
         aria-label="The shapes in a' = sigma(W a + b): W is m by n, a is n by 1, b and a' are m by 1">
      {/* a' */}
      <rect x={20} y={77} width={28} height={90} className="shape-m" />
      <rect x={20} y={ROW_TOP} width={28} height={ROW_H} className="shape-row" />
      <text x={34} y={70} textAnchor="middle" className="shape-name">a′</text>
      <text x={34} y={215} textAnchor="middle" className="shape-dim dim-m">m tall</text>

      <text x={66} y={128} className="shape-op">=</text>
      <text x={92} y={128} className="shape-op">σ(</text>

      {/* W */}
      <rect x={150} y={77} width={170} height={90} className="shape-w" />
      {[1, 2].map((k) => (
        <line key={`h${k}`} x1={150} x2={320} y1={77 + k * 30} y2={77 + k * 30} className="shapes-grid" />
      ))}
      {[1, 2, 3].map((k) => (
        <line key={`v${k}`} x1={150 + k * 42.5} x2={150 + k * 42.5} y1={77} y2={167} className="shapes-grid" />
      ))}
      <rect x={150} y={ROW_TOP} width={170} height={ROW_H} className="shape-row" />
      <text x={235} y={70} textAnchor="middle" className="shape-name">W</text>
      <text x={144} y={128} textAnchor="end" className="shape-dim dim-m">m</text>
      <text x={235} y={215} textAnchor="middle" className="shape-dim dim-n">n wide</text>

      {/* a */}
      <rect x={336} y={44} width={28} height={156} className="shape-n" />
      <text x={350} y={36} textAnchor="middle" className="shape-name">a</text>
      <text x={350} y={215} textAnchor="middle" className="shape-dim dim-n">n tall</text>

      <text x={386} y={128} className="shape-op">+</text>

      {/* b */}
      <rect x={406} y={77} width={28} height={90} className="shape-m" />
      <rect x={406} y={ROW_TOP} width={28} height={ROW_H} className="shape-row" />
      <text x={420} y={70} textAnchor="middle" className="shape-name">b</text>
      <text x={420} y={215} textAnchor="middle" className="shape-dim dim-m">m tall</text>

      <text x={448} y={128} className="shape-op">)</text>
    </svg>
  );
}
