import { AfterThis, Aside, Figure, ModuleToc, Recap, SectionHeader, fig } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExerciseCard } from "../../components/ExerciseCard";
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
      <ModuleToc />

      <p>
        Your code so far computes one neuron: a column of inputs in, one confidence
        out. Reading a handwritten digit needs many neurons side by side, and a
        second layer behind them reading their answers.
      </p>

      <SectionHeader id="m2-layer" title="A layer is one formula" />
      <p>
        Module 1 ended with three neurons wired into a network, and one phrase did
        the heavy lifting: <M tex="h_1" /> and <M tex="h_2" /> both read the same two
        inputs. Whenever several neurons sit side by side reading the same inputs,
        they form a layer, and this module is about computing a whole layer at once.
        Start small. Take the concert neuron, weights 6 and 2, and give it a
        colleague with weights 1 and 5. Feed both the input <M tex="x_1 = 1" />,{" "}
        <M tex="x_2 = 0" /> and do the multiply-and-add twice by hand, biases set
        aside for a moment: the first neuron's total is{" "}
        <M tex="6 \cdot 1 + 2 \cdot 0 = 6" />, the second's is{" "}
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
        <M tex="b" />, and a whole layer of neurons becomes one formula:
      </p>
      <Eq
        tex="a' = \sigma(W a + b)"
        gloss="The next layer's activations, written a' and read a-prime, come from multiplying the weight matrix by the current activations a, adding the biases, and squashing every entry with sigmoid."
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

      <SectionHeader id="m2-shapes" title="Shape discipline" />
      <p>
        The shapes deserve slow reading, because nearly every bug you will
        write in this course is a shape bug. NumPy describes every array by its
        shape, written as a pair: <M tex="(m, n)" /> means <M tex="m" /> rows and{" "}
        <M tex="n" /> columns. In the worked example, two neurons reading two
        inputs made <M tex="W" /> of shape <M tex="(2, 2)" />; in general a layer
        of <M tex="m" /> neurons reading <M tex="n" /> inputs has <M tex="W" /> of
        shape <M tex="(m, n)" />, and activations are always columns, shape{" "}
        <M tex="(n, 1)" />, never flat: Module 1's trailing-comma rule stands, and
        it matters more here, because a flat array in a matrix product misbehaves
        silently instead of failing loudly.
      </p>
      <p>
        One warning on top of that, because everyone trips here once: the shape
        names the receiving layer first. Data may flow from 2 inputs to 1 neuron,
        but that layer's matrix is <M tex="(1, 2)" />: rows first, and the rows
        belong to the layer being computed. Said with Module 1's anatomy,{" "}
        <M tex="W" /> is the layer's wire ledger: the entry in row j, column k is
        the weight on the wire from input k to neuron j, so a row collects one
        neuron's incoming wires, and a column collects one input's outgoing
        wires. The payoff is that multiplication just
        works, by the rule that the inner numbers must touch: <M tex="(1, 2)" />{" "}
        times <M tex="(2, 1)" /> works because 2 meets 2, and the outer numbers,{" "}
        <M tex="(1, 1)" />, are the answer's shape. With columns everywhere,{" "}
        <M tex="Wa" /> is <M tex="(m, 1)" />, the biases are <M tex="(m, 1)" />,
        and everything adds up cleanly. A bias column added to several columns
        side by side lands on each of them, and a plain number added to an array
        lands on every entry: NumPy stretches the smaller shape to fit, and calls
        it broadcasting. Here is the whole contract as a picture:
      </p>
      <Figure caption="How the shapes lock together. Teal is the input side: W is n wide because a is n tall, one weight per input, and the two must match or the multiply is impossible. Purple is the neuron side: W has m rows, so Wa, b, and a' are all m tall, one entry per neuron. The shaded strip shows one neuron's whole story: its row of W, times all of a, plus its own bias entry, becomes its entry of a'. For Module 2's hidden layer, m = 15 and n = 784.">
        <ShapesDiagram />
      </Figure>
      <Aside>
        <p>
          Why matrices instead of a Python loop over neurons? The loop runs in the
          interpreter, one neuron at a time; the matrix product hands the whole
          layer to fast numerical code in one call. The same discipline that keeps
          shapes honest makes the code hundreds of times faster. To be precise
          about what got banned: looping over the hundreds of neurons inside a
          layer. Looping over the two or three layers of a network is normal, and
          you are about to write exactly that loop.
        </p>
      </Aside>

      <SectionHeader id="m2-digits" title="A digit is 784 numbers" />
      <p>
        Time to leave four-dot toy worlds. The real task of this course is reading
        handwritten digits from MNIST, a classic collection of 70,000 of them
        (this course bundles a slice of it). And here is the bridge from fuzzy to
        numeric: a digit image is already numbers. Each one is a 28-by-28 grid of
        ink levels, 0 for blank paper, 1 for full ink. Read the grid row by row into
        one long column and you get 784 numbers: that column is the network's input,
        exactly like <M tex="(x_1, x_2)" /> was, just taller. Point at the image
        below and see for yourself:
      </p>
      <Figure caption="An image is a grid of numbers, and the grid unrolls into the (784, 1) input column. There is no other magic between a picture and a network.">
        <PixelsInteractive />
      </Figure>

      <SectionHeader id="m2-design" title="The 784-15-10 design" />
      <p>
        What about the other end? Give the network ten output neurons, one per
        digit, each answering its own yes-or-no question: is this a 0? is this a 1?
        and so on up to 9. Each answer is an ordinary sigmoid confidence between 0
        and 1, exactly like the outputs you watched in Module 1, and the network's
        final verdict is simply the question that got the most confident yes.
      </p>

      <p>
        Between the 784 inputs and the 10 outputs sits one hidden layer, and its
        size is a genuine choice, the first you get to make as a network designer.
        Nothing about the problem dictates it: more hidden neurons means more
        little pattern-detectors, which usually means more accuracy and always
        means slower training; fewer means the opposite. The network on this page
        uses 15 because 15 circles fit in a diagram and train in seconds (the
        classic setup you will train in Module 5 uses 30). So how many numbers do
        15 hidden neurons leave us to choose? Count layer by layer, the way we
        counted Module 1's nine:
      </p>
      <Eq
        tex="\begin{aligned} &\underbrace{15 \times 784}_{\text{hidden } W} + \underbrace{15}_{\text{hidden } b} + \underbrace{10 \times 15}_{\text{output } W} + \underbrace{10}_{\text{output } b} \\[0.8em] &= 11{,}760 + 15 + 150 + 10 = 11{,}935 \end{aligned}"
        gloss="The hidden layer's W has one row per neuron (15) and one column per input (784), plus one bias per neuron; the output layer reads the 15 hidden activations the same way. Nearly twelve thousand knobs, far too many to set by hand; Module 3 is about finding them automatically."
      />

      <p>
        And where did Module 1's geometry go? Nowhere: it scaled. A point in a
        784-axis space sounds exotic, but it is nothing deeper than the column you
        unrolled above: a list of 784 ink levels in order. So every image is one
        point in that space, one axis per pixel (move along an axis and one pixel
        brightens). The training images are dots scattered through that space, each
        labeled with the digit it shows. Module 1's four concert dots were the same
        thing in two axes, green for the go answers and gold for the stay answers. Each hidden neuron still makes one
        straight cut through the space, yes on one side and no on the other. Aiming
        that cut now takes 784 weights instead of two, one weight for every pixel
        in the grid.
      </p>
      <p>
        The hidden layer's 15 reports are themselves a list of numbers, one per
        hidden neuron, exactly like the 784 pixels. Read that list the way you read
        the pixels and each image becomes one point in a space of 15 axes instead
        of 784. Module 1's two hidden neurons did this on a smaller scale. Their
        pair of reports became the axes of a second plot, and on it the two go dots
        ended up almost on top of each other. Trading each image's 784 numbers for
        its 15 reports deserves a name, re-description, because every network does
        it.
      </p>
      <p>
        The ten output neurons never see pixels at all. They make their ten cuts in
        that 15-axis space, on the re-described points, where training will have
        arranged for each digit class to be separable from the rest.
      </p>

      <SectionHeader id="m2-weights" title="What learned weights look like" />
      <p>
        The network below has already been trained for you (86% test accuracy; the
        network you train in Module 5 does better), so you can study what learned
        numbers actually look
        like. Focus on a single hidden neuron. It works like every neuron so far:
        784 incoming wires, one per pixel, each carrying its weight, and its
        evidence rises when ink lands on pixels
        where its weight is positive, falls when ink lands where its weight is
        negative. (You have met a negative weight before: the output neuron's{" "}
        <M tex="-8" /> in the XOR network, the veto.) Now use the fact that each of
        the 784 weights belongs to one specific pixel: draw the weights themselves
        as a 28-by-28 image, each weight on its own pixel's square, red where it is
        positive (ink here excites the neuron), blue where it is negative (ink here
        suppresses it), pale where it is near zero (this neuron ignores that
        pixel). The result is literally a picture of what the neuron looks for.
        Hover the hidden neurons below and browse: you will see strokes, arcs, and
        blobs, because those are the shapes that tell digits apart.
      </p>
      <Figure caption="A 784-15-10 network. Edge colors show the sign and strength of hidden-to-output weights. Hover a hidden neuron (middle column) to see its incoming weights as a 28x28 patch. The digit panel below the diagram unlocks when your feedforward passes the tests.">
        <NetworkDiagram />
      </Figure>

      <SectionHeader id="m2-code" title="Implement feedforward" />
      <p>
        Now implement feedforward yourself; the exercise below spells out the plan
        and the two NumPy tools it needs. It goes into the same file as Module 1's
        work, under its own section line, so the sigmoid you call is the one you
        wrote a module ago rather than a copy of it. When the tests pass, come back
        up to the diagram: the button there runs your code on real digits with the
        trained weights, and every activation it lights up will have been computed
        by your function.
      </p>

      <ExerciseCard exercise={feedforwardExercise} />

      <Recap
        items={[
          "A layer is several neurons reading the same inputs; stack their weight rows into W and the layer is one formula: a' = sigmoid(Wa + b).",
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
    <svg {...fig(8, 13, 455, 213)} className="shapes-diagram" role="img"
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
