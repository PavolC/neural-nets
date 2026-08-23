import { AfterThis, Figure, Recap } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExercisePage } from "../../components/ExercisePage";
import { feedforwardExercise } from "../../exercises/feedforward";
import { NetworkDiagram } from "./interactives/NetworkDiagram";

export function Module2() {
  return (
    <article className="module">
      <h2>Module 2: Feedforward</h2>
      <AfterThis
        items={[
          "Compute a whole layer of neurons with one matrix multiplication.",
          "Keep shapes straight: the discipline that prevents most NumPy bugs.",
          "Run your own feedforward on real MNIST digits and watch it read them.",
        ]}
      />

      <p>
        Module 1 computed one neuron at a time. A layer of them, all reading the same
        inputs, is one matrix multiplication. Stack each neuron's weights as a row of a
        matrix <M tex="W" />, stack the biases into a vector <M tex="b" />, and the
        whole layer is:
      </p>
      <Eq
        tex="a' = \sigma(W a + b)"
        gloss="The next layer's activations a' come from multiplying the weight matrix by the current activations a, adding the biases, and squashing every entry with sigmoid."
      />
      <p>
        Applying that rule layer after layer, input to output, is called feedforward.
        That is the entire forward story of a neural network: the rest of this course
        is about choosing <M tex="W" /> and <M tex="b" />.
      </p>

      <p>
        The shapes deserve one careful paragraph, because nearly every bug you will
        write in this course is a shape bug. Activations are column vectors: an MNIST
        image is <M tex="(784, 1)" />, never <M tex="(784,)" />. The weight matrix into
        a layer of <M tex="m" /> neurons from a layer of <M tex="n" /> has shape{" "}
        <M tex="(m, n)" />: one row per receiving neuron, one column per input. Then{" "}
        <M tex="W a" /> is <M tex="(m, 1)" />, the biases are <M tex="(m, 1)" />, and
        everything adds up cleanly. Why matrices instead of loops? Python loops run in
        the interpreter, one neuron at a time; a matrix multiplication hands the whole
        layer to optimized numerical code at once. The same discipline that keeps
        shapes honest also makes the code hundreds of times faster.
      </p>

      <p>
        Here is what you are about to power. This is a real trained network, 784 inputs,
        15 hidden neurons, 10 outputs, trained beforehand to read digits (86% test
        accuracy; you will beat it later). Its weights are not code, they are learned
        pictures: hover the hidden neurons and look at what each one scans for.
      </p>
      <Figure caption="A 784-15-10 network. Edge colors show the sign and strength of hidden-to-output weights. Hover a hidden neuron to see its incoming weights as a 28x28 patch. The digit panel unlocks when your feedforward passes the tests below.">
        <NetworkDiagram />
      </Figure>

      <p>
        Now implement feedforward yourself. Your function receives the lists weights
        and biases (one entry per layer, input side first) and an input column vector,
        and returns the output activations. When the tests pass, go back up to the
        diagram: the button there runs your code, on real digits, with the trained
        weights. Every activation it lights up will have been computed by the function
        you are about to write.
      </p>

      <ExercisePage exercise={feedforwardExercise} />

      <Recap
        items={[
          "One layer is a' = sigmoid(Wa + b); feedforward is that rule applied layer by layer.",
          "Column vectors and (next layer, this layer) weight shapes keep every product well formed.",
          "Your feedforward plus someone else's trained weights already reads digits; learning those weights yourself is the next three modules.",
        ]}
        chapter="Chapter 1 (the architecture of neural networks)"
        href="http://neuralnetworksanddeeplearning.com/chap1.html"
      />
    </article>
  );
}
