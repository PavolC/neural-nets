import { AfterThis, Figure, Recap } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExercisePage } from "../../components/ExercisePage";
import { sigmoidNeuronExercise } from "../../exercises/sigmoid-neuron";
import { SeparatingLine } from "./interactives/SeparatingLine";
import { XorNetwork } from "./interactives/XorNetwork";

export function Module1() {
  return (
    <article className="module">
      <h2>Module 1: From neurons to networks</h2>
      <AfterThis
        items={[
          "Read a neuron as a weighted vote: what its weights and bias each control.",
          "Explain, with a picture, why a single neuron cannot solve XOR and a small network can.",
          "Implement sigmoid and a single neuron in NumPy, and pass your first tests.",
        ]}
      />

      <p>
        A neuron is a tiny decision. It takes some numbers in (call them{" "}
        <M tex="x_1, x_2, \dots" />), weighs each one by how much it should matter (the
        weights <M tex="w_1, w_2, \dots" />), adds them up, and adds a personal offset
        called the bias <M tex="b" />. That sum is the neuron's evidence:
      </p>
      <Eq
        tex="z = w \cdot x + b"
        gloss="The evidence z is each input times its weight, summed, plus the bias; the bias sets how much evidence the neuron demands before it leans yes."
      />
      <p>
        Positive evidence means the neuron leans yes, negative means no. The earliest
        artificial neuron, the perceptron, made this a hard rule: output 1 if{" "}
        <M tex="z > 0" />, else 0. Everything the neuron believes is packed into those
        few numbers. Learning, which is where this course is headed, will mean nothing
        more than adjusting them.
      </p>

      <p>
        Geometrically, one neuron is one straight line: the set of points where{" "}
        <M tex="z = 0" /> splits the plane in two, yes on one side, no on the other.
        That is a real limitation, and you should feel it yourself. Below are four
        points. In the OR and AND datasets one class can be separated from the other by
        a single line. Then switch to XOR (points agree: class 0, points differ: class
        1) and try to do the same.
      </p>
      <Figure caption="Drag the two round handles to move the line. OR and AND fall quickly. XOR never does: no single straight cut puts both green points on one side and both gold points on the other.">
        <SeparatingLine />
      </Figure>

      <p>
        The fix is not a cleverer line; it is more neurons. But before stacking them,
        one change to the neuron itself. A perceptron's hard threshold is brittle: a
        tiny nudge to a weight can flip its output from 0 to 1 with nothing in between,
        which makes gradual learning impossible. So we swap the cliff for a slope, the
        sigmoid function:
      </p>
      <Eq
        tex="\sigma(z) = \frac{1}{1 + e^{-z}}"
        gloss="Sigmoid squashes any evidence z into a number strictly between 0 and 1, smoothly: large positive z gives almost 1, large negative gives almost 0, and z = 0 gives exactly one half."
      />
      <p>
        A sigmoid neuron outputs <M tex="\sigma(w \cdot x + b)" />. It still draws the
        same line, but near the line it hedges, and that hedging is what lets small
        weight changes produce small output changes. Hold onto that: it is the whole
        reason networks can learn by nudging.
      </p>

      <p>
        Now the stacking. Give two neurons the same two inputs: each draws its own
        line. Feed both of their outputs into a third neuron, and that third neuron no
        longer sees raw inputs; it sees which side of each line the input is on. Its
        own straight cut, made in that new space, can look bent in the original one.
        Below is exactly that: a 2-2-1 network with every weight on a slider. Try to
        make it solve XOR before you press the solution button.
      </p>
      <Figure caption="The two thin lines belong to the hidden neurons; the shading is the output neuron's verdict (green means class 1). One classic solution: one hidden neuron computes roughly OR, the other roughly AND, and the output says OR but not AND.">
        <XorNetwork />
      </Figure>

      <p>
        That is a neural network: layers of neurons, each layer reading the previous
        one's outputs. The input layer is just your data. Layers between input and
        output are called hidden layers, and the pattern you just built by hand, carve
        with simple features then combine them, is what deep networks do at scale.
        Enough looking; time to build the neuron for real. The tests below are your
        first trip through the workflow you will use for the whole course.
      </p>

      <ExercisePage exercise={sigmoidNeuronExercise} />

      <Recap
        items={[
          "A neuron computes sigmoid(w . x + b): weighted evidence, squashed smoothly into (0, 1).",
          "One neuron is one straight cut, so XOR is out of reach for any single neuron.",
          "Hidden layers re-describe the input; a straight cut in that new description can bend in the original space.",
        ]}
        chapter="Chapter 1 (perceptrons and sigmoid neurons)"
        href="http://neuralnetworksanddeeplearning.com/chap1.html"
      />
    </article>
  );
}
