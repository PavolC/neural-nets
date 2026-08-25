import { AfterThis, Figure, ModuleToc, Recap, SectionHeader } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExercisePage } from "../../components/ExercisePage";
import { backpropExercise } from "../../exercises/backprop";
import { BackpropTrainPanel } from "./interactives/BackpropTrainPanel";

export function Module5() {
  return (
    <article className="module">
      <h2>Module 5: Backpropagation, for real</h2>
      <AfterThis
        items={[
          "Implement backprop in NumPy: one forward pass that keeps receipts, one backward sweep, every slope exact.",
          "Pass the gradient check: your slopes against nudge-and-measure, agreeing parameter by parameter.",
          "Train the 784-30-10 digit reader end to end with your own code, and price what the old way would have cost.",
        ]}
      />
      <ModuleToc />

      <SectionHeader id="m5-plan" title="The deliverable" />
      <p>
        Module 4 worked out how to get every slope from one forward pass and one
        backward sweep of blame, exact instead of estimated, and wrote the method
        down as four equations. In this module you build it. Most of the machinery
        already exists: your feedforward computes the forward pass (Module 2),
        and your sgd walks downhill along whatever slopes it is handed
        (Module 3). The one slow part left is where those slopes come from: the
        course's gradient function, which still measures each one by nudging its
        parameter twice and rescoring the batch. The deliverable here is a
        replacement for exactly that part: a function named{" "}
        <code>backprop</code>, small enough to fit on one screen. When it is
        written, the nudge method takes on the referee
        job Module 4 described: the tests measure every slope of a small fixed
        network the slow way and demand your answers match, number by number.
        And once they do, the panel at the bottom of this page plugs your
        backprop into your own sgd, unchanged, and trains the digit reader for
        real.
      </p>

      <SectionHeader id="m5-cost" title="One example at a time" />
      <p>
        One design decision before the code, and what it buys: backprop handles
        a single training example per call, and mini-batches then come free.
        For one example the cost keeps its shape from Module 3 but drops the
        averaging:
      </p>
      <Eq
        tex="C = \tfrac12 \, \lVert a - y \rVert^2"
        gloss="The double bars as in Module 3: take the gap in every output entry, square each, add them up. Nothing divides by a count, because this is one example's score. The half is the same bookkeeping half, there so the slope comes out as the bare gap."
      />
      <p>
        With a private cost per example comes a private slope per example, and
        a batch's slope is just the average of them, because averages pass
        changes through: if each example's cost changes by some amount when a
        knob moves, the average cost changes by the average of those amounts.
        Numbers you already have can check that. Take Module 3's starting
        network and the knob it worked by hand, the output bias, and
        nudge-and-measure that bias four times, each time scoring a single
        corner alone:
      </p>
      <Eq
        tex="\frac{\underbrace{0.0460}_{(0,0)\text{, stay}} + \underbrace{(-0.1337)}_{(1,0)\text{, go}} + \underbrace{(-0.1337)}_{(0,1)\text{, go}} + \underbrace{0.0460}_{(1,1)\text{, stay}}}{4} = -0.04385"
        gloss="Four private slopes, one per corner. The two stay corners pull the bias up, the two go corners pull it down harder, and the average of the four is the -0.044 Module 3 measured on the full four-corner cost, one more digit shown."
      />
      <p>
        So a mini-batch's gradient is: run backprop on each example separately,
        average the returned slopes. Your sgd never notices the difference; it
        just receives slopes, as always. (The opposite pulls above are also
        Module 3's wobble seen up close: each mini-batch's average is a
        different compromise between examples like these.)
      </p>
      <p>
        For a digit image, the right answer <M tex="y" /> is a column of ten
        numbers, 1.0 in the digit's slot and 0.0 everywhere else (the packing is
        called one-hot). Each entry grades one output neuron's yes-or-no question
        from Module 2, so the ten gaps in the cost are the ten questions'
        misses.
      </p>

      <SectionHeader id="m5-lines" title="The equations as code" />
      <p>
        Here are the four equations again, in the order the code meets them:
      </p>
      <Eq
        tex="\begin{aligned} \delta^L &= (a^L - y) \odot \sigma'(z^L) && \text{(BP1)} \\ \delta^l &= \big( (w^{l+1})^T \, \delta^{l+1} \big) \odot \sigma'(z^l) && \text{(BP2)} \\ \frac{\partial C}{\partial b^l} &= \delta^l && \text{(BP3)} \\ \frac{\partial C}{\partial w^l} &= \delta^l \, (a^{l-1})^T && \text{(BP4)} \end{aligned}"
        gloss="Module 4's equations, with one upgrade: the layer letter l stands for each layer in turn (2 then 3 on this page; the tests also try a deeper network), which is exactly the loop you are about to write, and L names the last layer. Blame starts at the output as gap times steepness, flows backward through the transposed wires, and every slope reads off a blame; if any of the four feels foreign, the Module 4 stepper is the place to rebuild it."
      />
      <p>
        The translation to NumPy is nearly mechanical, because Module 4 already
        glossed every symbol: the circled dot is the elementwise <code>*</code>,
        the raised T is <code>.T</code>, and matrix products are <code>@</code>.
        Each equation becomes one NumPy statement. The shapes column below runs
        each statement on the digit reader's hidden layer, where Module 2's
        inner-numbers-touch check confirms every product:
      </p>
      <table className="truth-table">
        <thead>
          <tr>
            <th>equation</th>
            <th>in NumPy</th>
            <th>shapes on the digit reader</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>BP1</td>
            <td><code>delta = (a - y) * sigmoid_prime(z)</code></td>
            <td>(10, 1), one blame per output neuron</td>
          </tr>
          <tr>
            <td>BP2</td>
            <td><code>delta = (w_next.T @ delta) * sigmoid_prime(z)</code></td>
            <td>(30, 10) @ (10, 1) gives (30, 1)</td>
          </tr>
          <tr>
            <td>BP3</td>
            <td><code>nabla_b = delta</code></td>
            <td>(30, 1), exactly the biases' shape</td>
          </tr>
          <tr>
            <td>BP4</td>
            <td><code>nabla_w = delta @ a_prev.T</code></td>
            <td>(30, 1) @ (1, 784) gives (30, 784)</td>
          </tr>
        </tbody>
      </table>

      <SectionHeader id="m5-receipts" title="Keep receipts" />
      <p>
        The one genuinely new implementation idea is bookkeeping: keep receipts.
        BP1 needs the output layer's <M tex="z" />, BP2 needs every earlier{" "}
        <M tex="z" />, and BP4 needs every layer's incoming activations. All of
        them get computed on the forward trip anyway, so the forward half of
        backprop is your feedforward with two accumulator lists: every{" "}
        <M tex="z" /> appended to <code>zs</code>, every activation appended to{" "}
        <code>activations</code> (which starts as <code>[x]</code>, since the
        input doubles as layer 1's activations). The backward sweep then never
        recomputes anything, which is the entire point: Module 4 put it as
        arranging the multiplications so no factor is computed twice.
      </p>
      <p>
        The natural way to read those lists is from the end, and Python has
        notation for that worth learning here: a negative index counts from the
        back. <code>activations[-1]</code> is the last entry (the network's
        output), <code>activations[-2]</code> the one before it,{" "}
        <code>zs[-1]</code> the last stored <M tex="z" />. Assignment works the
        same way, so <code>nabla_b[-1] = delta</code> fills the last layer's
        slot. The figure shows both counting directions on a two-layer network's
        receipts:
      </p>
      <Figure caption="What the forward pass stores, for a network of two layers (like the 2-3-1 fixture in the tests, or the digit reader). Each box shows its entry's positive index on top and its negative index below: both name the same slot. BP1 reads zs[-1]; BP4 at the output layer reads activations[-2]; the backward loop steps both pointers left together.">
        <ReceiptsDiagram />
      </Figure>

      <SectionHeader id="m5-exercise" title="Write backprop" />
      <ExercisePage exercise={backpropExercise} />

      <SectionHeader id="m5-train" title="The real training run" />
      <p>
        The network for the real run is 784-30-10. Module 2's diagram used 15
        hidden neurons so the circles fit on screen, and named 30 as the
        classic training setup; thirty detectors instead of fifteen is the same
        free choice Module 2 called out, more little pattern-detectors against
        more parameters to train. The counting rule from Module 2 prices it:
      </p>
      <Eq
        tex="\underbrace{30 \times 784}_{\text{hidden } w} + \underbrace{30}_{\text{hidden } b} + \underbrace{10 \times 30}_{\text{output } w} + \underbrace{10}_{\text{output } b} = 23{,}520 + 30 + 300 + 10 = 23{,}860"
        gloss="Almost exactly double Module 2's 11,935: doubling the hidden layer doubles everything except the ten output biases. At the nudge method's price of two rescores per knob, one step of descent on this network would cost 47,720 passes over the mini-batch. Your backward sweep replaces all of them."
      />
      <p>
        The data is the course's bundled slice of MNIST: 5,000 training images,
        plus 1,000 test images that stay held out, never trained on. Every
        accuracy figure below is scored on those held-out images, so memorizing
        the training set cannot inflate it. For scale: Nielsen's Chapter 1
        trains this same architecture on the full 50,000 images and reports
        about 95 percent; the bundled slice gives up a few points of accuracy
        so the run fits in this page.
      </p>
      <p>
        The panel below hands your code the whole pipeline. It swaps the
        course's nudge-measured gradient for a small adapter that calls your
        backprop once per column of the mini-batch and averages the slopes,
        then runs your Module 3 sgd, unchanged, with learning rate 3.0 and
        mini-batches of 10 (both found by trying, like Module 3's 2.0) for 15
        epochs (Module 3's word: one full pass through the training data, so
        5,000 images in batches of 10 is 500 mini-batches per epoch). And
        before training starts, it prices one step of descent both ways, your
        backprop against nudge-and-measure, on the same mini-batch: Module 3
        priced that comparison by counting passes; here it is wall-clock, on
        your machine, with your own network.
      </p>
      <BackpropTrainPanel />

      <p>
        Module 2's pretrained network, the one whose 11,935 numbers arrived in a
        file, read 86 percent of the test digits. This run lands near 89, and
        nothing about it arrived in a file: the weights come out of your sgd
        walking downhill along slopes computed by your backprop through
        activations computed by your feedforward, against a cost you first
        worked out by hand for four concert corners. Every function in that
        sentence is code you wrote.
      </p>

      <Recap
        items={[
          "Backprop is a forward pass that keeps receipts (every z, every activation) plus a backward sweep that turns them into every slope: BP1 once at the output, then BP2, BP3, BP4 walking toward the input.",
          "It handles one example per call; a mini-batch's gradient is the average of its examples' slopes, so your sgd plugged in unchanged.",
          "The gradient check is the strongest guarantee in this course: two independent methods, your equations and nudge-and-measure, agreeing on every parameter of a network to within one part in ten million.",
          "The digit reader trains in seconds where nudge-measured gradients would need hours: same descent, same cost, same data, cheaper slopes. Module 6 is a playground interlude; Module 7 returns here to make this training better, starting from quadratic cost's saturation weakness.",
        ]}
        chapter="Chapter 2 (how the backpropagation algorithm works)"
        href="http://neuralnetworksanddeeplearning.com/chap2.html"
      />
    </article>
  );
}

// Static diagram: the two receipt lists after a forward pass through a
// two-layer network, each entry labeled with both its positive and negative
// index, so the exercise's negative-indexing reads have a picture.
function ReceiptsDiagram() {
  const BW = 118;
  const BH = 58;
  const GAP = 18;
  const X0 = 150;
  const rowY = { acts: 40, zs: 138 };
  const acts = [
    { label: "x", note: "the input", pos: "[0]", neg: "[-3]" },
    { label: "hidden a", note: "layer 2's confidences", pos: "[1]", neg: "[-2]" },
    { label: "output a", note: "the answer", pos: "[2]", neg: "[-1]" },
  ];
  const zs = [
    { label: "hidden z", note: "layer 2's evidence", pos: "[0]", neg: "[-2]" },
    { label: "output z", note: "layer 3's evidence", pos: "[1]", neg: "[-1]" },
  ];
  const box = (
    x: number,
    y: number,
    b: { label: string; note: string; pos: string; neg: string },
    key: string,
  ) => (
    <g key={key}>
      <text x={x + BW / 2} y={y - 7} textAnchor="middle" className="ripple-change">
        {b.pos}
      </text>
      <rect x={x} y={y} width={BW} height={BH} rx={6} className="ripple-box" />
      <text x={x + BW / 2} y={y + 24} textAnchor="middle" className="ripple-title">
        {b.label}
      </text>
      <text x={x + BW / 2} y={y + 43} textAnchor="middle" className="ripple-value">
        {b.note}
      </text>
      <text x={x + BW / 2} y={y + BH + 16} textAnchor="middle" className="ripple-change">
        {b.neg}
      </text>
    </g>
  );
  return (
    <svg
      viewBox="-116 0 812 220"
      className="chain-ripple"
      role="img"
      aria-label="The two lists a forward pass stores: activations with three entries indexed 0, 1, 2 or -3, -2, -1, and zs with two entries indexed 0, 1 or -2, -1"
    >
      <text x={X0 - 14} y={rowY.acts + BH / 2 + 4} textAnchor="end" className="ripple-title">
        activations
      </text>
      {acts.map((b, i) => box(X0 + i * (BW + GAP), rowY.acts, b, `a${i}`))}
      <text x={X0 - 14} y={rowY.zs + BH / 2 + 4} textAnchor="end" className="ripple-title">
        zs
      </text>
      {zs.map((b, i) => box(X0 + (i + 1) * (BW + GAP), rowY.zs, b, `z${i}`))}
    </svg>
  );
}
