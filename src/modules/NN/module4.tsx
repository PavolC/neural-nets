import { AfterThis, Figure, Recap } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { BackpropStepper } from "./interactives/BackpropStepper";
import { DeltaQuiz } from "./interactives/DeltaQuiz";

export function Module4() {
  return (
    <article className="module">
      <h2>Module 4: Backpropagation, the idea</h2>
      <AfterThis
        items={[
          "Trace a nudge through a network by hand and get the exact slope by multiplying local factors.",
          "Say in words what each of backpropagation's four equations computes, and why its shapes work out.",
          "Predict where blame flows, and where it dies, before a visualization confirms you.",
        ]}
      />

      <p>
        Module 3 ended with a bill. To train, you needed the slope of the cost for
        every knob, and you measured each one the honest way: nudge that knob, rerun
        the whole network on the batch, twice, divide. Two full runs bought one
        slope. On the toy network's 33 knobs that was affordable. On the digit
        reader from Module 2, with its 11,935 weights and biases, one training step
        costs 23,870 full runs, and the waste is obvious once you say it out loud:
        run number fourteen thousand recomputes every neuron in the network just to
        ask a question about one knob. Backpropagation gets every slope, all 11,935
        of them, from one forward pass plus one backward sweep of comparable price.
        And they are exact, not nudge-and-divide estimates. This module is about
        understanding how that can possibly work; you will not write code today
        (that is Module 5, and it will be short). Instead: one worked example, four
        equations, one visualization you can step through, and a quiz.
      </p>

      <p>
        Numbers first, sized for by-hand. Take the smallest network that has
        something hidden inside it: a chain of two neurons. The first neuron reads
        a single input <M tex="x = 1.0" /> with weight <M tex="w_1 = 1.0" /> and
        bias <M tex="-0.5" />; its confidence is the only input to the second
        neuron (weight <M tex="w_2 = 2.0" />, bias <M tex="-1.0" />), whose
        confidence is the network's answer. The right answer for this example is{" "}
        <M tex="y = 1" />. Run it forward, exactly Module 2's arithmetic:
      </p>
      <Eq
        tex="\begin{aligned} z_1 &= 1.0 \times 1.0 - 0.5 = 0.5, & a_1 &= \sigma(0.5) = 0.6225, \\ z_2 &= 2.0 \times 0.6225 - 1.0 = 0.2449, & a_2 &= \sigma(0.2449) = 0.5609 \end{aligned}"
        gloss="Each neuron does what every neuron has done since Module 1: multiply-and-add for its evidence z, sigmoid for its confidence a. The second neuron's input is the first one's confidence."
      />
      <p>
        The answer should have been 1, so the cost (Module 3's quadratic score, on
        this one example) is <M tex="C = \tfrac12 (1 - 0.5609)^2 = 0.0964" />. Now
        ask Module 3's question about the farthest knob, <M tex="w_1" />, and answer
        it Module 3's way. Nudge <M tex="w_1" /> from 1.0 to 1.01 and rerun
        everything: <M tex="a_1" /> becomes 0.6248, <M tex="a_2" /> becomes 0.5621,
        and the cost drops to 0.0959. Slope: change over nudge, about{" "}
        <M tex="-0.050" />. Two full runs, one slope, same as always.
      </p>

      <p>
        This time, though, do not just read off the final cost. Watch where the
        nudge went. It passed five stations, and each station multiplied it by a
        number you can compute locally, without rerunning anything:
      </p>
      <p>
        Station one: <M tex="z_1 = w_1 x + b_1" />. Nudge <M tex="w_1" /> by a tiny
        amount and <M tex="z_1" /> moves by <M tex="x" /> times that amount, because{" "}
        <M tex="w_1" /> gets multiplied by <M tex="x" /> and nothing else. Local
        factor: <M tex="1.0" />.
      </p>
      <p>
        Station two: <M tex="a_1 = \sigma(z_1)" />. When <M tex="z_1" /> moves a
        little, <M tex="a_1" /> moves by sigmoid's own slope at that spot, exactly
        the slope you watched in Module 1's slider (eager near the middle, flat
        once the neuron is sure). That slope has a shorthand,{" "}
        <M tex="\sigma'(z)" /> (the tick mark is read "prime" and means "the slope
        of"), and it has a convenient formula:
      </p>
      <Eq
        tex="\sigma'(z) = \sigma(z)\,(1 - \sigma(z))"
        gloss="Sigmoid's slope at z is its own output times one minus its output: confidence times doubt. Biggest at the fence (a quarter, when the output is one half), and nearly zero once the neuron is sure either way."
      />
      <p>
        You do not have to take the formula on faith; test it the Module 3 way. At{" "}
        <M tex="z = 0.5" /> the formula says <M tex="0.6225 \times 0.3775 = 0.2350" />,
        and the nudge measurement <M tex="(\sigma(0.51) - \sigma(0.50))/0.01" />{" "}
        gives 0.2347. Local factor at station two: <M tex="0.2350" />.
      </p>
      <p>
        Station three: <M tex="z_2 = w_2 a_1 + b_2" />. A wiggle in <M tex="a_1" />{" "}
        is amplified by the wire it rides on. Local factor: <M tex="w_2 = 2.0" />.
        Station four: sigmoid again, at the second neuron's evidence. Local factor:{" "}
        <M tex="\sigma'(0.2449) = 0.5609 \times 0.4391 = 0.2463" />. Station five:
        the cost. Call the gap <M tex="g = a_2 - y = -0.4391" />, so{" "}
        <M tex="C = \tfrac12 g^2" />, and nudge the gap by a tiny <M tex="\varepsilon" />:
      </p>
      <Eq
        tex="\tfrac12 (g + \varepsilon)^2 - \tfrac12 g^2 = g\,\varepsilon + \tfrac12 \varepsilon^2 \approx g\,\varepsilon"
        gloss="Expand the square, high-school style. For a tiny nudge the epsilon-squared term is negligible, so the cost moves by the gap times the nudge: the last local factor is the gap itself, here −0.4391."
      />
      <p>
        Now multiply the five factors:
      </p>
      <Eq
        tex="1.0 \times 0.2350 \times 2.0 \times 0.2463 \times (-0.4391) = -0.0508"
        gloss="The slope of the cost with respect to w1, computed with no reruns at all: just the product of the local factors along the nudge's path."
      />
      <p>
        The nudge measurement said <M tex="-0.050" />, and the product says{" "}
        <M tex="-0.0508" />. They agree, and of the two, the product is the exact
        one: the nudge method was always an estimate (it used a small step instead
        of a truly tiny one). This multiply-the-local-slopes fact is called the
        chain rule, named after exactly the picture you just walked through, and it
        is the entire mathematical content of backpropagation. Everything else is
        bookkeeping.
      </p>
      <Figure caption="The path of the nudge. Each box is a quantity the forward pass computed (its value inside); each arrow multiplies a passing wiggle by its local factor. The slope of the cost for w1 is the product of all five factors: −0.0508.">
        <ChainRippleDiagram />
      </Figure>

      <p>
        The bookkeeping matters, though, because done naively you would compute one
        product per knob. Look at <M tex="b_1" />'s product: nudging{" "}
        <M tex="b_1" /> moves <M tex="z_1" /> one-for-one (its local factor is 1),
        and from <M tex="z_1" /> onward its path is identical to{" "}
        <M tex="w_1" />'s. Same for the second neuron's two knobs: both paths merge
        at <M tex="z_2" />. So compute each shared tail once and give it a name:
        a neuron's <M tex="\delta" /> (the Greek letter delta) is the slope of the
        cost with respect to that neuron's evidence <M tex="z" />. Call it the
        neuron's blame: how much the cost cares about this neuron's total. In the
        chain, the second neuron's blame is{" "}
        <M tex="\delta_B = 0.2463 \times (-0.4391) = -0.1081" /> (its own sigmoid
        slope times the gap), and the first neuron's is{" "}
        <M tex="\delta_A = 0.2350 \times 2.0 \times (-0.1081) = -0.0508" />. Read
        that second one closely: it is built from <M tex="\delta_B" />, multiplied
        by the wire between them (2.0) and by the first neuron's own sigmoid slope.
        Blame flows backward, one cheap step per neuron.
      </p>
      <p>
        And once a neuron's blame is known, every one of its knobs gets its slope
        for free. Its bias's slope is the blame itself (the bias moves{" "}
        <M tex="z" /> one-for-one): <M tex="-0.1081" /> for <M tex="b_2" />,{" "}
        <M tex="-0.0508" /> for <M tex="b_1" />. A weight's slope is the blame
        times the activation arriving on that wire, because that activation is what
        the weight gets multiplied by: for <M tex="w_2" /> that is{" "}
        <M tex="0.6225 \times (-0.1081) = -0.0673" />, and for <M tex="w_1" /> it
        is <M tex="1.0 \times (-0.0508) = -0.0508" /> (the same number as the bias
        only because the wire happens to carry exactly 1.0). Count what just
        happened: one forward pass, one backward sweep of two blames, and all four
        slopes are on the table. That is backpropagation, in miniature.
      </p>

      <p>
        One thing changes in a real network: a hidden neuron does not feed one
        neuron, it feeds many (in the digit reader, every hidden neuron feeds all
        ten outputs). Then blame arrives along every outgoing wire, and the
        arrivals simply add. Small example: suppose a hidden neuron feeds two
        output neurons, through a wire of weight 4.0 into a neuron with blame{" "}
        <M tex="-0.10" />, and a wire of weight <M tex="-4.0" /> into a neuron with
        blame <M tex="0.08" />. The blame it collects is{" "}
        <M tex="4.0 \times (-0.10) + (-4.0) \times 0.08 = -0.72" />, and then it
        multiplies by its own sigmoid slope as always, say 0.15, giving{" "}
        <M tex="\delta = -0.108" />. Collect along each wire, add, multiply by your
        own <M tex="\sigma'" />: that is the whole upgrade.
      </p>

      <p>
        Now the four equations, in the compact matrix language of Module 2. They
        need four pieces of notation, each small. First, layers get numbers, and
        the numbers ride as superscripts: the input column is layer 1, the hidden
        layer is 2, the output layer is 3, so <M tex="w^2" /> means "the weight
        matrix into layer 2" and <M tex="\delta^3" /> means "the blame column of
        layer 3". The superscript is a label, never a power; nothing in these
        equations gets squared. Second, the curly <M tex="\partial" />:{" "}
        <M tex="\partial C / \partial w" /> is written like a fraction but read as
        one name, "the slope of C per unit nudge of w", exactly the number Module 3
        measured with nudge-and-divide. Third, <M tex="\odot" /> means multiply
        matching entries (what NumPy's plain <code>*</code> did in Module 1's
        exercise, entry by entry; no adding). Fourth, the superscript{" "}
        <M tex="T" />, read "transpose", flips a matrix so rows become columns.
        Meaning: the same wires, read in the reverse direction. Check it with
        shapes, the inner-numbers-must-touch rule from Module 2: in a 2-3-1
        network, <M tex="w^3" /> is <M tex="(1, 3)" /> (one output neuron reading
        three hidden ones), so <M tex="(w^3)^T" /> is <M tex="(3, 1)" />, and{" "}
        <M tex="(w^3)^T" /> times the <M tex="(1, 1)" /> blame <M tex="\delta^3" />{" "}
        gives <M tex="(3, 1)" />: one collected blame per hidden neuron, which is
        exactly the sum-along-your-wires from the last paragraph, done for a whole
        layer at once. Here they are, numbered BP1 to BP4 following Nielsen's
        Chapter 2, whose framing this module adapts:
      </p>
      <Eq
        tex="\delta^L = (a^L - y) \odot \sigma'(z^L) \tag{BP1}"
        gloss="Blame starts at the output layer (L is the last layer's number, 3 in our stepper): the gap in each output entry, times how responsive that neuron currently is. A confident (saturated) neuron has sigma-prime near zero and soaks up almost no blame, even when it is wrong."
      />
      <Eq
        tex="\delta^l = \big( (w^{l+1})^T \, \delta^{l+1} \big) \odot \sigma'(z^l) \tag{BP2}"
        gloss="Blame flows backward: layer l collects the next layer's blame through the transposed wires (add along each neuron's outgoing connections), then scales by its own responsiveness. Apply repeatedly and blame reaches every layer."
      />
      <Eq
        tex="\frac{\partial C}{\partial b^l} = \delta^l \tag{BP3}"
        gloss="Every bias's slope is exactly its neuron's blame, because a bias nudges its neuron's evidence one-for-one. No extra computation at all."
      />
      <Eq
        tex="\frac{\partial C}{\partial w^l} = \delta^l \, (a^{l-1})^T \tag{BP4}"
        gloss="Every weight's slope is the receiving neuron's blame times the activation the wire carried. The shapes tell the story: delta is (m, 1), the transposed activations are (1, n), and their product is (m, n), one slope per weight, in exactly the shape of w itself; the entry in row j, column k is delta-j times a-k."
      />
      <p>
        Read the four as one machine. The forward pass computes and stores every{" "}
        <M tex="z" /> and <M tex="a" /> (you built this in Module 2). BP1 turns the
        final gap into the last layer's blame. BP2 walks the blame backward one
        layer at a time, reusing the same weight matrices the forward pass used.
        BP3 and BP4 then read every slope directly off the blames and the stored
        activations: no equation in the list reruns the network, and none of them
        estimates anything.
      </p>

      <p>
        Now watch the machine run. The network below is Module 1's contrarian
        network grown by one hidden neuron, 2-3-1 instead of 2-2-1, so that no two
        layers have the same size and every shape stays honest. Count its knobs
        layer by layer, like the 11,935:
      </p>
      <Eq
        tex="\underbrace{3 \times 2}_{\text{hidden } w} + \underbrace{3}_{\text{hidden } b} + \underbrace{1 \times 3}_{\text{output } w} + \underbrace{1}_{\text{output } b} = 6 + 3 + 3 + 1 = 13"
        gloss="Thirteen knobs, so the backward pass owes us thirteen slopes. The weights shown on the wires are a hand-picked starting point, not a trained network: training them is Module 5's job."
      />
      <p>
        The input is fixed at one training example, good weather without the
        friend, where the contrarian's answer is go (<M tex="y = 1" />). How to
        read the diagram: each circle is a neuron except the two gray ones, which
        are just the input numbers. Wire color is the weight's sign (red excites,
        blue vetoes, the same code as Module 2's weight images), thickness is its
        size, and the number on the wire is the weight itself. A neuron's circle
        fills with green as its activation rises, its bias sits under it, and its
        blame <M tex="\delta" /> appears under that, in dark red, once the backward
        pass reaches it. The equation card above the diagram always shows the step
        you are on. Step forward through the forward pass, cost, and backward pass;
        then click any wire or circle to select a knob and drag its slider, and
        every revealed number recomputes live. Two experiments worth running: nudge
        the wire from <M tex="h_2" /> to the output (weight <M tex="-4.0" />) and
        watch how a negative wire flips the sign of the blame it delivers; and drag
        the output bias far negative, then watch what saturation does to every
        blame in the network.
      </p>
      <Figure caption="Backpropagation on a 2-3-1 network, one step at a time: forward pass, cost, then BP1, BP2, and BP3-BP4. The table lists every evidence z, activation a, and blame delta; the panel beside it edits whichever of the 13 knobs is selected.">
        <BackpropStepper />
      </Figure>

      <p>
        No code this module, deliberately: Module 5 is where you implement all four
        equations, and it goes much better if you can already think in them. So
        before moving on, three predictions. Commit to an answer before you click,
        and verify each one afterward by actually doing it in the stepper above
        (the questions describe the stepper's starting weights; press Reset weights
        first if you have been experimenting).
      </p>
      <DeltaQuiz />

      <p>
        Close with the tally that justifies the whole module. For the digit
        reader's 11,935 knobs, Module 3's method needs two full network runs per
        knob per training step: 23,870 forward passes to take one step downhill.
        The four equations need one forward pass, plus one backward sweep that
        touches each neuron and each weight about once, roughly the price of a
        second forward pass. Call it two passes against twenty-four thousand: about
        ten thousand times cheaper, and exact instead of estimated. That is why
        backpropagation, and not a faster computer, is what made neural networks
        trainable. In Module 5 you will implement BP1 through BP4 in NumPy, and the
        nudge method gets a retirement job: the course will nudge-and-measure every
        knob of a tiny network the slow way and compare your backward pass against
        it, number by number. When the two agree, your gradients are right, and the
        real training run begins.
      </p>

      <Recap
        items={[
          "Backprop answers 'which way should each knob move' with one forward pass and one backward sweep of blame, instead of two full reruns per knob, and its answers are exact.",
          "A nudge's effect on the cost is the product of the local factors along its path (the chain rule); a neuron's blame delta is the shared tail of those products, computed once per neuron.",
          "The four equations: blame starts at the output as gap times sigma-prime (BP1), flows backward through the transposed wires (BP2), every bias's slope is its neuron's blame (BP3), and every weight's slope is blame times the activation its wire carried (BP4).",
          "A saturated neuron has sigma-prime near zero and soaks up almost no blame even when badly wrong: quadratic cost's weakness, and Module 7's opening problem.",
        ]}
        chapter="Chapter 2 (how the backpropagation algorithm works)"
        href="http://neuralnetworksanddeeplearning.com/chap2.html"
      />
    </article>
  );
}

// Static diagram: the five-station path of a nudge to w1 through the
// two-neuron chain worked in the prose, each arrow labeled with its local
// factor, and the product line underneath.
function ChainRippleDiagram() {
  const boxes = [
    { title: "nudge w₁", value: "1.0 → ?" },
    { title: "z₁", value: "0.5" },
    { title: "a₁", value: "0.6225" },
    { title: "z₂", value: "0.2449" },
    { title: "a₂", value: "0.5609" },
    { title: "cost C", value: "0.0964" },
  ];
  const factors = [
    { f: "× 1.0", why: "the input x" },
    { f: "× 0.235", why: "σ′ at z₁" },
    { f: "× 2.0", why: "the wire w₂" },
    { f: "× 0.246", why: "σ′ at z₂" },
    { f: "× −0.439", why: "the gap" },
  ];
  const BW = 82; // box width
  const GAP = 42; // arrow length between boxes
  const X0 = 14;
  const Y0 = 42; // box top
  const BH = 48;
  return (
    <svg viewBox="0 0 720 160" className="chain-ripple" role="img"
         aria-label="A nudge to w1 passes through z1, a1, z2, a2 and the cost; each hop multiplies it by a local factor, and the product of all five factors is the slope, −0.0508">
      {boxes.map((b, i) => {
        const x = X0 + i * (BW + GAP);
        return (
          <g key={i}>
            <rect x={x} y={Y0} width={BW} height={BH} rx={6} className="ripple-box" />
            <text x={x + BW / 2} y={Y0 + 20} textAnchor="middle" className="ripple-title">
              {b.title}
            </text>
            <text x={x + BW / 2} y={Y0 + 38} textAnchor="middle" className="ripple-value">
              {b.value}
            </text>
          </g>
        );
      })}
      {factors.map((f, i) => {
        const x1 = X0 + BW + i * (BW + GAP);
        const x2 = x1 + GAP;
        const y = Y0 + BH / 2;
        return (
          <g key={i}>
            <line x1={x1 + 2} y1={y} x2={x2 - 6} y2={y} className="ripple-arrow" markerEnd="url(#ripple-head)" />
            <text x={(x1 + x2) / 2} y={Y0 - 12} textAnchor="middle" className="ripple-factor">
              {f.f}
            </text>
            <text x={(x1 + x2) / 2} y={Y0 + BH + 22} textAnchor="middle" className="ripple-why">
              {f.why}
            </text>
          </g>
        );
      })}
      <defs>
        <marker id="ripple-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6"
                markerHeight="6" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" className="ripple-head" />
        </marker>
      </defs>
      <text x={360} y={152} textAnchor="middle" className="ripple-product">
        slope for w₁ = 1.0 × 0.235 × 2.0 × 0.246 × (−0.439) = −0.0508
      </text>
    </svg>
  );
}
