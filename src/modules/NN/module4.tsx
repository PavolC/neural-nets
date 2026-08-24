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
        Module 3's method has a cost worth stating exactly. Each knob's slope came
        from nudging that one knob and rerunning the whole network on the batch,
        twice: two full runs per slope. On the toy network's 33 knobs that was
        affordable. On the digit reader from Module 2, with its 11,935 weights and
        biases, one training step costs 23,870 full runs, and almost all of that
        work is redundant: each run recomputes every neuron in the network to answer
        a question about a single knob. Backpropagation removes the redundancy. One
        forward pass, plus one backward sweep of comparable price, produces every
        slope, all 11,935 of them, exact rather than nudge-and-divide estimates.
        This module explains how; Module 5 is where you implement it. There is no
        code here, just a worked example, a visualization you can step through, the
        four equations, and a quiz.
      </p>

      <p>
        Numbers first, sized for by-hand. Take the smallest network that has
        something hidden inside it: a chain of two neurons, call them A and B.
        Neuron A reads a single input <M tex="x = 1.0" /> with weight{" "}
        <M tex="w_1 = 1.0" /> and bias <M tex="b_1 = -0.5" />; its confidence is
        the only input to neuron B (weight <M tex="w_2 = 2.0" />, bias{" "}
        <M tex="b_2 = -1.0" />), whose confidence is the network's answer. The
        right answer for this example is <M tex="y = 1" />. Run it forward, exactly
        Module 2's arithmetic:
      </p>
      <Eq
        tex="\begin{aligned} z_1 &= 1.0 \times 1.0 - 0.5 = 0.5, & a_1 &= \sigma(0.5) = 0.6225, \\ z_2 &= 2.0 \times 0.6225 - 1.0 = 0.2449, & a_2 &= \sigma(0.2449) = 0.5609 \end{aligned}"
        gloss="Each neuron does what every neuron has done since Module 1: multiply-and-add for its evidence z, sigmoid for its confidence a. Neuron B's input is neuron A's confidence."
      />
      <p>
        The answer should have been 1, so the cost (Module 3's quadratic score, on
        this one example) is <M tex="C = \tfrac12 (1 - 0.5609)^2 = 0.0964" />. Now
        ask Module 3's question about the farthest knob, <M tex="w_1" />, and answer
        it Module 3's way. Nudge <M tex="w_1" /> from 1.00 to 1.01, rerun
        everything, and the cost comes back lower: 0.0959. Slope: change over
        nudge, about <M tex="-0.050" />. Two full runs, one slope, same as always.
      </p>

      <p>
        This time, though, do not just read off the final cost. Rerun the nudge and
        log every value in between, next to its old one. Only five numbers sit
        between <M tex="w_1" /> and the cost, and each of them moved. The picture
        below is that log. In every box: the value before and after the nudge, and
        the change in parentheses. On every arrow: a factor, and the claim that
        each change is the previous change times that factor. The next three
        paragraphs say where each factor comes from; check each one against the log
        as you go (the last digit drifts by one here and there, because the log is
        rounded to five decimals).
      </p>
      <Figure caption="The nudge's whole path. w1 moves by +0.01, and the change ripples through z1, a1, z2, a2 into the cost, multiplied at every arrow by that arrow's local factor. The product of all five factors is the slope of the cost for w1.">
        <ChainRippleDiagram />
      </Figure>

      <p>
        The first factor is the input. <M tex="z_1 = w_1 x + b_1" />, and{" "}
        <M tex="x" /> is 1.0, so when <M tex="w_1" /> moves by 0.01,{" "}
        <M tex="z_1" /> moves by <M tex="1.0 \times 0.01 = 0.0100" />. (Had the
        input been 0.5, only half the nudge would have gotten through: the factor
        is whatever the knob gets multiplied by.)
      </p>
      <p>
        The second factor is sigmoid's own slope. When <M tex="z_1" /> moves a
        little, <M tex="a_1" /> moves by however steep sigmoid is at that spot:
        the responsiveness you watched in Module 1's slider, strong near the
        middle, flat once the neuron is sure. That slope has a shorthand,{" "}
        <M tex="\sigma'(z)" /> (the tick mark is read "prime" and means "the slope
        of"), and a convenient formula:
      </p>
      <Eq
        tex="\sigma'(z) = \sigma(z)\,(1 - \sigma(z))"
        gloss="Sigmoid's slope at z is its own output times one minus its output: confidence times doubt. Biggest at the fence (a quarter, when the output is one half), and nearly zero once the neuron is sure either way."
      />
      <p>
        You do not have to take the formula on faith; it is checkable the Module 3
        way (nudge <M tex="z" />, divide). At <M tex="z_1 = 0.5" /> it says{" "}
        <M tex="0.6225 \times 0.3775 = 0.235" />, and the log agrees:{" "}
        <M tex="0.235 \times 0.0100 = 0.00235" />, exactly <M tex="a_1" />'s
        change.
      </p>
      <p>
        The third factor is the wire. <M tex="z_2 = w_2 a_1 + b_2" />, so{" "}
        <M tex="a_1" />'s change rides through multiplied by <M tex="w_2 = 2.0" />:
        that predicts <M tex="2.0 \times 0.00235 = 0.00470" />, matching the log.
        The fourth is sigmoid again, at neuron B's own evidence:{" "}
        <M tex="\sigma'(0.2449) = 0.5609 \times 0.4391 = 0.246" />, predicting{" "}
        <M tex="a_2" />'s change of 0.00115. The fifth and last factor is the gap.
        The cost is half the gap squared, and its slope formula comes on the same
        checkable terms as sigmoid's: the slope of <M tex="\tfrac12(\text{gap})^2" />{" "}
        is the gap itself, here <M tex="a_2 - y = -0.4391" /> (this is what the half
        in Module 3's cost was for: squaring puts a factor of 2 into the slope, and
        the half cancels it, leaving only the gap). Against the log:{" "}
        <M tex="-0.439 \times 0.00116 = -0.00051" />, the cost's change exactly.
      </p>
      <p>
        Now stack the five predictions. Each change was the previous change times a
        factor, so the final change is the original nudge times all five factors
        multiplied together, and the slope is just those factors:
      </p>
      <Eq
        tex="1.0 \times 0.2350 \times 2.0 \times 0.2463 \times (-0.4391) = -0.0508"
        gloss="The slope of the cost with respect to w1, computed with no reruns at all: the product of the local factors along the nudge's path."
      />
      <p>
        The nudge measurement said <M tex="-0.050" />, and the product says{" "}
        <M tex="-0.0508" />. They agree, and the product is the exact one: the
        nudge method approximates, because it uses a small step where the true
        slope wants a vanishingly small one. The multiply-the-local-slopes rule is
        called the chain rule, and it is the only piece of mathematics
        backpropagation needs. The rest of the algorithm is arranging the
        multiplications so that no factor is ever computed twice.
      </p>

      <p>
        Arranged naively, it is still one product of five factors per knob, so look
        for shared work. Write <M tex="w_1" />'s product next to{" "}
        <M tex="b_1" />'s. A nudge to <M tex="b_1" /> starts at the same neuron and
        rides the same path; the only difference is the first factor, because{" "}
        <M tex="b_1" /> reaches <M tex="z_1" /> one-for-one (add 0.01 to{" "}
        <M tex="b_1" /> and <M tex="z_1" /> moves by 0.01, no multiplying by{" "}
        <M tex="x" />):
      </p>
      <Eq
        tex="\begin{gathered} T = \underbrace{0.2350 \times 2.0 \times 0.2463 \times (-0.4391)}_{\text{everything from } z_1 \text{ to the cost}} = -0.0508 \\[0.9em] \text{slope for } w_1 = 1.0 \times T, \qquad \text{slope for } b_1 = 1 \times T \end{gathered}"
        gloss="Both knobs live at neuron A, so from z1 onward their ripples are identical. Compute the shared part T once and both slopes are one extra multiplication each."
      />
      <p>
        That shared part is the thing to name. A neuron's <M tex="\delta" /> (the
        Greek letter delta) is the slope of the cost with respect to that neuron's
        evidence <M tex="z" />: everything downstream of the neuron, collapsed into
        one number. Call it the neuron's blame: how much the cost cares about this
        neuron's total. The chain has two neurons, so two blames, and the second
        one is built from the first:
      </p>
      <Eq
        tex="\delta_B = \underbrace{0.2463}_{\text{own } \sigma'} \times \underbrace{(-0.4391)}_{\text{gap}} = -0.1081, \qquad \delta_A = \underbrace{0.2350}_{\text{own } \sigma'} \times \underbrace{2.0}_{\text{wire}} \times \underbrace{(-0.1081)}_{\delta_B} = -0.0508"
        gloss="Neuron B touches the cost directly, so its blame is its responsiveness times the gap. Neuron A's blame arrives from downstream: delta-B, carried back through the connecting wire, scaled by A's own responsiveness. Blame flows backward, one cheap step per neuron."
      />
      <p>
        And once a neuron's blame is known, its knobs read their slopes straight
        off it. A bias's slope is the blame itself, because a bias nudges its{" "}
        <M tex="z" /> one-for-one. A weight's slope is the blame times the
        activation arriving on its wire, because that activation is what the weight
        gets multiplied by. All four knobs of the chain:
      </p>
      <table className="truth-table">
        <thead>
          <tr>
            <th>knob</th>
            <th>slope</th>
            <th>read off as</th>
          </tr>
        </thead>
        <tbody>
          <tr><td><M tex="b_2" /></td><td>−0.1081</td><td><M tex="\delta_B" /></td></tr>
          <tr><td><M tex="w_2" /></td><td>−0.0673</td><td><M tex="a_1 \times \delta_B = 0.6225 \times (-0.1081)" /></td></tr>
          <tr><td><M tex="b_1" /></td><td>−0.0508</td><td><M tex="\delta_A" /></td></tr>
          <tr><td><M tex="w_1" /></td><td>−0.0508</td><td><M tex="x \times \delta_A = 1.0 \times (-0.0508)" /></td></tr>
        </tbody>
      </table>
      <p>
        The total: one forward pass, one backward sweep of two blames, and all four
        slopes. That is backpropagation, in miniature.
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
        Time to run the same story on a network instead of a chain. The one below
        is Module 1's contrarian network grown by one hidden neuron, 2-3-1 instead
        of 2-2-1, so that no two layers have the same size and every shape stays
        honest. With whole layers in play, one piece of notation: number the
        layers 1, 2, 3 (inputs, hidden, output) and hang the layer number on each
        symbol as a superscript, so <M tex="a^2" /> is the hidden layer's column
        of activations, <M tex="w^2" /> the weight matrix into it, and{" "}
        <M tex="\delta^2" /> its column of blames, one entry per neuron. The
        superscript is a label, never a power; nothing here is squared. Count the
        knobs, like the 11,935:
      </p>
      <Eq
        tex="\underbrace{3 \times 2}_{\text{hidden } w} + \underbrace{3}_{\text{hidden } b} + \underbrace{1 \times 3}_{\text{output } w} + \underbrace{1}_{\text{output } b} = 6 + 3 + 3 + 1 = 13"
        gloss="Thirteen knobs, so the backward pass must deliver thirteen slopes. The weights shown on the wires are a hand-picked starting point, not a trained network: training them is Module 5's job."
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
      <Figure caption="Backpropagation on a 2-3-1 network, one step at a time: forward pass, cost, then the backward sweep. The table lists every evidence z, activation a, and blame delta; the panel beside it edits whichever of the 13 knobs is selected.">
        <BackpropStepper />
      </Figure>

      <p>
        Those six steps are the whole algorithm, at any network size. Written down,
        they are the four equations of backpropagation, numbered BP1 to BP4 after
        Nielsen's Chapter 2, whose framing this module adapts; every symbol in them
        appeared on the cards you just stepped through. One bookkeeping note before
        reading them: in the chain, blames wore neuron names (<M tex="\delta_A" />,{" "}
        <M tex="\delta_B" />); in layer language, <M tex="\delta^2" /> is a whole
        column of blames at once, one entry per neuron of layer 2, and{" "}
        <M tex="L" /> names the last layer (3 in the stepper).
      </p>
      <Eq
        tex="\delta^L = (a^L - y) \odot \sigma'(z^L) \tag{BP1}"
        gloss="Blame starts at the output layer: the gap in each output entry, times how responsive that neuron currently is (the circled dot means multiply matching entries, NumPy's plain *, no adding). A saturated neuron has sigma-prime near zero and soaks up almost no blame, even when it is wrong."
      />
      <Eq
        tex="\delta^l = \big( (w^{l+1})^T \, \delta^{l+1} \big) \odot \sigma'(z^l) \tag{BP2}"
        gloss="Blame flows backward: each neuron of layer l collects blame along its outgoing wires and adds, then scales by its own responsiveness. The raised T (transpose) flips the matrix so the same wires read backward, and the shapes agree by Module 2's inner-numbers-touch rule: in the stepper, (3, 1) times (1, 1) gives (3, 1), one collected blame per hidden neuron."
      />
      <Eq
        tex="\frac{\partial C}{\partial b^l} = \delta^l \tag{BP3}"
        gloss="The curly-d expression is read as one name, 'the slope of C per nudge of b', the number Module 3 measured with nudge-and-divide. Every bias's slope is exactly its neuron's blame, because a bias nudges its neuron's evidence one-for-one."
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
        Module 5 is where you implement all four equations, and the implementation
        goes better if you can already think in them. So before moving on, three
        predictions. Commit to an answer before you click, and verify each one
        afterward by actually doing it in the stepper above (the questions describe
        the stepper's starting weights; press Reset weights first if you have been
        experimenting).
      </p>
      <DeltaQuiz />

      <p>
        The tally, one last time, at the digit reader's scale. Module 3's method
        needs two full network runs per knob per training step: 23,870 forward
        passes to take one step downhill. The four equations need one forward pass,
        plus one backward sweep that touches each neuron and each weight about
        once, roughly the price of a second forward pass. Two passes against
        twenty-four thousand: about ten thousand times cheaper, and exact instead
        of estimated. That is why backpropagation, and not a faster computer, is
        what made neural networks trainable. In Module 5 you implement BP1 through
        BP4 in NumPy, and the nudge method takes on a new job: the course will
        nudge-and-measure every knob of a tiny network the slow way and compare
        your backward pass against it, number by number. When the two agree, your
        gradients are right, and the real training run begins.
      </p>

      <Recap
        items={[
          "Backprop answers 'which way should each knob move' with one forward pass and one backward sweep of blame, instead of two full reruns per knob, and its answers are exact.",
          "A nudge's effect on the cost is the product of the local factors along its path (the chain rule); a neuron's blame delta is the shared part of those products, computed once per neuron.",
          "The four equations: blame starts at the output as gap times sigma-prime (BP1), flows backward through the transposed wires (BP2), every bias's slope is its neuron's blame (BP3), and every weight's slope is blame times the activation its wire carried (BP4).",
          "A saturated neuron has sigma-prime near zero and soaks up almost no blame even when badly wrong: quadratic cost's weakness, and Module 7's opening problem.",
        ]}
        chapter="Chapter 2 (how the backpropagation algorithm works)"
        href="http://neuralnetworksanddeeplearning.com/chap2.html"
      />
    </article>
  );
}

// Static diagram: the logged ripple of the +0.01 nudge to w1 through the
// two-neuron chain worked in the prose. Each box shows a quantity's value
// before and after the nudge (change in parentheses); each arrow carries the
// local factor that predicts the next change from the previous one.
function ChainRippleDiagram() {
  const boxes = [
    { title: "nudge w₁", value: "1.00 → 1.01", change: "(+0.01000)" },
    { title: "z₁", value: "0.5000 → 0.5100", change: "(+0.01000)" },
    { title: "a₁", value: "0.6225 → 0.6248", change: "(+0.00235)" },
    { title: "z₂", value: "0.2449 → 0.2496", change: "(+0.00469)" },
    { title: "a₂", value: "0.5609 → 0.5621", change: "(+0.00116)" },
    { title: "cost C", value: "0.09639 → 0.09589", change: "(−0.00051)" },
  ];
  const factors = [
    { f: "× 1.0", why: "the input x" },
    { f: "× 0.235", why: "σ′ at z₁" },
    { f: "× 2.0", why: "the wire w₂" },
    { f: "× 0.246", why: "σ′ at z₂" },
    { f: "× −0.439", why: "the gap" },
  ];
  const BW = 110; // box width
  const GAP = 26; // arrow length between boxes
  const X0 = 12;
  const Y0 = 44; // box top
  const BH = 62;
  return (
    <svg viewBox="0 0 812 182" className="chain-ripple" role="img"
         aria-label="A +0.01 nudge to w1 ripples through z1, a1, z2, a2 and the cost; each hop multiplies the change by a local factor, and the product of all five factors is the slope, −0.0508">
      {boxes.map((b, i) => {
        const x = X0 + i * (BW + GAP);
        return (
          <g key={i}>
            <rect x={x} y={Y0} width={BW} height={BH} rx={6} className="ripple-box" />
            <text x={x + BW / 2} y={Y0 + 18} textAnchor="middle" className="ripple-title">
              {b.title}
            </text>
            <text x={x + BW / 2} y={Y0 + 36} textAnchor="middle" className="ripple-value">
              {b.value}
            </text>
            <text x={x + BW / 2} y={Y0 + 53} textAnchor="middle" className="ripple-change">
              {b.change}
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
            <text x={(x1 + x2) / 2} y={Y0 - 14} textAnchor="middle" className="ripple-factor">
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
      <text x={406} y={174} textAnchor="middle" className="ripple-product">
        slope for w₁ = 1.0 × 0.235 × 2.0 × 0.246 × (−0.439) = −0.0508
      </text>
    </svg>
  );
}
