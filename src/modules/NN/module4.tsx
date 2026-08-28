import { AfterThis, Aside, Figure, ModuleToc, Recap, SectionHeader, fig } from "../../components/ModuleBits";
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
      <ModuleToc />

      <p>
        Module 3 ended with a bill and a factor it could not remove. Mini-batches
        shrank the crowd of examples, but every training step still pays 11,935
        times 2: each knob's slope needs the cost measured twice (the both-sides
        nudge), each measurement is a full pass over the mini-batch, and that is
        23,870 passes per step. Almost all of the work is redundant: every pass
        recomputes every neuron in the network to answer a question about a single
        knob. Backpropagation removes the redundancy. One forward pass, plus one
        backward sweep of comparable price, produces every slope, all 11,935 of
        them, exact rather than nudge-and-divide estimates.
        This module explains how; Module 5 is where you implement it. There is no
        code here, just a worked example, a visualization you can step through, the
        four equations, and a quiz.
      </p>

      <SectionHeader id="m4-chain" title="The two-neuron chain" />
      <p>
        Numbers first, sized for by-hand. Take the smallest network that has
        something hidden inside it: a chain of two neurons, call them A and B.
        Neuron A reads the lone input; neuron B reads only A's report and gives
        the network's answer. The arrows connecting them are the network's wires:
        a wire is the route a value travels from one place to the next, and each
        wire carries one weight, the multiplier applied to whatever travels it.
        Here is the whole thing, every knob in view:
      </p>
      <Figure caption="The worked example, drawn like Module 1's diagrams: gray is the input (a number, not a neuron), each green circle is one sigmoid neuron. Four knobs in total: one weight on each of the two wires, one bias inside each neuron.">
        <ChainNetDiagram />
      </Figure>
      <p>
        Module 2 wrote a whole layer as <M tex="a' = \sigma(Wa + b)" />. Both of
        this chain's layers are one neuron wide, so each symbol shrinks from a
        column to a single number and keeps its usual name: a neuron totals its
        evidence <M tex="z" /> (Module 1's multiply-and-add) and reports its
        confidence <M tex="a = \sigma(z)" />. Subscripts number the position along
        the chain: neuron A turns the input <M tex="x" /> into <M tex="z_1" /> and{" "}
        <M tex="a_1" /> using its knobs <M tex="w_1" /> and <M tex="b_1" />, and
        neuron B turns <M tex="a_1" /> into <M tex="z_2" /> and <M tex="a_2" />{" "}
        using <M tex="w_2" /> and <M tex="b_2" />. One habit to unlearn: in
        Module 1, <M tex="w_1" /> and <M tex="w_2" /> were two weights into the
        same neuron, numbered by input; here each neuron has one weight, so the
        subscript numbers the neuron instead. The right answer for this example is{" "}
        <M tex="y = 1" />. Run it forward:
      </p>
      <Eq
        tex="\begin{aligned} z_1 &= 1.0 \times 1.0 - 0.5 = 0.5, & a_1 &= \sigma(0.5) = 0.6225, \\ z_2 &= 2.0 \times 0.6225 - 1.0 = 0.2449, & a_2 &= \sigma(0.2449) = 0.5609 \end{aligned}"
        gloss="Neuron A first, then neuron B, which reads A's confidence 0.6225 as its only input. Your sigmoid from Module 1 can confirm every entry."
      />
      <p>
        The answer should have been 1. Score it with Module 3's quadratic cost,
        which with a single example (<M tex="n = 1" />) reduces to{" "}
        <M tex="C = \tfrac12 (1 - 0.5609)^2 = 0.0964" />. Now ask Module 3's
        question about the farthest knob, <M tex="w_1" />, and answer it Module 3's
        way. Nudge <M tex="w_1" /> from 1.00 to 1.01 with the other three knobs
        held still, rerun everything, and the cost comes back lower: 0.0959. Slope: change over nudge,{" "}
        <M tex="(0.0959 - 0.0964) / 0.01 \approx -0.050" />. Two cost measurements,
        one slope, same as always.
      </p>

      <SectionHeader id="m4-log" title="The ripple, logged" />
      <p>
        This time, though, do not just read off the final cost. Rerun the nudge and
        log every value in between, next to its old one. Only five numbers sit
        between <M tex="w_1" /> and the cost, and each of them moved. The picture
        below is that log, and its boxes are not new places: it is the wiring
        diagram above, unrolled into the numbers it makes. <M tex="z_1" /> and{" "}
        <M tex="a_1" /> live inside neuron A (its total, then its squash),{" "}
        <M tex="z_2" /> and <M tex="a_2" /> inside neuron B, the arrow between
        the two is the wire, and the cost hangs off the end, outside the
        network, keeping score. In every box: the value before and after the
        nudge, and the change in parentheses. On every arrow: a factor, and the claim that
        each change is the previous change times that factor. Where the factors
        come from is next, one kind at a time; check each against the log as you
        go (the last digit drifts by one here and there, because the log is
        rounded to five decimals).
      </p>
      <Figure caption="The nudge's whole path. w1 moves by +0.01, and the change ripples through z1, a1, z2, a2 into the cost, multiplied at every arrow by that arrow's local factor. The green bands mark which boxes live inside which neuron; the arrow crossing between bands is the wire. The factor colors sort the arrows into their three kinds. Purple: whatever the wiggling quantity is multiplied by (the input, the wire). Green: sigmoid's steepness where that neuron sits. Red: the gap between output and right answer. The product of all five factors is the slope of the cost for w1.">
        <ChainRippleDiagram />
      </Figure>

      <SectionHeader id="m4-rates" title="Factors are posted rates" />
      <p>
        What a factor is, and why it costs nothing to know, is easiest to see
        far from any network for one beat.
      </p>
      <Aside>
        <p>
          You are paid in euros and support family in Mexico, and two currency
          booths stand between: euros to dollars at 1 to 2, dollars to pesos at
          1 to 3. You get a 5 euro raise. How much more reaches the family? Pass
          the raise through the booths, one at a time: 5 extra euros become 10
          extra dollars, which become 30 extra pesos. Notice what you never
          needed: the salary. And notice that the two rates collapse into one
          number, <M tex="2 \times 3 = 6" /> pesos per euro, the chain's
          through-rate, good for any raise your boss might pick, and computable
          before any raise exists, because the rates are posted on the booth
          windows.
        </p>
        <Figure caption="A 5 euro raise crossing two currency booths. Each booth multiplies the incoming change by its posted rate, and the through-rate, 2 x 3 = 6 pesos per euro, prices any possible raise without a single peso moving. At no point did anyone need to know the salary.">
          <CurrencyDiagram />
        </Figure>
      </Aside>
      <p>
        The nudge's path is this exact situation with five booths. The raise is
        the nudge to <M tex="w_1" />; the far-end currency is the cost; each
        arrow's factor is that booth's rate; and the product of all five is the
        chain's through-rate, cost gained per unit of <M tex="w_1" />, which is
        precisely the number Module 3 called the knob's slope and measured the
        expensive way. So everything turns on one question: are the network's
        rates posted like the booths' rates, readable with no transaction, or
        must they be discovered by actually running the nudge? Posted, all five.
        The chain has three kinds of booth, each posting its rate in its own
        way (the picture colors them to match), and the log is there to certify
        every one.
      </p>
      <RippleSlice arrows={[0, 2]} />
      <p>
        The purple booths are the multiplications, and one subtraction explains
        both of their rates. Take the arrow from <M tex="a_1" /> to{" "}
        <M tex="z_2" />, the stretch where the change crosses the wire from A to
        B, because there the log shows something real happening: <M tex="a_1" />'s wiggle of 0.00235 lands on{" "}
        <M tex="z_2" /> as 0.00470, twice as big. Why exactly twice? The old
        total is <M tex="z_2 = 2.0\,a_1 + b_2" />; the new total is the same
        thing with <M tex="a_1 + 0.00235" /> in place of <M tex="a_1" />.
        Subtract old from new:
      </p>
      <Eq
        tex="\begin{aligned} &\underbrace{2.0\,(a_1 + 0.00235) + b_2}_{\text{new } z_2} - \underbrace{(2.0\,a_1 + b_2)}_{\text{old } z_2} \\[0.8em] &= 2.0 \times 0.00235 = 0.00470 \end{aligned}"
        gloss="The b2's cancel, the 2.0-times-a1 parts cancel, and what remains is the wiggle times the wire. A change riding into a multiplication comes out multiplied: that is the whole mechanism."
      />
      <p>
        So a purple booth's rate is the other partner in the product. The first
        arrow is the same subtraction with the roles swapped: there the wiggling
        thing is <M tex="w_1" />, so the partner is the input <M tex="x" />, and
        our input happens to be 1.0, which is why the nudge passed through
        unchanged. An input of 0.5 would let half through; an input of 0 would
        kill it, and <M tex="w_1" /> would be a knob that changes nothing. Wiggle
        the activation and the rate is the weight; wiggle the weight and the rate
        is the activation. Both rates are posted: the input came with the
        example, and the wire is a knob you can read off.
      </p>
      <RippleSlice arrows={[1, 3]} />
      <p>
        The green booths are the two sigmoids, and you met their kind in
        Module 1's slider. How much a neuron's confidence moves when its evidence moves
        depends on where the neuron currently sits on the curve: eager on the
        steep middle, nearly deaf out on the flat ends. The booth's rate is the
        steepness at that exact spot. Steepness has a shorthand,{" "}
        <M tex="\sigma'(z)" /> (the tick mark is read "prime" and means "the slope
        of"), and a convenient formula:
      </p>
      <Eq
        tex="\sigma'(z) = \sigma(z)\,(1 - \sigma(z))"
        gloss="Sigmoid's slope at z is its own output times one minus its output: confidence times doubt. Biggest at the fence (a quarter, when the output is one half), and nearly zero once the neuron is sure either way."
      />
      <p>
        The formula turns a confidence the log already holds into that booth's
        rate, with no wiggle involved. Neuron A's confidence is{" "}
        <M tex="a_1 = 0.6225" /> (the log's <M tex="a_1" /> box), so its doubt
        is <M tex="1 - 0.6225 = 0.3775" />, and its rate is{" "}
        <M tex="0.6225 \times 0.3775 = 0.235" />: exactly the × 0.235 posted on
        the arrow. Neuron B the same way: confidence <M tex="a_2 = 0.5609" />,
        doubt <M tex="1 - 0.5609 = 0.4391" />, rate 0.246. And a sigmoid booth
        is always a tax: confidence times doubt can never beat a quarter
        (0.5 × 0.5, at the fence), and far out on a flat end it is nearly zero,
        where a nudge dies entirely.
      </p>
      <p>
        Certify both against the log by multiplying each rate by the change
        arriving at its booth: <M tex="0.235 \times 0.0100 = 0.00235" /> (that
        0.0100 is <M tex="z_1" />'s change), exactly <M tex="a_1" />'s change;
        and <M tex="0.246 \times 0.00469 = 0.00115" /> (<M tex="z_2" />'s
        change), matching <M tex="a_2" />'s change up to the rounded last digit.
        If it is the formula itself you distrust, it too is checkable the
        Module 3 way: wiggle <M tex="z" />, divide.
      </p>
      <RippleSlice arrows={[4]} />
      <p>
        The red booth is the cost's, and it is the strange one: it can invert.
        Its rate's size is the size of the current miss: wiggling an output that
        already sits on its target barely changes the squared miss, while the
        same wiggle on a badly wrong output changes it a lot. So the rate is the
        gap itself, posted by the first run's output, and it arrives with a flip
        worth noticing:
        Module 3 wrote gaps as right answer minus output, but the slope comes out
        the other way around, output minus right answer,{" "}
        <M tex="a_2 - y = 0.5609 - 1 = -0.4391" />. The sign carries the
        direction: this output sits below its target, so raising it closes the
        miss and the cost falls, a negative factor. Squaring erases the sign, so
        the cost itself cannot tell the two directions apart, but slopes can, and
        output minus right answer is the convention every later formula uses.
        (The half in Module 3's cost pays off here: squaring would put a factor
        of 2 into this slope, and the half cancels it, leaving the gap alone.)
        Against the log: <M tex="-0.439 \times 0.00116 = -0.00051" />, the cost's
        change exactly.
      </p>
      <p>
        That is all three kinds, and not one rate needed the nudged run: the
        forward pass is what posts them, every confidence, every evidence total,
        and the gap. Multiply the five booths together and the through-rate, the
        slope, falls out:
      </p>
      <Eq
        tex="1.0 \times 0.2350 \times 2.0 \times 0.2463 \times (-0.4391) = -0.0508"
        gloss="w1's slope, computed with no reruns at all: how much the cost moves per unit of nudge to w1, every other knob held still, exactly Module 3's per-knob question."
      />
      <p>
        The nudge measurement said <M tex="-0.050" />, and the product says{" "}
        <M tex="-0.0508" />. They agree, and only the measurement needed a second
        run. The product is also the exact one: the
        nudge method approximates, because it uses a small step where the true
        slope wants a vanishingly small one (the smearing Module 3's aside
        described). The multiply-the-local-slopes rule is
        called the chain rule, and it is the only piece of mathematics
        backpropagation needs.
      </p>

      <SectionHeader id="m4-blame" title="Blame: price the road once" />
      <p>
        That prices one knob, and notice how little of the price is the knob's
        own. Of the five factors, only the first, the entry, is about{" "}
        <M tex="w_1" />; the other four belong to the road, and they would
        multiply any change passing through, whoever sent it. Calling −0.0508{" "}
        <M tex="w_1" />'s slope names the question it answers, not who owns the
        factors: nudge <M tex="w_1" /> alone and the cost moves by this per
        unit. Training needs that question answered for every knob, and the
        chain has four: <M tex="w_1" />, <M tex="b_1" />, <M tex="w_2" />,{" "}
        <M tex="b_2" /> (the digit reader has 11,935). Each knob enters the road
        somewhere, and two knobs entering at the same point share everything
        from there to the cost. So the rest of backpropagation is bookkeeping
        about exactly this ownership: price each stretch of road once, and let
        every knob reuse the price.
      </p>
      <Figure caption="The road with its on-ramps, in the log's visual language: the green bands are the two neurons again, and each knob's ramp joins the road at its neuron's z, the neuron's front door. On the way in, a nudge is multiplied by its private entry factor: its product partner, or 1 for a bias. Everything past the junction is shared, and a shared stretch has one price, the bracketed deltas, which the next paragraphs compute. A knob's slope is its entry factor times the delta where it joins. Notice w2 is in the picture twice: as the road's x 2.0 booth when something else's change passes through it, and as a knob on its own ramp when it is the thing being nudged. Which role a weight plays depends on what is wiggling: the partner rule again.">
        <RoadDiagram />
      </Figure>
      <p>
        Watch it with the two knobs that enter at neuron A. <M tex="w_1" />'s
        change enters at <M tex="z_1" />; so does <M tex="b_1" />'s. The only
        difference is how each arrives: <M tex="w_1" />'s nudge got multiplied by
        the input on the way in, while <M tex="b_1" /> reaches <M tex="z_1" />{" "}
        one-for-one (add 0.01 to <M tex="b_1" /> and <M tex="z_1" /> moves by
        0.01; there is no <M tex="x" /> to multiply by). From <M tex="z_1" />{" "}
        onward, identical booths. Give the shared stretch's through-rate a name,{" "}
        <M tex="T" />, and both slopes fall out of it:
      </p>
      <Eq
        tex="\begin{gathered} T = \underbrace{0.2350 \times 2.0 \times 0.2463 \times (-0.4391)}_{\text{everything from } z_1 \text{ to the cost}} = -0.0508 \\[0.8em] \text{slope for } w_1 = 1.0 \times T, \qquad \text{slope for } b_1 = 1 \times T \end{gathered}"
        gloss="Both knobs live at neuron A, so from z1 onward their ripples are identical. Compute the shared part T once and both slopes are one extra multiplication each."
      />
      <p>
        T was a temporary name; the idea underneath it is permanent. A neuron's{" "}
        <M tex="\delta" /> (the Greek letter delta) is the slope of the cost with
        respect to that neuron's evidence <M tex="z" />: the through-rate of
        everything downstream, all the booths between this neuron and the cost
        priced as one number. Call it the neuron's blame: how much the cost
        cares about this neuron's total. The T you just computed is exactly
        neuron A's blame, and the chain has two neurons, so two blames; neuron
        A's is built from neuron B's:
      </p>
      <Eq
        tex="\begin{aligned} \delta_B &= \underbrace{0.2463}_{\text{own } \sigma'} \times \underbrace{(-0.4391)}_{\text{gap}} = -0.1081 \\[0.8em] \delta_A &= \underbrace{0.2350}_{\text{own } \sigma'} \times \underbrace{2.0}_{\text{wire}} \times \underbrace{(-0.1081)}_{\delta_B} = -0.0508 \end{aligned}"
        gloss="Neuron B touches the cost directly, so its blame is its own steepness times the gap. Neuron A's blame arrives from downstream: delta-B, carried back through the connecting wire, scaled by A's own steepness. Blame flows backward, one cheap step per neuron."
      />
      <p>
        And once a neuron's blame is known, its knobs read their slopes straight
        off it. A bias's slope is the blame itself, because a bias nudges its{" "}
        <M tex="z" /> one-for-one. A weight's slope is the blame times the
        activation arriving on its wire, because that activation is what the weight
        gets multiplied by. All four knobs of the chain:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
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
      </div>
      <p>
        Now count what all four slopes cost. One forward pass, already paid,
        posted every rate. One backward walk priced the two stretches:{" "}
        <M tex="\delta_B" /> first, then <M tex="\delta_A" /> built from it.
        After that, each knob's slope was a single multiplication, entry factor
        times a <M tex="\delta" />. Forward pass, backward sweep of blames,
        slopes read off: that three-step shape is the whole backpropagation
        algorithm, run here on the smallest possible chain. Everything after
        this point is the same three steps with more neurons in them.
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
      <Figure caption="Blame collection at a fork. The hidden neuron feeds two output neurons whose blames are already priced (the red numbers under them). Each red dashed arrow carries its receiver's blame back along the wire, multiplied by the wire's weight, the purple rule read backward. The arrivals add at the neuron, and its own steepness scales the sum: 0.15 x (-0.72) = -0.108, its delta.">
        <ForkDiagram />
      </Figure>

      <SectionHeader id="m4-stepper" title="Watch it run on a network" />
      <p>
        Time to run the same story on a network instead of a chain. The one below
        is Module 1's contrarian network grown by one hidden neuron, 2-3-1 instead
        of 2-2-1, so that no two layers have the same size and every shape stays
        honest. With whole layers in play, one piece of notation: number the
        layers 1, 2, 3 (inputs, hidden, output) and hang the layer number on each
        symbol as a superscript, so <M tex="a^2" /> is the hidden layer's column
        of activations, <M tex="w^2" /> the weight matrix into it, and{" "}
        <M tex="\delta^2" /> its column of blames, one entry per neuron. (The
        chain's subscripts counted single neurons; superscripts count layers, and
        each symbol is a whole column or matrix again.) The
        superscript is a label, never a power; nothing here is squared. Count the
        knobs, like the 11,935:
      </p>
      <Eq
        tex="\underbrace{3 \times 2}_{\text{hidden } w} + \underbrace{3}_{\text{hidden } b} + \underbrace{1 \times 3}_{\text{output } w} + \underbrace{1}_{\text{output } b} = 6 + 3 + 3 + 1 = 13"
        gloss="Thirteen knobs, so the backward pass must deliver thirteen slopes. The weights shown on the wires are a hand-picked starting point, not a trained network: training them is Module 5's job."
      />
      <Figure caption="The network with every symbol placed. Layer-2 symbols are whole columns, one entry per neuron: z² holds the three evidences, a² the three confidences, δ² the three blames. w² and w³ are the wire ledgers, one entry per wire, each row one receiving neuron's incoming wires. The target and cost sit outside the network, keeping score.">
        <NetworkAnatomyDiagram />
      </Figure>
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

      <SectionHeader id="m4-equations" title="The four equations" />
      <p>
        Those six steps are the whole algorithm, and nothing new arrives now: the
        four equations below are facts you have already computed, written in
        Module 2's matrix shorthand so that Module 5 can turn each one into a
        single line of NumPy. The job here is recognition, not derivation. They
        are numbered BP1 to BP4 after Nielsen's Chapter 2, whose framing this
        module adapts, and they map onto the stepper: BP1 ran at step 4, BP2 at
        step 5, BP3 and BP4 together at step 6. They are also written for the
        stepper's own layers, 2 hidden and 3 output, no abstract layer letters.
        Two bookkeeping notes: <M tex="\delta^2" /> is a whole column of blames,
        one entry per neuron of layer 2 (the chain's blames wore neuron names
        instead, <M tex="\delta_A" /> and <M tex="\delta_B" />); and the input
        column doubles as layer 1's activations, <M tex="a^1 = x" />, which is
        what BP4 reads at the hidden layer.
      </p>
      <Eq
        tex="\delta^3 = (a^3 - y) \odot \sigma'(z^3) \tag{BP1}"
        gloss="Blame starts at the output layer: the gap in each output entry, times that neuron's current steepness (the circled dot means multiply matching entries, NumPy's plain *, no adding). Steepness is named by position on the curve, sigma-prime at z3, but sigmoid lets you compute it from the height instead: a3 times (1 minus a3), same number through either door. A saturated neuron has sigma-prime near zero and soaks up almost no blame, even when it is wrong."
      />
      <Eq
        tex="\delta^2 = \big( (w^3)^T \, \delta^3 \big) \odot \sigma'(z^2) \tag{BP2}"
        gloss="Blame flows backward: each hidden neuron collects blame along its outgoing wires and adds, then scales by its own steepness. The raised T (transpose) regroups Module 2's wire ledger by sender instead of receiver, so each row becomes one neuron's outgoing wires, exactly what collecting needs; the shapes agree by the inner-numbers-touch rule: (3, 1) times (1, 1) gives (3, 1), one collected blame per hidden neuron."
      />
      <Eq
        tex="\frac{\partial C}{\partial b^2} = \delta^2, \qquad \frac{\partial C}{\partial b^3} = \delta^3 \tag{BP3}"
        gloss="The curly-d expression is read as one name, 'the slope of C per nudge of b', the number Module 3 measured with nudge-and-divide. Every bias's slope is exactly its neuron's blame: a bias's ramp factor is 1."
      />
      <Eq
        tex="\frac{\partial C}{\partial w^2} = \delta^2 \, (a^1)^T, \qquad \frac{\partial C}{\partial w^3} = \delta^3 \, (a^2)^T \tag{BP4}"
        gloss="Every weight's slope is the receiving neuron's blame times the activation its wire carried: entry factor times the delta at the junction, for all wires at once. The shapes tell the story: delta-2 is (3, 1), the transposed input column is (1, 2), and their product is (3, 2), one slope per weight, in exactly the shape of w2; the entry in row j, column k is delta-j times a-k."
      />
      <p>
        And each equation is a receipt for work already done, twice over: once
        in the chain, by hand, and once in the stepper you just walked (its
        numbers are still on screen above; the BP2 and BP4 entries follow{" "}
        <M tex="h_1" />, whose activation is 0.818 and whose steepness is
        0.818 × 0.182 = 0.149):
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
        <thead>
          <tr>
            <th>equation</th>
            <th>in road words</th>
            <th>in the chain, by hand</th>
            <th>in the stepper, steps 4 to 6</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>BP1</td>
            <td>the last stretch's price: own steepness × the gap</td>
            <td><M tex="\delta_B = 0.2463 \times (-0.4391) = -0.1081" /></td>
            <td><M tex="\delta^3 = (0.593 - 1) \times 0.241 = -0.098" /></td>
          </tr>
          <tr>
            <td>BP2</td>
            <td>collect along outgoing wires, add, × own steepness</td>
            <td><M tex="\delta_A = 0.2350 \times 2.0 \times (-0.1081) = -0.0508" /></td>
            <td><M tex="4.0 \times (-0.098) \times 0.149 \approx -0.059" /></td>
          </tr>
          <tr>
            <td>BP3</td>
            <td>a bias's ramp factor is 1</td>
            <td><M tex="\text{slope for } b_2 = \delta_B" /></td>
            <td><M tex="\text{slope for } b^3 = \delta^3 = -0.098" /></td>
          </tr>
          <tr>
            <td>BP4</td>
            <td>entry factor × the δ at the junction</td>
            <td><M tex="\text{slope for } w_2 = 0.6225 \times (-0.1081) = -0.0673" /></td>
            <td><M tex="h_1\text{'s wire: } 0.818 \times (-0.098) \approx -0.080" /></td>
          </tr>
        </tbody>
        </table>
      </div>
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
        Two reading habits, because Module 5's debugging leans on both.
        Direction: the equations run right to left along the network, BP1 once
        at the output, then BP2 once per remaining layer, each use producing the
        layer to its left. A deeper network just uses BP2 more times; Nielsen
        writes that repetition with a general layer letter where this page wrote
        2 and 3, and in Module 5 it becomes a loop. Signs: a blame's sign is
        advice, negative meaning raising this neuron's evidence would lower the
        cost; the gap's sign says which side of its target an output sits on;
        and a negative wire flips blame in transit, which you watched{" "}
        <M tex="h_2" />'s wire do in the stepper.
      </p>

      <SectionHeader id="m4-quiz" title="Three predictions" />
      <p>
        Module 5 is where you implement all four equations, and the implementation
        goes better if you can already think in them. So before moving on, three
        predictions. Commit to an answer before you click, and verify each one
        afterward by actually doing it in the stepper above (the questions describe
        the stepper's starting weights; press Reset weights first if you have been
        experimenting).
      </p>
      <DeltaQuiz />

      <SectionHeader id="m4-bill" title="The bill, closed" />
      <p>
        The bill, one last time, at the digit reader's scale. Nudge-measured
        gradients pay two cost measurements per knob per step: 23,870 passes over
        the batch to walk downhill once. The four equations pay one forward pass,
        which also posts every rate, plus one backward sweep that touches each
        neuron and each weight about once, roughly the price of a second pass. Two
        passes against 23,870 of them: about twelve thousand times cheaper, and
        exact instead of estimated. That is why backpropagation, and not a faster computer, is
        what made neural networks trainable. In Module 5 you implement BP1 through
        BP4 in NumPy, and the nudge method takes on a new job: the course will
        nudge-and-measure every knob of a tiny network the slow way and compare
        your backward pass against it, number by number. When the two agree, your
        gradients are right, and the real training run begins.
      </p>

      <Recap
        items={[
          "Backprop answers 'which way should each knob move' with one forward pass and one backward sweep of blame, instead of two cost measurements per knob, and its answers are exact.",
          "A nudge's effect on the cost is the product of the local factors along its path (the chain rule); the forward pass posts every factor, and a neuron's blame delta is the downstream stretch priced once per neuron.",
          "The four equations: blame starts at the output as gap times sigma-prime (BP1), flows backward through the transposed wires (BP2), every bias's slope is its neuron's blame (BP3), and every weight's slope is blame times the activation its wire carried (BP4).",
          "A saturated neuron has sigma-prime near zero and soaks up almost no blame even when badly wrong: quadratic cost's weakness at the output layer, and the problem Module 7's cross-entropy cost exists to solve.",
        ]}
        chapter="Chapter 2 (how the backpropagation algorithm works)"
        href="http://neuralnetworksanddeeplearning.com/chap2.html"
      />
    </article>
  );
}

// Static diagram: the currency chain used to introduce factors as posted
// exchange rates. Reuses the ripple diagram's visual language so the two
// pictures read as the same kind of object.
function CurrencyDiagram() {
  const boxes = [
    { title: "the raise", value: "+5 euros" },
    { title: "in dollars", value: "+10" },
    { title: "at the far end", value: "+30 pesos" },
  ];
  const rates = [
    { f: "× 2", why: "posted: 1 € = 2 $" },
    { f: "× 3", why: "posted: 1 $ = 3 pesos" },
  ];
  const BW = 150;
  const GAP = 80;
  const X0 = 85;
  const Y0 = 40;
  const BH = 52;
  return (
    <svg {...fig(76, 7, 627, 146)} className="chain-ripple" role="img"
         aria-label="A 5 euro raise passes through two currency booths, times 2 into dollars and times 3 into pesos, arriving as 30 extra pesos; the through-rate is 6 pesos per euro">
      {boxes.map((b, i) => {
        const x = X0 + i * (BW + GAP);
        return (
          <g key={i}>
            <rect x={x} y={Y0} width={BW} height={BH} rx={6} className="ripple-box" />
            <text x={x + BW / 2} y={Y0 + 21} textAnchor="middle" className="ripple-title">
              {b.title}
            </text>
            <text x={x + BW / 2} y={Y0 + 41} textAnchor="middle" className="ripple-change">
              {b.value}
            </text>
          </g>
        );
      })}
      {rates.map((r, i) => {
        const x1 = X0 + BW + i * (BW + GAP);
        const x2 = x1 + GAP;
        const y = Y0 + BH / 2;
        return (
          <g key={i}>
            <line x1={x1 + 4} y1={y} x2={x2 - 8} y2={y} className="ripple-arrow" markerEnd="url(#currency-head)" />
            <text x={(x1 + x2) / 2} y={Y0 - 12} textAnchor="middle"
                  className="ripple-factor ripple-kind-mult">
              {r.f}
            </text>
            <text x={(x1 + x2) / 2} y={Y0 + BH + 22} textAnchor="middle"
                  className="ripple-why ripple-kind-mult">
              {r.why}
            </text>
          </g>
        );
      })}
      <defs>
        <marker id="currency-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6"
                markerHeight="6" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" className="ripple-head" />
        </marker>
      </defs>
      <text x={390} y={142} textAnchor="middle" className="ripple-product">
        through-rate: 2 × 3 = 6 pesos per euro, posted before any raise exists
      </text>
    </svg>
  );
}

// Static diagram: the shared road from z1 to the cost with the four knobs as
// on-ramps, and the two priced stretches (the deltas) bracketed above it.
// Draws the blame section's whole idea: entry factors are private, the road
// is shared, and a shared stretch has one price.
function RoadDiagram() {
  const stops = ["z₁", "a₁", "z₂", "a₂", "cost C"];
  const roadFactors = [
    { f: "× 0.235", kind: "sig" },
    { f: "× 2.0", kind: "mult" },
    { f: "× 0.246", kind: "sig" },
    { f: "× −0.439", kind: "gap" },
  ];
  const BW = 74;
  const GAP = 56;
  const X0 = 44;
  const ROAD_Y = 88; // road box top
  const BH = 34;
  const left = (i: number) => X0 + i * (BW + GAP);
  const cx = (i: number) => left(i) + BW / 2;
  const bracket = (x1: number, x2: number, y: number, label: string) => (
    <g key={label}>
      <path d={`M ${x1} ${y + 6} L ${x1} ${y} L ${x2} ${y} L ${x2} ${y + 6}`} className="road-delta" />
      <text x={(x1 + x2) / 2} y={y - 6} textAnchor="middle" className="ripple-factor ripple-kind-gap">
        {label}
      </text>
    </g>
  );
  // knobs: [label, pill center x, junction stop index, ramp offset, entry factor, factor note]
  const knobs: [string, number, number, number, string, string][] = [
    ["w₁", 40, 0, -12, "× 1.0", "the input"],
    ["b₁", 124, 0, 12, "× 1", ""],
    ["w₂", 300, 2, -12, "× 0.6225", "a₁"],
    ["b₂", 384, 2, 12, "× 1", ""],
  ];
  return (
    <svg {...fig(-25, -1, 683, 261)} className="chain-ripple" role="img"
         aria-label="The road from z1 through a1, z2, a2 to the cost, with knobs w1 and b1 entering at z1 and w2 and b2 entering at z2; delta-A prices the road from z1, delta-B from z2. Green bands group z1 and a1 inside neuron A and z2 and a2 inside neuron B.">
      <defs>
        <marker id="road-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6"
                markerHeight="6" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" className="ripple-head" />
        </marker>
      </defs>
      {bracket(left(0), left(4) + BW, 26, "δ_A = −0.0508: the road from z₁, priced")}
      {bracket(left(2), left(4) + BW, 52, "δ_B = −0.1081: from z₂")}
      {/* neuron ownership, matching the ripple log's bands */}
      {[{ i: 0, label: "inside neuron A" }, { i: 2, label: "inside neuron B" }].map((band) => (
        <g key={band.label}>
          <rect x={left(band.i) - 6} y={ROAD_Y - 6} width={2 * BW + GAP + 12} height={BH + 12}
                rx={9} className="ripple-neuron-band" />
          <text x={left(band.i) + BW + GAP / 2} y={ROAD_Y - 12} textAnchor="middle"
                className="ripple-band-label">
            {band.label}
          </text>
        </g>
      ))}
      <text x={cx(4)} y={ROAD_Y + BH + 34} textAnchor="middle" className="ripple-why">
        the scorekeeper
      </text>
      {stops.map((s, i) => (
        <g key={s}>
          <rect x={left(i)} y={ROAD_Y} width={BW} height={BH} rx={6} className="ripple-box" />
          <text x={cx(i)} y={ROAD_Y + 22} textAnchor="middle" className="ripple-title">
            {s}
          </text>
        </g>
      ))}
      {roadFactors.map((f, i) => (
        <g key={i}>
          <line x1={left(i) + BW + 2} y1={ROAD_Y + BH / 2} x2={left(i + 1) - 6} y2={ROAD_Y + BH / 2}
                className="ripple-arrow" markerEnd="url(#road-head)" />
          <text x={(left(i) + BW + left(i + 1)) / 2} y={ROAD_Y + BH + 16} textAnchor="middle"
                className={`ripple-why ripple-kind-${f.kind}`}>
            {f.f}
          </text>
        </g>
      ))}
      {knobs.map(([name, px, stop, off, factor, note]) => {
        const tx = cx(stop) + off;
        const midX = (px + tx) / 2;
        const outside = off < 0; // left ramps label to the left, right ramps to the right
        const anchor = outside ? "end" : "start";
        const lx = outside ? midX - 8 : midX + 8;
        return (
          <g key={name}>
            <rect x={px - 30} y={186} width={60} height={26} rx={13} className="ripple-box" />
            <text x={px} y={203} textAnchor="middle" className="ripple-title">{name}</text>
            <line x1={px} y1={186} x2={tx} y2={ROAD_Y + BH + 24} className="ripple-arrow"
                  markerEnd="url(#road-head)" />
            <text x={lx} y={162} textAnchor={anchor} className="ripple-factor ripple-kind-mult">
              {factor}
            </text>
            {note && (
              <text x={lx} y={176} textAnchor={anchor} className="ripple-why ripple-kind-mult">
                ({note})
              </text>
            )}
          </g>
        );
      })}
      <text x={350} y={248} textAnchor="middle" className="ripple-product">
        a knob's slope = its ramp's entry factor × the δ where it joins
      </text>
    </svg>
  );
}

// Static anatomy chart of the stepper's 2-3-1 network: every symbol from the
// layer notation placed on the picture, so the stepper's equation cards have
// a map. Unicode superscripts stand in for the KaTeX forms.
function NetworkAnatomyDiagram() {
  const IN = [{ x: 110, y: 105 }, { x: 110, y: 195 }];
  const HID = [{ x: 330, y: 65 }, { x: 330, y: 150 }, { x: 330, y: 235 }];
  const OUT = { x: 550, y: 150 };
  const R = 22;
  const trim = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    return { x1: a.x + (dx / len) * R, y1: a.y + (dy / len) * R, x2: b.x - (dx / len) * R, y2: b.y - (dy / len) * R };
  };
  const colLabels = (
    x: number,
    rows: string[],
  ) => rows.map((t, i) => (
    <text key={i} x={x} y={286 + i * 16} textAnchor="middle"
          className={i === 0 ? "bp-name" : "ripple-why"}>
      {t}
    </text>
  ));
  return (
    <svg {...fig(50, 4, 673, 325)} className="chain-net" role="img"
         aria-label="The 2-3-1 network labeled: layer 1 is the input column a1 equals x; layer 2 has three neurons with columns z2, a2, delta2 and biases b2; layer 3 has one neuron with z3, a3, delta3 and bias b3; w2 and w3 are the wire ledgers; the target y and cost C sit outside the network">
      <text x={110} y={22} textAnchor="middle" className="ripple-band-label">layer 1</text>
      <text x={330} y={22} textAnchor="middle" className="ripple-band-label">layer 2</text>
      <text x={550} y={22} textAnchor="middle" className="ripple-band-label">layer 3</text>

      {HID.map((h, j) =>
        IN.map((inp, k) => {
          const t = trim(inp, h);
          return <line key={`w2-${j}${k}`} {...t} className="tiny-net-edge" />;
        }),
      )}
      {HID.map((h, k) => {
        const t = trim(h, OUT);
        return <line key={`w3-${k}`} {...t} className="tiny-net-edge" />;
      })}
      <text x={220} y={40} textAnchor="middle" className="ripple-factor ripple-kind-mult">
        w², the wire ledger
      </text>
      <text x={220} y={55} textAnchor="middle" className="ripple-why ripple-kind-mult">
        (3, 2): one entry per wire
      </text>
      <text x={440} y={40} textAnchor="middle" className="ripple-factor ripple-kind-mult">
        w³
      </text>
      <text x={440} y={55} textAnchor="middle" className="ripple-why ripple-kind-mult">
        (1, 3)
      </text>

      {IN.map((p, i) => (
        <g key={`in${i}`}>
          <circle cx={p.x} cy={p.y} r={R} className="tiny-net-input" />
          <text x={p.x} y={p.y + 5} textAnchor="middle" className="tiny-net-label">
            {i === 0 ? "x₁" : "x₂"}
          </text>
        </g>
      ))}
      {HID.map((p, i) => (
        <g key={`h${i}`}>
          <circle cx={p.x} cy={p.y} r={R} className="tiny-net-neuron" />
          <text x={p.x} y={p.y + 5} textAnchor="middle" className="tiny-net-label">
            {["h₁", "h₂", "h₃"][i]}
          </text>
        </g>
      ))}
      <circle cx={OUT.x} cy={OUT.y} r={R} className="tiny-net-neuron" />
      <text x={OUT.x} y={OUT.y + 5} textAnchor="middle" className="tiny-net-label">out</text>

      <line x1={OUT.x + R} y1={OUT.y} x2={618} y2={OUT.y} className="tiny-net-edge" />
      <text x={626} y={144} className="bp-name">y = 1, the target</text>
      <text x={626} y={164} className="bp-cost">C = ½ (y − a³)²</text>
      <text x={626} y={182} className="ripple-why">the scorekeeper</text>

      {colLabels(110, ["the input column", "a¹ = x, shape (2, 1)", "numbers, not neurons"])}
      {colLabels(330, ["three neurons", "z², a², δ²: columns, (3, 1)", "biases b², (3, 1)"])}
      {colLabels(550, ["one neuron", "z³, a³, δ³: (1, 1)", "bias b³, (1, 1)"])}
    </svg>
  );
}

// Static diagram: blame collecting at a fork. One hidden neuron feeds two
// output neurons whose blames are already priced; blame flows backward along
// each wire multiplied by the wire's weight, the arrivals add, and the
// neuron's own steepness scales the sum. The upgrade BP2 generalizes.
function ForkDiagram() {
  return (
    <svg {...fig(47, 17, 518, 229)} className="chain-net" role="img"
         aria-label="A hidden neuron feeds two output neurons with blames −0.10 and +0.08 through wires of weight 4.0 and −4.0; blame flows backward along each wire multiplied by its weight, the arrivals −0.40 and −0.32 add to −0.72, and times the neuron's own steepness 0.15 gives δ = −0.108">
      <defs>
        <marker id="fork-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6"
                markerHeight="6" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" className="fork-head" />
        </marker>
      </defs>

      {/* wires, forward */}
      <line x1={154} y1={107} x2={425} y2={57} className="tiny-net-edge" />
      <line x1={154} y1={117} x2={425} y2={167} className="tiny-net-edge" />
      {/* Above their own wires, not on them: at x = 290 the upper wire is at
          y 82 and the lower at 142, and a baseline of 79 or 149 put the line
          through the digits, which read "w = =4:0". */}
      <text x={290} y={76} textAnchor="middle" className="bp-wlabel">w = 4.0</text>
      <text x={290} y={134} textAnchor="middle" className="bp-wlabel">w = −4.0</text>

      {/* blame, backward */}
      <line x1={420} y1={38} x2={168} y2={86} className="fork-blame" markerEnd="url(#fork-head)" />
      <line x1={420} y1={186} x2={168} y2={138} className="fork-blame" markerEnd="url(#fork-head)" />
      <text x={294} y={44} textAnchor="middle" className="bp-delta">−0.10 × 4.0 = −0.40</text>
      <text x={294} y={200} textAnchor="middle" className="bp-delta">0.08 × (−4.0) = −0.32</text>

      {/* the hidden neuron */}
      <text x={130} y={70} textAnchor="middle" className="bp-name">a hidden neuron</text>
      <circle cx={130} cy={112} r={26} className="tiny-net-neuron" />
      <text x={130} y={117} textAnchor="middle" className="tiny-net-label">h</text>
      <text x={130} y={162} textAnchor="middle" className="bp-delta">−0.40 − 0.32 = −0.72</text>
      <text x={130} y={180} textAnchor="middle" className="bp-delta">× own σ′ 0.15 → δ = −0.108</text>

      {/* the two receivers, blames already priced */}
      <circle cx={450} cy={52} r={26} className="tiny-net-neuron" />
      <text x={450} y={57} textAnchor="middle" className="tiny-net-label">out₁</text>
      <text x={506} y={57} className="bp-delta">δ = −0.10</text>
      <circle cx={450} cy={172} r={26} className="tiny-net-neuron" />
      <text x={450} y={177} textAnchor="middle" className="tiny-net-label">out₂</text>
      <text x={506} y={177} className="bp-delta">δ = +0.08</text>

      <text x={320} y={234} textAnchor="middle" className="ripple-product">
        rates along a chain multiply; rates of parallel routes add
      </text>
    </svg>
  );
}

// Static wiring diagram of the two-neuron chain worked in the prose, in the
// visual language of Module 1's TinyNetDiagram: gray input, green neurons,
// every knob labeled in place.
function ChainNetDiagram() {
  const IN = { x: 70, y: 95 };
  const A = { x: 230, y: 95 };
  const B = { x: 390, y: 95 };
  const R = 22;
  const wire = (x1: number, x2: number, key: string) => (
    <line
      key={key}
      x1={x1} y1={95} x2={x2} y2={95}
      className="tiny-net-edge" markerEnd="url(#chain-arrow)"
    />
  );
  return (
    <svg {...fig(35, 29, 533, 122)} className="chain-net" role="img"
         aria-label="A chain: input x feeds neuron A through weight w1, neuron A feeds neuron B through weight w2, and neuron B's confidence is the answer, compared against the target y = 1">
      <defs>
        <marker id="chain-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7"
                markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 8 4 L 0 8 z" className="tiny-net-arrowhead" />
        </marker>
      </defs>
      {wire(IN.x + R, A.x - R - 6, "xa")}
      {wire(A.x + R, B.x - R - 6, "ab")}
      {wire(B.x + R, 452, "bout")}
      <text x={(IN.x + A.x) / 2} y={86} textAnchor="middle" className="bp-wlabel">
        w₁ = 1.0
      </text>
      <text x={(A.x + B.x) / 2} y={86} textAnchor="middle" className="bp-wlabel">
        w₂ = 2.0
      </text>

      <circle cx={IN.x} cy={IN.y} r={R} className="tiny-net-input" />
      <text x={IN.x} y={IN.y + 5} textAnchor="middle" className="tiny-net-label">x</text>
      <text x={IN.x} y={48} textAnchor="middle" className="tiny-net-caption">the input</text>
      <text x={IN.x} y={140} textAnchor="middle" className="tiny-net-caption">= 1.0</text>

      <circle cx={A.x} cy={A.y} r={R} className="tiny-net-neuron" />
      <text x={A.x} y={A.y + 5} textAnchor="middle" className="tiny-net-label">A</text>
      <text x={A.x} y={48} textAnchor="middle" className="tiny-net-caption">computes z₁, reports a₁</text>
      <text x={A.x} y={140} textAnchor="middle" className="tiny-net-caption">b₁ = −0.5</text>

      <circle cx={B.x} cy={B.y} r={R} className="tiny-net-neuron" />
      <text x={B.x} y={B.y + 5} textAnchor="middle" className="tiny-net-label">B</text>
      <text x={B.x} y={48} textAnchor="middle" className="tiny-net-caption">computes z₂, reports a₂</text>
      <text x={B.x} y={140} textAnchor="middle" className="tiny-net-caption">b₂ = −1.0</text>

      <text x={462} y={90} className="tiny-net-label">a₂, the answer</text>
      <text x={462} y={112} className="tiny-net-caption">should be y = 1</text>
    </svg>
  );
}

// The logged ripple of the +0.01 nudge to w1, shared by the full diagram and
// the per-kind recall slices. Each box is a quantity's value before and after
// the nudge (change in parentheses); each arrow carries the local factor.
const RIPPLE_BOXES = [
  { title: "nudge w₁", value: "1.00 → 1.01", change: "(+0.01000)" },
  { title: "z₁", value: "0.5000 → 0.5100", change: "(+0.01000)" },
  { title: "a₁", value: "0.6225 → 0.6248", change: "(+0.00235)" },
  { title: "z₂", value: "0.2449 → 0.2496", change: "(+0.00469)" },
  { title: "a₂", value: "0.5609 → 0.5621", change: "(+0.00116)" },
  { title: "cost C", value: "0.09639 → 0.09589", change: "(−0.00051)" },
];
// kind groups the arrows for the prose's three explanations:
// mult = a multiplication passes the change through scaled,
// sig = sigmoid's steepness where the neuron sits, gap = the current miss.
const RIPPLE_FACTORS = [
  { f: "× 1.0", why: "the input x", kind: "mult" },
  { f: "× 0.235", why: "σ′ at z₁", kind: "sig" },
  { f: "× 2.0", why: "the wire w₂", kind: "mult" },
  { f: "× 0.246", why: "σ′ at z₂", kind: "sig" },
  { f: "× −0.439", why: "the gap", kind: "gap" },
];

// The log's box geometry, shared with the recall strips below, so a strip is
// the same drawing at the same size rather than a redraw of it. 116 rather
// than 110 because the widest logged value, "0.09639 → 0.09589", measured 107
// units and left a letter almost touching each edge; the arrow gap gives back
// what the boxes take, so the diagram's width is unchanged.
const RIPPLE_BW = 116;
const RIPPLE_GAP = 20;
const RIPPLE_BH = 62;

// A recall strip: re-renders just the log segments a kind-beat discusses
// (arrow i connects box i to box i+1), so the reader never scrolls back to
// the full diagram.
function RippleSlice({ arrows }: { arrows: number[] }) {
  const BW = RIPPLE_BW;
  const GAP = RIPPLE_GAP;
  const SEP = 40; // space between segments
  const X0 = 14;
  const Y0 = 44;
  const BH = RIPPLE_BH;
  const SEG = BW + GAP + BW;
  const W = X0 * 2 + arrows.length * SEG + (arrows.length - 1) * SEP;
  const markerId = `slice-arrow-${arrows.join("_")}`;
  const box = (b: (typeof RIPPLE_BOXES)[number], x: number, key: string) => (
    <g key={key}>
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
  return (
    <div className="figure-scroll scroll-x" tabIndex={0}>
    <svg {...fig(5, -3, W - 11, 142)} className="ripple-slice" role="img"
         aria-label={`Recall from the log: ${arrows
           .map((i) => `${RIPPLE_BOXES[i].title} to ${RIPPLE_BOXES[i + 1].title}, factor ${RIPPLE_FACTORS[i].f}`)
           .join("; ")}`}>
      <defs>
        <marker id={markerId} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6"
                markerHeight="6" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" className="ripple-head" />
        </marker>
      </defs>
      <text x={X0} y={16} className="ripple-why">from the log:</text>
      {arrows.map((ai, si) => {
        const xL = X0 + si * (SEG + SEP);
        const x1 = xL + BW;
        const x2 = x1 + GAP;
        const f = RIPPLE_FACTORS[ai];
        return (
          <g key={ai}>
            {box(RIPPLE_BOXES[ai], xL, `l${ai}`)}
            <line x1={x1 + 2} y1={Y0 + BH / 2} x2={x2 - 6} y2={Y0 + BH / 2}
                  className="ripple-arrow" markerEnd={`url(#${markerId})`} />
            <text x={(x1 + x2) / 2} y={Y0 - 12} textAnchor="middle"
                  className={`ripple-factor ripple-kind-${f.kind}`}>
              {f.f}
            </text>
            <text x={(x1 + x2) / 2} y={Y0 + BH + 22} textAnchor="middle"
                  className={`ripple-why ripple-kind-${f.kind}`}>
              {f.why}
            </text>
            {box(RIPPLE_BOXES[ai + 1], x2, `r${ai}`)}
          </g>
        );
      })}
    </svg>
    </div>
  );
}

// Static diagram: the full logged ripple.
function ChainRippleDiagram() {
  const boxes = RIPPLE_BOXES;
  const factors = RIPPLE_FACTORS;
  const BW = RIPPLE_BW;
  const GAP = RIPPLE_GAP;
  const X0 = 12;
  const Y0 = 44; // box top
  const BH = RIPPLE_BH;
  const bandLeft = (i: number) => X0 + i * (BW + GAP); // band spans boxes i, i+1
  return (
    <svg {...fig(0, 9, 817, 205)} className="chain-ripple" role="img"
         aria-label="A +0.01 nudge to w1 ripples through z1, a1, z2, a2 and the cost; each arrow multiplies the change by a local factor, and the product of all five factors is the slope, −0.0508. Green bands group z1 and a1 inside neuron A and z2 and a2 inside neuron B.">
      {/* which boxes live inside which neuron: the wiring diagram's circles,
          unrolled */}
      {[{ i: 1, label: "inside neuron A" }, { i: 3, label: "inside neuron B" }].map((band) => (
        <g key={band.label}>
          <rect x={bandLeft(band.i) - 6} y={Y0 - 6} width={2 * BW + GAP + 12} height={BH + 12}
                rx={9} className="ripple-neuron-band" />
          <text x={bandLeft(band.i) + BW + GAP / 2} y={Y0 + BH + 40} textAnchor="middle"
                className="ripple-band-label">
            {band.label}
          </text>
        </g>
      ))}
      <text x={X0 + BW / 2} y={Y0 + BH + 40} textAnchor="middle" className="ripple-why">
        the knob you moved
      </text>
      <text x={X0 + 5 * (BW + GAP) + BW / 2} y={Y0 + BH + 40} textAnchor="middle" className="ripple-why">
        the scorekeeper
      </text>
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
            <text x={(x1 + x2) / 2} y={Y0 - 14} textAnchor="middle"
                  className={`ripple-factor ripple-kind-${f.kind}`}>
              {f.f}
            </text>
            <text x={(x1 + x2) / 2} y={Y0 + BH + 22} textAnchor="middle"
                  className={`ripple-why ripple-kind-${f.kind}`}>
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
      <text x={406} y={202} textAnchor="middle" className="ripple-product">
        slope for w₁ = 1.0 × 0.235 × 2.0 × 0.246 × (−0.439) = −0.0508
      </text>
    </svg>
  );
}
