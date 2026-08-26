import { AfterThis, Aside, Figure, ModuleToc, Recap, SectionHeader } from "../../components/ModuleBits";
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
      <Figure caption="Where backprop goes. Your sgd from Module 3 asks one question per step and does not care who answers it; the answer is a slope for every knob. Two machines can answer. The dashed one is what you have been using: correct, and priced per knob, because each knob needs its own pair of rescores. The solid one is this module's exercise, priced per pass instead: every knob's slope falls out of one trip forward and one sweep back.">
        <PipelineDiagram />
      </Figure>

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
      <Aside>
        <p>
          Module 2 banned Python loops over neurons, and averaging per-example
          slopes means looping over examples, so it is worth saying which loops
          that ban covers. The objection there was scale: a loop over the
          hundreds of neurons inside a layer spends hundreds of interpreter
          steps on work one matrix product does in a single call. A mini-batch
          is ten examples, so this loop spends ten steps, and each step still
          hands a whole layer's multiplication to the same fast numerical code.
          A faster arrangement does exist, and production code uses it: stack
          the batch as columns, run the four equations on whole matrices, and
          have BP3 add along the batch instead of reading the blame column
          directly. It returns the same gradients to the last decimal place.
          One example per call is the version that matches the four equations
          one for one, and that is the version worth writing by hand.
        </p>
      </Aside>
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
        Each equation becomes one NumPy statement.
      </p>
      <p>
        Shapes are what make those statements concrete, so here is the network
        they are aimed at: the digit reader this module trains. 784 inputs, one
        per pixel of a 28-by-28 image; one hidden layer of 30 neurons; 10
        outputs, one per digit. Module 2 drew this same three-layer shape with
        15 hidden neurons, few enough for its circles to fit in a diagram, and
        named 30 as the classic setup for reading digits. Either size is a free
        choice: more pattern-detectors against more parameters to train.
      </p>
      <p>
        The size fixes the two weight matrices: (30, 784) into the hidden layer,
        (10, 30) into the output, receiving layer named first as in Module 2.
        Choosing differently would change no symbol in the four equations, only
        these numbers. Two names in the table below are the ones your{" "}
        <code>sgd</code> already takes: <code>nabla_b</code> for the biases'
        slopes and <code>nabla_w</code> for the weights', the two halves of
        Module 3's <M tex="\nabla C" />. Read the right-hand column for how
        often each equation runs, because that differs across the four:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
        <thead>
          <tr>
            <th>equation</th>
            <th>in NumPy</th>
            <th>where it runs, and the shapes it makes</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>BP1</td>
            <td><code>delta = (a - y) * sigmoid_prime(z)</code></td>
            <td>once, at the output layer: delta is (10, 1), one blame per output neuron</td>
          </tr>
          <tr>
            <td>BP2</td>
            <td><code>delta = (w_next.T @ delta) * sigmoid_prime(z)</code></td>
            <td>once per earlier layer: <code>.T</code> makes (10, 30) into (30, 10), and @ (10, 1) gives (30, 1)</td>
          </tr>
          <tr>
            <td>BP3</td>
            <td><code>nabla_b[l] = delta</code></td>
            <td>once per layer: (10, 1) then (30, 1), exactly the biases' shapes</td>
          </tr>
          <tr>
            <td>BP4</td>
            <td><code>nabla_w[l] = delta @ a_prev.T</code></td>
            <td>once per layer: (10, 30) then (30, 784), the weights' own shapes</td>
          </tr>
        </tbody>
        </table>
      </div>

      <p>
        Add that column up, because the four equations do not produce four
        numbers. Two of them produce nothing you keep: BP1 and BP2 build{" "}
        <M tex="\delta" />, one blame per neuron, and that column lives inside
        the function and is gone once it returns. The other two are the
        answers, and they run once per layer, so this network's two layers
        fill four arrays. Nothing gets combined at the end; the loop fills the
        slots as it walks. Here is everything that comes back:
      </p>
      <Figure caption="What backprop returns, beside what it is the gradient of. Each slope array has the shape of the parameter array directly above it, so the two lists pair up slot for slot: nabla_w[0] against weights[0], and so on. Count the entries and the gradient holds one number for every knob in the network.">
        <GradientShapeDiagram />
      </Figure>
      <p>
        Matching those shapes is not tidiness, it is what lets your sgd run
        unchanged. Its update subtracts entry by entry,{" "}
        <code>w - eta * nabla_w</code>, so every slope has to sit in the same
        spot as the knob it belongs to. What backprop returns is a
        slope-shaped shadow of the network: same layers, same shapes, one
        number per knob.
      </p>

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
      <Figure caption="What the forward pass stores, for a network of two layers (like the 2-3-1 fixture in the tests, or the digit reader). Each box shows its entry's positive index on top and its negative index below: both name the same slot. BP1 reads zs[-1]; BP4 at the output layer reads activations[-2].">
        <ReceiptsDiagram />
      </Figure>

      <p>
        Reading those lists is one thing; walking them is the other. The
        backward sweep visits one layer per pass. BP1 starts the walk at the
        output layer, and every later pass is BP2 carrying the blame one layer
        further back. What makes the code fiddly is that each pass reads the
        same slots as the pass before, one step further left: the{" "}
        <M tex="z" /> of the layer being blamed, and the activations that fed
        that layer. Two passes finish the digit reader; a deeper network
        repeats the second panel with everything shifted left again.
      </p>
      <Figure caption="The backward walk over the same receipts, pass by pass. Shaded slots are the ones that pass reads. The window slides one box left each pass, which is the whole motion of the loop: get it wrong by one and the tests say so, because the sigmoid steepness would come from the wrong layer's z.">
        <BackwardWalkDiagram />
      </Figure>

      <SectionHeader id="m5-exercise" title="Write backprop" />
      <ExercisePage exercise={backpropExercise} />

      <SectionHeader id="m5-train" title="The real training run" />
      <p>
        The network is the digit reader the shapes table set up: 784 pixels in,
        30 hidden neurons, 10 digit outputs, and the 23,860 knobs counted with
        its gradient above. That is almost exactly double Module 2's 11,935,
        because doubling the hidden layer doubles everything except the ten
        output biases. At the nudge method's price of two rescores per knob,
        one step of descent here would cost 47,720 passes over the mini-batch.
        Your backward sweep replaces all of them.
      </p>
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
          "The digit reader trains in seconds where nudge-measured gradients would need hours: same descent, same cost, same data, cheaper slopes.",
          "You have now written a sigmoid neuron, a feedforward pass, stochastic gradient descent and backpropagation, and together they read handwritten digits. Module 6 steps away from training to ask what a network of sigmoid neurons can represent at all.",
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

// Static diagram: where backprop sits in the machinery the learner already
// owns. sgd asks one question per step; two implementations can answer it,
// one priced per knob (the nudge method) and one priced per pass (backprop).
// No numbers here on purpose: the bill is derived later, in m5-train.
function PipelineDiagram() {
  const SGD = { x: 24, y: 74, w: 166, h: 84 };
  const FORK = 358; // where the two answers branch
  const IMPL = { x: 388, w: 400, h: 84 };
  const TOP_Y = 20;
  const BOT_Y = 144;
  const topMid = TOP_Y + IMPL.h / 2;
  const botMid = BOT_Y + IMPL.h / 2;
  return (
    <svg viewBox="0 0 812 242" className="chain-ripple" role="img"
         aria-label="Your sgd from Module 3 asks which way is downhill and receives one slope per knob. Two implementations can answer: nudge-and-measure, the course's gradient function, priced per knob; or your backprop, priced per pass, one forward pass and one backward sweep for every knob at once.">
      <defs>
        <marker id="pipe-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6"
                markerHeight="6" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" className="ripple-head" />
        </marker>
      </defs>

      <rect x={SGD.x} y={SGD.y} width={SGD.w} height={SGD.h} rx={6} className="ripple-box" />
      <text x={SGD.x + SGD.w / 2} y={SGD.y + 32} textAnchor="middle" className="ripple-title">
        your sgd
      </text>
      <text x={SGD.x + SGD.w / 2} y={SGD.y + 52} textAnchor="middle" className="ripple-value">
        Module 3, unchanged
      </text>
      <text x={SGD.x + SGD.w / 2} y={SGD.y + 69} textAnchor="middle" className="ripple-value">
        one step per mini-batch
      </text>

      {/* the exchange: a question out, slopes back */}
      <line x1={SGD.x + SGD.w} y1={104} x2={FORK - 2} y2={104}
            className="ripple-arrow" markerEnd="url(#pipe-head)" />
      <text x={(SGD.x + SGD.w + FORK) / 2} y={96} textAnchor="middle" className="ripple-why">
        which way is downhill?
      </text>
      <line x1={FORK - 2} y1={134} x2={SGD.x + SGD.w} y2={134}
            className="ripple-arrow" markerEnd="url(#pipe-head)" />
      <text x={(SGD.x + SGD.w + FORK) / 2} y={152} textAnchor="middle" className="ripple-why">
        one slope per knob
      </text>

      {/* branch to the two answers */}
      <line x1={FORK} y1={topMid} x2={FORK} y2={botMid} className="ripple-arrow" />
      <line x1={FORK} y1={topMid} x2={IMPL.x} y2={topMid} className="ripple-arrow" />
      <line x1={FORK} y1={botMid} x2={IMPL.x} y2={botMid} className="ripple-arrow" />

      <text x={IMPL.x} y={TOP_Y - 8} className="ripple-change">
        what you have been using
      </text>
      <rect x={IMPL.x} y={TOP_Y} width={IMPL.w} height={IMPL.h} rx={6}
            className="ripple-box-old" />
      <text x={IMPL.x + IMPL.w / 2} y={TOP_Y + 30} textAnchor="middle" className="ripple-title">
        nudge-and-measure
      </text>
      <text x={IMPL.x + IMPL.w / 2} y={TOP_Y + 50} textAnchor="middle" className="ripple-value">
        the course's gradient function
      </text>
      <text x={IMPL.x + IMPL.w / 2} y={TOP_Y + 70} textAnchor="middle" className="ripple-value">
        two rescores of the batch, for each knob separately
      </text>

      <text x={IMPL.x} y={BOT_Y - 8} className="ripple-change">
        what you write in this module
      </text>
      <rect x={IMPL.x} y={BOT_Y} width={IMPL.w} height={IMPL.h} rx={6} className="ripple-box" />
      <text x={IMPL.x + IMPL.w / 2} y={BOT_Y + 30} textAnchor="middle" className="ripple-title">
        your backprop
      </text>
      <text x={IMPL.x + IMPL.w / 2} y={BOT_Y + 50} textAnchor="middle" className="ripple-value">
        the four equations, once per example
      </text>
      <text x={IMPL.x + IMPL.w / 2} y={BOT_Y + 70} textAnchor="middle" className="ripple-value">
        one pass forward, one sweep back, every knob at once
      </text>
    </svg>
  );
}

// Static small-multiple: the backward sweep's two passes over the receipts of
// a two-layer network, with the slots each pass reads shaded. Makes the
// loop's one-step-left motion visible instead of asking the reader to
// imagine it (and the off-by-one the tests diagnose).
function BackwardWalkDiagram() {
  const PANEL_W = 390;
  const BW = 112;
  const GAP = 8;
  const BH = 44;
  const ACTS_Y = 44;
  const ZS_Y = 112;
  const panels = [
    {
      x: 8,
      title: "pass 1 · BP1, at the output layer",
      acts: [
        { label: "a[-3]", note: "the input x", read: false },
        { label: "a[-2]", note: "fed the output layer", read: true },
        { label: "a[-1]", note: "the answer", read: true },
      ],
      zs: [
        { label: "z[-2]", note: "", read: false },
        { label: "z[-1]", note: "output evidence", read: true },
      ],
      writes: "writes nabla_b[-1] and nabla_w[-1]",
    },
    {
      x: 8 + PANEL_W + 16,
      title: "pass 2 · BP2, one layer back",
      acts: [
        { label: "a[-3]", note: "fed the hidden layer", read: true },
        { label: "a[-2]", note: "", read: false },
        { label: "a[-1]", note: "", read: false },
      ],
      zs: [
        { label: "z[-2]", note: "hidden evidence", read: true },
        { label: "z[-1]", note: "", read: false },
      ],
      writes: "writes nabla_b[-2] and nabla_w[-2]",
    },
  ];
  const slot = (
    x: number,
    y: number,
    s: { label: string; note: string; read: boolean },
    key: string,
  ) => (
    <g key={key}>
      <rect x={x} y={y} width={BW} height={BH} rx={5}
            className={s.read ? "ripple-box-read" : "ripple-box-idle"} />
      <text x={x + BW / 2} y={y + 19} textAnchor="middle" className="ripple-change">
        {s.label}
      </text>
      <text x={x + BW / 2} y={y + 35} textAnchor="middle" className="ripple-value">
        {s.note}
      </text>
    </g>
  );
  return (
    <svg viewBox="0 0 812 212" className="chain-ripple" role="img"
         aria-label="Two panels. Pass 1, BP1 at the output layer, reads activations[-1], activations[-2] and zs[-1], and writes the last layer's slopes. Pass 2, BP2 one layer back, reads activations[-3] and zs[-2], and writes the hidden layer's slopes: the same window of slots, moved one box left.">
      {panels.map((p, pi) => (
        <g key={pi}>
          <text x={p.x + PANEL_W / 2} y={20} textAnchor="middle" className="ripple-title">
            {p.title}
          </text>
          <text x={p.x + 4} y={ACTS_Y - 6} className="ripple-band-label">activations</text>
          {p.acts.map((s, i) => slot(p.x + 4 + i * (BW + GAP), ACTS_Y, s, `a${pi}${i}`))}
          <text x={p.x + 4} y={ZS_Y - 6} className="ripple-band-label">zs</text>
          {p.zs.map((s, i) =>
            slot(p.x + 4 + (i + 1) * (BW + GAP), ZS_Y, s, `z${pi}${i}`),
          )}
          <text x={p.x + PANEL_W / 2} y={ZS_Y + BH + 26} textAnchor="middle"
                className="ripple-why">
            {p.writes}
          </text>
        </g>
      ))}
      <text x={406} y={206} textAnchor="middle" className="ripple-change">
        shaded = read on this pass
      </text>
    </svg>
  );
}

// Static diagram: the gradient's shape, beside the parameters it is the
// gradient of. The course states "shaped like weights and biases" in five
// places and had never drawn it, which is what let the four equations read
// as four numbers. The counts under the slope row derive the 23,860 the
// training section then spends.
function GradientShapeDiagram() {
  const ROW_TOP = 32;
  const ROW_BOT = 116;
  const BH = 52;
  const panels = [
    {
      title: "weights, and their slopes",
      x: 18,
      bw: 214,
      gap: 12,
      pairs: [
        { param: "weights[0]", grad: "nabla_w[0]", shape: "(30, 784)", count: "23,520" },
        { param: "weights[1]", grad: "nabla_w[1]", shape: "(10, 30)", count: "300" },
      ],
    },
    {
      title: "biases, and their slopes",
      x: 496,
      bw: 146,
      gap: 12,
      pairs: [
        { param: "biases[0]", grad: "nabla_b[0]", shape: "(30, 1)", count: "30" },
        { param: "biases[1]", grad: "nabla_b[1]", shape: "(10, 1)", count: "10" },
      ],
    },
  ];
  const box = (x: number, y: number, w: number, name: string, shape: string,
               cls: string, key: string) => (
    <g key={key}>
      <rect x={x} y={y} width={w} height={BH} rx={6} className={cls} />
      <text x={x + w / 2} y={y + 21} textAnchor="middle" className="ripple-title">{name}</text>
      <text x={x + w / 2} y={y + 40} textAnchor="middle" className="ripple-change">{shape}</text>
    </g>
  );
  return (
    <svg viewBox="0 0 812 232" className="chain-ripple" role="img"
         aria-label="Two panels. On the left, weights[0] of shape (30, 784) and weights[1] of shape (10, 30), each with the slope array nabla_w[0] and nabla_w[1] of the same shape directly below it, holding 23,520 and 300 numbers. On the right, biases[0] of shape (30, 1) and biases[1] of shape (10, 1), with nabla_b[0] and nabla_b[1] below them, holding 30 and 10. The four slope arrays hold 23,860 numbers in total, one per knob.">
      <defs>
        <marker id="grad-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5"
                markerHeight="5" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" className="ripple-head" />
        </marker>
      </defs>
      <line x1={478} y1={20} x2={478} y2={196} className="shapes-grid" />
      {panels.map((p, pi) => {
        const span = p.bw * 2 + p.gap;
        return (
          <g key={pi}>
            <text x={p.x + span / 2} y={18} textAnchor="middle" className="ripple-band-label">
              {p.title}
            </text>
            {p.pairs.map((pair, i) => {
              const x = p.x + i * (p.bw + p.gap);
              const mid = x + p.bw / 2;
              return (
                <g key={i}>
                  {box(x, ROW_TOP, p.bw, pair.param, pair.shape, "ripple-box", `p${pi}${i}`)}
                  <line x1={mid} y1={ROW_TOP + BH + 4} x2={mid} y2={ROW_BOT - 4}
                        className="ripple-arrow" markerEnd="url(#grad-head)" />
                  {pi === 0 && i === 0 && (
                    <text x={mid + 8} y={ROW_BOT - 12} className="ripple-why">same shape</text>
                  )}
                  {box(x, ROW_BOT, p.bw, pair.grad, pair.shape, "ripple-box-read", `g${pi}${i}`)}
                  <text x={mid} y={ROW_BOT + BH + 22} textAnchor="middle" className="ripple-change">
                    {pair.count}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
      <text x={406} y={222} textAnchor="middle" className="ripple-product">
        23,520 + 300 + 30 + 10 = 23,860 numbers, one for every knob in the network
      </text>
    </svg>
  );
}
