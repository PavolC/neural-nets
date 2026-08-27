import { AfterThis, Aside, Figure, ModuleToc, Recap, SectionHeader } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { BumpBuilder } from "./interactives/BumpBuilder";
import { CurveSculptor } from "./interactives/CurveSculptor";
import { scale, sigmoid } from "./interactives/utils";

export function Module6() {
  return (
    <article className="module">
      <h2>Module 6: Universality (an interlude)</h2>
      <AfterThis
        items={[
          "Say which of the three candidates for your digit reader's 89 percent this module rules out, and which two it leaves standing.",
          "Build a bump out of two neurons, and place it anywhere on an input's range by choosing two biases.",
          "Fit any curve you can draw with a hidden layer of bumps, and price the accuracy in neurons.",
        ]}
      />
      <ModuleToc />

      <p>
        Module 5 left your digit reader at about 89 percent, where Nielsen's
        Chapter 1 reports about 95 on the same 784-30-10 shape with ten times
        the data. Three things could be capping it. The data: 5,000 training
        images instead of 50,000. The training: one learning rate, 15 epochs,
        the quadratic cost. Or the shape itself, meaning that one hidden layer
        of sigmoid neurons feeding ten outputs cannot express the
        pixels-to-digit rule at all, however wide you make it.
      </p>
      <p>
        This page rules out the third, and only the third. The result is called
        universality: one hidden layer, made wide enough, can express any
        relationship between its inputs and its outputs, to any accuracy you
        name. Width is what has to grow, not the number of layers. So a wide
        enough hidden layer can express a function that gets every training
        image right, which leaves the data and the training to explain the 89
        percent. The data is fixed here so the run fits in a browser tab. That
        leaves the training, and the training is Module 7.
      </p>
      <p>
        The network below has one input, one output, and weights placed by hand
        instead of trained: the smallest version where the argument fits in a
        picture. There is no code and no exercise here.
      </p>

      <SectionHeader id="m6-ask" title="One dial, one answer" />
      <p>
        Give a network this job. One number in, one number out: hand it the
        forecast, and it answers how much you want to go to the concert, from 0
        to 10. Module 1's weather input was a switch, 1 for good and 0 for bad.
        This one is a dial, 0 at freezing and 1 at scorching, which is the kind
        of input your digit reader has had all along: Module 2 fed it ink levels,
        0 for blank paper and 1 for full ink.
      </p>
      <p>Your own answers, temperature by temperature, are the job.</p>
      <Figure caption="The curve to match (dashed) and two attempts by a single neuron (green). A neuron on the dial reports sigma(w x + b), and its outgoing wire carries one weight that scales the report: the neuron can climb and flatten, or fall and flatten.">
        <TargetCurveFigure />
      </Figure>
      <p>
        Freezing rates about 1, mild about 9, scorching about 1 again, and no
        formula produced that: it is just what you would say, temperature by
        temperature.
      </p>
      <p>
        One neuron cannot do that job, and no choice of numbers fixes it. A
        neuron on the dial computes its evidence <M tex="z = wx + b" />, one
        weight and one bias, and reports <M tex="\sigma(z)" />. Turn the dial up
        and <M tex="z" /> moves one way only: up if <M tex="w" /> is positive,
        down if it is negative. So its report climbs and flattens, or falls and
        flattens, which is what the two green curves are doing. Neither turns
        around, so the far side of the hump stays out of reach.
      </p>
      <p>
        Module 1 hit this kind of wall before, when no single straight line could
        separate the four concert dots under the rule go when exactly one of the
        two things is good. The fix there was a hidden layer, and it is the fix
        here too, except that this one gets built by hand instead of trained.
      </p>
      <p>
        Hold 783 of your digit reader's pixels still and turn the last one: each
        of its ten answers traces a curve like this one, so a curve out of reach
        here is out of reach there.
      </p>

      <SectionHeader id="m6-steps" title="Turn the steepness up" />
      <p>
        That curve sits at 9 in the middle and near 1 at both ends, so whatever
        builds it has to act on one stretch of the dial and leave the rest
        alone. A neuron's report does the opposite: it climbs once and stays up.
        Turn its weight up anyway, and watch what the climb does.
      </p>
      <Figure caption="The same neuron at three weights, each bias chosen to put its switchover at 0.40 of the dial. A bigger weight does not move the switchover, it narrows it: the shaded part is the switchover itself, where the report is on its way from 0 to 1. At 400 the neuron is a step.">
        <StepSharpnessFigure />
      </Figure>
      <p>
        Two numbers describe each panel, and both read off the neuron directly.
        The switchover sits where the evidence is exactly zero: solve{" "}
        <M tex="wx + b = 0" /> and it is at <M tex="x = -b/w" />, minus the bias
        over the weight. Module 1 solved the same equation with two inputs,{" "}
        <M tex="w_1 x_1 + w_2 x_2 + b = 0" />, and got a straight line across the
        concert plane, the frontier with go on one side and stay on the other.
        With one input that frontier has nowhere to run, so it shrinks to the one
        point on the dial where the neuron switches over.
      </p>
      <p>
        The switchover's width is <M tex="6/w" />, because the sigmoid needs its
        evidence to run from about <M tex="-3" /> to <M tex="+3" /> to climb from
        0.05 to 0.95, and one unit of dial buys <M tex="w" /> units of evidence.
      </p>
      <p>
        Read the position rule the other way and it becomes the instruction used
        from here on: <M tex="b = -ws" /> puts a step at <M tex="s" />. The third
        panel is that rule with <M tex="w = 400" /> and <M tex="s = 0.40" />, so
        its bias is <M tex="-400 \times 0.40 = -160" />.
      </p>
      <p>
        Work one out before moving on. Keeping the weight at 400, what bias puts a
        step three quarters of the way along the dial? The rule says{" "}
        <M tex="-400 \times 0.75 = -300" />. The panel in the next section prints
        the bias for whichever position you choose, so drag its left step to 0.75
        and check the prediction rather than taking it.
      </p>
      <Aside>
        <p>
          Module 1 argued for the sigmoid over the perceptron's hard step,
          because a step's slope is zero everywhere and leaves learning nothing
          to work with, and that argument still holds. Nothing on this page gets
          trained, so flat parts cost nothing here, and a large weight only
          makes a sigmoid behave like a step; it never quite becomes one.
        </p>
      </Aside>

      <SectionHeader id="m6-bump" title="Two steps make a bump" />
      <p>
        A step still acts on everything to its right: past 0.40 it reports 1
        forever, including out at the scorching end where the rating belongs
        back down near 1. So use two step neurons side by side in one hidden
        layer, and let the second undo the first. Wire them like this:
      </p>
      <Figure caption="The whole network for one bump: the dial, a hidden layer of two neurons, one output neuron. Each green-bordered circle is a neuron, the gray circle is the input number, and every arrow is a wire carrying one weight. Four wires, four weights, one bias inside each hidden neuron, and the output neuron's own bias left at 0.">
        <BumpNetDiagram />
      </Figure>
      <p>
        Both hidden neurons read the same dial, each through its own incoming
        wire, and both those wires carry weight 400, so both neurons are steps.
        Their biases are what differ, by the rule from the last section:{" "}
        <M tex="-160" /> puts the first step at 0.40, and <M tex="-240" /> puts
        the second at 0.60. Each hidden neuron then has
        one wire into the output neuron, and those two wires carry{" "}
        <M tex="+6" /> and <M tex="-6" />.
      </p>
      <Figure caption="The same two neurons, plotted. Each dashed curve is one hidden neuron's report times the weight on its outgoing wire: one rises to +6 at 0.40, the other falls to −6 at 0.60. The solid curve is their total, which is what the output neuron answers. At the steepness it opens on, 400, the top is flat at exactly the height you set; drop it to 40 and the two switchovers meet in the middle, so the bump peaks at 5.78 instead of 6. Drag the right step past the left one and the bump becomes a dip.">
        <BumpBuilder />
      </Figure>
      <p>The total is 0, then 6, then 0 again, and the three rows say why:</p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>the dial</th>
              <th>left neuron</th>
              <th>right neuron</th>
              <th>total: +6 times left, −6 times right</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>below 0.40</td><td>0</td><td>0</td><td>0</td></tr>
            <tr><td>between 0.40 and 0.60</td><td>1</td><td>0</td><td>+6</td></tr>
            <tr><td>above 0.60</td><td>1</td><td>1</td><td>6 − 6 = 0</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        A bump, 6 tall, from 0.40 to 0.60: the two biases place its edges, and
        the two weights on the outgoing wires set its height.
      </p>
      <p>
        Below 0.40 the bump contributes 0, and above 0.60 the two steps cancel,
        so it contributes 0 there too: the bump acts on its own stretch of the
        dial and nowhere else. A step never does that. Past its switchover a
        step reports 1 forever, so it keeps adding to the total everywhere
        further up the dial.
      </p>
      <p>
        Add a second pair of neurons, stepping at 0.60 and 0.80, and the first
        bump's stretch is untouched: between 0.40 and 0.60 the total is still 6,
        whatever height the second bump is given. Each height gets chosen once,
        from the target over its own stretch, and no height already chosen has
        to be revisited.
      </p>
      <Eq
        tex="\underbrace{2}_{\text{weights in}} + \underbrace{2}_{\text{biases}} + \underbrace{2}_{\text{weights out}} = 6"
        gloss="Six numbers per bump, set by three decisions: where it starts, where it ends, how tall it is. Both weights in are the shared steepness, both biases come from the edges, and the two weights out are one height with opposite signs. The output neuron's own bias stays 0 all through this page, so it is not one of the six."
      />
      <p>
        You have built this network before. Module 1's rule, go when exactly one
        of the two things is good, came out as two hidden neurons with weights 6
        and 6 and biases <M tex="-3" /> and <M tex="-9" />, feeding an output
        through wires carrying <M tex="+8" /> and <M tex="-8" /> with a bias of{" "}
        <M tex="-4" />. Both hidden neurons multiplied both inputs by 6, so both
        cared about one number only: the total <M tex="x_1 + x_2" />, which is 0,
        1 or 2 across the four dots. The first switched on once that total passed
        0.5, the second once it passed 1.5.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>the total <M tex="x_1 + x_2" /></th>
              <th>first neuron</th>
              <th>second neuron</th>
              <th>evidence into the output: 8 times the first, −8 times the second, minus 4</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>below 0.5</td><td>0</td><td>0</td><td>−4</td></tr>
            <tr><td>between 0.5 and 1.5</td><td>1</td><td>0</td><td>+4</td></tr>
            <tr><td>above 1.5</td><td>1</td><td>1</td><td>−4</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        Low, high, low: a bump, with its edges at 0.5 and 1.5 and its height set
        by the weights on the two outgoing wires. Module 1 then squashed that total, and the bias{" "}
        <M tex="-4" /> acted as a threshold, turning inside the bump into a yes.
        This page leaves the squash off, so the height is the answer instead of a
        verdict.
      </p>
      <Aside>
        <p>
          One departure from the anatomy the course has used so far. In every
          earlier module the last neuron squashed its total with{" "}
          <M tex="\sigma" />, because the answer was a confidence between 0 and
          1. Here the answer is a rating out of 10, so the last neuron reports
          its total unsquashed, which is how networks that predict a quantity
          are built. Nielsen's chapter keeps the squash and aims the total at the
          un-squashed version of the target instead; underneath, the
          construction is the same.
        </p>
      </Aside>

      <SectionHeader id="m6-sculpt" title="Sculpt any curve" />
      <p>
        A bump is a bar. Slice the dial into equal pieces, give each piece its
        own bump, and set each height to where the target runs over that piece:
        the curve, drawn as a bar chart. One pair of neurons per bar, so six
        bars is a hidden layer of twelve, against the 30 hidden neurons in the
        network you trained, which is 15 bars' worth.
      </p>
      <p>
        The weights now hold a lookup table. Module 1's truth table had four
        rows, one per situation, with the answer written beside it. This one has
        a row per slice of the dial, each row's answer is that bar's height, and
        the network answers by looking up whichever slice the dial is in.
      </p>
      <p>
        The construction does not change as the bar count goes up. Each bar still
        costs two neurons, still has an edge at each end, and still has one height
        to set. The slices just get narrower.
      </p>
      <Figure caption="Sculpting a curve out of bumps. Press Fit it for me to put every bar at the target's average height across its own slice, then move the bar count and watch the shaded area between the curves: 4 bars score 1.030 and 8 score 0.445. The second curve is a friend's, with an extra peak at the cold end for the crisp winter shows; the third setting lets you draw your own and fit that.">
        <CurveSculptor />
      </Figure>

      <SectionHeader id="m6-price" title="What accuracy costs" />
      <p>
        Here is what Fit it for me scores on the first curve, at the sharpness
        the playground starts with. The dial is one unit wide, so the area
        between the curves is also the average miss in rating points: the same
        kind of number as Module 3's cost, one score for the whole fit, though it
        averages the plain gap rather than the squared one.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>bars</th>
              <th>hidden neurons</th>
              <th>numbers in the network</th>
              <th>area between the curves</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>2</td><td>4</td><td>12</td><td>2.311</td></tr>
            <tr><td>4</td><td>8</td><td>24</td><td>1.030</td></tr>
            <tr><td>8</td><td>16</td><td>48</td><td>0.445</td></tr>
            <tr><td>16</td><td>32</td><td>96</td><td>0.204</td></tr>
            <tr><td>24</td><td>48</td><td>144</td><td>0.118</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        Each doubling of the bars cuts the area by slightly more than half, and
        the picture says why. A bar is flat where the target is not, so the miss
        inside one slice is at most how far the target drifts across that slice.
        Halve the slice and the target has half as far to drift.
      </p>
      <p>
        So why not a thousand bars? Two things stop you, and the second is the
        more interesting. The bill comes first: a thousand bars is 2,000 hidden
        neurons and 6,000 numbers, spent tracing one curve of one input, where the
        digit reader reads 784 pixels into ten answers with 23,860. And the returns
        dry up before the bill does. At the sharpness the playground starts on, 96
        bars score 0.008 and a thousand score 0.006, because a bar narrower than
        the switchover (0.015 of the dial at weight 400) blurs into its neighbours.
        Past that, the weights have to grow along with the bar count or the extra
        slices buy nothing.
      </p>
      <p>
        That argument has no stopping point. Name an accuracy, slice the dial
        finely enough, and you reach it, which is universality for one input. The
        bill comes to two neurons and six numbers per bar.
      </p>
      <p>
        One experiment before moving on. At 16 bars, lower the sharpness from 400
        to 100: the area falls from 0.204 to 0.110, because rounded shoulders
        follow a smooth curve better than square steps do. Bars are the version
        whose accuracy is easy to argue about, not the best these neurons can do.
      </p>

      <SectionHeader id="m6-inputs" title="More than one input" />
      <p>
        One dial gave a curve. Two dials give a surface: Module 1's concert
        plane with both inputs turned continuous, the forecast across and how
        keen your friend is up, and a rating at every point of the square. The
        four dots become every point of it, and the tinted region where one
        neuron said go becomes the shaded band in the panels below. Building a
        bump there takes one extra stage.
      </p>
      <Figure caption="Building a tower on the concert plane. Each square is the plane of two dials, forecast across and friend's keenness up. Four neurons make the two bands; the fifth thresholds their sum at 9, which only the crossing clears.">
        <TowerDiagram />
      </Figure>
      <p>
        Two neurons make a bump along the forecast: 6 inside a vertical band, 0
        outside. Two more make a bump along the friend: 6 inside a horizontal
        band. Added, the square reads 0 outside both bands, 6 in either band
        alone, and 12 where they cross. Then one more neuron, thresholding a sum
        of hidden reports the way the XOR network's output neuron did with its
        bias of <M tex="-4" />: give this one a large weight and a bias of{" "}
        <M tex="-9" />. Nine sits between 6 and 12, so only the crossing clears
        it, and that neuron reports 1 on a small rectangle
        and 0 everywhere else.
      </p>
      <p>
        That rectangle is a tower, the two-input version of the bar. Everything
        after it repeats the one-input story: the weight on each tower's
        outgoing wire sets its height, towers tile the square, and finer tiles bring any
        surface within reach. More inputs need more bands per tower and nothing
        else. More outputs need nothing at all,
        since every output neuron reads the same hidden layer through its own
        weights.
      </p>
      <Aside>
        <p>
          The thresholding neuron sits between the bump neurons and the output,
          so this two-input construction spends two hidden layers where the
          one-input construction spent one. One hidden layer is enough for two
          inputs as well, and that is how universality is usually stated, but
          this picture does not show it, and neither does Nielsen's chapter. He
          states the result for one hidden layer, proves the two-layer version
          drawn here because it is the one that can be drawn, and leaves the
          one-layer version as a problem with the route sketched: make steps in
          slanted directions instead of along the two axes, add those into
          towers that are round rather than rectangular, and tile with the round
          ones.
        </p>
      </Aside>

      <SectionHeader id="m6-digits" title="Back to the digit reader" />
      <p>
        Your digit reader has 784 inputs, and towers work there too: two neurons
        fence a band along each input, one more thresholds the sum, and the
        tower becomes a box. Put one box around each training image, tight
        enough to hold one image each, and let the weight on each box's
        outgoing wire report that image's digit.
      </p>
      <Figure caption="That construction used as a digit reader, drawn with two pixels so it fits on paper. At 784 pixels the boxes are boxes in 784 directions, and the white space is the same.">
        <BoxesFigure />
      </Figure>
      <p>
        At 784 inputs instead of 2, that network gets all 5,000 training images
        right. That network spends two hidden layers doing it, bands then
        thresholds, and one hidden layer would do it by the slanted-steps route.
        Price it:
      </p>
      <Eq
        tex="\underbrace{2}_{\text{neurons per band}} \times \underbrace{784}_{\text{pixels}} \times \underbrace{5{,}000}_{\text{training images}} = 7{,}840{,}000"
        gloss="Two neurons to fence each pixel's band, 784 pixels per box, one box per training image, plus one thresholding neuron per box. Your trained network does the job with 30 hidden neurons, so this construction is about 260,000 times the size of the network you trained."
      />
      <p>
        And that network knows nothing between its boxes. The bar chart's lookup
        table is back, one row per box, except that the rows no longer tile
        anything: six bars covered every temperature on the dial, while 5,000
        boxes leave almost all of 784 directions uncovered. Your 30-neuron
        network reads 89 percent of images it never saw, while this one answers 0
        for anything outside the 5,000 boxes it was built from. The box network proves that
        the parameters can express a perfect fit on the training set, and that is
        all it proves; it does not read digits.
      </p>
      <p>
        So the architecture was never the cap: a wide enough hidden layer can
        express a function that fits every training image, and can approximate
        any pixels-to-digit rule you could write down. The argument does not say
        how to find such a function by descent, and it gives no reason for the
        one you do find to work on images it has not seen. Module 7 takes on both
        of those. The argument also leaves one thing genuinely open: whether 30
        hidden neurons in particular are enough, since universality is about the
        family of these networks rather than one size of one of them.
      </p>

      <SectionHeader id="m6-limits" title="What universality does not say" />
      <p>Four limits, because the claim is easy to over-read.</p>
      <p>
        A target that jumps is out of reach. Sums of sigmoids always move a
        little when the input moves a little, so a curve answering 2 just below
        0.5 of the dial and 9 just above cannot be matched exactly. Such a curve
        can still be matched everywhere except a narrow strip at the jump, which
        for most purposes is the same thing. Universality covers any curve you
        could draw without lifting the pen.
      </p>
      <p>
        The match is never exact, either. You name an accuracy and a network
        exists that meets it, which is a weaker promise than it first sounds.
        Every row of the table above buys a smaller number, and no row reaches
        zero.
      </p>
      <p>
        Existing and being findable are different things. Every number on this
        page was placed by hand, and descent would not produce weights like
        these. Module 2's pretrained network had weights that looked like blurry
        strokes of ink, nothing like bar edges, and no training run would land on
        the box network.
      </p>
      <p>
        The construction also wastes neurons: two per slice, and 7.8 million of
        them for the box network. Module 2 called a hidden neuron a little
        pattern-detector, and each of your trained network's 30 detectors helps
        describe every image it sees. The box network's detectors each know one
        image and share nothing, which is where the 260,000-fold difference in
        size comes from. Deeper networks reuse their detectors across the input
        instead of tiling it, which is Module 8's subject.
      </p>
      <p>
        So universality is not a technique you apply. It removes one explanation
        for a network that will not work, which means everything going wrong from
        here is about finding the weights rather than about whether they exist.
      </p>

      <Recap
        items={[
          "One neuron on one input can only climb and flatten, or fall and flatten. A hump needs at least two: two large-weight neurons in one hidden layer, their outgoing wires carrying opposite weights, make a bump whose edges are the two biases and whose height is those two weights.",
          "A neuron's switchover sits at minus its bias over its weight, and is 6/w of the input range wide, so a large weight turns a sigmoid into a step.",
          "Slice the input, give each slice a bump, and the network traces the target as a bar chart. Doubling the bars roughly halves the area between the curves, at two neurons and six numbers per bar. Two inputs replace bars with towers, built by thresholding a pair of crossed bumps.",
          "Universality settles one of the three candidates for your digit reader's 89 percent: not the architecture. A wide enough hidden layer can express a function that fits every training image, so the shortfall is in the data and the training, and Module 7 goes after the training.",
          "What it does not say: that descent finds those weights, that they are efficient (7.8 million hidden neurons for the box network against your 30), or that a function fitting the training images reads new ones.",
        ]}
        chapter="Chapter 4 (a visual proof that neural nets can compute any function)"
        href="http://neuralnetworksanddeeplearning.com/chap4.html"
      />
    </article>
  );
}

// Static figure: the target curve, plus what a single neuron can produce.
// Both attempts are genuine single-neuron outputs: one wire's weight times a
// sigmoid, one with a positive weight and one with a negative one.
function TargetCurveFigure() {
  const W = 480;
  const H = 250;
  const px = (x: number) => scale(x, 0, 1, 40, W - 14);
  const py = (v: number) => scale(v, 0, 10.6, H - 44, 14);
  const target = (x: number) => 1.2 + 8.0 * Math.exp(-Math.pow((x - 0.45) / 0.17, 2));
  const rising = (x: number) => 9.2 * sigmoid(14 * (x - 0.32));
  const falling = (x: number) => 9.2 * sigmoid(-14 * (x - 0.62));
  const path = (f: (x: number) => number) =>
    Array.from({ length: 161 }, (_, i) => {
      const x = i / 160;
      return `${i === 0 ? "M" : "L"}${px(x).toFixed(1)},${py(f(x)).toFixed(1)}`;
    }).join(" ");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="curve-figure"
      role="img"
      aria-label="A hump-shaped target curve rising from about 1 at the freezing end of the dial to about 9 in the middle and back to 1 at the scorching end, with two single-neuron curves that each rise or fall once and then stay flat."
    >
      {[0, 2, 4, 6, 8, 10].map((v) => (
        <g key={v}>
          <line x1={px(0)} x2={px(1)} y1={py(v)} y2={py(v)} className="curve-grid" />
          <text x={px(0) - 7} y={py(v) + 4} className="chart-tick" textAnchor="end">
            {v}
          </text>
        </g>
      ))}
      <text x={px(0) - 7} y={16} className="chart-tick" textAnchor="end">rating</text>
      <line x1={px(0)} x2={px(1)} y1={py(0)} y2={py(0)} className="axis-line" />
      {[0, 0.5, 1].map((x) => (
        <text key={x} x={px(x)} y={py(0) + 18} className="chart-tick" textAnchor="middle">
          {x}
        </text>
      ))}
      <text x={px(0)} y={py(0) + 34} className="chart-tick" textAnchor="start">freezing</text>
      <text x={px(0.5)} y={py(0) + 34} className="chart-tick" textAnchor="middle">
        the forecast dial
      </text>
      <text x={px(1)} y={py(0) + 34} className="chart-tick" textAnchor="end">scorching</text>

      <path d={path(rising)} className="curve-attempt" />
      <path d={path(falling)} className="curve-attempt" />
      <path d={path(target)} className="curve-target" />

      {/* The one band of the plot that no curve crosses: right of 0.68 the
          two green attempts sit at 9.2 and below 2.8, the target below 2.5. */}
      <g>
        <line x1={px(0.68)} x2={px(0.745)} y1={py(6.6)} y2={py(6.6)} className="curve-target" />
        <text x={px(0.755)} y={py(6.6) + 4} className="chart-tick">the curve to match</text>
        <line x1={px(0.68)} x2={px(0.745)} y1={py(5.4)} y2={py(5.4)} className="curve-attempt" />
        <text x={px(0.755)} y={py(5.4) + 4} className="chart-tick">a single neuron</text>
      </g>
    </svg>
  );
}

// Static diagram: the two-input tower, one panel per stage. Each panel is the
// plane of the two dials; the numbers are what the running total reads in
// each region, so the thresholding step at the end has something concrete to
// cut between.
function TowerDiagram() {
  const S = 160; // panel square
  const GAP = 40;
  const X0 = 26;
  const TOP = 46;
  const bandX = [0.34, 0.62]; // the forecast bump's edges, as fractions
  const bandY = [0.32, 0.60]; // the friend bump's edges
  // Region centers: left of the forecast band, inside it, right of it, and the
  // same three up the friend axis, top row first. Every region carries the
  // total it reads there, so the third panel is the first plus the second,
  // entry by entry, and the fourth is the third put through the threshold.
  const XR = [0.17, 0.48, 0.81];
  const YR = [0.8, 0.46, 0.16];
  const reads = (panel: number, cx: number, cy: number) => {
    const forecast = cx === 1 ? 6 : 0;
    const friend = cy === 1 ? 6 : 0;
    if (panel === 0) return forecast;
    if (panel === 1) return friend;
    const total = forecast + friend;
    return panel === 2 ? total : total >= 9 ? 1 : 0;
  };
  const sx = (i: number, f: number) => X0 + i * (S + GAP) + f * S;
  const sy = (f: number) => TOP + (1 - f) * S;
  const panels = [
    { title: "two neurons", sub: "a bump along the forecast" },
    { title: "two more", sub: "a bump along the friend" },
    { title: "add all four", sub: "three levels on the plane" },
    { title: "one more neuron", sub: "threshold the total at 9" },
  ];
  return (
    <svg
      viewBox="0 0 812 274"
      className="chain-ripple"
      role="img"
      aria-label="Four panels of the same square plane. First, a vertical band worth 6. Second, a horizontal band worth 6. Third, both bands added: 6 in each band alone and 12 where they cross. Fourth, a neuron with a bias of minus nine keeps only the crossing, a tower."
    >
      {panels.map((p, i) => (
        <g key={i}>
          <text x={sx(i, 0.5)} y={18} textAnchor="middle" className="ripple-title">
            {p.title}
          </text>
          <text x={sx(i, 0.5)} y={35} textAnchor="middle" className="ripple-value">
            {p.sub}
          </text>
          <rect x={sx(i, 0)} y={TOP} width={S} height={S} className="tower-plane" />

          {(i === 0 || i === 2) && (
            <rect
              x={sx(i, bandX[0])} y={TOP}
              width={(bandX[1] - bandX[0]) * S} height={S}
              className="tower-band"
            />
          )}
          {(i === 1 || i === 2) && (
            <rect
              x={sx(i, 0)} y={sy(bandY[1])}
              width={S} height={(bandY[1] - bandY[0]) * S}
              className="tower-band"
            />
          )}
          {(i === 2 || i === 3) && (
            <rect
              x={sx(i, bandX[0])} y={sy(bandY[1])}
              width={(bandX[1] - bandX[0]) * S}
              height={(bandY[1] - bandY[0]) * S}
              className={i === 3 ? "tower-tower" : "tower-cross"}
            />
          )}

          {/* what the total reads in each region */}
          {YR.map((fy, cy) =>
            XR.map((fx, cx) => (
              <text
                key={`${cx}-${cy}`}
                x={sx(i, fx)}
                y={sy(fy) + 5}
                textAnchor="middle"
                className="tower-num"
              >
                {reads(i, cx, cy)}
              </text>
            )),
          )}
          {i === 3 && (
            <text x={sx(i, 0.5)} y={TOP + S + 22} textAnchor="middle" className="ripple-change">
              a tower
            </text>
          )}
        </g>
      ))}
      <text x={X0} y={TOP + S + 22} className="ripple-value">
        forecast →
      </text>
      {/* Upright at the top-left, this label ran into the first panel's
          subtitle, so it stands on its side beside the square instead. */}
      <text
        x={X0 - 10}
        y={TOP + S / 2}
        transform={`rotate(-90 ${X0 - 10} ${TOP + S / 2})`}
        textAnchor="middle"
        className="ripple-value"
      >
        friend ↑
      </text>
      <text x={X0} y={TOP + S + 48} className="ripple-value">
        Each number is the running total in that region; the shading marks where it is
        not 0.
      </text>
    </svg>
  );
}

// Static small-multiple: one neuron on the dial at three weights, each bias
// chosen to hold the switchover at 0.40. Replaces a table of widths and
// biases: the shaded switchover IS the width, so the numbers are read off a
// picture instead of asserted in prose.
function StepSharpnessFigure() {
  const PW = 140; // panel width
  const X0 = 16;
  const GAP = 16;
  const TOP = 34;
  const BASE = 140;
  const panels = [
    { w: 10, width: "0.600", bias: "−4" },
    { w: 50, width: "0.120", bias: "−20" },
    { w: 400, width: "0.015", bias: "−160" },
  ];
  const STEP_AT = 0.4;
  const px = (i: number, x: number) => X0 + i * (PW + GAP) + 6 + x * (PW - 12);
  const py = (v: number) => scale(v, 0, 1.06, BASE, TOP);
  return (
    <svg
      viewBox="0 0 480 208"
      className="curve-figure"
      role="img"
      aria-label="Three panels of one neuron on the dial, at weights 10, 50 and 400, each switching over at 0.40. The switchover, where its report climbs from 0 to 1, shrinks from 0.600 of the dial to 0.120 and then to 0.015, where the curve is a step."
    >
      {panels.map((p, i) => {
        const sw = 6 / p.w;
        const curve = Array.from({ length: 121 }, (_, k) => {
          const x = k / 120;
          return `${k === 0 ? "M" : "L"}${px(i, x).toFixed(1)},${py(sigmoid(p.w * (x - STEP_AT))).toFixed(1)}`;
        }).join(" ");
        return (
          <g key={p.w}>
            <text x={px(i, 0.5)} y={16} textAnchor="middle" className="ripple-title">
              weight {p.w}
            </text>
            {[0, 1].map((v) => (
              <line
                key={v}
                x1={px(i, 0)} x2={px(i, 1)} y1={py(v)} y2={py(v)}
                className="curve-grid"
              />
            ))}
            <rect
              x={px(i, Math.max(0, STEP_AT - sw / 2))}
              y={py(1.06)}
              width={px(i, Math.min(1, STEP_AT + sw / 2)) - px(i, Math.max(0, STEP_AT - sw / 2))}
              height={BASE - py(1.06)}
              className="step-band"
            />
            <line
              x1={px(i, STEP_AT)} x2={px(i, STEP_AT)} y1={py(1.06)} y2={BASE}
              className="step-mark"
            />
            <path d={curve} className="curve-net" />
            <line x1={px(i, 0)} x2={px(i, 1)} y1={BASE} y2={BASE} className="axis-line" />
            <text x={px(i, STEP_AT)} y={BASE + 15} textAnchor="middle" className="chart-tick">
              0.40
            </text>
            <text x={px(i, 0.5)} y={BASE + 36} textAnchor="middle" className="chart-tick">
              bias {p.bias}
            </text>
            <text x={px(i, 0.5)} y={BASE + 52} textAnchor="middle" className="chart-tick">
              switchover {p.width} wide
            </text>
          </g>
        );
      })}
      <text x={px(0, 0) - 4} y={py(1) + 4} textAnchor="end" className="chart-tick">1</text>
      <text x={px(0, 0) - 4} y={py(0) + 4} textAnchor="end" className="chart-tick">0</text>
    </svg>
  );
}

// Static wiring diagram of the 1-2-1 bump network, drawn to Module 1's
// TinyNetDiagram conventions (same node radius, same column captions, same
// natural-scale 490 viewBox) so the reader recognizes it as the same kind of
// picture. Weights sit on the wires and biases inside the neurons, which is
// the anatomy Modules 1, 2 and 4 use; the numbers are the ones the prose and
// the playground below both carry.
function BumpNetDiagram() {
  const R = 22;
  const dial = { x: 60, y: 116 };
  const h1 = { x: 210, y: 66 };
  const h2 = { x: 210, y: 166 };
  const out = { x: 360, y: 116 };
  const trim = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    return {
      x1: a.x + (dx / len) * R,
      y1: a.y + (dy / len) * R,
      x2: b.x - (dx / len) * (R + 6),
      y2: b.y - (dy / len) * (R + 6),
    };
  };
  const wires: { from: typeof dial; to: typeof dial; label: string; lx: number; ly: number }[] = [
    { from: dial, to: h1, label: "400", lx: 140, ly: 80 },
    { from: dial, to: h2, label: "400", lx: 140, ly: 158 },
    { from: h1, to: out, label: "+6", lx: 285, ly: 82 },
    { from: h2, to: out, label: "−6", lx: 285, ly: 156 },
  ];
  const column = (x: number, lines: string[]) =>
    lines.map((t, i) => (
      <text key={t} x={x} y={248 + i * 14} textAnchor="middle" className="tiny-net-caption">
        {t}
      </text>
    ));
  return (
    <svg
      viewBox="-50 0 490 276"
      className="tiny-net"
      role="img"
      aria-label="Diagram: the dial feeds two hidden neurons through wires of weight 400 each. The first has bias minus 160, so it steps at 0.40; the second has bias minus 240, so it steps at 0.60. Their two wires into the output neuron carry plus 6 and minus 6, and the output neuron adds without squashing."
    >
      <defs>
        <marker
          id="bn-arrow" viewBox="0 0 8 8" refX="7" refY="4"
          markerWidth="7" markerHeight="7" orient="auto-start-reverse"
        >
          <path d="M 0 0 L 8 4 L 0 8 z" className="tiny-net-arrowhead" />
        </marker>
      </defs>
      {wires.map((wire) => (
        <g key={`${wire.lx}-${wire.ly}`}>
          <line {...trim(wire.from, wire.to)} className="tiny-net-edge" markerEnd="url(#bn-arrow)" />
          <text x={wire.lx} y={wire.ly} textAnchor="middle" className="chart-tick">
            {wire.label}
          </text>
        </g>
      ))}

      <circle cx={dial.x} cy={dial.y} r={R} className="tiny-net-input" />
      <text x={dial.x} y={dial.y + 5} textAnchor="middle" className="tiny-net-label">x</text>
      {[h1, h2].map((n, i) => (
        <g key={i}>
          <circle cx={n.x} cy={n.y} r={R} className="tiny-net-neuron" />
          <text x={n.x} y={n.y + 5} textAnchor="middle" className="tiny-net-label">
            {i === 0 ? "h₁" : "h₂"}
          </text>
        </g>
      ))}
      <circle cx={out.x} cy={out.y} r={R} className="tiny-net-neuron" />
      <text x={out.x} y={out.y + 5} textAnchor="middle" className="tiny-net-label">out</text>

      {/* Each hidden neuron's bias, next to the neuron that holds it, with the
          step it buys: the pair the prose reads off this picture. */}
      <text x={h1.x} y={22} textAnchor="middle" className="chart-tick">bias −160</text>
      <text x={h1.x} y={36} textAnchor="middle" className="chart-tick">so it steps at 0.40</text>
      <text x={h2.x} y={206} textAnchor="middle" className="chart-tick">bias −240</text>
      <text x={h2.x} y={220} textAnchor="middle" className="chart-tick">so it steps at 0.60</text>

      {column(dial.x, ["the dial", "(a number, not a neuron)"])}
      {column(h1.x, ["hidden layer", "(2 neurons)"])}
      {column(out.x, ["output layer", "(1 neuron, no squash)"])}
    </svg>
  );
}

// Static figure: the tower construction used as a classifier, at two pixels
// so it can be drawn. One box per training image, and white space everywhere
// else, which is the whole objection to it in one picture. The key sits on
// the figure, so the caption stays one sentence.
function BoxesFigure() {
  const S = 236;
  const X0 = 22;
  const Y0 = 24;
  const sx = (f: number) => X0 + f * S;
  const sy = (f: number) => Y0 + (1 - f) * S;
  const BOX = 38;
  const KEY_X = 286;
  const shown = [
    { x: 0.18, y: 0.74, label: "3" },
    { x: 0.42, y: 0.84, label: "7" },
    { x: 0.72, y: 0.66, label: "0" },
    { x: 0.26, y: 0.28, label: "1" },
    { x: 0.6, y: 0.16, label: "8" },
    { x: 0.88, y: 0.42, label: "4" },
  ];
  return (
    <svg
      viewBox="0 0 480 292"
      className="curve-figure"
      role="img"
      aria-label="A square of two pixel values holding six training images, each fenced inside its own small box labelled with that image's digit. A seventh image sits in the empty middle, inside no box, where the network answers zero."
    >
      <rect x={X0} y={Y0} width={S} height={S} className="tower-plane" />
      {shown.map((d) => (
        <g key={d.label}>
          <rect
            x={sx(d.x) - BOX / 2} y={sy(d.y) - BOX / 2}
            width={BOX} height={BOX}
            className="tower-tower"
          />
          <circle cx={sx(d.x)} cy={sy(d.y)} r={3} className="pt-class1" />
          <text x={sx(d.x)} y={sy(d.y) - BOX / 2 - 5} textAnchor="middle" className="tower-num">
            {d.label}
          </text>
        </g>
      ))}
      <circle cx={sx(0.5)} cy={sy(0.5)} r={4.5} className="pt-wrong" />

      <text x={X0} y={Y0 + S + 18} className="chart-tick">pixel 1 →</text>
      <text
        x={X0 - 12} y={Y0 + S / 2}
        transform={`rotate(-90 ${X0 - 12} ${Y0 + S / 2})`}
        textAnchor="middle"
        className="chart-tick"
      >
        pixel 2 ↑
      </text>

      {/* the key, in the space the square does not use */}
      <rect x={KEY_X} y={54} width={20} height={20} className="tower-tower" />
      <circle cx={KEY_X + 10} cy={64} r={3} className="pt-class1" />
      {["one tower per training", "image, reporting the", "digit of the dot inside"].map((line, i) => (
        <text key={line} x={KEY_X + 30} y={58 + i * 14} className="chart-tick">
          {line}
        </text>
      ))}
      <circle cx={KEY_X + 10} cy={140} r={4.5} className="pt-wrong" />
      {["an image no box holds:", "every tower reports 0,", "so the answer is 0"].map((line, i) => (
        <text key={line} x={KEY_X + 30} y={134 + i * 14} className="chart-tick">
          {line}
        </text>
      ))}
    </svg>
  );
}
