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
        picture. Later sections carry it to two inputs, and then to 784. There
        is no code and no exercise here, and the playground in the middle is the
        argument.
      </p>

      <SectionHeader id="m6-ask" title="The question, shrunk to one dial" />
      <p>
        The claim is about relationships. Your digit reader holds one: 784 pixel
        values in, ten confidences out, and every setting of its 23,860 numbers
        picks a different one. Which relationships can be reached that way, and
        which cannot? At 784 inputs the question has no picture, so shrink it to
        one input and one output, where a relationship is a curve you can draw.
      </p>
      <p>
        Shrinking keeps the question intact. Hold 783 of the digit reader's
        pixels still and turn the remaining one: each of its ten outputs traces
        a curve over that pixel. A curve out of reach here is out of reach
        there.
      </p>
      <p>
        So: one dial, one answer. Module 1's weather input was 1 for good
        weather and 0 for bad, so make it the forecast high instead, 0 at
        freezing to 1 at scorching. And ask for a rating out of 10 rather than a
        verdict. Your own ratings, temperature by temperature, are the dashed
        curve below.
      </p>
      <Figure caption="The curve to match (dashed) and two attempts by a single neuron (green). A neuron on the dial reports sigma(w x + b), scaled by one output weight: it can climb and flatten, or fall and flatten.">
        <TargetCurveFigure />
      </Figure>
      <p>
        Mild weather rates about 9 there, and both ends about 1. No formula
        produced that curve; it is what you would say, temperature by
        temperature.
      </p>
      <p>
        The two green curves are single neurons trying to match it, and no
        choice of numbers does better. A neuron on the dial computes its
        evidence <M tex="z = wx + b" />, one weight and one bias, and reports{" "}
        <M tex="\sigma(z)" />. Turn the dial up and <M tex="z" /> moves one way
        only: up if <M tex="w" /> is positive, down if it is negative. So the
        report climbs and flattens, or falls and flattens. It never turns
        around, which leaves the far side of the hump out of reach.
      </p>
      <p>
        Module 1 hit this kind of wall before: no single straight line could
        separate the four concert dots when the rule was go when exactly one of
        the two things is good. The fix was a hidden layer. Same fix here, built
        by hand instead of trained.
      </p>

      <SectionHeader id="m6-steps" title="Turn the steepness up" />
      <p>
        A hidden layer needs parts to build from, and one kind of part will do:
        something that acts on one stretch of the dial and does nothing outside
        it, so that many of them add up without interfering. This section makes
        a sharp edge; the next turns two edges into that part.
      </p>
      <p>Start with one neuron on the dial, and turn its weight up.</p>
      <Figure caption="The same neuron at three weights, each bias chosen to put its switchover at 0.40 of the dial. A bigger weight does not move the switchover, it narrows it: the shaded band is where the report is on its way from 0 to 1. At 400 the neuron is a step.">
        <StepSharpnessFigure />
      </Figure>
      <p>
        Two numbers describe each panel, and both read off the neuron directly.
        The switchover sits where the evidence is exactly zero: solve{" "}
        <M tex="wx + b = 0" /> and it is at <M tex="x = -b/w" />, minus the bias
        over the weight. Its width is <M tex="6/w" />, because the sigmoid needs
        its evidence to run from about <M tex="-3" /> to <M tex="+3" /> to climb
        from 0.05 to 0.95, and one unit of dial buys <M tex="w" /> units of
        evidence.
      </p>
      <p>
        Read the position rule the other way and it becomes the instruction used
        from here on: <M tex="b = -ws" /> puts a step at <M tex="s" />. The third
        panel is that rule with <M tex="w = 400" /> and <M tex="s = 0.40" />.
      </p>
      <p>
        Module 1 argued for the sigmoid over the perceptron's hard step, because
        a step's slope is zero everywhere and leaves learning nothing to work
        with. That still holds. Nothing on this page is trained, so flat parts
        cost nothing here, and a large weight only makes a sigmoid behave like a
        step, never quite become one.
      </p>

      <SectionHeader id="m6-bump" title="Two steps make a bump" />
      <p>
        One step acts on everything to its right, which is the opposite of what
        the part needs. Two steps fix that, if the second undoes the first. Put
        a step at 0.40 and another at 0.60, both with weight 400, and give the
        output neuron the weights <M tex="+6" /> and <M tex="-6" /> on them.
      </p>
      <Figure caption="Two step neurons added with opposite output weights. Each dashed curve is one neuron's report times its own output weight: one rises to +6 at 0.40, the other falls to −6 at 0.60. The solid curve is their total, which is what the network answers. Lower the steepness to see that both edges are sigmoids, and drag the right step past the left one to turn the bump into a dip.">
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
        A bump, 6 tall, from 0.40 to 0.60. The two edges are the two biases, the
        height is the pair of output weights, and the zero outside is what makes
        the part usable: a step never stops acting, so anything placed to its
        right has to be chosen around it, while two bumps on different slices
        cannot reach each other at all. Their heights can be set one at a time,
        in any order.
      </p>
      <Eq
        tex="\underbrace{2}_{\text{weights in}} + \underbrace{2}_{\text{biases}} + \underbrace{2}_{\text{output weights}} = 6"
        gloss="Six numbers per bump, set by three decisions: where it starts, where it ends, how tall it is. Both weights in are the shared steepness, both biases come from the edges, and the two output weights are one height with opposite signs."
      />
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
        bars is a hidden layer of twelve.
      </p>
      <p>
        Nothing about the construction changes as the bar count goes up. Same
        two neurons per bar, same edges, same heights, narrower slices.
      </p>
      <Figure caption="Sculpting a curve out of bumps. Press Fit it for me to put every bar at the target's average height across its own slice, then move the bar count and watch the shaded area between the curves. The second curve is a friend's, with an extra peak at the cold end for the crisp winter shows; the third setting lets you draw your own and fit that.">
        <CurveSculptor />
      </Figure>

      <SectionHeader id="m6-price" title="What accuracy costs" />
      <p>
        Here is what Fit it for me scores on the first curve, at the sharpness
        the playground starts with. The dial is one unit wide, so the area
        between the curves is also the average miss in rating points.
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
        Nothing in that argument stops anywhere: name an accuracy, slice finely
        enough, reach it. That is universality for one input, and its price is
        the rest of the table, two neurons and six numbers per bar. One
        experiment before moving on: at 16 bars, lower the sharpness from 400 to
        100, and the area falls from 0.204 to 0.110, because rounded shoulders
        follow a smooth curve better than square steps do. The bar chart is the
        version whose accuracy is easy to argue about, not the closest fit these
        neurons can make.
      </p>
      <p>
        That settles one input. Two sections left: the same argument at two
        inputs, and then at 784.
      </p>

      <SectionHeader id="m6-inputs" title="More than one input" />
      <p>
        One dial gave a curve. Two dials give a surface: Module 1's concert
        plane with both inputs turned continuous, the forecast across and how
        keen your friend is up, and a rating at every point of the square. The
        construction gains one step there, and the four panels are it.
      </p>
      <Figure caption="Building a tower on the concert plane. Each square is the plane of two dials, forecast across and friend's keenness up, and the numbers are what the running total reads in each region. Four neurons make the two bands; the fifth thresholds their sum at 9, which only the crossing clears.">
        <TowerDiagram />
      </Figure>
      <p>
        Two neurons make a bump along the forecast: 6 inside a vertical band, 0
        outside. Two more make a bump along the friend: 6 inside a horizontal
        band. Added, the square reads 0 outside both bands, 6 in either band
        alone, and 12 where they cross. Then one more neuron, with a large
        weight and a bias of <M tex="-9" />. Nine sits between 6 and 12, so only
        the crossing clears it, and that neuron reports 1 on a small rectangle
        and 0 everywhere else.
      </p>
      <p>
        That rectangle is a tower, the two-input version of the bar. The rest
        repeats: an output weight sets each tower's height, towers tile the
        square, and finer tiles bring any surface within reach. More inputs need
        more bands per tower and nothing else. More outputs need nothing at all,
        since every output neuron reads the same hidden layer through its own
        weights.
      </p>
      <p>
        One honest note on the wiring, and on the layer count from the opener:
        the thresholding neuron sits between the bump neurons and the output, so
        this two-input construction uses two hidden layers, where the one-input
        construction used one. A single hidden layer is enough for two inputs as
        well, and that is how the theorem is usually stated, but this picture
        does not show it, and neither does Nielsen's chapter. He states the
        theorem for one hidden layer, proves the two-layer version drawn here
        because it is the one that can be drawn, and sets the one-layer version
        as a problem with the route sketched: make steps in slanted directions
        instead of along the two axes, add those into towers that are round
        rather than rectangular, and tile with the round ones.
      </p>

      <SectionHeader id="m6-digits" title="Back to the digit reader" />
      <p>
        Now the question this page opened with. Towers work at any number of
        inputs: two neurons fence a band along each input, one more thresholds
        their sum, and the tower becomes a box. Put one box around each training
        image, tight enough to hold one image each, and give each box an output
        weight that reports that image's digit.
      </p>
      <Figure caption="That construction used as a digit reader, drawn with two pixels so it fits on paper. At 784 pixels the boxes are boxes in 784 directions, and the white space is the same.">
        <BoxesFigure />
      </Figure>
      <p>
        At 784 inputs instead of 2, that network gets all 5,000 training images
        right. It spends two hidden layers doing it, bands then thresholds, as
        the last section noted, and one layer would do by the route sketched
        there. Price it:
      </p>
      <Eq
        tex="\underbrace{2}_{\text{neurons per band}} \times \underbrace{784}_{\text{pixels}} \times \underbrace{5{,}000}_{\text{training images}} = 7{,}840{,}000"
        gloss="Two neurons to fence each pixel's band, 784 pixels per box, one box per training image, plus one thresholding neuron per box. Your trained network does the job with 30 hidden neurons, so this construction is about 260,000 times the size of the network you trained."
      />
      <p>
        And it knows nothing between its boxes. Your 30-neuron network reads 89
        percent of images it never saw; this one answers 0 for anything outside
        the 5,000 boxes it was built from. It proves that the parameters can
        express a perfect fit on the training set. It does not read digits.
      </p>
      <p>
        Which is the split worth keeping. The architecture was never the cap: a
        wide enough hidden layer can express a function that fits every training
        image, and can approximate any pixels-to-digit rule you could write
        down. What the argument does not provide is a way to find such a
        function by descent, or a reason for the one you find to work on images
        it has not seen. Finding is Module 7's first subject, and working on
        unseen images is its second. One thing it leaves genuinely open: whether
        30 hidden neurons in particular are enough. Universality is about the
        family of these networks, not one size of one of them, and Module 2
        already put the hidden layer's size down as a free choice with a
        trade-off.
      </p>

      <SectionHeader id="m6-limits" title="What universality does not say" />
      <p>Four limits, because the claim is easy to over-read.</p>
      <p>
        It covers continuous curves. Sums of sigmoids move a little when the
        input moves a little, always, so a target that jumps, answering 2 just
        below 0.5 of the dial and 9 just above, cannot be matched exactly. It
        can be matched everywhere except a narrow strip at the jump, which for
        most purposes is the same thing. What the claim covers is any curve you
        could draw without lifting the pen.
      </p>
      <p>
        It promises approximation, not equality. You name an accuracy and a
        network exists that meets it. The table above is that promise in its
        honest form: every row buys a smaller number, and no row reaches zero.
      </p>
      <p>
        It says can, not will. Every number on this page was placed by hand, and
        nothing about descent would produce weights like these: Module 2's
        pretrained network had weights that looked like blurry strokes of ink,
        nothing like bar edges, and no training run would land on the box
        network.
      </p>
      <p>
        And it is wasteful: two neurons per slice, 7.8 million of them for the
        box network, and nothing shared between slices, since each pair knows
        its own strip and nothing else. Deeper networks reuse their detectors
        across the input instead of tiling it, which is Module 8's subject.
      </p>
      <p>
        So universality is not a technique. It is a constraint removed: the
        shape of the problem is not what stops a network, so everything that
        goes wrong from here is about finding the weights, not about whether
        they exist.
      </p>

      <Recap
        items={[
          "One neuron on one input can only climb and flatten, or fall and flatten. A hump needs at least two, and two large-weight neurons with opposite output weights make a bump whose edges are its biases and whose height is its output weights.",
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
// Both attempts are honest single-neuron outputs, one output weight times a
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
      viewBox="0 0 812 262"
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
          {i === 0 && (
            <>
              <text x={sx(i, 0.48)} y={sy(0.86)} textAnchor="middle" className="tower-num">6</text>
              <text x={sx(i, 0.15)} y={sy(0.86)} textAnchor="middle" className="tower-num">0</text>
            </>
          )}
          {i === 1 && (
            <>
              <text x={sx(i, 0.5)} y={sy(0.42)} textAnchor="middle" className="tower-num">6</text>
              <text x={sx(i, 0.5)} y={sy(0.86)} textAnchor="middle" className="tower-num">0</text>
            </>
          )}
          {i === 2 && (
            <>
              <text x={sx(i, 0.48)} y={sy(0.42)} textAnchor="middle" className="tower-num">12</text>
              <text x={sx(i, 0.48)} y={sy(0.86)} textAnchor="middle" className="tower-num">6</text>
              <text x={sx(i, 0.83)} y={sy(0.42)} textAnchor="middle" className="tower-num">6</text>
              <text x={sx(i, 0.15)} y={sy(0.86)} textAnchor="middle" className="tower-num">0</text>
            </>
          )}
          {i === 3 && (
            <>
              <text x={sx(i, 0.48)} y={sy(0.42)} textAnchor="middle" className="tower-num">1</text>
              <text x={sx(i, 0.15)} y={sy(0.86)} textAnchor="middle" className="tower-num">0</text>
              <text x={sx(i, 0.5)} y={TOP + S + 22} textAnchor="middle" className="ripple-change">
                a tower
              </text>
            </>
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
    </svg>
  );
}

// Static small-multiple: one neuron on the dial at three weights, each bias
// chosen to hold the switchover at 0.40. Replaces a table of widths and
// biases: the shaded band IS the width, so the numbers are read off a
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
      aria-label="Three panels of one neuron on the dial, at weights 10, 50 and 400, each switching over at 0.40. The band where its report climbs from 0 to 1 shrinks from 0.600 of the dial to 0.120 and then to 0.015, where the curve is a step."
    >
      {panels.map((p, i) => {
        const band = 6 / p.w;
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
              x={px(i, Math.max(0, STEP_AT - band / 2))}
              y={py(1.06)}
              width={px(i, Math.min(1, STEP_AT + band / 2)) - px(i, Math.max(0, STEP_AT - band / 2))}
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
              band {p.width} wide
            </text>
          </g>
        );
      })}
      <text x={px(0, 0) - 4} y={py(1) + 4} textAnchor="end" className="chart-tick">1</text>
      <text x={px(0, 0) - 4} y={py(0) + 4} textAnchor="end" className="chart-tick">0</text>
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
