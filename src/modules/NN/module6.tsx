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
          "Say why one neuron cannot trace a hump, and build one out of two neurons that can.",
          "Read a step's position and sharpness off its bias and weight, and place a bump where you want it.",
          "Fit any curve you can draw with a hidden layer of bumps, and price the accuracy in neurons.",
        ]}
      />
      <ModuleToc />

      <p>
        Module 5 finished the machinery: your feedforward, your sgd, your
        backprop, and a digit reader trained by all three. This module spends
        none of it. There is no code here and no exercise, just one question
        and a playground. The question is what a network of these neurons can
        express at all, and the answer is the reason people bother with hidden
        layers: given enough hidden neurons, a network can trace any curve you
        can draw, as closely as you ask. That claim is called universality.
        Everything below builds the machine behind it out of neurons you
        already know, by hand, with no training involved.
      </p>

      <SectionHeader id="m6-ask" title="A curve, not a verdict" />
      <p>
        Every network so far answered yes-or-no questions. Module 1's neuron
        decided go or stay; Module 5's digit reader answered ten such questions
        at once, one per digit. Change the question.
      </p>
      <p>
        Keep the concert, and turn its first input from a switch into a dial.
        Module 1's weather input was 1 for good and 0 for bad. Make it the
        forecast high instead, running from 0 at freezing to 1 at scorching, so
        0.45 on the dial is about 18 degrees. And ask for a quantity instead of
        a verdict: rate how much you want to go, from 0 to 10.
      </p>
      <p>
        Your answer is a curve over the dial now. Near freezing you would
        rather stay in, so the rating starts near 1. It climbs through the mild
        middle to about 9. In a heatwave with no shade it drops back to about
        1. No formula produced that curve. It is just what you would say,
        temperature by temperature.
      </p>
      <Figure caption="The curve to match (dashed) and two attempts by a single neuron (green). A neuron on the dial reports sigma(w x + b), scaled by one output weight: it can climb and flatten, or fall and flatten. Neither one comes back down, so neither can trace a hump.">
        <TargetCurveFigure />
      </Figure>
      <p>
        One neuron cannot trace that curve, and the reason has nothing to do
        with picking better numbers. A neuron on this dial computes its
        evidence <M tex="z = wx + b" />, one weight and one bias, and reports{" "}
        <M tex="\sigma(z)" />. Turn the dial up and <M tex="z" /> moves in one
        direction only: up if <M tex="w" /> is positive, down if it is
        negative. So the report climbs and then flattens, or falls and then
        flattens. It never turns around. The far side of the hump is out of
        reach for every choice of <M tex="w" /> and <M tex="b" />.
      </p>
      <p>
        Module 1 hit the same kind of wall: no single straight line could cut
        the contrarian's four dots apart (the rule that says go when exactly
        one of two things is good). The fix there was a hidden layer of two
        neurons. The fix here is a hidden layer too, and this time it gets
        built by hand rather than trained.
      </p>

      <SectionHeader id="m6-steps" title="Turn the steepness up" />
      <p>
        Start from what one neuron does do, and push it to an extreme. Two
        questions about <M tex="\sigma(wx + b)" />: where does it switch over,
        and how sharply?
      </p>
      <p>
        Where: at the dial setting whose evidence is exactly zero. Solve{" "}
        <M tex="wx + b = 0" /> and the switchover sits at{" "}
        <M tex="x = -b/w" />, read as minus the bias divided by the weight.
        Module 1 did this same rearrangement with two inputs and got a line;
        with one input it gives a single point on the dial.
      </p>
      <p>
        How sharply: Module 1's slider showed the sigmoid climbing from about
        0.05 to about 0.95 while its evidence runs from <M tex="-3" /> to{" "}
        <M tex="+3" />, a span of 6. On the dial that span is <M tex="6/w" />{" "}
        wide, because one unit of dial buys <M tex="w" /> units of evidence. So
        the weight alone sets the sharpness, and it sets it upside down: the
        bigger the weight, the narrower the switchover.
      </p>
        <div className="table-scroll scroll-x" tabIndex={0}>
          <table className="truth-table">
          <thead>
            <tr>
              <th>weight</th>
              <th>switchover width (6 / w)</th>
              <th>bias for a switchover at 0.40</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>10</td><td>0.600 of the dial</td><td>−4</td></tr>
            <tr><td>50</td><td>0.120</td><td>−20</td></tr>
            <tr><td>400</td><td>0.015</td><td>−160</td></tr>
          </tbody>
          </table>
        </div>
      <p>
        With a weight of 400 the neuron is a step: 0 below 0.40 of the dial, 1
        above it, and a switchover too narrow to see. The bias is what places
        the step, and placing it takes one multiplication:{" "}
        <M tex="b = -w s" /> puts the step at <M tex="s" />, which is the{" "}
        <M tex="x = -b/w" /> rule read the other way.
      </p>
      <p>
        Module 1 drew the sigmoid next to the perceptron's hard step and argued
        for the smooth one, because a step's slope is zero everywhere and
        leaves learning nothing to work with. Nothing here contradicts that.
        These networks are built by hand rather than trained, so flat parts
        cost nothing, and a large weight makes a sigmoid behave like a step
        without ever quite being one.
      </p>

      <SectionHeader id="m6-bump" title="Two steps make a bump" />
      <p>
        One step still only goes one way. Two of them, subtracted, do not.
      </p>
      <p>
        Take two neurons on the dial, both with weight 400. The first switches
        on at 0.40 (bias <M tex="-160" />), the second at 0.60 (bias{" "}
        <M tex="-240" />). Send both into the output neuron, with output weight{" "}
        <M tex="+6" /> on the first and <M tex="-6" /> on the second, and
        follow the total across the dial:
      </p>
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
        A bump: 6 tall, from 0.40 to 0.60, zero on both sides. Two hidden
        neurons, one bump, and every feature of it is a number you set. The two
        edges are the two biases. The height is the pair of output weights.
      </p>
      <Figure caption="One bump out of two neurons. The dashed curves are the two hidden neurons' reports, each already scaled by its output weight (+h and −h); the solid curve is their total, which is what the network answers. The steepness starts at 40, low enough to see that the edges are sigmoids; raise it to 400 and the picture becomes the table above. Drag the right step past the left one and the bump becomes a dip.">
        <BumpBuilder />
      </Figure>
      <Eq
        tex="\underbrace{2}_{\text{weights in}} + \underbrace{2}_{\text{biases}} + \underbrace{2}_{\text{output weights}} = 6"
        gloss="Six numbers per bump. The construction sets them from three decisions: where the bump starts, where it ends, and how tall it is. Both weights in are the shared steepness, both biases come from the edges, and the two output weights are one height with opposite signs."
      />
      <Aside>
        <p>
          One departure from the anatomy the course has used so far. In every
          earlier module the last neuron squashed its total with{" "}
          <M tex="\sigma" />, because the answer was a confidence between 0 and
          1. Here the answer is a rating from 0 to 10, so the last neuron
          reports its total as it stands, with no squash. Networks that predict
          a quantity rather than a class are built exactly this way. Nielsen's
          chapter keeps the squash and aims the total at the un-squashed
          version of the target instead; the construction underneath is the
          same either way.
        </p>
      </Aside>

      <SectionHeader id="m6-sculpt" title="Sculpt any curve" />
      <p>
        A bump is a bar. So slice the dial into equal pieces, give each piece
        its own bump, and set each bump's height to where the target runs over
        that piece: the target curve, drawn as a bar chart. Each bar is one
        pair of hidden neurons, so a six-bar approximation is a network with
        twelve hidden neurons, and its answer is the sum of its bars.
      </p>
      <p>
        The playground below is that network, with the bar count in your hands.
        Nothing else about the construction changes as that number goes up:
        same two neurons per bar, same edges, same heights. Only the slices get
        narrower.
      </p>
      <Figure caption="Sculpting a curve out of bumps. Press Fit it for me to put every bar at the target's average height across its own slice, then move the bar count and watch the shaded area between the curves. The second target is a friend's curve, which has an extra peak down at the cold end for the crisp winter shows; the third setting lets you draw a curve of your own and fit that one instead.">
        <CurveSculptor />
      </Figure>

      <SectionHeader id="m6-price" title="What accuracy costs" />
      <p>
        Here is what Fit it for me scores on the first curve, at the step
        sharpness the playground starts with. The area between the curves is
        the score, and since the dial is one unit wide, it is also the average
        miss in rating points.
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
        Every doubling of the bars cuts the area by slightly more than half,
        and the picture says why. A bar is flat where the target is not, so the
        miss inside one slice is at most how far the target drifts across that
        slice. Halve the slice's width and the target has half as far to drift,
        so the worst miss inside it halves too, in every slice at once.
      </p>
      <p>
        Nothing in that argument stops anywhere. Name an accuracy, and slicing
        the dial finely enough reaches it, which is the universality claim for
        one input. The price is the rest of the table: two neurons per bar, six
        numbers per bar, and 144 of them to trace one curve of one input.
      </p>
      <p>
        One more experiment, because it says what bars are for. Set 16 bars,
        fit them, and then lower the step sharpness from 400 to 100. The area
        falls from 0.204 to 0.110. The edges stop being vertical, and rounded
        shoulders follow a smooth curve better than square steps do. So the bar
        chart is not the closest fit these neurons can produce; it is the
        version whose accuracy is easy to argue about.
      </p>

      <SectionHeader id="m6-inputs" title="More than one input" />
      <p>
        The dial was one input. Module 1's concert plane had two, and there the
        construction gains one step. Make both inputs dials: the forecast
        across, how keen your friend is up, each running 0 to 1. The answer is
        a rating at every point of that square.
      </p>
      <p>
        Two neurons give a bump along the forecast axis: 6 inside a vertical
        band, 0 outside it. Two more give a bump along the friend axis: 6
        inside a horizontal band. Add all four together and the square has
        three levels, 6 in each band alone, 12 where the bands cross, 0
        elsewhere. Now pass that total through one more neuron with a large
        weight and a bias of <M tex="-9" />. Nine sits between 6 and 12, so
        only the crossing clears the threshold: the neuron reports 1 on a small
        rectangle of the square and 0 everywhere else.
      </p>
      <Figure caption="Building a tower on the concert plane. Each square is the plane of two dials, forecast across and friend's keenness up, and the numbers are what the running total reads in each region. Four neurons make the two bands; the fifth thresholds their sum at 9, which only the crossing clears.">
        <TowerDiagram />
      </Figure>
      <p>
        That rectangle is a tower, and it is the two-input version of the bar.
        Everything after it repeats the one-input story: give each tower its
        height with an output weight, tile the square with towers, and any
        surface over the square comes within reach as the tiling gets finer.
        More inputs work the same way with more bands per tower. More outputs
        need no new idea at all, since each output neuron reads the same hidden
        layer through its own weights.
      </p>
      <p>
        One honest note on the wiring: the thresholding neuron sits between the
        bump neurons and the output, so this two-input construction uses two
        hidden layers rather than one. A single hidden layer is enough for two
        inputs as well, which is the version Nielsen's chapter proves, at the
        cost of a picture this clean.
      </p>

      <SectionHeader id="m6-limits" title="What universality does not say" />
      <p>
        Four limits, because the claim is easy to over-read.
      </p>
      <p>
        It covers continuous curves. Sums of sigmoids move a little when the
        input moves a little, always. A target that jumps, answering 2 just
        below 0.5 of the dial and 9 just above, cannot be matched exactly by
        any network of these neurons. It can be matched everywhere except a
        narrow strip around the jump, which for most purposes is the same
        thing. What the claim really covers is any curve you could draw without
        lifting the pen.
      </p>
      <p>
        It promises approximation, not equality. The claim is that you name an
        accuracy and a network exists that meets it, not that some network
        reproduces the target exactly. The table above is the promise in its
        honest form: every row buys a smaller number, and no row reaches zero.
      </p>
      <p>
        It says can, not will. Every number in this module was placed by hand:
        the biases from the edges, the output weights from the heights. Nothing
        here says gradient descent would find weights like these, and it does
        not: the pretrained network in Module 2 had weights that looked like
        blurry strokes of ink, nothing like bar edges. Universality is about
        what a set of parameters can express. Training is about what descent
        finds, and Modules 7 and 8 are about that.
      </p>
      <p>
        And it is wasteful. The bar chart spends two neurons per slice and
        shares nothing between slices: each pair knows its own strip of the
        dial and nothing else. Between the points it was fitted to it says
        whatever its bar says, which is a poor way to guess about inputs it has
        not seen. That is Module 7's subject. Deeper networks reuse their
        detectors instead of tiling, which is Module 8's.
      </p>
      <p>
        So universality is not a technique to use. It is a constraint removed:
        the shape of a problem is not what stops a network, so everything that
        goes wrong from here is about finding the weights, not about whether
        they exist.
      </p>

      <Recap
        items={[
          "One neuron on one input can only climb and flatten, or fall and flatten. A hump needs at least two, and two large-weight neurons with opposite output weights make a bump whose edges are its biases and whose height is its output weights.",
          "A neuron's switchover sits at minus its bias over its weight, and is 6/w of the input range wide, so a large weight turns a sigmoid into a step.",
          "Slice the input, give each slice a bump, and the network traces the target as a bar chart. Doubling the bars roughly halves the area between the curves, at two neurons and six numbers per bar. Two inputs replace bars with towers, built by thresholding a pair of crossed bumps.",
          "Universality says the weights exist. It does not say descent finds them, that they are efficient, or that they generalize. Module 7 returns to training the digit reader and makes it work better.",
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
