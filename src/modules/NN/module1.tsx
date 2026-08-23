import { AfterThis, Figure, Recap } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExercisePage } from "../../components/ExercisePage";
import { sigmoidNeuronExercise } from "../../exercises/sigmoid-neuron";
import { SeparatingLine } from "./interactives/SeparatingLine";
import { XorNetwork } from "./interactives/XorNetwork";
import { scale } from "./interactives/utils";

export function Module1() {
  return (
    <article className="module">
      <h2>Module 1: From neurons to networks</h2>
      <AfterThis
        items={[
          "Explain what it means for a machine to learn from examples instead of following rules.",
          "Compute a neuron's decision by hand, and say what its weights and bias each control.",
          "See exactly where one neuron hits a wall, and how stacking neurons breaks through it.",
        ]}
      />

      <p>
        Start with the problem this whole course is about. Computers are superb at
        following exact instructions and terrible at fuzzy judgment calls. You can
        recognize a sloppily handwritten 7 instantly, but try writing down rules for
        it: "a horizontal stroke, then a diagonal one," and immediately the exceptions
        bury you. Neural networks flip the approach: instead of writing rules, you
        show the machine thousands of labeled examples, and it adjusts itself until
        its answers match. That self-adjustment is what "learning" means here, nothing
        more mystical than that. By Module 5 you will have built, with your own code,
        a program that learns to read handwritten digits. This module builds its
        smallest part.
      </p>

      <p>
        That smallest part is a neuron, and it is just a weighted decision. Suppose
        you are deciding whether to go to an outdoor concert, and three yes-or-no
        facts matter: is the weather good (<M tex="x_1" />), is a friend coming (
        <M tex="x_2" />), is it easy to get to (<M tex="x_3" />). Each is 1 for yes, 0
        for no. They do not matter equally: say weather matters 6 points to you, the
        friend 2, transport 1. Those importances are the weights. Add up the points
        for the facts that are true, and go if the total clears your personal
        threshold, say 5. Try it: good weather, no friend, bad transport gives 6
        points, clears 5, you go. Bad weather with everything else in favor gives 3,
        you stay home. Written as one formula, with the threshold moved to the left
        side and renamed the bias <M tex="b = -5" />:
      </p>
      <Eq
        tex="z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b"
        gloss="Multiply each input by its weight, add all of it up, then add the bias; the bias is just the threshold in disguise, so that 'clears the threshold' becomes simply z > 0."
      />
      <p>
        That multiply-each-pair-then-add-everything operation comes up so often that
        it has a name and a shorthand: the dot product, written{" "}
        <M tex="w \cdot x" />. Whenever you see <M tex="z = w \cdot x + b" /> in this
        course, it means exactly the sum above, nothing extra. A neuron that outputs 1
        when <M tex="z > 0" /> and 0 otherwise is called a perceptron, the original
        artificial neuron. Everything it believes about the world is stored in a few
        numbers, and learning will mean adjusting those numbers.
      </p>

      <p>
        Now a picture, and it comes straight out of numbers you can compute by hand.
        Stay with the concert, but keep only two facts so we can draw them: the
        weather (<M tex="x_1" />, weight 6) and the friend (<M tex="x_2" />, weight
        2), bias still <M tex="-5" />. Only four situations exist, so compute{" "}
        <M tex="z = 6x_1 + 2x_2 - 5" /> for each: bad weather alone gives{" "}
        <M tex="-5" />, friend but bad weather gives <M tex="-3" />, good weather
        alone gives <M tex="1" />, both gives <M tex="3" />. Now put those four
        situations on graph paper, <M tex="x_1" /> across and <M tex="x_2" /> up,
        each dot labeled with the <M tex="z" /> you just computed:
      </p>
      <Figure caption="The concert neuron, drawn. Each dot is one of the four possible situations; green means the neuron says go (z above 0), gold means stay. The line is every point where z would be exactly 0, and the tinted region is the neuron's whole go zone.">
        <ConcertPlot />
      </Figure>
      <p>
        The two stay dots landed on one side, the two go dots on the other, and the
        frontier between them, every point where <M tex="z" /> is exactly zero, is
        the straight line in the picture. A line is exactly what the algebra
        predicts: rearrange <M tex="z = 0" /> the way you did in school:
      </p>
      <Eq
        tex="w_1 x_1 + w_2 x_2 + b = 0 \quad\Longrightarrow\quad x_2 = -\tfrac{w_1}{w_2}\, x_1 - \tfrac{b}{w_2}"
        gloss="This is y = mx + c with different letters: a straight line, whose slope and position are set entirely by the two weights and the bias."
      />
      <p>
        So one neuron, seen geometrically, is one straight line: yes on one side, no
        on the other. The three numbers <M tex="w_1, w_2, b" /> and the line are the
        same object in two costumes. Change a weight and the line tilts; change the
        bias and it slides. Keep that translation in your head for the rest of the
        course.
      </p>

      <p>
        So far you picked the weights and read off the verdicts. Learning runs the
        other way around: you are handed inputs whose correct answers are already
        known (these known examples are the two "classes" of dots below, green and
        gold), and you must find weights that make the neuron's verdict match every
        one. In the picture, that job is: place the line so all green ends up on one
        side and all gold on the other. The interactive below lets you do it with
        your hands, and this is the part to internalize: dragging the line IS
        choosing <M tex="w_1, w_2, b" />. The readout under the chart shows, live,
        which three numbers your line corresponds to. Start on the OR dataset (green
        means "at least one input is 1"), then try AND (green means "both are 1").
        Both should take you a few seconds.
      </p>
      <Figure caption="Drag the two round handles to place the line. OR and AND fall in seconds. Then switch to XOR and keep trying: no single straight cut ever puts both green points on one side and both gold points on the other.">
        <SeparatingLine />
      </Figure>

      <p>
        Then there is XOR, the third dataset. XOR ("exclusive or") is a tiny rule:
        output 1 when the two inputs differ, 0 when they agree. The whole task is four
        points:
      </p>
      <table className="truth-table">
        <thead>
          <tr>
            <th><M tex="x_1" /></th>
            <th><M tex="x_2" /></th>
            <th>inputs differ?</th>
            <th>class</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>0</td><td>0</td><td>no</td><td>0 (gold)</td></tr>
          <tr><td>0</td><td>1</td><td>yes</td><td>1 (green)</td></tr>
          <tr><td>1</td><td>0</td><td>yes</td><td>1 (green)</td></tr>
          <tr><td>1</td><td>1</td><td>no</td><td>0 (gold)</td></tr>
        </tbody>
      </table>
      <p>
        Four points, and no line works: the greens sit on one diagonal, the golds on
        the other, so any line that separates them would have to bend. This is not a
        curiosity. When it was proved in 1969 that a single neuron cannot represent
        XOR, funding and faith in the whole field collapsed for years. The eventual
        answer was not a smarter neuron. It was neurons feeding other neurons.
      </p>

      <p>
        Before stacking them, one upgrade to the neuron itself. Learning, remember,
        will mean nudging weights a little and checking whether the answers improve.
        A perceptron fights that: its output is a hard 0 or 1, so a tiny nudge
        usually changes nothing at all, until suddenly it flips everything. What we
        want is a neuron whose output moves smoothly as its evidence moves. So we
        pass <M tex="z" /> through the sigmoid function:
      </p>
      <Eq
        tex="\sigma(z) = \frac{1}{1 + e^{-z}}"
        gloss="Sigmoid squashes any number z into the range 0 to 1, smoothly: strong positive evidence gives almost 1, strong negative gives almost 0, and z = 0 lands exactly on one half. (e is just a fixed number, about 2.718; np.exp computes powers of it.)"
      />
      <p>
        A sigmoid neuron outputs <M tex="\sigma(w \cdot x + b)" />: not a verdict but
        a confidence, like 0.93 or 0.02. It still draws the same line through the
        plane, but near the line it hedges, and that hedging is what makes gradual
        learning possible. Every neuron in this course from here on is a sigmoid
        neuron.
      </p>

      <p>
        Now the stacking. If one neuron is one line, use three neurons. Wire them
        like this:
      </p>
      <Figure caption="The smallest network that can beat XOR. Every arrow carries a weight, every neuron adds its own bias and applies sigmoid. h1 and h2 are called the hidden layer: neither input nor final output.">
        <TinyNetDiagram />
      </Figure>
      <p>
        There is nothing new inside any of these neurons. The two middle ones,{" "}
        <M tex="h_1" /> and <M tex="h_2" />, each read <M tex="x_1, x_2" /> with
        their own weights and bias, exactly like the concert neuron, so each one
        draws its own line through the picture. The only novelty is the output
        neuron's wiring: its two inputs are not <M tex="x_1" /> and <M tex="x_2" />{" "}
        but the outputs of <M tex="h_1" /> and <M tex="h_2" />. It never sees the
        original point at all; it only hears the two reports.
      </p>
      <p>
        Here is one way to set the nine numbers, in words first. Make{" "}
        <M tex="h_1" /> ask "is at least one input on?" (weights 6 and 6, bias{" "}
        <M tex="-3" />: a single on input gives <M tex="z = 3" />, already positive).
        Make <M tex="h_2" /> ask "are both on?" (weights 6 and 6, bias{" "}
        <M tex="-9" />: one on input only reaches <M tex="z = -3" />, so it takes
        both). Then let the output listen for "<M tex="h_1" /> yes, <M tex="h_2" />{" "}
        no" (weight <M tex="+8" /> from <M tex="h_1" />, <M tex="-8" /> from{" "}
        <M tex="h_2" />, bias <M tex="-4" />). "At least one, but not both" is
        exactly XOR. Check every case:
      </p>
      <table className="truth-table">
        <thead>
          <tr>
            <th><M tex="x_1" /></th>
            <th><M tex="x_2" /></th>
            <th><M tex="h_1" /> (at least one?)</th>
            <th><M tex="h_2" /> (both?)</th>
            <th>output</th>
            <th>class</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>0</td><td>0</td><td>≈ 0</td><td>≈ 0</td><td>≈ 0</td><td>0 (gold)</td></tr>
          <tr><td>0</td><td>1</td><td>≈ 1</td><td>≈ 0</td><td>≈ 1</td><td>1 (green)</td></tr>
          <tr><td>1</td><td>0</td><td>≈ 1</td><td>≈ 0</td><td>≈ 1</td><td>1 (green)</td></tr>
          <tr><td>1</td><td>1</td><td>≈ 1</td><td>≈ 1</td><td>≈ 0</td><td>0 (gold)</td></tr>
        </tbody>
      </table>
      <p>
        The playground below is this exact network with all nine numbers on sliders.
        How to read it: the shading is the output neuron's verdict for every possible
        input at once, green for class 1, sand for class 0. The solid purple line is
        where <M tex="h_1" />'s own evidence crosses zero; the dashed teal line is{" "}
        <M tex="h_2" />'s. Each slider group is labeled with its line. Two
        suggestions: press "Show a solution" first and then break it, one slider at a
        time, to feel what each number does. Then start over and rebuild it. Getting
        all four points right by hand is genuinely fiddly, and that is foreshadowing:
        Module 3 is about making the computer do the fiddling.
      </p>
      <Figure caption="The 2-2-1 network on its four XOR points. Move any slider and watch that neuron's line, and the output's shading, respond.">
        <XorNetwork />
      </Figure>

      <p>
        Time to build the neuron in code, which means two minutes of NumPy, the
        Python library for arrays of numbers. Everything in this exercise uses five
        ideas:
      </p>
      <pre className="hint-pre">{`import numpy as np

w = np.array([[6.0], [2.0], [1.0]])   # an array: 3 rows, 1 column
x = np.array([[1.0], [0.0], [0.0]])   # same shape: (3, 1)

w * x           # multiplies matching entries -> [[6.0], [0.0], [0.0]]
(w * x).sum()   # adds every entry up -> 6.0  (that's the dot product!)
np.exp(-2.0)    # e^(-2); given an array, it does every entry at once`}</pre>
      <p>
        One convention to adopt now, because the whole course uses it: lists of
        numbers are always columns, shape (n, 1), meaning n rows and 1 column. The
        shape is the first thing the tests check, and the error messages will coach
        you when it is off.
      </p>

      <ExercisePage exercise={sigmoidNeuronExercise} />

      <Recap
        items={[
          "Learning means adjusting numbers until answers match examples, not writing rules.",
          "A neuron computes sigmoid(w . x + b): multiply each input by its weight, add, add the bias, squash smoothly into (0, 1).",
          "With two inputs, a neuron is a straight line on graph paper: that is why XOR, whose classes sit on crossing diagonals, defeats any single neuron.",
          "Hidden neurons re-describe the input, and a straight cut in their re-description can look bent in the original space.",
        ]}
        chapter="Chapter 1 (perceptrons and sigmoid neurons)"
        href="http://neuralnetworksanddeeplearning.com/chap1.html"
      />
    </article>
  );
}

// Static plot of the two-input concert neuron (weights 6 and 2, bias -5):
// the four possible situations, their z values, and the z = 0 line.
function ConcertPlot() {
  const W = 400;
  const H = 340;
  const D0 = -0.35;
  const D1 = 1.35;
  const px = (v: number) => scale(v, D0, D1, 40, W - 14);
  const py = (v: number) => scale(v, D0, D1, H - 34, 14);
  // 6*x1 + 2*x2 - 5 = 0  ->  x1 = (5 - 2*x2) / 6
  const lineX1 = (x2: number) => (5 - 2 * x2) / 6;
  const dots = [
    { x1: 0, x2: 0, z: "−5", go: false, labelAbove: false },
    { x1: 0, x2: 1, z: "−3", go: false, labelAbove: true },
    { x1: 1, x2: 0, z: "+1", go: true, labelAbove: true },
    { x1: 1, x2: 1, z: "+3", go: true, labelAbove: true },
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="concert-plot" role="img"
         aria-label="The four concert situations plotted; a straight line separates the two stay dots from the two go dots">
      <polygon
        points={`${px(lineX1(D0))},${py(D0)} ${px(D1)},${py(D0)} ${px(D1)},${py(D1)} ${px(lineX1(D1))},${py(D1)}`}
        className="concert-go-zone"
      />
      {[0, 1].map((v) => (
        <g key={v} className="axis-guides">
          <line x1={px(v)} y1={py(D0)} x2={px(v)} y2={py(D1)} />
          <line x1={px(D0)} y1={py(v)} x2={px(D1)} y2={py(v)} />
          <text x={px(v) - 4} y={H - 16}>{v}</text>
          <text x={22} y={py(v) + 4}>{v}</text>
        </g>
      ))}
      <line
        x1={px(lineX1(D0))} y1={py(D0)} x2={px(lineX1(D1))} y2={py(D1)}
        className="sep-line"
      />
      <text x={px(0.56)} y={py(0.62)} className="concert-label" textAnchor="end">
        z = 0
      </text>
      {dots.map((d, i) => (
        <g key={i}>
          <circle cx={px(d.x1)} cy={py(d.x2)} r={9}
                  className={d.go ? "pt-class1" : "pt-class0"} />
          <text
            x={px(d.x1)} y={py(d.x2) + (d.labelAbove ? -16 : 26)}
            className="concert-label" textAnchor="middle"
          >
            z = {d.z} · {d.go ? "go" : "stay"}
          </text>
        </g>
      ))}
      <text x={W - 14} y={H - 4} className="concert-label" textAnchor="end">
        x₁ = weather
      </text>
      <text x={8} y={12} className="concert-label">
        x₂ = friend
      </text>
    </svg>
  );
}

// Static wiring diagram of the 2-2-1 network discussed in the prose.
function TinyNetDiagram() {
  const nodes = {
    x1: { x: 60, y: 60, label: "x₁" },
    x2: { x: 60, y: 160, label: "x₂" },
    h1: { x: 210, y: 60, label: "h₁" },
    h2: { x: 210, y: 160, label: "h₂" },
    out: { x: 360, y: 110, label: "out" },
  };
  const edges: [keyof typeof nodes, keyof typeof nodes][] = [
    ["x1", "h1"], ["x1", "h2"], ["x2", "h1"], ["x2", "h2"], ["h1", "out"], ["h2", "out"],
  ];
  return (
    <svg viewBox="0 0 430 220" className="tiny-net" role="img"
         aria-label="Diagram: inputs x1 and x2 feed hidden neurons h1 and h2, which feed one output neuron">
      <defs>
        <marker id="tn-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7"
                markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 8 4 L 0 8 z" className="tiny-net-arrowhead" />
        </marker>
      </defs>
      {edges.map(([from, to]) => {
        const a = nodes[from];
        const b = nodes[to];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = Math.hypot(dx, dy);
        const R = 24;
        return (
          <line
            key={`${from}-${to}`}
            x1={a.x + (dx / len) * R} y1={a.y + (dy / len) * R}
            x2={b.x - (dx / len) * (R + 6)} y2={b.y - (dy / len) * (R + 6)}
            className="tiny-net-edge" markerEnd="url(#tn-arrow)"
          />
        );
      })}
      {Object.entries(nodes).map(([id, n]) => (
        <g key={id}>
          <circle cx={n.x} cy={n.y} r={22}
                  className={id.startsWith("x") ? "tiny-net-input" : "tiny-net-neuron"} />
          <text x={n.x} y={n.y + 5} textAnchor="middle" className="tiny-net-label">
            {n.label}
          </text>
        </g>
      ))}
      <text x={60} y={205} textAnchor="middle" className="tiny-net-caption">inputs</text>
      <text x={210} y={205} textAnchor="middle" className="tiny-net-caption">hidden layer</text>
      <text x={360} y={205} textAnchor="middle" className="tiny-net-caption">output</text>
    </svg>
  );
}
