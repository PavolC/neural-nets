import { AfterThis, Aside, Figure, ModuleToc, Recap, SectionHeader } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExerciseCard } from "../../components/ExerciseCard";
import { sgdExercise } from "../../exercises/sgd";
import { CostVsKnob } from "./interactives/CostVsKnob";
import { Descent1D, Descent2D } from "./interactives/DescentPlayground";
import { BatchVsSgd } from "./interactives/BatchVsSgd";
import { SgdLivePanel } from "./interactives/SgdLivePanel";

export function Module3() {
  return (
    <article className="module">
      <h2>Module 3: Learning as descent</h2>
      <AfterThis
        items={[
          "Read a cost function as a landscape and gradient descent as walking downhill in it.",
          "Predict what the learning rate does before you move the slider.",
          "Implement mini-batch SGD and train a real (tiny) network with it.",
        ]}
      />
      <ModuleToc />

      <SectionHeader id="m3-cost" title="Scoring a network" />
      <p>
        So far the weights were given. Learning means finding them, and the first step
        is admitting you need a score. Here is a network to score: the little
        three-neuron network that beat XOR in Module 1 (two hidden neurons h1 and h2,
        one output neuron, nine numbers in total), with its numbers set where that
        module's sliders start: hidden weights all 2, hidden biases -1 and -3, output
        weights 4 and -4, output bias -2. Ask it about good weather alone, the corner
        (1, 0), by running the two layers the way your feedforward from Module 2 does:
      </p>
      <Eq
        tex="\begin{aligned} h_1 &= \sigma(2 \cdot 1 + 2 \cdot 0 - 1) = \sigma(1) = 0.731 \\ h_2 &= \sigma(2 \cdot 1 + 2 \cdot 0 - 3) = \sigma(-1) = 0.269 \\ \text{output} &= \sigma(4 \cdot 0.731 - 4 \cdot 0.269 - 2) = \sigma(-0.152) = 0.462 \end{aligned}"
        gloss="sigma is the sigmoid from Module 1, and each line is one neuron doing the usual move: multiply each input by its weight, add everything up including the bias, squash. The two hidden neurons read the corner's inputs 1 and 0; the output neuron reads their confidences 0.731 and 0.269."
      />
      <p>
        The right answer on (1, 0) is 1, and the network says 0.462: just under one
        half, leaning stay when it should say go. The other corners run the same
        way, and your own feedforward can confirm each one. (0, 1) also gives 0.462,
        because both inputs carry the same weight 2. (0, 0) and (1, 1) each work out
        to 0.247. So the four answers are 0.247, 0.462, 0.462, 0.247, against the
        right answers 0, 1, 1, 0. Score it the obvious way: take each gap between
        right answer and output, square it so a miss in either direction counts
        against you, and average over the corners:
      </p>
      <Eq
        tex="\begin{aligned} C &= \frac{\underbrace{(0 - 0.247)^2}_{(0,0)\text{, stay}} + \underbrace{(1 - 0.462)^2}_{(1,0)\text{, go}} + \underbrace{(1 - 0.462)^2}_{(0,1)\text{, go}} + \underbrace{(0 - 0.247)^2}_{(1,1)\text{, stay}}}{2 \times 4} \\[0.8em] &= \frac{0.700}{8} \approx 0.0875 \end{aligned}"
        gloss="One term per corner of the truth table: the squared gaps are 0.061, 0.289, 0.289 and 0.061, which sum to 0.700. The 4 below the line is the number of examples (an average, so a bigger dataset does not automatically score worse). The extra 2 is a bookkeeping choice with a delayed payoff: in Module 4 it will make the cost's slope work out to exactly the gap between output and right answer, with no stray factor of 2."
      />
      <p>
        One number for the whole network: 0.0875. Lower is better, and a perfect
        contrarian would score zero. This recipe works for any network and any
        dataset, and its general form is called the quadratic cost:
      </p>
      <Eq
        tex="C(w, b) = \frac{1}{2n} \sum_x \lVert y(x) - a(x) \rVert^2"
        gloss="For each training input x, compare the desired output y(x) with the network's actual output a(x): the double bars squared mean take the gap in every output entry, square each gap, and add them up. Average that over all n examples, with the same bookkeeping half as above."
      />
      <p>
        That number has other names, and they are the commoner ones. The course
        says cost throughout, following Nielsen; most of the field says{" "}
        <b>loss</b>, and
        formal treatments say the <b>objective function</b>. All three name the one
        number this equation computes, so a sentence elsewhere about minimizing the
        loss is a sentence about the recipe above. The particular cost above has its
        own second name, <b>mean squared error</b>, for the obvious reason.
      </p>
      <p>
        You scored one particular choice of the nine numbers. But nothing stops you
        changing them: pick any other nine values, run the same recipe, and you get a
        score for that version of the network too. Learning is now a search problem:
        out of all possible settings of the nine knobs, find the one with the
        smallest cost. The knobs get their real name here: all of a network's weights
        and biases together are its <b>parameters</b>, so every count of knobs in
        this course is a parameter count. Knob stays in the prose because you can
        turn a knob.
      </p>
      <SectionHeader id="m3-slope" title="The slope of one knob" />
      <p>
        The way in is a question you can ask of each knob separately: if I nudge this
        one knob a tiny bit, does the cost go up or down, and how steeply? Try it on
        the network just scored. Keeping two more decimals, its cost is 0.08758.
        Nudge the output bias up by 0.01, from -2.00 to -1.99, and change nothing
        else: the other eight numbers stay exactly where they are. Recompute the cost
        over the four corners: 0.08714. Divide the change in cost by the size of the
        nudge: (0.08714 - 0.08758) / 0.01, about -0.044. That number is the knob's
        slope, the same m as in y = mx + c, measured at the point where you currently
        stand. Its minus sign says the cost falls when this knob goes up.
      </p>
      <p>
        One knob's slope with every other knob held still is called a{" "}
        <b>partial derivative</b>, and it is written <M tex="\partial C / \partial b" />{" "}
        for the bias you just nudged: read it "the partial derivative of C with
        respect to b". The curly <M tex="\partial" /> is an ordinary derivative's d
        with a note attached, and the note says the other eight numbers are frozen.
        Calling it that changes nothing about it. The -0.044 you just computed is a
        partial derivative, measured rather than derived.
      </p>
      <Aside>
        <p>
          The nudge size is a free choice with a trade-off: the ground curves,
          so a big nudge smears the measurement over a stretch of hillside,
          while a smaller one reads the slope right under your feet. We used
          0.01 to keep the numbers readable; the course's gradient helper, which
          you will meet in the exercise, nudges by 0.00001. It also nudges to
          both sides, once down and once up, and reads the slope between the two
          measurements, which centers the reading on the exact spot where you
          stand; your version read between -2.00 and -1.99, so it sat a
          half-nudge to one side. Either way, one slope costs two cost
          measurements.
        </p>
      </Aside>
      <p>
        Every knob answers this same question, and you can put it to all nine below.
        The curve is the real cost of this network as one chosen knob moves, the
        other eight frozen at their start values; the tilted segment through the
        ball is the slope you would measure at that spot. Each knob still has the
        geometric job Module 1 gave it: a hidden neuron's weights tilt its line, its
        bias slides the line, and the output neuron's knobs re-blend the two
        half-planes into the shaded go region. So each curve below is one of those
        motions, scored: how the total mismatch between the shading and the four
        dots changes as that one line tilts, slides, or re-blends.
      </p>
      <Figure caption="The cost along one knob at a time; the nine knobs are grouped by the neuron they belong to, as in Module 1's sliders. Pick a knob, drag its slider; the gray dot marks the start value from the prose, and switching knobs puts the previous one back. On the output neuron's bias at -2.00 the readout shows the 0.0876 and -0.044 you computed by hand. Try the output neuron's weight from h2: its curve is nearly level, a knob that barely matters right now.">
        <CostVsKnob />
      </Figure>
      <p>
        Now repeat this for each of the nine knobs, always putting the previous knob
        back before nudging the next. Nine measurements, nine slopes. That full list,
        one slope per knob, is called the gradient and written <M tex="\nabla C" />.
        The symbol is an upside-down capital delta, and it is read "nabla", so{" "}
        <M tex="\nabla C" /> is spoken as "nabla C" or "the gradient of C". It names
        the whole list, not a tenth number. For this network, at this exact setting
        of the knobs, it comes out as:
      </p>
      <Eq
        tex="\begin{gathered} \nabla C = (\underbrace{-0.024,\; -0.024,\; -0.041}_{\text{h1's knobs}},\; \underbrace{0.017,\; 0.017,\; 0.041}_{\text{h2's knobs}}, \\[0.8em] \underbrace{-0.035,\; -0.009,\; -0.044}_{\text{output's knobs}}) \end{gathered}"
        gloss="Nine slopes, three per neuron in the interactive's order: each neuron's two weights then its bias, h1 first, the output neuron last. The final entry is the -0.044 you measured on the output bias. A different setting of the knobs would give a different list: the gradient is a local reading, taken where you stand."
      />
      <p>
        The way you got those nine numbers has a name too: measuring a slope by
        nudging one input and rescoring is the method of <b>finite differences</b>,
        and a gradient built that way is a <b>numerical gradient</b>. It needs no
        calculus and works on any network and any cost, which is why Module 5 uses
        it to check the fast method entry by entry, and why the section at the end
        of this module can price it.
      </p>

      <SectionHeader id="m3-descent" title="Walking downhill" />
      <p>
        Read the signs. Six slopes are negative: for those knobs the cost falls when
        the knob goes up, so they should be pushed up. Three are positive (the second
        hidden neuron's two weights and its bias): for those the cost falls when the
        knob goes down. One formula handles both directions at once: subtract each
        slope from its knob. Subtracting a negative slope pushes the knob up,
        subtracting a positive one pushes it down; either way, downhill:
      </p>
      <Eq
        tex="w \leftarrow w - \eta \, \nabla C"
        gloss="Move every parameter a small step against its slope; the learning rate eta (a small positive number you choose) sets the step size. Eta is not the 0.01 nudge from a moment ago: the nudge is how you measure which way is downhill, eta is how far you then walk."
      />
      <p>
        Apply it once, with η = 1 to keep the arithmetic plain: every knob moves by
        its own slope, the output bias from -2.00 up to -1.956, the second hidden
        bias from -3.00 down to -3.041, and so on for all nine. Rescore: the cost
        drops from 0.08758 to 0.0799. That is one step of gradient descent, and
        training a network is repeating it.
      </p>
      <p>
        Why repeat, instead of reading each knob's best value straight off its chart
        above? Because every curve was drawn with the other eight knobs frozen. The
        moment any knob moves, all the other curves are redrawn and their old bottoms
        stop being true; each neuron's best setting depends on what the rest are
        doing, as Module 1's playground made visible. Try the shortcut anyway: jump
        all nine knobs to the bottoms of their current curves at once, and the cost
        lands at 0.1665, nearly double the 0.0876 you started from, because all nine
        jumps broke each other's assumptions. A slope is a compass, not a map: it
        says which way is downhill from where you stand, nothing more. So descent
        inches all the knobs forward together, re-measures everything, and inches
        again.
      </p>

      <p>
        In the nine charts above, you turned the knobs yourself. Below, the rule does
        the turning. The landscape is made up, one knob w whose cost is w times w,
        and made up on purpose: it is far steeper than any of the nine real curves,
        and it keeps rising forever, so the rule's failure modes have room to show.
        Press Step and watch the walk. Each step is the learning rate times the
        slope, so the steps shrink by themselves as the ball nears the flat bottom.
        Then push the learning rate up and watch the same rule fall apart.
      </p>
      <Figure caption="Gradient descent on a bowl. Below roughly 0.5 the ball settles; above it the ball overshoots the bottom and bounces between the walls; past 1.0 each bounce is bigger than the last.">
        <Descent1D />
      </Figure>

      <p>
        Real cost landscapes have many knobs and are not round. The next picture has
        two knobs, w1 across and w2 up, so the cost needs a third dimension; it is
        drawn the way a hiking map draws a mountain. Each ring connects points of
        equal cost, inner rings are lower, and the dot at the center is the bottom of
        the valley. This valley is stretched: steep in the w2 direction, nearly flat
        in the w1 direction, and one learning rate has to serve both. That tension has
        a signature look: zigzag. The gradient keeps pointing across the valley
        instead of along it.
      </p>
      <Figure caption="An elongated valley: 8 times steeper vertically than horizontally. Click to drop the ball anywhere. A learning rate small enough not to bounce across the steep direction is painfully slow along the flat one.">
        <Descent2D />
      </Figure>

      <SectionHeader id="m3-sgd" title="Mini-batches" />
      <p>
        One more idea and you can build it. Look at what one step of descent costs.
        Every slope needs the cost measured twice (once for each side of the nudge),
        every cost measurement is a full pass over the dataset, and every example in
        that pass is one complete trip through the network. For the XOR network that
        multiplies out to nine knobs, times two rescores each, times four corners per
        rescore: 72 trips per step, nothing. Now price the digit reader, whose
        training set is the standard 50,000 images (Module 5 trains on a slice of
        it):
      </p>
      <Eq
        tex="\underbrace{11{,}935}_{\text{knobs}} \times \underbrace{2}_{\text{rescores per slope}} \times \underbrace{50{,}000}_{\text{trips per rescore}} = 1{,}193{,}500{,}000 \text{ trips}"
        gloss="About 1.2 billion complete runs of the network to move every knob once, and training takes thousands of such steps. Three factors multiply: knobs, rescores per knob, and examples per rescore. The last is the biggest and the easiest to attack."
      />
      <p>
        Stochastic gradient descent saves exactly that expense ("stochastic" is just
        an older word for random). Do not score against all the examples: grab a small
        random handful of them, called a mini-batch, and score against the handful.
        To be clear about what shrinks: nothing about the network. Every neuron and
        every weight still runs on each example you do look at, and the scoring
        recipe is unchanged; only the averaging gets shorter. With a mini-batch of
        10, the digit reader's step bill becomes:
      </p>
      <Eq
        tex="\underbrace{11{,}935}_{\text{knobs}} \times \underbrace{2}_{\text{rescores per slope}} \times \underbrace{10}_{\text{trips per rescore}} = 238{,}700 \text{ trips}"
        gloss="Same first two factors: SGD does not touch the cost of measuring slopes, it only shrinks the crowd, and the step gets five thousand times cheaper. The 11,935 times 2 that remains is the factor backpropagation removes, in Modules 4 and 5."
      />
      <p>
        The handful's verdict about which way is downhill is only roughly right. Take
        the slightly wrong step, grab a fresh handful, repeat. Each step is noisy,
        and you can afford five thousand of them for the old price of one.
      </p>
      <p>
        The picture below stages the contest on the earlier valley, built this time
        from 12 examples. Each example on its own scores the network a little
        differently, so each has a private bottom in a slightly different spot, and
        the rings drawn are the average of all 12. The full-batch walker reads all 12
        examples before each step, so it follows the average pull, smoothly. The SGD
        walker reads one example per step and follows that example's pull alone:
        wobbly, but twelve times cheaper. The counter next to the buttons tallies
        examples looked at; judge the race by that instead of by steps.
      </p>
      <Figure caption="Same valley, same start, same learning rate. Full batch consults all 12 examples per step; SGD consults one. Count what each has looked at: SGD reaches the neighborhood of the minimum having read a fraction of the data.">
        <BatchVsSgd />
      </Figure>

      <SectionHeader id="m3-code" title="Implement SGD" />
      <p>
        Time to implement it. First, meet the dataset as your code will receive it.
        One situation stands up as a column of numbers, exactly like the input
        columns in Module 2. A whole dataset is those columns side by side, a matrix
        called <M tex="X" />, and the right answers line up in a second matrix{" "}
        <M tex="Y" />, so that column k of Y grades column k of X. For the
        contrarian's four corners:
      </p>
      <Eq
        tex="X = \begin{pmatrix} 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 1 \end{pmatrix} \qquad Y = \begin{pmatrix} 0 & 1 & 1 & 0 \end{pmatrix}"
        gloss="Four situations, four columns: the top row of X is the weather, the bottom row is the friend, and the four entries of Y are the right answers in the same column order. X has shape (2, 4) and Y has shape (1, 4); the digit dataset would just be a much wider pair, 50,000 columns."
      />
      <p>
        A mini-batch is nothing new either: just some of the columns. Take columns 3
        and 0 of X together with columns 3 and 0 of Y and you hold a batch of two
        examples. With that, the exercise: you write the update step and the loop
        that feeds it: shuffle the column order, walk it in mini-batches, repeat
        (one full pass through the dataset is called an epoch). The shuffling is
        done by rng, the last argument your loop takes: a NumPy random number
        generator, made by np.random.default_rng(0), where the 0 is the seed. Two
        generators made from the same seed deal the same order, so the same
        training run repeats exactly. The gradient
        arrives written for you, in its own section of your file just above the one
        you are about to write, measured by nudging every parameter to both sides
        and rescoring, as described earlier. A section line that says "written for
        you" marks a stretch the course supplies so the file runs on its own; you
        can read it, and nothing in the course asks you to change it. This one is
        correct, and it is slow; the panel after the exercise measures how slow.
      </p>

      <ExerciseCard exercise={sgdExercise} />

      <SectionHeader id="m3-bill" title="The bill" />
      <p>
        The panel below points your sgd at a concert-shaped dataset, forty noisy
        copies of the contrarian's four corners, and hands it a network with two
        inputs, eight hidden neurons, and one output. Eight hidden neurons instead of
        XOR's two is a free choice: spare detectors make learning from a random start
        more forgiving, at the price of more parameters to train. Count them:
      </p>
      <Eq
        tex="\underbrace{8 \times 2}_{\text{hidden } W} + \underbrace{8}_{\text{hidden } b} + \underbrace{1 \times 8}_{\text{output } W} + \underbrace{1}_{\text{output } b} = 16 + 8 + 8 + 1 = 33"
        gloss="Same counting as Module 2's 11,935: each layer's weight matrix is (neurons in this layer) times (inputs feeding it), plus one bias per neuron."
      />
      <p>
        Training works, and it is visibly slow. The slowness is the factor SGD left
        standing in the bill you priced before the exercise: two cost measurements
        per parameter, every step. With 33 parameters that is 66 passes over the
        batch per step, and the panel counts them as it runs. On the digit reader's
        11,935 parameters, training this way stops being realistic. What's needed is
        a way to get every slope from roughly one pass, and that is what
        backpropagation is.
      </p>
      <SgdLivePanel />

      <Recap
        items={[
          "The cost function (the loss, elsewhere) turns learning into search; the gradient, the list of one partial derivative per parameter, turns search into walking downhill.",
          "The learning rate trades speed against stability, and elongated valleys punish any single choice.",
          "Mini-batch SGD buys many cheap noisy steps instead of few exact ones, and it wins.",
          "Nudge-and-measure, which is the method of finite differences, costs two cost measurements per parameter per step: the motivation for backpropagation.",
        ]}
        chapter="Chapter 1 (learning with gradient descent)"
        href="http://neuralnetworksanddeeplearning.com/chap1.html"
      />
    </article>
  );
}
