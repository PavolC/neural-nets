import { AfterThis, Figure, Recap } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExercisePage } from "../../components/ExercisePage";
import { sgdExercise } from "../../exercises/sgd";
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

      <p>
        So far the weights were given. Learning means finding them, and the first step
        is admitting you need a score. Here is a network to score: the little
        three-neuron network that beat XOR in Module 1 (two hidden neurons, one output
        neuron, nine numbers in total), with its numbers set where that module's
        sliders start: hidden weights all 2, hidden biases -1 and -3, output weights 4
        and -4, output bias -2. Feed it the four corners of the contrarian's truth
        table (your own feedforward from Module 2 can confirm this) and it answers
        0.247, 0.462, 0.462, 0.247. The right answers are 0, 1, 1, 0. Score it the
        obvious way: take each gap between right answer and output, square it so a
        miss in either direction counts against you, and average over the corners:
      </p>
      <Eq
        tex="C = \frac{\underbrace{(0 - 0.247)^2}_{(0,0)\text{, stay}} + \underbrace{(1 - 0.462)^2}_{(1,0)\text{, go}} + \underbrace{(1 - 0.462)^2}_{(0,1)\text{, go}} + \underbrace{(0 - 0.247)^2}_{(1,1)\text{, stay}}}{2 \times 4} = \frac{0.700}{8} \approx 0.0875"
        gloss="One term per corner of the truth table: the squared gaps are 0.061, 0.289, 0.289 and 0.061, which sum to 0.700. The 4 below the line is the number of examples (an average, so a bigger dataset does not automatically score worse); the extra 2 is a convention that makes later math tidier."
      />
      <p>
        One number for the whole network: 0.0875. Lower is better, and a perfect
        contrarian would score zero. This recipe works for any network and any
        dataset, and its general form is called the quadratic cost:
      </p>
      <Eq
        tex="C(w, b) = \frac{1}{2n} \sum_x \lVert y(x) - a(x) \rVert^2"
        gloss="For each training input x, compare the desired output y(x) with the network's actual output a(x): the double bars squared mean take the gap in every output entry, square each gap, and add them up. Average that over all n examples, with the same tidiness half as above."
      />
      <p>
        You scored one particular choice of the nine numbers. But nothing stops you
        changing them: pick any other nine values, run the same recipe, and you get a
        score for that version of the network too. Learning is now a search problem:
        out of all possible settings of the nine knobs, find the one with the
        smallest cost. (The knobs, all the weights and biases together, get their
        proper name here: the network's parameters.) The way in is a question you can ask of each knob
        separately: if I nudge this one knob a tiny bit, does the cost go up or down,
        and how steeply? Try it on the network just scored. Keeping two more decimals,
        its cost is 0.08758. Nudge the output bias up by 0.01, from -2.00 to -1.99,
        and rescore everything: 0.08714. Divide the change in cost by the size of the
        nudge: (0.08714 - 0.08758) / 0.01, about -0.044. That number is the knob's
        slope, the same m as in
        y = mx + c, measured at the point where you currently stand. Its minus sign
        says the cost falls when this knob goes up. The nudge size is a free choice
        with a trade-off: the ground curves, so a big nudge smears the measurement
        over a stretch of hillside, while a smaller one reads the slope right under
        your feet. We used 0.01 to keep the numbers readable; the course's gradient
        helper, which you will meet in the exercise, nudges by 0.00001. The full list
        of slopes, one per knob, is called the gradient and written{" "}
        <M tex="\nabla C" />. It points the
        way the cost rises fastest, so to descend, step every knob the other way:
      </p>
      <Eq
        tex="w \leftarrow w - \eta \, \nabla C"
        gloss="Move every parameter a small step against its slope; the learning rate eta (a small positive number you choose) sets the step size. Eta is not the 0.01 nudge from a moment ago: the nudge is how you measure which way is downhill, eta is how far you then walk."
      />

      <p>
        Feel this rule before you code it. Below is the simplest possible landscape:
        not a network, just one made-up knob w whose cost is w times w, a single bowl.
        Watch how the steps shrink by themselves as the ball nears the bottom. That is
        the rule working: each step is the learning rate times the slope, and the
        slope shrinks as the ground flattens. Then push the learning rate up and watch
        the walk fall apart.
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

      <p>
        One more idea and you can build it. Look at what one score costs. Scoring the
        XOR network meant feeding it 4 examples, the corners of the truth table.
        Scoring the digit reader the same way means feeding it the whole digit
        training set, and the standard one holds 50,000 images (Module 5 trains on a
        slice of it). Every score is a full pass over every example, and the recipe
        above wants fresh scores all the time: nudge a knob, rescore, for every knob,
        on every step. The expensive part of learning is how many examples every
        single score has to look at.
      </p>
      <p>
        Stochastic gradient descent cheats on exactly that ("stochastic" is just an
        older word for random). Do not score against all the examples: grab a small
        random handful of them, called a mini-batch, and score against the handful.
        The handful's verdict about which way is downhill is only roughly right, but
        it costs a tiny fraction of a full pass. Take the slightly wrong step, grab a
        fresh handful, repeat. Each step is noisy, and you can afford vastly more of
        them.
      </p>
      <p>
        The picture below stages the contest on the earlier valley, built this time
        from 12 examples. Each example on its own scores the network a little
        differently, so each has a private bottom in a slightly different spot, and
        the rings drawn are the average of all 12. The full-batch walker reads all 12
        examples before each step, so it follows the average pull, smoothly. The SGD
        walker reads one example per step and follows that example's pull alone:
        wobbly, but twelve times cheaper. The counter next to the buttons keeps the
        honest tally, examples looked at, so judge the race by that instead of by
        steps.
      </p>
      <Figure caption="Same valley, same start, same learning rate. Full batch consults all 12 examples per step; SGD consults one. Count what each has looked at: SGD reaches the neighborhood of the minimum having read a fraction of the data.">
        <BatchVsSgd />
      </Figure>

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
        (one full pass through the dataset is called an epoch). The course supplies
        the gradient, computed exactly the way you did it by hand above: nudge a
        parameter, measure the cost twice, divide. It is honest and correct, and you
        are about to discover its price.
      </p>

      <ExercisePage exercise={sgdExercise} />

      <p>
        Your sgd is real. The panel below points it at a concert-shaped dataset, forty
        noisy copies of the contrarian's four corners, and hands it a network with two
        inputs, eight hidden neurons, and one output. Eight hidden neurons instead of
        XOR's two is a free choice: spare detectors make learning from a random start
        more forgiving, at the price of more parameters to train. Count them:
      </p>
      <Eq
        tex="\underbrace{8 \times 2}_{\text{hidden } W} + \underbrace{8}_{\text{hidden } b} + \underbrace{1 \times 8}_{\text{output } W} + \underbrace{1}_{\text{output } b} = 16 + 8 + 8 + 1 = 33"
        gloss="Same counting as Module 2's 11,935: each layer's weight matrix is (neurons in this layer) times (inputs feeding it), plus one bias per neuron."
      />
      <p>
        Now watch the clock, and count what training costs. Estimating the gradient
        numerically takes two cost evaluations per parameter per step, and each cost
        evaluation is a full forward pass over the batch. With 33 parameters that
        bill is merely annoying. The digit reader from Module 2 has 11,935 parameters,
        and there the bill is fatal. What we need is a way to get every partial
        derivative from roughly one pass, and that is exactly what backpropagation is.
      </p>
      <SgdLivePanel />

      <Recap
        items={[
          "The cost function turns learning into search; the gradient turns search into walking downhill.",
          "The learning rate trades speed against stability, and elongated valleys punish any single choice.",
          "Mini-batch SGD buys many cheap noisy steps instead of few exact ones, and it wins.",
          "Numerical gradients cost two forward passes per parameter: the motivation for backpropagation.",
        ]}
        chapter="Chapter 1 (learning with gradient descent)"
        href="http://neuralnetworksanddeeplearning.com/chap1.html"
      />
    </article>
  );
}
