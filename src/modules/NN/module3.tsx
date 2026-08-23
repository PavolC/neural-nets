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
        is admitting you need a score. Go back to Module 1's slider network, in the
        position the sliders start at, before you fixed it. On the four corners of the
        contrarian's truth table it answers 0.247, 0.462, 0.462 and 0.247; the right
        answers are 0, 1, 1, 0. Score it the obvious way. Take each gap between right
        answer and output: -0.247, 0.538, 0.538, -0.247. Square each gap, so a miss in
        either direction counts against you: 0.061, 0.289, 0.289, 0.061. Add them up
        (0.700) and divide by twice the number of examples (8) to get 0.0875. One
        number for the whole network: that recipe is called the quadratic cost, and
        its shorthand is:
      </p>
      <Eq
        tex="C(w, b) = \frac{1}{2n} \sum_x \lVert y(x) - a(x) \rVert^2"
        gloss="For each training input x, compare the desired output y(x) with the network's actual output a(x): the double bars squared mean take the gap in every output entry, square each gap, and add them up. Average that over all n examples; the half just makes later math tidier."
      />
      <p>
        Every setting of the nine knobs gets one number this way; lower is better, and
        a perfect contrarian would score zero. Learning is now a search problem: find
        the knob settings with the smallest cost. The way in is a question you can ask
        of each knob separately: if I nudge this one knob a tiny bit, does the cost go
        up or down, and how steeply? Try it on the starting network. Keeping two more
        decimals, its cost is 0.08758. Move the output neuron's bias from -2.00 to
        -1.99 and rescore everything: 0.08714. The change divided by the nudge,
        (0.08714 - 0.08758) / 0.01, is about -0.044. That is this knob's slope, the
        same m as in y = mx + c, just measured at the point where you currently stand,
        and its minus sign says: push this knob up and the cost falls. The full list
        of slopes, one per knob, is called the gradient and written{" "}
        <M tex="\nabla C" />. It points the way the cost rises fastest, so to descend,
        step every knob the other way:
      </p>
      <Eq
        tex="w \leftarrow w - \eta \, \nabla C"
        gloss="Move every parameter a small step against its slope; the learning rate eta (a small positive number you choose) sets the step size."
      />

      <p>
        Feel this rule before you code it. Below is the simplest possible landscape,
        one parameter, one bowl. Watch how the steps shrink by themselves as the ground
        flattens, then push the learning rate up and watch the walk fall apart.
      </p>
      <Figure caption="Gradient descent on a bowl. Below roughly 0.5 the ball settles; above it the ball overshoots the bottom and bounces between the walls; past 1.0 each bounce is bigger than the last.">
        <Descent1D />
      </Figure>

      <p>
        Real cost landscapes are not round. Some directions are steep cliffs, others
        near-flat valleys, and one learning rate has to serve both. That tension has a
        signature look: zigzag. The gradient keeps pointing across the valley instead
        of along it.
      </p>
      <Figure caption="An elongated valley: 8 times steeper vertically than horizontally. Click to drop the ball anywhere. A learning rate small enough not to bounce across the steep direction is painfully slow along the flat one.">
        <Descent2D />
      </Figure>

      <p>
        One more idea and you can build it. The true gradient of <M tex="C" /> averages
        over every training example, so one exact step on 50,000 images costs 50,000
        evaluations. Stochastic gradient descent cheats: estimate the gradient from a
        small random mini-batch, take the slightly wrong step, repeat. Each step is
        noisy, but you can afford vastly more of them.
      </p>
      <Figure caption="Same valley, same start, same learning rate. Full batch consults all 12 examples per step; SGD consults one. Count what each has looked at: SGD reaches the neighborhood of the minimum having read a fraction of the data.">
        <BatchVsSgd />
      </Figure>

      <p>
        Time to implement it. You write the update step and the loop that feeds it:
        shuffle the dataset, walk it in mini-batches, repeat (one full pass through
        the dataset is called an epoch). The course supplies the gradient, computed
        exactly the way you did it by hand above: nudge a parameter, measure the cost
        twice, divide. It is honest and correct, and you are about to discover its
        price.
      </p>

      <ExercisePage exercise={sgdExercise} />

      <p>
        Your sgd is real. The panel below points it at a concert-shaped dataset, forty
        noisy copies of the contrarian's four corners, and it learns the rule. But
        watch the clock, and count. Estimating the gradient numerically costs two cost
        evaluations per parameter per step, and each cost evaluation is a full forward
        pass over the batch. The tiny network below has 33 parameters, so the bill is
        merely annoying. The digit reader from Module 2 has 11,935, and there the bill
        is fatal. What we need is a way to get every partial derivative from roughly
        one pass, and that is exactly what backpropagation is.
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
