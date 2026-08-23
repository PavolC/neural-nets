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
        is admitting you need a score. The quadratic cost measures how wrong the
        network is across the training set:
      </p>
      <Eq
        tex="C(w, b) = \frac{1}{2n} \sum_x \lVert y(x) - a(x) \rVert^2"
        gloss="For each training input x, compare the desired output y(x) with the network's actual output a(x): the double bars squared mean take the gap in every output entry, square each gap, and add them up. Average that over all n examples; the half just makes later math tidier."
      />
      <p>
        Every choice of weights and biases gets one number: lower is better. That turns
        learning into a search problem, and a strange one: there are thousands of knobs
        to turn at once. The way in is a question you can ask of each knob separately:
        if I nudge this one parameter up a tiny bit, does the cost go up or down, and
        how steeply? That steepness is the parameter's slope, the same idea as the m
        in y = mx + c, just measured at the point where you currently stand. The full
        list of slopes, one per parameter, is called the gradient and written{" "}
        <M tex="\nabla C" />. It points the way the cost rises fastest, so to descend,
        step every parameter the other way:
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
        Time to implement it. You write the update step and the epoch loop; the course
        supplies the gradient, computed numerically by nudging every parameter and
        measuring the cost twice. It is honest and correct, and you are about to
        discover its price.
      </p>

      <ExercisePage exercise={sgdExercise} />

      <p>
        Your sgd is real: point it at a toy dataset and it learns. But watch the clock,
        and count. Estimating the gradient numerically costs two cost evaluations per
        parameter per step, and each cost evaluation is a full forward pass over the
        batch. Seventeen parameters make that bill merely annoying. The MNIST network
        you will train in Module 5 has 23,860, and there the bill is fatal. What we
        need is a way to get every partial derivative from roughly one pass, and that
        is exactly what backpropagation is.
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
