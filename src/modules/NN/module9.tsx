import { AfterThis, Aside, Figure, ModuleToc, SectionHeader } from "../../components/ModuleBits";
import { M } from "../../components/Math";
import { ExerciseCard } from "../../components/ExerciseCard";
import { trainExercise } from "../../exercises/train";
import { FullTrainPanel } from "./interactives/FullTrainPanel";

export function Module9() {
  return (
    <article className="module">
      <h2>Module 9: Assembling the program</h2>
      <AfterThis
        items={[
          "Write the loop that ties every function you have built into one program, and run it on the digit reader.",
          "Say what each of this course's words is called everywhere else, so that the next thing you read is readable.",
          "Name what this course did not teach you, and pick what to read next.",
        ]}
      />
      <ModuleToc />

      <SectionHeader id="m9-loop" title="The loop that was never yours" />
      <p>
        Count what you have written: a squash and one neuron (Module 1), a whole
        network's forward pass as one matrix multiplication per layer (Module 2),
        and a descent step and the loop around it (Module 3). Then the four
        equations that turn one forward pass into every knob's slope (Module 5), and
        a cost, a starting draw and an update rule (Module 7). Every training run in
        this course has been made of those functions.
      </p>
      <p>
        And every one of those runs was started by a panel. The panel loaded the
        images, drew the network, walked the epochs, cut the mini-batches, called
        your gradient, applied your update, scored the result against the held-out
        thousand and drew the chart. Your functions did the work inside that loop.
        The loop was the course's.
      </p>
      <p>
        Module 3 is the exception that shows the shape of the gap. You wrote a loop
        there, <code>sgd</code>, and it is the right loop: shuffle, cut, step,
        repeat. It has been running on your own gradient since Module 5, where the
        panel pointed its <code>gradient</code> at the adapter around your backprop,
        and Module 7's first two panels ran it as well. But every one of those
        panels called it with epochs set to 1, once per epoch, so the walk through
        the epochs stayed the panel's. And your <code>sgd</code> has never drawn the
        network it trains, never scored one, and never taken a decayed step: Module
        7's decay panel called your <code>l2_step</code> from a loop of its own.
      </p>
      <p>
        So write it. The exercise below is two functions, and the second one is the
        program: draw a network, run the epochs, and hand back what it learned
        along with a score after each one.
      </p>
      <p>
        The first is smaller and has been missing for longer.{" "}
        <code>accuracy</code> takes a network and a batch of images and reports the
        share it reads correctly. Module 2 said how: ten output neurons, each
        answering its own yes-or-no question, and the network's verdict is whichever
        one is most confident. In NumPy that is a single call,{" "}
        <code>np.argmax(out, axis=0)</code>, which hands back the row number of the
        largest value in each column (argmax is read "arg max", the argument at
        which a maximum occurs, so what comes back is the row rather than the
        confidence sitting in it). You have never written it, because every accuracy
        figure in eight modules was computed for you.
      </p>

      <ExerciseCard exercise={trainExercise} />

      <SectionHeader id="m9-run" title="Your program, on the digits" />
      <p>
        The panel below runs it on the real thing: 5,000 training images and the
        held-out thousand, at the settings printed under its own button. Run it at
        0.5 first, the middle of the three step sizes, and let all fifteen epochs
        finish. The number worth reading is where the chart settles, not the best
        single epoch. Your <code>train</code> imports nothing. Every
        name it calls is defined higher up the same file: the draw from Module 7,
        the blame from Module 7, the decayed step from Module 7, the mini-batch
        adapter written for you in Module 5, the backprop under that from Module 5,
        and the forward pass your <code>accuracy</code> reads the ten confidences
        from, from Module 2. The panel runs your file and calls your{" "}
        <code>train</code>; nothing is substituted underneath it.
      </p>
      <p>
        That is worth one sentence of its own, because it is what nine modules of
        writing code into one file were for. Every arrow in the map below, except
        the shuffle NumPy performs and the adapter marked as written for you, ends
        at something you wrote. The panel says so itself under its results: when it
        has borrowed nothing it prints that it ran entirely on your own code.
      </p>
      <Figure caption="Your program's call map, indented by who calls whom. One name in it is not yours, the mini-batch adapter, and your file carries it under a section line that says it was written for you. Everything else is code you wrote, read from your own file in the order you wrote it.">
        <div className="table-scroll scroll-x" tabIndex={0}>
          <pre className="torch-listing">{`train(sizes, X, Y, X_test, y_test, ...)   yours, this module
    init_network(sizes, rng)              yours, Module 7
    each epoch: rng.permutation(n)        NumPy's shuffle
        each mini-batch:
            batch_gradient(...)           written for you, Module 5
                backprop(...)             yours, Module 5
                    cross_entropy_delta   yours, Module 7
            l2_step(...)                  yours, Module 7
        accuracy(w, b, X_test, y_test)    yours, this module
            feedforward(w, b, X)          yours, Module 2`}</pre>
        </div>
      </Figure>
      <Figure caption="Your whole program on the digit reader, with the step size in your hands. Each epoch takes a few seconds, and Stop ends the run.">
        <FullTrainPanel />
      </Figure>
      <p>
        At a step size of 0.5 it averages about 90.4 percent over the last five
        epochs, while Module 7's run of the same network reported 92.1. Did
        assembling the program yourself cost you almost two points? No: the decay is
        switched on here (lambda 1, which Module 7's run did not use), and one
        generator now does both jobs, the draw and the shuffles, so the mini-batches
        fall in a different order than they did there. Same program, same data, a
        different stream through it. Module 7 measured what a different stream alone
        is worth: three shuffles of one unchanged setting landed between 85.4 and
        86.4 percent there, a point of wobble with nothing changed at all.
      </p>
      <SectionHeader id="m9-step-size" title="The step size, from the inside" />
      <p>
        The other two settings reach past Module 7's grid: 3.0 is one of its four
        columns, and 0.1 is smaller than any step size that table tried. At 0.1 the
        first epoch reads 77.2 percent against 0.5's 85.5, and by
        the end the two are level: a smaller step is not a worse answer here, only
        a slower start. At 3.0 the last five epochs average 86.6, and the per-epoch
        line stops being a curve: 80.9, then 89.3, then 85.1 on consecutive passes.
        That is the overshoot Module 3 drew on a bowl, at the scale of a real
        network.
      </p>
      <Aside>
        <p>
          Worth naming, since you can now do it in one click: choosing the step size
          by looking at that chart is choosing against the number you are reporting.
          Module 7 said what to do about it, and this is the moment the temptation is
          real. If you were reporting this result to someone, you would pick the step
          size on a validation split and quote the held-out score once, at the end.
        </p>
      </Aside>

      <SectionHeader id="m9-words" title="The same ideas, in everyone else's words" />
      <p>
        This course chose plain words on purpose. Blame, steepness, evidence, the
        wire ledger, the bill: each was picked so that a sentence could be read
        without a glossary. Almost nobody else uses those words, and that becomes a
        problem on the first page of the next thing you open. The table below gives
        every one of them its field name, and the module where you first met it.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>in this course</th>
              <th>everywhere else</th>
              <th>first met</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>evidence, <M tex="z" /></td>
              <td>the weighted input, or the pre-activation</td>
              <td>Module 1</td>
            </tr>
            <tr>
              <td>the squash</td>
              <td>the activation function</td>
              <td>Module 1</td>
            </tr>
            <tr>
              <td>a neuron's confidence, <M tex="a" /></td>
              <td>its activation, or its output</td>
              <td>Module 1</td>
            </tr>
            <tr>
              <td>the line a neuron cuts</td>
              <td>
                its decision boundary, or a hyperplane once there are more than two
                inputs
              </td>
              <td>Module 1</td>
            </tr>
            <tr>
              <td>the wire ledger, <M tex="W" /></td>
              <td>the weight matrix</td>
              <td>Module 2</td>
            </tr>
            <tr>
              <td>a knob</td>
              <td>a parameter</td>
              <td>Module 3</td>
            </tr>
            <tr>
              <td>the score, the cost</td>
              <td>the loss, or the objective function</td>
              <td>Module 3</td>
            </tr>
            <tr>
              <td>one knob's slope</td>
              <td>a partial derivative, <M tex="\partial C / \partial w" /></td>
              <td>Module 3</td>
            </tr>
            <tr>
              <td>nudge-and-measure</td>
              <td>finite differences, a numerical gradient</td>
              <td>Module 3</td>
            </tr>
            <tr>
              <td>steepness, <M tex="\sigma'(z)" /></td>
              <td>the activation function's derivative</td>
              <td>Module 4</td>
            </tr>
            <tr>
              <td>a posted rate, a factor</td>
              <td>a local derivative</td>
              <td>Module 4</td>
            </tr>
            <tr>
              <td>a neuron's blame, <M tex="\delta" /></td>
              <td>its error, or the error signal at that layer</td>
              <td>Module 4</td>
            </tr>
            <tr>
              <td>the four equations</td>
              <td>backpropagation, or just backprop</td>
              <td>Module 4</td>
            </tr>
            <tr>
              <td>receipts</td>
              <td>the cached forward pass</td>
              <td>Module 5</td>
            </tr>
            <tr>
              <td>the bill</td>
              <td>computational cost, usually counted in operations</td>
              <td>Modules 3 and 4</td>
            </tr>
            <tr>
              <td>the divided start</td>
              <td>scaled initialization (the Xavier and He schemes)</td>
              <td>Module 7</td>
            </tr>
            <tr>
              <td>the decay factor</td>
              <td>L2 regularization, or weight decay</td>
              <td>Module 7</td>
            </tr>
            <tr>
              <td>a layer's learning speed</td>
              <td>the norm of its gradient</td>
              <td>Module 8</td>
            </tr>
            <tr>
              <td>a hop well under 1</td>
              <td>the vanishing gradient problem</td>
              <td>Module 8</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Two of those rows are worth a sentence each. A partial derivative is the
        thing you have been computing since Module 3: the slope of the cost for one
        knob, with every other knob held still. The course measured them by nudging,
        then computed them exactly with the chain rule, and never used the word.
        And "loss" is simply the commoner name for the cost; this course said cost
        throughout because a bill is easier to picture than a loss.
      </p>

      <SectionHeader id="m9-built" title="What you built" />
      <p>
        In Module 1 a neuron was a picture on a page. Here is what is running in
        your browser now, all of it written by you:
      </p>
      <ul className="m9-built-list">
        <li>
          <b>sigmoid and one neuron</b> (Module 1), the squash and the
          weighted-sum-plus-bias every layer since has been made of.
        </li>
        <li>
          <b>feedforward</b> (Module 2), a whole network as one matrix
          multiplication per layer, running 784-30-10 on real digits.
        </li>
        <li>
          <b>sgd_step and sgd</b> (Module 3), gradient descent with mini-batches
          and a shuffle.
        </li>
        <li>
          <b>backprop</b> (Module 5), BP1 to BP4 in about fifteen lines, checked
          entry by entry against numerically measured gradients, and training
          784-30-10 to 89 percent in seconds where Module 5's own timing put the
          nudge-and-measure route at hours.
        </li>
        <li>
          <b>cross_entropy_delta, init_network and l2_step</b> (Module 7), three
          one-line changes worth about three points of accuracy and a great deal of
          stability.
        </li>
        <li>
          <b>train and accuracy</b> (this module), the loop that runs all of the
          above and the score that says whether it worked.
        </li>
      </ul>
      <p>
        That is the whole of Nielsen's Chapters 1 to 3 as working code, plus the
        parts of Chapters 4 to 6 that are best understood by argument. The digit
        reader in this browser is not a toy version of a real network. It is a small
        real one, and now it is a program rather than a collection of parts.
      </p>

      <SectionHeader id="m9-missing" title="What this course did not teach you" />
      <p>
        Four things, named plainly, because each one is waiting in the next thing
        you read, and none of the eight modules behind you made you do it yourself.
      </p>
      <p>
        <b>Nobody writes backprop by hand.</b> Every framework computes gradients by
        recording the operations of the forward pass as they happen and then
        applying the chain rule to that record, mechanically, whatever the network
        turns out to be. The name for it is automatic differentiation, and what it
        automates is exactly what you did in Module 5: your <code>zs</code> and{" "}
        <code>activations</code> lists are a hand-built version of that record. This
        is why a modern network can have a shape nobody wrote a backward pass for.
      </p>
      <p>
        <b>Nobody uses plain descent either.</b> The update you wrote steps against
        the current gradient and forgets it. The ones in use keep some memory:
        momentum adds a fraction of the previous step to this one, so a consistent
        direction builds speed and Module 3's zigzag across a valley cancels itself
        out; Adam, the common default, additionally gives each parameter its own
        step size based on the recent size of its own gradient. Those are names, not
        explanations, and the arithmetic underneath is the update rule you wrote
        with two extra running totals beside it.
      </p>
      <p>
        <b>The data arrived prepared.</b> Pixels were already between 0 and 1,
        labels already one-hot, the split into training and held-out already made.
        Module 10 is the one item on this list that the course goes on to teach. It
        points the functions you just assembled at a file with words in two of its
        columns, holes in another, and two measurements 245 times apart in scale. A
        real problem asks for that work again every time, in whatever shape the file
        happens to arrive in. One part of that job is load-bearing for what you
        learned here: Module 7's argument for
        dividing the weights by <M tex="\sqrt{n}" /> counted about a hundred lit
        pixels each contributing a value near 1. Inputs that run to the thousands
        would break that argument, and the usual fix is to scale the inputs rather
        than the weights.
      </p>
      <p>
        <b>Accuracy is one number.</b> Your run above misreads about a hundred of
        the thousand held-out digits, and this course has broken a score like that
        apart exactly once: Module 5's panel put a per-digit row and the eight
        mistakes the network was most confident about under its chart. Looking once
        is a demonstration rather than a habit. Which digits it confuses, whether it
        is confidently wrong or barely wrong, and what the misread images have in
        common are all answerable with the code you have, and looking is what
        improves a real model fastest. Module 10 makes it a step of its own, on a
        smaller problem where a 73.5 percent score turns out to hide a class the
        network never once answers.
      </p>

      <SectionHeader id="m9-next" title="Where to go next" />
      <p>
        Module 10 is still ahead of you, and it is the shortest step of anything on
        this list: the same program you just ran, pointed at a file that nobody
        prepared for you. After that, in rough order of how far each one is from
        where you now stand:
      </p>
      <ul className="m9-next-list">
        <li>
          <a href="http://neuralnetworksanddeeplearning.com/chap5.html">
            Nielsen, Chapter 5
          </a>{" "}
          and{" "}
          <a href="http://neuralnetworksanddeeplearning.com/chap6.html">Chapter 6</a>.
          The two chapters Module 8 summarized, at full length: the unstable
          gradient argument with the algebra written out, then convolutional
          networks built up to above 99 percent on MNIST. The book this course is
          adapted from, and the smallest step from here.
        </li>
        <li>
          <a href="https://karpathy.ai/zero-to-hero.html">
            Andrej Karpathy, Neural Networks: Zero to Hero
          </a>
          . A video series that builds automatic differentiation from scratch, then
          a language model, in the same implement-it-yourself spirit. It starts with
          the first item in the list above this one, and ends where attention
          begins, which is exactly the gap this course leaves.
        </li>
        <li>
          <a href="https://pytorch.org/tutorials/beginner/basics/intro.html">
            The PyTorch tutorials
          </a>
          . What all of this looks like when the backward pass and the optimizer are
          written for you. Worth doing once you can say what they are doing on your
          behalf, which you now can.
        </li>
        <li>
          <a href="https://jalammar.github.io/illustrated-transformer/">
            Jay Alammar, The Illustrated Transformer
          </a>
          , and then the paper it explains,{" "}
          <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a>.
          The one ingredient Module 8 named and did not cover.
        </li>
        <li>
          <a href="https://www.deeplearningbook.org/">
            Goodfellow, Bengio and Courville, Deep Learning
          </a>
          . The reference textbook, free online. Heavier on mathematics than
          anything here, and the place to look up a technique properly.
        </li>
      </ul>
      <p>
        Whichever you open, the arithmetic will be the arithmetic in the table
        above. A column of numbers goes in, a matrix multiplies it, a squash bends
        it, a cost scores the answer, a chain of factors says which way each number
        should move, and a small step is taken. You have written all of it, and the
        rest of this field is that machine, built larger and arranged better.
      </p>
    </article>
  );
}
