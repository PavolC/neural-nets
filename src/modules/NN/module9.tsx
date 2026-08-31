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
          "Say what your file holds once that loop is in it, and what it takes to run the whole thing outside this page.",
          "Name the two jobs every framework does that you did by hand here, and say what each one automates.",
        ]}
      />
      <ModuleToc />

      <p>
        Open the workbench and press Download my nn.py. What lands is fifteen
        functions in the order you met them, eleven of them written by you:{" "}
        <code>sigmoid</code>, <code>feedforward</code>, <code>sgd_step</code>,{" "}
        <code>sgd</code>, <code>sigmoid_prime</code>, <code>backprop</code>,{" "}
        <code>cross_entropy_delta</code>, <code>init_network</code>,{" "}
        <code>l2_step</code>, and the four the course wrote for you. Run that file
        with <code>python3 nn.py</code> and nothing happens. Nothing is broken
        either: every line in it is a definition, and no line calls one.
      </p>
      <p>
        Nothing has called them on this page either. Every training run you have
        watched since Module 5 was started by a panel, and the panel did the
        loading, the epoch loop and the scoring around your code. This module is
        that loop, and it is one function long.
      </p>

      <SectionHeader id="m9-loop" title="The loop that was never yours" />
      <p>
        Count what those fifteen give you: a squash and one neuron (Module 1), a
        whole network's forward pass as one matrix multiplication per layer
        (Module 2), a descent step and the loop around it (Module 3), the four
        equations that turn one forward pass into every parameter's partial
        derivative (Module 5), and a cost, a starting draw and an update rule
        (Module 7). Every training run in this course has been made of those
        functions. The loop around them, loading the images, walking the epochs,
        scoring each pass against the held-out thousand, was always the course's.
      </p>
      <p>
        Module 3's <code>sgd</code> came closest: the right loop, shuffle, cut,
        step, repeat, running on your own gradient in every panel that has trained
        it since Module 5. But the course only ever called it one epoch at a time,
        and it has never drawn the network it trains, never scored one, and never
        taken a decayed step. In your file, <code>sgd</code> still calls Module 3's
        nudge-and-measure <code>gradient</code>, and <code>train</code>, the
        function you are about to write, is the one that reaches your{" "}
        <code>backprop</code>.
      </p>
      <p>
        So write it. The exercise below is two functions, and the second one is the
        program: draw a network, run the epochs, and hand back what it learned
        along with a score after each one.
      </p>
      <p>
        The first, <code>accuracy</code>, answers one question: out of a batch of
        images, what share did the network read right? You have never written it,
        because every accuracy figure in eight modules was computed for you, but
        every ingredient is from Module 2. <code>feedforward</code> on a batch
        returns ten confidences per image, one column each, and row 0 holds the
        "is it a 0?" neuron's answer for every image, row 7 the "is it a 7?"
        neuron's. So the network's verdict on an image is a row number: the row
        where its column peaks. Here is the whole computation on a batch of three:
      </p>
      <Figure caption="The whole of accuracy, run on three images. A starred entry is its column's largest, and its row number is the network's verdict for that image.">
        <div className="table-scroll scroll-x" tabIndex={0}>
          <pre className="torch-listing">{`out = feedforward(weights, biases, X)   three images in, so out is (10, 3)

           image 1  image 2  image 3
  row 0    0.03     0.91*    0.02
  row 1    0.05     0.02     0.11
  row 2    0.01     0.04     0.08
  row 3    0.86*    0.01     0.07
  row 4    0.02     0.03     0.30
  row 5    0.04     0.02     0.09
  row 6    0.01     0.01     0.05
  row 7    0.11     0.02     0.44*
  row 8    0.03     0.08     0.12
  row 9    0.02     0.05     0.38

guesses = np.argmax(out, axis=0)  ->  array([3, 0, 7])   the winning rows
y                                     array([3, 0, 9])   the right answers
guesses == y                      ->  array([True, True, False])
(guesses == y).mean()             ->  0.6666...          two of three`}</pre>
        </div>
      </Figure>
      <p>
        Two of those lines are new notation. <code>np.argmax(out, axis=0)</code>,
        read "arg max", the argument at which a maximum occurs, hands back each
        column's winning row: the position of the largest value, not the value
        sitting there. And the right answers arrive as <code>y</code>, one plain
        integer per image rather than the one-hot columns training uses, so the
        comparison is integer against integer, verdict against answer.{" "}
        <code>guesses == y</code> checks every image at once; True counts as 1 and
        False as 0 in arithmetic, so the mean of the comparison is the share read
        correctly. Image 3 shows what the single number hides: 0.44 over 0.38 is a
        coin flip of a verdict, and it counts against the score exactly as a
        confident miss would.
      </p>

      <ExerciseCard exercise={trainExercise} />

      <SectionHeader id="m9-run" title="Your program, on the digits" />
      <p>
        Your <code>train</code> imports nothing. Every name it calls is defined
        higher up the same file, and apart from the shuffle NumPy performs and the
        mini-batch adapter written for you in Module 5, every line in the map below
        is code you wrote. The panel runs your file and nothing is substituted
        underneath it, which is what nine modules of writing code into one file
        were for.
      </p>
      <Figure caption="Your program's call map, indented by who calls whom, each line labeled with who wrote it and where.">
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
      <p>
        The panel below runs it on the real thing: 5,000 training images scored
        against the held-out thousand, at the settings printed under its button.
        Run it at 0.5 first, the middle of the three step sizes, and let all
        fifteen epochs finish. The number worth reading is where the chart settles,
        not the best single epoch.
      </p>
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
        Only one of the other two settings reaches past Module 7's grid: 3.0
        re-runs one of its four columns with the decay switched on, and 0.1 is
        below every step size that table tried. At 0.1 the
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

      <SectionHeader id="m9-program" title="A file that runs anywhere" />
      <p>
        Press Download my nn.py again. The file is the same fifteen functions plus
        the two you just wrote, and running it still prints nothing, because it is
        still all definitions. What changed is what those definitions add up to.
        Three lines at a Python prompt now train a network:
      </p>
      <Figure caption="Your downloaded file, driven from outside this page. The import needs NumPy installed; X, Y, X_test and y_test are whatever data you put in the packing Module 2 fixed, and Module 10 is about producing them from a file nobody prepared.">
        <div className="table-scroll scroll-x" tabIndex={0}>
          <pre className="torch-listing">{`>>> import numpy as np
>>> from nn import train, accuracy
>>> w, b, history = train([784, 30, 10], X, Y, X_test, y_test,
...                       epochs=15, eta=0.5, lmbda=1.0,
...                       batch_size=10, rng=np.random.default_rng(1))
>>> history[-1]
0.904`}</pre>
        </div>
      </Figure>
      <p>
        Two things that were true a moment ago are not true now. Before{" "}
        <code>train</code> existed, nothing in the file called anything else in it,
        so the only way to run any of it was a panel on this page holding the loop.
        And the browser was doing the arithmetic: Pyodide, CPython compiled to
        WebAssembly, which is why a run of the digit reader takes a few seconds an
        epoch here. The same file on a machine with ordinary NumPy is the same code
        at ordinary speed. Nothing about the download is a demo version: it is about
        250 lines of Python with one import in it, and the import is NumPy.
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

      <SectionHeader id="m9-words" title="Which words were only ours" />
      <p>
        Each module named the field's word for its own ideas as it went. The squash
        is an activation function and the line a neuron cuts is its decision
        boundary (Module 1); a knob is a parameter, the cost is the loss, and one
        knob's slope is a partial derivative (Module 3); a factor is a local
        derivative and blame is the error (Module 4); receipts are the cached
        forward pass (Module 5); the divided start is Xavier initialization and the
        decay factor is L2 regularization (Module 7); a layer's learning speed is
        its gradient norm, and a hop well under 1 is the vanishing gradient problem
        (Module 8). The whole lookup, both ways round, is folded into the start page
        under "Notation and NumPy reference".
      </p>
      <p>
        What is worth listing instead is the other set: the words this course made
        up, which have no counterpart anywhere. Nobody else uses these, so there is
        nothing to search for and nothing to translate. They were carriers, and
        they can be put down here.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>ours only</th>
              <th>what it was carrying</th>
              <th>from</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>the concert, the contrarian, the easygoing and picky voters</td>
              <td>
                one dataset carried through nine modules, so that a new idea never
                arrived with a new world attached
              </td>
              <td>Module 1</td>
            </tr>
            <tr>
              <td>booths, posted rates, the through-rate</td>
              <td>
                the chain rule, as a thing you could already do: two currency
                booths and a raise
              </td>
              <td>Module 4</td>
            </tr>
            <tr>
              <td>the road and its on-ramps</td>
              <td>
                why every parameter at one neuron shares a price, which is the
                whole reason <M tex="\delta" /> is worth computing
              </td>
              <td>Module 4</td>
            </tr>
            <tr>
              <td>the dial, and slicing it</td>
              <td>
                one input moving while 783 are held still, which is what makes
                universality drawable
              </td>
              <td>Module 6</td>
            </tr>
            <tr>
              <td>bars, towers, boxes</td>
              <td>
                sigmoid pairs assembled by hand into a lookup table, one dimension
                at a time
              </td>
              <td>Module 6</td>
            </tr>
            <tr>
              <td>the hop</td>
              <td>
                one backward step of BP2 as a single number, so that depth becomes
                that number raised to a power
              </td>
              <td>Module 8</td>
            </tr>
          </tbody>
        </table>
      </div>

      <SectionHeader id="m9-missing" title="The two jobs you did by hand" />
      <p>
        Two things in your file are things nobody writes any more. Both are worth
        naming here, because both are waiting in the first framework you open, and
        in both cases what the framework replaced is code you have now written.
      </p>
      <p>
        <b>Nobody writes backprop by hand.</b> Every framework computes gradients by
        recording the operations of the forward pass as they happen and then
        applying the chain rule to that record, mechanically, whatever the network
        turns out to be. The name for it is automatic differentiation, and what it
        automates is exactly what you did in Module 5: your <code>zs</code> and{" "}
        <code>activations</code> lists are a hand-built version of that record, and
        Module 5 called them the cached forward pass for this reason. This is why a
        modern network can have a shape nobody wrote a backward pass for.
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
        Two more things are missing, and both are the next page rather than this
        list. Every number your network has seen arrived prepared: pixels already
        between 0 and 1, labels already one-hot, the split already made, every
        example complete. And your run above misreads about a hundred of the
        thousand held-out digits, which this course has broken apart exactly once,
        in Module 5's panel. Looking once is a demonstration rather than a habit.
      </p>
      <p>
        Module 10 points the program you just assembled at a file with none of the
        preparation done, and makes reading a score apart a step of its own, on a
        problem where 73.5 percent turns out to hide a class the network never once
        answers. One part of that job is load-bearing for what you learned here:
        Module 7's argument for dividing the weights by <M tex="\sqrt{n}" /> counted
        about a hundred lit pixels each contributing a value near 1, so inputs that
        run to the thousands break that argument, and the usual fix is to scale the
        inputs rather than the weights.
      </p>
    </article>
  );
}
