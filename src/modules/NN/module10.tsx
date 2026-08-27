import { AfterThis, Aside, Figure, ModuleToc, SectionHeader } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExercisePage } from "../../components/ExercisePage";
import { prepareExercise } from "../../exercises/prepare";
import { PenguinsPanel } from "./interactives/PenguinsPanel";

export function Module10() {
  return (
    <article className="module">
      <h2>Module 10: Your own problem</h2>
      <AfterThis
        items={[
          "Turn a file of mixed numbers, categories and holes into the matrix your network expects, and say why each step was needed.",
          "Read a score against a baseline and a per-class breakdown rather than on its own.",
          "Pick a first network for a problem nobody has set up for you, and know what to check when it does not train.",
        ]}
      />
      <ModuleToc />

      <SectionHeader id="m10-file" title="A file, not a dataset" />
      <p>
        Every number your network has seen so far arrived ready. MNIST's pixels
        were already numeric, already between 0 and 1, already complete, already
        split into training and held-out, already labelled as one-hot columns.
        The course did all of that before you saw a single digit:{" "}
        <code>tools/make_mnist_subset.py</code> took the subset and kept MNIST's
        own split, and <code>src/python/data_loader.py</code> divided every
        pixel by 255 and packed the labels into one-hot columns. That
        preparation is most of the work in a problem of your own, and every step
        of it is a decision somebody makes (and can get wrong) before any
        network sees a number.
      </p>
      <p>
        Here is the file this module uses. It is a real survey of penguins from
        the Palmer Archipelago in Antarctica: three species, and for each bird a
        few measurements, the island it was found on, and its sex.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>species</th>
              <th>island</th>
              <th>bill length</th>
              <th>bill depth</th>
              <th>flipper</th>
              <th>body mass</th>
              <th>sex</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Adelie</td>
              <td>Torgersen</td>
              <td>39.1</td>
              <td>18.7</td>
              <td>181</td>
              <td>3750</td>
              <td>male</td>
            </tr>
            <tr>
              <td>Adelie</td>
              <td>Torgersen</td>
              <td>39.5</td>
              <td>17.4</td>
              <td>186</td>
              <td>3800</td>
              <td>female</td>
            </tr>
            <tr>
              <td>Adelie</td>
              <td>Torgersen</td>
              <td>(none)</td>
              <td>(none)</td>
              <td>(none)</td>
              <td>(none)</td>
              <td>(none)</td>
            </tr>
            <tr>
              <td>Gentoo</td>
              <td>Biscoe</td>
              <td>46.1</td>
              <td>13.2</td>
              <td>211</td>
              <td>4500</td>
              <td>female</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Four problems are already visible, and none of them exists in MNIST. Two
        of the columns hold words rather than numbers. The third row is empty:
        344 penguins were surveyed, two of them have no measurements at all and
        eleven have no recorded sex. The species do not appear in equal numbers
        (152 Adelie, 124 Gentoo, 68 Chinstrap), so a network that always answers
        Adelie is right more than four times in ten. And the measurements are on
        wildly different scales: body mass averages 4,202 while bill depth
        averages 17.15, which is a factor of 245.
      </p>

      <SectionHeader id="m10-shape" title="What your network expects" />
      <p>
        The target is the packing every module since Module 3 has used. One
        column per example, one row per input number, and a matching column of
        right answers:
      </p>
      <Eq
        tex="X \;\text{is}\; (\text{inputs},\, m), \qquad Y \;\text{is}\; (\text{outputs},\, m)"
        gloss="Example k is column k of X, and its right answer is column k of Y. A row of X is one input across every example: one pixel in Module 2, one measurement here. Everyone outside this course calls one such row a feature, and this module uses that word from here on, since the file it prepares arrives with some of its features written as words."
      />
      <p>
        The outputs, and the cost that scores them, follow from what you are
        asking rather than from taste. Three cases cover almost everything, and
        you have met all three:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>what you are asking</th>
              <th>output layer</th>
              <th>cost</th>
              <th>the right answer is</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>yes or no</td>
              <td>one sigmoid neuron</td>
              <td>cross-entropy</td>
              <td>a single 0 or 1</td>
            </tr>
            <tr>
              <td>which one of several (this module)</td>
              <td>one sigmoid per class, or softmax</td>
              <td>cross-entropy</td>
              <td>a one-hot column</td>
            </tr>
            <tr>
              <td>how much</td>
              <td>one neuron, no squash</td>
              <td>quadratic</td>
              <td>the quantity itself</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The first two rows are your digit reader with a different number of
        outputs. The third is the one departure the course has already shown you:
        Module 6's rating over a dial ended in a neuron that reported its total
        with no squash, because an answer between 0 and 1 is the wrong shape for
        a quantity. Predicting a house price or a temperature is that row, and
        the quadratic cost from Module 3 is what scores it.
      </p>

      <SectionHeader id="m10-scale" title="The step that decides everything" />
      <p>
        Now the measurements. A hidden neuron computes{" "}
        <M tex="w \cdot x + b" />, so every weight meets whatever numbers its own
        row of <M tex="X" /> happens to hold. Feed the penguins in as they come
        and one weight meets body masses near 4,200 while its neighbour meets
        bill depths near 17.
      </p>
      <p>
        Module 7 already worked out what that does. Its whole argument for
        dividing the starting weights by <M tex="\sqrt{n}" /> counted about a
        hundred lit pixels, each contributing a value near 1, and concluded that
        the evidence piles up to about the square root of the squared input
        values added together, which for those images gave 9.3. Here is that
        same sum on one average penguin, straight out of the file:
      </p>
      <Eq
        tex="\sqrt{\underbrace{4200^2}_{\text{mass}} + \underbrace{200^2}_{\text{flipper}} + \underbrace{44^2}_{\text{bill length}} + \underbrace{17^2}_{\text{bill depth}}} = \sqrt{17{,}682{,}225} \approx 4{,}200"
        gloss="The four averages from the file, squared and added, then square-rooted: that is the spread of a hidden neuron's evidence when its weights are drawn at spread 1. Module 7's draw divides those weights by the square root of the input count, which is 3 for nine input rows. Dividing 4,200 by 3 still leaves the evidence near 1,400, where the same division landed MNIST's at 0.33."
      />
      <p>
        Every hidden neuron is saturated flat before training starts, and the
        sigmoid overflows on the way. The fix is the same fix, applied at the
        other end: instead of shrinking the weights to match the inputs, put the
        inputs on a scale the weights were designed for.
      </p>
      <Eq
        tex="x' = \frac{x - \text{mean}}{\text{spread}}"
        gloss="For each feature separately: subtract that feature's average across the training examples, then divide by its typical distance from that average. Every feature comes out centred on 0 and about 1 wide, whatever it was measured in. The name for it is standardizing."
      />
      <p>
        One detail in that formula decides whether your final number means
        anything. The mean and the spread must be measured on the training
        examples alone and then applied unchanged to everything else. Measure
        them over the whole file and the validation and test rows have quietly
        contributed to how the training rows were scaled, which is Module 7's
        lesson about optimizing against a number you also report, wearing a
        different hat.
      </p>
      <Aside>
        <p>
          Why not leave the file alone and shrink the weights instead? Divide
          the starting draw by four thousand and body masses come out near 1,
          which fixes that row. Bill depths arrive near 17, so the same division
          leaves bill depth contributing about 0.004 where body mass contributes
          1, and the network is deaf to it. One division cannot serve two rows
          that are 245 times apart: a layer has one weight scale, while
          standardizing gives every row its own mean and spread. You could pick
          a scale per row by hand, which is this formula in disguise, done where
          nobody can check it.
        </p>
      </Aside>

      <SectionHeader id="m10-categories" title="Words, holes and splits" />
      <p>
        The island column holds one of three words. A network cannot multiply a
        word, and numbering them 1, 2, 3 would be worse than useless: it would
        tell the network that Torgersen is three times Biscoe and that Dream sits
        between them. The honest encoding is the one your labels have used since
        Module 5. Give each level its own input row, and put a 1 in the row that
        applies:
      </p>
      <Eq
        tex="\text{Dream} \;\longrightarrow\; \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} \quad \text{(Biscoe, Dream, Torgersen)}"
        gloss="One row per level, a single 1 per column. This is one-hot encoding, the same packing as a one-hot label, used on the input side. Three islands cost three input rows."
      />
      <p>
        Holes need a decision rather than a formula. The two penguins with no
        measurements at all cannot be fed to anything, so they leave. The eleven
        with no recorded sex still have every measurement, so throwing them away
        would cost real information to avoid a small gap; instead their sex
        column becomes all zeros, which says none of these rather than guessing.
        Both choices are judgement, and the useful habit is to write down which
        one you made and why.
      </p>
      <p>
        Scaling and one-hot rows together turn the first penguin in the file at
        the top of this page into one column of nine rows, and that column is
        what your network reads:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>rows of X</th>
              <th>what they hold</th>
              <th>the Adelie from Torgersen</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1 to 4</td>
              <td>the four measurements, standardized</td>
              <td>
                body mass, row 4, is 3,750. The masses average 4,202 with a
                spread of about 800, the two numbers your <code>standardize</code>{" "}
                computes for that row, so this bird sits 452 below the average, a
                bit over half a spread, and the row reads about -0.6
              </td>
            </tr>
            <tr>
              <td>5, 6, 7</td>
              <td>Biscoe, Dream, Torgersen</td>
              <td>0, 0, 1</td>
            </tr>
            <tr>
              <td>8, 9</td>
              <td>female, male</td>
              <td>0, 1</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Then the split, and it has to be shuffled. This file is sorted by
        species, the way files usually arrive sorted by something. Cut it 60, 20
        and 20 without shuffling and the training rows are 151 Adelie and 54
        Gentoo, while the held-back rows are 68 Chinstrap and a single Gentoo: a
        network trained on two species and scored on a third it has never seen.
        Shuffling once, before cutting, is the whole defence.
      </p>
      <Aside>
        <p>
          Three splits, not two, for the reason Module 7 gave. The validation
          rows are the ones you may look at as often as you like, because
          choosing the step size or the layer size against them is what they are
          for. The test rows are spent once, at the end, to report. In this
          module the panel scores validation after every epoch and touches the
          test rows exactly once, when the run is over.
        </p>
      </Aside>

      <ExercisePage exercise={prepareExercise} />

      <SectionHeader id="m10-run" title="What the preparation was worth" />
      <p>
        The panel below runs the whole pipeline: your <code>standardize</code>,{" "}
        <code>one_hot</code> and <code>split</code> turn the file into a matrix,
        and your <code>train</code> from Module 9 does the rest.
      </p>
      <Figure caption="The penguin file, prepared by your code and trained by your loop. The switches change the preparation, not the network: same shape, same step size, same epochs, same seeds.">
        <PenguinsPanel />
      </Figure>
      <p>
        Turn the scaling off first. The network reads 42.6 percent of the 68
        held-back penguins, and answering Adelie for every one of them would have
        read 42.6 percent too, because 29 of those 68 are Adelie. (The whole file
        is 44.2 percent Adelie, 152 of 344, so one shuffled fifth of it landing a
        little under that is ordinary.) Turn the scaling on and the same network,
        on the same rows, reads 100 percent, with 80.9 percent after a single
        epoch. One preprocessing step is the difference between a network that
        works and a network that does not.
      </p>
      <p>
        Why does the unscaled network match that score exactly, rather than
        landing somewhere worse? A body mass in the thousands drives every hidden
        neuron flat, and to the same flat value for every bird, so the hidden
        layer hands the output layer the same column whichever penguin arrived.
        The output layer is reading a constant, and the best a constant answer
        can do is name the commonest species. The network cannot climb out of it
        either, for the reason Module 7 gave for dividing the starting weights: a
        flat neuron's steepness is near zero, and BP2 multiplies its blame by
        that steepness, so the wires feeding it barely move.
      </p>
      <p>
        A perfect score should make you suspicious rather than pleased, so why
        did this one happen? The three species really are far
        apart in these four measurements: a Gentoo is much heavier with a
        shallower bill, and Chinstraps and Adelies differ clearly in bill length.
        The problem is easy, and the honest report of an easy problem is that it
        was easy. The other thing that produces a perfect score is a leak
        (anything the held-back rows told you before you scored them, a mean and
        a spread included), which is what the training-only scaling and the
        shuffled split were guarding against.
      </p>

      <SectionHeader id="m10-errors" title="One number hides the failure" />
      <p>
        Now switch the features to bill depth and body mass alone, keeping the
        scaling on, and the score falls to 73.5 percent. Against a 42.6 percent
        baseline that sounds like a working model. The per-class row underneath
        says otherwise:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>two measurements, held-back penguins</th>
              <th>Adelie</th>
              <th>Chinstrap</th>
              <th>Gentoo</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>read correctly</td>
              <td>29 of 29</td>
              <td>0 of 18</td>
              <td>21 of 21</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        It never answers Chinstrap. Not rarely: never. All 18 are read as Adelie,
        and the 73.5 percent is the two easy species carrying the score. This is
        what a single number cannot tell you and a per-class count can, and the
        class it hid is the smallest one, which is where it usually hides.
      </p>
      <p>
        The cause is in the data rather than in the network. Chinstraps and
        Adelies have almost the same body mass and almost the same bill depth;
        what separates them is bill length, which was the feature you just
        removed. With no way to tell them apart, answering Adelie every time is
        the best the network can do, because Adelie is the commoner of the two.
        It is behaving correctly and uselessly, and only the breakdown shows it.
      </p>
      <p>
        So: compute a baseline before you train anything, so the score has
        something to be measured against; then look at the score per class, and
        at what was confused with what. Both are a few lines, and between them
        they catch most of the ways a model can be quietly wrong.
      </p>

      <SectionHeader id="m10-first" title="Picking a first network" />
      <p>
        Nothing so far said how big the network should be. There is no equation
        for it, the way there was none for the step size, so here is a procedure
        that works.
      </p>
      <p>
        Start with one hidden layer. Module 8 measured what depth costs when you
        add it without needing it, and Module 6 showed that a single hidden
        layer, made wide enough, can already express whatever relationship you
        are asking the network for. That argument was about such weights existing
        rather than about descent finding them, so treat one layer as the
        cheapest thing to try rather than as the right answer. Size that layer
        somewhere between the number of inputs and the number of outputs: the
        penguin network is 9, 8 and 3, and MNIST's was 784, 30 and 10. Neither is
        optimal, and neither needed to be.
      </p>
      <Eq
        tex="\underbrace{8 \times 9}_{\text{hidden } W} + \underbrace{8}_{\text{hidden } b} + \underbrace{3 \times 8}_{\text{output } W} + \underbrace{3}_{\text{output } b} = 72 + 8 + 24 + 3 = 107"
        gloss="Counting the penguin network's knobs, by the same two products as Module 2's count: one weight per wire into a layer, one bias per neuron in it. The nine inputs are the four measurements plus one row per island and one per sex. The 784-30-10 digit reader you trained came to 23,860 knobs by this counting, and the whole penguin network comes to 107."
      />
      <p>
        Then prove the machinery works before you worry about accuracy: take
        twenty examples, train on them alone, and check the network can reach
        100 percent on those twenty. A network that cannot memorize twenty
        examples has a bug, a step size that is wrong, or inputs that were never
        scaled, and no amount of extra data will help until that is fixed. Once
        it can, put the data back and read the two curves Module 7 taught: if
        training and held-out are both poor, the network is too small or has not
        trained long enough; if training is good and held-out is not, you are
        overfitting and want more data, weight decay, or a smaller network.
      </p>

      <SectionHeader id="m10-diagnose" title="When it does not train" />
      <p>
        The failures in this course are the common ones, and they are
        distinguishable by symptom. The rows below run in the order worth
        checking, from the failures you can see in the cost to the ones that show
        up only in the scores.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>what you see</th>
              <th>what it usually is</th>
              <th>where you met it</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>the cost does not move at all</td>
              <td>inputs never scaled, or the step size far too small</td>
              <td>this module, and Module 3</td>
            </tr>
            <tr>
              <td>the cost jumps around or grows</td>
              <td>the step size is too large</td>
              <td>Module 3's bowl</td>
            </tr>
            <tr>
              <td>NaN appears and stays</td>
              <td>a logarithm of zero, or a step size that overflowed</td>
              <td>Module 7's clip</td>
            </tr>
            <tr>
              <td>the score equals the commonest class</td>
              <td>it has learned the baseline and nothing else</td>
              <td>this module, and Module 8</td>
            </tr>
            <tr>
              <td>training keeps improving, held-out stops</td>
              <td>overfitting</td>
              <td>Module 7</td>
            </tr>
            <tr>
              <td>both stop, well short of good</td>
              <td>too few neurons, or features that cannot separate the classes</td>
              <td>this module's Chinstraps</td>
            </tr>
            <tr>
              <td>the first layers barely move</td>
              <td>too deep for sigmoids</td>
              <td>Module 8</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Two checks are worth running before any of that, because they are the
        ones that separate a wrong network from wrong code. The first is the
        gradient check Module 5's tests ran against your backprop: nudge every
        parameter, rescore, and compare with the slopes your equations produced.
        The course wrote that one, but the recipe is Module 3's
        nudge-and-measure, so you can write it again for a network of your own.
        The check works on any network and any cost, and it is the only way to be
        sure the slopes you are stepping against are the slopes of the thing you
        are scoring. The second is the
        initial cost. Before training a network answers about 0.5 at every
        output, and your cost charges each output neuron on its own, so every
        neuron bills <M tex="-\ln 0.5 = 0.693" /> whether its right answer is 1
        or 0. Ten outputs come to 6.93 and three come to 2.08. Print the cost
        before the first step and check it against that count: a first cost far
        from it means the labels, the outputs or the cost are not lined up.
        (Networks with a softmax output layer report{" "}
        <M tex="\ln 10 = 2.30" /> here instead, because their cost charges the
        layer once rather than charging ten neurons, which is worth knowing
        before you compare your number with someone else's.)
      </p>

      <SectionHeader id="m10-torch" title="How this is done in practice" />
      <p>
        One honest note to end on. Everything you have written is what a
        framework does for you, and in a real project you would use the
        framework. Here is the same digit reader in PyTorch, beside the parts of
        it you built.
      </p>
      <p>
        This is the one piece of code in the course you cannot run where you read
        it: PyTorch will not load in this page, so this block is for reading and
        recognising rather than for pressing a button.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <pre className="torch-listing">{`model = nn.Sequential(          # your list of weights and biases
    nn.Linear(784, 30),         # w[0] @ a + b[0], drawn for you
    nn.Sigmoid(),               # your sigmoid
    nn.Linear(30, 10),
)
loss_fn = nn.CrossEntropyLoss() # your cross_entropy_cost
opt = torch.optim.SGD(model.parameters(), lr=0.5)   # your l2_step

for epoch in range(15):                    # your train
    for x, y in loader:                    # your shuffle and mini-batches
        loss = loss_fn(model(x), y)        # your feedforward, then the cost
        opt.zero_grad()
        loss.backward()                    # your backprop: BP1 to BP4
        opt.step()                         # your update rule`}</pre>
      </div>
      <p>
        Fourteen lines, and there is nothing in them you have not implemented.{" "}
        <code>nn.Linear</code> holds a weight matrix and a bias column and
        computes <M tex="Wa + b" />. <code>loss.backward()</code> is the backward
        sweep, worked out by the automatic differentiation Module 9 named, from
        a record of the forward pass rather than from equations you wrote.{" "}
        <code>opt.step()</code> is your update rule, and swapping{" "}
        <code>SGD</code> for <code>Adam</code> on that line is what changes the
        optimizer. The framework's contribution is not the ideas; it is that
        every one of them is written once, tested, fast, and able to run on a
        graphics card.
      </p>
      <p>
        Two conventions differ from this course and are worth knowing before you
        read anyone's code. Frameworks put examples in <em>rows</em> rather than
        columns, so their matrices are the transpose of yours and their layers
        compute <M tex="aW^T + b" />. And their cross-entropy takes the raw
        evidence rather than a squashed answer, folding the softmax inside the
        cost for numerical reasons, which is why the model above ends at{" "}
        <code>nn.Linear</code> with no final sigmoid.
      </p>
      <p>
        What you take to a framework is not the code. It is knowing what{" "}
        <code>lr</code> does to a run, why a loss that will not move is usually
        the inputs, what a per-class breakdown is for, and what{" "}
        <code>loss.backward()</code> is actually doing when it takes a
        millisecond. That is the difference between using a tool and operating
        one, and it is the whole reason this course made you write the parts.
      </p>
      <p className="m10-credit">
        The penguin data is from the Palmer Station Antarctica LTER survey by Dr
        Kristen Gorman, published as the palmerpenguins package by Allison Horst,
        Alison Hill and Kristen Gorman, and released into the public domain
        (CC0). Rebuild the bundled copy with{" "}
        <code>python3 tools/make_penguins.py</code>.
      </p>
    </article>
  );
}
