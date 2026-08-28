import { AfterThis, Aside, Figure, ModuleToc, Recap, SectionHeader } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { ExerciseCard } from "../../components/ExerciseCard";
import { crossEntropyExercise } from "../../exercises/cross-entropy";
import { smartInitExercise } from "../../exercises/smart-init";
import { l2Exercise } from "../../exercises/l2";
import { BlameCurves } from "./interactives/BlameCurves";
import { CostSwapPanel } from "./interactives/CostSwapPanel";
import { InitStartPanel } from "./interactives/InitStartPanel";
import { OverfitFigure } from "./interactives/OverfitFigure";
import { RegularizePanel } from "./interactives/RegularizePanel";
import { SlowNeuron } from "./interactives/SlowNeuron";

export function Module7() {
  return (
    <article className="module">
      <h2>Module 7: Making it actually work</h2>
      <AfterThis
        items={[
          "Read one descent step as two links, find the one that goes slack when an output neuron is confidently wrong, and pick a scoring rule that covers for it.",
          "Change three lines of your own network (the output blame, the scale of the starting weights, the weight update) and measure what each one buys.",
          "Tell learning from memorizing by watching four numbers, and say what regularization does and does not fix.",
        ]}
      />
      <ModuleToc />

      <SectionHeader id="m7-plan" title="Three complaints" />
      <p>
        Module 5 finished a working digit reader: your feedforward, your sgd,
        your backprop, 89 percent of a thousand held-out digits read
        correctly. Nielsen's Chapter 1 trains this same 784-30-10 shape on the
        full 50,000 images and reports about 95 percent. Module 6 crossed one
        explanation for that gap off the list: a hidden layer, made wide
        enough, can express the pixels-to-digit rule, so the shape of the
        network is not what caps the score. Some of the gap is the bundled
        slice being ten times smaller, which is fixed here so the run fits in
        a browser tab. The rest is the setup, and this module makes three
        specific complaints about it.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>the complaint</th>
              <th>where it bites</th>
              <th>what changes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>an output neuron that is confidently wrong barely learns</td>
              <td>the last layer, throughout training</td>
              <td>BP1, the output layer's blame</td>
            </tr>
            <tr>
              <td>most hidden neurons start flat, before anything is learned</td>
              <td>the hidden layer, before the first step</td>
              <td>how the starting weights are drawn</td>
            </tr>
            <tr>
              <td>training keeps improving on the images it has and stops improving on the rest</td>
              <td>everywhere, once the data runs short</td>
              <td>the update rule for the weights</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Each fix is one line, and each one comes with a measurement of what it
        bought. To keep them one-liners, the course now hands your Module 5
        algorithm back to you with one step lifted out of it: the output
        layer's blame, BP1, is now something you pass in.
      </p>
      <div className="play-snippet">
        <pre>{`from course import backprop
nabla_w, nabla_b = backprop(weights, biases, x, y, output_delta)`}</pre>
      </div>
      <p>
        Same four equations, same forward pass keeping receipts, same backward
        sweep. Leave that last argument out and you get exactly what you wrote
        in Module 5. The first exercise below writes a different one.
      </p>

      <SectionHeader id="m7-slowdown" title="Badly wrong, barely learning" />
      <p>
        Start with one neuron, because the effect is easiest to see there and
        it is the same effect in the digit reader. One input, pinned at 1. One
        weight and one bias, both set to 2. The right answer is 0.
      </p>
      <Eq
        tex="\begin{gathered} z = \underbrace{2}_{\text{the weight}} \times \underbrace{1}_{\text{the input}} + \underbrace{2}_{\text{the bias}} = 4 \\[0.8em] a = \sigma(4) = 0.982 \end{gathered}"
        gloss="The input never moves, so the evidence is just the weight plus the bias, and the squash turns that 4 into an answer of 0.982."
      />
      <p>
        The neuron answers 0.982 when its job is to answer 0. Now train by
        descent, the same rule as everywhere else: measure both knobs' slopes,
        step against them at <M tex="\eta = 0.15" /> (eta, Module 3's step
        size), repeat. Here is the log, one line per checkpoint, and the figure
        below walks the same run at those settings.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>after this many steps</th>
              <th>1</th>
              <th>25</th>
              <th>100</th>
              <th>200</th>
              <th>300</th>
              <th>400</th>
              <th>500</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>the answer</td>
              <td>0.9819</td>
              <td>0.9794</td>
              <td>0.9641</td>
              <td>0.7887</td>
              <td>0.2028</td>
              <td>0.1217</td>
              <td>0.0930</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        A hundred steps move the answer by 0.018. It takes 469 steps to get
        below 0.1. And the neuron is not stuck at a bad spot in the
        landscape: given enough steps it walks all the way down, and the
        second half of the walk is fast. The slow part is the beginning, which
        is where it is most wrong.
      </p>
      <p>
        Start it closer to the answer instead: set the weight to 0.6 and the
        bias to 0.9, so the evidence is 1.5 and the answer is{" "}
        <M tex="\sigma(1.5) = 0.818" /> rather than 0.982. It reaches 0.1 in
        273 steps, so being more wrong at the start made the first stretch
        slower. That is the complaint, and the figure is where to check it. Its
        two chips are exactly these two starts. The last column of its table
        counts the steps to 0.1 for whatever weight and bias you drag to.
      </p>
      <Figure caption="One neuron learning to answer 0, with the input pinned at 1, under the quadratic cost. Press Play to walk the run, and drag w and b to start it somewhere else: the further the starting answer is from 0, the longer the flat stretch at the beginning.">
        <SlowNeuron />
      </Figure>
      <SectionHeader id="m7-flat" title="The land under the knobs" />
      <p>
        That run was a walk on a landscape, and it pays to be exact about what
        the landscape is. Module 3 drew it as a hiking map: the floor is every
        possible setting of the knobs, and the height above a spot is the
        score. To find the height at one spot, freeze the knobs there, run the
        images through the network as it stands, and add up the penalties. The
        land is already there at every setting, including the ones you will
        never visit. Descent does not build it. Descent walks on it.
      </p>
      <p>
        So measure it in two places. One neuron, the input pinned at 1, so the
        evidence is <M tex="w + b" />, and the score for an answer{" "}
        <M tex="a" /> against a right answer of 0 is <M tex="\tfrac12 a^2" />.
        Take the same step in both places: add 0.25 to each knob, which raises
        the evidence by 0.5.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>where the neuron stands</th>
              <th>w</th>
              <th>b</th>
              <th>evidence z</th>
              <th>answer</th>
              <th>score</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>somewhere unsure</td><td>0</td><td>0</td><td>0</td><td>0.500</td><td>0.125</td></tr>
            <tr><td>after the step</td><td>0.25</td><td>0.25</td><td>0.5</td><td>0.622</td><td>0.194</td></tr>
            <tr><td>confidently wrong</td><td>1.946</td><td>1.946</td><td>3.892</td><td>0.980</td><td>0.480</td></tr>
            <tr><td>after the same step</td><td>2.196</td><td>2.196</td><td>4.392</td><td>0.988</td><td>0.488</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        The same step raises the land by 0.069 in the first place and by 0.008
        in the second, nine times less. That second spot is nearly the panel's
        saturated start, so the flat opening of the run above is this: the walk
        begins on a shelf.
      </p>
      <p>
        Which half of the step went slack? A step reaches the score through two
        links. Turning the knobs moves the answer, and the answer moves the
        score. Module 4's chain has this shape, and its rule holds here too:
        what each link passes on multiplies.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>starting from</th>
              <th>the knobs move</th>
              <th>so the answer moves</th>
              <th>so the score moves</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>somewhere unsure</td><td>+0.5</td><td>+0.122</td><td>+0.069</td></tr>
            <tr><td>confidently wrong</td><td>+0.5</td><td>+0.008</td><td>+0.008</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        The first link took the same push both times and passed on fifteen
        times less out at the extreme, 0.008 against 0.122. The second link
        cannot be read that way, because what arrives at it differs. Ask
        instead what fraction of its input it passes on. Where the neuron is
        unsure it turns 0.122 of answer into 0.069 of score, 56 percent. Where
        the neuron is confidently wrong it turns 0.008 into 0.008, 98 percent.
        The second link is not weak out there. It passes on more than it does
        in the middle.
      </p>
      <p>
        So the shelf is the first link's doing, and the first link is the
        sigmoid's own shape: steep in the middle, nearly level at both ends,
        which is what squashing means. Its rate is the squash's slope, which
        Module 4 wrote <M tex="\sigma'(z)" /> and gave the formula{" "}
        <M tex="a(1-a)" />: 0.25 at an answer of 0.5, and 0.0196 at 0.98.
      </p>
      <p>
        Multiply the two rates together and you have the number BP1 computes.
        That is what blame is: how steeply the land rises under one neuron's
        knobs.
      </p>
      <Eq
        tex="\delta^L = \underbrace{(a - y)}_{\text{the gap}} \odot \underbrace{\sigma'(z^L)}_{\text{the squash's slope}}"
        gloss="Module 4's BP1, read as the two links multiplied. The gap does double duty here: it is how wrong the answer is, and under the quadratic cost it is also what the score charges for one more unit of answer. The squash's slope is what the first link passes on. The circled dot is the elementwise product, NumPy's plain star. Every slope in the network is built from this number, so whatever happens to it happens to all of them."
      />
      <p>
        Both rates depend on the answer, and they pull in opposite directions.
        The gap grows as the answer gets worse. The squash's slope shrinks,
        because a sigmoid at 0.98 is nearly level. Multiply them and read the
        product across the range:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>the answer</th>
              <th>the gap (what the score charges)</th>
              <th>the squash's slope a(1 − a)</th>
              <th>the blame, the two multiplied</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>0.500</td><td>0.500</td><td>0.2500</td><td>0.1250</td></tr>
            <tr><td>0.667</td><td>0.667</td><td>0.2222</td><td>0.1481</td></tr>
            <tr><td>0.900</td><td>0.900</td><td>0.0900</td><td>0.0810</td></tr>
            <tr><td>0.980</td><td>0.980</td><td>0.0196</td><td>0.0192</td></tr>
            <tr><td>0.999</td><td>0.999</td><td>0.0010</td><td>0.0010</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        The blame peaks at 0.148, when the answer is two thirds wrong, and
        falls away on both sides. Past that peak the squash's slope shrinks faster
        than the gap grows, so the worse the answer, the smaller the blame,
        and the smaller every slope the backward sweep produces from it. At an
        answer of 0.999 the blame is 0.001: as wrong as a sigmoid can be, with
        a blame 148 times smaller than the 0.148 the table peaks at.
      </p>
      <Figure caption="How steeply the land rises under the output neuron's knobs, against how wrong its answer is (right answer 0, so the answer is also the gap). The dots mark three rows of the table. Everything right of the peak is the complaint: more wrong, less blame, slower learning.">
        <BlameCurves />
      </Figure>
      <p>
        The same two rates predict the log in the last section. The
        neuron starts at 0.98201, so the gap is 0.98201 and the squash's slope
        is the gap times 0.01799, or 0.01766 (not the 0.0196 in the table's
        rounded 0.980 row: the fourth decimal of the answer moves it by a
        tenth). Both knobs then take the same step, because the
        input pinned at 1 makes the evidence just <M tex="w + b" />, and the
        run's step size is 0.15.
      </p>
      <Eq
        tex="\begin{gathered} \underbrace{0.98201}_{\text{the gap}} \times \underbrace{0.01766}_{\text{the squash's slope}} = \underbrace{0.0173}_{\text{the blame}} \\[0.8em] 0.15 \times \underbrace{0.0173}_{\text{the blame}} = \underbrace{0.0026}_{\text{per knob}} \\[0.8em] \underbrace{0.0052}_{\text{the drop in } z} \times \underbrace{0.01766}_{\text{the squash's slope}} = 0.00009 \end{gathered}"
        gloss="In order: the blame, the step each knob takes against it, and how far the answer moves once the evidence has fallen by both steps together. The log's first step takes the answer from 0.9820 to 0.9819, a drop of 0.00009, and 500 steps at that rate would leave the answer above 0.93. The walk finishes in 469 because the blame grows as the answer falls back toward two thirds, not because that first rate holds."
      />
      <p>
        Module 4's quiz already met this on its little 2-3-1 network: drag the
        output bias to −8 and the network answers 0.0036 when the right answer is 1,
        while its blame collapses to almost nothing. The same neuron, at the
        same time, is as wrong as it can be and as slow to learn as it can be.
      </p>

      <SectionHeader id="m7-ce" title="A cost that notices" />
      <p>
        The first link is not yours to change. It is what a sigmoid neuron
        does, and squashing is also what you want at the output, where the
        answer has to be a confidence between 0 and 1. The second link is not
        part of the network at all. It is the cost, the yardstick you grade
        answers with, chosen in Module 3 as half the squared gap because
        squaring was the obvious way to score. Swap the yardstick and every
        weight, wire and neuron stays exactly as it was, while the land
        underneath them is replaced. The replacement is the cross-entropy
        cost.
      </p>
      <p>
        So the second link has to cover for the first, and the question is how
        much cover to ask for. Two things are worth asking of the land's slope.
        It should be zero where the answer is right, so a neuron with nothing
        to fix stays put. It should grow as the answer gets worse, so the worst
        answers bring the largest corrections. The gap has both. It also never
        exceeds 1, so no step can be wild, and it is the simplest quantity that
        qualifies. Asking for exactly the gap is a design choice rather than a
        forced one: any rate that keeps growing with wrongness would break the
        shelf, and this is the plainest of them.
      </p>
      <p>
        Module 3's rule already charges at that rate, since the slope of half
        the squared gap is the gap. The multiplication that follows is what
        ruins it, and any fix has to survive that multiplication:
      </p>
      <Eq
        tex="\begin{gathered} \underbrace{0.98}_{\text{Module 3 charges}} \times \underbrace{0.0196}_{\text{the squash's slope}} = \underbrace{0.0192}_{\text{the land's slope}} \\[0.8em] \underbrace{50}_{\text{the new rule charges}} \times \underbrace{0.0196}_{\text{the squash's slope}} = \underbrace{0.98}_{\text{the gap, as asked}} \end{gathered}"
        gloss="Both lines sit at an answer of 0.98 against a right answer of 0, and the middle factor is the same in both, because the neuron has not changed. The top line is what the land does now. The bottom line is what it has to do, and the only number free to move is the charge."
      />
      <p>
        So the new rule has to charge the gap divided by the squash's slope,{" "}
        <M tex="(a - y) / (a(1-a))" />, which is 2 at an answer of 0.5, 50 at
        0.98, and 1000 at 0.999.
      </p>
      <p>
        Module 3's rule cannot be stretched to charge that, and the obstacle
        is a ceiling. The
        most one output can ever cost under half the squared gap is 0.5,
        reached by answering 1 when the right answer is 0. At an answer of 0.98
        it has already charged 0.48, so the whole rule has 0.02 points left in
        it for every remaining degree of wrongness. Nudging the answer from
        0.98 to 0.99 spends 0.0099 of that. For that one nudge to cost 0.5
        instead, and the next nudge to cost more again, the charge has to keep
        rising with no upper limit.
      </p>
      <Figure caption="How steeply the land rises, against how wrong the answer is (right answer 0, so the answer is also the gap). The hump is the quadratic cost, the curve from the last section. The straight line is what a yardstick with no ceiling gives: the land's slope equal to the gap, so a worse answer always brings a bigger correction. The two dots at an answer of 0.98 are 0.0192 and 0.98, a factor of 51 apart.">
        <BlameCurves showCrossEntropy />
      </Figure>
      <p>
        A charge that rises without limit needs one function the course has not
        used yet. Module 1 met <M tex="e" />, the number 2.718, inside the
        sigmoid. The natural logarithm <M tex="\ln a" /> is the reverse
        question: the power you have to raise <M tex="e" /> to in order to get{" "}
        <M tex="a" />. For answers between 0 and 1 it is negative, and it dives
        without limit as the answer approaches 0:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>a</th>
              <th>1</th>
              <th>0.5</th>
              <th>0.1</th>
              <th>0.02</th>
              <th>0.001</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>ln a</td>
              <td>0</td>
              <td>−0.693</td>
              <td>−2.303</td>
              <td>−3.912</td>
              <td>−6.908</td>
            </tr>
            <tr>
              <td>−ln a</td>
              <td>0</td>
              <td>0.693</td>
              <td>2.303</td>
              <td>3.912</td>
              <td>6.908</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Read the second row as a price list. If the right answer is 1, charge{" "}
        <M tex="-\ln a" />: nothing for answering 1, 0.693 for answering 0.5,
        6.908 for answering 0.001. If the right answer is 0, charge the same
        thing about the other end, <M tex="-\ln(1-a)" />. One line covers both
        cases, because one of the two terms is always multiplied by zero:
      </p>
      <Eq
        tex="C = -\big[\, y \ln a + (1-y) \ln (1 - a) \,\big]"
        gloss="The cross-entropy cost, for one output neuron. With y = 1 the second term vanishes and the charge is -ln a; with y = 0 the first vanishes and the charge is -ln(1-a). A network's cost is this summed over the output neurons and averaged over the examples, the same shape of bookkeeping as the quadratic cost, with no bookkeeping half."
      />
      <p>
        Check its rate by nudging, the way Module 3 measured every rate. At an
        answer of 0.98 against a right answer of 0 the charge is{" "}
        <M tex="-\ln(0.02) = 3.912" />. Nudge the answer up by 0.001 and the
        charge becomes <M tex="-\ln(0.019) = 3.963" />. The rise is 0.051, and
        50 times 0.001 is 0.050. The new yardstick does charge about 50 per
        unit of answer at that spot, where the old one charged 0.98.
      </p>
      <p>
        Now set that rate beside the first link's rate, which has not moved,
        because nothing about the neuron has changed:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>the answer</th>
              <th>the squash's slope</th>
              <th>what the new yardstick charges</th>
              <th>the two multiplied</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>0.500</td><td>0.250000</td><td>2</td><td>0.500</td></tr>
            <tr><td>0.980</td><td>0.019600</td><td>50</td><td>0.980</td></tr>
            <tr><td>0.999</td><td>0.000999</td><td>1000</td><td>0.999</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        Wherever the first link goes slack, the second link's rate rises by
        exactly the amount needed to undo it, and the two multiplied come out
        at the gap every time. In symbols, the squash's slope divides out of
        the yardstick's rate and multiplies back in at the neuron:
      </p>
      <Eq
        tex="\delta^L = \underbrace{\frac{a - y}{a(1-a)}}_{\text{the yardstick's rate}} \times \underbrace{a(1-a)}_{\text{the squash's slope}} = a - y"
        gloss="The squash's slope sits once in the yardstick's denominator and once as itself, so it divides out: the output layer's blame is the gap alone. At an answer of 0.98 against a right answer of 0 that is 50 times 0.0196, which is 0.98. Module 4's identity that sigma-prime equals a(1-a) is what makes the cancellation exact rather than approximate."
      />
      <p>
        The land has changed shape. Under Module 3's rule it climbs away from
        the bottom and then levels off at 0.5 in every wrong direction, so a
        network far enough out stands on a tabletop with no downhill left to
        find. Under the new rule it keeps climbing, at a rate that approaches 1
        and never flattens. Same floor, same bottom where the answers are
        right, no shelf anywhere in between.
      </p>
      <p>
        In the code, one line changes. BP1 was{" "}
        <code>delta = (a - y) * sigmoid_prime(z)</code> and becomes{" "}
        <code>delta = a - y</code>. Nothing has been struck out: the squash's
        slope is divided out by the yardstick and multiplied back in by the
        neuron, so computing both would be doing a thing and then undoing it.
        The short line is the full slope of the new land. BP2, BP3 and BP4 are
        untouched, and so is the forward pass.
      </p>
      <p>
        Swap the blame and run the same neuron again. From the saturated start
        it is below 0.1 in 50 steps instead of 469.
      </p>
      <Figure caption="The same neuron as at the top of the module, one line per cost. Both start by answering 0.982 and take the same step size; the cross-entropy line has no flat stretch at the beginning.">
        <SlowNeuron showCrossEntropy />
      </Figure>
      <p>
        BP2 keeping its <M tex="\sigma'" /> is worth a sentence, because it
        looks like an oversight. It is not. The <M tex="\sigma'" /> in BP1 was
        the output neuron's own first link, and the new yardstick was built to
        cover for that one. Every hidden neuron has a first link of its own,
        and no choice of yardstick reaches inside it: the score is a function
        of what comes out of the network, so only the last link into it can be
        cancelled this way. A flat hidden neuron still swallows the blame
        passing through it. Thirty flat hidden neurons are
        the second complaint, and they can be measured before training takes a
        single step.
      </p>
      <Aside>
        <p>
          One more pairing worth naming, since it is what most classifiers
          use today. Instead of ten independent sigmoids, a
          softmax output layer divides each neuron's{" "}
          <M tex="e^z" /> by the total across the layer, so the ten answers
          add to 1 and can be read as probabilities: the network says how it
          splits its confidence rather than answering ten separate yes-or-no
          questions. Pair softmax with a cost called the log-likelihood and the
          output layer's blame comes out as <M tex="a - y" /> again, by the
          same kind of cancellation. Nothing in this course needs it, and the
          ten-sigmoid version you have is what Nielsen's Chapters 1 to 3
          use.
        </p>
      </Aside>
      <p>
        One consequence: the step size has to change with the yardstick. The
        old land never rose faster than 0.148 per unit of evidence and the new
        one approaches 1, so at their steepest the new slopes are about seven
        times the old ones (1 divided by 0.148 is 6.8). Module 5 trained at{" "}
        <M tex="\eta = 3.0" />, found by trying; the runs below use{" "}
        <M tex="\eta = 0.5" />, found the same way. Trying landed on a factor
        of six between the two step sizes, close to the factor of seven between
        the two blames. This is not a cost of the fix, just a
        reminder that eta and the cost are not independent choices.
      </p>

      <ExerciseCard exercise={crossEntropyExercise} />
      <p>
        The panel below trains the digit reader three times from the same
        starting parameters, the same ones Module 5 used: 5,000 images, 8
        epochs, mini-batches of 10, your sgd doing the walking. Two runs share
        a step size of 0.5 and differ only in the output blame. The third is
        the quadratic cost again at Module 5's step size of 3.0, which is that
        module's own run cut short at 8 epochs.
      </p>
      <Figure caption="Three runs, one chart. Everything is held fixed except the output blame and the step size: the same wiring, the same starting numbers, the same shuffle, the same sgd. Each run takes a few seconds per epoch, and Stop ends it.">
        <CostSwapPanel />
      </Figure>
      <p>
        Try the obvious shortcut first: if the trouble is that the blames are
        small, why not take bigger steps? The third run prices it. At a shared
        step size of 0.5 the two costs are far apart: the quadratic run reaches
        62.6 percent against the cross-entropy run's 86.5, and the first epoch
        alone is 32.4 against 71.7. Cranking eta to 3.0 brings the quadratic
        run level at 86.6 percent, so on this network the cross-entropy cost
        does not make a smarter network. It makes one that learns at a step
        size where the other crawls, and that starts learning in its first
        epoch rather than climbing out of the slowdown first.
      </p>
      <p>
        That leaves the accuracy roughly where Module 5 left it, and it leaves
        the step size found the way Module 5 found it, by trying whole runs.
        The flat hidden layer is the complaint that moves the accuracy.
      </p>

      <SectionHeader id="m7-birth" title="Saturated at birth" />
      <p>
        Measure the hidden layer of Module 5's network before it takes a single
        step. Thirty neurons, 5,000 training images, so 150,000 readings of
        one neuron's evidence <M tex="z" /> on one image. Here is how they are
        spread, and what the squash's slope is where they land.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>distance of z from zero</th>
              <th>0 to 1</th>
              <th>1 to 2</th>
              <th>2 to 4</th>
              <th>4 to 8</th>
              <th>8 or more</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>share of the readings</td>
              <td>8.8%</td>
              <td>8.6%</td>
              <td>16.4%</td>
              <td>27.8%</td>
              <td>38.5%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The typical distance is 7.43, and the squash's slope has a median of 0.0020,
        against the 0.25 a neuron has at its steepest. Almost 62 percent of
        the readings are flatter than 0.01. A hidden neuron in that state
        passes almost nothing backward and moves almost nowhere, and it is in
        that state before it has learned anything at all: the flatness is an
        accident of how the weights were drawn.
      </p>
      <p>
        Why does the draw land the evidence out there, before anything has
        been learned? Count the terms. A hidden neuron's evidence is a sum of
        784 of them, one per pixel, each a weight times a pixel value, plus the
        bias. Most pixels of a digit are black, and on average 103 of the 784
        are above half brightness (measured over the bundled images, and the
        number the exercise below builds its stand-in digits from). So about a
        hundred weights contribute, each drawn from a bell of spread 1, meaning
        a typical draw sits about 1 away from the bell's middle.
      </p>
      <p>
        A hundred random pushes of size about 1 do not cancel out to nothing,
        and they do not add up to a hundred either. They pile up to about the
        square root of the count: <M tex="\sqrt{100} = 10" />. That
        square-root rule for adding up independent random quantities is the
        one fact here taken on trust rather than derived, and it can at least
        be checked. A spread runs a little larger than the plain average
        distance quoted above (7.43 for these readings), because it squares the
        far draws before averaging them. The exact version of the rule says the
        spread of a weighted sum is the square root of the pixel values squared
        and added up. For one bundled image those squares add to 87.4 on
        average (fewer than the 103 lit pixels, because a digit's grey edge
        pixels sit below 1), so the rule predicts{" "}
        <M tex="\sqrt{87.4} = 9.35" /> against a measured spread of{" "}
        <M tex="z" /> of 9.28.
      </p>
      <p>
        Nine is a long way out on a sigmoid. So the fix is to divide the pile
        back down, and the rule gives the divisor: shrink each weight by the
        square root of the number of inputs feeding its layer. For the hidden
        layer that is <M tex="\sqrt{784} = 28" />.
      </p>
      <Eq
        tex="w = \frac{\text{a draw of spread } 1}{\sqrt{n_{\text{in}}}} \qquad\text{and}\qquad \frac{9.28}{28} = 0.33"
        gloss="Every weight into a layer is drawn as before and then divided by the square root of that layer's input count, n-in: 28 for the 784 pixels, and the square root of 30 for the layer reading the 30 hidden neurons. The weighted sum's spread falls from 9.28 to 0.33, so the bias, still drawn at spread 1, becomes the larger term and the evidence lands within about 1 of zero."
      />
      <p>
        Measured on the same 150,000 readings, the typical distance from zero
        falls from 7.43 to 0.78, and the median of the squash's slope rises from 0.0020 to
        0.2203, close to the sigmoid's maximum of 0.25. Nothing is flatter than
        0.01 any more.
      </p>
      <Aside>
        <p>
          The biases keep their spread of 1, and after the division they are
          the bigger of the two terms in <M tex="z" />. Shrinking them too
          would push the evidence closer to zero still, and it is not worth
          doing: a bias is one number per neuron rather than one per wire, so
          nothing piles up in it, and starting every neuron at exactly the
          middle of the sigmoid removes some of the variety the layer starts
          with. Nielsen's Chapter 3 leaves the biases at spread 1 as well, and
          reports that the choice makes little difference either way.
        </p>
      </Aside>

      <ExerciseCard exercise={smartInitExercise} />
      <p>
        The comparison below is as controlled as this course gets. Same
        wiring, same cross-entropy blame, same sgd, same step size of 0.5,
        same shuffle, and the same random numbers from the same seed. One run
        divides them; the other does not.
      </p>
      <Figure caption="Two starting points, fifteen epochs each. The table above the chart is the hidden layer measured before either run takes a step; the chart is test accuracy per epoch afterwards. The dashed line is Module 5's start, the solid line yours.">
        <InitStartPanel />
      </Figure>
      <p>
        The divided start reads 85.3 percent of the held-out digits after one
        epoch, which is more than the other start reaches in its first four,
        and 92.1 percent after fifteen against 87.8. Module 5's own run, the quadratic
        cost at a step size of 3.0, reached 89.2 percent over the same fifteen
        epochs, so the network has gained about three points on where this
        module started, and this change is where they came from.
      </p>
      <p>
        One footnote on the two fixes together. Each cost can be given a step
        size that suits it, and at their best columns in the grid at the end of
        this module the two land close: 89.9 percent for the quadratic cost
        against 89.3 for cross-entropy, both from Module 5's start. The
        cross-entropy cost reliably removes the slowdown, which is why it needs
        no cranked step size, and the division reliably removes a flat start.
        They are not competing.
      </p>

      <SectionHeader id="m7-overfit" title="Learning the training set" />
      <p>
        The third complaint needs a smaller dataset to become visible, so take
        the first 1,000 of the 5,000 training images and train on those alone.
        The held-out thousand stays exactly as it was. (The slice is not
        rigged: its ten digits appear between 87 and 117 times each.) Watch
        four numbers rather than two, the cost and the accuracy on the images
        the network trains on, and the same pair on the images it never sees.
      </p>
      <Figure caption="Eighty epochs on 1,000 images, with the cross-entropy cost and the divided start. Solid lines are the images the network trains on, dashed lines the thousand held out. Green is accuracy on the left axis, red is cost on the right. The two solid lines keep improving to the end of the run; the two dashed ones stop at about epoch 10 and the dashed cost then turns around and rises.">
        <OverfitFigure />
      </Figure>
      <p>
        By epoch 29 the network answers all 1,000 training images correctly,
        and it keeps going: the training cost falls from 0.065 at epoch 20 to
        0.0093 at epoch 80, still dropping. Nothing in that stretch improves
        what the network does with a digit it has not seen. Accuracy on the held-out digits
        sat at 85.5 percent at epoch 20 and sits at 86.3 percent at epoch 80,
        and the held-out cost bottomed out at 0.84 around epoch 9 and has
        risen to 1.09 by the end.
      </p>
      <p>
        The two held-out lines are saying different things and both are worth
        having. Accuracy counts how often the biggest of the ten outputs is
        the right digit, so it ignores confidence. Cost prices the confidence.
        A rising held-out cost with flat held-out accuracy means the network is
        getting more sure of itself, right and wrong alike, on digits it has
        never seen. That is what the last seventy epochs bought: the same
        digits recognized, with more confidence behind each answer.
      </p>
      <p>
        The name for this is overfitting, and it is not a bug in the
        implementation. Descent was asked to make the cost on 1,000 images as
        small as possible, and it is doing exactly that. Whether small cost on
        those 1,000 has anything to do with the next digit is not a question
        the cost function was ever asked.
      </p>
      <p>
        Three words for the rest of the course, since this is where they all
        become visible at once. Doing well on digits you were not trained on is
        called generalizing, and it is the only thing anyone actually wants;
        every score in this course is on held-out digits for that reason.
        Overfitting is one way to fail at it, the one on the chart: the network
        has capacity to spare and spends it on the particular thousand images in
        front of it. The other way is underfitting, which is the network not
        managing the training images either, and it looks completely different:
        both lines poor, both still flat, no gap between them. The two failures
        want opposite treatments, which is why it is worth being able to tell
        them apart at a glance. A gap that keeps widening is too much capacity
        for the data; two low flat lines are too little.
      </p>
      <p>
        Two things follow. The first is a habit rather than a technique: hold
        data out, score on it, and read more than one number off it. The 1,000
        held-out digits in this course have never been trained on, in any
        module. Without them, the training cost falling to 0.0093 would look
        like success. Without the second number, the flat held-out accuracy
        here would hide the held-out cost rising beside it, the way Module 5's
        89 percent hid a 1 coming back at 122 of 126 and an 8 at only 68 of
        89.
      </p>
      <p>
        The second is that the cure closest to hand is more data. This same
        network on all 5,000 images reached 92.1 percent a section ago, against
        86.3 here. Five times the data, six points. When more data is available
        that is the thing to do, and when it is not there is a technique for
        getting part of the way.
      </p>

      <SectionHeader id="m7-l2" title="Shrink every weight" />
      <p>
        Track one more number through that run: the total of every weight
        squared, which is a plain measure of how big the weights have become.
        The divided start puts it at 39. By epoch 20, with the training images
        essentially all correct, it is 1,189, and by epoch 80 it is 1,926. The
        weights keep growing through the whole stretch where nothing improves.
      </p>
      <p>
        Big weights are what confident answers are made of. Evidence far from
        zero needs large weights to get there, and large weights also mean a
        small change in a few pixels swings the answer a long way. So the
        network that has driven its training cost to 0.0093 is a network that
        reacts sharply to the exact pixels it was trained on. That is the
        informal story, and it is worth saying that it is informal: why big
        weights and poor generalization go together is understood in
        particular cases rather than in general.
      </p>
      <p>
        The technique acts on the measurable part. Add the weights' own size to
        the cost, so that descent has a reason to keep them small unless the
        data insists otherwise:
      </p>
      <Eq
        tex="C_{\text{total}} = \underbrace{C}_{\text{the cross-entropy cost}} + \underbrace{\frac{\lambda}{2n} \sum_w w^2}_{\text{the new term}}"
        gloss="Lambda is a positive number you choose, the exchange rate between fitting the data and keeping the weights small: zero recovers the old cost exactly, and large values care more about small weights than about right answers. n is the number of training examples, there so that the same lambda means the same thing whatever the dataset size. The sum runs over every weight in the network; biases are not included."
      />
      <p>
        Descent needs the new term's slope for one weight, which means the
        slope of <M tex="w^2" />. Nudge it by hand, the Module 3 way: move{" "}
        <M tex="w" /> by a small amount <M tex="h" /> and read what the square
        does.
      </p>
      <Eq
        tex="\begin{gathered} (w + h)^2 - w^2 = \underbrace{2wh}_{\text{proportional to } h} + \underbrace{h^2}_{\text{negligible for small } h} \\[0.8em] \Rightarrow \quad \text{per unit of nudge, } 2w + h \approx 2w \end{gathered}"
        gloss="Multiply the square out, subtract the old value, and divide by the nudge: the rise per unit of nudge is 2w + h, which for a small nudge is 2w. Check it at w = 3 and h = 0.01, where the two squares are 9 and 9.0601: a rise of 0.0601 for a nudge of 0.01 is 6.01 per unit, against the 6 the rule predicts."
      />
      <p>
        Two things follow. The new term contributes{" "}
        <M tex="\lambda w / n" /> to that weight's slope, the 2 cancelling the
        half. And Module 3's bookkeeping half in{" "}
        <M tex="\tfrac12 (a - y)^2" /> is this same cancellation, the one
        Module 4 cashed in when the output layer's slope came out as the plain
        gap with no stray 2: halving a squared quantity makes its slope the
        quantity itself. Put the new slope into the update rule and collect the
        two terms in <M tex="w" />:
      </p>
      <Eq
        tex="w \;\leftarrow\; w - \eta \frac{\lambda}{n} w - \eta \frac{\partial C}{\partial w} \;=\; \Big( 1 - \frac{\eta \lambda}{n} \Big) w - \eta \frac{\partial C}{\partial w}"
        gloss="The same update as Module 3's, with the weight multiplied by a number slightly below 1 before the usual step. That factor is the whole technique, and the name for it says what it does: weight decay. Biases keep the old rule exactly, since the new term does not mention them."
      />
      <p>
        Put numbers in it. The run below uses <M tex="\eta = 0.5" />,{" "}
        <M tex="n = 1000" /> and <M tex="\lambda = 1" />, so the factor is{" "}
        <M tex="1 - 0.5/1000 = 0.9995" />. One epoch is 100 mini-batches, so
        100 steps, and left entirely alone a weight would shrink to{" "}
        <M tex="0.9995^{100} = 0.951" /> of itself per epoch, about five
        percent. No weight is left alone, of course, since the gradient is
        pulling the other way the whole time. So every weight faces a standing
        test each epoch: shrink by five percent unless the training images keep
        pulling it back, and a weight that only earns its keep on a handful of
        the thousand is not pulled back hard enough.
      </p>

      <ExerciseCard exercise={l2Exercise} />
      <p>
        Both runs in the panel go through your l2_step, one with{" "}
        <M tex="\lambda = 0" />, which is your Module 3 update exactly, and one
        with the lambda you pick. Same 1,000 images, same eighty epochs, same
        shuffle. The starting point is a control too, because the answer
        depends on it.
      </p>
      <Figure caption="The same 1,000-image run, with and without weight decay. Switch between accuracy (solid for the images trained on, dashed for the held-out ones) and the cost on the held-out digits. The starting point matters more than lambda does, so try both settings of it.">
        <RegularizePanel />
      </Figure>
      <p>
        Two readings, and they disagree. From your divided start, weight decay
        at <M tex="\lambda = 1" /> leaves the held-out accuracy where it was,
        85.7 percent averaged over the last twenty epochs against 86.4 without
        it. (Both are averages over the last twenty epochs of the chart, not
        the epoch-80 numbers the table under it prints, 84.7 with decay and
        86.3 without: one epoch of an eighty-epoch run wobbles by more than
        these two runs differ.) That gap is smaller than the run-to-run wobble
        in any case: the shuffle the panel holds fixed averages 86.4 over those
        last twenty epochs, and two other shuffles of the same unregularized
        run give 86.0 and 85.4. What decay does change is the two numbers it aims at. The
        held-out cost ends at 0.86 instead of 1.09, and the total of the
        squared weights at 655 instead of 1,926, so the network stopped growing
        and stopped becoming more confident while reading the same digits.
      </p>
      <p>
        Switch the start to Module 5's undivided draw and the same comparison
        answers differently. Those weights begin at a total of 23,538, since
        that is what not dividing by 28 means, and without decay they stay
        there: 25,528 after eighty epochs, with the held-out accuracy stalled
        at 77.8 percent. With <M tex="\lambda = 1" /> the decay pulls them down
        to 679 over the course of the run and the accuracy reaches 85.1
        percent, seven points better.
      </p>
      <p>
        Which is the same fix as the last section's, arriving from the other
        end. The division set the weights to a sane size before training; the
        decay pulls them to a sane size during it. Where the division has
        already been done, weight decay has no runaway to catch. So the summary
        of this cycle is a narrow one: decay reliably holds the weights and the
        held-out cost down, and it buys accuracy where something else was
        leaving the weights too large. Lambda is one more number found by trying, and the
        panel's other settings are there to be tried:{" "}
        <M tex="\lambda = 5" /> holds the weights down hard enough that the
        network reads fewer digits.
      </p>

      <SectionHeader id="m7-more" title="The rest of the toolbox" />
      <p>
        Everything in this module was a constant chosen by trying: the step
        size, lambda, thirty hidden neurons, mini-batches of ten, fifteen
        epochs. Those choices are called hyperparameters, to separate them from
        the parameters descent finds on its own, and there is no equation for
        them. Here is the trying, for two of them, at fifteen epochs on the
        5,000 images.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>test accuracy after 15 epochs</th>
              <th>η = 0.5</th>
              <th>η = 1.0</th>
              <th>η = 3.0</th>
              <th>η = 6.0</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>quadratic cost, Module 5's start</td>
              <td>70.8%</td>
              <td>79.4%</td>
              <td>89.2%</td>
              <td>89.9%</td>
            </tr>
            <tr>
              <td>cross-entropy cost, Module 5's start</td>
              <td>87.8%</td>
              <td>89.3%</td>
              <td>87.6%</td>
              <td>80.9%</td>
            </tr>
            <tr>
              <td>quadratic cost, the divided start</td>
              <td>90.0%</td>
              <td>90.3%</td>
              <td>91.0%</td>
              <td>91.3%</td>
            </tr>
            <tr>
              <td>cross-entropy cost, the divided start</td>
              <td>92.1%</td>
              <td>92.3%</td>
              <td>87.7%</td>
              <td>69.4%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Three things are readable there. Each row has a best column and it is
        not the same column, which is why a cost cannot be judged at a fixed
        step size. The quadratic rows are still climbing at the right-hand
        edge, where the cross-entropy rows have already fallen off: the
        slowdown showing up as a demand for a bigger step, and the absence of
        it showing up as a lower ceiling on how big a step is useful. And the
        best of each row runs 89.9, 89.3, 91.3, 92.3, so the two changes
        together are worth about two and a half points at their own best
        settings, while two of the four rows span nineteen points or more from
        their worst column to their best. Getting the step size wrong costs more than
        either fix earns.
      </p>
      <p>
        How to search, when there is no equation: change one thing, rerun,
        keep it if the held-out score improved. Move in factors of about three
        rather than small increments, because the interesting range spans
        orders of magnitude. Start with the step size, since a bad one makes
        everything else unreadable. And score on held-out data, never on the
        training images.
      </p>
      <p>
        That last rule needs one more turn of the screw, and this section is
        where it bites. The overfitting cycle showed what happens to a number
        that descent is allowed to optimize: the training cost fell to 0.0093
        and stopped meaning anything about unseen digits. A search does the same
        thing by hand. Sixteen runs were scored on those thousand held-out
        digits and the best was kept, so the winning 92.3 percent is the best of
        sixteen tries against that particular thousand, not what the network
        would score on a fresh one. Some of that number is the fix and some of
        it is the choosing.
      </p>
      <p>
        The standard repair is a third split. Hold out two sets rather than one:
        a validation set you may look at as often as you like, because choosing
        against it is what it is for, and a test set you look at once, at the
        end, to report. Everything in the search above then happens on
        validation data, and the test number is spent last and only once.
      </p>
      <p>
        This course has one held-out thousand and uses it for both jobs, so its
        quoted numbers are a little generous, and the last section measured
        how generous. Three shuffles of one unchanged setting landed between
        85.4 and 86.4 percent there: a point of wobble with nothing
        changed at all. Picking the best of sixteen runs pockets some of that
        wobble as if it were an improvement, which is why a repeated tenth of a
        point is not worth chasing and why the number to trust is one scored on
        digits nothing was chosen against.
      </p>
      <p>
        Four more techniques exist for the overfitting problem, named here and
        not implemented. More training data, which is the strongest of them and
        the reason this course's numbers sit below Nielsen's. Artificial data,
        made by shifting and rotating the images you have, which works because
        a rotated 3 is still a 3. Dropout, which switches off a random half of
        the hidden neurons on each mini-batch, so no neuron can rely on any
        particular other one. And early stopping, which is simply reading the
        chart above and keeping the network from epoch 10 rather than epoch 80.
      </p>
      <p>
        The machinery underneath all three fixes has not changed. The cost is a
        different formula, the starting weights are divided by a square root,
        and the update carries one extra factor. Your forward pass, your
        backward sweep with its four equations, your mini-batch shuffle and
        your descent are exactly the code you wrote in Modules 2, 3 and 5. Module 8 asks
        what happens to those four equations when the network gets deep, and
        the answer is already visible in BP2.
      </p>

      <Recap
        items={[
          "A descent step reaches the score through two links: the knobs move the answer, and the answer moves the score. BP1 multiplies their two rates, the squash's slope and the gap, which is why blame is just how steeply the land rises under a neuron's knobs. Where an output neuron is confidently wrong the first link goes slack: the blame peaks at 0.148 two thirds of the way wrong and falls away past it, and at an answer of 0.999 it is 0.001.",
          "The first link is the neuron's own machinery and is not open to change; the second is the yardstick, and it is. The quadratic yardstick has a ceiling of 0.5 per output, so its rate can never grow enough to cover for a slack first link. The cross-entropy cost charges -ln of the confidence in the right answer, which rises without limit, at a rate of exactly the gap divided by the squash's slope. The two multiply out to the gap everywhere, so BP1 becomes delta = a - y with nothing struck out. BP2 keeps its own sigma-prime, which is a hidden neuron's own first link and out of the yardstick's reach.",
          "Weights drawn at spread 1 pile up over about a hundred lit pixels to an evidence spread of 9.3, leaving 62 percent of hidden neurons flatter than 0.01 before training starts. Dividing each layer's weights by the square root of its input count puts the typical evidence within 1 of zero, with no neuron below 0.01, and is worth 4.3 points of accuracy against the same cost started undivided.",
          "Training on 1,000 images reaches 100 percent on those images while the held-out accuracy stops improving and the held-out cost turns around and rises: the network is buying confidence, not recognition. Weight decay multiplies every weight by a factor just under 1 each step, which reliably holds the weights and the held-out cost down, and buys accuracy only where something else left the weights too large.",
          "Every constant here was found by trying, and the grid shows the step size mattering more than the cost. Module 8 takes the four equations into deep networks and finds a limit that no choice of cost fixes.",
        ]}
        chapter="Chapter 3 (improving the way neural networks learn)"
        href="http://neuralnetworksanddeeplearning.com/chap3.html"
      />
    </article>
  );
}
