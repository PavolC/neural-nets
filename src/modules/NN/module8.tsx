import { AfterThis, Aside, Figure, ModuleToc, Recap, SectionHeader, fig } from "../../components/ModuleBits";
import { Eq, M } from "../../components/Math";
import { DepthTrainPanel } from "./interactives/DepthTrainPanel";
import { LayerSpeedBars } from "./interactives/LayerSpeedBars";
import { sigmoid } from "./interactives/utils";

export function Module8() {
  return (
    <article className="module">
      <h2>Chapter 8: Why deep is hard (and what came next)</h2>
      <AfterThis
        items={[
          "Measure how fast each layer of a network is learning, and read a deep network's first epochs off those numbers before it takes a step.",
          "Take one backward hop apart into the two factors BP2 multiplies by, and say which one Chapter 7's division fixed and which one no choice of cost can.",
          "Say what ReLU changes, what a convolutional layer shares, and where a network keeps the description it has learned.",
        ]}
      />
      <ModuleToc />

      <SectionHeader id="c8-deeper" title="One more layer" />
      <p>
        Chapter 7 left the digit reader at 92.1 percent of a thousand held-out
        digits: 784 pixels in, one hidden layer of 30 neurons, ten outputs, the
        cross-entropy blame at the output, every weight divided by the square
        root of its layer's input count. The next thing to try is another hidden
        layer.
      </p>
      <p>
        There is a reason to expect that to buy something, and Chapter 6 priced
        the alternative. Its box network fenced every training image separately,
        two neurons per pixel per image, 7.8 million of them, and nothing in that
        construction was shared between one part of the input and another. A
        hidden neuron reads 784 pixels and reports one number, and whatever it is
        looking for it has to find in one step, straight from raw brightness.
        Stack two hidden layers and the second one reads the first one's thirty
        reports instead of pixels, so a stroke can be built out of edges and a
        digit out of strokes, with no single layer having to go from pixels to
        digits by itself. That is the intuition. Whether it pays is a
        measurement.
      </p>
      <p>
        So run it. Same 5,000 images, same shuffle, same cross-entropy blame,
        same divided start, same fifteen epochs, same step size of 0.5, and one,
        two, three or four hidden layers of 30 neurons each. The last row averages
        the final five epochs rather than quoting the fifteenth, because a single
        epoch's score wobbles by a point or two and the deeper runs wobble more.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>test accuracy</th>
              <th>1 hidden layer</th>
              <th>2 hidden layers</th>
              <th>3</th>
              <th>4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>after epoch 1</td>
              <td>85.3%</td>
              <td>81.8%</td>
              <td>41.2%</td>
              <td>12.6%</td>
            </tr>
            <tr>
              <td>after epoch 5</td>
              <td>91.2%</td>
              <td>89.8%</td>
              <td>87.6%</td>
              <td>61.1%</td>
            </tr>
            <tr>
              <td>averaged over epochs 11 to 15</td>
              <td>91.6%</td>
              <td>92.0%</td>
              <td>89.2%</td>
              <td>86.5%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The second hidden layer lands within half a point of the first: 92.0
        against 91.6. The third costs two and a half points and the fourth costs
        five. And the last row hides how each run got there. One hidden layer
        reads 85.3 percent of the held-out digits after a single pass. Four hidden
        layers read 12.6 percent, which is exactly the share of those digits that
        are 1s.
      </p>
      <p>
        That is not a coincidence: after its first epoch the four-layer network
        answers 1 for all thousand images, whatever the image shows. Every
        one of its ten outputs, on every one of those images, lands between 0.054
        and 0.179, and on a single image the highest sits within 0.13 of the
        lowest. So whichever output is highest wins every image, and the one that
        is highest is the digit that appeared most often in training.
        After a full pass over 5,000 images it has learned which digit is
        commonest and nothing else.
      </p>
      <p>
        Chapter 7 closed by showing the step size mattering more than either of its
        fixes, so that is the first thing to rule out. Doubling it to 1.0 lifts the
        four-layer network's first epoch from 12.6 percent to 23.2 and drops its
        last five from 86.5 to 84.0. At 2.0 the pair is 24.2 and 74.4; at 3.0, 21.0
        and 65.4. No setting gets the first epoch anywhere near the 85.3 percent
        one hidden layer reaches, and every setting above 0.5 ends the run lower.
      </p>
      <p>
        The deep network is not stuck for good either. Left running it climbs:
        61.1 percent by epoch 5, 86.5 by epoch 10, and it is still swinging across
        five points from epoch to epoch at the end of the run, where the one-layer
        network has stayed between 91.3 and 92.2 since its ninth. Some part of this
        network is learning very slowly, and the rest of it works around that part
        over the first several epochs. Which part, though, and what would you
        measure to find out?
      </p>

      <SectionHeader id="c8-speed" title="Which layer is learning" />
      <p>
        Backprop already computes the number this needs. BP3 said a neuron's bias
        slope is its blame, so the gradient the update rule uses for a layer's
        biases, <M tex="\partial C / \partial b^l" />, is that whole layer's
        column of blames. And the update turns that column into a distance: the
        biases move by <M tex="\eta" /> times it, so its size is how far the layer
        travels in one step, per unit of step size.
      </p>
      <p>
        The size of a column is what the double bars mean: square every entry,
        add them up, take the square root. Chapter 3 put those bars inside the
        quadratic cost with a squared on them. The square cancelled the root, so
        the recipe there stopped at square and add. Applied to a layer's bias
        gradient, call it that layer's learning speed. The field's name for the size
        of a gradient measured this way is its <b>norm</b>, so a paper that reports
        "the gradient norm at layer l" is reporting the number in the table below.
        Here is one
        four-hidden-layer network measured at Chapter 7's start, before it has
        taken a single step (every speed here comes from the gradient averaged
        over 200 of the thousand held-out digits, the same batch the panel below
        uses). Layer 1 is the 784 pixels, which have no biases and nothing to
        learn, so the count starts at 2.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>784-30-30-30-30-10, before the first step</th>
              <th>layer 2</th>
              <th>layer 3</th>
              <th>layer 4</th>
              <th>layer 5</th>
              <th>layer 6 (output)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>learning speed</td>
              <td>0.002539</td>
              <td>0.0124</td>
              <td>0.06668</td>
              <td>0.2652</td>
              <td>1.439</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        On one step of descent, at whatever step size, the output layer moves 567
        times as far as the first hidden layer. That answers which part is slow,
        and it was settled before the network saw a single correction, because
        these are the drawn weights and nothing else. The panel below is the same
        measurement with the depth on a slider. Press Draw again a few times: over
        forty draws at this depth the middle value of that gap is about 650 times,
        so the 567 in the table is an ordinary draw and not a picked one.
      </p>
      <Figure caption="Every layer's learning speed at the start, for a network with the chosen number of hidden layers of 30 neurons. Draw again redraws all the weights from a new random seed: the exact numbers move, the staircase does not.">
        <LayerSpeedBars />
      </Figure>
      <p>
        Two things in that picture. The bars fall by about the same amount at
        every hop, which on a scale by tens means a constant factor rather than a
        constant subtraction. And the factor does not depend on the depth: adding
        a layer does not make the existing hops steeper, it adds one more of
        them. One through five hidden layers put the output ahead of the first by
        4.8, 26.2, 117, 567 and 3,002 times.
      </p>

      <SectionHeader id="c8-hop" title="One fifth per hop" />
      <p>
        That constant factor is BP2, and BP2 has not changed since Chapter 4. It
        is the equation that carries blame one layer back:
      </p>
      <Eq
        tex="\delta^l = \underbrace{\big( (w^{l+1})^T \delta^{l+1} \big)}_{\text{back through the wire ledger}} \odot \underbrace{\sigma'(z^l)}_{\text{scale by the squash's slope}}"
        gloss="Chapter 4's BP2. The blame column of the layer above goes back through that layer's weight matrix transposed, which regroups the wires by sender, and then every entry is multiplied by that neuron's own squash slope at its own evidence. The circled dot is the elementwise product, NumPy's plain star."
      />
      <p>
        Two steps, in the order the code performs them, and each one changes the
        size of the column. Measuring what each does is the whole explanation.
        Here is the four-hidden-layer network again, one row per hop, with the
        bar it started from and the bar it produced.
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>hop</th>
              <th>from</th>
              <th>back through the ledger</th>
              <th>times the squash-slope step</th>
              <th>= the hop</th>
              <th>lands on</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>layer 6 to layer 5</td>
              <td>1.439</td>
              <td>0.878</td>
              <td>0.210</td>
              <td>0.184</td>
              <td>0.2652</td>
            </tr>
            <tr>
              <td>layer 5 to layer 4</td>
              <td>0.2652</td>
              <td>1.159</td>
              <td>0.217</td>
              <td>0.251</td>
              <td>0.06668</td>
            </tr>
            <tr>
              <td>layer 4 to layer 3</td>
              <td>0.06668</td>
              <td>0.906</td>
              <td>0.205</td>
              <td>0.186</td>
              <td>0.0124</td>
            </tr>
            <tr>
              <td>layer 3 to layer 2</td>
              <td>0.0124</td>
              <td>0.962</td>
              <td>0.213</td>
              <td>0.205</td>
              <td>0.002539</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Read one row in the multiplying direction. Layer 5's speed is 0.2652;
        the ledger sends it back at 1.159 times that, and the squash-slope step
        keeps 0.217 of what comes out:
      </p>
      <Eq
        tex="\underbrace{0.2652}_{\text{layer 5's speed}} \times \underbrace{1.159}_{\text{the ledger}} \times \underbrace{0.217}_{\text{the squash's slope}} = \underbrace{0.0667}_{\text{layer 4's speed}}"
        gloss="Layer 4's measured speed is 0.06668, so the two factors account for it to four figures. Check the other three rows the same way, and every product lands on the speed in that row's own last column. This multiplying is the only arithmetic the backward sweep does to the size of a blame column."
      />
      <p>
        The two middle columns hold two different kinds of number, and each kind
        has one plain reason for the value it takes.
      </p>
      <p>
        The ledger column is about 1, every time. That is not luck; it is what
        Chapter 7 bought. Dividing every weight by the square root of its layer's
        input count was chosen to keep a neuron's evidence near zero, and it has
        a second consequence: a ledger drawn that way sends a column back at
        about its own length. Chapter 7's square-root rule is doing its work a
        second time here. Take one of the hops between two layers of 30. Every
        entry of the returned column adds up 30 terms, one
        per neuron in the layer above. Each term is a weight of size about 1
        divided by the square root of 30, times a blame entry of the size that
        arrived. Thirty
        random pushes that small pile up to the square root of 30 times one of
        them, and the two square roots cancel, so the entries come back the size
        they went in. The first hop is the one exception, because the layer above
        it is the output layer with 10 neurons. Ten pushes pile up to the square
        root of 10 times one of them, and the weights are still divided by the
        square root of 30. Each entry there comes back at about 0.58 of the size
        that arrived. The returned column carries 30 of those entries against the
        10 that arrived, and 30 entries at 0.58 come to the same length as 10 at
        full size. Averaged over 200 draws of a 30-by-30 ledger the
        length comes back at 0.99 of what went in, with the middle 90 percent of
        those draws between 0.80 and 1.19. The three 30-to-30 values above, 0.906
        to 1.159, are ordinary members of that spread, and the first row's 0.878
        lands with them.
      </p>
      <p>
        The squash-slope column is about 0.21, every time, and it can never be more
        than 0.25. Chapter 4 gave the squash's slope as{" "}
        <M tex="a(1-a)" />, which is largest when the answer <M tex="a" /> is 0.5
        and equals 0.25 there, and shrinks toward zero at both ends. So the second
        factor is a fraction with a ceiling, and the divided start puts most
        neurons near the top of it: Chapter 7 measured the median squash slope at that
        start as 0.22.
      </p>
      <p>Multiply the two kinds together and the hop is settled:</p>
      <Eq
        tex="\text{one hop} \;=\; \underbrace{\approx 1}_{\text{the ledger}} \;\times\; \underbrace{\approx 0.21}_{\text{the squash's slope, at most } 0.25} \;\approx\; \tfrac{1}{5}"
        gloss="A ledger that returns a column at about its own length, times a squash slope that averages 0.21 here and cannot exceed 0.25 anywhere. The hop lands near one fifth, and nothing in the layer sizes or in the images moved it there."
      />
      <p>
        A hop of one fifth, applied four times between the output and the first
        hidden layer, is what the bars showed:
      </p>
      <Eq
        tex="\underbrace{5}_{\text{1 hidden}} \quad \underbrace{25}_{\text{2}} \quad \underbrace{125}_{\text{3}} \quad \underbrace{625}_{\text{4}} \quad \underbrace{3125}_{\text{5}}"
        gloss="Five to the power of the number of hidden layers, against the 4.8, 26.2, 117, 567 and 3,002 the bars measured. The prediction is a plain repeated multiplication, because that is all the backward sweep does."
      />
      <p>
        Nothing in this is a bug. The four equations are right, the gradient check
        in Chapter 5 said so, and the gradient really is that small: nudge one of
        the first layer's biases and the cost really does barely move. The first
        hidden layer of a four-hidden-layer network is learning about 500 times
        slower than the last one, which is why its first epoch scored the share of
        1s. The layer closest to the pixels is the one that barely moves.
      </p>
      <Aside>
        <p>
          The obvious repair is to give each layer its own step size, large at the
          front and small at the back, and per-layer step sizes are a real
          technique. It does not fix this, because the ratio is not a fixed number
          to compensate for. It depends on the depth (about 5 per layer, so a few
          hundred at four hidden layers and a few thousand at five), it moves with
          the draw, and it moves during training. Measured on the four-layer
          network the panel below trains: 520 times before the first step, 178
          after one epoch, 1.9 after five, 1.2 after fifteen. The imbalance is a
          property of the starting point that the network works its way out of,
          slowly, with the early epochs as the bill.
        </p>
      </Aside>

      <SectionHeader id="c8-relu" title="A neuron with no ceiling" />
      <p>
        Of the hop's two factors, only one has a ceiling on it, and that ceiling
        is a fact about the sigmoid rather than about neurons. Chapter 1 picked the
        sigmoid for a specific reason: a perceptron's hard step has slope zero
        everywhere, which leaves descent nothing to work with, and the sigmoid is
        the smooth version. Smooth was the requirement. Bounded was not.
      </p>
      <p>Here is a squash that is smooth enough and unbounded above:</p>
      <Eq
        tex="\mathrm{ReLU}(z) = \max(0, z)"
        gloss="Pass the evidence straight through when it is positive, and answer 0 when it is not. The name is short for rectified linear unit, which describes the shape: a straight line, with the negative half flattened. Read it as ray-loo."
      />
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>evidence z</th>
              <th>−3</th>
              <th>−0.5</th>
              <th>0</th>
              <th>0.5</th>
              <th>3</th>
              <th>9</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>sigmoid answers</td>
              <td>0.0474</td>
              <td>0.3775</td>
              <td>0.5000</td>
              <td>0.6225</td>
              <td>0.9526</td>
              <td>0.9999</td>
            </tr>
            <tr>
              <td>its squash slope</td>
              <td>0.0452</td>
              <td>0.2350</td>
              <td>0.2500</td>
              <td>0.2350</td>
              <td>0.0452</td>
              <td>0.0001</td>
            </tr>
            <tr>
              <td>ReLU answers</td>
              <td>0</td>
              <td>0</td>
              <td>0</td>
              <td>0.5</td>
              <td>3</td>
              <td>9</td>
            </tr>
            <tr>
              <td>its squash slope</td>
              <td>0</td>
              <td>0</td>
              <td>0</td>
              <td>1</td>
              <td>1</td>
              <td>1</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The last row is the whole change. A ReLU neuron with positive evidence has
        a squash slope of exactly 1, so BP2's second step multiplies its blame by 1 and
        leaves it alone. A neuron with negative evidence has a squash slope of 0 and its
        blame is deleted. There is no shrinking: a blame either passes through
        untouched or stops. (At exactly zero the slope has no single value; the
        usual convention picks 0, and it makes no measurable difference, because
        an evidence of exactly 0.0 essentially never happens.)
      </p>
      <p>
        Swap the hidden layers of the same four-hidden-layer network to ReLU and
        the hops become 0.634, 0.571, 0.900 and 0.771, and the output ends up 4.0
        times ahead of the first hidden layer instead of 567. The hops land under
        1 rather than on it because a hidden neuron is live only about half the
        time, and a silent one's share of the blame is deleted, which shortens the
        column. Deleting half a column's entries shortens it far less than
        multiplying every entry by 0.21 does.
      </p>
      <p>
        Leave the weight multiplier on the panel below at 1, which is Chapter 7's
        divided draw exactly, and switch between the two squashes. A ReLU neuron's
        squash slope is 1 when it is live and 0 when it is not, so averaging it across
        a layer's neurons and images gives a share. The last column of the table
        under the chart reports exactly that average, so with ReLU chosen it reads
        as how often those neurons are live.
      </p>
      <Figure caption="Every layer's learning speed at the start, now with the hidden layers' squash and the size of the drawn weights as controls. The table under the chart is BP2's two steps measured separately for each hop: the two columns multiply to the hop, and the hop is the bar divided by the bar it came from.">
        <LayerSpeedBars full />
      </Figure>
      <p>
        What that buys in training is a network where depth stops costing
        anything. Same 5,000 images, same fifteen epochs, same divided start, ReLU
        in the hidden layers, at a step size of 0.05:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>test accuracy, ReLU hidden layers</th>
              <th>1 hidden layer</th>
              <th>2 hidden layers</th>
              <th>3</th>
              <th>4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>after epoch 1</td>
              <td>83.5%</td>
              <td>83.7%</td>
              <td>83.0%</td>
              <td>63.6%</td>
            </tr>
            <tr>
              <td>after epoch 5</td>
              <td>89.8%</td>
              <td>90.1%</td>
              <td>89.8%</td>
              <td>90.3%</td>
            </tr>
            <tr>
              <td>averaged over epochs 11 to 15</td>
              <td>91.0%</td>
              <td>91.6%</td>
              <td>90.8%</td>
              <td>90.9%</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Three things in that table are worth stating plainly. Its last row spans
        0.8 of a point across the four depths where the sigmoid's spanned five, and
        the four-layer network is at 90.3 percent by epoch 5 where the sigmoid's was
        at 61.1: depth has stopped costing anything, which is what taking the
        ceiling off was for. The one-layer network gains nothing from the change and loses
        half a point, 91.0 against the sigmoid's 91.6, so this is not a better
        neuron, it is a neuron that survives being stacked. And the step size had to
        come down by a factor of ten, from 0.5 to 0.05, because a squash with no
        ceiling passes blame back at full size and a step a sigmoid would have
        absorbed now overshoots.
      </p>
      <p>
        What overshooting costs here is specific enough to watch. Run the same
        four-layer ReLU network at 0.5 and after one epoch 28 of the 30 neurons in
        its layer 4 answer 0 on every one of the 5,000 training images; by the
        third epoch it is all 30. A layer that always answers 0 sends nothing
        forward and passes nothing back, so the network sits between 9 and 13
        percent for the rest of the run. At 0.05 that same layer has 3 silent
        neurons after one epoch and 1 after three.
      </p>
      <Aside>
        <p>
          Follow that silent layer one step further. A ReLU neuron whose evidence
          is negative on every training image has a squash slope of 0 on every image, so
          every gradient it receives is 0, so nothing ever changes it, so it is
          negative on every image forever. The name for that is a dead unit, and a
          large step size creates them in bulk: one oversized update pushes a bias
          far enough negative that its neuron never fires again. This is the price
          of the flat half, and it is the one thing the sigmoid was better at,
          since a sigmoid neuron's squash slope gets small but never reaches zero.
          Variants exist that leave a small slope on the negative side (leaky ReLU
          is the plainest, with 0.01 in place of 0) so that nothing is ever
          switched off completely.
        </p>
      </Aside>
      <p>
        The panel below runs the whole experiment in one press: one hidden
        layer and four, under each squash, all of it your own code. Your
        init_network builds every network, your sgd walks them, and for the
        sigmoid runs it is your backprop computing every gradient, reached
        through the adapter written for you in Chapter 5. ReLU's cannot be,
        because your BP2 has <code>sigmoid_prime</code> written into it. For
        the ReLU runs the gradient comes from a copy of your backprop with two
        lines swapped, written into the panel. Its forward
        pass squashes each hidden layer with <code>np.maximum(0.0, z)</code>{" "}
        instead of <code>sigmoid(z)</code>, and its BP2 line multiplies by{" "}
        <code>(zs[-l] &gt; 0)</code> instead of{" "}
        <code>sigmoid_prime(zs[-l])</code>, since that comparison is already the
        1-or-0 squash-slope row of the table above. The sgd around every run is
        still yours.
      </p>
      <Figure caption="One hidden layer against four, under each squash: all four runs, drawn as they finish. Hue is the squash, dashed lines are the one-hidden-layer networks. The table above the chart is each deep network's per-layer learning speed, measured before its run starts. Each run takes a few seconds per epoch, and Stop ends it.">
        <DepthTrainPanel />
      </Figure>

      <SectionHeader id="c8-unstable" title="The hop is a product" />
      <p>
        Does a hop have to be smaller than 1? BP2 does not say so: the hop is a
        product of two measured numbers, and the weight multiplier on the
        layer-speed panel above changes one of them. Turn it up with the sigmoid
        selected:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>sigmoid, 4 hidden layers, every weight multiplied by</th>
              <th>1</th>
              <th>2</th>
              <th>4</th>
              <th>8</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>back through the ledger</td>
              <td>0.962</td>
              <td>1.914</td>
              <td>3.630</td>
              <td>7.220</td>
            </tr>
            <tr>
              <td>times the squash-slope step</td>
              <td>0.213</td>
              <td>0.206</td>
              <td>0.170</td>
              <td>0.126</td>
            </tr>
            <tr>
              <td>average squash slope across the hidden layers</td>
              <td>0.20</td>
              <td>0.18</td>
              <td>0.14</td>
              <td>0.08</td>
            </tr>
            <tr>
              <td>the hop into layer 2</td>
              <td>0.205</td>
              <td>0.394</td>
              <td>0.618</td>
              <td>0.908</td>
            </tr>
            <tr>
              <td>output ahead of layer 2 by</td>
              <td>567×</td>
              <td>60.4×</td>
              <td>15.0×</td>
              <td>4.2×</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The last row is not the row above it raised to the fourth power, the way
        one fifth per hop was. Four hops of 0.908 would put the output only 1.5
        times ahead of layer 2, not the 4.2 the table reports. Bigger weights move
        the four hops by different amounts, so they stop being near enough equal to
        raise to a power, and this table lists only the hop into layer 2 (the
        panel's own table prints all four).
      </p>
      <p>
        The ledger factor rises roughly in step with the multiplier, which is
        what a multiplier does. The squash-slope factor falls as it rises, because
        bigger weights put a neuron's evidence further from zero and a sigmoid out
        there is flatter. At a multiplier of 8 the average squash slope across the
        hidden layers is 0.08, against 0.20 at Chapter 7's draw, and Chapter 7 spent
        a section on why a start that flat cannot learn. So the two factors fight,
        the sigmoid wins, and the hop climbs toward 1 without reaching it however
        far the multiplier goes. A sigmoid network cannot be made to hand its early
        layers too much.
      </p>
      <p>
        Select ReLU and turn the same dial. Its squash-slope step does not fall,
        because a live neuron's slope is 1 whatever the evidence is:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>ReLU, 4 hidden layers, every weight multiplied by</th>
              <th>1</th>
              <th>2</th>
              <th>4</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>the hop into layer 2</td>
              <td>0.634</td>
              <td>1.456</td>
              <td>2.990</td>
            </tr>
            <tr>
              <td>layer 2's learning speed</td>
              <td>0.35</td>
              <td>9.74</td>
              <td>136.32</td>
            </tr>
            <tr>
              <td>the output layer's</td>
              <td>1.40</td>
              <td>1.73</td>
              <td>1.52</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        At a multiplier of 2 the first hidden layer moves 5.6 times as far as the
        output on every step; at 4, ninety times as far. That is the other
        failure, and it breaks a run in a different way: a layer taking enormous
        steps does not creep toward an answer, it jumps past it and comes back on
        the far side, further out. Each failure has a name of its own, and they are
        the two you meet first anywhere else: a hop well under 1 makes the early
        layers' gradients <b>vanish</b>, which is the <b>vanishing gradient
        problem</b>, and a hop above 1 makes them <b>explode</b>. Together they
        share one name, unstable gradients, and the table above shows why one name
        covers both. The hop is
        a single number that has to sit near 1, and the multiplier walks it from
        0.63 through 1.46 to 2.99 with nothing in the network objecting.
      </p>
      <p>
        Then raise it to the power of the depth. A hop of 0.8 over ten layers is a
        factor of 0.107; a hop of 1.25 over ten layers is a factor of 9.3. Those
        two hops are the same size of mistake in opposite directions, because 1.25
        is 1 divided by 0.8, and ten layers turn either one into a factor of about
        nine, one way down and the other way up. Depth turns a small error in the
        hop into a large one in the network, in whichever direction the error
        happened to fall. That is the sense in which deep is hard: not that the
        equations stop working, but that a stack of layers only trains when the
        product of its hops is arranged to stay near 1, and nothing in the four
        equations arranges it.
      </p>
      <p>
        Arranging it is most of what changed after 2010. Initialization schemes
        pick the drawn spread from the layer's input count so the ledger factor
        starts at the right size, which is Chapter 7's division and its relatives
        (the ReLU version multiplies by a further{" "}
        <M tex="\sqrt{2}" /> to pay for the dead half). ReLU and its variants take
        the ceiling off the squash-slope factor. Normalization layers rescale each
        layer's evidence during training, so the hop is corrected as it drifts
        rather than only set at the start. Residual connections add a path that
        skips a layer entirely, so blame has a route back that no hop multiplies
        at all, which is what made networks of hundreds of layers trainable. Those
        are names, not explanations, and each is a paper rather than a paragraph.
        The point of the list is that they all act on the same number.
      </p>

      <SectionHeader id="c8-conv" title="Sharing the weights" />
      <p>
        Depth is one of the two things that made image networks work. The other
        changes what a layer is allowed to look at, and it can be read straight
        off a count you already have.
      </p>
      <p>
        The first weight matrix of your digit reader has one row per hidden
        neuron, and a row holds that neuron's incoming weights, one per pixel:
        30 rows of 784, <M tex="30 \times 784 = 23{,}520" /> weights. Every
        hidden neuron holds a private opinion about every pixel,
        and each of those opinions is learned separately. Two facts about
        handwriting are missing from that arrangement. A stroke in the top left
        and the same stroke in the middle are the same stroke, and the network has
        to learn it twice. And a pixel's neighbours tell you far more about it than
        a pixel on the other side of the image does, which nothing in a full row of
        784 weights expresses. The column of 784 does not even record which pixels
        touch: scramble every image's pixels in the same fixed order and your
        network would learn just as well, because the scramble only relabels which
        weight reads which pixel.
      </p>
      <p>
        A convolutional layer states both. One neuron looks at a 5-by-5 window of
        the image, 25 pixels, so it has 25 weights and a bias, 26 numbers. The 5
        is a choice, the way the hidden layer's 30 was: big enough to hold a
        piece of a stroke, small enough to stay local. Slide that window across
        the 28-by-28 image one pixel at a time. Its left edge can start anywhere
        from column 1 to column 24; one further and the 5-wide window would run
        past the image's edge. The same 24 fits down the page, so the window has{" "}
        <M tex="24 \times 24 = 576" /> positions. Put one neuron at every
        position, and give all 576 of them the same 26 numbers. That is weight
        sharing. Each of the 576 has its own 25 wires into its own patch, but the
        numbers on those wires come from one stored set, not from 576 private
        copies, so the window is 576 neurons holding 26 numbers between them. In
        every layer you have built, those two counts moved together because a
        neuron owned its weights. Here they come apart.
      </p>
      <Figure caption="One 5-by-5 window, three of its 576 positions, and the single set of 26 numbers all of them use. The layer's output is a 24-by-24 grid of neurons, one per position, each reporting how well its own patch of the image matches that one window.">
        <SharedWeightsFigure />
      </Figure>
      <p>
        Watch what the sharing does, on numbers small enough to hold. Shrink the
        image to one row of six pixels and the window to three wide; nothing
        changes but the counts. The window fits at four positions, by the same
        edge argument as the 24: from position 5, a 3-wide window would run past
        pixel 6. Pick the three shared weights by hand for the demonstration, 1,
        0 and −1, set the bias to 0, and leave the squash off so the raw sums
        stay visible. The row of pixels is 0.2, 0.9, 0.1, 0.0, 0.7, 0.3, and here
        are all four neurons:
      </p>
      <div className="table-scroll scroll-x" tabIndex={0}>
        <table className="truth-table">
          <thead>
            <tr>
              <th>position</th>
              <th>pixels it reads</th>
              <th>multiply and add</th>
              <th>output</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>0.2, 0.9, 0.1</td>
              <td>1 × 0.2 + 0 × 0.9 − 1 × 0.1</td>
              <td>0.1</td>
            </tr>
            <tr>
              <td>2</td>
              <td>0.9, 0.1, 0.0</td>
              <td>1 × 0.9 + 0 × 0.1 − 1 × 0.0</td>
              <td>0.9</td>
            </tr>
            <tr>
              <td>3</td>
              <td>0.1, 0.0, 0.7</td>
              <td>1 × 0.1 + 0 × 0.0 − 1 × 0.7</td>
              <td>−0.6</td>
            </tr>
            <tr>
              <td>4</td>
              <td>0.0, 0.7, 0.3</td>
              <td>1 × 0.0 + 0 × 0.7 − 1 × 0.3</td>
              <td>−0.3</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Read the table twice. Along the arithmetic: every row uses the same three
        weights, no fourth weight appears anywhere, and four outputs cost three
        weights and a bias. Along the meaning: 1, 0, −1 computes left pixel minus
        right pixel, which scores high where brightness drops from left to right,
        an edge. Read that way, the four outputs are one question asked at four
        places, and the row of them is a map: the bright-to-dark edge sits at
        position 2, where the pixels fall from 0.9 to 0.0. The map exists because
        the weights repeat. Give each position its own three weights and each
        position develops its own private notion of an edge, and the four outputs
        stop having anything in common. The 24-by-24 grid in the figure above is
        the same object at full size: one question, answered at 576 places.
      </p>
      <p>
        Outside a demonstration nobody picks the weights. The 26 numbers start
        random and Chapter 3's descent moves them like any others, so training
        decides which question the window ends up asking. The sharing guarantees
        only the shape of the outcome: whatever the numbers become, they are one
        question, asked everywhere. And a trained window can be looked at. Draw
        its 25 weights as a tiny image, the way Chapter 2 drew each hidden
        neuron's 784 as a patch of excite and suppress, and the learned questions
        show up as short edges at various angles, small curves, dots.
      </p>

      <SectionHeader id="c8-layer" title="A layer of twenty windows" />
      <p>
        One window asks one question, so a useful layer runs many, side by side.
        Twenty is another designer's pick, sized to cover a spread of small
        shapes: edges at a few angles, corners, the ends of strokes. Each of the
        twenty has its own 26 numbers, slides over the same original image, and
        fills its own 24-by-24 grid. None of them reads another's grid; delete
        one and the other nineteen compute exactly what they did before. The
        organization is one you already own. Your hidden layer is 30 neurons
        side by side, all reading the same 784 pixels, none reading each other,
        and a layer in both networks means that: the units that read one input
        in parallel. What changed is the repeated unit, there a neuron owning
        784 weights and reporting one number, here a window holding 26 and
        reporting a grid.
      </p>
      <Figure caption="Inside the window layer: every window reads the same image, hunts its own shape, and fills its own grid. Nothing flows between windows; the twenty run in parallel, the way your hidden layer's 30 neurons all read the same 784 pixels.">
        <WindowFanFigure />
      </Figure>
      <p>
        So the sharing has a boundary, and the purposes sit exactly on it.
        Inside one window, 576 positions read one set of 26 numbers: one
        purpose, applied everywhere. Across windows nothing is shared: twenty
        sets of 26, twenty purposes. Sharing across windows would buy nothing,
        which is the check that the boundary is in the right place: a window
        reading another's numbers would slide the same detector over the same
        image and fill in an identical grid, a copy paid for twice.
      </p>
      <p>
        Count the layer both ways. Neurons: twenty windows at 576 positions
        each, <M tex="20 \times 576 = 11{,}520" />, every one an ordinary
        multiply, add and squash. Numbers to learn:{" "}
        <M tex="20 \times 26 = 520" />. Put that beside your hidden layer, 30
        neurons owning 23,520 weights, and the two counts have traded places:
        far more neurons, far fewer numbers. The trade is not a memory trick;
        it is a claim about the world, that a detector worth having at one
        place is worth having at every place. Fewer numbers to learn is also
        fewer numbers to overfit, which is Chapter 7's third complaint answered
        by the architecture instead of by weight decay. The 520 is this layer's
        count, not the network's: the fully connected tail after the pooling
        still spends numbers the old way, and the saving lives where the
        network touches pixels, which is exactly where a private opinion per
        pixel wastes the most.
      </p>
      <p>
        One more piece usually follows: a pooling layer, which replaces each small
        block of a grid by a single number, typically the largest in the block.
        That halves the grid's width and height and throws away exactly the
        information that was making the layer sensitive to a shape's exact
        position. The max is meaningful only because a grid answers one question:
        the largest of a block's four numbers reads as "the shape was somewhere
        in this block". Stack window layers and pooling layers alternately and
        the later layers see larger and larger parts of the image, described in
        terms of what the earlier ones found.
      </p>
      <Figure caption="The whole network, four working layers. A window layer turns the image into twenty grids, one per window, 520 numbers in total. A pooling layer halves each grid to 12-by-12 and holds no numbers at all: keeping a block's largest needs nothing learned. The tail is the kind of network you trained in Chapter 5, a fully connected layer reading all twenty grids as one long column, then the 10 outputs.">
        <ConvNetFigure />
      </Figure>
      <p>
        This course does not implement any of it. A convolution written from scratch
        in NumPy inside Pyodide is slow enough that a training run would stop being
        something you watch, and the conceptual return past "the same weights, used
        at every position" is small: BP1 to BP4 are unchanged, and sharing a weight
        only means adding up its slope over every position it was used at. Nielsen's
        Chapter 6 builds one and reports above 99 percent on the full MNIST test
        set, which nothing in his Chapters 1 to 3 reaches.
      </p>

      <SectionHeader id="c8-embed" title="Where the network thinks" />
      <p>
        Every network in this course turns the numbers it is handed into numbers
        of its own choosing, and that habit connects a 2015 digit reader to the
        models people use now.
      </p>
      <p>
        Start with where data arrives. Chapter 1's concert had two inputs, weather
        and friend, so every situation was a point on a plane with two axes, and
        the four situations were four dots. An MNIST digit is the same idea with
        more axes: 784 of them, one per pixel, and an image is one point in that
        space. Call it the input space. Nobody chose it. It is whatever the data
        happened to be measured as.
      </p>
      <p>
        In that space one rule defeats a single neuron: go when exactly one of the
        two things is good, which puts the two go dots on opposite corners of the
        square. One neuron cuts with one straight line, and no straight line puts
        those two dots on one side and the two stay dots on the other. Chapter 1
        fixed it with two hidden neurons, and the fix is worth looking at again in
        terms of spaces rather than lines. Each hidden neuron reports a number, so
        the pair of them reports two numbers, so every input lands somewhere on a
        second plane with axes <M tex="h_1" /> and{" "}
        <M tex="h_2" />.
      </p>
      <Figure caption="Chapter 1's four concert situations, on the left as they arrive and on the right as the hidden layer reports them, using Chapter 1's solution (h1 asks whether at least one input is on, h2 whether both are). Green means go, gold means stay. On the left no straight line separates the colours. On the right the two green points have landed on top of each other at (0.95, 0.05), and the dashed line is the cut the output neuron makes.">
        <HiddenSpaceFigure />
      </Figure>
      <p>
        The right-hand picture is a different description of the same four
        situations, and in that description the problem is easy. Nothing was added:
        the same four inputs, re-expressed as what two neurons said about them.
        That second plane is a learned space, in the strict sense that the numbers
        which place a point in it are weights, and Chapter 3's descent is what finds
        weights. Your digit reader does the same thing at a larger size: its hidden
        layer turns 784 pixel values into 30 numbers, and those 30 numbers are a
        description of the digit that the network chose, by training, because
        descriptions like it made the cost small.
      </p>
      <p>
        Now give the same treatment to something that has no numbers at all. To
        put a word into a network you first have to make it a column, and the
        plainest starting point is one slot per word in the vocabulary: the column
        is all zeros except a single 1 in that word's slot. That is a one-hot
        column, the same shape as the one-hot label columns your digit reader has
        been training against. It carries no information beyond which word it is:
        every pair of distinct words is exactly as far apart as every other pair.
      </p>
      <p>
        Multiply it by a weight matrix, the same operation a layer performs, and
        something specific happens:
      </p>
      <Eq
        tex="\begin{bmatrix} 0.3 & -1.2 & 0.8 \\ -0.5 & 0.4 & 1.1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -1.2 \\ 0.4 \end{bmatrix}"
        gloss="A matrix times a one-hot column returns that column of the matrix, and nothing else: every other column is multiplied by zero. So this weight matrix is a table with one column per word, and the multiplication is a lookup. Here the vocabulary has three words and each gets two numbers; real ones have tens of thousands of words and hundreds of numbers each. Written the other way round, with the one-hot as a row on the left, the lookup returns a row instead, which is how you will most often see it described."
      />
      <p>
        That table is not written by hand. It is weights, so it starts as a random
        draw and it is trained by the descent you implemented, on a task plain
        enough to state in one line: given a word, predict the next one. The
        input is a word's one-hot column (a real model reads a run of words
        before it guesses, but one word shows the machinery), and the lookup
        hands the layers after it that word's column from the table. The output
        is built the way your digit reader's is, one slot per possible answer:
        ten slots there, one per digit; here one slot per word in the
        vocabulary, each holding a score for "the next word is this one". The
        label is the word that actually came next, as a one-hot column, the
        same kind your digit reader trains against, and nobody writes those
        labels: cover the next word in any sentence ever printed, and the
        sentence itself supplies the answer. From there the loop is Chapter 5's:
        compare the output column to the label, take the slopes, step every
        number against them, the table's columns included.
      </p>
      <p>
        Watch the task shape the table. Tuesday and Wednesday stand in front of
        nearly the same next words: morning, afternoon, evening, the ninth. The
        cost therefore asks for nearly the same output whichever of the two
        arrives. And past the lookup the network cannot see which word arrived:
        every later layer computes on the looked-up column alone, so the direct
        way to give two words the same prediction is to give them nearly the
        same column. That is what descent settles into. Every sentence that
        uses the two words alike hands their columns another pair of nearly
        identical corrections, and the columns travel together, step after
        step; a pair the task must tell apart, Tuesday and yellow, lowers the
        cost only by holding its columns apart. What comes out is a space where
        distance means interchangeability, not because anyone asked for that
        meaning, but because arranging it is what lowered the cost. The columns
        are called embeddings, and the space is the embedding space.
      </p>
      <p>
        A large language model is that substrate at a size this course cannot
        demonstrate, and made of the same parts. Columns of numbers, weight
        matrices multiplying them, a cost that scores predictions, gradients from
        the chain of factors, descent stepping against them, with the parameter
        count in the billions rather than the tens of thousands in your digit
        reader, and the training run measured in months of many machines rather
        than seconds of one browser tab. One genuinely new ingredient sits in the middle of it, called
        attention, which lets each position in a sequence decide which other
        positions to read from; this course does not cover it, and Chapter 9's
        closing list is where to go for it.
      </p>
      <p>
        Input space is where the data arrives, and whoever collected it chose it.
        Embedding space is where the network chooses to think, and it learns that
        choice by the descent you wrote in Chapter 3. Chapter 1's two hidden neurons
        and a word's column in an embedding table are the same move at two sizes.
      </p>
      <p>
        One thing is still missing before any of that is usable, and it is not
        another idea. The functions this course had you write have never been
        assembled: every training run so far was started by a panel that did the
        loading, the looping and the scoring around them. Chapter 9 is where you
        write that loop, and where this course's words get translated into the ones
        everyone else uses.
      </p>

      <Recap
        items={[
          "A layer's learning speed is the size of its bias gradient, which is how far its biases move in one step per unit of step size. At Chapter 7's start a 784-30-30-30-30-10 network measures 0.00254 at the first hidden layer against 1.439 at the output: 567 times slower, before a single step, and its first epoch on 5,000 images ends by answering the commonest digit for every image.",
          "That ratio is BP2 applied repeatedly. One hop multiplies the blame column by about 1 going back through the wire ledger, which is what dividing the weights by the square root of the input count buys, and by at most 0.25 for the squash's slope. The hop is about one fifth, so the gap is about five to the power of the depth: 5, 25, 125, 625, 3,125 against 4.8, 26.2, 117, 567, 3,002 measured.",
          "ReLU answers max(0, z), so its squash slope is 1 where the evidence is positive and 0 where it is not: a live neuron passes blame back untouched. The same network's hop rises to about 0.7, the gap falls from 567 to 4, four hidden layers average 90.9 percent over the last five epochs instead of 86.5, and the step size has to fall by a factor of ten to keep the flat half from killing neurons outright.",
          "The hop is a product with no reason to sit near 1. Multiplying every weight by 4 takes the sigmoid's hop to 0.62 and ReLU's to 2.99, where the first hidden layer moves ninety times as far as the output. Both failures are called unstable gradients, and careful initialization, ReLU, normalization layers and residual connections are all ways of holding that one number near 1.",
          "A convolutional layer uses one 5-by-5 window's 26 numbers at all 576 positions of the image, so twenty windows cost 520 numbers against 23,520 for one fully connected layer of 30. A hidden layer is a learned re-description of its input, which is what makes a word's column in an embedding table mean something, and what a language model is built out of at a scale of billions.",
        ]}
        deeper="Chapter 5 (why deep neural networks are hard to train)"
        href="http://neuralnetworksanddeeplearning.com/chap5.html"
      />
    </article>
  );
}

// One 5-by-5 window at three of its positions, all reading from the same 26
// numbers. Plot family (CLAUDE.md): natural scale, capped at the viewBox
// width, centred.
function SharedWeightsFigure() {
  const IMG = 150; // the 28x28 image, drawn at this size
  const cell = IMG / 28;
  const win = 5 * cell;
  const OUT = 120; // the 24x24 output grid
  const ocell = OUT / 24;
  const ox = 400;
  const oy = 40;
  // Three window positions, in pixel coordinates on the 28x28 image.
  const spots = [
    { r: 2, c: 3 },
    { r: 11, c: 10 },
    { r: 19, c: 17 },
  ];
  const boxX = 232;
  const boxY = 88;

  return (
    <svg
      {...fig(12, 11, 534, 208)}
      className="m8-conv"
      role="img"
      aria-label="A 5 by 5 window shown at three positions on a 28 by 28 image; all three point at one shared set of 26 numbers, which feeds a 24 by 24 grid of neurons, one per position."
    >
      {/* Both frames before everything else: the cells and wire ends paint
          on top, and m8-conv-image's fill is opaque. */}
      <rect x={20} y={40} width={IMG} height={IMG} className="m8-conv-image" />
      <rect x={ox} y={oy} width={OUT} height={OUT} className="m8-conv-image" />
      {[7, 14, 21].map((k) => (
        <g key={k}>
          <line x1={20 + k * cell} x2={20 + k * cell} y1={40} y2={40 + IMG} className="m8-conv-grid" />
          <line x1={20} x2={20 + IMG} y1={40 + k * cell} y2={40 + k * cell} className="m8-conv-grid" />
        </g>
      ))}
      {spots.map((s, i) => (
        <g key={i}>
          <rect
            x={20 + s.c * cell}
            y={40 + s.r * cell}
            width={win}
            height={win}
            className="m8-conv-window"
          />
          <path
            d={`M${20 + s.c * cell + win},${40 + s.r * cell + win / 2} L${boxX},${boxY + 16}`}
            className="m8-conv-wire"
          />
          <rect
            x={ox + s.c * ocell}
            y={oy + s.r * ocell}
            width={ocell}
            height={ocell}
            className="m8-conv-cell"
          />
          <path
            d={`M${boxX + 118},${boxY + 16} L${ox + s.c * ocell},${oy + s.r * ocell + ocell / 2}`}
            className="m8-conv-wire"
          />
        </g>
      ))}
      <rect x={boxX} y={boxY} width={118} height={34} rx={4} className="m8-conv-box" />
      <text x={boxX + 59} y={boxY + 15} textAnchor="middle" className="m8-conv-boxtext">
        25 weights
      </text>
      <text x={boxX + 59} y={boxY + 28} textAnchor="middle" className="m8-conv-boxtext">
        + 1 bias
      </text>
      <text x={20 + IMG / 2} y={30} textAnchor="middle" className="tiny-net-caption">
        the image, 28 × 28
      </text>
      <text x={ox + OUT / 2} y={30} textAnchor="middle" className="tiny-net-caption">
        one grid, 24 × 24
      </text>
      <text x={20 + IMG / 2} y={40 + IMG + 18} textAnchor="middle" className="tiny-net-caption">
        576 window positions
      </text>
      <text x={ox + OUT / 2} y={oy + OUT + 18} textAnchor="middle" className="tiny-net-caption">
        576 neurons, one each
      </text>
      <text x={boxX + 59} y={boxY + 52} textAnchor="middle" className="tiny-net-caption">
        shared by all 576
      </text>
    </svg>
  );
}

// The whole small network, once: the image, a window layer of twenty grids,
// a pooling layer, and the fully connected tail from Chapter 5. Plot family
// (CLAUDE.md): natural scale, tight viewBox. Four grids stand in for the
// twenty; the caption carries the count.
function ConvNetFigure() {
  const IMG = 84; // the 28x28 image at 3px per pixel
  const GRID = 72; // a 24x24 feature map at the same scale
  const POOL = 36; // the same map after pooling, 12x12
  const STEP = 7; // stack offset between the drawn grids
  const N = 4;
  const iy = 54;
  const mid = iy + IMG / 2;
  // Each stack's front grid sits at the footprint's bottom-left; the rest
  // step up and right behind it. Footprint = size + (N-1)*STEP, centred on
  // the wire axis.
  const gridX = 148;
  const gridFront = mid + (GRID + (N - 1) * STEP) / 2 - GRID; // 71
  const poolX = 293;
  const poolFront = mid + (POOL + (N - 1) * STEP) / 2 - POOL; // 89
  const gridRight = gridX + (N - 1) * STEP + GRID;
  const poolRight = poolX + (N - 1) * STEP + POOL;
  const fcX = 400;
  const outX = 470;
  const outYs = Array.from({ length: 10 }, (_, i) => iy + 4 + (i * (IMG - 8)) / 9);
  const capY = 161;

  return (
    <svg
      {...fig(8, 8, 481, 178)}
      className="m8-net"
      role="img"
      aria-label="Five columns left to right: the 28 by 28 image, a stack of twenty 24 by 24 grids labelled one window layer, a stack of twenty 12 by 12 grids labelled one pooling layer, then a fully connected layer and 10 output neurons. A bracket groups the two stacks as the new layers."
    >
      <path d={`M${gridX},40 L${gridX},36 L${poolRight},36 L${poolRight},40`} className="m8-net-brace" />
      <text x={(gridX + poolRight) / 2} y={28} textAnchor="middle" className="tiny-net-caption">
        the new layers
      </text>
      <rect x={16} y={iy} width={IMG} height={IMG} className="m8-conv-image" />
      {Array.from({ length: N }, (_, i) => (
        <rect
          key={i}
          x={gridX + (N - 1 - i) * STEP}
          y={gridFront - (N - 1 - i) * STEP}
          width={GRID}
          height={GRID}
          className="m8-conv-image"
        />
      ))}
      {Array.from({ length: N }, (_, i) => (
        <rect
          key={i}
          x={poolX + (N - 1 - i) * STEP}
          y={poolFront - (N - 1 - i) * STEP}
          width={POOL}
          height={POOL}
          className="m8-conv-image"
        />
      ))}
      <rect x={fcX} y={iy} width={18} height={IMG} rx={9} className="tiny-net-neuron" />
      {outYs.map((y, i) => (
        <circle key={i} cx={outX} cy={y} r={4} className="tiny-net-neuron" />
      ))}
      <path d={`M${16 + IMG + 4},${mid} L${gridX - 4},${mid}`} className="m8-conv-wire" />
      <path d={`M${gridRight + 4},${mid} L${poolX - 4},${mid}`} className="m8-conv-wire" />
      <path d={`M${poolRight + 4},${mid} L${fcX - 4},${mid}`} className="m8-conv-wire" />
      <path d={`M${fcX + 18 + 4},${mid} L${outX - 8},${mid}`} className="m8-conv-wire" />
      <text x={16 + IMG / 2} y={capY} textAnchor="middle" className="tiny-net-caption">
        the image
      </text>
      <text x={16 + IMG / 2} y={capY + 14} textAnchor="middle" className="tiny-net-caption">
        28 × 28
      </text>
      <text x={(gridX + gridRight) / 2} y={capY} textAnchor="middle" className="tiny-net-caption">
        one window layer
      </text>
      <text x={(gridX + gridRight) / 2} y={capY + 14} textAnchor="middle" className="tiny-net-caption">
        20 grids of 24 × 24
      </text>
      <text x={(poolX + poolRight) / 2} y={capY} textAnchor="middle" className="tiny-net-caption">
        one pooling layer
      </text>
      <text x={(poolX + poolRight) / 2} y={capY + 14} textAnchor="middle" className="tiny-net-caption">
        20 grids of 12 × 12
      </text>
      <text x={(fcX + outX + 4) / 2} y={capY} textAnchor="middle" className="tiny-net-caption">
        fully connected
      </text>
      <text x={(fcX + outX + 4) / 2} y={capY + 14} textAnchor="middle" className="tiny-net-caption">
        then 10 outputs
      </text>
    </svg>
  );
}

// The window layer from the inside: parallel windows, none reading another's
// output. Three are drawn and the captions carry the twenty. Plot family
// (CLAUDE.md): natural scale, tight viewBox. Shares .m8-net's stylesheet
// entries with ConvNetFigure.
function WindowFanFigure() {
  const IMG = 84; // the 28x28 image at 3px per pixel
  const BOX_W = 104;
  const BOX_H = 26;
  const boxX = 160;
  const GRID = 48; // a 24x24 grid at 2px per cell
  const gridX = 320;
  const rows = [60, 112, 200]; // box and grid centre lines; dots between 2 and 3
  const labels = ["window 1", "window 2", "window 20"];
  const iy = 130 - IMG / 2; // the image, centred on the middle of the rows
  const capY = 246;

  return (
    <svg
      {...fig(8, 28, 382, 244)}
      className="m8-net"
      role="img"
      aria-label="The 28 by 28 image on the left. Wires fan out to three boxes named window 1, window 2 and window 20, with dots standing in for the rest, and each box has its own wire to its own small grid on the right. Captions read twenty windows, 26 numbers each, and twenty grids, 24 by 24 each."
    >
      <rect x={16} y={iy} width={IMG} height={IMG} className="m8-conv-image" />
      {rows.map((r, i) => (
        <g key={i}>
          <path d={`M${16 + IMG + 4},${130} L${boxX - 4},${r}`} className="m8-conv-wire" />
          <rect x={boxX} y={r - BOX_H / 2} width={BOX_W} height={BOX_H} rx={4} className="m8-conv-box" />
          <text x={boxX + BOX_W / 2} y={r + 4} textAnchor="middle" className="m8-conv-boxtext">
            {labels[i]}
          </text>
          <path d={`M${boxX + BOX_W + 4},${r} L${gridX - 4},${r}`} className="m8-conv-wire" />
          <rect x={gridX} y={r - GRID / 2} width={GRID} height={GRID} className="m8-conv-image" />
        </g>
      ))}
      <text x={boxX + BOX_W / 2} y={166} textAnchor="middle" className="tiny-net-caption">
        ⋮
      </text>
      <text x={gridX + GRID / 2} y={166} textAnchor="middle" className="tiny-net-caption">
        ⋮
      </text>
      <text x={16 + IMG / 2} y={capY} textAnchor="middle" className="tiny-net-caption">
        the image
      </text>
      <text x={16 + IMG / 2} y={capY + 14} textAnchor="middle" className="tiny-net-caption">
        28 × 28
      </text>
      <text x={boxX + BOX_W / 2} y={capY} textAnchor="middle" className="tiny-net-caption">
        twenty windows
      </text>
      <text x={boxX + BOX_W / 2} y={capY + 14} textAnchor="middle" className="tiny-net-caption">
        26 numbers each
      </text>
      <text x={gridX + GRID / 2} y={capY} textAnchor="middle" className="tiny-net-caption">
        twenty grids
      </text>
      <text x={gridX + GRID / 2} y={capY + 14} textAnchor="middle" className="tiny-net-caption">
        24 × 24 each
      </text>
    </svg>
  );
}

// Chapter 1's contrarian, twice: as the inputs arrive, and as the hidden layer
// reports them. The hidden coordinates come from Chapter 1's own solution, so
// the two pages cannot drift apart.
function HiddenSpaceFigure() {
  const H1 = (x1: number, x2: number) => sigmoid(6 * x1 + 6 * x2 - 3);
  const H2 = (x1: number, x2: number) => sigmoid(6 * x1 + 6 * x2 - 9);
  const CORNERS: { x1: number; x2: number; go: boolean }[] = [
    { x1: 0, x2: 0, go: false },
    { x1: 1, x2: 0, go: true },
    { x1: 0, x2: 1, go: true },
    { x1: 1, x2: 1, go: false },
  ];
  const SIZE = 150;
  const pad = 34;
  // Both coordinates run 0 to 1 and the corners are real data, so the plot is
  // inset inside its frame: without it half of every dot sits on the border.
  const INSET = 0.09;
  const at = (v: number) => INSET + v * (1 - 2 * INSET);
  // How far past 0 and 1, in data units, the drawn frame reaches. The output
  // neuron's cut is extended by this so it spans the frame instead of floating.
  const EDGE = INSET / (1 - 2 * INSET);
  const px = (v: number, left: number) => left + pad + at(v) * SIZE;
  const py = (v: number) => pad + (1 - at(v)) * SIZE;

  const panel = (left: number, title: string, coords: (c: (typeof CORNERS)[0]) => [number, number]) => (
    <g>
      <text x={left + pad + SIZE / 2} y={20} textAnchor="middle" className="tiny-net-caption">
        {title}
      </text>
      <rect x={left + pad} y={pad} width={SIZE} height={SIZE} className="m8-space-frame" />
      {CORNERS.map((c, i) => {
        const [a, b] = coords(c);
        return (
          <circle
            key={i}
            cx={px(a, left)}
            cy={py(b)}
            r={6}
            className={c.go ? "pt-class1" : "pt-class0"}
          />
        );
      })}
    </g>
  );

  return (
    <svg
      {...fig(12, 1, 456, 238)}
      className="m8-spaces"
      role="img"
      aria-label="Left: the four concert situations plotted on the weather and friend axes, with the two go points on opposite corners so no straight line separates them from the two stay points. Right: the same four plotted on the two hidden neurons' reports, where both go points land at 0.95, 0.05 and a single dashed line separates them from the other two."
    >
      {panel(0, "input space (x₁, x₂)", (c) => [c.x1, c.x2])}
      {panel(250, "hidden space (h₁, h₂)", (c) => [H1(c.x1, c.x2), H2(c.x1, c.x2)])}
      {/* The output neuron's cut: 8 h1 - 8 h2 - 4 = 0, so h1 - h2 = 0.5. Slope
          1, so stepping out by EDGE in both coordinates stays on the line and
          reaches the frame. */}
      <line
        x1={px(0.5 - EDGE, 250)}
        y1={py(-EDGE)}
        x2={px(1 + EDGE, 250)}
        y2={py(0.5 + EDGE)}
        className="m8-space-cut"
      />
      {[0, 250].map((left) => (
        <g key={left}>
          <text x={px(0, left)} y={pad + SIZE + 13} textAnchor="middle" className="chart-tick">
            0
          </text>
          <text x={px(1, left)} y={pad + SIZE + 13} textAnchor="middle" className="chart-tick">
            1
          </text>
          <text x={left + pad - 7} y={py(0) + 4} textAnchor="end" className="chart-tick">
            0
          </text>
          <text x={left + pad - 7} y={py(1) + 4} textAnchor="end" className="chart-tick">
            1
          </text>
        </g>
      ))}
      <text x={34 + SIZE / 2} y={pad + SIZE + 28} textAnchor="middle" className="chart-tick">
        weather across, friend up
      </text>
      <text x={284 + SIZE / 2} y={pad + SIZE + 28} textAnchor="middle" className="chart-tick">
        h₁ across, h₂ up
      </text>
      <text x={284 + SIZE / 2} y={pad + SIZE + 44} textAnchor="middle" className="chart-tick">
        both green points are at (0.95, 0.05)
      </text>
    </svg>
  );
}
