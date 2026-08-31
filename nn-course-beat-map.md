# Neural Networks: the course map

A beat sheet and syllabus for the whole course, assembled from a full read of all ten chapters,
the workbench section table, the front page, and the design doc. This is the picture to hold in
your head: what story the course tells, in what order, and what each chapter hands the next.

Written 2026-08-30 by the course review session and committed the day after, alongside that
review's findings report (`nn-course-review-findings.md`). The review's fixes landed in PR #21,
and none of them changed the course's structure (no exercise was cut, moved or rescoped, and
every quoted number reproduced), so the beats and the number chain here describe the course as
merged.

---

## The story in one paragraph

A person teaches a machine to read handwriting by building the reader themselves, in one growing
Python file. Act I builds the machine by hand: a neuron is a weighted decision, which is a line;
lines fail XOR, so neurons stack into layers, layers are matrices, and a digit is 784 numbers,
which yields a working 784-15-10 reader whose 11,935 knobs nobody can set by hand. Act II makes
it learn: learning is descent on a cost landscape, but measuring slopes by nudging costs 23,870
network passes per step, and the chain rule collapses that to two, so the learner implements
backpropagation, proves it against the nudge method, and trains the reader to 89 percent. That is
the summit. Act III interrogates the 89: not the architecture's fault (universality), so fix the
training (three one-line changes, 92.1 percent), then ask why "just go deeper" fails (vanishing
gradients), which opens the door to ReLU, convolutions, embeddings, and the modern field. Act IV
hands over ownership: assemble the whole training program yourself, translate the course's
invented vocabulary into the field's, then point your own program at a raw, messy dataset and
learn the two habits that survive contact with real data: scale your inputs, and never trust one
number.

## The four acts

**Act I. Build the machine (m1, m2).**
From one hand-computed decision to a full digit reader running the learner's own `feedforward`
on real MNIST digits with pretrained weights. Cliffhanger: 11,935 knobs, far too many to set by hand.

**Act II. Make it learn (m3, m4, m5).**
Descent (m3) works but the bill is 1.19 billion network trips per full step; even mini-batched,
every step still pays two cost measurements per knob. Chapter 4 is the idea that deletes the bill
(the chain rule, blame, the four equations, no code), Chapter 5 is the implementation, the gradient
check, and the real training run. Cliffhanger resolved: 89 percent, in seconds instead of hours.
This is the course's summit and the two-minute demo.

**Act III. Understand it, then improve it (m6, m7, m8).**
Three suspects for the missing points (data, training, shape). Chapter 6 acquits the shape
(universality, an interlude, no code). Chapter 7 convicts the training and fixes it three times,
one line each: cross-entropy, the divided start, weight decay, worth about three points (92.1).
Chapter 8 tries the obvious next move, another layer, watches it fail, explains the failure out of
BP2 itself (the hop, one fifth per layer), and uses the repair kit as the bridge to the field:
ReLU, convolutions, embedding spaces, LLMs.

**Act IV. Make it yours (m9, m10).**
Chapter 9: the loop that was never yours. Assemble `train` and `accuracy` from your own parts (the
course's only assembly test), read the call map where every line says "yours, Chapter N", translate
19 coined words into the field's vocabulary, name the four things the course did not teach.
Chapter 10: a file, not a dataset. Palmer penguins, with words, holes, unequal classes and a
245-times scale gap; write `standardize`, `one_hot`, `split`, and watch the two lessons land:
unscaled scores exactly the majority-class baseline, and 73.5 percent hides a class the network
never answers.

## The number chain

Each chapter ends by producing a number, and the next chapter opens by picking that number up. This
chain IS the plot; if you remember nothing else, remember this ladder.

| # | The number it produces | What it does next |
|---|---|---|
| m1 | 9 knobs, hand-set to beat XOR, and it was fiddly | "Make the computer do the fiddling" = m3's promise |
| m2 | 11,935 knobs; a pretrained net reads 86% | Too many to set by hand; and 86% is the score to beat |
| m3 | 1.19B trips per full-batch step; mini-batch leaves 2 passes per knob per step; the toy panel's "360 times worse at real scale" | "That is why Chapter 5 exists" |
| m4 | 2 passes instead of 23,870, about 12,000 times cheaper, and exact | Chapter 5 builds it |
| m5 | **89%** on the held-out thousand, learner's code end to end; seconds vs hours | The gap to Nielsen's 95 is the next mystery |
| m6 | A 7,840,000-neuron box net fits all 5,000 training images and reads nothing else | The shape is acquitted; the training is charged |
| m7 | **92.1%** from three one-line fixes; a wrong step size costs more than all three earn | Obvious next move: go deeper |
| m8 | The 567x staircase; one fifth per hop; 5^depth | Deep needed new parts (ReLU, conv); and "your functions were never assembled" |
| m9 | **90.4%** from the learner's own complete program (gap to 92.1 explained: decay on, shuffle wobble) | Point the program at data nobody prepared |
| m10 | 42.6% unscaled = the baseline exactly; 100% scaled; 73.5% hiding 0-of-18 Chinstrap | The habits that transfer: scale inputs, breakdown before believing |

## The threads that run the whole course

1. **One world.** The concert decision (weather, friend, go/stay, the three personalities) opens
   m1 and never leaves: m3 scores the XOR net's cost, m4's stepper feeds it a concert corner,
   m6 turns the concert plane continuous, m8 replots the corners in hidden space. MNIST is the
   real scale; penguins (m10) are the wild scale.
2. **One growing file.** Eleven sections of one `nn.py` in the workbench. Every payoff panel runs
   the learner's own code with nothing borrowed, and says so.
3. **The bill.** The tally thread: 9, then 11,935, then 23,870 passes per step, then 2. Parameter
   counting (m2's underbrace tally) is reused at every scale down to m10's 107-knob penguin net.
4. **Blame (delta).** Coined in m4, coded in m5, re-read and swapped in m7 (the seam: the course's
   only edit-your-own-working-code moment), measured per layer in m8 (learning speed, the hop).
5. **The nudge method's career.** Hero in m3 (measures every slope), rival in m4 (priced out),
   referee in m5 (the gradient check, the course's strongest guarantee), derivation tool in m7
   (derives weight decay, certifies cross-entropy), diagnostic in m10 (the symptom table).
6. **Re-description.** m1's hidden space (the XOR playground's phase 2), m2's 784-numbers-for-15-
   reports trade, m8's embedding spaces where distance means interchangeability.
7. **A score gets a breakdown.** m5's per-digit table and eight most-confident mistakes, m7's
   four-line overfitting chart, m10's baseline-then-per-class habit.

## Rhythm

Code chapters: m1, m2, m3, m5, m7, m9, m10 (nine exercises total). Idea chapters, no exercise:
m4 (the algorithm's idea), m6 (expressiveness), m8 (depth and the field). Nielsen mapping:
m1-m3 ch.1, m4-m5 ch.2, m7 ch.3, m6 ch.4, m8 ch.5 (ch.6 pointed at); m9 and m10 follow no
chapter (added after the teaching review) and carry no Recap or Go deeper.

---

## Chapter cards

### m1 · From neurons to networks
**Logline:** A neuron is a weighted decision, a weighted decision is a line, and three neurons
beat the problem one line cannot touch.
- The concert decision (weather 6, friend 2, transport 1, threshold 5) computed by hand; threshold
  becomes bias; multiply-and-add earns its name, the dot product; z > 0 is the perceptron.
- The two-input neuron plotted: four situations at z = -5, -3, +1, +3; the z = 0 frontier is a
  straight line (y = mx + c in costume). Learning = run the picture backwards: place the line.
- OR and AND fall to a dragged line in seconds; XOR (the contrarian) cannot fall: the 1969 wall.
- The sigmoid upgrade: small nudges give small visible changes (the ground all later training
  stands on).
- The 2-2-1 network, nine numbers, hand-set: h1 the at-least-one detector (6, 6, -3), h2 the both
  detector (6, 6, -9), output listens h1-yes-h2-no (+8, -8, -4). The playground's hidden space
  shows re-description for the first time.
- **Exercise:** `sigmoid`, `fire` (3 lines, deliberately tiny, first trip through the workflow).
- **Hands forward:** the nine-knob fiddliness (to m3), the anatomy (weights on wires, bias in the
  neuron), the (n, 1) column law, the whole concert world.

### m2 · Feedforward
**Logline:** A layer is one formula, a digit is 784 numbers, and your own forward pass reads real
handwriting before any training theory.
- Two neurons' multiply-and-adds stacked = the matrix; a' = sigma(Wa + b); network = the rule
  repeated; "the rest of the course is finding good numbers for W and b".
- Shape discipline: rows name the receiver, W is the wire ledger, inner numbers must touch.
- MNIST: 28x28 ink levels unrolled to (784, 1). The 784-15-10 design; ten yes/no questions; the
  11,935 tally in display math.
- Geometry scales: images are points in 784-axis space, a hidden layer re-describes them in 15.
- A pretrained net (86%) dissected: each hidden neuron's weights drawn as a red/blue 28x28 image.
- **Exercise:** `feedforward` (4 lines; the zip-over-layers loop).
- **Payoff:** the diagram runs the learner's own saved code on ten real digits.
- **Hands forward:** 11,935 (to m3's bill), 86% (to m5's 89), re-description (to m8), the batch
  convention every trainer uses.

### m3 · Learning as descent
**Logline:** Score the network with one number, measure which way each knob tilts it, and walk
downhill; then price what that costs at real scale.
- Chapter 1's XOR net scored by hand: cost 0.0875. Learning = search over nine knobs.
- One slope by nudging (-0.044); nine slopes = the gradient (nabla, named); the update rule
  w <- w - eta * nabla C; one step drops the cost. Jumping to each curve's bottom backfires
  (0.1665): a slope is a compass, not a map.
- Overshoot on a 1D bowl (thresholds 0.5, 1.0); zigzag on an 8x-elongated valley.
- The bill: 11,935 knobs x 2 rescores x 50,000 images = 1.19 billion trips. Mini-batches cut the
  averaging only: 238,700 at batch 10. The race: per example looked at, the wobble wins.
- **Exercise:** `sgd_step`, `sgd` (the update rule + the shuffle/slice/step epoch loop).
- **Payoff:** a 2-8-1 toy (33 knobs) trains live on the learner's own sgd; 66 passes per step;
  "roughly 360 times that at digit-reader scale. That is why Chapter 5 exists."
- **Hands forward:** the X/Y column packing, epoch and mini-batch vocabulary, the unremoved
  2-passes-per-knob factor (m4's opening), the mysterious half in the cost (paid off in m7).

### m4 · Backpropagation, the idea (no code)
**Logline:** Every slope is already posted along the wires; walk backward once and read them all off.
- The bill reopens: 23,870 passes per step. Promise: all slopes, exact, for two passes.
- A two-neuron chain worked entirely by hand; the nudge's ripple logged box by box; every arrow
  posts a factor (log first, explain second).
- The currency-booth aside: factors are posted rates; through-rate = product, known before any
  raise exists.
- Three kinds of factor (multiplication partner, sigma-prime, the gap); their product -0.0508
  agrees with the nudge's -0.050 but is exact. Named: the chain rule.
- Only the entry factor is a knob's own; the shared road is priced once per neuron: delta, blame.
  Forks add. Forward pass, backward sweep, slopes read off: the whole algorithm.
- BP1-BP4 arrive as receipts for work already done, matched against the by-hand numbers and the
  2-3-1 stepper (13 knobs, live-draggable). Quiz: three predict-then-verify questions (the
  saturation one plants m7).
- **Hands forward:** delta and the four equations (m5 implements), sigma-prime's flatness at the
  ends (m7's complaint one, m8's hop factor), the nudge method's demotion to referee.

### m5 · Backpropagation, for real (the summit)
**Logline:** The four equations become fifteen lines of NumPy, an independent referee certifies
all 54 slopes, and the learner's own code trains the digit reader.
- One example per call (averages pass changes through, checked on owned numbers); one-hot defined.
- Each equation = one NumPy statement, with the 784-30-10 shapes; the gradient is a slope-shaped
  shadow of the network: 23,860 numbers.
- Keep receipts: feedforward plus two accumulator lists; negative indexing; the backward window
  sliding one slot per pass.
- **Exercise:** `sigmoid_prime`, `backprop`. The test ladder ends at the gradient check: 54 knobs
  of a 3-5-4-2 net vs the nudge oracle, agreement within 1e-7. The strongest guarantee in the course.
- The given-batch adapter arrives written for the learner, carrying a dormant `output_delta=None`
  seam (m7 will swap BP1 through it).
- **Payoff:** wall-clock duel (backprop vs nudge, on the learner's machine), then the real run:
  5,000 images, 15 epochs, ~89%. Under the chart, what 89 hides: per-digit counts and the eight
  most-confident mistakes.
- **Hands forward:** 89% (m6 opens on it), the seam (m7), the receipts idiom (m9 names it
  hand-built autodiff), the breakdown habit (m10).

### m6 · Universality, an interlude (no code)
**Logline:** The missing points are not the architecture's fault: a wide enough hidden layer can
trace any curve, proven with hand-placed weights, and the proof also shows why that is not enough.
- Three suspects for 89-vs-95: data (held fixed), training, shape. This page acquits the shape.
- One dial in, a hump target out; a single neuron only climbs-and-flattens. Declared a sub-case,
  not a detour: hold 783 pixels still, turn one.
- Crank the weight: the sigmoid becomes a step at -b/w (Chapter 1's boundary with nowhere to run),
  width 6/w. Placement rule b = -ws.
- Two steps make a bump; six numbers per bump; Chapter 1's XOR net was already a bump (re-read,
  numbers restated).
- Slice the dial, one bump per slice: a bar chart, a lookup table. Price table: doubling bars
  roughly halves the miss (2.311 down to 0.118), until bars are narrower than the switchover.
- Two inputs: bands cross into a tower (thresholded at 9); at 784 inputs, boxes: 7,840,000 hidden
  neurons fit all 5,000 training images and answer 0 everywhere else.
- Four limits: jumps unreachable, never exact, existing is not findable (the pretrained weights
  look like ink strokes, not bar edges), and the construction shares nothing (m8's subject).
- **Hands forward:** the verdict (training is charged: m7), findability and generalization left
  standing, the shareless-detectors waste (m8's conv answer), the unsquashed output neuron (m10's
  how-much row).

### m7 · Making it actually work
**Logline:** Three complaints against the training, each fixed by one line, each fix measured:
about three points, and none of them worth a wrong step size.
- The first-ever edit to working code: the two-line `output_delta` seam in the learner's own m5
  backprop (shipped verbatim; no auto-splice button by design).
- Complaint 1, badly wrong but barely learning: the saturated one-neuron demo (469 steps vs 273
  vs, later, 50). BP1 re-read: blame peaks at 0.148 then collapses; more wrong, less learning.
- Fix 1: cross-entropy built from a -ln price list; the cancellation leaves delta = a - y.
  CostSwapPanel: at a shared eta the new cost wins 86.5 to 62.6, but quadratic at eta 3.0 ties
  (86.6). Honest verdict: it buys learning at a sane step size, not a smarter network.
- Complaint 2, saturated at birth: the untrained hidden layer's evidence sits 7.43 from zero, 62%
  of squash slopes flatter than 0.01 (the sqrt pile-up of ~100 lit pixels).
- Fix 2: divide each layer's draw by sqrt(n_in). 7.43 becomes 0.78; 92.1 vs 87.8 after 15 epochs.
- Complaint 3, overfitting: 80 epochs on 1,000 images; training cost falls to 0.0093 while
  held-out cost bottoms at epoch 9 and rises. Named: overfitting, generalizing, underfitting.
- Fix 3: weight decay, derived by the nudge method into the (1 - eta lambda/n) factor (finally
  paying off m3's mysterious half). RegularizePanel trains all six runs on one press.
- Closing: hyperparameters, the 16-cell grid (a wrong eta costs 19+ points; the fixes earn ~2.5),
  the best-of-sixteen caveat, the validation/test split, four unimplemented cures.
- **Exercises:** `cross_entropy_cost`/`cross_entropy_delta` (+ the seam edit), `init_network`, `l2_step`.
- **Hands forward:** 92.1 and the 0.22 median slope (m8 opens on both), the seam that every later
  panel trains through, the vocabulary rows (held-out, overfitting, hyperparameter...), the
  validation discipline (m9's aside, m10's split).

### m8 · Why deep is hard, and what came next (no code)
**Logline:** Add a layer and the network gets worse; the reason was sitting in BP2 all along, and
each repair for it is a doorway into the modern field.
- Depth at eta 0.5: 91.6 / 92.0 / 89.2 / 86.5. The four-layer net's first epoch answers "1" for
  every image: 12.6%, exactly the share of 1s. No step size rescues it.
- Learning speed defined (the size of a layer's bias-gradient column, BP3's blame). Measured
  before any step: the 567x staircase between output and first hidden layer.
- The hop: BP2 split into two measured factors, ledger (~1, bought by m7's sqrt division working
  a second time) times squash slope (~0.21, ceiling 0.25). One fifth per hop; 5^depth predicted
  5/25/125/625/3125, measured 4.8/26.2/117/567/3002.
- ReLU: slope exactly 1 or 0; hops ~0.7; depth stops costing (at a tenfold smaller step), but an
  oversized step kills a layer (dead units). Umbrella term: unstable gradients; every post-2010
  fix holds one number near 1.
- Convolution answers m2's 23,520-weight count: one 5x5 window's 26 numbers used at 576 positions;
  a 3-weight edge detector worked by hand; twenty windows = 520 numbers; the whole conv net drawn
  with m5's tail.
- Where the network thinks: m1's XOR corners replotted in hidden space (one dashed line finishes);
  matrix-times-one-hot is a lookup; embedding space, where distance means interchangeability;
  an LLM is the same parts at billions, attention the one new ingredient.
- **Hands forward:** "your functions have never been assembled" (m9's whole job), the translation
  promise (m9's table), Nielsen ch.6 as the pointed-at follow-on.

### m9 · Assembling the program (closing chapter + capstone)
**Logline:** The one piece that was always the course's, the loop itself, becomes yours; then the
course translates its own vocabulary and names what it did not teach.
- Opens by tallying what the learner owns, chapter by chapter, and naming the gap: every training
  run so far was a panel doing the loading, looping, scoring.
- `accuracy` taught on a walked figure first: a (10, 3) confidence grid, argmax down columns,
  guesses [3, 0, 7] vs y [3, 0, 9], mean 0.666, the coin-flip 0.44-over-0.38 flagged.
- **Exercise:** `accuracy` + `train` (draw, permute, slice, batch_gradient with cross-entropy
  blame, l2_step, score once per epoch, in a prescribed order the tests pin to exact weights).
- The call map: every line labeled "yours, Chapter N" (only rng.permutation and the adapter are not).
- **Payoff:** FullTrainPanel runs the learner's file: ~90.4% at eta 0.5; the gap to m7's 92.1
  explained honestly (decay on, one generator, a measured point of shuffle wobble). Step size
  from the inside: 0.1 starts slower but ends level; 3.0 zigzags (m3's overshoot at real scale).
- The 19-row translation table (evidence = pre-activation, blame = error signal, the divided
  start = Xavier/He, a hop under 1 = vanishing gradients...).
- What the course did not teach: autodiff (your receipts lists are a hand-built version),
  momentum/Adam, data preparation (m10), error analysis (accuracy is one number).
- Where to go next: m10 first, then Nielsen 5-6, Karpathy, PyTorch tutorials, the Illustrated
  Transformer + attention paper, Goodfellow-Bengio-Courville.

### m10 · Your own problem
**Logline:** The same program pointed at a file nobody prepared, where the two habits worth
having are: scale the inputs, and never believe one number.
- Palmer penguins as data actually arrives: 344 rows, words in two columns, holes, 152/124/68
  (always-Adelie is right 4 times in 10), body mass and bill depth 245 times apart in scale.
- What the network expects: the (features, m) packing named; three kinds of question mapped to
  output layer and cost (all three already met).
- The step that decides everything: m7's evidence-pile argument rerun on one penguin (~4,200,
  still ~1,400 after the sqrt division): every hidden neuron saturated at birth. Standardize,
  with mean and spread measured on training rows alone (the leak rule).
- Words and holes: one-hot on the input side (m5's packing moved); all-zeros means none-of-these;
  the shuffled 60/20/20 (a species-sorted file cut unshuffled trains on zero Chinstrap).
- **Exercise:** `standardize`, `one_hot`, `split` (none of them a network; that is the point).
- **Payoff:** PenguinsPanel. Unscaled: 42.6%, exactly the always-Adelie baseline, with the BP2
  flat-slope explanation of why it cannot climb out. Scaled, same rows: 100%. Two features only:
  73.5%, which sounds fine until the per-class row reads 29/29, 0/18, 21/21.
- Picking a first network (one hidden layer, sized between in and out; the 9-8-3 net's 107 knobs
  by m2's counting; memorize twenty examples first); the seven-row symptom table; the initial-cost
  sanity number (0.693 per output).
- Closes on 14 lines of PyTorch, each mapped to the piece the learner wrote, and the claim that
  what transfers is understanding, not code.

---

## The exercise spine (the second telling of the same story, in code)

| Order | Section | Chapter | You write | Requires | The concept the code teaches |
|---|---|---|---|---|---|
| 1 | sigmoid-neuron | m1 | `sigmoid`, `fire` | nothing | one neuron's whole computation; the course's contracts |
| 2 | feedforward | m2 | `feedforward` | sigmoid-neuron | the layer rule repeated = the forward pass |
| 3 | given-cost | m3 | (written for you) | feedforward | the score and the nudge-and-measure gradient |
| 4 | sgd | m3 | `sgd_step`, `sgd` | given-cost | the update rule; training = repeating the step |
| 5 | backprop | m5 | `sigmoid_prime`, `backprop` | sigmoid-neuron | the four equations as code; receipts |
| 6 | given-batch | m5 | (written for you) | backprop | per-example slopes averaged; the dormant seam |
| 7 | cross-entropy | m7 | `cross_entropy_cost`, `cross_entropy_delta`, + seam edit | feedforward, backprop | the cost is a swappable yardstick |
| 8 | smart-init | m7 | `init_network` | nothing | the start is a design choice; comparable runs |
| 9 | l2 | m7 | `l2_step` | nothing | decay as one factor on the update |
| 10 | train | m9 | `accuracy`, `train` | five sections | assembly: the program is yours |
| 11 | prepare | m10 | `standardize`, `one_hot`, `split` | nothing | preparation is part of the model |

Structural notes worth knowing (from the section table, not visible in any one chapter):
- The graph forks at sigmoid-neuron: the sgd branch (feedforward, given-cost, sgd) and the
  backprop branch rejoin at cross-entropy. **sgd is structurally a dead end**: nothing downstream
  requires it (deliberate: the numerical-gradient trainer exists to be priced out), though m9's
  `train` re-writes its loop shape with better parts.
- Three exercises are doable cold (smart-init, l2, prepare); prepare is an island whose only
  consumer is m10's panel.
- Chapters 4, 6, 8 own no sections: the idea chapters.
- m9's train/accuracy is the course's only assembly assessment; everything earlier tests pieces
  (possibly with dependencies on loan).
