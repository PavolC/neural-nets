# The method

How to build one of these. Written after finishing the first one (neural networks, five
days, 64 commits), so the numbers below are measured rather than estimated.

The short version: **you are not writing a course, you are debugging one against a
reader.** Building the machine and the first draft is a morning's work. Making it teach is
the project.

---

## What the shape actually is

Day one of course one, before lunch:

| 08:40 | repo skeleton, CLAUDE.md, feasibility spike (a real network training in the browser) |
| 09:17 | the exercise pipeline: editor, worker, tests, results, saved progress |
| 10:37 | chapters 1 to 3, complete with content and interactives |
| 12:50 | **chapters 1 to 3 rewritten from scratch**, because the learner's actual floor was below what the plan assumed |

Everything after 12:50 on day one was revision. Of 64 commits: 10 added chapter content,
7 added infrastructure, 38 rewrote content that already existed, 9 were fixes. **Budget
three revision passes for every unit of new content.** A plan that schedules only
authoring is off by a factor of three.

The productive unit is not a work session. It is **a reading session followed by two to
five commits within a few hours**, while the confusion is still articulable. The tightest
and most valuable stretch of course one was three commits in two hours, each from a live
read-through of the same chapter.

And the payoff for writing the rules down is measurable. Chapters written before the
playbook existed absorbed 11 to 13 revision commits each. Chapters written after it
absorbed 1 to 3.

One honest caveat, and a prediction for your own build: those later chapters had also had
less reading. The rules helped, but a chapter no one has read carefully is not a chapter
known to work. Expect the back half of your course to owe revisions it has not yet been
asked for.

## Roles

**You are the learner, and the learner is the test suite.** Thirty-one percent of course
one's commits would not exist without a real reader reporting real confusion. Do not
delegate that. If you already know the topic, this method is the wrong one: its whole
engine is your own confusion, reported fast and quoted verbatim.

Claude authors, draws, implements, benches, and keeps the rules file. You read, get stuck,
and say exactly where in your own words. "Over my head" and "we're just talking about
curves..... why??" were the two most valuable sentences anyone said during course one.
Both produced structural rewrites; neither was polite, specific, or well-formed feedback,
and neither needed to be.

## Phase 0: the design doc, before any code

Write it with Claude in one sitting, from `DESIGN-DOC-TEMPLATE.md`. What it must settle:
the goals in priority order, the non-goals, the chapter list with what each covers and
what the learner writes in it, the stack, the exercise contract, and the milestone order.

Two conventions from course one worth copying exactly:

- **Append, never revise.** When an open question gets answered, annotate it in place
  rather than editing the question away. The document stays a record of intent plus
  divergence rather than a retconned spec.
- **A "what the build actually produced" section** at the end, added as you go, recording
  every deliberate departure and why. Six months later this is how a reader tells a
  decision from an accident.

## Phase 1: day one, in this order

The four guards below all arrived in course one *after* the harm they prevent had already
happened, and each was cheap to build and expensive to retrofit. None needed topic
knowledge. Build them before the first chapter.

1. **The feasibility spike.** Take the riskiest runtime constraint (can this actually run
   in a browser tab in under a minute?) and prove it end to end with a reference
   implementation, before any content exists. Record the measurement, pin the runtime to
   it, and write down that the pin cannot move without re-running the spike.
2. **The learner floor**, as a list of what you do not know. Then confirm it: have Claude
   write two paragraphs at that floor and read them. If they land, the floor is right. If
   they are over your head, the floor is wrong and every chapter written against it will
   be too.
3. **The register section.** Copy it from `CLAUDE.md` and keep it. It touches every
   sentence, cannot be enforced mechanically, and course one retrofitted it over four
   finished chapters and then had to chase the drift five more times.
4. **The four cheap machines**, all committed day one:
   - the exercise checker (solutions pass, untouched skeletons fail for their own reason);
   - the bench harness, so the first measured number quoted in prose is already
     reproducible;
   - the notation reference on the front page, empty, with the rule that adding notation
     means adding a row in the same change;
   - the deploy path, green, before there is anything to deploy.

Also decide, on day one, things that are nearly free now and cost a course-wide retrofit
later: the component vocabulary (aside box, section header, figure families, recap), the
figure geometry families and their phone behaviour, the design tokens, and the front-door
question **"what does a first-time visitor see?"** Course one left its feasibility spike
hanging off the end of the nav as a build artifact for three and a half days because
nobody asked that.

## Phase 2: one cheap chapter, then the rules

Write **one** chapter. Read it. It will be wrong at the floor, not at the facts. Fix it,
and write down what kind of wrong it was.

Do not write chapters two through eight before this. Course one's first two chapters paid
for the other eight.

## Phase 3: the loop

Per chapter, repeat until the reading is clean:

1. Claude drafts the chapter to the playbook.
2. **You read it in one sitting**, out loud if you can, and stop at the first place you
   are lost. Do not push through to be nice.
3. Say what happened in your own words. Not "this could be clearer": "I read this and
   thought X". Course one's best fixes came from questions like "is this a single neuron
   per layer? what are we talking about here?" and "changing by 0.01 changes it by 0.01,
   so what?"
4. Claude fixes the passage, adds the generalized rule to `CLAUDE.md` with your quote as
   its evidence, and adds the incident to `CASEBOOK.md`. One commit.
5. **Seam review.** Before moving on, reconcile the new chapter against its neighbours:
   shared vocabulary, numbers quoted across chapters, cross-references, and anything the
   new chapter renamed.

For the hardest conceptual chapter in the course, a review is not enough: **book a live
tutoring session** on it. Course one's backpropagation chapter was restructured four times
in one day, and the version that worked came out of a session where the learner's
questions traced every failure in order.

## Phase 4: the passes at the end

Three of these, each a distinct kind of audit. Run them as separate commits so a
regression is attributable.

- **The teaching review.** Read the whole course as a stranger and ask the two questions
  that are invisible from inside a chapter: *what does the learner run end to end,
  unaided?* and *has any input ever arrived messy?* Course one's answers were "nothing"
  and "no", which is how it gained two chapters and a fourth goal at the end.
- **The house-style pass.** Measure every chapter against the ones that worked, on
  countable features: cleft openers, pronoun aphorisms, median sentence length, callbacks
  per paragraph, parentheticals. Course one's was the largest single commit in the project
  (20 files) and found the same drift from chapter 6 onward. Commit the script that
  computes the bands, so the thresholds in `CLAUDE.md` are checkable rather than
  aspirational.
- **The reproducibility pass.** Every quoted number re-derived by its committed bench,
  every command in `CLAUDE.md` actually run, every claim in the README checked against the
  code. Course one's front page said "the eight modules" over a list of eight when ten
  existed, because chapters nine and ten were added late and nothing tied the front door
  to the chapter registry.

## Working with more than one session

Course one ran parallel sessions and they collided: two branches independently fixed the
same class of mobile-overflow bug and both got merged six minutes apart.

- **Partition by artifact, not by bug class.** "Chapter 4" and "the stylesheet" are
  boundaries; "fix overflow everywhere" is not.
- **Pin shared infrastructure first** (dev server port, generated file paths) before
  fanning out.
- **Never bundle a content chapter with an unrelated cross-cutting pass.** Course one's
  chapter 6 arrived in a 25-file commit that also touched the stylesheet, the app shell
  and the runtime; two regressions landed in it that nobody was reviewing for UI, and the
  chapter itself then needed four rewrites.
- **A second reviewer with a different model is worth it.** One found a real numeric error
  in course one three commits from the end.

## Watch for these three

They are the failure modes that are invisible from inside the work.

1. **Structural conformance is not playbook conformance.** Course one's worst chapter had
   every required component (section headers, nav, aside, recap) at birth and still failed
   the reader completely, on motivation, register and callback density. A checklist of
   components does not check whether a chapter teaches.
2. **Fixing something adjacent to what was asked for.** It is where regressions come from,
   and it is worth naming out loud when it happens.
3. **Writing numbers or backward claims from memory.** Course one did it repeatedly and
   the bench caught it every time: "both numbers I first wrote from memory were wrong,
   which is what the bench is for."
