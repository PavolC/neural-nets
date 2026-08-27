# Course kit

Everything needed to build another interactive, self-contained course that teaches a
technical topic to one named learner by making them build the thing.

Extracted from the first one ([Grokking Nets](../README.md): five days, 64 commits, ten
chapters, nine coding exercises, a real network training in the browser), and specifically
from the parts of it that were not about neural networks.

## Start here

Drop this folder's contents into an empty repo and say:

```
/new-course music theory, for someone who plays guitar by ear
```

That command interviews you, tests your stated floor before believing it, writes the design
doc, and builds the day-one scaffold. Everything else is what it reads while doing that.

Without Claude Code slash commands, the equivalent is: read `METHOD.md`, then work through
`DESIGN-DOC-TEMPLATE.md`, then close every `FILL:` in `CLAUDE.md`.

## What is in the box

| file | what it is | when to read it |
|---|---|---|
| `CLAUDE.md` | the portable rules, with 20 topic-shaped holes marked `FILL:` | every session |
| `METHOD.md` | the process: phases, the feedback loop, the end passes | once, first |
| `CASEBOOK.md` | fifteen incidents from course one and the rules they produced | once |
| `DESIGN-DOC-TEMPLATE.md` | the plan to write before any code | once, at the start |
| `BRAND.md` | the shared visual identity, and how to wire it in | once |
| `brand/` | five files, ~480 lines: the identity itself | copy it |
| `.claude/commands/` | six slash commands that run the loop | they run themselves |

`CLAUDE.md` is the deliverable. The other files exist because two things it cannot contain
turned out to matter as much as its rules: **the loop that generates new rules**
(`METHOD.md`) and **the evidence that the existing ones are load-bearing** (`CASEBOOK.md`).
A rule with no incident behind it gets bent; a rule with a quote behind it does not.

## The six commands

| command | what it does |
|---|---|
| `/new-course <topic>` | interview, floor test, design doc, day-one scaffold, one chapter |
| `/chapter <n>` | seam check, beat plan, then draft or revise to the playbook |
| `/stuck <what happened>` | fix the passage, write the rule, log the incident, sweep for the same bug |
| `/seam-review <n>` | reconcile a chapter against its neighbours: vocabulary, numbers, claims |
| `/house-style` | measure every chapter on countable prose features against the bands |
| `/teaching-review` | read the finished course for what no single chapter can show |

`/stuck` is the important one. Thirty-one percent of course one's commits exist because a
real reader said something like "over my head" or "we're just talking about curves.....
why??" and that got turned into a fix plus a rule in the same commit.

## What this method assumes

**You do not know the topic.** The engine is your own confusion, reported fast and quoted
verbatim. If you already know the material, this is the wrong method: there is nothing to
drive the revisions, and the revisions are the project.

**You will read every chapter, in one sitting, and stop where you get lost.** Not push
through to be polite. The stopping point is the data.

**The rewriting is the work.** Course one built its whole machine and its first three
chapters in 117 minutes, then spent five days making them teach. Of 64 commits, 10 added
content and 38 rewrote content that already existed. Budget three revision passes for every
unit of new content.

## Code worth copying, if the new course has code exercises

The kit is deliberately docs and brand rather than a second application scaffold: a generic
scaffold for a topic that does not exist yet is a guess. But roughly 2,900 of course one's
21,700 lines under `src/` are topic-free, and the dependency direction never inverts
(nothing in `components/`, `runtime/` or `state/` imports a chapter), so they lift cleanly.
Worth cribbing from `../src/`:

- `runtime/messages.ts` and `runtime/workerClient.ts`: the test protocol is two opaque code
  strings in, one structured verdict out. It knows nothing about the language.
- `python/harness.py` (98 lines): runs the learner's code as a `submission` module and
  collects `test_*` functions. The only place the Python assumption lives.
- `components/ExercisePage.tsx` (467 lines): three couplings to Python, everything else
  generic. Its accumulated UX fixes are the expensive part.
- `components/CodeEditor.tsx`: three lines couple it to Python. Swap one language package.
- `components/ModuleBits.tsx`: the chapter building blocks, including a table of contents
  that discovers its own sections from the DOM and needs no configuration.
- `state/progress.ts` (172 lines): no chapter ids, no exercise ids, no topic knowledge.
  Progress export and import come along free.
- `exercises/types.ts` (19 lines) and the per-exercise folder convention.
- `tools/check_exercises.py` (115 lines): the invariant is the reusable part. Solutions
  pass; untouched skeletons fail for their own reason.
- `App.tsx` and the `ModuleDef` registry: a complete tabbed course shell with lazy chapters
  and preloading, in under 200 lines.
- `tools/check_brand.py` and `tools/brand_palette.py`: copy both with `brand/`.

What does not transfer: the chapter prose, the interactives, the exercise Python, the
datasets and their loaders, and every bench. The benches carry a discipline rather than
code, and it is cheap to re-establish.

If the new topic has **no** code exercises, about 1,100 of those lines go dead and what is
left is a themed article shell with saved progress. At that point `CLAUDE.md` is the whole
asset, which is fine: it is the part that took five days.
