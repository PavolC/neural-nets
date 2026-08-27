import { useState } from "react";
import { START, cloneNet, compute, fmt } from "./backpropNet";

// Module 4's exercise substitute: three predict-then-verify questions about
// where blame flows. Every number in the explanations is computed live from
// the same network definition the stepper uses, on its starting weights.

interface Question {
  setup: string;
  prompt: string;
  options: string[];
  correct: number;
  explain: string;
}

function buildQuestions(): Question[] {
  const base = compute(START);

  const cutWire = cloneNet(START);
  cutWire.W3[0][0] = 0;
  const q1 = compute(cutWire);

  const saturated = cloneNet(START);
  saturated.b3[0] = -8;
  const q2 = compute(saturated);

  return [
    {
      setup:
        `In the stepper, the wire from h₁ to the output carries weight 4.0, the strongest ` +
        `wire into the output. Suppose you slide it down to 0.`,
      prompt: "What happens to h₁'s blame δ?",
      options: [
        "It grows: the network leans on h₁ less, so h₁ must try harder",
        "It shrinks to exactly zero",
        "It stays the same: h₁'s own weights did not change",
      ],
      correct: 1,
      explain:
        `Blame reaches h₁ only through that one wire, and BP2 multiplies δ³ by the wire's ` +
        `weight on the way back: zero times anything is zero, so δ goes from ${fmt(base.d2[0])} ` +
        `to ${fmt(q1.d2[0])}. A neuron that cannot affect the output cannot be blamed for it. ` +
        `(δ³ itself changed too, from ${fmt(base.d3[0])} to ${fmt(q1.d3[0])}, because cutting ` +
        `the wire moved the output; but h₁'s share of it is exactly zero.) Verify in the ` +
        `stepper: click that wire, drag it to 0, watch h₁'s δ.`,
    },
    {
      setup:
        `Select the output neuron's bias and drag it from −2.0 down to −8.0. The output ` +
        `collapses to a³ = ${fmt(q2.a3[0], 4)}, about as wrong as possible: the right answer is 1.`,
      prompt: "What happens to the output's own blame δ³?",
      options: [
        "It grows: bigger mistake, bigger blame",
        "It stays roughly the same",
        "It nearly vanishes",
      ],
      correct: 2,
      explain:
        `δ³ goes from ${fmt(base.d3[0])} to ${fmt(q2.d3[0], 4)}, almost nothing. BP1 multiplies ` +
        `the gap (now nearly −1, as large as it gets) by σ′(z³), and at z³ = ${fmt(q2.z3[0])} ` +
        `the sigmoid is flat: its slope is ${fmt(q2.a3[0] * (1 - q2.a3[0]), 4)}. A saturated ` +
        `neuron barely responds to nudges, so no nudge looks worth taking, so gradient descent ` +
        `barely moves it: badly wrong and barely learning, at the same time. Module 7 ` +
        `starts from exactly this problem.`,
    },
    {
      setup:
        `On the starting weights, h₁'s activation is ${fmt(base.a2[0])} and h₃'s is ` +
        `${fmt(base.a2[2])}. Both feed the same output neuron, so at the last step both of ` +
        `their outgoing wires are judged against the same blame δ³.`,
      prompt: "Which outgoing wire gets the bigger slope (in size), h₁'s or h₃'s?",
      options: [
        "h₁'s wire: it carried the stronger signal",
        "h₃'s wire: it has more room to improve",
        "Equal: same receiving neuron, same blame, same slope",
      ],
      correct: 0,
      explain:
        `BP4 says a wire's slope is the receiver's blame times the activation the wire carried. ` +
        `The blame is the same for both, so the busier wire wins: ${fmt(base.gW3[0][0], 4)} for ` +
        `h₁'s wire against ${fmt(base.gW3[0][2], 4)} for h₃'s. A wire that carried almost ` +
        `nothing on this example gets almost no adjustment from it, which is fair: it barely ` +
        `participated in the mistake.`,
    },
  ];
}

const QUESTIONS = buildQuestions();

export function DeltaQuiz() {
  const [answers, setAnswers] = useState<(number | null)[]>(
    QUESTIONS.map(() => null),
  );
  const answered = answers.filter((a) => a !== null).length;

  return (
    <div className="interactive delta-quiz">
      <div className="interactive-controls">
        <button
          className="button-secondary"
          onClick={() => setAnswers(QUESTIONS.map(() => null))}
        >
          Reset quiz
        </button>
        <span className="bp-step-count">
          {answered} of {QUESTIONS.length} predictions checked
        </span>
      </div>
      {QUESTIONS.map((q, qi) => {
        const picked = answers[qi];
        return (
          <div key={qi} className="quiz-question">
            <p className="quiz-setup">
              <strong>Prediction {qi + 1}.</strong> {q.setup}
            </p>
            <p className="quiz-prompt">{q.prompt}</p>
            <div className="quiz-options">
              {q.options.map((opt, oi) => {
                const cls =
                  picked === null
                    ? ""
                    : oi === q.correct
                      ? "quiz-correct"
                      : oi === picked
                        ? "quiz-wrong"
                        : "quiz-dimmed";
                return (
                  <button
                    key={oi}
                    className={`quiz-option ${cls}`}
                    // aria-disabled, not disabled: the answer stays focusable
                    // so it can be read back, and the global disabled styling
                    // no longer fades the correct/wrong colours to a wash.
                    aria-disabled={picked !== null}
                    onClick={() => {
                      if (picked !== null) return;
                      setAnswers((prev) => prev.map((a, i) => (i === qi ? oi : a)));
                    }}
                  >
                    {picked !== null && oi === q.correct && (
                      <span className="quiz-tag">correct answer</span>
                    )}
                    {picked !== null && oi === picked && oi !== q.correct && (
                      <span className="quiz-tag quiz-tag-wrong">your answer</span>
                    )}
                    {opt}
                  </button>
                );
              })}
            </div>
            {picked !== null && (
              <p className="quiz-explain">
                {picked === q.correct ? "Right. " : "Not quite. "}
                {q.explain}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
