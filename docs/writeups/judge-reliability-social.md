# Judge-reliability writeup: platform versions

Distribution copy for the essay in `judge-reliability.md`. Same numbers, shorter
forms per platform. All figures trace to the committed ADR-0014 audit.

- **Substack / dev.to / Medium:** publish `judge-reliability.md` almost as-is
  (they render Markdown, including the tables). Use the essay's first line as the
  title.
- **LinkedIn:** the version below (no Markdown, short paragraphs, numbers inline).
- **X / Twitter:** the thread below.

Put the canonical essay somewhere with a stable URL first (Substack or a
personal site), then point the LinkedIn post and the thread at it.

---

## LinkedIn

My LLM judge said it agreed with the truth 94% of the time. That number was a
lie, and the gap taught me something every team using LLM-as-judge should check.

I run a reliability-first RAG system. An LLM judge decides whether each change is
good enough to ship. So I audited the judge the same way I audit the system it
grades. Four findings, on a 160-case golden set:

1. Raw agreement flatters the judge. It reported 0.938 agreement with ground
truth. Correct for chance with Cohen's kappa and the real number is 0.875. Six
points of confidence that were never there. Raw agreement is the default, and it
overstates every judge in the same direction.

2. The errors all point one way. Zero correct answers were failed. Twenty
wrong-but-fluent answers were passed. A judge tuned to be agreeable is optimized
for the exact failure you least want in production: a confident, plausible,
incorrect answer reaching a user who trusts it.

3. The verdict flips on formatting. Reword a borderline answer in a
meaning-preserving way and up to 23% of verdicts flip with a small judge, 13%
with a large one. That is presentation deciding your pass rate, not correctness.

4. Judge choice is a measurable tradeoff. The cheap judge was 16x cheaper and
flipped nearly twice as often. You can only make that call with both numbers in
front of you.

The takeaway: an unaudited judge is not a measurement, it is a number that feels
like one. Report chance-corrected agreement, look at the direction of the errors,
and reword a few borderline answers to see if the verdict holds. You will trust
your gate more for having looked.

Full writeup and the reproducible audit in the comments.

---

## X / Twitter thread

1/
My LLM judge said it agreed with the truth 94% of the time.

That number was a lie. Here is what auditing your own evaluator actually reveals,
with numbers.

2/
Raw agreement flatters the judge.

Mine: 0.938 agreement with ground truth. Correct for chance (Cohen's kappa) and
it drops to 0.875.

Raw agreement is the default metric, and it overstates every judge in the same
direction. 6 points of confidence that were never there.

3/
The errors all point one way.

0 correct answers failed. 20 wrong-but-fluent answers passed.

A judge tuned to be agreeable is optimized for the worst production failure: a
confident, plausible, wrong answer reaching a user who trusts it.

4/
The verdict flips on formatting.

Reword a borderline answer, same meaning, different format, and up to 23% of
verdicts flip with a small judge, 13% with a big one.

That is presentation moving your pass rate, not correctness.

5/
Judge choice is a measurable tradeoff, not a vibe.

The cheap judge: 16x cheaper, flips about twice as often.

"Use the biggest model" is a guess. You can only decide with both numbers in
front of you.

6/
The takeaway:

An unaudited judge is not a measurement. It is a number that feels like one.

Three cheap checks: report chance-corrected agreement, look at the direction of
the errors, reword a few borderline answers and see if the verdict holds.

7/
A model provider grading its own output cannot close this gap. A judge that
validates itself is the fox guarding the henhouse.

Full writeup and the reproducible audit: [link]
