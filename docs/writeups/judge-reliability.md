# My LLM judge agreed with the truth 94% of the time. That number was a lie.

If you build with LLMs, you have probably reached for an LLM as a judge. You
have a golden set, you ask a strong model to grade each answer pass or fail,
you average the verdicts, and you get a reliability number you put on a
dashboard. Mine said 94% agreement with ground truth. It felt great. It was
also the wrong number to trust, and the gap is the whole point of this post.

I run a reliability-first RAG system over Kubernetes documentation. The gate
that decides whether a change ships is an LLM judge scoring correctness. A
judge that quietly grades itself generously is not a small problem: it is the
measurement the entire product rests on. So I audited the judge the same way I
audit the system it grades. Here is what four probes found, with the exact
numbers, on a 160-case golden set.

## Finding 1: raw agreement flatters the judge by six points

The headline number most teams report is raw agreement: out of every answer
where the truth is known, how often did the judge agree? Mine was 0.938. The
problem is that raw agreement gives the judge full credit for verdicts it could
have gotten right by chance. Correct for chance with Cohen's kappa and the same
judge scores 0.875.

| Cohen's kappa | Raw agreement | False accepts | False rejects |
|---|---|---|---|
| 0.875 | 0.938 | 20/160 | 0/160 |

Six points does not sound like much until you remember what the number is for.
It is the confidence you have that your gate is telling you the truth. Reporting
0.938 when the chance-corrected reality is 0.875 is not lying on purpose. It is
using the flattering statistic because it is the default, and the default
overstates every judge in the same direction.

## Finding 2: the judge waves through confident, fluent, wrong answers

The direction of the errors matters more than the count. My judge produced zero
false rejects and twenty false accepts. It never failed a genuinely correct
answer. It passed twenty answers that were fluent, on-topic, and wrong.

That is the exact failure mode that hurts in production. Nobody ships because a
correct answer got a low score. Things break when a plausible, well-written,
incorrect answer sails through the gate and reaches a user who trusts it. A
judge tuned to be agreeable is a judge optimized for the failure you least want.

## Finding 3: the verdict flips when only the format changes

A correctness judge should grade substance, not presentation. So I took answers
that sit near the pass line and reworded them in meaning-preserving ways: same
facts, different formatting. A stable judge should not change its mind. Mine
did, and how often depended on the model.

| Judge model | Format-flip rate | False-accept | Cost per audit |
|---|---|---|---|
| gpt-4o-mini | 23% | 0% | $0.029 |
| gpt-4o | 13% | 0% | $0.48 |

Up to 23% of borderline verdicts flipped on formatting alone with the cheap
judge. The expensive judge was steadier at 13%, but 13% is still one in eight
borderline calls decided by presentation rather than correctness. If your eval
harness reformats answers anywhere in its pipeline, that fragility is silently
moving your pass rate around.

## Finding 4: judge choice is a measurable tradeoff, not a vibe

The received wisdom is to use the biggest judge you can afford. The numbers say
choose on evidence. The cheap judge is sixteen times cheaper and flips almost
twice as often. Neither false-accepts at the calibration probe. Whether the
extra stability is worth sixteen times the cost is a decision you can only make
with both numbers in front of you, which is the argument for auditing the judge
before you standardize on one, not after.

## The uncomfortable takeaway

An unaudited judge is not a measurement. It is a number that feels like one. The
same discipline you apply to the system under test, adversarial cases,
chance-corrected statistics, sensitivity probes, has to apply to the grader, or
the grader becomes the least-tested and most-trusted component you own. The
model provider grading its own output cannot close this gap: a judge that
validates itself is the fox guarding the henhouse. Third-party validation is the
only kind that means anything.

## A note on a number that is supposed to be zero

While auditing, I also measured resistance to fabricated facts planted directly
in the retrieved context. It scored zero, and that is by design. My system
trusts its corpus because the corpus is pinned to a specific commit and every
chunk carries its provenance. The defense against poisoned content is at
ingestion, not generation. Reporting that zero as an injection vulnerability
would be the same category error as reporting raw agreement as reliability:
a true number attached to the wrong claim. The honest version names what each
number actually measures, and what defends it.

## What I would ask you to check

If you run an LLM judge, three cheap checks are worth an afternoon. Report
chance-corrected agreement, not raw. Look at the direction of the errors, not
just the rate, because false accepts and false rejects are not equally costly.
And reword a handful of borderline answers to see whether the verdict holds.
Whatever you find, you will trust your gate more for having looked, and if you
never looked, you do not actually know how reliable your reliability number is.

---

Every figure here comes from a reproducible audit committed alongside the
system it grades (160 golden cases, ADR-0014). If you want the same report on
your own judge or retrieval, that is exactly what the free Reliability Audit
produces.
