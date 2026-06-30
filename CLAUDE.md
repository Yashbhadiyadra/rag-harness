# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this every session.

## What this project is

A reliability-first Retrieval-Augmented Generation (RAG) system over the **Kubernetes
documentation** (corpus: `github.com/kubernetes/website`, CC BY 4.0). The point of the
project is not "answer questions" — it is to **measure** answer quality and catch
regressions. A RAG pipeline has three failure modes (retrieval miss, grounding miss,
generation miss); this system scores each independently.

Corpus scope: the English docs under `content/en/docs/`, ingested from a **pinned baseline
git commit**, with a bounded window of release history retained as the change stream. See
ADR-0002.

## Project context & forward compatibility (read before designing ingest/eval)

This is **Project 1 of a three-part reliability portfolio**, and later projects depend on
choices made here. Do not paint them into a corner:

- **Project 2 (stale-embedding detection)** consumes this repo's ingest history. Therefore
  ingest MUST record, for every chunk, the **source file path and the git commit / doc
  version it came from**. Never discard that provenance — it is Project 2's input signal.
- **Project 3 (agent trajectory evaluator)** reuses this repo's `evaluation` layer. Keep
  evaluation metrics and the golden-set format reusable and not hard-wired to single-turn
  RAG only.
- **Planned Project 1 v2 — corrective / "self-healing" RAG.** A later feature will add a
  critic step that checks answer faithfulness and, if low, re-retrieves with a reformulated
  query or returns "not enough information." Don't build it now, but keep `generation` and
  `evaluation` modular enough to insert a critic-and-retry loop later without a rewrite.

## Who you are working with

The owner is an engineer building this to learn and to defend it in interviews. Your job
is to be a senior pair-programmer and **teacher**, not an autonomous agent. Optimize for
their understanding, not just for shipping code.

## Working agreement (IMPORTANT — follow exactly)

1. **Plan before you build.** For anything beyond a one-line change, use plan mode: state
   what you intend to change, in which files, and why. Wait for explicit approval before
   editing files.
2. **Explain as you go.** After each change, explain in plain language what you did, why,
   and what the owner should understand about it. Assume they will be asked to defend this
   change in a technical interview.
3. **Never commit without explicit approval.** Show the diff, explain it, and wait for a
   clear "approved" / "commit it" before running `git commit`. No surprise commits.
4. **If the owner seems unsure, slow down.** Re-explain rather than proceed. A change they
   don't understand is a failure, even if it works.
5. **Small, reviewable steps.** Prefer several small, understandable commits over one large
   one.

## Before every commit

Run the full local check and make sure it passes:

```bash
make check     # ruff lint + ruff format + mypy + pytest
```

Do not commit if `make check` fails.

## Commit conventions — Conventional Commits

Format: `<type>(<scope>): <summary>` (imperative, lowercase, no trailing period).

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`, `build`.
Scopes (this repo): `ingest`, `retrieval`, `generation`, `evaluation`, `api`, `eval`,
`docs`, `ci`.

Examples:
- `feat(ingest): chunk markdown docs preserving heading hierarchy`
- `fix(retrieval): deduplicate near-identical chunks before reranking`
- `test(eval): add golden cases for RBAC questions`

## Branching

Trunk-based. `main` stays releasable. Branch as `feat/<slug>`, `fix/<slug>`,
`docs/<slug>`, or `chore/<slug>`. Open a PR into `main`; CI must be green.

## Architecture & where code goes

Package lives in `src/rag_harness/`:

| Module        | Responsibility |
|---------------|----------------|
| `ingest`      | Load K8s docs from a pinned git snapshot, clean, chunk, embed, index. |
| `retrieval`   | Query → candidate chunks (dense; later hybrid + rerank). |
| `generation`  | Retrieved context + query → grounded answer. |
| `evaluation`  | Score outputs: context recall/precision, faithfulness, correctness. |
| `api`         | FastAPI service + CLI. |

See `docs/architecture.md`. Record non-trivial decisions as a new ADR in `docs/adr/`.

## Coding standards

- Python 3.12, fully type-annotated. `mypy --strict` must pass.
- Public functions and modules have docstrings.
- Pydantic for data models and settings (`src/rag_harness/config.py`).
- Keep the core app dependency-light; evaluation tooling lives behind the `eval` extra.
- Update `CHANGELOG.md` (Unreleased) when behavior changes; update the relevant ADR or add
  one when an architectural decision changes.

## Corpus & attribution

The Kubernetes docs are CC BY 4.0 — attribution in `NOTICE` must be preserved. The project's
own code is MIT (`LICENSE`). Keep these distinct.

## Evaluation discipline

- The golden set lives in `evals/golden/` as `(question, reference_answer,
  relevant_doc_ids)` cases. It is **version-controlled and reviewed like code.** Start small
  (~30 hand-checked cases) and grow; never auto-generate cases into it without review.
- The reliability gate fails when a metric drops below its threshold. Treat a metric
  regression as a build failure, not a warning.

## Cost awareness (the owner is budget-conscious)

- Default to cheap models (`text-embedding-3-small`, `gpt-4o-mini`); don't switch to
  expensive models without flagging it.
- The eval gate calls an LLM, so it runs **nightly / on-demand, not on every PR**. Don't
  wire LLM-calling evals into the per-PR CI without discussing cost.
- Cache embeddings and avoid needless re-embedding during development.

## Do not

- Do not commit secrets. Keys live only in a git-ignored `.env`; `.env.example` documents them.
- Do not add heavyweight dependencies without flagging the tradeoff and getting approval.
- Do not bypass the eval layer "to move faster" — measuring reliability is the product.
