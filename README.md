# rag-harness

A reliability-first Retrieval-Augmented Generation (RAG) system over the [Kubernetes documentation](https://github.com/kubernetes/website). The goal is not just to answer questions — it is to **measure** answer quality and catch regressions.

## Why

A RAG pipeline has three independent failure modes:

| Stage | Failure | Metric |
|---|---|---|
| Retrieval | Wrong chunks fetched | Context Recall |
| Generation | Answer not grounded in context | Faithfulness |
| Generation | Answer is factually wrong | Correctness |

Scoring each failure mode independently means you can pinpoint exactly what broke — not just that "something went wrong."

## Setup

**Requirements:** Python 3.12, an OpenAI API key.

```bash
git clone https://github.com/Yashbhadiyadra/rag-harness.git
cd rag-harness

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,eval]"

cp .env.example .env
# add your OPENAI_API_KEY to .env
```

## Usage

```bash
# Ingest the Kubernetes docs (clone, chunk, embed, index)
python -m rag_harness ingest

# Ask a question
python -m rag_harness query "How do I configure RBAC in Kubernetes?"

# Run the evaluation suite against golden cases
python -m rag_harness eval
```

Or run as an API server:

```bash
uvicorn rag_harness.api.server:app --reload
# POST /query  {"question": "..."}
```

## Quality gate

The eval suite scores every golden case in `evals/golden/` and fails if any metric drops below its threshold:

| Metric | Threshold |
|---|---|
| Context Recall | ≥ 0.80 |
| Faithfulness | ≥ 0.85 |
| Correctness | ≥ 0.75 |

Run the full check locally before committing:

```bash
make check   # lint + type-check + tests
```

## Project structure

```
src/rag_harness/
├── config.py       # Pydantic settings (models, paths, thresholds)
├── models.py       # Shared data models: Chunk, GoldenCase, EvalResult
├── ingest/         # Load → chunk → embed → index
├── retrieval/      # Query → top-k chunks
├── generation/     # Context + query → grounded answer
├── evaluation/     # Score outputs against golden cases
└── api/            # FastAPI server + CLI
```

Architecture decisions are documented in [`docs/adr/`](docs/adr/).

## Attribution

Kubernetes documentation is © The Kubernetes Authors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [NOTICE](NOTICE). Project source code is MIT licensed.
