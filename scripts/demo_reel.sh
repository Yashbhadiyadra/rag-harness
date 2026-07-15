#!/usr/bin/env bash
#
# Demo reel - runs the showcase flows end to end for a terminal recording.
#
# Each flow is a real command against the live system, not a mock:
#   1. a grounded answer with inline citations and sources
#   2. an honest refusal when the corpus cannot answer
#   3. the reliability gate passing on the per-PR golden subset
#
# Injection resistance, judge reliability, and citation accuracy are their own
# measured commands (security-eval / judge-audit / citation-eval) rather than a
# single query, because their threat models live in the retrieved context and
# the judge, not in one prompt - see the README table.
#
# Prerequisites:
#   - a populated Chroma index (make ingest)
#   - OPENAI_API_KEY in .env
#
# Record it with, for example:
#   asciinema rec demo.cast -c ./scripts/demo_reel.sh
# or feed it into a GIF tool (vhs, ttygif, agg).

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "Need a .env with OPENAI_API_KEY (cp .env.example .env, then add your key)." >&2
  exit 1
fi
set -a
. ./.env
set +a

PY=".venv/bin/python -m rag_harness"

banner() {
  printf '\n\033[1;36m===== %s =====\033[0m\n\n' "$1"
  sleep 1
}

banner "1/3  A grounded answer, with inline citations and sources"
$PY query "How do I expose a Deployment as a Service?"
sleep 2

banner "2/3  It refuses when the corpus cannot answer (no hallucination)"
$PY query "What is the airspeed velocity of an unladen swallow?"
sleep 2

banner "3/3  The reliability gate on the per-PR golden subset"
$PY eval --subset pr

printf '\n\033[1;32mDeeper measurements:\033[0m '
printf 'rag-harness judge-audit / security-eval / citation-eval / ablation\n'
