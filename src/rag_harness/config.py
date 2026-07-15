"""Application settings loaded from environment variables and the .env file."""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for models, corpus, storage, and evaluation thresholds."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str

    # Models - default to cheap options; change only after flagging cost tradeoff
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"

    # Corpus - pinned to an immutable git ref for reproducible ingest (ADR-0002,
    # ADR-0019). Defaults to the Kubernetes docs; override CORPUS_* to point the
    # harness at any markdown docs repo (bring-your-own-corpus). The resolved
    # commit SHA is the corpus checksum and is attached to every chunk.
    # Default pin: snapshot-initial-v1.32  SHA: bbb60b97e9bade8f5bd9cf3c4543243c55a4c0ca
    corpus_name: str = "k8s"
    corpus_repo_url: str = "https://github.com/kubernetes/website.git"
    corpus_git_ref: str = "bbb60b97e9bade8f5bd9cf3c4543243c55a4c0ca"
    corpus_docs_subpath: str = "content/en/docs"
    corpus_doc_glob: str = "*.md"

    # Storage
    chroma_db_path: str = "./chroma_db"
    chroma_collection: str = "rag_harness"
    embedding_cache_path: str = "./embedding_cache.db"

    # Ingest concurrency - bound parallel embedding-API calls to avoid rate
    # limits and preserve back-pressure. 4 is a reasonable default for
    # text-embedding-3-small at 512-input batches.
    ingest_embed_concurrency: int = 4

    # OpenAI SDK resilience - the SDK does exponential backoff + jitter
    # automatically. Two retries × ~20s timeout gives roughly a 40s worst-case
    # per call before we degrade to the honest refusal path. Tighter than
    # the SDK default (2 retries × 600s) but sensible for an interactive demo.
    openai_max_retries: int = 2
    openai_timeout_seconds: float = 20.0

    # Composite per-IP rate limit: sustained + burst. slowapi parses the
    # semicolon-separated form via limits.parse_many. Sized for the public
    # demo per ADR-0010; loosen via API_RATE_LIMIT for local dev.
    api_rate_limit: str = "10/hour;3/minute"
    api_max_question_length: int = 2000

    # Horizontal scale-out (ADR-0024). Empty = in-memory limiter + daily cap
    # (single-instance demo, no external infra). Set to redis://<host>:<port>
    # to share both across instances via one Redis; requires the [redis] extra
    # and lets Cloud Run max-instances rise above 1.
    redis_url: str = ""

    # Public-demo cost guardrails (ADR-0010). The daily cap is enforced
    # in-process - Cloud Run max-instances=1 keeps the counter single-writer.
    # DEMO_ENABLED=false is the emergency kill switch: /query returns 503 with
    # a demo_disabled body while health/ready/metrics stay reachable.
    demo_enabled: bool = True
    demo_daily_request_cap: int = 200

    # API authentication (ADR-0023). Opt-in: the public demo runs with auth off
    # (rate-limited + capped). Set API_AUTH_ENABLED=true for any non-demo
    # deployment. API_KEYS is a comma-separated list of SHA-256 hex digests of
    # accepted keys - never plaintext; generate one with `rag-harness hash-key`.
    # Enabling auth with an empty allowlist is a configuration error and fails
    # at startup (see _validate_auth_config), so a deployment can never believe
    # it is protected while accepting every request.
    api_auth_enabled: bool = False
    api_keys: str = ""

    # Retrieval
    retrieval_top_k: int = 5
    # "dense" | "hybrid" | "hybrid-rerank" | "hyde" | "full" | "decompose"
    retrieval_strategy: str = "dense"
    hybrid_rrf_k: int = 60  # RRF fusion constant; 60 is the Cormack et al. 2009 default

    # Reranking (requires `pip install -e '.[rerank]'`)
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_multiplier: int = 4  # retrieve top_k*4 candidates before reranking

    # Corrective RAG - critic-and-retry loop (higher quality, higher cost)
    corrective_rag_enabled: bool = False
    critic_correct_threshold: float = 0.7  # any chunk above this = Correct
    critic_incorrect_threshold: float = 0.3  # top chunk below this = Incorrect
    corrective_max_retries: int = 1  # extra retrieval attempts after query reformulation

    # Observability - override the built-in pricing table without editing source.
    # Values are (input_rate_per_million, output_rate_per_million) in USD.
    model_rates_overrides: dict[str, tuple[float, float]] = {}

    # Tracing (requires `pip install -e '.[observability]'`; see ADR-0009)
    tracing_enabled: bool = False
    tracing_endpoint: str = "http://localhost:6006/v1/traces"
    tracing_service_name: str = "rag-harness"

    # LLM judge response cache - see observability/llm_cache.py docstring.
    # Judges are near-deterministic at temperature=0 AND a stale judge score
    # is negligible-cost, so caching is safe for eval and ablation reruns.
    # Never caches generate() or the corrective critic - those are user-facing.
    llm_cache_enabled: bool = False
    llm_cache_path: str = "./llm_cache.db"

    # Ablation study - "relevant-but-incorrect" divergence category
    rbi_relevancy_min: float = 0.7  # answer_relevancy above this AND ...
    rbi_correctness_max: float = 0.5  # correctness below this = highlighted failure

    # Per-PR eval gate - a representative subset that runs on every PR to main.
    # Two cases from each of the eight golden categories (16 total), so the
    # gate exercises topic answers, the refusal path (unanswerable), and
    # version-sensitive facts. Sized so no single hard case dominates the
    # mean: at 5 cases one refusal swung correctness by 0.20; at 16 it moves
    # it by ~0.05, keeping the recalibrated thresholds stable. Cheap
    # (~$0.06/PR); the full 160-case suite runs nightly.
    eval_pr_subset_ids: list[str] = [
        "cluster-001",
        "cluster-002",
        "networking-001",
        "networking-002",
        "rbac-001",
        "rbac-002",
        "scheduling-001",
        "scheduling-002",
        "storage-001",
        "storage-002",
        "pods-001",
        "workloads-001",
        "unanswerable-001",
        "unanswerable-002",
        "version-sensitive-001",
        "version-sensitive-002",
    ]

    # Closed-loop eval (ADR-0020) - capture low-confidence /query traces as
    # golden-set review candidates. Off by default; the owner still reviews
    # every captured candidate before it can enter the golden set.
    closed_loop_enabled: bool = False
    closed_loop_queue_path: str = "./evals/review-queue/closed-loop.jsonl"

    # Logging
    log_level: str = "INFO"

    # Evaluation thresholds - dropping below any triggers a build failure.
    # Recalibrated from the 2026-07-14 ablation on the expanded 160-case
    # golden set (evals/experiments/ablation_20260714T064826+0000_164ff8c.md).
    # The gate protects the production config (dense); values sit below its
    # bootstrap CI lower bounds (recall 0.912, faithfulness 0.903,
    # correctness 0.875 per ADR-0011) with a buffer for small-sample and
    # LLM-judge noise, so production passes reliably while genuine
    # regressions fail. The heavy 'full' pipeline underperforms on this
    # factoid-heavy set and is not the gated config.
    threshold_context_recall: float = 0.85
    threshold_faithfulness: float = 0.85
    threshold_correctness: float = 0.82

    @property
    def api_key_hashes(self) -> set[str]:
        """The parsed allowlist of accepted key hashes (API_KEYS is comma-separated)."""
        return {h.strip() for h in self.api_keys.split(",") if h.strip()}

    @model_validator(mode="after")
    def _validate_auth_config(self) -> "Settings":
        """Refuse to start with authentication enabled but no keys configured.

        Fail-closed on misconfiguration: an operator who sets API_AUTH_ENABLED
        must also supply API_KEYS, so "auth on" can never silently accept every
        request.
        """
        if self.api_auth_enabled and not self.api_key_hashes:
            raise ValueError(
                "API_AUTH_ENABLED is true but API_KEYS is empty - refusing to "
                "start authenticated with no keys configured. Add at least one "
                "key hash (rag-harness hash-key) or set API_AUTH_ENABLED=false."
            )
        return self


settings = Settings()
