"""Application settings loaded from environment variables and the .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for models, corpus, storage, and evaluation thresholds."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str

    # Models — default to cheap options; change only after flagging cost tradeoff
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"

    # Corpus — pinned to an immutable tag for reproducible ingest (see ADR-0002)
    # Tag: snapshot-initial-v1.32  SHA: bbb60b97e9bade8f5bd9cf3c4543243c55a4c0ca
    k8s_repo_url: str = "https://github.com/kubernetes/website.git"
    k8s_git_commit: str = "bbb60b97e9bade8f5bd9cf3c4543243c55a4c0ca"
    k8s_docs_subpath: str = "content/en/docs"

    # Storage
    chroma_db_path: str = "./chroma_db"
    chroma_collection: str = "rag_harness"
    embedding_cache_path: str = "./embedding_cache.db"

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_strategy: str = "dense"  # "dense" | "hybrid" | "hybrid-rerank" | "hyde" | "full"
    hybrid_rrf_k: int = 60  # RRF fusion constant; 60 is the Cormack et al. 2009 default

    # Reranking (requires `pip install -e '.[rerank]'`)
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_multiplier: int = 4  # retrieve top_k*4 candidates before reranking

    # Corrective RAG — critic-and-retry loop (higher quality, higher cost)
    corrective_rag_enabled: bool = False
    critic_correct_threshold: float = 0.7  # any chunk above this = Correct
    critic_incorrect_threshold: float = 0.3  # top chunk below this = Incorrect
    corrective_max_retries: int = 1  # extra retrieval attempts after query reformulation

    # Observability — override the built-in pricing table without editing source.
    # Values are (input_rate_per_million, output_rate_per_million) in USD.
    model_rates_overrides: dict[str, tuple[float, float]] = {}

    # Tracing (requires `pip install -e '.[observability]'`; see ADR-0009)
    tracing_enabled: bool = False
    tracing_endpoint: str = "http://localhost:6006/v1/traces"
    tracing_service_name: str = "rag-harness"

    # Logging
    log_level: str = "INFO"

    # Evaluation thresholds — dropping below any triggers a build failure
    threshold_context_recall: float = 0.80
    threshold_faithfulness: float = 0.85
    threshold_correctness: float = 0.75


settings = Settings()
