from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str

    # Models — default to cheap options; change only after flagging cost tradeoff
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"

    # Corpus — pinned git commit ensures reproducible ingest (see ADR-0002)
    k8s_repo_url: str = "https://github.com/kubernetes/website.git"
    k8s_git_commit: str = "main"  # replace with a pinned SHA before first ingest
    k8s_docs_subpath: str = "content/en/docs"

    # Storage
    chroma_db_path: str = "./chroma_db"
    chroma_collection: str = "rag_harness"

    # Retrieval
    retrieval_top_k: int = 5

    # Evaluation thresholds — dropping below any triggers a build failure
    threshold_context_recall: float = 0.80
    threshold_faithfulness: float = 0.85
    threshold_correctness: float = 0.75


settings = Settings()
