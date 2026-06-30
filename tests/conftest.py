import os

# Set before any rag_harness module is imported — config.py instantiates Settings()
# at module level, so the env var must exist before collection begins.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
