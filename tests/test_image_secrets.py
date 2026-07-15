"""Guard tests: no secret ever ends up baked into the container image (ADR-0026).

These are cheap static checks over the build inputs, not an image scan. They
fail if someone removes the mechanism that keeps secrets out of the image -
un-ignoring .env, hardcoding a key, or copying .env into the build.
"""

from pathlib import Path

_REPO = Path(__file__).parent.parent


def test_dockerignore_excludes_env() -> None:
    """.env must be docker-ignored so it can never be copied into the image."""
    lines = {line.strip() for line in (_REPO / ".dockerignore").read_text().splitlines()}
    assert ".env" in lines, ".env must be listed in .dockerignore (ADR-0026)"


def test_dockerfile_bakes_no_secret() -> None:
    """The Dockerfile must not hardcode a key or copy .env into the image."""
    dockerfile = (_REPO / "Dockerfile").read_text()
    assert "sk-" not in dockerfile, "no OpenAI key literal in the Dockerfile"
    assert "OPENAI_API_KEY=" not in dockerfile, "no baked OPENAI_API_KEY value"
    # No `COPY .env ...` sneaking the local secrets file in.
    copy_lines = [ln for ln in dockerfile.splitlines() if ln.strip().upper().startswith("COPY")]
    assert not any(".env" in ln for ln in copy_lines), "the Dockerfile must not COPY .env"


def test_manifest_sources_openai_key_from_secret_manager() -> None:
    """Production must inject OPENAI_API_KEY from Secret Manager, not plaintext."""
    manifest = (_REPO / "deploy" / "cloud-run.yaml").read_text()
    assert "secretKeyRef" in manifest, "OPENAI_API_KEY must come from Secret Manager"
    # It must not be set as a plaintext value in the manifest.
    assert "sk-" not in manifest, "no key literal in the deploy manifest"
