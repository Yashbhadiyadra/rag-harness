"""Tests for the stateless HTTP MCP server: auth, tenant scoping, provenance (ADR-0028)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_harness.api import mcp_server
from rag_harness.api.auth import TenantContext
from rag_harness.api.mcp_server import (
    TenantAuthMiddleware,
    _current_tenant,
    _provenance,
    query_docs_http_impl,
)
from rag_harness.config import settings
from rag_harness.models import Chunk


def _chunk(source_file: str, commit: str, version: str) -> Chunk:
    return Chunk(
        id=f"{source_file}::0",
        text="content",
        source_file=source_file,
        git_commit=commit,
        doc_version=version,
        chunk_index=0,
        heading_path=["H"],
    )


def test_provenance_reports_distinct_corpus_snapshots() -> None:
    chunks = [_chunk("a.md", "c1", "v1"), _chunk("b.md", "c1", "v2"), _chunk("c.md", "c2", "v1")]
    prov = _provenance(chunks)
    assert prov == {"corpus_commits": ["c1", "c2"], "doc_versions": ["v1", "v2"]}


@pytest.mark.asyncio
async def test_query_docs_http_uses_authenticated_tenant_collection() -> None:
    """The query tool must retrieve from the tenant's collection, never a default."""
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("a.md", "c1", "v1")])
    build = MagicMock(return_value=retriever)

    token = _current_tenant.set(TenantContext(tenant_id="acme", collection="tenant_acme"))
    try:
        with (
            patch("rag_harness.retrieval.factory.build_retriever", build),
            patch(
                "rag_harness.generation.generator.generate_async",
                new=AsyncMock(return_value="Answer [1]."),
            ),
        ):
            result = await query_docs_http_impl("q?", top_k=3)
    finally:
        _current_tenant.reset(token)

    # collection came from the tenant context, not the request
    assert build.call_args.kwargs["collection_name"] == "tenant_acme"
    assert result["answer"] == "Answer [1]."
    assert result["provenance"]["corpus_commits"] == ["c1"]


def _http_scope(auth: str | None) -> dict:
    headers = [(b"authorization", auth.encode())] if auth is not None else []
    return {"type": "http", "headers": headers}


class _RecordingApp:
    """Downstream ASGI app that records whether it ran and the tenant it saw."""

    def __init__(self) -> None:
        self.called = False
        self.seen_tenant: TenantContext | None = None

    async def __call__(self, scope: object, receive: object, send: object) -> None:
        self.called = True
        self.seen_tenant = _current_tenant.get()


class _Sends:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)


async def _noop_receive() -> dict:  # pragma: no cover - never awaited in these paths
    return {}


@pytest.mark.asyncio
async def test_auth_enabled_rejects_missing_token() -> None:
    downstream = _RecordingApp()
    sends = _Sends()
    mw = TenantAuthMiddleware(downstream)
    with patch.object(settings, "api_auth_enabled", True):
        await mw(_http_scope(None), _noop_receive, sends)

    assert downstream.called is False
    assert sends.messages[0]["status"] == 401
    assert (b"www-authenticate", b"Bearer") in sends.messages[0]["headers"]


@pytest.mark.asyncio
async def test_auth_enabled_accepts_valid_token_and_sets_tenant() -> None:
    downstream = _RecordingApp()
    tenant = TenantContext(tenant_id="acme", collection="tenant_acme")
    mw = TenantAuthMiddleware(downstream)
    with (
        patch.object(settings, "api_auth_enabled", True),
        patch("rag_harness.api.auth.resolve_tenant", return_value=tenant),
    ):
        await mw(_http_scope("Bearer good-key"), _noop_receive, _Sends())

    assert downstream.called is True
    assert downstream.seen_tenant == tenant
    # ContextVar is reset after the request
    assert _current_tenant.get() is None


@pytest.mark.asyncio
async def test_auth_disabled_falls_back_to_default_tenant() -> None:
    downstream = _RecordingApp()
    mw = TenantAuthMiddleware(downstream)
    with patch.object(settings, "api_auth_enabled", False):
        await mw(_http_scope(None), _noop_receive, _Sends())

    assert downstream.called is True
    assert downstream.seen_tenant is not None
    assert downstream.seen_tenant.tenant_id == "default"


def test_build_http_app_wraps_in_auth_middleware() -> None:
    # build_http_app needs the optional 'mcp' extra; skip where it is not
    # installed (CI installs .[dev] only). The auth/tenant/provenance logic
    # above is covered without mcp, so the important behaviour is always tested.
    pytest.importorskip("mcp")
    app = mcp_server.build_http_app()
    assert isinstance(app, TenantAuthMiddleware)
