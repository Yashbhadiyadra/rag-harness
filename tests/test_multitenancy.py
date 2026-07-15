"""Tests for multi-tenant corpus isolation (ADR-0025).

The headline measurement: a retriever bound to tenant A's collection can only
ever return A's chunks, and a request authenticated as A is routed to A's
collection - never B's. Plus the routing, config-validation, and
no-silent-fallback guarantees the ADR promises.
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from rag_harness.api.auth import (
    TenantContext,
    default_tenant,
    require_api_key,
    resolve_tenant,
)
from rag_harness.api.server import app
from rag_harness.config import Settings, TenantSpec, settings
from rag_harness.models import Chunk
from rag_harness.retrieval.dense import DenseRetriever
from rag_harness.retrieval.factory import build_retriever

client = TestClient(app)

ACME_KEY = "acme-tenant-key"
ACME_HASH = hashlib.sha256(ACME_KEY.encode()).hexdigest()
GLOBEX_KEY = "globex-tenant-key"
GLOBEX_HASH = hashlib.sha256(GLOBEX_KEY.encode()).hexdigest()


def _tenants() -> dict[str, TenantSpec]:
    return {
        "acme": TenantSpec(key_hashes={ACME_HASH}, collection="acme_col"),
        "globex": TenantSpec(key_hashes={GLOBEX_HASH}, collection="globex_col"),
    }


def _http_request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("203.0.113.7", 12345),
    }
    return Request(scope)


# --- a fake Chroma so isolation is provable without onnxruntime -------------


class _FakeCollection:
    def __init__(self, name: str, rows: list[tuple[str, str, str]]) -> None:
        self.name = name
        self._rows = rows  # (id, document, source_file)

    def _meta(self, source_file: str) -> dict[str, str]:
        return {
            "source_file": source_file,
            "git_commit": "c",
            "doc_version": "v1",
            "chunk_index": "0",
            "heading_path": "[]",
        }

    def query(self, query_embeddings: object, n_results: int, include: object) -> dict[str, object]:
        rows = self._rows[:n_results]
        return {
            "ids": [[r[0] for r in rows]],
            "documents": [[r[1] for r in rows]],
            "metadatas": [[self._meta(r[2]) for r in rows]],
        }


_FAKE_STORE = {
    "acme_col": _FakeCollection("acme_col", [("a1", "ACME only document", "acme/a.md")]),
    "globex_col": _FakeCollection("globex_col", [("g1", "GLOBEX only document", "globex/g.md")]),
}


class _FakeChromaClient:
    def __init__(self, path: str) -> None:
        self._path = path

    def get_collection(self, name: str) -> _FakeCollection:
        if name not in _FAKE_STORE:
            raise ValueError(f"collection {name!r} does not exist")
        return _FAKE_STORE[name]


# --- tenant resolution ------------------------------------------------------


def test_resolve_tenant_routes_by_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tenants", _tenants())
    monkeypatch.setattr(settings, "api_keys", "")
    acme = resolve_tenant(ACME_KEY)
    assert acme == TenantContext(tenant_id="acme", collection="acme_col")
    globex = resolve_tenant(GLOBEX_KEY)
    assert globex == TenantContext(tenant_id="globex", collection="globex_col")
    assert resolve_tenant("stranger-key") is None


def test_flat_allowlist_key_maps_to_default_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    flat_key = "plain-key"
    monkeypatch.setattr(settings, "tenants", {})
    monkeypatch.setattr(settings, "api_keys", hashlib.sha256(flat_key.encode()).hexdigest())
    ctx = resolve_tenant(flat_key)
    assert ctx == default_tenant()
    assert ctx is not None and ctx.collection == settings.chroma_collection


async def test_require_api_key_returns_default_when_auth_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_auth_enabled", False)
    ctx = await require_api_key(_http_request({}))
    assert ctx == default_tenant()


async def test_require_api_key_resolves_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setattr(settings, "tenants", _tenants())
    monkeypatch.setattr(settings, "api_keys", "")
    ctx = await require_api_key(_http_request({"Authorization": f"Bearer {ACME_KEY}"}))
    assert ctx.tenant_id == "acme" and ctx.collection == "acme_col"


# --- retriever binding: the isolation boundary ------------------------------


async def test_dense_retriever_only_returns_its_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retriever bound to acme_col returns ACME docs and nothing from globex."""
    with (
        patch("rag_harness.retrieval.dense.build_async_client"),
        patch("rag_harness.retrieval.dense.chromadb.PersistentClient", _FakeChromaClient),
    ):
        acme = DenseRetriever(collection_name="acme_col")
        globex = DenseRetriever(collection_name="globex_col")

    acme._embed_query = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    globex._embed_query = AsyncMock(return_value=[0.0, 1.0])  # type: ignore[method-assign]

    acme_hits = await acme.retrieve_async("q", top_k=5)
    globex_hits = await globex.retrieve_async("q", top_k=5)

    assert [c.source_file for c in acme_hits] == ["acme/a.md"]
    assert all("GLOBEX" not in c.text for c in acme_hits)
    assert [c.source_file for c in globex_hits] == ["globex/g.md"]
    assert all("ACME" not in c.text for c in globex_hits)


def test_build_retriever_threads_collection() -> None:
    """build_retriever binds the leaf DenseRetriever to the requested collection."""
    with (
        patch("rag_harness.retrieval.dense.build_async_client"),
        patch("rag_harness.retrieval.dense.chromadb.PersistentClient", _FakeChromaClient),
    ):
        retriever = build_retriever("dense", collection_name="globex_col")
    assert isinstance(retriever, DenseRetriever)
    assert retriever._collection.name == "globex_col"  # type: ignore[attr-defined]


def test_missing_collection_raises_no_fallback() -> None:
    """An unprovisioned tenant collection errors instead of serving another corpus."""
    with (
        patch("rag_harness.retrieval.dense.build_async_client"),
        patch("rag_harness.retrieval.dense.chromadb.PersistentClient", _FakeChromaClient),
    ):
        with pytest.raises(ValueError):
            build_retriever("dense", collection_name="unprovisioned_col")


# --- end-to-end routing through /query --------------------------------------


def test_query_routes_each_tenant_to_its_own_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request as acme sees only acme's sources; as globex only globex's."""
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setattr(settings, "tenants", _tenants())
    monkeypatch.setattr(settings, "api_keys", "")

    def _chunk(source_file: str, text: str) -> Chunk:
        return Chunk(
            id=f"{source_file}::0",
            text=text,
            source_file=source_file,
            git_commit="c",
            doc_version="v1",
            chunk_index=0,
            heading_path=[],
        )

    def fake_get_retriever(collection: str) -> MagicMock:
        m = MagicMock()
        if collection == "acme_col":
            m.retrieve_async = AsyncMock(return_value=[_chunk("acme/a.md", "ACME doc")])
        elif collection == "globex_col":
            m.retrieve_async = AsyncMock(return_value=[_chunk("globex/g.md", "GLOBEX doc")])
        else:
            raise AssertionError(f"unexpected collection {collection!r}")
        return m

    with (
        patch("rag_harness.api.server._get_retriever", side_effect=fake_get_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="answer",
        ),
    ):
        acme_resp = client.post(
            "/query",
            json={"question": "q?"},
            headers={"Authorization": f"Bearer {ACME_KEY}"},
        )
        globex_resp = client.post(
            "/query",
            json={"question": "q?"},
            headers={"Authorization": f"Bearer {GLOBEX_KEY}"},
        )

    assert acme_resp.status_code == 200
    assert [s["source_file"] for s in acme_resp.json()["sources"]] == ["acme/a.md"]
    assert globex_resp.status_code == 200
    assert [s["source_file"] for s in globex_resp.json()["sources"]] == ["globex/g.md"]


# --- config validation ------------------------------------------------------


def test_overlapping_key_hash_rejected() -> None:
    """A hash in both the flat allowlist and a tenant is a startup error."""
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="x",
            api_keys=ACME_HASH,
            tenants={"acme": {"key_hashes": [ACME_HASH], "collection": "acme_col"}},
        )


def test_hash_shared_across_two_tenants_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="x",
            tenants={
                "a": {"key_hashes": [ACME_HASH], "collection": "a_col"},
                "b": {"key_hashes": [ACME_HASH], "collection": "b_col"},
            },
        )


def test_auth_enabled_with_only_tenant_keys_is_valid() -> None:
    """Tenant keys count toward "auth has keys"; API_KEYS need not be set."""
    s = Settings(
        _env_file=None,
        openai_api_key="x",
        api_auth_enabled=True,
        api_keys="",
        tenants={"acme": {"key_hashes": [ACME_HASH], "collection": "acme_col"}},
    )
    assert s.all_key_hashes == {ACME_HASH}


def test_tenant_collection_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        TenantSpec(key_hashes={ACME_HASH}, collection="")
