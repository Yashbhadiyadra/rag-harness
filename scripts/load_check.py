"""Concurrent load check for the /query endpoint.

Fires N concurrent POST /query requests against a locally-running
FastAPI server whose retriever + LLM are mocked at the module level.
Measures p50/p95/p99 latency and success rate at multiple concurrency
levels; emits a compact markdown table.

The mocks inject a fixed 200ms latency per LLM call so the numbers
report the async wiring's overhead, not OpenAI's throughput.

Usage:
    python scripts/load_check.py [--concurrency 10 25 50 100] \\
                                 [--requests-per-level 200] \\
                                 [--output docs/load-check/<ts>.md]

Nothing about this script hits the network. It boots the app in-process,
patches the retriever and LLM, and uses httpx.AsyncClient against the
FastAPI ASGI transport directly.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


async def _run_one(client: httpx.AsyncClient, question: str) -> tuple[float, bool]:
    """Fire one request, return (latency_ms, success)."""
    start = time.perf_counter()
    try:
        resp = await client.post("/query", json={"question": question})
        ok = resp.status_code == 200
    except Exception:
        ok = False
    latency_ms = (time.perf_counter() - start) * 1000.0
    return latency_ms, ok


async def _run_level(
    client: httpx.AsyncClient, concurrency: int, requests: int, question: str
) -> dict[str, float]:
    """Run `requests` calls, up to `concurrency` in flight at once."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _capped() -> tuple[float, bool]:
        async with semaphore:
            return await _run_one(client, question)

    start_wall = time.perf_counter()
    results = await asyncio.gather(*(_capped() for _ in range(requests)))
    total_seconds = time.perf_counter() - start_wall

    latencies = [ms for ms, _ in results]
    successes = sum(1 for _, ok in results if ok)

    return {
        "concurrency": concurrency,
        "requests": requests,
        "success_rate": successes / requests if requests else 0.0,
        "latency_p50": statistics.median(latencies) if latencies else 0.0,
        "latency_p95": _percentile(latencies, 95) if latencies else 0.0,
        "latency_p99": _percentile(latencies, 99) if latencies else 0.0,
        "throughput_rps": requests / total_seconds if total_seconds > 0 else 0.0,
        "wall_seconds": total_seconds,
    }


def _percentile(values: list[float], pct: int) -> float:
    """Nearest-rank percentile - matches the eval runner's helper."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    rank = math.ceil(pct / 100.0 * len(sorted_vals))
    return sorted_vals[max(0, rank - 1)]


async def _fake_chunk() -> object:
    from rag_harness.models import Chunk

    return Chunk(
        id="fake::0",
        text="fake chunk text",
        source_file="docs/fake.md",
        git_commit="abc",
        doc_version="v1",
        chunk_index=0,
        heading_path=["Fake"],
    )


async def _run_all_levels(
    levels: list[int], requests_per_level: int, injected_llm_ms: int
) -> list[dict[str, float]]:
    """Boot the app in-process with mocks and run the load check."""
    from rag_harness.api.server import app
    from rag_harness.models import Chunk

    chunk = Chunk(
        id="fake::0",
        text="fake chunk text",
        source_file="docs/fake.md",
        git_commit="abc",
        doc_version="v1",
        chunk_index=0,
        heading_path=["Fake"],
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[chunk])

    async def _fake_generate(question: str, chunks: list) -> str:
        await asyncio.sleep(injected_llm_ms / 1000.0)  # simulate LLM latency
        return "mocked answer"

    results: list[dict[str, float]] = []
    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            side_effect=_fake_generate,
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            for n in levels:
                # Bypass slowapi rate limits during the load check by patching
                # the limiter to a no-op for the run. Real rate limiting is
                # tested elsewhere.
                with patch.object(app.state.limiter, "enabled", False):
                    result = await _run_level(client, n, requests_per_level, "test?")
                results.append(result)
                print(
                    f"  concurrency={n:>4} → success={result['success_rate']:.0%} "
                    f"p50={result['latency_p50']:.0f}ms "
                    f"p95={result['latency_p95']:.0f}ms "
                    f"p99={result['latency_p99']:.0f}ms "
                    f"throughput={result['throughput_rps']:.0f}rps"
                )
    return results


def _render_markdown(results: list[dict[str, float]], injected_llm_ms: int) -> str:
    """Render the load-check results as a markdown table."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"# Load check - {ts}",
        "",
        f"Mocked retriever + LLM ({injected_llm_ms} ms injected per call).",
        "Numbers report the async wiring's overhead, not OpenAI's throughput.",
        "",
        "| Concurrency | Requests | Success | p50 ms | p95 ms | p99 ms | Throughput rps |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {int(r['concurrency'])} | {int(r['requests'])} | "
            f"{r['success_rate']:.0%} | "
            f"{r['latency_p50']:.0f} | {r['latency_p95']:.0f} | {r['latency_p99']:.0f} | "
            f"{r['throughput_rps']:.0f} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, nargs="+", default=[10, 25, 50, 100])
    p.add_argument("--requests-per-level", type=int, default=200)
    p.add_argument("--injected-llm-ms", type=int, default=200)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    print(
        f"Load check: {args.concurrency} × {args.requests_per_level} requests, "
        f"{args.injected_llm_ms}ms injected LLM latency"
    )
    results = await _run_all_levels(
        args.concurrency, args.requests_per_level, args.injected_llm_ms
    )
    md = _render_markdown(results, args.injected_llm_ms)

    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("docs/load-check") / f"{ts}.md"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
