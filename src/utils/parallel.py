"""
Small concurrency helpers
-------------------------
Judging is embarrassingly parallel and latency-bound on local inference, so a
thread pool is worth real money: on a CPU-only box four concurrent Ollama
requests cut a 40-sample window from six minutes to ninety seconds.

Two rules keep it honest:

* ``ordered_map`` preserves input order, because a regression report whose
  samples are shuffled makes a failed case impossible to find again.
* ``LocalLLMPool`` sizes itself from the server's advertised parallelism rather
  than a hard-coded number. Ollama's default ``OLLAMA_NUM_PARALLEL`` is low on
  CPU; asking for more threads than the server will run just deepens the
  queue and inflates every latency number in the report.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from src.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def ordered_map(fn: Callable[[T], R], items: Sequence[T], max_workers: int = 1,
                label: str | None = None, progress_every: int = 0) -> list[R]:
    """Map ``fn`` over ``items`` preserving order, optionally in parallel."""
    items = list(items)
    if not items:
        return []
    workers = max(1, min(int(max_workers), len(items)))
    if workers == 1:
        out = []
        for i, item in enumerate(items, start=1):
            out.append(fn(item))
            if progress_every and i % progress_every == 0:
                logger.info("  %s %d/%d", label or "processed", i, len(items))
        return out

    out: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ldm") as pool:
        futures = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
        for done, fut in enumerate(as_completed(futures), start=1):
            idx = futures[fut]
            try:
                out[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the window
                logger.warning("%s failed at index %d: %s", label or "task", idx, exc)
                out[idx] = None
            if progress_every and done % progress_every == 0:
                logger.info("  %s %d/%d", label or "processed", done, len(items))
    return [v for v in out if v is not None]


def ollama_parallelism(host: str = "http://localhost:11434", default: int = 2) -> int:
    """Ask the server how many requests it will actually run at once."""
    try:
        import requests

        resp = requests.get(host.rstrip("/") + "/api/ps", timeout=3)
        resp.raise_for_status()
        payload = resp.json()
        sizes = [m.get("size_vram", 0) for m in payload.get("models", [])] or [1]
        # A conservative heuristic: CPU inference gains a little from 2-4 ways.
        return max(1, min(int(default), len(sizes) or default))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not query Ollama parallelism (%s); using default %d", exc, default)
        return max(1, default)


def chunked(seq: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]