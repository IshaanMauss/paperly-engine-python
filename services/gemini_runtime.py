"""
Shared Gemini runtime controls for Paperly.

All Gemini calls in the Python engine should pass through this module. This
keeps extraction ownership inside Python and prevents separate modules from
quietly multiplying concurrency against the same API key.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Awaitable, Callable, TypeVar
import asyncio

T = TypeVar("T")


def _max_concurrency() -> int:
    raw_value = os.getenv("GEMINI_GLOBAL_MAX_CONCURRENCY", "3")
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return 3


def _min_gap_seconds() -> float:
    raw_value = os.getenv("GEMINI_MIN_SECONDS_BETWEEN_CALLS", "1.25")
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return 1.25


GEMINI_GLOBAL_MAX_CONCURRENCY = _max_concurrency()
GEMINI_MIN_SECONDS_BETWEEN_CALLS = _min_gap_seconds()

_SYNC_SEMAPHORE = threading.BoundedSemaphore(GEMINI_GLOBAL_MAX_CONCURRENCY)
_RATE_LOCK = threading.Lock()
_LAST_CALL_AT = 0.0


def _wait_for_rate_slot() -> None:
    global _LAST_CALL_AT
    if GEMINI_MIN_SECONDS_BETWEEN_CALLS <= 0:
        return

    with _RATE_LOCK:
        now = time.monotonic()
        wait_for = GEMINI_MIN_SECONDS_BETWEEN_CALLS - (now - _LAST_CALL_AT)
        if wait_for > 0:
            time.sleep(wait_for)
        _LAST_CALL_AT = time.monotonic()


async def run_gemini_async(call: Callable[[], Awaitable[T]]) -> T:
    """Run an async Gemini operation under the process-wide Gemini limiter."""
    await asyncio.to_thread(_SYNC_SEMAPHORE.acquire)
    try:
        await asyncio.to_thread(_wait_for_rate_slot)
        return await call()
    finally:
        _SYNC_SEMAPHORE.release()


def run_gemini_sync(call: Callable[[], T]) -> T:
    """Run a sync Gemini operation under the process-wide Gemini limiter."""
    with _SYNC_SEMAPHORE:
        _wait_for_rate_slot()
        return call()
