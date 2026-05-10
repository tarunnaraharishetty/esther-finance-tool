"""Rate limiter + retry helpers for external API calls.

Token-bucket limiter for steady throttling, plus a tenacity-based decorator
for transient-failure retries with exponential backoff. All Alpaca / news API
calls should pass through one of these.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import ParamSpec, TypeVar

from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class TokenBucket:
    """Simple thread-/coroutine-safe token bucket.

    capacity: max burst size. refill_rate: tokens added per second.
    """

    capacity: float
    refill_rate: float
    _tokens: float = 0.0
    _last_refill: float = 0.0

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + delta * self.refill_rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block (asyncio sleep) until ``tokens`` are available."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                await asyncio.sleep(deficit / self.refill_rate)


def with_retry(
    *,
    attempts: int = 5,
    min_wait: float = 0.5,
    max_wait: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: retry on transient failures with exponential backoff."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in Retrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=min_wait, max=max_wait),
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            ):
                with attempt:
                    return func(*args, **kwargs)
            raise RuntimeError("unreachable")

        return wrapper

    return decorator


def with_async_retry(
    *,
    attempts: int = 5,
    min_wait: float = 0.5,
    max_wait: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Async variant of :func:`with_retry`."""

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=min_wait, max=max_wait),
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)
            raise RuntimeError("unreachable")

        return wrapper

    return decorator
