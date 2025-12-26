import asyncio
import threading
import time
from typing import Optional


class AsyncTokenBucketLimiter:
    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._rate = float(rate_per_sec)
        self._capacity = (
            float(capacity) if capacity is not None else max(1.0, self._rate)
        )
        self._tokens = float(self._capacity)
        self._lock = asyncio.Lock()
        self._last_ts: Optional[float] = None

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            raise ValueError("tokens must be > 0")
        tokens = float(tokens)
        if tokens > self._capacity:
            self._capacity = tokens
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                if self._last_ts is None:
                    self._last_ts = now
                elapsed = now - self._last_ts
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity, self._tokens + (elapsed * self._rate)
                    )
                    self._last_ts = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait_for = (tokens - self._tokens) / self._rate
            await asyncio.sleep(wait_for)


class SyncTokenBucketLimiter:
    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._rate = float(rate_per_sec)
        self._capacity = (
            float(capacity) if capacity is not None else max(1.0, self._rate)
        )
        self._tokens = float(self._capacity)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            raise ValueError("tokens must be > 0")
        tokens = float(tokens)
        if tokens > self._capacity:
            self._capacity = tokens
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity, self._tokens + elapsed * self._rate
                    )
                    self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait_for = (tokens - self._tokens) / self._rate
            time.sleep(wait_for)
