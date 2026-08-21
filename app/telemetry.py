"""Privacy-preserving rolling latency percentiles for public product modes."""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque


class LatencyTracker:
    def __init__(self, capacity: int = 100, minimum_samples: int = 5):
        self.capacity = capacity
        self.minimum_samples = minimum_samples
        self._samples = defaultdict(lambda: deque(maxlen=capacity))
        self._lock = threading.Lock()

    def record(self, channel: str, mode: str, latency_ms: float) -> dict:
        key = (channel, mode)
        with self._lock:
            self._samples[key].append(float(latency_ms))
            values = list(self._samples[key])
        return self._summary(channel, mode, values)

    def snapshot(self, channel: str, mode: str) -> dict:
        with self._lock:
            values = list(self._samples[(channel, mode)])
        return self._summary(channel, mode, values)

    def _summary(self, channel: str, mode: str, values: list[float]) -> dict:
        count = len(values)
        ready = count >= self.minimum_samples
        ordered = sorted(values)

        def percentile(p: float) -> float | None:
            if not ready:
                return None
            index = max(0, math.ceil(p * len(ordered)) - 1)
            return round(ordered[index], 2)

        return {
            "category": f"{channel}_{mode}",
            "samples": count,
            "p50_ms": percentile(0.50),
            "p70_ms": percentile(0.70),
            "collecting": not ready,
        }


RUNTIME_LATENCIES = LatencyTracker()
