"""
Deterministic cache for answerability verdicts.

Why this exists
---------------
The judge PASSED its quality gate (EXPERIMENTS.md E15: false answers cut
62-80%) but its measured latency is P50 909 ms and P70 11.1 s against a 137 ms
RAG budget. It cannot sit synchronously in the voice path.

The resolution is not to weaken the judge, it is to move it off the critical
path:

    cold query  -> answer from the cosine guard immediately;
                   judge runs in the BACKGROUND and stores its verdict
    warm query  -> the cached verdict is available in microseconds and gates
                   the answer before it is generated

So repeat traffic gets the judge's accuracy at cache-lookup cost, and first-time
traffic keeps the measured fast path. Nothing here is claimed to be part of the
<200 ms critical path -- the background call is timed separately as
`judge_async_ms`.

Key design
----------
The key must identify "this question against this evidence", so a verdict is
reused only when it is genuinely still applicable:

    sha256( normalized_query | sorted(evidence doc_ids) | index_version )

* the query is normalised (casefold, collapse whitespace, strip punctuation)
  so trivial phrasing differences hit the same entry
* evidence doc_ids are SORTED, because retrieval order may vary while the
  evidence set is identical
* index_version is included so rebuilding the index invalidates every verdict
  rather than silently serving judgements about passages that no longer exist

Bounded, thread-safe, in-process. An LRU with a hard cap; no persistence,
because a stale verdict surviving a restart is a correctness risk for a
guardrail and the recompute cost is one background call.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import regex

_WS = regex.compile(r"\s+")
_PUNCT = regex.compile(r"[^\p{L}\p{N}\p{M}\s]+")


def normalize_query(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace. Unicode-aware."""
    t = _PUNCT.sub(" ", (text or ""))
    t = _WS.sub(" ", t).strip().casefold()
    return t


def verdict_key(query: str, evidence_ids: list[str],
                index_version: str = "") -> str:
    payload = "|".join([
        normalize_query(query),
        ",".join(sorted(str(i) for i in evidence_ids)),
        str(index_version),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    in_flight_skips: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def as_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "stores": self.stores, "evictions": self.evictions,
                "in_flight_skips": self.in_flight_skips,
                "hit_rate": round(self.hit_rate, 4)}


class VerdictCache:
    """Thread-safe bounded LRU with optional TTL."""

    def __init__(self, max_entries: int = 5000, ttl_s: float | None = None):
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: str):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            stored_at, verdict = entry
            if self.ttl_s is not None and time.time() - stored_at > self.ttl_s:
                del self._data[key]
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return verdict

    def put(self, key: str, verdict) -> None:
        with self._lock:
            self._data[key] = (time.time(), verdict)
            self._data.move_to_end(key)
            self.stats.stores += 1
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.stats.evictions += 1
            self._in_flight.discard(key)

    def claim(self, key: str) -> bool:
        """
        Reserve a key for a background judge call.

        Returns False if this key is already cached or already being judged,
        so concurrent identical queries fire exactly one background call
        instead of N. Without this, a burst of the same question would
        multiply the LLM cost by the burst size.
        """
        with self._lock:
            if key in self._data or key in self._in_flight:
                self.stats.in_flight_skips += 1
                return False
            self._in_flight.add(key)
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._in_flight.discard(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
