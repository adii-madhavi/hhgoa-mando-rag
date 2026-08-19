"""
Multilingual embedding backends.

Model notes
-----------
The E5 family REQUIRES asymmetric prefixes -- "query: " on queries and
"passage: " on documents. Omitting them silently costs several points of
Recall@1, and it is a common benchmarking bug, so the prefix lives in the model
registry rather than at the call site.

BGE-M3 is symmetric (no prefix) and 1024-dim. It is the strongest of the three
on multilingual retrieval, but at 568M params on a CPU-only box its query
encode latency is the thing to watch against the 200 ms end-to-end budget --
see evaluation/ for measured numbers rather than assumptions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    dim: int
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512


REGISTRY: dict[str, ModelSpec] = {
    "e5-small": ModelSpec(
        "e5-small", "intfloat/multilingual-e5-small", 384,
        query_prefix="query: ", passage_prefix="passage: "),
    "e5-base": ModelSpec(
        "e5-base", "intfloat/multilingual-e5-base", 768,
        query_prefix="query: ", passage_prefix="passage: "),
    "bge-m3": ModelSpec(
        "bge-m3", "BAAI/bge-m3", 1024),
}


class Embedder:
    """Thin wrapper over sentence-transformers with prefix handling."""

    def __init__(self, model_key: str, device: str = "cpu",
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        if model_key not in REGISTRY:
            raise KeyError(f"unknown model {model_key!r}; "
                           f"known: {sorted(REGISTRY)}")
        self.spec = REGISTRY[model_key]
        self.batch_size = batch_size
        t0 = time.perf_counter()
        self.model = SentenceTransformer(self.spec.hf_id, device=device)
        self.model.max_seq_length = self.spec.max_seq_length
        self.load_seconds = time.perf_counter() - t0

    def _encode(self, texts: list[str], prefix: str,
                batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.spec.dim), dtype="float32")
        payload = [prefix + t for t in texts] if prefix else texts
        vecs = self.model.encode(
            payload,
            batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine == dot product downstream
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def encode_queries(self, texts: list[str], **kw) -> np.ndarray:
        return self._encode(texts, self.spec.query_prefix, **kw)

    def encode_passages(self, texts: list[str], **kw) -> np.ndarray:
        return self._encode(texts, self.spec.passage_prefix, **kw)

    def time_single_query(self, text: str = "what is a corporation?",
                          repeats: int = 20) -> dict:
        """Measured per-query encode latency -- the runtime-path cost."""
        self.encode_queries([text])          # warm up
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.encode_queries([text], batch_size=1)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        return {
            "model": self.spec.name,
            "dim": self.spec.dim,
            "repeats": repeats,
            "p50_ms": round(samples[len(samples) // 2], 2),
            "p70_ms": round(samples[int(len(samples) * 0.7)], 2),
            "p100_ms": round(samples[-1], 2),
            "mean_ms": round(sum(samples) / len(samples), 2),
            "min_ms": round(samples[0], 2),
        }
