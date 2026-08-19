"""
Hybrid retrieval: dense + BM25 -> fusion -> dedup -> passage-level candidates.

Fusion choice
-------------
We fuse with Reciprocal Rank Fusion rather than a weighted sum of scores.
Dense cosine similarities live in [-1, 1] and are tightly clustered; BM25
scores are unbounded and corpus-dependent. Normalising them onto a common
scale requires per-query min/max normalisation, which is unstable when one
retriever returns few hits -- a frequent case for BM25 on short Devanagari
queries. RRF only consumes RANKS, so it sidesteps the calibration problem
entirely and is why it is the default in most production hybrid stacks.

    RRF(d) = sum over retrievers of  weight / (k + rank(d))

k=60 is the standard constant from Cormack et al.; it damps the influence of
any single retriever's top hit so one confident-but-wrong retriever cannot
dominate the fused list.

Dedup
-----
Chunks collapse to their parent passage, keeping the best-ranked chunk per
passage, because the generator consumes passages (with sentence-window or
hierarchical context attached), not raw chunks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidate:
    doc_id: str
    text: str            # the chunk that matched
    context: str         # what the generator should actually see
    query_id: int
    passage_index: int
    lang: str
    # Row of this chunk in the dense matrix. Carried so downstream stages
    # (reranking, grounding) can REUSE the embedding the index already holds
    # instead of re-encoding the same text. On CPU an encode is ~23 ms/text,
    # and the naive pipeline encoded each evidence passage three times.
    row: int | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class RetrievalTiming:
    dense_ms: float = 0.0
    bm25_ms: float = 0.0
    fusion_ms: float = 0.0
    # The encoded query, carried out of the stage so later stages (reranking,
    # grounding) do not pay a second ~15 ms encode for the identical string.
    query_vector: "np.ndarray | None" = None

    @property
    def total_ms(self) -> float:
        return self.dense_ms + self.bm25_ms + self.fusion_ms


class HybridRetriever:
    """
    Holds a dense matrix and a BM25 index over the SAME chunk list, so row i
    means the same chunk to both retrievers. That invariant is what makes rank
    fusion valid; it is asserted at construction.
    """

    def __init__(self, chunks, dense_matrix: np.ndarray, bm25_index,
                 embedder, rrf_k: int = 60,
                 dense_weight: float = 1.0, bm25_weight: float = 1.0):
        if len(chunks) != dense_matrix.shape[0]:
            raise ValueError(
                f"chunk/dense row mismatch: {len(chunks)} vs "
                f"{dense_matrix.shape[0]} -- fusion by rank would be invalid")
        if bm25_index is not None and len(chunks) != len(bm25_index.texts):
            raise ValueError("chunk/bm25 row mismatch")
        self.chunks = chunks
        self.matrix = dense_matrix
        self.bm25 = bm25_index
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def _dense(self, query: str, top_k: int):
        qv = self.embedder.encode_queries([query], batch_size=1)[0]
        scores = self.matrix @ qv
        n = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx], qv

    def retrieve(self, query: str, top_k: int = 10, candidate_k: int = 50,
                 use_dense: bool = True, use_bm25: bool = True):
        timing = RetrievalTiming()
        fused: dict[int, Candidate] = {}

        def slot(row: int) -> Candidate:
            if row not in fused:
                c = self.chunks[row]
                fused[row] = Candidate(
                    doc_id=c.chunk_id, text=c.text, context=c.context,
                    query_id=c.query_id, passage_index=c.passage_index,
                    lang=c.lang, row=row,
                    meta={"chunk_type": c.chunk_type, "strategy": c.strategy,
                          "parent_id": c.parent_id, **c.meta})
            return fused[row]

        if use_dense:
            t0 = time.perf_counter()
            hits, qv = self._dense(query, candidate_k)
            timing.query_vector = qv
            for rank, (row, score) in enumerate(hits):
                cand = slot(row)
                cand.dense_rank, cand.dense_score = rank, score
                cand.fused_score += self.dense_weight / (self.rrf_k + rank + 1)
            timing.dense_ms = (time.perf_counter() - t0) * 1000

        if use_bm25 and self.bm25 is not None:
            t0 = time.perf_counter()
            for rank, (row, score) in enumerate(self.bm25.search(query,
                                                                candidate_k)):
                cand = slot(row)
                cand.bm25_rank, cand.bm25_score = rank, score
                cand.fused_score += self.bm25_weight / (self.rrf_k + rank + 1)
            timing.bm25_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ordered = sorted(fused.values(), key=lambda c: -c.fused_score)

        # Collapse to passages, keeping the best-ranked chunk of each.
        seen, out = set(), []
        for cand in ordered:
            key = (cand.query_id, cand.passage_index)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
            if len(out) >= top_k:
                break
        timing.fusion_ms = (time.perf_counter() - t0) * 1000
        return out, timing
