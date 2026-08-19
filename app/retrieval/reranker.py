"""
Reranking.

The CPU reality, stated up front
--------------------------------
A cross-encoder scores every (query, candidate) pair jointly, so cost is
linear in candidates and each pair is a full forward pass. This box encodes
~44 short texts/s with a 118M bi-encoder, so a 278M-568M multilingual
cross-encoder over 20 candidates lands in the high hundreds of milliseconds --
on its own, several times the 200 ms end-to-end target.

So we provide two rerankers and MEASURE both instead of assuming:

  cross-encoder : best quality, likely disqualified by latency on CPU
  mmr           : near-free; diversifies the fused list using embeddings we
                  have ALREADY computed, costing one small matrix op

MMR is not a quality substitute for a cross-encoder. It is a different job:
it removes near-duplicate evidence so the generator sees k distinct facts
rather than k paraphrases of one. On MSMARCO, where a query's 10 passages are
scraped from overlapping web pages, redundancy is the common failure and MMR
addresses it at a cost the budget can actually absorb.

Which one ships is decided by evaluation/, not by preference.
"""

from __future__ import annotations

import time

import numpy as np


class MMRReranker:
    """
    Maximal Marginal Relevance over already-computed chunk embeddings.

        score(d) = lambda * sim(q, d) - (1 - lambda) * max sim(d, selected)

    lambda=1.0 reduces to pure relevance; lower values buy diversity. Requires
    no model load and no extra forward passes.
    """

    name = "mmr"

    def __init__(self, embedder, lambda_: float = 0.7, dense_matrix=None):
        self.embedder = embedder
        self.lambda_ = lambda_
        # The index's dense matrix. When supplied, candidate vectors are looked
        # up instead of re-encoded -- measured to cut this stage from ~110 ms
        # to ~1 ms on CPU, since an encode costs ~23 ms per text here.
        self.dense_matrix = dense_matrix

    def _vectors(self, candidates):
        """Reuse indexed embeddings where possible; encode only the rest."""
        if self.dense_matrix is None:
            return self.embedder.encode_passages(
                [c.context for c in candidates],
                batch_size=len(candidates)), 0

        # CORRECTNESS GUARD: the matrix holds embeddings of chunk.TEXT, while
        # MMR scores chunk.CONTEXT. Identical for `fixed`/`sentence`; NOT for
        # hierarchical or sentence-window, where context is deliberately wider.
        # A cached vector is only valid when the two strings are the same.
        rows = [c.row if (c.row is not None and c.text == c.context) else None
                for c in candidates]
        if all(r is not None for r in rows):
            return self.dense_matrix[rows], 0

        # Mixed case: fill the gaps with a single batched encode.
        missing = [i for i, r in enumerate(rows) if r is None]
        mat = np.empty((len(candidates), self.dense_matrix.shape[1]),
                       dtype="float32")
        for i, r in enumerate(rows):
            if r is not None:
                mat[i] = self.dense_matrix[r]
        if missing:
            enc = self.embedder.encode_passages(
                [candidates[i].context for i in missing],
                batch_size=len(missing))
            for j, i in enumerate(missing):
                mat[i] = enc[j]
        return mat, len(missing)

    def rerank(self, query: str, candidates, top_k: int = 5,
               query_vector=None):
        t0 = time.perf_counter()
        if not candidates:
            return [], 0.0
        qv = (query_vector if query_vector is not None
              else self.embedder.encode_queries([query], batch_size=1)[0])
        mat, _ = self._vectors(candidates)
        rel = mat @ qv

        selected: list[int] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda i: rel[i])
            else:
                sims = mat[remaining] @ mat[selected].T   # (n_rem, n_sel)
                redundancy = sims.max(axis=1)
                mmr = self.lambda_ * rel[remaining] - (1 - self.lambda_) * redundancy
                best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)

        out = []
        for rank, i in enumerate(selected):
            c = candidates[i]
            c.rerank_score = float(rel[i])
            c.meta["rerank_rank"] = rank
            out.append(c)
        return out, (time.perf_counter() - t0) * 1000


class CrossEncoderReranker:
    """Joint (query, passage) scoring. Highest quality, highest cost."""

    name = "cross-encoder"

    def __init__(self, model_id: str = "jinaai/jina-reranker-v2-base-multilingual",
                 device: str = "cpu", max_length: int = 512,
                 trust_remote_code: bool = True):
        from sentence_transformers import CrossEncoder

        t0 = time.perf_counter()
        self.model = CrossEncoder(model_id, device=device,
                                  max_length=max_length,
                                  trust_remote_code=trust_remote_code)
        self.load_seconds = time.perf_counter() - t0
        self.model_id = model_id

    def rerank(self, query: str, candidates, top_k: int = 5,
               query_vector=None):
        t0 = time.perf_counter()
        if not candidates:
            return [], 0.0
        pairs = [(query, c.context) for c in candidates]
        scores = self.model.predict(pairs, batch_size=8,
                                    show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))[:top_k]
        out = []
        for rank, i in enumerate(order):
            c = candidates[int(i)]
            c.rerank_score = float(scores[int(i)])
            c.meta["rerank_rank"] = rank
            out.append(c)
        return out, (time.perf_counter() - t0) * 1000


class NoOpReranker:
    """Passthrough, for isolating the reranker's contribution in ablations."""

    name = "none"

    def rerank(self, query: str, candidates, top_k: int = 5,
               query_vector=None):
        return candidates[:top_k], 0.0
