"""
Inverted-index BM25.

Why replace rank_bm25
---------------------
`rank_bm25.BM25Okapi.get_scores` allocates and fills a score array over the
ENTIRE corpus for every query term, in Python. Measured on this index
(72,411 chunks, evaluation/latency.py, 360 queries):

    rank_bm25          P50  95.16 ms   P100 345.44 ms
    this implementation  -- see EXPERIMENTS.md E11 --

It was the single largest stage in the runtime path, larger than dense
retrieval and every guardrail combined.

The fix is the standard one: an inverted index. A query touches only the
documents that actually contain one of its terms, which for a 3-6 term query
over 72k chunks is a small fraction of the corpus. We accumulate scores into a
dense buffer indexed by document, but only over posting lists, so cost scales
with postings-hit rather than corpus size.

Scoring is textbook BM25 with the same k1/b defaults as rank_bm25, and the
implementation is verified against it in tests/ so the swap is a pure speed
change, not a silent quality change.
"""

from __future__ import annotations

import math

import numpy as np

from app.retrieval.bm25 import tokenize


class FastBM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.k1 = k1
        self.b = b

        n_docs = len(texts)
        self.n_docs = n_docs
        doc_len = np.zeros(n_docs, dtype=np.float32)

        # term -> {doc_id: term_frequency}
        postings: dict[str, dict[int, int]] = {}
        for doc_id, text in enumerate(texts):
            toks = tokenize(text)
            doc_len[doc_id] = len(toks)
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for term, tf in counts.items():
                postings.setdefault(term, {})[doc_id] = tf

        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if n_docs else 0.0

        # Freeze postings into contiguous arrays, and precompute the per-term
        # IDF and the length-normalisation denominator, which do not depend on
        # the query and so should never be recomputed per request.
        self.idf: dict[str, float] = {}
        self.doc_ids: dict[str, np.ndarray] = {}
        self.tf_weight: dict[str, np.ndarray] = {}

        denom_norm = k1 * (1 - b + b * (doc_len / max(self.avgdl, 1e-9)))
        for term, plist in postings.items():
            ids = np.fromiter(plist.keys(), dtype=np.int32, count=len(plist))
            tfs = np.fromiter(plist.values(), dtype=np.float32, count=len(plist))
            df = len(plist)
            # Okapi IDF, matching rank_bm25's BM25Okapi formulation.
            self.idf[term] = math.log(n_docs - df + 0.5) - math.log(df + 0.5)
            self.doc_ids[term] = ids
            # tf * (k1 + 1) / (tf + k1*(1-b+b*dl/avgdl)) -- query independent.
            self.tf_weight[term] = (tfs * (k1 + 1)) / (tfs + denom_norm[ids])

        self._scores = np.zeros(n_docs, dtype=np.float32)   # reused buffer

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []

        scores = self._scores
        scores.fill(0.0)
        touched: list[np.ndarray] = []

        for term in set(toks):
            ids = self.doc_ids.get(term)
            if ids is None:
                continue                      # term absent from the corpus
            idf = self.idf[term]
            if idf <= 0:
                # Appears in nearly every document; Okapi IDF goes negative and
                # would penalise documents for containing the term.
                continue
            np.add.at(scores, ids, idf * self.tf_weight[term])
            touched.append(ids)

        if not touched:
            return []

        candidates = np.unique(np.concatenate(touched))
        cand_scores = scores[candidates]
        n = min(top_k, candidates.size)
        part = np.argpartition(-cand_scores, n - 1)[:n]
        part = part[np.argsort(-cand_scores[part])]
        return [(int(candidates[i]), float(cand_scores[i])) for i in part
                if cand_scores[i] > 0.0]
