r"""
Lexical (BM25) retrieval.

Tokenisation across en/hi/mr -- and why NOT `re.\w`
---------------------------------------------------
The obvious `re.compile(r"\w+", re.UNICODE)` is BROKEN for Devanagari, and
silently so. Python's `re` excludes Unicode category Mn/Mc (combining marks)
from `\w`, and Devanagari carries its vowels as exactly those marks. Measured:

    re  : "कॉर्पोरेशन" -> ['क', 'र', 'प', 'र', 'शन']      # shattered
    this: "कॉर्पोरेशन" -> ['कॉर्पोरेशन']                    # correct

Every matra and the virama become token boundaries, so Hindi/Marathi BM25
degrades to near-random while English BM25 looks fine -- a failure mode that
shows up as "hybrid retrieval just doesn't help much for Indic", not as an
error. We therefore use the `regex` module with explicit Unicode properties:
letters, numbers and marks are word characters; the danda (U+0964) and double
danda (U+0965) are punctuation and correctly fall out as separators.

We deliberately do NOT stem. There is no reliable light stemmer for Marathi in
the dependency set, and stemming English but not Marathi would bias the
cross-language comparison that this whole project is measuring.
"""

from __future__ import annotations

import regex

from rank_bm25 import BM25Okapi

# \p{L} letters, \p{N} numbers, \p{M} combining marks (the critical one).
_TOKEN = regex.compile(r"[\p{L}\p{N}\p{M}]+")

# Latin stopwords only. Applying an English stoplist to Devanagari would be a
# no-op; applying a hand-rolled Hindi list but not a Marathi one would skew the
# comparison, so we leave Indic text unfiltered in both.
_EN_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will",
    "with", "what", "which", "who", "how", "when", "where", "why",
}


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    if drop_stopwords:
        kept = [t for t in toks if t not in _EN_STOP]
        # Never return empty for an all-stopword query -- fall back to raw.
        return kept or toks
    return toks


class BM25Index:
    """BM25 over chunk texts, returning (row_index, score) pairs."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.corpus_tokens = [tokenize(t) for t in texts]
        # rank_bm25 divides by average doc length; guard the empty-corpus case.
        if not any(self.corpus_tokens):
            raise ValueError("BM25 corpus is empty after tokenisation")
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b)

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        import numpy as np

        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        n = min(top_k, len(scores))
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0.0]
