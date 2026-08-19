"""
Chunking strategies for MANDO.

Honest framing
--------------
MSMARCO-XI evidence units are short web snippets (~300-600 chars, typically a
single paragraph), NOT long documents. Strategies designed for book-length text
have little to bite on at passage level, so we anchor the hierarchy one level
UP, using structure the dataset genuinely has:

    query_id group  (all 10 retrieved passages for one query -- the "document")
        -> passage        (one web snippet)
            -> sentence-window
                -> sentence

Every strategy below emits ChunkS with the same schema so the retrieval
benchmark can swap strategies without touching the index code.

Strategies
----------
fixed        char-budget baseline with overlap        (control, not a proposal)
sentence     split on sentence boundaries, pack to a budget
window       sentence-window: embed 1 sentence, return +/-k neighbours as context
semantic     split where adjacent-sentence embedding similarity drops
hierarchical parent=passage, child=sentence-pack; retrieve child, serve parent
adaptive     pick per passage based on length / sentence count / list-iness

All strategies are metadata-aware: every chunk carries query_id, passage_index,
lang, is_selected, parent_id and chunk_type so the evaluator can score a chunk
hit against the passage-level qrels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

# Sentence terminators across our three languages. Devanagari uses the danda
# (U+0964) and double danda (U+0965); Hindi/Marathi text from this dataset also
# uses ASCII '.' heavily because it is machine-translated from English.
_SENT_END = re.compile(r"(?<=[.!?।॥])\s+")
_ABBREV = re.compile(r"\b(?:[A-Z]|Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|e\.g|i\.e)\.$")
_LISTY = re.compile(r"(?:^|\s)(?:\d+\.|[-*•])\s")


@dataclass
class Chunk:
    chunk_id: str
    text: str                 # what gets embedded
    context: str              # what gets shown to the LLM (>= text)
    chunk_type: str
    strategy: str
    query_id: int
    passage_index: int
    lang: str
    is_selected: bool
    parent_id: str
    position: int = 0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    """Multilingual sentence split for en/hi/mr. No extra dependency."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts, buf = [], ""
    for piece in _SENT_END.split(text):
        buf = f"{buf} {piece}".strip() if buf else piece
        # Don't break on "Dr." / "U.S." style abbreviations.
        if _ABBREV.search(buf):
            continue
        parts.append(buf)
        buf = ""
    if buf:
        parts.append(buf)
    return [p for p in parts if p.strip()]


def _pack(sentences: list[str], budget: int) -> list[list[str]]:
    """Greedily pack sentences into groups of <= budget chars (>=1 sentence)."""
    groups, cur, size = [], [], 0
    for s in sentences:
        if cur and size + len(s) + 1 > budget:
            groups.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += len(s) + 1
    if cur:
        groups.append(cur)
    return groups


def _mk(strategy: str, chunk_type: str, rec: dict, position: int,
        text: str, context: str | None = None, **meta) -> Chunk:
    qid, pi, lang = rec["query_id"], rec["passage_index"], rec["lang"]
    parent_id = f"{qid}:{pi}:{lang}"
    return Chunk(
        chunk_id=f"{parent_id}:{strategy}:{position}",
        text=text,
        context=context if context is not None else text,
        chunk_type=chunk_type,
        strategy=strategy,
        query_id=qid,
        passage_index=pi,
        lang=lang,
        is_selected=bool(rec.get("is_selected")),
        parent_id=parent_id,
        position=position,
        meta=meta,
    )


# --------------------------------------------------------------------------
# A. Fixed-size baseline (control)
# --------------------------------------------------------------------------
def chunk_fixed(rec: dict, size: int = 400, overlap: int = 64) -> list[Chunk]:
    text = rec["text"].strip()
    step = max(size - overlap, 1)
    out, pos = [], 0
    for start in range(0, max(len(text), 1), step):
        piece = text[start:start + size].strip()
        if not piece:
            continue
        out.append(_mk("fixed", "fixed_span", rec, pos, piece,
                       char_start=start, size=size, overlap=overlap))
        pos += 1
        if start + size >= len(text):
            break
    return out


# --------------------------------------------------------------------------
# B. Sentence-based
# --------------------------------------------------------------------------
def chunk_sentence(rec: dict, budget: int = 350) -> list[Chunk]:
    sents = split_sentences(rec["text"])
    out = []
    for pos, group in enumerate(_pack(sents, budget)):
        out.append(_mk("sentence", "sentence_pack", rec, pos, " ".join(group),
                       n_sentences=len(group)))
    return out


# --------------------------------------------------------------------------
# C. Sentence-window
# --------------------------------------------------------------------------
def chunk_window(rec: dict, window: int = 1) -> list[Chunk]:
    """Embed ONE sentence (precise match) but serve +/-window as context."""
    sents = split_sentences(rec["text"])
    out = []
    for i, s in enumerate(sents):
        lo, hi = max(0, i - window), min(len(sents), i + window + 1)
        out.append(_mk("window", "sentence_window", rec, i, s,
                       context=" ".join(sents[lo:hi]),
                       window=window, n_sentences=len(sents)))
    return out


# --------------------------------------------------------------------------
# D. Semantic
# --------------------------------------------------------------------------
def chunk_semantic(rec: dict, encode: Callable[[list[str]], "object"],
                   percentile: float = 25.0, budget: int = 500) -> list[Chunk]:
    """
    Split where consecutive-sentence similarity dips.

    `encode` takes a list of sentences and returns L2-normalised vectors
    (numpy array). The breakpoint threshold is the `percentile`-th percentile of
    the adjacent-similarity distribution WITHIN THIS PASSAGE, so it adapts to
    passages that are uniformly on-topic instead of forcing a fixed cut count.
    """
    import numpy as np

    sents = split_sentences(rec["text"])
    if len(sents) < 3:
        return chunk_sentence(rec, budget)

    vecs = np.asarray(encode(sents), dtype="float32")
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
    threshold = float(np.percentile(sims, percentile))

    groups, cur, size = [], [sents[0]], len(sents[0])
    for i, s in enumerate(sents[1:]):
        too_big = size + len(s) > budget
        topic_shift = sims[i] <= threshold
        if cur and (topic_shift or too_big):
            groups.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += len(s) + 1
    if cur:
        groups.append(cur)

    return [_mk("semantic", "semantic_span", rec, pos, " ".join(g),
                n_sentences=len(g), threshold=round(threshold, 4))
            for pos, g in enumerate(groups)]


# --------------------------------------------------------------------------
# E. Hierarchical
# --------------------------------------------------------------------------
def chunk_hierarchical(rec: dict, child_budget: int = 220) -> list[Chunk]:
    """
    Parent = the whole passage, child = a small sentence pack.

    Children are what we embed and retrieve; `context` carries the full parent
    passage so generation never sees a decapitated sentence. The parent chunk
    itself is also emitted (chunk_type=parent) so a coarse match can still hit.
    """
    sents = split_sentences(rec["text"])
    out = [_mk("hierarchical", "parent", rec, 0, rec["text"].strip(),
               level=0, n_sentences=len(sents))]
    for pos, group in enumerate(_pack(sents, child_budget)):
        out.append(_mk("hierarchical", "child", rec, pos + 1, " ".join(group),
                       context=rec["text"].strip(),
                       level=1, n_sentences=len(group)))
    return out


# --------------------------------------------------------------------------
# F. Adaptive
# --------------------------------------------------------------------------
def chunk_adaptive(rec: dict, encode: Callable | None = None) -> list[Chunk]:
    """
    Route each passage to the strategy its shape justifies:

      very short (< 300 chars or <= 2 sentences) -> keep whole; splitting a
          2-sentence snippet only destroys context
      list-like (bullets / numbered items)       -> sentence packs, since list
          items are semantically independent and windows would blur them
      long and prose-like (>= 6 sentences)       -> semantic if an encoder is
          available, else sentence-window
      everything else                            -> sentence-window
    """
    text = rec["text"].strip()
    sents = split_sentences(text)

    if len(text) < 300 or len(sents) <= 2:
        chunks = [_mk("adaptive", "whole_passage", rec, 0, text,
                      route="short", n_sentences=len(sents))]
    elif len(_LISTY.findall(text)) >= 3:
        chunks = chunk_sentence(rec, budget=250)
        for c in chunks:
            c.meta["route"] = "listy"
    elif len(sents) >= 6 and encode is not None:
        chunks = chunk_semantic(rec, encode)
        for c in chunks:
            c.meta["route"] = "long_prose_semantic"
    else:
        chunks = chunk_window(rec, window=1)
        for c in chunks:
            c.meta["route"] = "windowed"

    for c in chunks:
        c.strategy = "adaptive"
        c.chunk_id = f"{c.parent_id}:adaptive:{c.position}"
    return chunks


STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "window": chunk_window,
    "semantic": chunk_semantic,
    "hierarchical": chunk_hierarchical,
    "adaptive": chunk_adaptive,
}


def chunk_records(records: Iterable[dict], strategy: str,
                  encode: Callable | None = None, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    needs_encode = strategy in ("semantic", "adaptive")
    out: list[Chunk] = []
    for rec in records:
        if not rec.get("text", "").strip():
            continue
        if needs_encode:
            if strategy == "semantic" and encode is None:
                raise ValueError("semantic chunking requires an encoder")
            out.extend(fn(rec, encode, **kw) if strategy == "semantic"
                       else fn(rec, encode=encode, **kw))
        else:
            out.extend(fn(rec, **kw))
    return out
