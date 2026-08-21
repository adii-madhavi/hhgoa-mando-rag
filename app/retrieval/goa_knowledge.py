"""Small product-only Goa evidence index, isolated from MSMARCO-XI metrics."""

from __future__ import annotations

import json
import regex as re
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.hybrid import Candidate

_STOP = {"a", "an", "and", "are", "about", "is", "of", "the", "to", "what", "which", "who", "in", "tell", "me"}


@dataclass
class GoaRetrieval:
    candidates: list[Candidate]
    sufficient: bool
    strong_match: bool


class GoaKnowledgeBase:
    def __init__(self, embedder, path: str | Path | None = None):
        self.embedder = embedder
        source = Path(path) if path else Path(__file__).parents[2] / "data" / "goa_knowledge.json"
        with source.open(encoding="utf-8") as handle:
            self.passages = json.load(handle)
        self.matrix = embedder.encode_passages([p["text"] for p in self.passages])

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[\wÀ-ÿ]+", text.casefold()) if len(t) > 1 and t not in _STOP}

    def retrieve(self, query: str, query_vector=None, top_k: int = 3) -> GoaRetrieval:
        qv = query_vector
        if qv is None:
            qv = self.embedder.encode_queries([query], batch_size=1)[0]
        dense = self.matrix @ qv
        query_tokens = self._tokens(query)
        scored = []
        for index, passage in enumerate(self.passages):
            haystack = " ".join([passage["title"], passage["text"], *passage.get("aliases", [])])
            overlap = len(query_tokens & self._tokens(haystack)) / max(1, len(query_tokens))
            scored.append((index, float(dense[index]), overlap))
        scored.sort(key=lambda item: (item[2] >= 0.5, item[2], item[1]), reverse=True)

        candidates = []
        for index, cosine, overlap in scored[:top_k]:
            passage = self.passages[index]
            candidates.append(Candidate(
                doc_id=f"goa:{passage['id']}", text=passage["text"],
                context=passage["text"], query_id=-1, passage_index=index,
                lang="en", dense_score=cosine,
                meta={"knowledge_origin": "goa_knowledge",
                      "source_name": passage["source_name"],
                      "source_url": passage["source_url"],
                      "title": passage["title"], "lexical_overlap": overlap}))

        best_dense = scored[0][1] if scored else 0.0
        best_overlap = scored[0][2] if scored else 0.0
        sufficient = best_overlap >= 0.5 or best_dense >= 0.82
        return GoaRetrieval(candidates if sufficient else [], sufficient,
                            best_overlap >= 0.75)
