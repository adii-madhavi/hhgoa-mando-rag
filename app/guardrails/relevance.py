"""
Retrieval guardrail: is there enough evidence to answer at all?

This is the guardrail that actually prevents hallucination, and it is the one
we can calibrate honestly, because MSMARCO-XI hands us 780 queries (39% of the
corpus) whose gold label is literally "No Answer Present". So the threshold is
fitted to labelled data rather than picked because it felt about right.

Signals
-------
top_score      the best dense similarity. A query about something outside the
               corpus scores uniformly low against everything.
margin         top_score - second_score. A confident retrieval separates its
               winner from the pack; a diffuse one does not.
mean_top_k     average of the top-k, which resists a single lucky chunk.

We combine them rather than thresholding top_score alone: cosine similarity
from a bi-encoder is poorly calibrated in absolute terms (multilingual e5 puts
almost everything in a narrow band), so the SHAPE of the score distribution
carries more information than any single value.

The operating point is chosen in evaluation/threshold_calibration.py by
maximising balanced accuracy over answerable vs no-answer queries, and the
resulting numbers -- including the false-refusal rate we accept -- are reported
in EXPERIMENTS.md rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvidenceDecision:
    sufficient: bool
    confidence: float
    top_score: float
    margin: float
    mean_top_k: float
    detail: str = ""


# Defaults are placeholders until calibration runs; they are overwritten from
# experiments/threshold_calibration.json at startup when that file exists.
DEFAULT_THRESHOLDS = {
    "min_top_score": 0.80,
    "min_mean_top_k": 0.78,
    "min_candidates": 1,
    "k": 3,
}


def assess_evidence(candidates, thresholds: dict | None = None
                    ) -> EvidenceDecision:
    """
    `candidates` are post-rerank Candidate objects. We prefer rerank_score when
    present, else dense_score, else fused_score -- whichever reflects the final
    ordering the generator will see.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    k = th["k"]

    if not candidates or len(candidates) < th["min_candidates"]:
        return EvidenceDecision(False, 0.0, 0.0, 0.0, 0.0,
                                "no candidates retrieved")

    # SCALE DISCIPLINE: the threshold is expressed in COSINE-SIMILARITY units,
    # so only cosine-valued fields may feed it. rerank_score (MMR relevance)
    # and dense_score are both query-passage cosines and are interchangeable.
    # fused_score is NOT -- it is an RRF value of order 1/(60+rank) ~ 0.016,
    # and mixing it in would make any BM25-only candidate look like near-zero
    # similarity and silently drag mean_top_k below the threshold, refusing
    # good answers whenever lexical retrieval surfaced a passage dense missed.
    def score_of(c) -> float | None:
        for attr in ("rerank_score", "dense_score"):
            v = getattr(c, attr, None)
            if v is not None:
                return float(v)
        return None

    scores = sorted((s for s in map(score_of, candidates) if s is not None),
                    reverse=True)
    if not scores:
        # Every candidate came from BM25 alone and nothing rescored them, so we
        # have no calibrated signal. Refusing is the safe default.
        return EvidenceDecision(False, 0.0, 0.0, 0.0, 0.0,
                                "no cosine-scored candidates to threshold on")
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0
    margin = top - second
    mean_top_k = sum(scores[:k]) / min(k, len(scores))

    sufficient = top >= th["min_top_score"] and mean_top_k >= th["min_mean_top_k"]

    # Confidence is reported for the debug panel; it is a monotone squash of
    # the top score relative to the threshold, not a probability.
    span = max(1.0 - th["min_top_score"], 1e-6)
    confidence = max(0.0, min(1.0, (top - th["min_top_score"]) / span))

    detail = ("sufficient" if sufficient else
              f"top={top:.3f}<{th['min_top_score']} or "
              f"mean@{k}={mean_top_k:.3f}<{th['min_mean_top_k']}")
    return EvidenceDecision(sufficient, round(confidence, 3), round(top, 4),
                            round(margin, 4), round(mean_top_k, 4), detail)


def load_calibrated(path: str = "experiments/threshold_calibration.json"
                    ) -> dict:
    """Load the fitted operating point, falling back to defaults."""
    import json
    import os

    if not os.path.exists(path):
        return dict(DEFAULT_THRESHOLDS)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {**DEFAULT_THRESHOLDS, **data.get("chosen", {})}
