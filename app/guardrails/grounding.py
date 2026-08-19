"""
Grounding guardrail: is the generated answer actually supported by the evidence?

Runs AFTER generation. If it fails, the answer is discarded and replaced by a
refusal -- the persona never gets to talk its way past this check.

Why not "ask an LLM if the answer is grounded"
----------------------------------------------
An LLM judge is a second generation call: it adds hundreds of ms to a 200 ms
budget, and it fails in a correlated way with the generator (both are fluent-
text models, and the same fluent-but-unsupported sentence that fooled one tends
to fool the other). So the primary verifier is cheap and mechanical, and an LLM
verifier is available as an optional second opinion for offline evaluation.

Two independent checks
----------------------
1. NUMERIC GROUNDING (hard, high-precision). Every number in the answer must
   appear in the retrieved context. This targets the single most damaging RAG
   failure -- a fluent answer with an invented figure, date, or dose. Numbers
   are the one class of claim where exact matching is both possible and
   meaningful, so this check is a hard gate, not a score.

2. SEMANTIC GROUNDING (soft). Each answer sentence is embedded and compared to
   the evidence; the answer's score is the mean over sentences of each
   sentence's best match. Mean rather than min, because a well-formed answer
   legitimately contains connective sentences ("Here's what I found") that
   match nothing in particular.

Both must pass. The thresholds are calibrated in evaluation/, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import regex

# Matches integers, decimals, percentages and comma-grouped numbers.
_NUMBER = regex.compile(r"\d+(?:[.,]\d+)*")
# Devanagari digits, which Sarvam and Indic LLMs both emit.
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# CALIBRATION WARNING -- do not lower min_semantic on intuition.
# multilingual-e5 does not use the full [-1, 1] cosine range; it compresses
# everything into a narrow high band. Measured against a single corporation
# passage:
#     grounded paraphrase           0.939
#     grounded single sentence      0.827
#     COMPLETELY UNRELATED text     0.678   <- "best beaches in Goa are Baga"
# So the floor for unrelated text is ~0.68, and the initial 0.60 default was
# vacuous: it passed everything. Anything below ~0.70 is not a threshold, it is
# a no-op. 0.82 is the provisional operating point pending
# evaluation/threshold_calibration.py, which fits it on labelled data.
DEFAULTS = {
    "min_semantic": 0.82,
    "require_numeric": True,
    "max_unsupported_numbers": 0,
}

# Numbers that carry no factual claim on their own and would cause noisy
# failures if treated as evidence-bearing.
_TRIVIAL = {"0", "1", "2", "100"}


@dataclass
class GroundingResult:
    passed: bool
    semantic_score: float
    unsupported_numbers: list[str] = field(default_factory=list)
    per_sentence: list[float] = field(default_factory=list)
    detail: str = ""

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _normalise_numbers(text: str) -> set[str]:
    text = text.translate(_DEVA_DIGITS)
    out = set()
    for raw in _NUMBER.findall(text):
        # "1,234.5" and "1234.5" must compare equal; trailing zeros too.
        cleaned = raw.replace(",", "")
        out.add(cleaned)
        if "." in cleaned:
            out.add(cleaned.rstrip("0").rstrip("."))
    return {n for n in out if n and n not in _TRIVIAL}


def verify_grounding(answer: str, evidence_texts: list[str], embedder,
                     thresholds: dict | None = None,
                     evidence_vectors=None) -> GroundingResult:
    """
    `evidence_vectors` lets the caller pass embeddings the index already holds.
    Encoding costs ~23 ms/text on CPU, and without this the same five evidence
    passages are encoded by retrieval, again by the reranker, and again here.
    """
    th = {**DEFAULTS, **(thresholds or {})}
    answer = (answer or "").strip()

    if not answer:
        return GroundingResult(False, 0.0, detail="empty answer")
    if not evidence_texts:
        return GroundingResult(False, 0.0, detail="no evidence to check against")

    # --- check 1: numeric grounding (hard gate) ---------------------------
    context_numbers = _normalise_numbers(" ".join(evidence_texts))
    answer_numbers = _normalise_numbers(answer)
    unsupported = sorted(answer_numbers - context_numbers)

    # --- check 2: semantic grounding (soft) -------------------------------
    from ingestion.chunk import split_sentences

    sentences = split_sentences(answer) or [answer]
    sent_vecs = embedder.encode_passages(sentences,
                                         batch_size=max(len(sentences), 1))
    if evidence_vectors is not None:
        ev_vecs = evidence_vectors
    else:
        ev_vecs = embedder.encode_passages(
            evidence_texts, batch_size=max(len(evidence_texts), 1))
    sims = sent_vecs @ ev_vecs.T                 # both L2-normalised
    per_sentence = sims.max(axis=1)
    semantic = float(np.mean(per_sentence))

    numeric_ok = (not th["require_numeric"]
                  or len(unsupported) <= th["max_unsupported_numbers"])
    semantic_ok = semantic >= th["min_semantic"]
    passed = numeric_ok and semantic_ok

    detail = "grounded"
    if not numeric_ok:
        detail = f"unsupported numbers in answer: {unsupported}"
    elif not semantic_ok:
        detail = (f"semantic grounding {semantic:.3f} below "
                  f"{th['min_semantic']}")

    return GroundingResult(
        passed=passed,
        semantic_score=round(semantic, 4),
        unsupported_numbers=unsupported,
        per_sentence=[round(float(x), 4) for x in per_sentence],
        detail=detail,
    )
