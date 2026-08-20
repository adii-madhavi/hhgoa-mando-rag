"""
Structured I/O for the MANDO pipeline.

Every stage has an explicit contract. Nothing in the runtime path passes raw
dicts around, because the harness requirements (retries, timeouts, validation,
graceful failure, stage-level latency) all depend on knowing exactly what a
stage returned and whether it is usable.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Two different notions of "supported", kept apart on purpose.
#
# PRODUCT_LANGS  -- a user may speak/select this language and the system will
#                   try to serve them.
# EVALUATED_LANGS -- we hold LABELLED BENCHMARK DATA for this language, so
#                   every retrieval / answerability / calibration number in
#                   EXPERIMENTS.md is grounded in measurement.
#
# Konkani is a product language but NOT an evaluated one: MSMARCO-XI ships zero
# Konkani rows under any code (kok/kon/gom/knn), so there is nothing to compute
# Recall@k or a false-answer rate against. Any Konkani quality claim must come
# from the SEPARATE curated product evaluation set (evaluation/konkani_product_eval.py)
# and must never be merged into MSMARCO-XI benchmark metrics.
EVALUATED_LANGS = ("en", "hi", "mr")
PRODUCT_LANGS = ("en", "hi", "mr", "kok")
SUPPORTED_LANGS = PRODUCT_LANGS          # back-compat for existing callers

# Per-language capability, verified against Sarvam's published enums rather
# than assumed. Nothing in the UI or API may claim a capability that is False
# here.
LANG_CAPABILITIES = {
    "en":  {"stt": True, "tts": True,  "corpus": True,  "benchmark": True},
    "hi":  {"stt": True, "tts": True,  "corpus": True,  "benchmark": True},
    "mr":  {"stt": True, "tts": True,  "corpus": True,  "benchmark": True},
    # Konkani: Sarvam saaras accepts kok-IN, but bulbul:v3 does NOT, and the
    # corpus has no Konkani passages -- a Konkani query retrieves against
    # en/hi/mr evidence cross-lingually, which measured poorly for mr->en
    # (EXPERIMENTS.md E9). Both limits are surfaced, never papered over.
    "kok": {"stt": True, "tts": False, "corpus": False, "benchmark": False},
}

# Our internal code -> Sarvam BCP-47. STT codes; see SARVAM_TTS_LANG for the
# narrower set bulbul actually accepts.
SARVAM_LANG = {"en": "en-IN", "hi": "hi-IN", "mr": "mr-IN", "kok": "kok-IN"}

# TTS is a strict subset. Konkani is absent from bulbul:v3's enum, so it has no
# entry -- a lookup miss is the intended failure, not a silent fallback to
# another language's voice.
SARVAM_TTS_LANG = {"en": "en-IN", "hi": "hi-IN", "mr": "mr-IN"}


def has_capability(lang: str, capability: str) -> bool:
    return LANG_CAPABILITIES.get(lang, {}).get(capability, False)


class Lang(str, Enum):
    en = "en"
    hi = "hi"
    mr = "mr"
    kok = "kok"


class Stage(str, Enum):
    validate = "validate"
    stt = "stt"
    language_detection = "language_detection"
    input_guardrail = "input_guardrail"
    query_rewrite = "query_rewrite"
    dense_retrieval = "dense_retrieval"
    bm25_retrieval = "bm25_retrieval"
    fusion = "fusion"
    rerank = "rerank"
    evidence_check = "evidence_check"
    answerability = "answerability"
    generation = "generation"
    grounding = "grounding"
    tts = "tts"


class RefusalReason(str, Enum):
    empty_input = "empty_input"
    unintelligible = "unintelligible"
    unsafe = "unsafe"
    prompt_injection = "prompt_injection"
    off_topic = "off_topic"
    insufficient_evidence = "insufficient_evidence"
    ungrounded_answer = "ungrounded_answer"
    internal_error = "internal_error"


class Turn(BaseModel):
    """One exchange of session memory. Short-term only, never persisted."""
    query: str
    answer: str
    lang: Lang


class RAGRequest(BaseModel):
    audio_base64: str | None = None
    text: str | None = None            # bypasses STT; used by benchmarks/tests
    language: str = "auto"
    character: str = "mando"
    voice: str = "male"
    want_audio: bool = True
    context: list[Turn] = Field(default_factory=list, max_length=8)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])

    @field_validator("language")
    @classmethod
    def _lang_ok(cls, v: str) -> str:
        if v != "auto" and v not in SUPPORTED_LANGS:
            raise ValueError(f"language must be 'auto' or one of "
                             f"{SUPPORTED_LANGS}, got {v!r}")
        return v

    def has_input(self) -> bool:
        return bool((self.text or "").strip()) or bool(self.audio_base64)


class Evidence(BaseModel):
    doc_id: str
    text: str
    context: str
    lang: str
    query_id: int
    passage_index: int
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class StageTiming(BaseModel):
    stage: Stage
    ms: float
    ok: bool = True
    attempts: int = 1
    error: str | None = None


class Latency(BaseModel):
    """
    Stage-level latency. `total_rag_ms` covers transcript -> verified answer;
    `end_to_end_ms` additionally covers STT and TTS, i.e. the full voice loop.
    Offline chunking/embedding/indexing is NOT counted here -- it is not on the
    runtime path.
    """
    stages: list[StageTiming] = Field(default_factory=list)
    total_rag_ms: float = 0.0
    end_to_end_ms: float = 0.0

    def add(self, stage: Stage, ms: float, ok: bool = True,
            attempts: int = 1, error: str | None = None) -> None:
        self.stages.append(StageTiming(stage=stage, ms=round(ms, 2), ok=ok,
                                       attempts=attempts, error=error))

    def get(self, stage: Stage) -> float:
        return sum(s.ms for s in self.stages if s.stage == stage)

    def flat(self) -> dict:
        out = {f"{s.value}_ms": 0.0 for s in Stage}
        for s in self.stages:
            out[f"{s.stage.value}_ms"] += s.ms
        out["total_rag_ms"] = round(self.total_rag_ms, 2)
        out["end_to_end_ms"] = round(self.end_to_end_ms, 2)
        return out


class RAGResponse(BaseModel):
    request_id: str
    ok: bool
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    transcript: str | None = None
    rewritten_query: str | None = None
    language: Lang | None = None
    language_confidence: float | None = None
    answer: str | None = None
    audio_base64: str | None = None
    # Set when want_audio was requested but no audio was produced -- e.g. a
    # Konkani refusal/answer, where bulbul:v3 has no kok-IN voice and TTS
    # raises rather than substituting another language's voice.
    audio_unavailable_reason: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    grounded: bool | None = None
    grounding_score: float | None = None
    retrieval_confidence: float | None = None
    # Answerability judge (Phase 1). `judge_available=False` means the LLM
    # judge could not be consulted and the cosine guard decided alone.
    judge_available: bool | None = None
    judge_sufficient: bool | None = None
    judge_confidence: float | None = None
    judge_reason: str | None = None
    judge_supporting_ids: list[str] = Field(default_factory=list)
    # Async judge observability. `judge_async_dispatched` means a background
    # call was fired and this response did NOT wait for it -- so no judge cost
    # appears in total_rag_ms.
    judge_cache_hit: bool | None = None
    judge_async_dispatched: bool = False
    # Generation observability
    generator: str | None = None
    persona: str | None = None
    sources_used: list[int] = Field(default_factory=list)
    invalid_sources: list[int] = Field(default_factory=list)
    language_match: bool | None = None
    latency: Latency = Field(default_factory=Latency)
    warnings: list[str] = Field(default_factory=list)


class Timer:
    """Context manager that records a stage's wall time even when it raises."""

    def __init__(self, latency: Latency, stage: Stage):
        self.latency, self.stage = latency, stage
        self.attempts = 1

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        ms = (time.perf_counter() - self.t0) * 1000
        self.latency.add(self.stage, ms, ok=exc is None,
                         attempts=self.attempts,
                         error=None if exc is None else f"{exc_type.__name__}: {exc}")
        return False   # never swallow; the pipeline decides how to recover
