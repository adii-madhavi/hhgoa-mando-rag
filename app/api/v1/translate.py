"""
Translation from internal RAGResponse to the public v1 schema.

This is the ONLY place that reads from app.schemas.RAGResponse for the public
API. Every other v1 module works exclusively with app.api.v1.schemas types.
That single choke point is what makes "internal fields never leak" a property
you can test, rather than a convention every route has to remember to follow.

Deliberately NOT copied to the public response: judge_*, retrieval_confidence,
grounding_score (only the boolean grounded/refused survive), generator class
name, persona name, latency.stages (per-stage internal timing), warnings,
sources_used/invalid_sources (citation numbers -- internal bookkeeping),
request_id (replaced by the caller's own session_id, since the two are
unrelated concepts), rewritten_query, language_confidence, judge_cache_hit,
judge_async_dispatched.
"""

from __future__ import annotations

from app.api.v1.schemas import (GuardrailStatus, LatencyMeta, SourceItem,
                                TextQueryResponse, VoiceQueryResponse)
from app.schemas import RAGResponse


def _sources(resp: RAGResponse, limit: int = 5) -> list[SourceItem]:
    out = []
    for e in resp.evidence[:limit]:
        score = e.rerank_score if e.rerank_score is not None else (
            e.dense_score if e.dense_score is not None else 0.0)
        # Clamp: cosine/rerank scores are not guaranteed to sit in [0, 1]
        # (E10/E15 measured e5 cosine as high as ~0.94), and the public field
        # is documented as a 0-1 relevance number.
        out.append(SourceItem(text=e.text, language=e.lang,
                              relevance=round(max(0.0, min(1.0, score)), 4)))
    return out


def _guardrail(resp: RAGResponse) -> GuardrailStatus:
    return GuardrailStatus(
        grounded=resp.grounded,
        refused=resp.refused,
        refusal_reason=resp.refusal_reason.value if resp.refusal_reason
        else None,
    )


def _latency(resp: RAGResponse, streamed: bool = False) -> LatencyMeta:
    flat = resp.latency.flat()
    generation_ms = flat.get("generation_ms")
    return LatencyMeta(
        retrieval_ms=round(resp.latency.total_rag_ms - (generation_ms or 0), 2),
        generation_ms=(round(generation_ms, 2) if generation_ms else None),
        total_ms=round(resp.latency.total_rag_ms, 2),
        streamed=streamed,
    )


def to_text_response(resp: RAGResponse, session_id: str | None,
                     streamed: bool = False) -> TextQueryResponse:
    return TextQueryResponse(
        answer=resp.answer or "",
        sources=_sources(resp),
        language=resp.language.value if resp.language else "en",
        latency=_latency(resp, streamed=streamed),
        guardrail=_guardrail(resp),
        session_id=session_id,
    )


def to_voice_response(resp: RAGResponse, session_id: str | None,
                      streamed: bool = False) -> VoiceQueryResponse:
    base = to_text_response(resp, session_id, streamed=streamed)
    return VoiceQueryResponse(
        **base.model_dump(),
        transcript=resp.transcript or "",
        audio_base64=resp.audio_base64,
        audio_stream_url=None,   # not yet implemented; see schemas.py docstring
        audio_unavailable_reason=resp.audio_unavailable_reason,
    )
