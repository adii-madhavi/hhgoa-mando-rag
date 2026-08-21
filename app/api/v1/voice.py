"""
POST /api/voice/query

multipart/form-data, not base64-in-JSON: the frontend will record microphone
audio as a Blob/File and upload it directly. Internally this still becomes
`RAGRequest.audio_base64` (app/schemas.py) -- that internal contract is
unchanged; base64 encoding happens here, once, as an adapter detail the
frontend never sees.

Same content negotiation as /api/text/query: `Accept: text/event-stream` for
SSE, otherwise a single JSON body. See text.py for the streaming-honesty note;
it applies identically here.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.v1.schemas import (PUBLIC_ANSWER_MODES, PUBLIC_LANGUAGES,
                                PUBLIC_VOICES,
                                VoiceQueryResponse)
from app.api.v1.text import _sse_frame                        # reuse framing
from app.api.v1.translate import to_voice_response
from app.schemas import RAGRequest
from app.telemetry import RUNTIME_LATENCIES

router = APIRouter(tags=["voice"])


def _pipeline():
    from app.main import STATE
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"index not loaded: {STATE.get('error') or 'unknown'}")
    return pipeline


def _stream_body(payload: VoiceQueryResponse):
    yield _sse_frame("meta", {
        "transcript": payload.transcript,
        "language": payload.language,
        "guardrail": payload.guardrail.model_dump(),
        "answer_mode": payload.answer_mode,
        "answer_origin": payload.answer_origin,
        "external_verified": payload.external_verified,
        "language_match": payload.language_match,
    })
    if payload.guardrail.refused or not payload.answer:
        yield _sse_frame("done", {"answer": payload.answer,
                                  "latency": payload.latency.model_dump()})
        return
    for word in payload.answer.split(" "):
        yield _sse_frame("chunk", {"text": word + " "})
    yield _sse_frame("done", {
        "answer": payload.answer,
        "sources": [s.model_dump() for s in payload.sources],
        "audio_base64": payload.audio_base64,
        "latency": payload.latency.model_dump(),
        "session_id": payload.session_id,
        "answer_mode": payload.answer_mode,
        "answer_origin": payload.answer_origin,
        "external_verified": payload.external_verified,
        "language_match": payload.language_match,
    })


@router.post("/voice/query", response_model=VoiceQueryResponse)
async def voice_query(
    request: Request,
    audio: UploadFile,
    language: str = Form(default="auto"),
    voice: str = Form(default="male"),
    session_id: str | None = Form(default=None),
    answer_mode: str = Form(default="detailed"),
    want_audio: bool = Form(default=False),
):
    if language != "auto" and language not in PUBLIC_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"language must be 'auto' or one of {PUBLIC_LANGUAGES}, "
                   f"got {language!r}")
    if voice not in PUBLIC_VOICES:
        raise HTTPException(
            status_code=422,
            detail=f"voice must be one of {PUBLIC_VOICES}, got {voice!r}")
    if answer_mode not in PUBLIC_ANSWER_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"answer_mode must be one of {PUBLIC_ANSWER_MODES}, "
                   f"got {answer_mode!r}")

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=422, detail="empty audio upload")

    pipeline = _pipeline()
    internal = pipeline.run(RAGRequest(
        audio_base64=base64.b64encode(raw).decode("ascii"),
        language=language, voice=voice, want_audio=want_audio,
        answer_mode=answer_mode))
    if internal.ok and not internal.refused and internal.answer:
        runtime = RUNTIME_LATENCIES.record(
            "voice", answer_mode, internal.latency.end_to_end_ms)
    else:
        runtime = RUNTIME_LATENCIES.snapshot("voice", answer_mode)
    payload = to_voice_response(internal, session_id, runtime=runtime)

    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return StreamingResponse(_stream_body(payload),
                                 media_type="text/event-stream")
    return payload
