"""
POST /api/text/query

Content-negotiated: `Accept: text/event-stream` gets SSE, anything else
(including `application/json` or no header) gets a single JSON body. One
route, no duplicate `/stream` endpoint.

Streaming honesty note
-----------------------
`RAGPipeline.run()` is synchronous end-to-end -- retrieval, guardrails and
generation all complete before it returns. Wiring true mid-generation token
streaming through the full guarded pipeline was deliberately deferred (it
would mean either duplicating the guardrail chain here, which breaks the
"thin adapter" property, or modifying RAGPipeline itself, which is out of
scope for this task). So today the SSE path runs the pipeline once, then
emits the completed, already-validated answer as word-chunked SSE frames.
That gives the frontend a stable streaming CONTRACT to build against now,
without pretending a latency characteristic that has not been measured for
the guarded path. See docs/api-v1.md.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.v1.schemas import TextQueryRequest, TextQueryResponse
from app.api.v1.translate import to_text_response
from app.schemas import RAGRequest

router = APIRouter(tags=["text"])


def _pipeline():
    from app.main import STATE
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"index not loaded: {STATE.get('error') or 'unknown'}")
    return pipeline


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_body(payload: TextQueryResponse):
    yield _sse_frame("meta", {
        "language": payload.language,
        "guardrail": payload.guardrail.model_dump(),
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
        "latency": payload.latency.model_dump(),
        "session_id": payload.session_id,
    })


@router.post("/text/query", response_model=TextQueryResponse)
def text_query(body: TextQueryRequest, request: Request):
    pipeline = _pipeline()
    internal = pipeline.run(RAGRequest(
        text=body.query, language=body.language, want_audio=False))
    payload = to_text_response(internal, body.session_id)

    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return StreamingResponse(_stream_body(payload),
                                 media_type="text/event-stream")
    return payload
