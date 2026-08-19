"""
FastAPI surface for MANDO.

Endpoints
    GET  /health   readiness + which credentials are missing
    GET  /config   effective non-secret configuration
    POST /ask      the full pipeline (text or audio in, answer + audio out)
    POST /ask/text convenience: text-only, no TTS -- what the benchmarks use

The index loads once at startup, not per request. Loading it per request would
put a multi-second model load inside the latency budget and make every reported
number meaningless.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG                              # noqa: E402
from app.pipeline import RAGPipeline                       # noqa: E402
from app.retrieval.loader import load_index                # noqa: E402
from app.retrieval.reranker import (CrossEncoderReranker,  # noqa: E402
                                    MMRReranker, NoOpReranker)
from app.schemas import RAGRequest, RAGResponse            # noqa: E402
from app.stt.sarvam import SarvamSTT                       # noqa: E402
from app.tts.sarvam_tts import SarvamTTS                   # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("mando.api")

app = FastAPI(title="MANDO", version="0.1.0",
              description="Voice-enabled multilingual RAG (en/hi/mr)")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])

STATE: dict = {"pipeline": None, "manifest": None, "error": None}


def _build_reranker(embedder, dense_matrix=None):
    if CONFIG.reranker == "cross-encoder":
        return CrossEncoderReranker()
    if CONFIG.reranker == "none":
        return NoOpReranker()
    # Pass the index matrix so MMR looks vectors up instead of re-encoding.
    return MMRReranker(embedder, dense_matrix=dense_matrix)


@app.on_event("startup")
def startup() -> None:
    missing = CONFIG.missing_credentials()
    if missing:
        log.warning("starting with missing credentials: %s", missing)

    try:
        t0 = time.perf_counter()
        loaded = load_index(CONFIG.index_dir, rrf_k=CONFIG.rrf_k)
        stt = SarvamSTT(CONFIG.sarvam_api_key) if CONFIG.has_stt else None
        tts = SarvamTTS(CONFIG.sarvam_api_key) if CONFIG.has_stt else None
        STATE["pipeline"] = RAGPipeline(
            retriever=loaded.retriever, embedder=loaded.embedder,
            reranker=_build_reranker(loaded.embedder,
                                     loaded.retriever.matrix),
            config=CONFIG, stt=stt, tts=tts,
            dense_matrix=loaded.retriever.matrix)
        STATE["manifest"] = loaded.manifest
        log.info("index loaded: %d chunks in %.1fs (total startup %.1fs)",
                 len(loaded.chunks), loaded.load_seconds,
                 time.perf_counter() - t0)
    except Exception as exc:
        # Start anyway so /health can explain what is wrong, rather than the
        # container crash-looping with the reason buried in logs.
        STATE["error"] = str(exc)
        log.error("startup failed: %s", exc)


@app.get("/health")
def health() -> dict:
    return {
        "ok": STATE["pipeline"] is not None,
        "error": STATE["error"],
        "index": STATE["manifest"],
        "missing_credentials": CONFIG.missing_credentials(),
        "capabilities": {
            "speech_to_text": CONFIG.has_stt,
            "text_to_speech": CONFIG.has_stt,
            "llm_generation": CONFIG.has_llm,
            "extractive_fallback": True,
        },
        "languages": ["en", "hi", "mr"],
    }


@app.get("/config")
def config() -> dict:
    return {
        "embedding_model": CONFIG.embedding_model,
        "chunk_strategy": CONFIG.chunk_strategy,
        "reranker": CONFIG.reranker,
        "top_k": CONFIG.top_k, "candidate_k": CONFIG.candidate_k,
        "use_bm25": CONFIG.use_bm25, "use_dense": CONFIG.use_dense,
        "thresholds": {
            "evidence_min_top": CONFIG.evidence_min_top,
            "evidence_min_mean": CONFIG.evidence_min_mean,
            "grounding_min_semantic": CONFIG.grounding_min_semantic,
            "enforce_grounding": CONFIG.enforce_grounding,
        },
    }


def _run(request: RAGRequest) -> RAGResponse:
    pipeline = STATE["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"index not loaded: {STATE['error'] or 'unknown'}")
    return pipeline.run(request)


@app.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest) -> RAGResponse:
    return _run(request)


@app.post("/ask/text", response_model=RAGResponse)
def ask_text(request: RAGRequest) -> RAGResponse:
    request.want_audio = False
    return _run(request)


FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.get("/")
def index():
    """Serve the UI from the same origin, so the browser needs no CORS dance."""
    from fastapi.responses import FileResponse, HTMLResponse

    path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>MANDO</h1><p>frontend/index.html missing</p>",
                            status_code=404)
    return FileResponse(path)


@app.exception_handler(Exception)
def unhandled(request, exc):
    """No traceback ever reaches the client."""
    log.exception("unhandled error")
    return JSONResponse(status_code=500,
                        content={"ok": False, "error": "internal error"})
