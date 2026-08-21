"""
FastAPI surface for MANDO.

Two layers, deliberately kept separate:

  INTERNAL (used by evaluation/ scripts and this project's own benchmarks)
    GET  /health
    GET  /config
    POST /ask        full RAGResponse, all internal fields
    POST /ask/text   convenience: text-only, no TTS

  PUBLIC v1 (app/api/v1/) -- the contract a separately-developed frontend
  consumes. Frozen, documented in docs/api-v1.md, and never exposes internal
  pipeline fields, model names, or credentials.
    GET  /api/health
    GET  /api/config
    POST /api/text/query
    POST /api/voice/query

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
from app.guardrails.answerability import AnswerabilityJudge  # noqa: E402
from app.generation.external import ExternalLLMGenerator  # noqa: E402
from app.pipeline import RAGPipeline                       # noqa: E402
from app.retrieval.loader import load_index                # noqa: E402
from app.retrieval.goa_knowledge import GoaKnowledgeBase   # noqa: E402
from app.retrieval.reranker import (CrossEncoderReranker,  # noqa: E402
                                    MMRReranker, NoOpReranker)
from app.schemas import RAGRequest, RAGResponse            # noqa: E402
from app.stt.sarvam import SarvamSTT                       # noqa: E402
from app.stt.elevenlabs import ElevenLabsSTT               # noqa: E402
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

# Public v1 API -- the boundary a separately-developed frontend consumes.
# Deferred import: app.api.v1.* modules import STATE from this module
# lazily (inside request handlers), so this import is safe here even
# though it happens before STATE is populated by startup().
from app.api.v1.router import api_router                   # noqa: E402
app.include_router(api_router, prefix="/api")

STATE: dict = {"pipeline": None, "manifest": None, "error": None}


def _build_reranker(embedder, dense_matrix=None):
    if CONFIG.reranker == "cross-encoder":
        return CrossEncoderReranker()
    if CONFIG.reranker == "none":
        return NoOpReranker()
    # Pass the index matrix so MMR looks vectors up instead of re-encoding.
    return MMRReranker(embedder, dense_matrix=dense_matrix)


def _build_stt():
    if not CONFIG.has_stt:
        return None
    if CONFIG.stt_provider == "elevenlabs":
        return ElevenLabsSTT(
            CONFIG.elevenlabs_api_key, model=CONFIG.elevenlabs_stt_model,
            timeout_s=CONFIG.timeout_stt_ms / 1000)
    return SarvamSTT(
        CONFIG.sarvam_api_key, timeout_s=CONFIG.timeout_stt_ms / 1000)


@app.on_event("startup")
def startup() -> None:
    missing = CONFIG.missing_credentials()
    if missing:
        log.warning("starting with missing credentials: %s", missing)

    try:
        t0 = time.perf_counter()
        loaded = load_index(CONFIG.index_dir, rrf_k=CONFIG.rrf_k)
        stt = _build_stt()
        tts = (SarvamTTS(CONFIG.sarvam_api_key)
               if CONFIG.has_sarvam_tts else None)
        external = (ExternalLLMGenerator(
            api_key=CONFIG.external_llm_api_key,
            base_url=CONFIG.external_llm_base_url,
            model=CONFIG.external_llm_model,
            provider=CONFIG.external_llm_provider)
            if CONFIG.has_external_llm else None)
        # REGRESSION FIX: this constructor never passed judge= before, so the
        # deployed app silently ran on cosine-only guardrails -- the async
        # judge + verdict cache measured and decided as the production shape
        # (EXPERIMENTS.md E15/E16/E20) was built and benchmarked, but never
        # actually wired into app.main's own pipeline instance. Every eval
        # script (generation_latency.py, voice_e2e_check.py,
        # voice_loop_benchmark.py) constructs its OWN RAGPipeline with the
        # judge attached, which is why this gap went unnoticed: the numbers
        # in EXPERIMENTS.md are real, but they were never running here.
        judge = AnswerabilityJudge(timeout_s=8.0) if CONFIG.has_llm else None
        goa_knowledge = GoaKnowledgeBase(loaded.embedder)
        STATE["pipeline"] = RAGPipeline(
            retriever=loaded.retriever, embedder=loaded.embedder,
            reranker=_build_reranker(loaded.embedder,
                                     loaded.retriever.matrix),
            config=CONFIG, stt=stt, tts=tts,
            dense_matrix=loaded.retriever.matrix,
            judge=judge, judge_mode="async",
            index_version=str(loaded.manifest.get("n_chunks", "")),
            goa_knowledge=goa_knowledge,
            external_generator=external)
        STATE["manifest"] = loaded.manifest
        log.info("index loaded: %d chunks in %.1fs (total startup %.1fs)",
                 len(loaded.chunks), loaded.load_seconds,
                 time.perf_counter() - t0)
    except Exception as exc:
        # Start anyway so /health can explain what is wrong, rather than the
        # container crash-looping with the reason buried in logs.
        STATE["error"] = str(exc)
        log.error("startup failed: %s", exc)


@app.on_event("shutdown")
def shutdown() -> None:
    """Cleanly close the judge's background thread pool, now that
    startup() actually creates one (see the judge= fix above)."""
    pipeline = STATE.get("pipeline")
    if pipeline is not None:
        pipeline.shutdown(wait=False)


@app.get("/health")
def health() -> dict:
    return {
        "ok": STATE["pipeline"] is not None,
        "error": STATE["error"],
        "index": STATE["manifest"],
        "missing_credentials": CONFIG.missing_credentials(),
        "capabilities": {
            "speech_to_text": CONFIG.has_stt,
            "text_to_speech": CONFIG.has_sarvam_tts,
            "llm_generation": CONFIG.has_llm,
            "external_generation": CONFIG.has_external_llm,
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
