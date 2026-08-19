"""GET /api/health, GET /api/config -- public-safe status and capabilities."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.schemas import ConfigResponse, HealthResponse
from app.schemas import EVALUATED_LANGS, PRODUCT_LANGS

router = APIRouter(tags=["meta"])


def _state():
    # Deferred import: app.main imports this router at module load time, so a
    # top-level `from app.main import STATE` here would be circular. By the
    # time a request actually arrives, app.main has finished importing.
    from app.main import STATE
    return STATE


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = _state()
    pipeline = state.get("pipeline")
    return HealthResponse(
        ok=pipeline is not None,
        index_loaded=pipeline is not None,
        llm_available=bool(pipeline and getattr(
            pipeline.generator, "available", False)),
        stt_available=bool(pipeline and pipeline.stt is not None
                           and not getattr(pipeline.stt, "offline", True)),
        tts_available=bool(pipeline and pipeline.tts is not None
                           and getattr(pipeline.tts, "available", False)),
    )


@router.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    state = _state()
    pipeline = state.get("pipeline")
    has_llm = bool(pipeline and getattr(pipeline.generator, "available",
                                        False))
    has_judge = bool(pipeline and pipeline.judge is not None)
    has_tts = bool(pipeline and pipeline.tts is not None
                   and getattr(pipeline.tts, "available", False))
    has_stt = bool(pipeline and pipeline.stt is not None
                   and not getattr(pipeline.stt, "offline", True))

    return ConfigResponse(
        supported_languages=list(PRODUCT_LANGS),
        evaluated_languages=list(EVALUATED_LANGS),
        # bulbul:v3 has no kok-IN (app.schemas.SARVAM_TTS_LANG); never claim a
        # voice for a language TTS cannot actually produce.
        available_voices=["male", "female"] if has_tts else [],
        streaming_supported=has_llm,
        feature_flags={
            "llm_generation": has_llm,
            "speech_to_text": has_stt,
            "text_to_speech": has_tts,
            "answerability_judge": has_judge,
        },
    )
