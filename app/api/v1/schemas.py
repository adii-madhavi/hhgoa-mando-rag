"""
PUBLIC API v1 schemas — the frozen boundary between MANDO's backend and any
frontend.

Nothing in here is the internal RAGRequest/RAGResponse (app/schemas.py).
These models are deliberately smaller: they name a stable public contract, and
they are the ONLY thing a separately-developed frontend should ever import
knowledge of. Internal fields (judge_*, retrieval_confidence, generator class
name, per-stage Latency.stages, API keys, model names) never appear here --
enforced by translate.py, which is the single place internal responses are
converted, and by tests/test_api_v1.py, which asserts no internal field name
leaks into a serialized public response.

Once published, DO NOT rename or remove a field here without bumping to v2 --
that is what "stable schema" means for a contract a frontend is coded against.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Mirrors app.schemas.PRODUCT_LANGS, restated here (not imported) so the
# public contract does not silently change shape if the internal language set
# is ever refactored. "auto" is a request-only value, not a language.
PUBLIC_LANGUAGES = ("en", "hi", "mr", "kok")
PUBLIC_VOICES = ("male", "female")


class SourceItem(BaseModel):
    """One piece of retrieved evidence, trimmed to what a frontend needs."""

    text: str = Field(description="Evidence snippet shown to the user.")
    language: str = Field(description="Language of this evidence passage.")
    relevance: float = Field(
        description="0-1 relevance score. Internal scoring method "
                    "(dense/rerank/fused) is not exposed.")


class LatencyMeta(BaseModel):
    """
    Coarse, honest latency reporting. Stage-by-stage internal timings
    (judge_ms, rerank_ms, etc.) are not public API -- see EXPERIMENTS.md if
    that detail is ever needed for debugging.
    """

    retrieval_ms: float = Field(description="Retrieval + guardrail time.")
    generation_ms: float | None = Field(
        default=None, description="LLM generation time, if generation ran.")
    total_ms: float = Field(description="retrieval_ms + generation_ms.")
    streamed: bool = Field(
        default=False, description="Whether the answer was delivered via SSE.")


class GuardrailStatus(BaseModel):
    """
    What safety/quality gate produced this outcome. Internal refusal reason
    codes are mapped to a small public enum, not passed through verbatim, so
    internal guardrail names can change without breaking the contract.
    """

    grounded: bool | None = Field(
        default=None,
        description="True if the answer was verified against evidence. Null "
                    "when the request was refused before generation.")
    refused: bool = Field(default=False)
    refusal_reason: str | None = Field(
        default=None,
        description="One of: empty_input, unintelligible, unsafe, "
                    "prompt_injection, insufficient_evidence, "
                    "ungrounded_answer, internal_error. Null if not refused.")


class TextQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    language: str = Field(
        default="auto",
        description="One of en, hi, mr, kok, or 'auto' to detect.")
    session_id: str | None = Field(
        default=None,
        description="Opaque client-supplied id. The API is STATELESS -- this "
                    "is not looked up server-side. The frontend owns "
                    "conversation history and must resend it via future "
                    "context support; it is accepted here for forward "
                    "compatibility only.")

    @field_validator("language")
    @classmethod
    def _lang_ok(cls, v: str) -> str:
        if v != "auto" and v not in PUBLIC_LANGUAGES:
            raise ValueError(
                f"language must be 'auto' or one of {PUBLIC_LANGUAGES}, "
                f"got {v!r}")
        return v


class TextQueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    language: str = Field(description="Language the answer was given in.")
    latency: LatencyMeta
    guardrail: GuardrailStatus
    session_id: str | None = None


class VoiceQueryResponse(TextQueryResponse):
    transcript: str = Field(description="What Sarvam STT heard.")
    audio_base64: str | None = Field(
        default=None,
        description="Base64 WAV of the spoken answer. Null if TTS is "
                    "unavailable (no credentials, unsupported language such "
                    "as Konkani, or synthesis failed) -- the frontend should "
                    "fall back to displaying `answer` as text in that case.")
    audio_stream_url: str | None = Field(
        default=None,
        description="Reserved for a future chunked-audio delivery path. "
                    "Always null today; Sarvam TTS streaming is not yet "
                    "integrated. Kept in the schema now so the frontend "
                    "contract does not need to change when it is.")
    audio_unavailable_reason: str | None = Field(
        default=None,
        description="Set when audio_base64 is null despite audio being "
                    "requested -- explains why (e.g. Konkani has no Sarvam "
                    "TTS voice) so the frontend can show that reason instead "
                    "of silently falling back to text.")


class ConfigResponse(BaseModel):
    supported_languages: list[str] = Field(
        description="Languages a user may speak/select "
                    "(app.schemas.PRODUCT_LANGS).")
    evaluated_languages: list[str] = Field(
        description="Subset with labelled benchmark data "
                    "(app.schemas.EVALUATED_LANGS). A language can be "
                    "supported without being evaluated -- see Konkani.")
    available_voices: list[str]
    streaming_supported: bool
    feature_flags: dict[str, bool] = Field(
        description="llm_generation, speech_to_text, text_to_speech, "
                    "answerability_judge -- capability flags only, never "
                    "model names or credentials.")


class HealthResponse(BaseModel):
    ok: bool
    index_loaded: bool
    llm_available: bool
    stt_available: bool
    tts_available: bool


class ErrorResponse(BaseModel):
    """Uniform error body. FastAPI's default validation error shape is used
    for 422s; this is used for the handled 4xx/5xx cases in v1 routes."""

    error: str
    detail: str | None = None
