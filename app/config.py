"""Central configuration. Everything overridable by env; no secrets in code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")


@dataclass
class Config:
    # -- data / index ------------------------------------------------------
    processed_dir: str = os.environ.get("PROCESSED_DIR", "data/processed")
    index_dir: str = os.environ.get("INDEX_DIR", "data/index")
    qdrant_url: str | None = os.environ.get("QDRANT_URL") or None
    qdrant_path: str = os.environ.get("QDRANT_PATH", "data/qdrant")

    # -- models ------------------------------------------------------------
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "e5-small")
    chunk_strategy: str = os.environ.get("CHUNK_STRATEGY", "fixed")
    reranker: str = os.environ.get("RERANKER", "mmr")   # mmr|cross-encoder|none

    # -- retrieval ---------------------------------------------------------
    candidate_k: int = _i("CANDIDATE_K", 50)
    top_k: int = _i("TOP_K", 5)
    rrf_k: int = _i("RRF_K", 60)
    # DEFAULT OFF, on evidence. Equal-weight RRF fusion with BM25 measurably
    # HURT retrieval in all three languages (en MRR 0.654 -> 0.564), and a
    # weight sweep was monotonic: every reduction in BM25's weight improved
    # MRR, with dense-only best. BM25 stays implemented and switchable because
    # it is the right choice on corpora with real lexical overlap -- this one
    # is natural-language questions against paraphrased web text, where dense
    # retrieval handles the paraphrase and lexical matching adds only noise.
    # See EXPERIMENTS.md E8.
    use_bm25: bool = _b("USE_BM25", False)
    use_dense: bool = _b("USE_DENSE", True)
    cross_lingual: bool = _b("CROSS_LINGUAL", True)

    # -- stage timeouts (ms). Budget-driven, enforced by the harness. ------
    timeout_stt_ms: float = _f("TIMEOUT_STT_MS", 10_000)
    timeout_retrieval_ms: float = _f("TIMEOUT_RETRIEVAL_MS", 2_000)
    timeout_generation_ms: float = _f("TIMEOUT_GENERATION_MS", 15_000)
    timeout_tts_ms: float = _f("TIMEOUT_TTS_MS", 10_000)

    # -- guardrails --------------------------------------------------------
    grounding_min_semantic: float = _f("GROUNDING_MIN_SEMANTIC", 0.82)
    evidence_min_top: float = _f("EVIDENCE_MIN_TOP", 0.87)
    evidence_min_mean: float = _f("EVIDENCE_MIN_MEAN", 0.85)
    enforce_grounding: bool = _b("ENFORCE_GROUNDING", True)

    # -- credentials (never defaulted to a literal) ------------------------
    sarvam_api_key: str | None = field(
        default_factory=lambda: os.environ.get("SARVAM_API_KEY"))
    llm_api_key: str | None = field(
        default_factory=lambda: os.environ.get("LLM_API_KEY")
        or os.environ.get("SARVAM_API_KEY"))
    llm_model: str = os.environ.get("LLM_MODEL", "sarvam-105b-conversations")
    llm_base_url: str = os.environ.get("LLM_BASE_URL",
                                       "https://api.sarvam.ai/v1")

    # General-knowledge fallback, independent from the primary provider.
    external_llm_api_key: str | None = field(
        default_factory=lambda: os.environ.get("EXTERNAL_LLM_API_KEY"))
    external_llm_base_url: str | None = field(
        default_factory=lambda: os.environ.get("EXTERNAL_LLM_BASE_URL"))
    external_llm_model: str | None = field(
        default_factory=lambda: os.environ.get("EXTERNAL_LLM_MODEL"))
    external_llm_provider: str | None = field(
        default_factory=lambda: os.environ.get("EXTERNAL_LLM_PROVIDER"))

    stt_provider: str = os.environ.get("STT_PROVIDER", "sarvam").lower()
    elevenlabs_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY"))
    elevenlabs_stt_model: str = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v2")

    # -- voice -------------------------------------------------------------
    tts_model: str = os.environ.get("TTS_MODEL", "bulbul:v3")
    tts_speaker_male: str = os.environ.get("TTS_SPEAKER_MALE", "aditya")
    tts_speaker_female: str = os.environ.get("TTS_SPEAKER_FEMALE", "ritu")

    @property
    def has_stt(self) -> bool:
        if self.stt_provider == "elevenlabs":
            return bool(self.elevenlabs_api_key)
        return self.stt_provider == "sarvam" and bool(self.sarvam_api_key)

    @property
    def has_sarvam_tts(self) -> bool:
        return bool(self.sarvam_api_key)

    @property
    def has_external_llm(self) -> bool:
        return all((self.external_llm_api_key, self.external_llm_base_url,
                    self.external_llm_model))

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key)

    def missing_credentials(self) -> list[str]:
        missing = []
        if self.stt_provider not in ("sarvam", "elevenlabs"):
            missing.append("STT_PROVIDER (must be sarvam or elevenlabs)")
        elif self.stt_provider == "elevenlabs" and not self.elevenlabs_api_key:
            missing.append("ELEVENLABS_API_KEY (speech-to-text)")
        elif self.stt_provider == "sarvam" and not self.sarvam_api_key:
            missing.append("SARVAM_API_KEY (speech-to-text and text-to-speech)")
        if not self.llm_api_key:
            missing.append("LLM_API_KEY (answer generation)")
        return missing


CONFIG = Config()
