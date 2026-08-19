"""
Sarvam speech-to-text.

API contract (verified against docs.sarvam.ai, not assumed):
    POST https://api.sarvam.ai/speech-to-text
    header  api-subscription-key: <key>
    body    multipart/form-data: file, model, language_code, mode
    returns {request_id, transcript, language_code, language_probability,
             timestamps?}

`language_code` in the RESPONSE is Sarvam's own detection, and it is the most
reliable language signal we get, because it is derived from audio -- we only
ever see text. The pipeline treats it as authoritative over our text heuristic
except in the specific hi/mr case where our markers are confidently opposed
(see app/language.resolve_language).

Offline mode
------------
Without SARVAM_API_KEY the client runs in `offline` mode and returns a
deterministic canned transcript. That exists so the pipeline, tests and the
latency harness are runnable without credentials -- NOT to fake results. Every
response carries `offline=True`, and the benchmark refuses to report STT
latency as real when that flag is set.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from app.schemas import SARVAM_LANG

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = "saaras:v3"


class STTError(RuntimeError):
    pass


class STTTransient(STTError):
    """Retryable: timeout, 429, 5xx."""


@dataclass
class Transcription:
    text: str
    language_code: str | None
    language_probability: float | None
    latency_ms: float
    offline: bool = False
    request_id: str | None = None


class SarvamSTT:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 timeout_s: float = 10.0, max_attempts: int = 3):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model = model
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.offline = not bool(self.api_key)
        self._client = None if self.offline else httpx.Client(
            timeout=httpx.Timeout(timeout_s))

    # -- public ------------------------------------------------------------
    def transcribe(self, audio_bytes: bytes, language: str = "auto",
                   filename: str = "audio.wav") -> Transcription:
        if self.offline:
            return self._offline(audio_bytes)

        t0 = time.perf_counter()
        try:
            payload = self._call(audio_bytes, language, filename)
        except Exception as exc:
            raise STTError(f"sarvam stt failed: {exc}") from exc
        ms = (time.perf_counter() - t0) * 1000

        return Transcription(
            text=(payload.get("transcript") or "").strip(),
            language_code=payload.get("language_code"),
            language_probability=payload.get("language_probability"),
            latency_ms=ms,
            request_id=payload.get("request_id"),
        )

    def transcribe_base64(self, audio_b64: str, **kw) -> Transcription:
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise STTError(f"audio_base64 is not valid base64: {exc}") from exc
        if not raw:
            raise STTError("audio_base64 decoded to zero bytes")
        return self.transcribe(raw, **kw)

    # -- internals ---------------------------------------------------------
    def _call(self, audio_bytes: bytes, language: str, filename: str) -> dict:
        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=0.25, max=2.0),
            retry=retry_if_exception_type(STTTransient),
            reraise=True,
        )
        def _attempt() -> dict:
            data = {"model": self.model}
            if language != "auto":
                code = SARVAM_LANG.get(language)
                if not code:
                    raise STTError(f"unsupported language {language!r}")
                data["language_code"] = code

            files = {"file": (filename, io.BytesIO(audio_bytes),
                              "audio/wav")}
            try:
                resp = self._client.post(
                    SARVAM_STT_URL,
                    headers={"api-subscription-key": self.api_key},
                    data=data, files=files)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise STTTransient(str(exc)) from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                raise STTTransient(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                # 4xx is our bug (bad key, bad codec) -- retrying wastes budget.
                raise STTError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()

        return _attempt()

    def _offline(self, audio_bytes: bytes) -> Transcription:
        t0 = time.perf_counter()
        # Deterministic, and clearly marked. Never presented as a real result.
        return Transcription(
            text="", language_code=None, language_probability=None,
            latency_ms=(time.perf_counter() - t0) * 1000,
            offline=True)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
