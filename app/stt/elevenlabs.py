"""ElevenLabs Scribe speech-to-text adapter for the existing STT contract."""

from __future__ import annotations

import base64
import io
import time

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      stop_after_delay, wait_exponential)

from app.stt.sarvam import STTError, STTTransient, Transcription

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsSTT:
    def __init__(self, api_key: str, model: str = "scribe_v2",
                 timeout_s: float = 10.0, max_attempts: int = 3,
                 retry_deadline_s: float = 15.0,
                 client: httpx.Client | None = None):
        self.api_key = api_key
        self.model = model
        self.max_attempts = max_attempts
        self.retry_deadline_s = retry_deadline_s
        self.offline = not bool(api_key)
        self._client = client or (None if self.offline else httpx.Client(
            timeout=httpx.Timeout(timeout_s)))

    def transcribe(self, audio_bytes: bytes, language: str = "auto",
                   filename: str = "audio.wav") -> Transcription:
        if self.offline:
            raise STTError("ElevenLabs STT selected but API key is missing")
        if not audio_bytes:
            raise STTError("audio is empty")
        started = time.perf_counter()
        payload = self._call(audio_bytes, language, filename)
        return Transcription(
            text=(payload.get("text") or "").strip(),
            language_code=payload.get("language_code"),
            language_probability=payload.get("language_probability"),
            latency_ms=(time.perf_counter() - started) * 1000,
            request_id=payload.get("request_id"))

    def transcribe_base64(self, audio_b64: str, **kwargs) -> Transcription:
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise STTError(f"audio_base64 is not valid base64: {exc}") from exc
        if not raw:
            raise STTError("audio_base64 decoded to zero bytes")
        return self.transcribe(raw, **kwargs)

    def _call(self, audio_bytes: bytes, language: str, filename: str) -> dict:
        @retry(
            stop=stop_after_attempt(self.max_attempts)
                 | stop_after_delay(self.retry_deadline_s),
            wait=wait_exponential(multiplier=0.25, max=2.0),
            retry=retry_if_exception_type(STTTransient),
            reraise=True)
        def attempt() -> dict:
            data = {"model_id": self.model}
            if language != "auto":
                if language not in ("en", "hi", "mr", "kok"):
                    raise STTError(f"unsupported language {language!r}")
                data["language_code"] = language
            files = {"file": (filename, io.BytesIO(audio_bytes),
                              "application/octet-stream")}
            try:
                response = self._client.post(
                    ELEVENLABS_STT_URL,
                    headers={"xi-api-key": self.api_key},
                    data=data, files=files)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise STTTransient(str(exc)) from exc
            if response.status_code == 429 or response.status_code >= 500:
                raise STTTransient(
                    f"HTTP {response.status_code}: {response.text[:200]}")
            if response.status_code >= 400:
                raise STTError(
                    f"HTTP {response.status_code}: {response.text[:200]}")
            return response.json()
        return attempt()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
