"""
Sarvam text-to-speech.

API contract (verified against docs.sarvam.ai):
    POST https://api.sarvam.ai/text-to-speech
    header  api-subscription-key: <key>
    body    {text, language_code, speaker, model}
    returns {request_id, audios: [<base64 WAV>, ...]}

Language support, checked rather than assumed
---------------------------------------------
bulbul:v3 accepts: bn, en, gu, hi, kn, ml, mr, od, pa, ta, te (all -IN).
en-IN, hi-IN and mr-IN are natively supported, so none of those voices is an
approximation.

Konkani (kok-IN) is NOT in the enum. It IS in saaras's STT enum, so Konkani is
a product language we can hear but not speak. `synthesize()` raises for it
rather than substituting a Marathi voice -- the pipeline treats TTS failure as
non-fatal and returns the text answer, which is the honest degradation.

Text handling
-------------
* bulbul:v3 caps input at 2,500 chars. Mando's answers are 2-4 sentences, but
  we truncate at a sentence boundary rather than mid-word if one ever runs long.
* Sarvam's docs note that numbers above 4 digits are pronounced correctly only
  when comma-grouped, so we insert separators before synthesis. This is a
  presentation-layer fix and never touches the text used for grounding.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from app.schemas import SARVAM_TTS_LANG

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
DEFAULT_MODEL = "bulbul:v3"
MAX_CHARS = 2500

# bulbul:v3 speakers. Gender is not documented per-voice for v3, so these are
# configurable rather than hard-coded assumptions.
DEFAULT_SPEAKERS = {"male": "aditya", "female": "ritu"}


class TTSError(RuntimeError):
    pass


class TTSTransient(TTSError):
    """Retryable: timeout, 429, 5xx."""


@dataclass
class Speech:
    audio_base64: str
    latency_ms: float
    speaker: str
    language_code: str
    chars: int


def _group_digits(text: str) -> str:
    """10000 -> 10,000 so Sarvam pronounces long numbers correctly."""
    def repl(m: re.Match) -> str:
        return f"{int(m.group(0)):,}"
    return re.sub(r"(?<![\d,.])\d{5,}(?![\d,.])", repl, text)


def _truncate(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("। ", ". ", "! ", "? "):
        idx = cut.rfind(sep)
        if idx > limit * 0.6:
            return cut[:idx + 1]
    return cut.rsplit(" ", 1)[0]


class SarvamTTS:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 timeout_s: float = 10.0, max_attempts: int = 3,
                 speakers: dict | None = None):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model = model
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.speakers = {**DEFAULT_SPEAKERS, **(speakers or {})}
        self.available = bool(self.api_key)
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s)) \
            if self.available else None

    def synthesize(self, text: str, lang: str, voice: str = "male") -> Speech:
        if not self.available:
            raise TTSError("no SARVAM_API_KEY configured")

        # SARVAM_TTS_LANG, not SARVAM_LANG: bulbul:v3's language enum is a
        # strict subset of saaras's. Konkani is a product language we can
        # TRANSCRIBE but cannot SPEAK, so the lookup misses and we raise
        # instead of quietly substituting a Marathi voice and calling it
        # Konkani.
        code = SARVAM_TTS_LANG.get(lang)
        if not code:
            raise TTSError(
                f"no Sarvam TTS voice for {lang!r}; bulbul:v3 supports "
                f"{sorted(SARVAM_TTS_LANG)}. Konkani is transcribable but not "
                f"speakable -- return the text answer instead.")

        speaker = self.speakers.get(voice, self.speakers["male"])
        payload_text = _truncate(_group_digits(text.strip()))
        if not payload_text:
            raise TTSError("nothing to synthesize")

        @retry(stop=stop_after_attempt(self.max_attempts),
               wait=wait_exponential(multiplier=0.25, max=2.0),
               retry=retry_if_exception_type(TTSTransient), reraise=True)
        def _attempt() -> dict:
            try:
                resp = self._client.post(
                    SARVAM_TTS_URL,
                    headers={"api-subscription-key": self.api_key,
                             "Content-Type": "application/json"},
                    json={"text": payload_text, "language_code": code,
                          "speaker": speaker, "model": self.model})
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise TTSTransient(str(exc)) from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                raise TTSTransient(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                raise TTSError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()

        t0 = time.perf_counter()
        data = _attempt()
        ms = (time.perf_counter() - t0) * 1000

        audios = data.get("audios") or []
        if not audios:
            raise TTSError(f"no audio returned: {str(data)[:200]}")

        return Speech(audio_base64=audios[0], latency_ms=ms, speaker=speaker,
                      language_code=code, chars=len(payload_text))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
