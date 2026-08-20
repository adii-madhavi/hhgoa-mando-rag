"""
SMALL live Sarvam smoke test. Two real API calls, nothing else.

Run this manually, once, after adding a real SARVAM_API_KEY to .env:

    python evaluation/sarvam_live_check.py

What it does
------------
1. SarvamTTS.synthesize() a short fixed English phrase -> real audio bytes.
   Proves: TTS auth works, the request shape is accepted, bulbul:v3 returns
   audio.
2. Feeds THAT audio into SarvamSTT.transcribe() -> a transcript.
   Proves: STT auth works, the request shape is accepted, saaras returns text.

This exercises both real Sarvam endpoints without needing a microphone
recording or the full RAG index loaded, which is why it is "small" -- it is
deliberately NOT the full voice pipeline benchmark (that is Phase 10,
evaluation/streaming_latency.py's voice-loop successor, not built yet and not
in scope here). It proves the credential and the two HTTP contracts work; the
mocked suite in tests/test_voice.py already proves the pipeline wiring around
them.

Secret hygiene
--------------
This script NEVER logs, prints, or includes the API key value in any output.
Only non-secret fields are printed: transcript text, audio byte length,
latency, and HTTP status on failure (never headers).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG                    # noqa: E402
from app.stt.sarvam import SarvamSTT, STTError    # noqa: E402
from app.tts.sarvam_tts import SarvamTTS, TTSError  # noqa: E402

PHRASE = "Eagles fly thirty to fifty five miles per hour."


def main() -> None:
    if not CONFIG.sarvam_api_key:
        raise SystemExit(
            "\nNo SARVAM_API_KEY configured in .env.\n"
            "This script will not run without real credentials, and it will "
            "not fabricate a result. Add SARVAM_API_KEY to .env and re-run:\n"
            "  python evaluation/sarvam_live_check.py\n")

    print("=" * 60)
    print("SARVAM LIVE CHECK (2 real API calls)")
    print("=" * 60)

    tts = SarvamTTS(api_key=CONFIG.sarvam_api_key)
    stt = SarvamSTT(api_key=CONFIG.sarvam_api_key)

    print(f"\n[1/2] TTS: synthesizing {len(PHRASE)} chars of English...")
    try:
        speech = tts.synthesize(PHRASE, "en", voice="male")
    except TTSError as exc:
        # str(exc) may include an HTTP status/body from Sarvam, but NEVER the
        # key -- the key is never interpolated into any exception message in
        # app/tts/sarvam_tts.py (verified: it only appears as a request
        # header, never in an f-string).
        print(f"  FAILED: {exc}")
        raise SystemExit(1) from exc

    import base64
    audio_bytes = base64.b64decode(speech.audio_base64)
    print(f"  OK: {len(audio_bytes)} bytes of audio, "
          f"speaker={speech.speaker}, latency={speech.latency_ms:.0f}ms")

    print("\n[2/2] STT: transcribing that audio back...")
    try:
        result = stt.transcribe(audio_bytes, language="en")
    except STTError as exc:
        print(f"  FAILED: {exc}")
        raise SystemExit(1) from exc

    print(f"  OK: transcript={result.text!r}")
    print(f"      language_code={result.language_code} "
          f"confidence={result.language_probability} "
          f"latency={result.latency_ms:.0f}ms")

    print("\n" + "=" * 60)
    print("BOTH ENDPOINTS REACHABLE AND AUTHENTICATED.")
    print("=" * 60)

    tts.close()
    stt.close()


if __name__ == "__main__":
    main()
