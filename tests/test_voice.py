"""
Voice pipeline tests: Sarvam STT / TTS clients and their wiring into
RAGPipeline. Everything here is MOCKED -- no network call, no SARVAM_API_KEY
required. The one real Sarvam smoke test lives outside pytest, in
evaluation/sarvam_live_check.py, and is run manually (see its own docstring).

Why this file exists
---------------------
app/stt/sarvam.py and app/tts/sarvam_tts.py were written and wired into
RAGPipeline in an earlier phase, contract-verified against docs.sarvam.ai, but
never had dedicated tests exercising the HTTP layer (request shape, retry
classification, error handling) or the full pipeline round trip
(audio in -> transcript -> guarded generation -> audio out). This file closes
that gap before any live call is made.

Nothing in this file ever asserts on, logs, or constructs a real credential.
Test doubles use the literal string "test-key", which is not a secret.
"""

from __future__ import annotations

import base64
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.stt.sarvam import STTError, STTTransient, SarvamSTT        # noqa: E402
from app.tts.sarvam_tts import (TTSError, TTSTransient, SarvamTTS,  # noqa: E402
                                _group_digits, _truncate)


def stt_with(handler, **kw):
    c = SarvamSTT(api_key="test-key", **kw)
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def tts_with(handler, **kw):
    c = SarvamTTS(api_key="test-key", **kw)
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


WAV_BYTES = b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x00" * 40  # shape only


# --------------------------------------------------------------------------
# SarvamSTT — offline mode (no key)
# --------------------------------------------------------------------------
class TestSTTOffline:
    def test_no_key_is_offline(self):
        c = SarvamSTT(api_key=None)
        assert c.offline is True

    def test_offline_transcription_is_marked(self):
        c = SarvamSTT(api_key=None)
        result = c.transcribe(WAV_BYTES)
        assert result.offline is True
        assert result.text == ""

    def test_offline_never_makes_a_request(self):
        """No httpx client should even be constructed without a key."""
        c = SarvamSTT(api_key=None)
        assert c._client is None


# --------------------------------------------------------------------------
# SarvamSTT — HTTP contract, via a mock transport
# --------------------------------------------------------------------------
class TestSTTRequestShape:
    def test_multipart_fields_present(self):
        captured = {}

        def handler(request):
            captured["content_type"] = request.headers.get("content-type", "")
            captured["auth_header"] = request.headers.get(
                "api-subscription-key")
            captured["body"] = request.content
            return httpx.Response(200, json={
                "request_id": "r1", "transcript": "how fast is an eagle",
                "language_code": "en-IN", "language_probability": 0.94})

        c = stt_with(handler)
        result = c.transcribe(WAV_BYTES, language="en")

        assert "multipart/form-data" in captured["content_type"]
        assert captured["auth_header"] == "test-key"
        assert b'name="model"' in captured["body"]
        assert b'name="file"' in captured["body"]
        assert b'name="language_code"' in captured["body"]
        assert b"en-IN" in captured["body"]
        assert result.text == "how fast is an eagle"
        assert result.language_code == "en-IN"

    def test_auto_language_omits_language_code_field(self):
        captured = {}

        def handler(request):
            captured["body"] = request.content
            return httpx.Response(200, json={
                "request_id": "r1", "transcript": "x",
                "language_code": "hi-IN", "language_probability": 0.8})

        stt_with(handler).transcribe(WAV_BYTES, language="auto")
        assert b'name="language_code"' not in captured["body"]

    def test_unsupported_language_raises_before_any_request(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json={})

        with pytest.raises(STTError):
            stt_with(handler).transcribe(WAV_BYTES, language="ta")
        assert calls["n"] == 0


class TestSTTRetryAndErrors:
    def test_429_retried_then_succeeds(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(429, text="slow down")
            return httpx.Response(200, json={
                "request_id": "r", "transcript": "ok",
                "language_code": "en-IN", "language_probability": 0.9})

        result = stt_with(handler).transcribe(WAV_BYTES)
        assert calls["n"] == 2 and result.text == "ok"

    def test_5xx_retried(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return (httpx.Response(503, text="down") if calls["n"] == 1 else
                    httpx.Response(200, json={
                        "request_id": "r", "transcript": "ok",
                        "language_code": "en-IN",
                        "language_probability": 0.9}))

        assert stt_with(handler).transcribe(WAV_BYTES).text == "ok"

    def test_4xx_fails_fast_without_retry(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(401, text="bad key")

        with pytest.raises(STTError):
            stt_with(handler).transcribe(WAV_BYTES)
        assert calls["n"] == 1, "4xx must not be retried"

    def test_exhausted_retries_raise_stt_error(self):
        def handler(request):
            return httpx.Response(503, text="down")

        with pytest.raises(STTError):
            stt_with(handler, max_attempts=2).transcribe(WAV_BYTES)


class TestTranscribeBase64:
    def test_valid_base64_round_trips(self):
        def handler(request):
            return httpx.Response(200, json={
                "request_id": "r", "transcript": "hello",
                "language_code": "en-IN", "language_probability": 0.9})

        c = stt_with(handler)
        b64 = base64.b64encode(WAV_BYTES).decode("ascii")
        assert c.transcribe_base64(b64).text == "hello"

    def test_invalid_base64_raises(self):
        c = stt_with(lambda r: httpx.Response(200, json={}))
        with pytest.raises(STTError):
            c.transcribe_base64("not-valid-base64!!!")

    def test_empty_decoded_bytes_raises(self):
        c = stt_with(lambda r: httpx.Response(200, json={}))
        with pytest.raises(STTError):
            c.transcribe_base64("")


# --------------------------------------------------------------------------
# SarvamTTS
# --------------------------------------------------------------------------
class TestTTSOffline:
    def test_no_key_is_unavailable(self):
        assert SarvamTTS(api_key=None).available is False

    def test_synthesize_without_key_raises(self):
        with pytest.raises(TTSError):
            SarvamTTS(api_key=None).synthesize("hello", "en")


class TestTTSLanguageSupport:
    def test_konkani_raises_rather_than_substituting(self):
        """
        REGRESSION-adjacent: bulbul:v3 has no kok-IN voice. Must raise, never
        silently fall back to the Marathi voice under a Konkani label.
        """
        c = tts_with(lambda r: httpx.Response(200, json={"audios": ["x"]}))
        with pytest.raises(TTSError) as exc:
            c.synthesize("test", "kok")
        assert "mr-in" not in str(exc.value).lower().split("supports")[0]

    @pytest.mark.parametrize("lang,code", [("en", "en-IN"), ("hi", "hi-IN"),
                                           ("mr", "mr-IN")])
    def test_evaluated_languages_map_correctly(self, lang, code):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"audios": ["YWJj"]})

        tts_with(handler).synthesize("hello", lang)
        assert captured["body"]["language_code"] == code


class TestTTSRequestShape:
    def test_json_body_and_auth_header(self):
        captured = {}

        def handler(request):
            captured["auth"] = request.headers.get("api-subscription-key")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"audios": ["YWJj"]})

        speech = tts_with(handler).synthesize("Eagles fly fast.", "en",
                                              voice="female")
        assert captured["auth"] == "test-key"
        assert captured["body"]["speaker"] == "ritu"   # default female voice
        assert speech.audio_base64 == "YWJj"
        assert speech.language_code == "en-IN"

    def test_male_voice_maps_to_default_speaker(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"audios": ["YWJj"]})

        tts_with(handler).synthesize("hi", "en", voice="male")
        assert captured["body"]["speaker"] == "aditya"

    def test_long_numbers_are_comma_grouped_before_send(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"audios": ["YWJj"]})

        tts_with(handler).synthesize("Revenue was 1500000 dollars.", "en")
        assert "1,500,000" in captured["body"]["text"]

    def test_empty_text_raises_without_a_request(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json={"audios": ["x"]})

        with pytest.raises(TTSError):
            tts_with(handler).synthesize("   ", "en")
        assert calls["n"] == 0


class TestTTSRetryAndErrors:
    def test_429_retried_then_succeeds(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(429, text="slow down")
            return httpx.Response(200, json={"audios": ["YWJj"]})

        assert tts_with(handler).synthesize("hi", "en").audio_base64 == "YWJj"
        assert calls["n"] == 2

    def test_4xx_fails_fast(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(403, text="forbidden")

        with pytest.raises(TTSError):
            tts_with(handler).synthesize("hi", "en")
        assert calls["n"] == 1

    def test_empty_audios_list_raises(self):
        def handler(request):
            return httpx.Response(200, json={"audios": []})

        with pytest.raises(TTSError):
            tts_with(handler).synthesize("hi", "en")


class TestTTSTextHelpers:
    def test_group_digits(self):
        assert _group_digits("1500000") == "1,500,000"
        assert _group_digits("2024") == "2024", "4-digit years untouched"

    def test_truncate_respects_limit(self):
        long_text = "word " * 3000
        out = _truncate(long_text, limit=2500)
        assert len(out) <= 2500


# --------------------------------------------------------------------------
# Full pipeline round trip: audio in -> transcript -> guarded generation ->
# audio out, with BOTH Sarvam clients mocked. This is the actual
# "Sarvam STT -> RAGPipeline -> guarded generation -> Sarvam TTS" wiring.
# --------------------------------------------------------------------------
class FakeSTT:
    """Scripted STT double matching SarvamSTT's public surface."""

    def __init__(self, text="how fast do eagles fly", language_code="en-IN",
                offline=False, raises=None):
        self.text, self.language_code = text, language_code
        self.offline = offline
        self.raises = raises
        self.calls = []

    def transcribe_base64(self, audio_b64, language="auto"):
        self.calls.append((audio_b64, language))
        if self.raises:
            raise self.raises
        from app.stt.sarvam import Transcription
        return Transcription(text=self.text, language_code=self.language_code,
                             language_probability=0.9, latency_ms=1.0,
                             offline=self.offline)


class FakeTTS:
    """Scripted TTS double matching SarvamTTS's public surface."""

    def __init__(self, audio_b64="ZmFrZS1hdWRpbw==", raises=None):
        self.audio_b64 = audio_b64
        self.raises = raises
        self.calls = []

    def synthesize(self, text, lang, voice="male"):
        self.calls.append((text, lang, voice))
        if self.raises:
            raise self.raises
        from app.tts.sarvam_tts import Speech
        return Speech(audio_base64=self.audio_b64, latency_ms=1.0,
                     speaker=voice, language_code=f"{lang}-IN",
                     chars=len(text))


@pytest.fixture(scope="module")
def voice_pipeline():
    """
    A real RAGPipeline with a real tiny in-memory index (mirrors
    test_pipeline.py's fixture), but SCRIPTED Sarvam clients -- proving the
    pipeline wiring without any network dependency.
    """
    from app.config import Config
    from app.generation.generator import ExtractiveGenerator
    from app.pipeline import RAGPipeline
    from app.retrieval.embeddings import Embedder
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.reranker import MMRReranker
    from ingestion.chunk import chunk_records

    emb = Embedder("e5-small")
    recs = [{"query_id": 1, "passage_index": 0, "lang": "en",
             "is_selected": True,
             "text": "Eagles fly 30 to 55 mph and dive at over 100 mph."}]
    chunks = chunk_records(recs, "fixed")
    vectors = emb.encode_passages([c.text for c in chunks], batch_size=8)
    retriever = HybridRetriever(chunks, vectors, None, emb)

    cfg = Config()
    cfg.top_k, cfg.candidate_k, cfg.use_bm25 = 3, 10, False
    cfg.evidence_min_top = cfg.evidence_min_mean = 0.70

    def build(stt=None, tts=None):
        return RAGPipeline(retriever=retriever, embedder=emb,
                           reranker=MMRReranker(emb, dense_matrix=vectors),
                           config=cfg, generator=ExtractiveGenerator(emb),
                           dense_matrix=vectors, stt=stt, tts=tts)

    return build


class TestVoicePipelineRoundTrip:
    def test_audio_in_transcript_and_audio_out(self, voice_pipeline):
        from app.schemas import RAGRequest

        stt, tts = FakeSTT(), FakeTTS()
        p = voice_pipeline(stt=stt, tts=tts)
        r = p.run(RAGRequest(audio_base64="ZmFrZS13YXY=", want_audio=True))

        assert not r.refused
        assert r.transcript == "how fast do eagles fly"
        assert r.audio_base64 == "ZmFrZS1hdWRpbw=="
        assert stt.calls and tts.calls, "both clients must have been used"

    def test_stt_stage_latency_recorded(self, voice_pipeline):
        from app.schemas import RAGRequest, Stage

        p = voice_pipeline(stt=FakeSTT(), tts=FakeTTS())
        r = p.run(RAGRequest(audio_base64="ZmFrZQ==", want_audio=True))
        stages = {s.stage for s in r.latency.stages}
        assert Stage.stt in stages and Stage.tts in stages

    def test_want_audio_false_skips_tts_entirely(self, voice_pipeline):
        from app.schemas import RAGRequest

        stt, tts = FakeSTT(), FakeTTS()
        p = voice_pipeline(stt=stt, tts=tts)
        r = p.run(RAGRequest(audio_base64="ZmFrZQ==", want_audio=False))
        assert not r.refused
        assert r.audio_base64 is None
        assert tts.calls == [], "TTS must not run when want_audio=False"

    def test_offline_stt_refuses_cleanly(self, voice_pipeline):
        """No SARVAM_API_KEY -> STT client is offline -> honest refusal."""
        from app.schemas import RAGRequest, RefusalReason

        p = voice_pipeline(stt=FakeSTT(offline=True), tts=FakeTTS())
        r = p.run(RAGRequest(audio_base64="ZmFrZQ==", want_audio=True))
        assert r.refused
        assert r.refusal_reason == RefusalReason.internal_error
        assert r.ok, "a clean refusal is still a successful response"

    def test_stt_failure_refuses_without_crashing(self, voice_pipeline):
        from app.schemas import RAGRequest, RefusalReason

        p = voice_pipeline(stt=FakeSTT(raises=RuntimeError("network down")),
                           tts=FakeTTS())
        r = p.run(RAGRequest(audio_base64="ZmFrZQ==", want_audio=True))
        assert r.refused and r.refusal_reason == RefusalReason.internal_error

    def test_tts_failure_is_non_fatal_text_still_returned(self, voice_pipeline):
        """
        TTS is presentation-only. If it fails, the text answer must still
        reach the user, with a warning -- never a lost answer.
        """
        from app.schemas import RAGRequest

        p = voice_pipeline(stt=FakeSTT(),
                           tts=FakeTTS(raises=RuntimeError("tts down")))
        r = p.run(RAGRequest(audio_base64="ZmFrZQ==", want_audio=True))
        assert not r.refused
        assert r.answer and r.audio_base64 is None
        assert any("tts failed" in w for w in r.warnings)

    def test_no_stt_client_configured_refuses(self, voice_pipeline):
        from app.schemas import RAGRequest, RefusalReason

        p = voice_pipeline(stt=None, tts=None)
        r = p.run(RAGRequest(audio_base64="ZmFrZQ==", want_audio=True))
        assert r.refused and r.refusal_reason == RefusalReason.internal_error

    def test_text_path_still_bypasses_stt(self, voice_pipeline):
        """The pre-existing text-only path must be unaffected by this work."""
        from app.schemas import RAGRequest

        stt, tts = FakeSTT(), FakeTTS()
        p = voice_pipeline(stt=stt, tts=tts)
        r = p.run(RAGRequest(text="how fast do eagles fly", want_audio=True))
        assert not r.refused
        assert stt.calls == [], "text requests must never call STT"
        assert tts.calls, "TTS should still run for a text request wanting audio"

    def test_requested_voice_is_passed_to_tts(self, voice_pipeline):
        from app.schemas import RAGRequest

        tts = FakeTTS()
        p = voice_pipeline(stt=FakeSTT(), tts=tts)
        p.run(RAGRequest(audio_base64="ZmFrZQ==", voice="female",
                         want_audio=True))
        assert tts.calls[0][2] == "female"


# --------------------------------------------------------------------------
# Secret hygiene: the one property this task explicitly demands
# --------------------------------------------------------------------------
class TestNoSecretExposure:
    def test_api_key_never_in_repr_of_transcription_or_speech(self):
        from app.stt.sarvam import Transcription
        from app.tts.sarvam_tts import Speech

        t = Transcription(text="x", language_code="en-IN",
                          language_probability=0.9, latency_ms=1.0)
        s = Speech(audio_base64="YWJj", latency_ms=1.0, speaker="aditya",
                  language_code="en-IN", chars=1)
        assert "test-key" not in repr(t) and "test-key" not in repr(s)

    def test_error_messages_do_not_include_the_api_key(self):
        def handler(request):
            return httpx.Response(401, text="unauthorized")

        with pytest.raises(STTError) as exc:
            stt_with(handler).transcribe(WAV_BYTES)
        assert "test-key" not in str(exc.value)
