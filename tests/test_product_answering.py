"""Focused final-product regressions; no network or paid provider calls."""

from __future__ import annotations
import base64

import numpy as np
import httpx
import pytest

from app.api.v1.translate import to_text_response
from app.config import Config
from app.generation.generator import ExtractiveGenerator, Generation
from app.generation.external import ExternalLLMGenerator
from app.pipeline import RAGPipeline
from app.stt.elevenlabs import ElevenLabsSTT
from app.retrieval.goa_knowledge import GoaKnowledgeBase
from app.retrieval.hybrid import Candidate, RetrievalTiming
from app.schemas import AnswerOrigin, RAGRequest, Stage
from app.telemetry import LatencyTracker


class HashEmbedder:
    dim = 256

    def _encode(self, texts):
        rows = []
        for text in texts:
            vector = np.zeros(self.dim, dtype="float32")
            for token in text.casefold().replace("?", "").replace(".", "").split():
                vector[hash(token) % self.dim] += 1
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows)

    def encode_queries(self, texts, **_):
        return self._encode(texts)

    def encode_passages(self, texts, **_):
        return self._encode(texts)


class ProductRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.matrix = None

    def retrieve(self, query, **_):
        qv = self.embedder.encode_queries([query])[0]
        if "corporation" in query.casefold():
            text = ("A corporation is a company authorized by law to act as "
                    "a single entity. Corporations can own property in their "
                    "own name.")
            candidates = [Candidate(
                doc_id="corpus:corporation", text=text, context=text,
                query_id=1, passage_index=0, lang="en", dense_score=.96)]
        else:
            candidates = [Candidate(
                doc_id="corpus:unrelated", text="Eagles are birds.",
                context="Eagles are birds.", query_id=2, passage_index=0,
                lang="en", dense_score=.10)]
        return candidates, RetrievalTiming(query_vector=qv)


class PassthroughReranker:
    def rerank(self, query, candidates, **_):
        return candidates, 0.0


class ExternalGenerator:
    def generate(self, query, lang, answer_mode="detailed"):
        return Generation(answer="This answer comes from general knowledge.")


class FailingExternalGenerator:
    def generate(self, query, lang, answer_mode="detailed"):
        raise RuntimeError("provider unavailable")


def build_pipeline(external_type=ExternalGenerator):
    embedder = HashEmbedder()
    cfg = Config()
    cfg.top_k = 3
    cfg.candidate_k = 5
    cfg.evidence_min_top = cfg.evidence_min_mean = .80
    cfg.grounding_min_semantic = .20
    return RAGPipeline(
        ProductRetriever(embedder), embedder, PassthroughReranker(),
        config=cfg, generator=ExtractiveGenerator(embedder),
        goa_knowledge=GoaKnowledgeBase(embedder),
        external_generator=(external_type() if external_type else None))


def test_existing_corpus_answer_is_classified_and_grounded():
    result = build_pipeline().run(RAGRequest(
        text="What is a corporation?", want_audio=False))
    assert not result.refused
    assert result.answer_origin == AnswerOrigin.corpus
    assert result.grounded is True


def test_capital_of_goa_comes_from_actual_goa_evidence():
    result = build_pipeline().run(RAGRequest(
        text="What is the capital of Goa?", answer_mode="fast",
        want_audio=False))
    assert not result.refused
    assert result.answer_origin == AnswerOrigin.goa_knowledge
    assert "Panaji" in result.answer
    assert any("Panaji" in evidence.text and evidence.doc_id.startswith("goa:")
               for evidence in result.evidence)
    public = to_text_response(result, None)
    assert public.sources and public.sources[0].source_name


def test_external_answer_has_no_fake_sources_and_is_unverified():
    result = build_pipeline().run(RAGRequest(
        text="What is the weather on Mars today?", want_audio=False))
    public = to_text_response(result, None)
    assert public.answer_origin == "external"
    assert public.external_verified is False
    assert public.guardrail.grounded is False
    assert public.sources == []


def test_external_provider_failure_preserves_clean_refusal():
    result = build_pipeline(FailingExternalGenerator).run(RAGRequest(
        text="What is the weather on Mars today?", want_audio=False))
    assert result.ok and result.refused
    assert result.answer_origin is None


def test_fast_mode_stays_grounded_and_is_shorter_than_detailed():
    pipeline = build_pipeline()
    fast = pipeline.run(RAGRequest(
        text="What is a corporation?", answer_mode="fast", want_audio=False))
    detailed = pipeline.run(RAGRequest(
        text="What is a corporation?", answer_mode="detailed", want_audio=False))
    assert fast.grounded and detailed.grounded
    assert Stage.grounding in {timing.stage for timing in fast.latency.stages}
    assert len(fast.answer) <= len(detailed.answer)


def test_percentiles_require_multiple_samples_and_stay_separated():
    tracker = LatencyTracker(capacity=3, minimum_samples=2)
    one = tracker.record("text", "fast", 100)
    assert one["collecting"] and one["p50_ms"] is None
    tracker.record("text", "fast", 200)
    fast = tracker.record("text", "fast", 300)
    detailed = tracker.record("text", "detailed", 900)
    voice = tracker.record("voice", "fast", 1200)
    assert fast["samples"] == 3 and fast["p50_ms"] == 200
    assert detailed["samples"] == 1 and voice["samples"] == 1
    assert len(tracker._samples[("text", "fast")]) == 3


def test_frontend_listen_reuses_answer_and_paid_tts_is_not_default():
    text = open("frontend/index.html", encoding="utf-8").read()
    toggle = text[text.index("function toggleMandoSpeech"):
                  text.index("function startSpeech")]
    assert "fetch(" not in toggle
    assert "currentBackendAudio" in toggle
    assert "SpeechSynthesisUtterance(rawText)" in text
    assert "formData.append('want_audio', 'false')" in text
    assert "Not verified against the HHGoa corpus." in text
    assert "innerHTML = rows.map" in text and "escapeHtml(value)" in text


@pytest.mark.parametrize("query,language", [
    ("What is the capital of Goa?", "en"),
    ("गोवा की राजधानी क्या है?", "hi"),
    ("गोव्याची राजधानी काय आहे?", "mr"),
])
def test_multilingual_capital_queries_use_goa_knowledge(query, language):
    result = build_pipeline().run(RAGRequest(
        text=query, language=language, answer_mode="fast", want_audio=False))
    assert result.answer_origin == AnswerOrigin.goa_knowledge
    assert "Panaji" in result.answer
    assert result.evidence[0].source_url


def test_missing_external_configuration_preserves_refusal():
    result = build_pipeline(None).run(RAGRequest(
        text="What is the weather on Mars today?", want_audio=False))
    assert result.ok and result.refused
    assert result.answer_origin is None


def test_external_configuration_never_reuses_primary(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "primary-secret")
    monkeypatch.setenv("SARVAM_API_KEY", "sarvam-secret")
    external = ExternalLLMGenerator(
        api_key="external-secret", base_url="https://external.invalid/v1",
        model="external-model")
    assert external.client.api_key == "external-secret"
    assert external.client.base_url == "https://external.invalid/v1"
    assert external.client.model == "external-model"
    external.close()


def test_elevenlabs_stt_maps_provider_response_without_network():
    def handler(request):
        assert request.headers["xi-api-key"] == "test-key"
        assert b'name="model_id"' in request.content
        assert b"scribe_v2" in request.content
        return httpx.Response(200, json={
            "text": "Namaskar Goa",
            "language_code": "en",
            "language_probability": 0.98,
        })
    client = httpx.Client(transport=httpx.MockTransport(handler))
    stt = ElevenLabsSTT("test-key", client=client)
    result = stt.transcribe_base64(
        base64.b64encode(b"not-real-audio").decode(), language="en")
    assert result.text == "Namaskar Goa"
    assert result.language_code == "en"
    assert result.language_probability == 0.98
    stt.close()
