"""Final language-adherence regressions; all providers are mocked."""

from __future__ import annotations

import numpy as np
import pytest

import app.pipeline as pipeline_module
from app.config import Config
from app.generation.generator import Generation
from app.generation.persona import refusal_text_with_locale
from app.guardrails.answerability import AnswerabilityResult
from app.guardrails.grounding import GroundingResult
from app.language import detect_language, resolve_language
from app.pipeline import RAGPipeline
from app.retrieval.hybrid import Candidate, RetrievalTiming
from app.schemas import RAGRequest, RefusalReason


class FakeEmbedder:
    def encode_queries(self, texts, **_):
        return np.ones((len(texts), 2), dtype="float32")

    def encode_passages(self, texts, **_):
        return np.ones((len(texts), 2), dtype="float32")


class CountingRetriever:
    def __init__(self):
        self.calls = 0
        self.matrix = None

    def retrieve(self, query, **_):
        self.calls += 1
        text = "A corporation is a company authorized by law."
        candidate = Candidate(
            doc_id="d1", text=text, context=text, query_id=1,
            passage_index=0, lang="en", dense_score=0.99)
        timing = RetrievalTiming(query_vector=np.ones(2, dtype="float32"))
        return [candidate], timing


class CountingReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, candidates, **_):
        self.calls += 1
        return candidates, 0.0


class CountingJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, query, evidence, lang):
        self.calls += 1
        return AnswerabilityResult(
            available=True, sufficient=True, confidence=1.0,
            supporting_ids=["p1"], supporting_indices=[0])


class CountingExternal:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return Generation(answer="external")


class CorrectingGenerator:
    def __init__(self, first_answer: str, corrected_answer: str):
        self.first_answer = first_answer
        self.corrected_answer = corrected_answer
        self.generate_calls = []
        self.correction_calls = []

    def generate(self, query, evidence, lang, history=None,
                 answer_mode="detailed"):
        self.generate_calls.append((query, evidence, lang, answer_mode))
        return Generation(answer=self.first_answer, sources_used=[1])

    def correct_language(self, query, evidence, lang, previous_answer,
                         history=None, answer_mode="detailed"):
        self.correction_calls.append(
            (query, evidence, lang, previous_answer, answer_mode))
        return Generation(answer=self.corrected_answer, sources_used=[1])


def build_pipeline(monkeypatch, first_answer, corrected_answer,
                   fail_second_grounding=False):
    grounding_calls = []

    def verify(answer, evidence, *_args, **_kwargs):
        grounding_calls.append((answer, evidence))
        passed = not (fail_second_grounding and len(grounding_calls) == 2)
        return GroundingResult(passed, 0.99,
                               detail="grounded" if passed else "not grounded")

    monkeypatch.setattr(pipeline_module, "verify_grounding", verify)
    retriever = CountingRetriever()
    reranker = CountingReranker()
    judge = CountingJudge()
    external = CountingExternal()
    generator = CorrectingGenerator(first_answer, corrected_answer)
    cfg = Config()
    cfg.evidence_min_top = cfg.evidence_min_mean = 0.8
    cfg.enforce_grounding = True
    pipeline = RAGPipeline(
        retriever, FakeEmbedder(), reranker, config=cfg,
        generator=generator, judge=judge, judge_mode="sync",
        external_generator=external)
    return pipeline, retriever, reranker, judge, external, generator, grounding_calls


@pytest.mark.parametrize("lang,first,corrected", [
    ("mr", "निगम एक कंपनी है।",
     "कॉर्पोरेशन ही कायद्याने अधिकृत कंपनी आहे."),
    ("hi", "कॉर्पोरेशन ही कायद्याने अधिकृत कंपनी आहे।",
     "निगम कानून द्वारा अधिकृत कंपनी है।"),
    ("en", "निगम एक कंपनी है।",
     "A corporation is a company authorized by law."),
])
def test_explicit_language_gets_one_isolated_correction(
        monkeypatch, lang, first, corrected):
    (pipeline, retriever, reranker, judge, external, generator,
     grounding_calls) = build_pipeline(monkeypatch, first, corrected)

    result = pipeline.run(RAGRequest(
        text="what is a corporation?", language=lang, want_audio=False))

    assert result.language.value == lang
    assert detect_language(result.answer).lang == lang
    assert result.language_match is True
    assert len(generator.generate_calls) == 1
    assert len(generator.correction_calls) == 1
    assert retriever.calls == reranker.calls == judge.calls == 1
    assert external.calls == 0
    assert len(grounding_calls) == 2
    assert generator.generate_calls[0][0] == generator.correction_calls[0][0]
    assert generator.generate_calls[0][1] is generator.correction_calls[0][1]
    assert grounding_calls[0][1] is grounding_calls[1][1]


def test_correction_never_loops_when_second_answer_still_mismatches(monkeypatch):
    pipeline, *_, generator, grounding_calls = build_pipeline(
        monkeypatch, "निगम एक कंपनी है।", "निगम फिर भी कंपनी है।")
    result = pipeline.run(RAGRequest(
        text="what is a corporation?", language="mr", want_audio=False))
    assert not result.refused
    assert result.language.value == "mr"
    assert result.language_match is False
    assert len(generator.correction_calls) == 1
    assert len(grounding_calls) == 2
    assert any("after one corrective retry" in warning
               for warning in result.warnings)


def test_corrected_answer_is_grounded_again_and_refused_if_it_fails(monkeypatch):
    pipeline, *_, generator, grounding_calls = build_pipeline(
        monkeypatch, "निगम एक कंपनी है।",
        "कॉर्पोरेशन ही कायद्याने अधिकृत कंपनी आहे.",
        fail_second_grounding=True)
    result = pipeline.run(RAGRequest(
        text="what is a corporation?", language="mr", want_audio=False))
    assert len(generator.correction_calls) == 1
    assert len(grounding_calls) == 2
    assert result.refused
    assert result.refusal_reason == RefusalReason.ungrounded_answer


def test_language_signal_precedence_is_strict():
    assert resolve_language("mr", "निगम क्या है", "hi-IN").lang == "mr"
    assert resolve_language("auto", "कॉर्पोरेशन म्हणजे काय", "hi-IN").lang == "hi"
    assert resolve_language("auto", "कॉर्पोरेशन म्हणजे काय").lang == "mr"


@pytest.mark.parametrize("lang,marker", [("mr", "माहिती"), ("hi", "जानकारी")])
def test_refusal_copy_uses_resolved_language(lang, marker):
    text, localised = refusal_text_with_locale(lang, "insufficient_evidence")
    assert localised
    assert marker in text


def test_konkani_honesty_behavior_does_not_trigger_language_retry(monkeypatch):
    pipeline, *_, generator, grounding_calls = build_pipeline(
        monkeypatch, "I can only answer this safely in English.",
        "unused correction")
    result = pipeline.run(RAGRequest(
        text="Goa information", language="kok", want_audio=False))
    assert result.language.value == "kok"
    assert result.language_match is None
    assert generator.correction_calls == []
    assert len(grounding_calls) == 1
