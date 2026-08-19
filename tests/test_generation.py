"""
Phase 3 tests: Mando generation, persona, async judge, verdict cache.

No API key required -- a scripted ChatClient drives every path.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.generation.generator import (ExtractiveGenerator,      # noqa: E402
                                      GenerationError,
                                      GenerationOutput, LLMGenerator)
from app.generation.llm_client import ChatClient                # noqa: E402
from app.generation.persona import (DEFAULT_PERSONA,            # noqa: E402
                                    PersonaConfig, system_prompt)
from app.guardrails.verdict_cache import (VerdictCache,         # noqa: E402
                                          normalize_query, verdict_key)
from app.schemas import has_capability                          # noqa: E402

from test_answerability import FakeClient                       # noqa: E402

EVIDENCE = ["Eagles fly 30 to 55 mph and dive at over 100 mph.",
            "Eagles do not migrate the way many other birds do."]


def gen_json(answer="Eagles fly 30 to 55 mph.", sources=(1,), sufficient=True):
    return json.dumps({"answer": answer, "sources_used": list(sources),
                       "sufficient": sufficient})


def generator_with(replies, **kw):
    return LLMGenerator(client=FakeClient(replies), **kw)


# --------------------------------------------------------------------------
# 1. LLMGenerator uses the shared ChatClient
# --------------------------------------------------------------------------
class TestSharedClientMigration:
    def test_uses_chatclient(self):
        assert isinstance(LLMGenerator(api_key="k").client, ChatClient)

    def test_no_private_http_stack(self):
        """The duplicated httpx/tenacity path must be gone, not dormant."""
        g = LLMGenerator(api_key="k")
        assert not hasattr(g, "_client"), "old private httpx client still there"
        assert not hasattr(g, "max_attempts"), "old retry config still there"

    def test_max_tokens_is_reasoning_sized(self):
        """
        REGRESSION: the old default was 400. sarvam-105b spends ~900 tokens on
        reasoning_content before answering, so 400 returns content=None and no
        answer at all. 3000 is the value validated over 600 judge calls (E15).
        """
        assert LLMGenerator(api_key="k").max_tokens >= 3000

    def test_inherits_client_retry_policy(self):
        assert LLMGenerator(api_key="k").client.max_http_attempts >= 5


# --------------------------------------------------------------------------
# 2. Structured output
# --------------------------------------------------------------------------
class TestStructuredOutput:
    def test_valid_reply(self):
        g = generator_with([gen_json()])
        out = g.generate("how fast do eagles fly", EVIDENCE, "en")
        assert out.answer.startswith("Eagles fly")
        assert out.sources_used == [1] and out.sufficient

    def test_rejects_extra_keys_then_retries(self):
        bad = json.dumps({"answer": "x", "sources_used": [1],
                          "sufficient": True, "smuggled": "field"})
        g = generator_with([bad, gen_json()])
        assert g.generate("q", EVIDENCE, "en").schema_attempts == 2

    def test_malformed_json_recovered_by_retry(self):
        g = generator_with(["not json at all", gen_json()])
        assert g.generate("q", EVIDENCE, "en").answer

    def test_fenced_json_accepted_without_retry(self):
        g = generator_with(["```json\n" + gen_json() + "\n```"])
        assert g.generate("q", EVIDENCE, "en").schema_attempts == 1

    def test_repeated_malformed_output_raises(self):
        g = generator_with(["garbage", "still garbage"])
        with pytest.raises(GenerationError):
            g.generate("q", EVIDENCE, "en")

    def test_sufficient_true_with_empty_answer_is_rejected(self):
        """A 'sufficient' verdict with nothing in it is not an answer."""
        g = generator_with([gen_json(answer="   "), gen_json()])
        assert g.generate("q", EVIDENCE, "en").answer.strip()

    @pytest.mark.parametrize("raw,expected", [
        (["1", "2"], [1, 2]), ("1", [1]), (1, [1]),
        (["[1]", "source 2"], [1, 2]), (None, []),
    ])
    def test_source_coercion(self, raw, expected):
        out = GenerationOutput.model_validate(
            {"answer": "a", "sources_used": raw, "sufficient": True})
        assert out.sources_used == expected


# --------------------------------------------------------------------------
# 3. Citation validation (anti-fabrication)
# --------------------------------------------------------------------------
class TestCitationValidation:
    def test_out_of_range_citations_are_dropped(self):
        g = generator_with([gen_json(sources=(1, 99))])
        out = g.generate("q", EVIDENCE, "en")      # only 2 sources exist
        assert out.sources_used == [1]
        assert out.invalid_sources == [99]

    def test_all_invalid_citations_are_surfaced(self):
        g = generator_with([gen_json(sources=(42, 77))])
        out = g.generate("q", EVIDENCE, "en")
        assert out.sources_used == []
        assert sorted(out.invalid_sources) == [42, 77]

    def test_valid_citations_kept(self):
        g = generator_with([gen_json(sources=(1, 2))])
        assert g.generate("q", EVIDENCE, "en").sources_used == [1, 2]


# --------------------------------------------------------------------------
# 4. Evidence-only generation
# --------------------------------------------------------------------------
class TestEvidenceOnly:
    def test_refuses_to_generate_without_evidence(self):
        """No evidence must never reach the model at all."""
        client = FakeClient([gen_json()])
        with pytest.raises(GenerationError):
            LLMGenerator(client=client).generate("q", [], "en")
        assert client.calls == [], "must not spend a call with no evidence"

    def test_no_key_raises(self):
        with pytest.raises(GenerationError):
            LLMGenerator(client=FakeClient([], available=False)).generate(
                "q", EVIDENCE, "en")

    def test_grounding_rules_precede_persona_in_prompt(self):
        """Persona must never be able to override factual constraints."""
        p = system_prompt("en")
        assert "ONLY information present in the numbered sources" in p
        assert "Never state a fact" in p

    def test_evidence_is_numbered_in_the_prompt(self):
        g = generator_with([gen_json()])
        g.generate("q", EVIDENCE, "en")
        user = g.client.calls[0][1]["content"]
        assert "[1]" in user and "[2]" in user


# --------------------------------------------------------------------------
# 5. Mando persona
# --------------------------------------------------------------------------
class TestPersona:
    def test_enabled_by_default(self):
        assert DEFAULT_PERSONA.enabled and DEFAULT_PERSONA.name == "Mando"
        assert "Mando" in system_prompt("en")

    def test_prompt_is_voice_invariant(self):
        """
        THE architectural guarantee: voice selects a TTS speaker and nothing
        else, so the same question yields the same answer for male and female
        Mando. If voice ever leaks into the prompt this test fails.
        """
        male = system_prompt("en", PersonaConfig(voice="male"))
        female = system_prompt("en", PersonaConfig(voice="female"))
        assert male == female
        assert "male" not in male.lower().split()

    def test_voice_is_selectable_without_touching_rag(self):
        for v in ("male", "female"):
            assert PersonaConfig(voice=v).tts_voice() == v
        assert PersonaConfig(voice="nonsense").tts_voice() == "male"

    def test_persona_can_be_disabled(self):
        p = system_prompt("en", PersonaConfig(enabled=False))
        assert "Mando" not in p
        assert "ONLY information present" in p, "grounding survives persona off"

    def test_renaming_persona(self):
        assert "Maya" in system_prompt("en", PersonaConfig(name="Maya"))

    def test_discourages_stereotype(self):
        p = system_prompt("en")
        assert "slang" in p.lower()

    def test_asks_for_concise_voice_friendly_answers(self):
        assert "two to four sentences" in system_prompt("en")


# --------------------------------------------------------------------------
# 6. Language handling, including the Konkani honesty rule
# --------------------------------------------------------------------------
class TestLanguage:
    @pytest.mark.parametrize("lang,name", [
        ("en", "English"), ("hi", "Hindi"), ("mr", "Marathi"),
    ])
    def test_evaluated_languages_named_in_prompt(self, lang, name):
        assert name in system_prompt(lang)

    @pytest.mark.parametrize("lang", ["en", "hi", "mr"])
    def test_no_fallback_hedge_for_evaluated_languages(self, lang):
        """We measured these; the model should just answer in them."""
        assert "honest fallback" not in system_prompt(lang)

    def test_konkani_gets_an_honesty_clause(self):
        """
        Konkani is a PRODUCT language with no benchmark data. The prompt tells
        the model to fall back to English rather than fake fluency, because we
        cannot verify Konkani quality and will not imply that we can.
        """
        p = system_prompt("kok")
        assert "Konkani" in p
        assert "honest fallback" in p

    def test_konkani_capabilities_are_not_overclaimed(self):
        assert has_capability("kok", "stt")
        assert not has_capability("kok", "tts")
        assert not has_capability("kok", "benchmark")

    def test_generator_passes_language_through(self):
        g = generator_with([gen_json()])
        g.generate("q", EVIDENCE, "mr")
        assert "Marathi" in g.client.calls[0][0]["content"]


# --------------------------------------------------------------------------
# 7. Verdict cache (async judge support)
# --------------------------------------------------------------------------
class TestVerdictCache:
    def test_key_is_deterministic(self):
        a = verdict_key("What is a corporation?", ["d1", "d2"], "v1")
        b = verdict_key("What is a corporation?", ["d1", "d2"], "v1")
        assert a == b

    def test_key_ignores_evidence_order(self):
        """Retrieval order may vary while the evidence set is identical."""
        assert verdict_key("q", ["d2", "d1"], "v1") == \
               verdict_key("q", ["d1", "d2"], "v1")

    def test_key_normalises_query_phrasing(self):
        assert verdict_key("What is a corporation?", ["d1"]) == \
               verdict_key("what is  a CORPORATION", ["d1"])

    def test_key_changes_with_evidence(self):
        assert verdict_key("q", ["d1"]) != verdict_key("q", ["d2"])

    def test_index_version_invalidates(self):
        """Rebuilding the index must not serve verdicts about old passages."""
        assert verdict_key("q", ["d1"], "v1") != verdict_key("q", ["d1"], "v2")

    def test_devanagari_normalisation(self):
        assert normalize_query("कॉर्पोरेशन  क्या है?") == "कॉर्पोरेशन क्या है"

    def test_hit_and_miss(self):
        c = VerdictCache()
        assert c.get("k") is None
        c.put("k", "verdict")
        assert c.get("k") == "verdict"
        assert c.stats.hits == 1 and c.stats.misses == 1

    def test_lru_eviction(self):
        c = VerdictCache(max_entries=2)
        for k in ("a", "b", "c"):
            c.put(k, k)
        assert len(c) == 2 and c.stats.evictions == 1

    def test_ttl_expiry(self):
        c = VerdictCache(ttl_s=0.01)
        c.put("k", "v")
        time.sleep(0.02)
        assert c.get("k") is None

    def test_claim_prevents_duplicate_background_calls(self):
        """A burst of identical queries must fire ONE judge call, not N."""
        c = VerdictCache()
        assert c.claim("k") is True
        assert c.claim("k") is False
        assert c.stats.in_flight_skips == 1

    def test_claim_released_on_failure(self):
        c = VerdictCache()
        c.claim("k")
        c.release("k")
        assert c.claim("k") is True

    def test_put_clears_in_flight(self):
        c = VerdictCache()
        c.claim("k")
        c.put("k", "v")
        assert c.claim("k") is False, "cached key must not be re-judged"


# --------------------------------------------------------------------------
# 8. Extractive fallback still works (LLM unavailable)
# --------------------------------------------------------------------------
class TestExtractiveFallback:
    def test_extractive_is_verbatim(self):
        class _Emb:
            def encode_queries(self, t, **kw):
                import numpy as np
                return np.ones((len(t), 4), dtype="float32")

            def encode_passages(self, t, **kw):
                import numpy as np
                return np.ones((len(t), 4), dtype="float32")

        g = ExtractiveGenerator(_Emb())
        out = g.generate("q", EVIDENCE, "en")
        assert out.extractive
        for frag in out.answer.split(". "):
            if len(frag) > 20:
                assert frag.strip(".") in " ".join(EVIDENCE)
