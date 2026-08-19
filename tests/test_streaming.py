"""
Streaming generation, incremental JSON extraction, deadlines and degradation.

The extractor is tested at every chunk boundary because the network decides
where chunks split, and a parser that only works on tidy boundaries is a parser
that fails in production.
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.generation.generator import (Generation, GenerationError,   # noqa: E402
                                      LLMGenerator)
from app.generation.llm_client import (ChatClient, LLMError,         # noqa: E402
                                       LLMTransient, StreamTimeout)
from app.generation.stream_parser import StreamingAnswerExtractor    # noqa: E402

EVIDENCE = ["Eagles fly 30 to 55 mph and dive at over 100 mph."]


def obj(answer="Eagles fly 30 to 55 mph.", sources=(1,), sufficient=True):
    return {"answer": answer, "sources_used": list(sources),
            "sufficient": sufficient}


def feed_all(payload: str, chunk: int) -> tuple[str, StreamingAnswerExtractor]:
    e = StreamingAnswerExtractor()
    out = "".join(e.feed(payload[i:i + chunk])
                  for i in range(0, len(payload), chunk))
    return out, e


# --------------------------------------------------------------------------
# Incremental extraction
# --------------------------------------------------------------------------
class TestStreamingExtractor:
    @pytest.mark.parametrize("chunk", [1, 2, 3, 5, 17, 999])
    @pytest.mark.parametrize("ensure_ascii", [True, False])
    def test_ascii_at_every_boundary(self, chunk, ensure_ascii):
        o = obj("Eagles fly 30 to 55 mph. He said \"fast\".\nDone.")
        out, _ = feed_all(json.dumps(o, ensure_ascii=ensure_ascii), chunk)
        assert out == o["answer"]

    @pytest.mark.parametrize("chunk", [1, 2, 3, 5, 17])
    @pytest.mark.parametrize("ensure_ascii", [True, False])
    def test_devanagari_at_every_boundary(self, chunk, ensure_ascii):
        """
        REGRESSION: json.dumps emits Devanagari as \\uXXXX by default. The
        first implementation computed the escape length before knowing whether
        a lone trailing "\\" was a 2-char or 6-char escape, truncating every
        \\u sequence and dropping Devanagari entirely.
        """
        o = obj("गरुड ३० ते ५५ मैल प्रति तास वेगाने उडतात")
        out, _ = feed_all(json.dumps(o, ensure_ascii=ensure_ascii), chunk)
        assert out == o["answer"]

    @pytest.mark.parametrize("chunk", [1, 4, 999])
    def test_no_json_syntax_ever_leaks(self, chunk):
        """The user must never see braces, keys, or trailing fields."""
        out, _ = feed_all(json.dumps(obj(sources=(1, 2))), chunk)
        for forbidden in ("{", "}", "sources_used", "sufficient", '"answer"'):
            assert forbidden not in out

    def test_trailing_fields_are_not_emitted(self):
        out, e = feed_all(json.dumps(obj("short")), 1)
        assert out == "short"
        assert e.state.finished_answer

    def test_raw_is_retained_for_validation(self):
        payload = json.dumps(obj())
        _, e = feed_all(payload, 3)
        assert e.raw == payload, "full raw text must survive for the schema"

    def test_escapes_decoded(self):
        o = obj('line1\nline2\ttab "quoted" back\\slash')
        out, _ = feed_all(json.dumps(o), 1)
        assert out == o["answer"]

    def test_non_json_emits_nothing(self):
        out, e = feed_all("this is not json at all", 3)
        assert out == ""
        assert not e.looks_like_json()

    def test_empty_answer(self):
        out, _ = feed_all(json.dumps(obj("")), 2)
        assert out == ""


# --------------------------------------------------------------------------
# SSE transport
# --------------------------------------------------------------------------
def sse(chunks, include_reasoning=False):
    """Build an SSE body; optionally interleave reasoning_content deltas."""
    lines = []
    for c in chunks:
        if include_reasoning:
            lines.append("data: " + json.dumps(
                {"choices": [{"delta": {"reasoning_content": "SECRET"}}]}))
        lines.append("data: " + json.dumps(
            {"choices": [{"delta": {"content": c}}]}))
    lines.append("data: [DONE]")
    return "\n".join(lines) + "\n"


def streaming_client(body, status=200, **kw):
    def handler(request):
        return httpx.Response(status, text=body,
                              headers={"content-type": "text/event-stream"})
    c = ChatClient(api_key="k", base_url="https://example.invalid/v1", **kw)
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


class TestStreamTransport:
    def test_yields_content_deltas(self):
        c = streaming_client(sse(["a", "b", "c"]))
        assert list(c.stream_chat([{"role": "user", "content": "x"}])) == \
            ["a", "b", "c"]

    def test_reasoning_content_is_never_yielded(self):
        """
        Requirement: internal reasoning must not stream to a user. The API
        sends it as a separate delta FIELD, so this is structural filtering,
        not string matching on the output.
        """
        c = streaming_client(sse(["visible"], include_reasoning=True))
        out = "".join(c.stream_chat([{"role": "user", "content": "x"}]))
        assert out == "visible"
        assert "SECRET" not in out

    def test_done_sentinel_stops(self):
        body = sse(["a"]) + "data: " + json.dumps(
            {"choices": [{"delta": {"content": "after-done"}}]}) + "\n"
        c = streaming_client(body)
        assert "after-done" not in "".join(
            c.stream_chat([{"role": "user", "content": "x"}]))

    def test_malformed_frames_are_skipped(self):
        body = ("data: not-json\n"
                + "data: " + json.dumps(
                    {"choices": [{"delta": {"content": "ok"}}]}) + "\n"
                + "data: [DONE]\n")
        c = streaming_client(body)
        assert "".join(c.stream_chat([{"role": "user", "content": "x"}])) == "ok"

    def test_4xx_raises_non_transient(self):
        c = streaming_client("nope", status=401)
        with pytest.raises(LLMError):
            list(c.stream_chat([{"role": "user", "content": "x"}]))

    def test_5xx_raises_transient(self):
        c = streaming_client("boom", status=503)
        with pytest.raises(LLMTransient):
            list(c.stream_chat([{"role": "user", "content": "x"}]))

    def test_no_key_raises(self):
        c = ChatClient(api_key=None)
        c.available = False
        with pytest.raises(LLMError):
            list(c.stream_chat([{"role": "user", "content": "x"}]))


# --------------------------------------------------------------------------
# Generator streaming + degradation
# --------------------------------------------------------------------------
class StreamingFake:
    """Scripted streaming client. `delay` simulates a slow server."""

    def __init__(self, payload, chunk=4, delay=0.0, raises=None,
                 available=True):
        self.payload = payload
        self.chunk = chunk
        self.delay = delay
        self.raises = raises
        self.available = available
        self.model = "fake"

    def stream_chat(self, messages, temperature=0.0, max_tokens=3000,
                    response_format=None, deadline_s=None):
        if self.raises:
            raise self.raises
        started = time.perf_counter()
        for i in range(0, len(self.payload), self.chunk):
            if self.delay:
                time.sleep(self.delay)
            if deadline_s is not None and \
                    time.perf_counter() - started > deadline_s:
                raise StreamTimeout(f"exceeded {deadline_s}s")
            yield self.payload[i:i + self.chunk]

    def close(self):
        pass


def run_stream(gen, **kw):
    text, final = "", None
    for item in gen.generate_stream("q", EVIDENCE, "en", **kw):
        if isinstance(item, Generation):
            final = item
        else:
            text += item
    return text, final


class TestGeneratorStreaming:
    def test_streams_visible_text_and_returns_validated_final(self):
        g = LLMGenerator(client=StreamingFake(json.dumps(obj())))
        text, final = run_stream(g)
        assert text == obj()["answer"]
        assert final is not None and final.streamed
        assert final.answer == obj()["answer"]
        assert final.sources_used == [1] and not final.incomplete

    def test_instrumentation_is_populated(self):
        g = LLMGenerator(client=StreamingFake(json.dumps(obj()), chunk=2))
        _, final = run_stream(g)
        assert final.time_to_first_token_ms is not None
        assert final.time_to_first_visible_text_ms is not None
        assert final.stream_duration_ms is not None
        assert (final.time_to_first_visible_text_ms
                >= final.time_to_first_token_ms), \
            "visible text cannot precede the first token"

    def test_citations_still_validated_when_streaming(self):
        g = LLMGenerator(client=StreamingFake(
            json.dumps(obj(sources=(1, 99)))))
        _, final = run_stream(g)
        assert final.sources_used == [1]
        assert final.invalid_sources == [99]

    def test_refuses_without_evidence(self):
        g = LLMGenerator(client=StreamingFake(json.dumps(obj())))
        with pytest.raises(GenerationError):
            list(g.generate_stream("q", [], "en"))

    def test_no_key_raises(self):
        g = LLMGenerator(client=StreamingFake("", available=False))
        with pytest.raises(GenerationError):
            list(g.generate_stream("q", EVIDENCE, "en"))

    def test_devanagari_streams_correctly(self):
        o = obj("गरुड ३० ते ५५ मैल प्रति तास")
        g = LLMGenerator(client=StreamingFake(
            json.dumps(o, ensure_ascii=True), chunk=3))
        text, final = run_stream(g)
        assert text == o["answer"] and final.answer == o["answer"]


class TestDeadlineAndDegradation:
    def test_deadline_before_any_text_raises(self):
        """No partial output shown -> the caller must refuse cleanly."""
        g = LLMGenerator(client=StreamingFake(
            json.dumps(obj()), chunk=1, delay=0.05))
        with pytest.raises(GenerationError) as exc:
            list(g.generate_stream("q", EVIDENCE, "en", deadline_s=0.01))
        assert "deadline" in str(exc.value)

    def test_deadline_after_text_marks_incomplete(self):
        long_answer = "word " * 200
        g = LLMGenerator(client=StreamingFake(
            json.dumps(obj(long_answer)), chunk=8, delay=0.004))
        text, final = run_stream(g, deadline_s=0.12)
        assert text, "text already shown is not retracted"
        assert final is not None and final.incomplete
        assert final.sources_used == [], \
            "a truncated answer must not claim validated citations"

    def test_deadline_default_is_bounded(self):
        """E17 saw a 65 s call; the default must be far below that."""
        assert LLMGenerator(api_key="k").deadline_s <= 30

    def test_malformed_stream_raises_and_shows_nothing(self):
        g = LLMGenerator(client=StreamingFake("this is not json"))
        with pytest.raises(GenerationError):
            list(g.generate_stream("q", EVIDENCE, "en"))

    def test_transport_failure_raises_generation_error(self):
        g = LLMGenerator(client=StreamingFake(
            "", raises=LLMTransient("connection reset")))
        with pytest.raises(GenerationError):
            list(g.generate_stream("q", EVIDENCE, "en"))

    def test_blocking_path_still_works(self):
        """Streaming is additive; the non-streaming path must be untouched."""
        from test_answerability import FakeClient
        g = LLMGenerator(client=FakeClient([json.dumps(obj())]))
        out = g.generate("q", EVIDENCE, "en")
        assert out.answer and not out.streamed
