"""
Grounded answer generation.

Provider
--------
The client speaks the OpenAI-compatible `/v1/chat/completions` shape, which
Sarvam's chat API and most other providers expose. Base URL, model and key all
come from config, so switching providers is an env change.

Extractive fallback
-------------------
Without an LLM key, `ExtractiveGenerator` answers by selecting the best-
supported sentences from the retrieved evidence. This is not a stand-in
pretending to be an LLM -- it is a legitimate, fully grounded baseline that:

  * can NEVER hallucinate, since every word is copied from the evidence, which
    makes it a useful control when measuring the grounding guardrail
  * lets the whole pipeline and the latency harness run end to end without
    credentials
  * gives a floor to compare LLM answer quality against

Its output is marked `extractive=True` and the evaluation reports it
separately. It does not carry the Mando persona -- persona requires
generation -- and the README says so rather than implying the demo voice is
available offline.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.generation.llm_client import (ChatClient, LLMError, LLMSchemaError,
                                       LLMTransient, StreamTimeout,
                                       extract_json)
from app.generation.stream_parser import StreamingAnswerExtractor
from app.generation.persona import (DEFAULT_PERSONA, PersonaConfig,
                                    build_user_prompt, system_prompt)
from app.guardrails.input import sanitize_for_prompt


class GenerationError(RuntimeError):
    pass


class GenerationTransient(GenerationError):
    """Retryable: timeout, 429, 5xx."""


@dataclass
class Generation:
    answer: str
    sources_used: list[int] = field(default_factory=list)
    # Source numbers the model cited that were NOT shown to it. Dropped from
    # sources_used and surfaced here so fabricated citations are visible
    # rather than silently discarded.
    invalid_sources: list[int] = field(default_factory=list)
    sufficient: bool = True
    latency_ms: float = 0.0
    extractive: bool = False
    raw: str | None = None
    attempts: int = 1
    schema_attempts: int = 1
    # Streaming instrumentation. None when the blocking path was used.
    time_to_first_token_ms: float | None = None
    time_to_first_visible_text_ms: float | None = None
    stream_duration_ms: float | None = None
    streamed: bool = False
    # True when the deadline fired after some visible text had already been
    # emitted. The answer is real but truncated, and the caller must say so.
    incomplete: bool = False


def _parse_structured(content: str) -> dict:
    """
    Parse the model's JSON reply, tolerating the usual deviations.

    Models wrap JSON in markdown fences or add prose around it often enough
    that failing hard here would turn a cosmetic formatting slip into a total
    request failure. We recover the object; only genuinely unparseable output
    falls back to treating the whole reply as the answer text.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Unparseable: treat the raw text as the answer, and mark it sufficient so
    # the grounding verifier -- not a JSON parse error -- makes the call.
    return {"answer": content.strip(), "sources_used": [], "sufficient": True}


class GenerationOutput(BaseModel):
    """
    Strict schema for the generator's reply. Extra keys are rejected rather
    than ignored, so a model that invents a field fails validation and gets a
    corrective retry instead of silently smuggling something past us.
    """

    model_config = {"extra": "forbid"}

    answer: str
    sources_used: list[int] = Field(default_factory=list)
    sufficient: bool = True

    @field_validator("sources_used", mode="before")
    @classmethod
    def _coerce_sources(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, int)):
            v = [v]
        out = []
        for x in v:
            # Models emit "1", "[1]", "source 1"; keep the digits, drop prose.
            m = re.search(r"\d+", str(x))
            if m:
                out.append(int(m.group()))
        return out


class LLMGenerator:
    """
    Grounded generation on the shared ChatClient.

    Migrated off its own httpx/tenacity stack (Phase 3). What that fixed:

    * `max_tokens` was 400. sarvam-105b is a REASONING model that emits
      `reasoning_content` before its answer -- at 400 it returns
      finish_reason="length" with content=None, i.e. no answer at all. The
      judge hit exactly this and 3000 is the configuration validated across
      600 calibration calls (EXPERIMENTS.md E15).
    * retry classification, timeout handling and JSON recovery were duplicated
      and had already drifted (0.3s/3s backoff vs the client's rate-limit-sized
      2s/30s over 5 attempts).
    * the reply was parsed with `dict.get()` and no schema, so a malformed
      object became an empty answer rather than a retry.
    """

    def __init__(self, client: ChatClient | None = None,
                 api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout_s: float = 30.0,
                 temperature: float = 0.3, max_tokens: int = 3000,
                 max_schema_attempts: int = 2,
                 persona: PersonaConfig | None = None,
                 deadline_s: float = 20.0):
        self.client = client or ChatClient(
            api_key=api_key, model=model, base_url=base_url,
            timeout_s=timeout_s)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_schema_attempts = max_schema_attempts
        self.persona = persona or DEFAULT_PERSONA
        # HARD wall-clock budget for a single generation. E17 recorded a
        # 65,132 ms Marathi call; a per-read socket timeout does not stop a
        # server that dribbles bytes slowly, so this is total-elapsed.
        self.deadline_s = deadline_s

    @property
    def available(self) -> bool:
        return self.client.available

    @property
    def model(self) -> str:
        return self.client.model

    def generate(self, query: str, evidence_texts: list[str], lang: str,
                 history=None) -> Generation:
        if not self.client.available:
            raise GenerationError("no LLM API key configured")
        if not evidence_texts:
            raise GenerationError("refusing to generate with no evidence")

        n_sources = len(evidence_texts)
        messages = [
            {"role": "system", "content": system_prompt(lang, self.persona)},
            {"role": "user", "content": build_user_prompt(
                sanitize_for_prompt(query), evidence_texts, history)},
        ]

        def _validate(payload: dict) -> GenerationOutput:
            try:
                out = GenerationOutput.model_validate(payload)
            except ValidationError as exc:
                raise LLMSchemaError(
                    "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: "
                              f"{e['msg']}" for e in exc.errors()[:4])) from exc
            if out.sufficient and not out.answer.strip():
                raise LLMSchemaError(
                    "sufficient=true but answer is empty")
            return out

        try:
            parsed, meta = self.client.chat_json(
                messages, _validate, temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_schema_attempts=self.max_schema_attempts,
                response_format={"type": "json_object"})
        except (LLMSchemaError, LLMTransient, LLMError) as exc:
            raise GenerationError(f"{type(exc).__name__}: {exc}") from exc

        # CITATION VALIDATION. The model may only cite source numbers it was
        # actually shown. Out-of-range citations are dropped and recorded --
        # a claim of support we cannot verify is not support. This mirrors the
        # judge's anti-fabrication rule, enforced in code rather than trusted
        # to the prompt.
        valid = [i for i in parsed.sources_used if 1 <= i <= n_sources]
        invalid = [i for i in parsed.sources_used if i not in valid]

        return Generation(
            answer=parsed.answer.strip(),
            sources_used=valid,
            invalid_sources=invalid,
            sufficient=parsed.sufficient,
            latency_ms=meta.latency_ms,
            raw=None,
            attempts=meta.http_attempts,
            schema_attempts=meta.schema_attempts,
        )


    def generate_stream(self, query: str, evidence_texts: list[str], lang: str,
                        history=None, deadline_s: float | None = None):
        """
        Stream a grounded answer.

        Yields user-visible text deltas, then finally a validated `Generation`.
        Usage:

            for item in gen.generate_stream(...):
                if isinstance(item, Generation):
                    final = item        # validated, cited, safe to trust
                else:
                    emit(item)          # prose the user can see NOW

        What streaming does NOT relax
        -----------------------------
        The evidence is still the only factual source, the strict schema still
        validates the completed object, and citations are still range-checked.
        Streaming changes WHEN text appears, not WHAT is allowed to appear.

        Degradation, in the order it can occur:

        * deadline before any visible text  -> raise; caller refuses. Nothing
          half-formed is shown.
        * deadline after some visible text  -> stop, mark `incomplete=True`.
          The text already shown was real evidence-grounded output; we do not
          retract it, but the caller must not present it as a whole answer.
        * malformed / non-JSON reply        -> raise. Anything already emitted
          came from inside the `answer` string, so no JSON syntax leaked, but
          we refuse to hand back an unvalidated object.
        * transport failure                 -> raise; caller falls back.
        """
        if not self.client.available:
            raise GenerationError("no LLM API key configured")
        if not evidence_texts:
            raise GenerationError("refusing to generate with no evidence")

        deadline = deadline_s if deadline_s is not None else self.deadline_s
        n_sources = len(evidence_texts)
        messages = [
            {"role": "system", "content": system_prompt(lang, self.persona)},
            {"role": "user", "content": build_user_prompt(
                sanitize_for_prompt(query), evidence_texts, history)},
        ]

        extractor = StreamingAnswerExtractor()
        t0 = time.perf_counter()
        ttft = ttfv = None
        timed_out = False

        try:
            for piece in self.client.stream_chat(
                    messages, temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    deadline_s=deadline):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                visible = extractor.feed(piece)
                if visible:
                    if ttfv is None:
                        ttfv = (time.perf_counter() - t0) * 1000
                    yield visible
        except StreamTimeout:
            timed_out = True
            if not extractor.emitted:
                # Nothing was shown; a clean refusal is the honest outcome.
                raise GenerationError(
                    f"generation deadline ({deadline}s) exceeded before any "
                    f"answer text") from None
        except (LLMTransient, LLMError) as exc:
            raise GenerationError(f"{type(exc).__name__}: {exc}") from exc

        duration = (time.perf_counter() - t0) * 1000

        if timed_out:
            # Partial but real text. Return it marked incomplete rather than
            # discarding work the user has already seen.
            yield Generation(
                answer=extractor.emitted.strip(), sources_used=[],
                sufficient=True, latency_ms=duration, streamed=True,
                incomplete=True, time_to_first_token_ms=ttft,
                time_to_first_visible_text_ms=ttfv,
                stream_duration_ms=duration)
            return

        try:
            payload = extract_json(extractor.raw)
            parsed = GenerationOutput.model_validate(payload)
        except Exception as exc:
            raise GenerationError(
                f"stream produced unvalidatable output: {exc}") from exc

        valid = [i for i in parsed.sources_used if 1 <= i <= n_sources]
        invalid = [i for i in parsed.sources_used if i not in valid]

        yield Generation(
            answer=parsed.answer.strip(), sources_used=valid,
            invalid_sources=invalid, sufficient=parsed.sufficient,
            latency_ms=duration, streamed=True,
            time_to_first_token_ms=ttft,
            time_to_first_visible_text_ms=ttfv,
            stream_duration_ms=duration)

    def close(self) -> None:
        self.client.close()


class ExtractiveGenerator:
    """
    Grounded-by-construction baseline: returns the evidence sentences most
    similar to the query. Cannot hallucinate, because it never writes a word
    that is not already in the sources.
    """

    def __init__(self, embedder, max_sentences: int = 2,
                 max_passages: int = 3, sentence_budget: int = 14):
        self.embedder = embedder
        self.max_sentences = max_sentences
        # Encoding costs ~23 ms/text on CPU, and the naive version encoded
        # every sentence of all five evidence passages (~30 sentences,
        # ~150 ms). The best answer almost always comes from the top-ranked
        # passages, so we consider only those and cap the total.
        self.max_passages = max_passages
        self.sentence_budget = sentence_budget

    def generate(self, query: str, evidence_texts: list[str], lang: str,
                 history=None) -> Generation:
        from ingestion.chunk import split_sentences

        t0 = time.perf_counter()
        sentences: list[str] = []
        for text in evidence_texts[:self.max_passages]:
            sentences.extend(split_sentences(text))
        sentences = [s for s in sentences if len(s) > 25][:self.sentence_budget]

        if not sentences:
            return Generation(answer="", sufficient=False,
                              latency_ms=(time.perf_counter() - t0) * 1000,
                              extractive=True)

        qv = self.embedder.encode_queries([query], batch_size=1)[0]
        sv = self.embedder.encode_passages(sentences,
                                           batch_size=len(sentences))
        scores = sv @ qv
        top = np.argsort(-scores)[:self.max_sentences]
        top = sorted(top)                      # keep original reading order

        answer = " ".join(sentences[int(i)] for i in top)
        return Generation(answer=answer, sources_used=[1], sufficient=True,
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          extractive=True)
