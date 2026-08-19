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

import json
import os
import time
from dataclasses import dataclass, field

import httpx
import numpy as np
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from app.generation.persona import build_user_prompt, system_prompt
from app.guardrails.input import sanitize_for_prompt


class GenerationError(RuntimeError):
    pass


class GenerationTransient(GenerationError):
    """Retryable: timeout, 429, 5xx."""


@dataclass
class Generation:
    answer: str
    sources_used: list[int] = field(default_factory=list)
    sufficient: bool = True
    latency_ms: float = 0.0
    extractive: bool = False
    raw: str | None = None
    attempts: int = 1


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


class LLMGenerator:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout_s: float = 15.0,
                 max_attempts: int = 3, temperature: float = 0.3,
                 max_tokens: int = 400):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") \
            or os.environ.get("SARVAM_API_KEY")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or "https://api.sarvam.ai/v1").rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "sarvam-105b")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.available = bool(self.api_key)
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s)) \
            if self.available else None

    def generate(self, query: str, evidence_texts: list[str], lang: str,
                 history=None) -> Generation:
        if not self.available:
            raise GenerationError("no LLM API key configured")

        messages = [
            {"role": "system", "content": system_prompt(lang)},
            {"role": "user", "content": build_user_prompt(
                sanitize_for_prompt(query), evidence_texts, history)},
        ]

        state = {"attempts": 0}

        @retry(stop=stop_after_attempt(self.max_attempts),
               wait=wait_exponential(multiplier=0.3, max=3.0),
               retry=retry_if_exception_type(GenerationTransient),
               reraise=True)
        def _attempt() -> str:
            state["attempts"] += 1
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model, "messages": messages,
                          "temperature": self.temperature,
                          "max_tokens": self.max_tokens},
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise GenerationTransient(str(exc)) from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                raise GenerationTransient(
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                raise GenerationError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise GenerationError(
                    f"unexpected response shape: {str(data)[:200]}") from exc

        t0 = time.perf_counter()
        content = _attempt()
        ms = (time.perf_counter() - t0) * 1000

        parsed = _parse_structured(content)
        return Generation(
            answer=str(parsed.get("answer", "")).strip(),
            sources_used=[int(s) for s in parsed.get("sources_used", [])
                          if str(s).isdigit()],
            sufficient=bool(parsed.get("sufficient", True)),
            latency_ms=ms, raw=content, attempts=state["attempts"],
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


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
