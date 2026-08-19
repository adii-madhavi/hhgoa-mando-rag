"""
Shared OpenAI-compatible chat client.

Extracted so the answerability judge, the generator and the grounding verifier
share one implementation of the parts that are easy to get subtly wrong:
retry classification, timeout handling, and tolerant JSON recovery.

`app/generation/generator.py` is deliberately left untouched -- it works and is
covered by tests. New callers use this; migrating the generator is a Phase 3
concern.

Retry policy
------------
429 and 5xx are transient and retried with exponential backoff. 4xx is our bug
(bad key, bad model name, malformed request) and fails immediately -- retrying
a 400 just burns the latency budget three times over.

Backoff is sized for RATE LIMITS, not jitter. A 600-call calibration at
concurrency 6 hit HTTP 429 on 41 calls with the original 0.3s/3s-max policy:
too fast and too few attempts to outlast a rate-limit window. Now 2s base,
30s cap, 5 attempts.

A malformed JSON *body* is a third case: the HTTP call succeeded but the model
did not follow the schema. That is retried separately, because it is usually
fixed by simply asking again, and the retry prompt says what was wrong.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)


class LLMError(RuntimeError):
    """Non-retryable: bad credentials, bad request, unusable response shape."""


class LLMTransient(LLMError):
    """Retryable: timeout, connection error, 429, 5xx."""


class LLMSchemaError(LLMError):
    """The call succeeded but the body did not match the expected schema."""


@dataclass
class ChatResult:
    content: str
    latency_ms: float
    attempts: int = 1
    http_attempts: int = 1
    schema_attempts: int = 1
    model: str = ""
    usage: dict = field(default_factory=dict)


def extract_json(content: str) -> dict:
    """
    Recover a JSON object from a model reply.

    Models wrap JSON in markdown fences or add a sentence of preamble often
    enough that failing hard would turn a cosmetic slip into a request failure.
    We recover the object; genuinely unparseable output raises LLMSchemaError
    so the caller can retry with feedback.
    """
    text = (content or "").strip()

    if text.startswith("```"):
        body = text[3:]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        end = body.rfind("```")
        text = (body[:end] if end != -1 else body).strip()

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

    raise LLMSchemaError(f"no JSON object in reply: {content[:200]!r}")


class ChatClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout_s: float = 15.0,
                 max_http_attempts: int = 5):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") \
            or os.environ.get("SARVAM_API_KEY")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or "https://api.sarvam.ai/v1").rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "sarvam-105b")
        self.timeout_s = timeout_s
        self.max_http_attempts = max_http_attempts
        self.available = bool(self.api_key)
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s)) \
            if self.available else None

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 500,
             response_format: dict | None = None) -> ChatResult:
        if not self.available:
            raise LLMError("no LLM API key configured (set LLM_API_KEY)")

        state = {"attempts": 0}

        @retry(stop=stop_after_attempt(self.max_http_attempts),
               wait=wait_exponential(multiplier=2.0, max=30.0),
               retry=retry_if_exception_type(LLMTransient), reraise=True)
        def _attempt() -> dict:
            state["attempts"] += 1
            payload = {"model": self.model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}
            if response_format:
                # Providers that support it enforce the schema server-side; the
                # ones that ignore it still get the instruction in the prompt,
                # and validation catches the difference either way.
                payload["response_format"] = response_format
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise LLMTransient(f"{type(exc).__name__}: {exc}") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                raise LLMTransient(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()

        t0 = time.perf_counter()
        data = _attempt()
        ms = (time.perf_counter() - t0) * 1000

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"unexpected response shape: {str(data)[:200]}") from exc

        # REASONING MODELS: sarvam-105b (and similar) emit `reasoning_content`
        # first and only then the answer. If max_tokens runs out mid-reasoning
        # the API returns finish_reason="length" with content=None -- a
        # perfectly well-formed 200 response carrying no answer. Without this
        # branch that surfaced as an opaque "no JSON object in reply: None"
        # schema failure, which is exactly the wrong diagnosis: the fix is a
        # bigger token budget, not a better prompt. Measured: at max_tokens=300
        # the model spent all 300 on reasoning and returned None; at 1500 it
        # used 901 and returned valid JSON.
        if content is None:
            finish = choice.get("finish_reason")
            reasoning = (choice.get("message") or {}).get("reasoning_content")
            raise LLMTransient(
                f"model returned no content (finish_reason={finish!r}, "
                f"reasoning_content={len(reasoning) if reasoning else 0} chars"
                f"). If finish_reason is 'length', raise max_tokens -- this is "
                f"a reasoning model and the budget was consumed before it "
                f"produced an answer.")

        return ChatResult(content=content, latency_ms=ms,
                          attempts=state["attempts"],
                          http_attempts=state["attempts"],
                          model=self.model, usage=data.get("usage") or {})

    def chat_json(self, messages: list[dict], validate,
                  temperature: float = 0.0, max_tokens: int = 500,
                  max_schema_attempts: int = 2,
                  response_format: dict | None = None):
        """
        Call the model and validate the reply against `validate`, a callable
        that takes a dict and returns a parsed object or raises.

        On a schema failure we retry ONCE more with the error appended as a
        corrective user turn, which is far more effective than repeating the
        identical prompt. Total cost stays bounded by max_schema_attempts.
        """
        convo = list(messages)
        last_error: Exception | None = None
        total_ms = 0.0

        for attempt in range(1, max_schema_attempts + 1):
            result = self.chat(convo, temperature=temperature,
                               max_tokens=max_tokens,
                               response_format=response_format)
            total_ms += result.latency_ms
            try:
                parsed = validate(extract_json(result.content))
                result.latency_ms = total_ms
                result.schema_attempts = attempt
                return parsed, result
            except Exception as exc:                       # schema violation
                last_error = exc
                if attempt >= max_schema_attempts:
                    break
                convo = convo + [
                    {"role": "assistant", "content": result.content},
                    {"role": "user",
                     "content": (f"That reply was rejected: {exc}. "
                                 f"Reply again with ONLY a valid JSON object "
                                 f"matching the required schema. No prose, no "
                                 f"markdown fences.")},
                ]

        raise LLMSchemaError(
            f"schema validation failed after {max_schema_attempts} attempts: "
            f"{last_error}")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
