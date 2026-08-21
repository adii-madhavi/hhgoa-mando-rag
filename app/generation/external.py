"""Independent, uncited general-knowledge LLM fallback."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from app.generation.generator import Generation, GenerationError
from app.generation.llm_client import (ChatClient, LLMError, LLMSchemaError,
                                       LLMTransient)
from app.guardrails.input import sanitize_for_prompt


class ExternalOutput(BaseModel):
    model_config = {"extra": "forbid"}
    answer: str


class ExternalLLMGenerator:
    """Uses only EXTERNAL_LLM_* configuration, never primary credentials."""

    def __init__(self, *, api_key: str, base_url: str, model: str,
                 provider: str | None = None, timeout_s: float = 15.0,
                 max_tokens: int = 1200, max_schema_attempts: int = 2,
                 client: ChatClient | None = None):
        if not all((api_key, base_url, model)):
            raise ValueError("external API key, base URL and model are required")
        self.provider = provider
        self.client = client or ChatClient(
            api_key=api_key, base_url=base_url, model=model,
            timeout_s=timeout_s)
        self.max_tokens = max_tokens
        self.max_schema_attempts = max_schema_attempts

    @property
    def available(self) -> bool:
        return self.client.available

    def generate(self, query: str, lang: str,
                 answer_mode: str = "detailed") -> Generation:
        language = {"en": "English", "hi": "Hindi", "mr": "Marathi",
                    "kok": "Konkani"}.get(lang, "English")
        brevity = ("Use one short sentence." if answer_mode == "fast" else
                   "Answer concisely, normally in two to four sentences.")
        messages = [
            {"role": "system", "content":
             "Answer from general knowledge. Do not claim that the answer is "
             "verified against supplied sources or a corpus. Do not provide "
             "chain-of-thought or hidden reasoning. Return only JSON with the "
             "single key 'answer'."},
            {"role": "user", "content":
             f"Reply in {language}. {brevity}\nQuestion: "
             f"{sanitize_for_prompt(query)}"},
        ]

        def validate(payload: dict) -> ExternalOutput:
            try:
                out = ExternalOutput.model_validate(payload)
            except ValidationError as exc:
                raise LLMSchemaError(str(exc)) from exc
            if not out.answer.strip():
                raise LLMSchemaError("external answer is empty")
            return out

        try:
            parsed, meta = self.client.chat_json(
                messages, validate, temperature=0.2,
                max_tokens=(min(self.max_tokens, 400)
                            if answer_mode == "fast" else self.max_tokens),
                max_schema_attempts=self.max_schema_attempts,
                response_format={"type": "json_object"})
        except (LLMSchemaError, LLMTransient, LLMError) as exc:
            raise GenerationError(f"{type(exc).__name__}: {exc}") from exc
        return Generation(answer=parsed.answer.strip(), sources_used=[],
                          sufficient=True, latency_ms=meta.latency_ms,
                          attempts=meta.http_attempts,
                          schema_attempts=meta.schema_attempts)

    def close(self) -> None:
        self.client.close()
