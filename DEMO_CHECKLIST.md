# HHGoa — Demo Checklist

Run `uvicorn app.main:app --reload` and open <http://localhost:8000>.
Each item is what to actually click/say and what a correct result looks like.

**Sarvam account status at time of writing: zero credits for BOTH LLM
generation and TTS** (`HTTP 402 insufficient_quota_error` on
`api.sarvam.ai/v1/chat/completions`, confirmed live in the prior pass).
Items marked **BLOCKED BY CREDITS** cannot show a real grounded/spoken
answer until the account is topped up -- they will instead show a clean,
correctly-shaped refusal. That refusal behavior itself IS demonstrable and
is a legitimate thing to show (it proves the failure path works).

| # | Check | How | Expected | Status |
|---|---|---|---|---|
| 1 | English text | Ask "What is a corporation?" | Grounded answer, sources shown, `guardrail.refused=false` | BLOCKED BY CREDITS -- will show a clean refusal instead (verified live) |
| 2 | Hindi text | Select Hindi, ask "निगम क्या है?" | Answer in Hindi, `language: "hi"` | BLOCKED BY CREDITS (same reason) |
| 3 | Marathi text | Select Marathi, ask a Marathi question | Answer in Marathi | NOT LIVE-TESTED this pass (would 402 identically -- not repeated to save credits) |
| 4 | Konkani handling | Select Konkani, ask any question | Either a text answer with `audio_unavailable_reason` explaining no Sarvam TTS voice exists for `kok`, or (pre-credits) a clean refusal -- **never** Marathi audio substituted | Code path VERIFIED by unit tests (`TestRefusalAudio::test_konkani_refusal_gets_no_audio_and_no_marathi_substitution`); not live-tested this pass |
| 5 | Voice round trip | Mic button, speak a question | Transcript shown, real Sarvam STT text, spoken answer plays | BLOCKED BY CREDITS (STT itself still has credits per earlier checks, but the RAG answer step needs LLM credits) |
| 6 | Streaming | `curl -N -H "Accept: text/event-stream" ...` or open dev tools Network tab | `event: meta`, then `event: chunk` frames, then `event: done`; no raw JSON/model internals visible | VERIFIED by test suite (`tests/test_api_v1.py::TestSSE`, `tests/test_streaming.py`) -- SSE framing itself does not need Sarvam credits to verify structurally |
| 7 | Refusal / unanswerable | Ask something absurd, e.g. "population of Mars colonies in year 3000" | `guardrail.refused=true`, an honest apology answer, no fabricated fact, sources may still be listed (retrieved but judged insufficient) | VERIFIED live this pass (also true for every query right now, since the account is out of credits -- see note above) |
| 8 | No internal fields leak | Inspect any `/api/*` JSON response | No `judge_available`, `retrieval_confidence`, `grounding_score`, `request_id`, `warnings`, etc. | VERIFIED by test suite (`TestSchemaIsolation`) |
| 9 | No secrets in responses/frontend | Inspect `/`, `/api/health`, `/api/config` | No API key substrings anywhere | VERIFIED by test suite (`TestNoSecretExposure`, `TestFrontendIntegration::test_frontend_never_embeds_api_keys`) |
| 10 | Health/config accuracy | `GET /api/health`, `GET /api/config` | `answerability_judge: true` (fixed this pass -- was `false`), capability flags reflect real key presence | VERIFIED live this pass |

## What to say if asked "why isn't it answering for real right now"

"The Sarvam account used for both speech and generation has run out of
credits -- you're seeing the system's own honest degradation: it refuses
cleanly rather than making something up, exactly as designed. Every other
piece of the pipeline (retrieval, judge, grounding, guardrails, streaming,
the API contract) is fully wired and unit/integration tested; only the live
external call is blocked until the account is topped up."
