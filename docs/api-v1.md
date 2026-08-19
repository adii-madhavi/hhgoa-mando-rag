# MANDO Public API — v1

This is the frozen boundary between the MANDO backend and any frontend
(including the separately-developed one). Once published, fields are not
renamed or removed without a v2 bump — see `app/api/v1/schemas.py` for the
enforcement comment.

**Never expose:** API keys, model names, internal guardrail scores, per-stage
timing, or anything from `app.schemas.RAGResponse` that isn't explicitly
listed below. This is verified by `tests/test_api_v1.py`, not just documented.

Base path: `/api`. Internal endpoints (`/ask`, `/ask/text`, `/health`,
`/config` — no `/api` prefix) exist separately for this project's own
evaluation scripts and expose full internal detail; the frontend must never
call those.

---

## Contents

- [Authentication](#authentication)
- [CORS](#cors)
- [Languages](#languages)
- [Voice field](#voice-field)
- [POST /api/text/query](#post-apitextquery)
- [POST /api/voice/query](#post-apivoicequery)
- [GET /api/health](#get-apihealth)
- [GET /api/config](#get-apiconfig)
- [Streaming (SSE) format](#streaming-sse-format)
- [Error responses](#error-responses)
- [What is NOT implemented yet](#what-is-not-implemented-yet)

---

## Authentication

None. The frontend never sends or sees a Sarvam or LLM API key — those live
only in the backend's `.env` and are read from `app.config.CONFIG`, never
serialized into any response.

## CORS

`allow_origins=["*"]` for the whole app during development (no frontend
origin is locked down yet — do not read anything into this beyond "development
convenience"). Revisit before any production deployment.

## Languages

Two lists, not one — this distinction is load-bearing elsewhere in the project
(`EXPERIMENTS.md`, `app/schemas.py`) and is preserved here:

| | meaning |
|---|---|
| `supported_languages` | a user may speak/select it (`en`, `hi`, `mr`, `kok`) |
| `evaluated_languages` | subset with labelled benchmark data (`en`, `hi`, `mr`) |

Konkani (`kok`) is supported but not evaluated: no MSMARCO-XI ground truth
exists for it, and Sarvam TTS (`bulbul:v3`) has no `kok-IN` voice — see
`GET /api/config` below and `app/schemas.LANG_CAPABILITIES`.

Requests accept `"auto"` in addition to the four codes, to mean "detect from
the input."

## Voice field

`voice`: `"male"` | `"female"`. This selects a TTS speaker only — it has no
effect on retrieval, generation, or the answer text. If TTS is unavailable
(no credentials, or the language has no TTS voice), `audio_base64` in the
response is `null` regardless of which voice was requested; the frontend
should fall back to displaying `answer` as text.

---

## POST /api/text/query

**Request** (`application/json`)

```json
{
  "query": "how fast does an eagle travel",
  "language": "en",
  "session_id": "optional-client-id"
}
```

| field | type | required | notes |
|---|---|---|---|
| `query` | string | yes | 1–2000 chars |
| `language` | string | no | default `"auto"`; one of `en`/`hi`/`mr`/`kok`/`auto` |
| `session_id` | string \| null | no | opaque, echoed back only — see [Sessions](#sessions-are-stateless) |

**Response** (`200`, `application/json` by default)

```json
{
  "answer": "Eagles fly at 30 to 55 mph and can dive at over 100 mph.",
  "sources": [
    {"text": "Eagles fly 30 to 55 mph and dive at over 100 mph.",
     "language": "en", "relevance": 0.91}
  ],
  "language": "en",
  "latency": {
    "retrieval_ms": 42.1,
    "generation_ms": 2380.6,
    "total_ms": 2422.7,
    "streamed": false
  },
  "guardrail": {"grounded": true, "refused": false, "refusal_reason": null},
  "session_id": "optional-client-id"
}
```

**Refusal example** (still `200` — a correct refusal is a successful call,
not an error):

```json
{
  "answer": "I looked through my sources, but I couldn't find enough there to give you a reliable answer on that one.",
  "sources": [],
  "language": "en",
  "latency": {"retrieval_ms": 38.2, "generation_ms": null,
             "total_ms": 38.2, "streamed": false},
  "guardrail": {"grounded": null, "refused": true,
               "refusal_reason": "insufficient_evidence"},
  "session_id": null
}
```

**cURL**

```bash
curl -X POST http://localhost:8000/api/text/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how fast does an eagle travel", "language": "en"}'
```

**Streaming** (SSE) — same endpoint, different `Accept` header:

```bash
curl -N -X POST http://localhost:8000/api/text/query \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "how fast does an eagle travel", "language": "en"}'
```

---

## POST /api/voice/query

`multipart/form-data`, not base64-in-JSON — the frontend uploads a recorded
audio file directly.

| field | type | required | notes |
|---|---|---|---|
| `audio` | file | yes | any format Sarvam STT accepts (e.g. WAV) |
| `language` | form field | no | default `"auto"` |
| `voice` | form field | no | default `"male"`; `"male"` \| `"female"` |
| `session_id` | form field | no | echoed back only |

**Response** (`200`, `application/json` by default) — same shape as
`TextQueryResponse` plus:

```json
{
  "transcript": "how fast does an eagle travel",
  "audio_base64": "UklGRi4AAABXQVZFZm10...",
  "audio_stream_url": null,
  "answer": "...",
  "sources": [...],
  "language": "en",
  "latency": {...},
  "guardrail": {...},
  "session_id": null
}
```

`audio_base64` is `null` when TTS is unavailable — check `guardrail.refused`
and `answer` in that case; the text answer is still fully valid.

**cURL**

```bash
curl -X POST http://localhost:8000/api/voice/query \
  -F "audio=@question.wav;type=audio/wav" \
  -F "language=en" \
  -F "voice=female"
```

**Streaming**

```bash
curl -N -X POST http://localhost:8000/api/voice/query \
  -H "Accept: text/event-stream" \
  -F "audio=@question.wav;type=audio/wav" \
  -F "language=en"
```

---

## GET /api/health

```bash
curl http://localhost:8000/api/health
```

```json
{
  "ok": true,
  "index_loaded": true,
  "llm_available": true,
  "stt_available": false,
  "tts_available": false
}
```

Booleans only — no model names, no error strings, no credential status
detail. (The internal `/health` endpoint has that detail for operators; this
one doesn't.)

## GET /api/config

```bash
curl http://localhost:8000/api/config
```

```json
{
  "supported_languages": ["en", "hi", "mr", "kok"],
  "evaluated_languages": ["en", "hi", "mr"],
  "available_voices": [],
  "streaming_supported": true,
  "feature_flags": {
    "llm_generation": true,
    "speech_to_text": false,
    "text_to_speech": false,
    "answerability_judge": false
  }
}
```

`available_voices` is `[]` whenever `text_to_speech` is `false` — never a
fabricated list. This reflects real backend capability at call time, not a
static declaration.

---

## Streaming (SSE) format

Three event types, in order:

```
event: meta
data: {"language": "en", "guardrail": {...}}

event: chunk
data: {"text": "Eagles "}

event: chunk
data: {"text": "fly "}

...

event: done
data: {"answer": "Eagles fly ...", "sources": [...], "latency": {...}, "session_id": null}
```

If the request was refused, `meta` is still sent, then `done` directly (no
`chunk` events) with the refusal text as `answer`.

**Streaming honesty note.** Today the SSE path runs the full guarded pipeline
once (retrieval → guardrails → generation, all synchronous), then emits the
completed, already-validated answer as word-chunked `chunk` events. This gives
the frontend a stable streaming *contract* to build against now. It is **not**
yet true mid-generation token streaming — that requires wiring streaming
through `RAGPipeline` itself, which is a separate, larger change deliberately
out of scope here (see `app/api/v1/text.py` docstring, and
`app/generation/generator.LLMGenerator.generate_stream`, which *does* support
real token streaming at the generator level and is benchmarked in
`EXPERIMENTS.md` E18 — pipeline-level wiring is the next step, not yet done).

---

## Error responses

| status | when |
|---|---|
| `422` | invalid `language`, invalid `voice`, empty/missing `query`, empty/missing `audio` |
| `503` | index not loaded (check `/api/health` first) |
| `500` | unhandled internal error (never leaks a traceback or internal detail) |

`422` bodies use FastAPI's standard validation-error shape:

```json
{"detail": [{"type": "value_error", "loc": ["body", "language"],
            "msg": "Value error, language must be 'auto' or one of ..."}]}
```

`503`/`500` bodies:

```json
{"error": "...", "detail": "..."}
```

---

## Sessions are stateless

`session_id` is accepted and echoed back in every response, but the backend
does **not** store or look up conversation history by it. The frontend owns
conversation state and is expected to resend relevant context in a future
API revision. This is a deliberate simplicity choice, not an oversight — see
`app/api/v1/schemas.TextQueryRequest` docstring.

## What is NOT implemented yet

- **True mid-generation SSE token streaming** through the guarded pipeline
  (see streaming note above).
- **`audio_stream_url`** — reserved in the schema, always `null` today.
  Sarvam TTS streaming is not yet integrated (Phase 5/6, blocked on
  `SARVAM_API_KEY`).
- **Server-side session/context.**
- **Production CORS configuration** — currently wide open for development.
