# MANDO — TODO

Source of truth alongside `PROJECT_STATE.md`. Update both after every major task.
Deadline: 2026-08-22 23:59.

Phases 1-3 COMPLETE. `LLM_API_KEY` present and working.

---

## 🔴 STATUS — product knowledge + external-answer layer merged and verified offline

**DONE this pass (ZIP merge + credit-safe release verification):** the
supplied ZIP was byte-identical to the repo at `6e9b914` (already merged in
an earlier session) -- nothing to merge. Verified via code reading + the
full offline test suite that everything in that commit is wired correctly:
independent `EXTERNAL_LLM_*` credentials (never reuse Sarvam/primary LLM
key), the corpus/goa_knowledge/external `answer_origin` routing, Fast/
Detailed modes that never skip grounding, ElevenLabs/Sarvam STT provider
selection, and the runtime P50/P70 telemetry (bounded history, minimum
sample count, no query content stored). Two small corrections applied:
README's stale `--strategy hierarchical` index-build command corrected to
`--strategy fixed` (matches `CHUNK_STRATEGY` production default), and the
unused `CROSS_LINGUAL` config flag's default flipped from `True` to `False`
(the hypothesis it names was tested and rejected -- see EXPERIMENTS.md;
the flag is read nowhere in `app/` at runtime). Both pinned with new tests
in `tests/test_config.py`. 392/392 tests passing (was 377 -- the merged
commit's own `tests/test_product_answering.py` plus these 2 new tests).
Full repository secret scan clean; `.env` confirmed untracked and ignored;
no provider key names appear in `frontend/index.html`.

**NEXT ACTION for a future agent/session -- READ THIS FIRST:**

- [ ] Nothing is required before release. The items below are explicitly
      OPTIONAL or BLOCKED, not blockers. Do not treat any of them as a
      default next step -- see the credit-protection policy immediately
      below before running ANYTHING that calls Sarvam, ElevenLabs, or an
      external LLM.

---

### [OPTIONAL / BLOCKED — DO NOT RUN AUTOMATICALLY]

**Large Sarvam voice-loop benchmark (n>=100).**

Reason: requires external API quota and can consume significant LLM/STT/TTS
credits for a marginal demo improvement -- this was already tried once
(Phase 10, `EXPERIMENTS.md` E20) and burned through the account's credits
without reaching n=100. **The user has explicitly said this should no
longer be considered a requirement for the project to be finished.**

Existing benchmark/evaluation evidence (E1-E20) must be preserved exactly,
never rerun to "refresh" it.

Do NOT rerun this benchmark unless the user explicitly requests it in a
future session.

### [OPTIONAL LIVE PROVIDER VALIDATION]

Use the offline/mocked regression suite (`python -m pytest tests/ -q`,
392 tests) during normal development -- it already covers corpus, Goa
knowledge, external fallback, Fast/Detailed, and STT-provider-selection
behavior without spending any quota.

Before a final demo/deployment, IF usable free quota is confirmed to exist
(check for `HTTP 402`/quota-exhausted errors on ONE call first, do not
assume), perform AT MOST:
1. one corpus text query
2. one Goa knowledge query ("What is the capital of Goa?")
3. one external-fallback query (something outside corpus/Goa knowledge)
4. one voice/STT query

No repeated runs unless diagnosing a specific, confirmed defect.

### [VOICE CREDIT POLICY]

Normal "Listen" functionality in the frontend uses the browser's own
`SpeechSynthesis` API on the answer already in memory -- it must NEVER
re-run retrieval, call the LLM, call the external LLM, or call backend
TTS (Sarvam/ElevenLabs). Verified in `tests/test_product_answering.py::
test_frontend_listen_reuses_answer_and_paid_tts_is_not_default`.

Backend TTS remains available ONLY for an explicitly requested voice/audio
response (`want_audio=true` on `/api/voice/query`). Do not wire it into any
other path.

### [EXPERIMENT POLICY]

Do NOT automatically rerun:
- the 600-call answerability calibration (E15)
- the n>=100 voice-loop benchmark (E20)
- chunking experiments (E3-E7)
- embedding experiments
- MMR/reranking experiments
- cross-lingual experiments (rejected -- see CROSS_LINGUAL note above)
- historical generation benchmarks (E16-E18)

The E-series experiment results in `EXPERIMENTS.md` are historical evidence
and must remain unchanged. If a genuine regression is suspected, diagnose
with the smallest possible offline/mocked test first.

---

**REQUIRED BEFORE RELEASE:** none currently known -- code, tests, and
secret hygiene are all verified as of this pass.

**BLOCKED BY EXTERNAL CREDENTIALS (informational, not a blocker for
release):** the single quota-probe call made in this pass (one real
`/api/text/query`, per the credit-safe smoke-test policy above) returned
`HTTP 403 invalid_api_key_error: "Invalid or missing authentication
credentials"` from `api.sarvam.ai/v1/chat/completions` -- **not** the
`HTTP 402 insufficient_quota_error` seen in earlier passes. This suggests
the `SARVAM_API_KEY` in `.env` may have been rotated/invalidated since the
last working session, which is a DIFFERENT problem than running out of
credits and needs a fresh key, not a top-up. No further live calls were
made once this was seen (`EXTERNAL_LLM_*` and `ELEVENLABS_API_KEY` are also
both unconfigured -- `CONFIG.has_external_llm` and the ElevenLabs STT path
are both `False`, confirmed via the boolean flags only, no values printed).
This blocks only live demo verification, not the release itself -- the
code path for a provider outage/auth failure is itself tested and verified
(see `test_product_answering.py` and `TestNoSecretExposure`/degradation
tests elsewhere in the suite): the pipeline degraded correctly (clean
`internal_error` refusal, no crash, no leaked credential).

---

## Historical — Sarvam credentials, then pipeline-level streaming

## Older — Sarvam credentials, then pipeline-level streaming

Streaming is done and measured (E18): TTFT P50 246ms, **TTFV P50 391ms**
(10.7x better perceived latency than the 4,175ms non-streamed generation), a
hard 20s deadline enforced, and clean degradation on 4/12 Marathi timeouts (all
before any visible text -- never a broken partial answer reaching a user).

Public API v1 is done (`app/api/v1/`, `docs/api-v1.md`): thin adapter over
RAGPipeline, 4 endpoints, 35 isolation/validation tests. Frontend is now
OUT OF SCOPE for backend work -- it will be built separately against that doc.

Remaining, in order:

- [x] `SARVAM_API_KEY` added and live-verified (sarvam_live_check.py passed).
- [x] E19: one small end-to-end voice validation run (see below). STOPPING
      per instruction -- awaiting review before any further phase.
- [ ] **Pipeline-level streaming.** `LLMGenerator.generate_stream()` is fully
      guarded and works; `RAGPipeline.run()` is still synchronous end-to-end.
      The public API's SSE endpoint currently streams the COMPLETED answer
      word-chunked (documented honestly in docs/api-v1.md), not true
      mid-generation tokens through the guard chain. Wiring that through is
      the next real latency-perception improvement once STT/TTS exist to make
      it worth doing for voice.
- [ ] **Generation tail** (P100 9.4s in E18, was 65s in E17) -- improved by the
      hard deadline but not eliminated. Revisit if a faster/cheaper model
      becomes available.

Honest position on the 200 ms target, unchanged: only the **retrieval core**
(22-50 ms P50) is inside it. TTFV brings PERCEIVED latency to 391ms, which is
much closer but is not the same claim as "under 200ms" and is never presented
as such. Generation TOTAL is still ~4.2s P50. These are reported as separate
numbers and never merged.

---

## ✅ PHASE 3 DONE

- [x] LLMGenerator migrated to shared ChatClient; duplicate httpx/tenacity gone
- [x] max_tokens 400 -> 3000 (reasoning-model truncation fixed)
- [x] Strict Pydantic structured output + corrective retry
- [x] Citation validation; out-of-range sources dropped into invalid_sources
- [x] Configurable PersonaConfig; Mando default; prompt is voice-invariant
      (male/female addable without touching the RAG core)
- [x] Konkani honesty clause - no fabricated fluency claim
- [x] Judge async by default + deterministic VerdictCache
- [x] Same-language enforcement (warn, never refuse a correct answer)
- [x] 237 tests passing (was 177)

## ✅ STREAMING + PUBLIC API DONE

- [x] `ChatClient.stream_chat()` -- SSE parsing, reasoning_content structurally excluded
- [x] `StreamingAnswerExtractor` -- incremental JSON->prose decode, verified at
      6 chunk sizes x 2 escape modes x en/hi (Devanagari \uXXXX bug found+fixed)
- [x] `LLMGenerator.generate_stream()` -- hard 20s wall-clock deadline,
      citations still validated, incomplete-answer marking
- [x] 8-field latency instrumentation (TTFT/TTFV/generation/total/etc)
- [x] Benchmarked 36 queries (E18): TTFT P50 246ms, TTFV P50 391ms
- [x] `app/api/v1/` -- schemas.py, translate.py, router.py, text.py, voice.py, meta.py
- [x] 4 public endpoints: POST /api/text/query, POST /api/voice/query,
      GET /api/health, GET /api/config -- content-negotiated JSON/SSE
- [x] `docs/api-v1.md` -- full contract, examples, SSE format, error responses
- [x] python-multipart added (was missing; needed for UploadFile/Form)
- [x] 35 new tests: schema isolation, secret-leak checks, validation, SSE, multipart
- [x] 321 tests passing (was 237)

## ✅ VOICE PIPELINE WIRING DONE (mock-tested; live call still blocked)

- [x] Confirmed `app/pipeline.py` already implements Sarvam STT -> RAGPipeline
      -> guarded generation -> Sarvam TTS end-to-end (built in an earlier
      phase; this task found it structurally complete, just untested/unrun)
- [x] `tests/test_voice.py` -- 39 new tests:
      - SarvamSTT: multipart request shape, auto-language field omission,
        429/5xx retry, 4xx fail-fast, invalid/empty base64 handling, offline
        mode (no key)
      - SarvamTTS: JSON request shape, Konkani rejection (no Marathi
        substitution), voice->speaker mapping, digit-grouping, truncation,
        429/5xx retry, 4xx fail-fast, empty-audios-list error
      - Full pipeline round trip with scripted STT+TTS doubles: audio in ->
        transcript -> guarded generation -> audio out; Stage.stt/Stage.tts
        latency recorded; want_audio=False skips TTS; text path still
        bypasses STT
      - Degradation: STT failure/offline -> clean refusal, never a crash;
        TTS failure -> NON-fatal, text answer still returned with a warning
      - Secret hygiene: API key never appears in a repr or an error message
- [x] `evaluation/sarvam_live_check.py` -- small 2-call live smoke test
      (TTS synthesize -> STT transcribe the result back). Exits cleanly with
      no key, matching this project's "never fabricate a result" convention.
      Verified: currently exits correctly (no key set), makes zero network
      calls in that path.
- [x] 360 tests passing (was 321)
- [x] **Live call done.** See E19 below.

## E19 — small end-to-end voice validation (n=1 per query, NOT a benchmark)

`python evaluation/voice_e2e_check.py` -- 5 real pipeline runs: 3 fixed
queries (en/hi/mr, first answerable MSMARCO-XI query per language, Konkani
excluded per instruction) + male/female voice check. Real Sarvam TTS
synthesized each query's audio INPUT (no microphone), real Sarvam STT
transcribed it, real RAGPipeline ran retrieval+guardrails+generation, real
Sarvam TTS synthesized the spoken OUTPUT where reached.

| query | transcript matched | refused | grounded | valid audio out | e2e latency |
|---|---|---|---|---|---|
| en | yes | No | True | **Yes** | 15,041 ms |
| hi | yes | No | True | **Yes** | 35,046 ms |
| mr | yes | Yes (ungrounded_answer) | False | No (never reached TTS) | 5,190 ms |

Male/female voice check (same English query): both requests reached the
pipeline with the correct `voice` param; both were refused (ungrounded_answer,
same numeric-hard-gate cause as mr above) before reaching the TTS stage, so
neither produced audio in THIS run -- not a voice-selection bug, the voice
field itself was passed and used correctly.

**Diagnosed, not "fixed":** mr/male/female all failed the EXISTING numeric
grounding hard-gate (`app/guardrails/grounding.py`, the same gate that has
caught invented numbers since early development) -- the LLM (temperature=0.3)
volunteered a digit not present in evidence on 3 of 5 runs. This is the
guardrail working correctly, not an integration bug; left untouched per "never
silently downgrade grounding standards."

**One real script bug found and fixed** (in the test script, not the
pipeline): `pipeline.shutdown()` was called before the voice-check queries,
prematurely closing the async judge pool for those two runs. Fixed (shutdown
moved to the end). Confirmed this had zero effect on any reported result --
every query in the script is a fresh cache miss, so the async judge never
gates a first-time query regardless of whether the pool is alive.

**One real product gap found, NOT fixed (flagged for review, out of scope):**
`RAGPipeline` only reaches the TTS stage on the success path. Every
`_refuse()` return exits before stage 11, so a refused voice query produces
NO spoken audio at all -- silence, even though `resp.answer` already holds
correct, localised refusal text. Whether refusals should be spoken is a
product decision, not a bug fix, and this task was explicitly scoped to
validate + diagnose, not redesign.

360/360 tests passing (5 pre-existing failures were test-isolation bugs --
"offline" tests assumed no `SARVAM_API_KEY` would ever be in the environment;
now that one legitimately is, those 5 tests were fixed with
`monkeypatch.delenv` to force the true offline path regardless of the host
machine's `.env`).

**STOPPING HERE per instruction. Awaiting review.**

---

## ✅ DONE

- [x] Phase 0 — dataset inspection, selective download, corpus build, index
- [x] Chunking: 6 strategies benchmarked → fixed baseline won
- [x] Retrieval: dense/BM25/hybrid ablated → dense-only ships
- [x] Cross-lingual routing tested → refuted, not enabled
- [x] Language detection: 95.4% overall, measured on 6,000 labelled cases
- [x] Guardrails: input (7/7 adversarial) · cosine relevance · grounding
- [x] Threshold calibration on the 780 labelled no-answer queries
- [x] Latency: 137.45 ms P50 across 360 requests, per-stage breakdown
- [x] Orchestration harness: structured I/O, retries, request IDs, stage timing
- [x] FastAPI + push-to-talk frontend (text path working, voice path gated)
- [x] Docker / compose / requirements / .env.example
- [x] **Phase 1 — LLM answerability judge** (built, 66 tests, NOT measured)
- [x] Konkani reframed as product-not-benchmark language, enforced in code
- [x] Pushed to https://github.com/adii-madhavi/hhgoa-mando-rag (`main`)
- [x] `requirements.txt` regenerated from AST scan of real imports; `datasets`
      and `pandas` dropped (unused); all 16 packages pinned and verified
- [x] **Phase 2 calibration run #2 (600 calls) - VALID, GATE PASSED.**
      trustworthy in all 3 languages; 0 rate-limit failures. C balAcc
      0.7250/0.6414/0.6515 vs A 0.5700/0.5808/0.6061; false answers
      0.80->0.16 en, 0.64->0.24 hi, 0.55->0.17 mr. Latency P50 909ms.
- [x] **Phase 2 calibration run #1 (600 calls) - INVALID.** 103/600 judge
      calls failed and concentrated on one class by call order, inflating
      English false-answer 0.2034 -> 0.5300. Gate NOT applied, Phase 3 NOT
      unlocked. Root causes fixed (interleaving, backoff, token budget).
- [x] **Phase 2 smoke test (n=5/class) run live** — found and fixed: deprecated
      `sarvam-m` model, and reasoning-model `max_tokens` truncation returning
      `content=None`. 176 tests passing.

---

## Then, in order (each gated on the previous)

**Phase 4 — grounding verification**
- [ ] Structured `{grounded, supported_passage_ids, confidence}`
- [ ] Failed grounding → safe refusal, never the generated answer
- [ ] Hallucination / unsupported-claim tests

**Phase 5 — Sarvam STT** *(needs `SARVAM_API_KEY`)*
- [ ] Exercise the client against the live API (currently contract-verified only)
- [ ] Confirm each language actually works before claiming it
- [ ] Track `stt_ms`

**Phase 6 — TTS**
- [ ] Male/female voice selection (presentation only — answer must be identical)
- [ ] Konkani stays text-only; `bulbul:v3` has no `kok-IN`
- [ ] Track `tts_ms`

**Phase 7 — full voice loop**
- [ ] End-to-end audio in → audio out
- [ ] Separate `total_rag_ms` and `end_to_end_ms` — never merge them

**Phase 8 — UI**
- [ ] Character states: 🙂 idle · 👂 listening · 🤔 processing · 🗣️ speaking
- [ ] Language + voice selectors; capability-aware (no control the backend can't serve)

**Phase 9 — Mando character art** (fictional, not a real person)

**Phase 10 — voice latency benchmark**
- [ ] 100+ queries; report P50/P70/P100 for RAG core **and** full voice E2E
- [ ] State plainly that voice E2E will exceed 200 ms — two network round trips

**Phase 11 — final testing**
- [ ] Keep all 237 passing; add live-API failure/timeout/retry cases

---

## Repo hygiene

- [ ] Re-commit + push after **every** major task; keep `PROJECT_STATE.md` and
      `TODO.md` current in the same commit
- Excluded by `.gitignore`: `.env`, `.venv/`, `data/raw|processed|index|qdrant/`,
      `*.parquet`, `*.npy` — the 936 MB corpus and 206 MB index are rebuilt via
      `ingestion/`, not stored in git
- `experiments/*.json` **are** committed: they are the evidence behind every
      number in `EXPERIMENTS.md`

---

## Deliberately NOT doing

- ❌ Re-running completed retrieval experiments
- ❌ Changing `fixed` chunking, enabling BM25, or enabling cross-lingual routing
- ❌ Tuning another threshold without labelled evidence
- ❌ Reporting any Konkani benchmark metric (no labelled data exists)
- ❌ Emitting any number that was not measured
