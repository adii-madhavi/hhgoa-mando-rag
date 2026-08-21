# MANDO — Project State

## Current status — final correctness and demo-readiness pass

- Feature set is frozen. Baseline: `627316b` on `main`; this document is
  included in the successor correctness commit.
- Explicit `en`/`hi`/`mr` selection is authoritative. STT language_code
  is second; heuristic detection is used only when neither stronger signal
  exists.
- A confidently wrong grounded answer receives at most one corrective rewrite
  over the same query/evidence, then the final answer is grounded again.
  Konkani's existing honest fallback behavior is unchanged.
- Primary generation is independently configured by `LLM_API_KEY`,
  `LLM_BASE_URL`, and `LLM_MODEL`. External fallback uses only
  `EXTERNAL_LLM_*`.
- STT is selectable with `STT_PROVIDER=sarvam|elevenlabs`. Sarvam is optional;
  normal frontend Listen reuses the current answer through browser
  `SpeechSynthesis` and makes no paid provider call.
- Runtime P50/P70 needs five successful samples in one isolated channel/mode
  bucket; text/voice and Fast/Detailed samples never mix.
- Offline verification: **416/416 tests passed**.
- Bounded live verification: **5/5 user-level requests passed** on 2026-08-22:
  Groq corpus generation, curated Goa knowledge (Panaji + real source), Gemini
  external fallback (unverified, zero corpus evidence), explicit Marathi
  (detected Marathi), and ElevenLabs STT voice.
- Current code blocker: none. Deployment still requires operator-managed
  secrets and the normal deployment/demo checklist.

Last updated: 2026-08-22 · Deadline: 2026-08-22 23:59
Repo: https://github.com/adii-madhavi/hhgoa-mando-rag (branch `main`)

The older readiness snapshots and E-series measurements below are preserved as
historical evidence. Where they mention exhausted Sarvam credits or older test
counts, they describe that earlier pass and are not the current provider state.

---

---

## Historical status snapshot (superseded by current status above)

A measured multilingual RAG system over MSMARCO-XI with grounded Mando
generation live in English, Hindi and Marathi. **392 tests pass.**
Retrieval core **22-50 ms P50**; the answerability judge is off the critical
path (**inline 0.0 ms**). LLM generation adds **~4.3 s P50** after the E17
model switch — median fixed, but the **tail (P100 65 s)** is the open blocker
for a voice product.

---

## Historical blockers recorded during E18-E20

| blocker | blocks | effect |
|---|---|---|
| **Generation tail** | a reliable voice UX | Streaming (E18) gives TTFV P50 **391 ms** -- 10.7x better PERCEIVED latency -- but total generation is still ~4.2s P50, 9.4s P100. A 20s hard deadline is now enforced client-side (`LLMGenerator.deadline_s`); Marathi hit it 4/12 times in the benchmark, always cleanly (refusal, never a broken partial answer). |
| **Pipeline-level streaming** | true mid-generation SSE through the guarded chain | Streaming exists at `LLMGenerator.generate_stream()` (fully guarded: citations validated, evidence-only). `RAGPipeline.run()` itself is still synchronous end-to-end -- deliberately deferred, larger change. Public API's SSE today streams the COMPLETED answer word-chunked, documented as such in docs/api-v1.md. |
| **Voice-loop latency, at scale** | production readiness | PARTIALLY MEASURED (E20, see below). The Sarvam account ran out of TTS credits (HTTP 402) 8 queries into the English pass, then rate-limited (429) for the rest of the run. Only **n=7 English** real voice-loop measurements exist; **Hindi and Marathi have ZERO benchmark data.** Not fabricated -- re-run needs Sarvam credits topped up. |
| **Cache-hit-rate evidence for async judge** | production deployment shape (priority 4) | BLOCKED by the same credit exhaustion -- 0 of the planned repeat-query cache-hit checks completed. The deployment decision (below) is made on existing E15/E16 evidence only, not new data. |
| **Sarvam account has zero credits, LLM included** | live demo verification | Found in this pass: every real `/api/text/query` call now returns `HTTP 402 insufficient_quota_error` from `api.sarvam.ai/v1/chat/completions` -- not just TTS (already known), the LLM proxy too, since both share the same Sarvam account/key. The pipeline degrades correctly (clean `internal_error` refusal, no crash, no leaked secret) but no real grounded answer can be produced or demonstrated live until credits are topped up. |

`LLM_API_KEY` is present and working (API-key-wise; account-credit-wise it is
currently exhausted, see above). Judge latency is RESOLVED as a blocker: it
runs async with a verdict cache, measured inline cost 0.0 ms -- **and, as of
this pass, is now actually wired into the deployed app** (see below; it
previously was not).
The Sarvam STT/TTS clients remain verified against published contracts but
never exercised live successfully in this session (credits exhausted).

### Final production readiness pass (this task)

Audited, found nothing new to fix -- everything below was ALREADY CORRECT
and is left untouched, per the "if already correct, don't touch it"
instruction:

- **Judge wiring**: `AnswerabilityJudge` inspected end-to-end -- async mode,
  `VerdictCache` claim/put/release lifecycle, background-thread exception
  handling (caught, logged, cache key released so a retry can happen; never
  crashes the pool), `shutdown(wait=False)` closes it cleanly. Bounded by
  the Phase-10 `retry_deadline_s` on the shared `ChatClient`. DONE, VERIFIED
  by code inspection + the existing `TestAsyncJudge`/`TestJudgeUnavailable`
  suites; the 600-call calibration was NOT rerun, per instruction.
- **Streaming safety**: `_stream_body()` in both `text.py` and `voice.py`
  only ever iterates over an ALREADY-COMPUTED, already-guarded
  `RAGResponse` -- `pipeline.run()` fully completes (with its own internal
  exception handling) before any SSE frame is emitted, so no external-API
  exception can occur mid-stream. `reasoning_content` is filtered by name in
  `ChatClient.stream_chat()`, never yielded. Malformed upstream frames are
  skipped (`TestStreamTransport::test_malformed_frames_are_skipped`). DONE,
  VERIFIED by `tests/test_streaming.py` (30 tests) and
  `tests/test_api_v1.py::TestSSE`.
- **Exception safety**: `app.exception_handler(Exception)` in `app/main.py`
  catches everything at the FastAPI level and returns a generic error, never
  a traceback, covering the mounted `/api/*` router too.
- **CORS**: `allow_origins=["*"]` with `allow_credentials` left at its
  default `False` -- a valid combination (the invalid one is `*` origins
  WITH credentials=True, which is not this).
- **Docker/deployment**: `Dockerfile` copies `frontend/` into the image (so
  `/` serves correctly in a container), bakes the embedding model, correct
  `HEALTHCHECK` against `/health`. `requirements.txt` has `python-multipart`
  pinned and commented as required for the voice multipart upload.
- **Full-repo secret scan** (not just the diff): zero matches for API-key
  patterns across every tracked file. `.env` confirmed untracked and
  ignored.

Added `DEMO_CHECKLIST.md` -- a concrete run-through with each item marked
DONE / VERIFIED / BLOCKED BY CREDITS / NOT LIVE-TESTED, per instruction not
to claim anything was tested live that wasn't. No further live Sarvam calls
were made this pass, per the "don't repeatedly call Sarvam while credits are
exhausted" instruction -- the account's zero-credit state was already
confirmed in the prior pass and has not been re-checked.

### Resolved this task (Final Hardening Pass)

- **REAL BUG FOUND AND FIXED: the deployed app never ran the answerability
  judge.** `app/main.py`'s `startup()` constructed `RAGPipeline(...)` without
  `judge=`, so the async-judge-plus-cache configuration documented as the
  production shape since E15/E16 (and re-affirmed as a deliberate decision in
  E20) was built, benchmarked, and tested -- but never actually connected to
  the app real users/the frontend talk to. Every eval script constructs its
  own `RAGPipeline` with the judge attached, which is why this went
  unnoticed: the EXPERIMENTS.md numbers are real measurements, they just
  weren't running in production. Fixed by wiring
  `AnswerabilityJudge(timeout_s=8.0)` + `judge_mode="async"` into `startup()`
  when `CONFIG.has_llm`, plus a `shutdown()` handler to close its thread
  pool cleanly. Verified live: `GET /api/config` now reports
  `answerability_judge: true` (was `false`). Pinned with a new regression
  test, `TestHealthConfig::test_judge_is_actually_wired_when_llm_is_available`.
- **Audited for other real blockers** (clean-environment import, route
  registration, CORS, secret leakage, stale Node-backend references,
  internal-field leakage, dependency/import sanity) -- all clean, nothing
  else found. Per the "don't touch what's already correct" instruction for
  this pass, nothing else was changed.
- **Live E2E check attempted** (English + Hindi text queries against a real
  running server): both returned a clean, correctly-shaped refusal rather
  than a fabricated answer, because the Sarvam account's LLM credits are
  also exhausted (see blocker above). This is a real, honest finding, not a
  code defect -- the failure path itself is exactly what priority 6 asked
  to verify, and it worked correctly under a genuine live failure.

### Resolved this task (Phase 10)

- **Refusals now produce spoken audio.** `RAGPipeline._refuse()` synthesizes
  the same validated refusal text used for the text answer, whenever
  `want_audio` and a TTS client are present -- see E20 below. Konkani still
  correctly gets no substituted audio (never Marathi), with the reason now
  surfaced explicitly via a new `audio_unavailable_reason` field. Tested in
  `tests/test_voice.py::TestRefusalAudio` (5 tests).
- **Hard client-side deadlines added** to every network call in the voice
  path: `ChatClient.chat()` (shared by generation and the judge), `SarvamSTT`,
  `SarvamTTS` now all cap their tenacity retry loop with `stop_after_delay`
  in addition to `stop_after_attempt`, so a chronically-failing backend can no
  longer hang a request for the full exponential-backoff schedule (this is
  what produced the E15/E17 tails). Tested in `tests/test_generation.py`
  and `tests/test_voice.py` by proving wall time stays bounded against an
  always-failing mock backend, not by attempt count alone.
- **Latency finalization for refusals is now uniform.** Previously several
  refusal exits (`empty_input`, STT failures, prompt injection, retrieval
  failure, generation failures, empty answer) left `total_rag_ms`/
  `end_to_end_ms` at their zero default. All 13 refusal exits in
  `RAGPipeline.run()` now finalize both consistently.

### Live-API findings (from the smoke test)

1. **`sarvam-m` is deprecated.** API offers `sarvam-105b` and
   `sarvam-105b-conversations`. Default updated everywhere.
2. **`sarvam-105b` is a REASONING model.** It emits `reasoning_content` before
   its answer. At `max_tokens=300` it returned `finish_reason="length"` and
   `content=None` — a well-formed 200 carrying no answer, which surfaced as a
   misleading "schema failure". At 1500 it used 901 tokens and returned valid
   JSON. Judge budget raised to 1500; `content=None` now raises a diagnosable
   error naming `max_tokens`. Regression-tested.
3. **Judge latency — RESOLVED.** Runs async behind a verdict cache; measured
   inline cost is 0.0 ms in every request (E16).
4. **`sarvam-105b-conversations` is non-reasoning** and 3.6x faster with
   quality held. Now the default (E17).

---

## What is DONE and MEASURED

All numbers reproducible via the command in `EXPERIMENTS.md`.

| area | result |
|---|---|
| Dataset | 935.5 MB of 55.6 GB downloaded (1.7%); `hinval`/`marval` verified row-parallel, 0 join mismatches over 2,000 queries |
| Corpus | 2,000 queries · 1,220 answerable · **780 no-answer** (labelled refusal set) · 59,961 passages |
| Index | 72,411 chunks · `fixed` · e5-small · 206 MB · 1,865 s offline build |
| Chunking | 6 strategies benchmarked — **fixed baseline won**; on Marathi it beat hierarchical ΔMRR +0.027, p=0.0008 (paired bootstrap) |
| Retrieval | dense-only: en MRR **0.654** · hi **0.465** · mr **0.416** |
| Hybrid BM25 | Built, ablated, **measurably hurt** (en 0.654→0.564); weight sweep monotonic → shipped **disabled** |
| Cross-lingual | Tested and **refuted** (mr→en 0.223 vs 0.416) → not enabled |
| Language detection | 95.4% overall · en 1.000 · hi 0.996 · **mr 0.866** |
| Latency (extractive, E11) | RAG P50 **137.45 ms**, P70 149.84, P100 289.07 (360 requests) |
| Latency (with LLM, E16/E17) | retrieval core **22-50 ms** P50 · generation **4.3 s** P50 (en, after model switch) · judge async, excluded |
| Generation model | **sarvam-105b-conversations** (non-reasoning). 3.6x faster than sarvam-105b with quality held: grounded 0.86-0.88, correct language, valid citations - E17 |
| Streaming (E18) | TTFT P50 **246ms** / TTFV P50 **391ms** (10.7x perceived-latency win) / generation P50 4,175ms P100 9,430ms. Deadline refusal rate 11.1% (all Marathi), failure rate 0.0%. Hard 20s wall-clock deadline enforced; incomplete-answer marking never trusts citations from a truncated stream. |
| Public API v1 | `app/api/v1/` -- thin adapter over `RAGPipeline`, internal `RAGResponse` unmodified. 4 endpoints, content-negotiated JSON/SSE, multipart voice upload. 35 tests assert schema isolation (no internal field/secret ever serialized). Documented in `docs/api-v1.md`. |
| Voice pipeline (STT+TTS), mocked | Sarvam STT -> RAGPipeline -> guarded generation -> Sarvam TTS is fully wired (`app/pipeline.py` Stage.stt/Stage.tts). 39 mocked tests (request shape, retry/4xx-vs-5xx classification, Konkani-TTS rejection, full round trip, TTS-failure-is-non-fatal, STT-failure-refuses-cleanly). |
| **Voice pipeline, LIVE (E19)** | 5 real end-to-end runs (3 fixed queries en/hi/mr + male/female voice check), real Sarvam STT+TTS + real LLM generation, existing prebuilt index (not rebuilt). en: full success, grounded=True, valid audio out, e2e 15,041ms. hi: full success, grounded=True, valid audio out, e2e 35,046ms (TTS alone: 30,924ms -- real network variance, not a bug). mr + both voice-check runs: correctly refused by the EXISTING numeric grounding hard-gate (LLM volunteered an unsupported digit at temperature=0.3) -- guardrail behaving as designed, not relaxed. Male and female voice selection both reached the pipeline correctly (voice param passed through); neither produced audio in this run because both were refused before the TTS stage. |
| Guardrails | adversarial **7/7**; injection blocked in all 3 languages |
| Generation (Phase 3) | LLMGenerator on shared ChatClient, strict schema, citation validation. Live in en/hi/mr: grounded, correct language, valid citations |
| Judge deployment | **ASYNC** - inline cost 0.0 ms in every request; background P50 5.9 s excluded by construction |
| Answerability judge | **Gate PASSED** (E15). C balAcc 0.7250/0.6414/0.6515 vs A 0.5700/0.5808/0.6061; false answers cut 62-80%. Latency P50 909 ms blocks inline use. |
| Dependencies | `requirements.txt` regenerated from an AST scan of actual imports; `datasets` + `pandas` removed (nothing imports them), all 16 pinned |

### Known weakness, quantified

**False-answer rate: 78% en / 65% hi / 47% mr.** The cosine confidence signal
has balanced accuracy 0.61–0.65 vs 0.50 chance. This is what Phase 2 exists to
fix. The extractive generator cannot fabricate (verbatim evidence), so these
are relevance failures, not invented facts — still wrong, still reported.

---

## Frozen decisions — do not change without new evidence

1. `CHUNK_STRATEGY=fixed`
2. `USE_BM25=false`
3. Cross-lingual routing **off**
4. `EVIDENCE_MIN_TOP=0.87` (deliberately not the 0.891 balanced-accuracy optimum)
5. Devanagari tokenisation uses the `regex` module, never `re.\w` — regression-tested
6. e5 similarity floors ~0.68 for unrelated text; any threshold below ~0.70 is a no-op

---

## Language scope

| | languages |
|---|---|
| `PRODUCT_LANGS` | en, hi, mr, **kok** |
| `EVALUATED_LANGS` | en, hi, mr |

**Konkani is a product language, not a benchmark language.** MSMARCO-XI has
zero Konkani rows, so no Konkani metric exists or may be manufactured.
Capability matrix: STT ✅ · TTS ❌ · corpus ❌ · benchmark ❌.
Measured only by `evaluation/konkani_product_eval.py` (stamped
`"msmarco_xi": false`), whose question set needs a Konkani speaker to author.

---

## Phase 2 — RUN #2: VALID. GATE **PASSED**.

`results_trustworthy` = True for all three languages (failure rate 0.00 / 0.01 /
0.01, threshold 0.02). Zero HTTP 429; 4 reasoning truncations total, down from
62. Downgraded verdicts 0, invalid citations 0.

| lang | A balAcc | B balAcc | C balAcc | A falseAns | **C falseAns** |
|---|---|---|---|---|---|
| en | 0.5700 | 0.7250 | **0.7250** | 0.8000 | **0.1600** |
| hi | 0.5808 | 0.6313 | **0.6414** | 0.6364 | **0.2424** |
| mr | 0.6061 | **0.6566** | 0.6515 | 0.5455 | **0.1717** |

Judge latency: P50 **909 ms** · P70 **11,116 ms** · P100 102,540 ms.

**Clause 1 PASSES:** C beats A in all three languages, under both the in-run
baseline and the literal E10 thresholds.
**Clause 2 PASSES:** false answers cut 62-80% relative (the system's worst
weakness), for +0.27-0.33 false refusals.

**VERDICT: PASS. The judge ships.**

### The PASS was about QUALITY only

At P50 909 ms / P70 11.1 s the judge **cannot** be a synchronous inline stage:
that is 6.6x the entire measured 137 ms RAG budget at P50, and Hindi alone is
10.5 s at P50. Wiring it inline as-is turns a 137 ms system into a multi-second
one.

Deployment shape is unresolved and unmeasured. Options: async revise-after-answer,
cache verdicts per (query, evidence-set), a smaller/faster judge model, or keep
it as an offline eval gate with the cosine guard at runtime. **Decide and
measure before the judge enters the runtime path.**

Full detail: EXPERIMENTS.md E15. Run #1 (invalid) preserved as E14.

---

## Phase 2 — RUN #1 INVALID (superseded by run #2)

600 calls completed, but **103 failed** (41 HTTP 429 + 62 reasoning-truncation)
and the failures were **not random**. The harness processed all positives then
all negatives, so accumulating rate-limit pressure struck the back half — for
English, **41 of 41 failures landed on negatives**. Failed verdicts fail open,
so each became a false answer by construction: English B false-answer was
inflated **0.2034 → 0.5300** and balanced accuracy depressed **0.7083 → 0.5450**
by call *order*.

That swing is large enough to reverse the verdict, so the pre-registered rule
was **not** applied in either direction.

Four fixes applied (177 tests passing):
1. the two classes are shuffled together — failures can no longer concentrate
2. retry backoff 0.3 s/3 s/3 attempts → **2 s/30 s/5 attempts** (rate limits)
3. judge `max_tokens` 1500 → **3000** (reasoning-model tail)
4. harness records `judge_failure_rate` / `results_trustworthy`, warns above 2%

Contaminated numbers, latency percentiles and full diagnosis: EXPERIMENTS.md E14.

### Re-run command (needs go-ahead; ~60–90 min)

```bash
python evaluation/answerability_calibration.py --n-per-class 100 \
  --concurrency 3 --timeout 120
```

Concurrency lowered from 6 deliberately: at 6 the API rate-limited and per-call
latency inflated (hi P50 8.6 s, P100 96.8 s).

---

## Phase 2 — earlier SMOKE (n=5 per class — INDICATIVE, NOT CONCLUSIVE)

Each metric moves in 0.2 steps at this sample size. These do **not** satisfy
the gate; the full run decides.

| lang | A balAcc | B/C balAcc | B/C falseAnswer | B/C falseRefusal |
|---|---|---|---|---|
| en | 0.600 | **1.000** | 0.000 | 0.000 |
| hi | 0.600 | **0.900** | 0.000 | 0.200 |
| mr | **0.800** | 0.700 | 0.000 | **0.600** |

Signal: the judge drove false answers to **0.000 in all three languages** —
exactly what it was built for. Cost: Marathi false refusals hit 0.600 and
**A still wins Marathi**, consistent with Marathi being the weakest retrieval
language. Whether that trade is acceptable is what the gate decides, on the
full sample.

Raw: `experiments/answerability_smoke.json`.

---

## Phase 2 — the gate

Pre-registered **before** any numbers exist, so it cannot be adjusted to fit
the result:

> The judge ships only if **B** (judge only) or **C** (cosine + judge) beats
> **A**'s balanced accuracy — **0.6525 en / 0.6325 hi / 0.6100 mr** — *and* the
> false-answer reduction justifies the added latency.
>
> C can only *reduce* the answer rate vs A, so it must be judged on the
> false-answer/false-refusal trade, not accuracy alone.

`evaluation/answerability_calibration.py` **exits without emitting numbers**
when no key is present. Verified.

---

## Repository map

```
app/
  pipeline.py            orchestration harness (judge wired at Stage.answerability)
  schemas.py             PRODUCT_LANGS vs EVALUATED_LANGS, capability matrix
  guardrails/            input · relevance (cosine) · answerability (LLM) · grounding
  generation/            llm_client (shared) · generator · persona
  retrieval/             loader · hybrid · embeddings · bm25_fast · reranker · vector_store
  stt/sarvam.py          verified against docs, never run live
  tts/sarvam_tts.py      SARVAM_TTS_LANG excludes kok — raises, never substitutes
evaluation/              9 scripts; every number in EXPERIMENTS.md traces to one
tests/                   172 passing
frontend/index.html      push-to-talk UI with live stage breakdown
```

---

## Historical resume instructions — DO NOT RUN AUTOMATICALLY

```bash
# 1. add credentials
cp .env.example .env    # then fill LLM_API_KEY (and SARVAM_API_KEY)

# 2. run the gate (~600 LLM calls)
python evaluation/answerability_calibration.py --n-per-class 100

# 3. apply the pre-registered rule above, then proceed to Phase 3 or not
```

See `TODO.md` for the ordered task list.
