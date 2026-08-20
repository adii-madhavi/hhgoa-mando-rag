# MANDO — Project State

**Status: Phase 3 COMPLETE + streaming (E18) + public API v1 layer.**
**Generation P50 4.3s (E17); streaming TTFV P50 391ms gives ~10.7x perceived-latency win (E18).**
**Public API contract live at /api/* (docs/api-v1.md) as a thin adapter -- internal pipeline untouched.**
Last updated: 2026-08-19 · Deadline: 2026-08-22 23:59
Repo: https://github.com/adii-madhavi/hhgoa-mando-rag (branch `main`, pushed)

`PROJECT_STATE.md` and `TODO.md` are the source of truth. Update both after
every major task.

**Frontend is out of scope for backend work from this point on.** It will be
built separately against `docs/api-v1.md`. Do not create/modify frontend/,
React/Next.js, HTML, CSS, or Mando visual assets from the backend side.

Phases 1-3 complete. Voice pipeline (Sarvam STT -> RAGPipeline -> guarded
generation -> Sarvam TTS) is BUILT, MOCK-TESTED (39 tests), and NOW LIVE-
VALIDATED end-to-end with real Sarvam + LLM calls (3 fixed queries + 2 voice
checks, n=1 each -- not a benchmark). 2/3 languages produced a grounded,
audible answer; the numeric grounding gate correctly refused the other 2
query variants when the LLM volunteered an unsupported digit. UI remains out
of scope for backend work.

---

## One-line status

A measured multilingual RAG system over MSMARCO-XI with grounded Mando
generation live in English, Hindi and Marathi. **360 tests pass.**
Retrieval core **22-50 ms P50**; the answerability judge is off the critical
path (**inline 0.0 ms**). LLM generation adds **~4.3 s P50** after the E17
model switch — median fixed, but the **tail (P100 65 s)** is the open blocker
for a voice product.

---

## Blockers

| blocker | blocks | effect |
|---|---|---|
| **Generation tail** | a reliable voice UX | Streaming (E18) gives TTFV P50 **391 ms** -- 10.7x better PERCEIVED latency -- but total generation is still ~4.2s P50, 9.4s P100. A 20s hard deadline is now enforced client-side (`LLMGenerator.deadline_s`); Marathi hit it 4/12 times in the benchmark, always cleanly (refusal, never a broken partial answer). |
| **Pipeline-level streaming** | true mid-generation SSE through the guarded chain | Streaming exists at `LLMGenerator.generate_stream()` (fully guarded: citations validated, evidence-only). `RAGPipeline.run()` itself is still synchronous end-to-end -- deliberately deferred, larger change. Public API's SSE today streams the COMPLETED answer word-chunked, documented as such in docs/api-v1.md. |
| **Voice-loop latency, at scale** | Phase 10 (100+ query benchmark) | Single-run figures only exist so far (E19, n=1/query): en e2e 15.0s, hi e2e 35.0s (TTS alone hit 30.9s once). No percentiles yet -- that is explicitly Phase 10, not this task. |
| **Refusals produce no spoken audio** | voice UX completeness | Found during E19: `RAGPipeline` only reaches the TTS stage on the SUCCESS path; every `_refuse()` return exits before stage 11, so a refused voice query is answered with `resp.audio_base64=None` -- silence, even though `resp.answer` already holds correctly-localised refusal text. Not fixed (a real design decision, not a bug fix, and out of this task's explicit scope). Flagged for your review.

`LLM_API_KEY` is present and working. Judge latency is RESOLVED as a blocker:
it runs async with a verdict cache, measured inline cost 0.0 ms.
The Sarvam STT/TTS clients remain verified against published contracts but
never exercised live.

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

## Resume instructions

```bash
# 1. add credentials
cp .env.example .env    # then fill LLM_API_KEY (and SARVAM_API_KEY)

# 2. run the gate (~600 LLM calls)
python evaluation/answerability_calibration.py --n-per-class 100

# 3. apply the pre-registered rule above, then proceed to Phase 3 or not
```

See `TODO.md` for the ordered task list.
