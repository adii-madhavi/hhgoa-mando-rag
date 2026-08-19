# MANDO — Project State

**Status: Phase 2 run #1 INVALID (methodology bug, fixed). Re-run required.**
**Phase 3 NOT unlocked.**
Last updated: 2026-08-19 · Deadline: 2026-08-22 23:59
Repo: https://github.com/adii-madhavi/hhgoa-mando-rag (branch `main`, pushed)

`PROJECT_STATE.md` and `TODO.md` are the source of truth. Update both after
every major task.

Do not make architecture changes or run the calibration until the key is
provided. Phases 3–11 are gated behind Phase 2, which is gated on the key.

---

## One-line status

A working, measured, voice-ready multilingual RAG system over MSMARCO-XI.
**172 tests pass.** RAG core runs at **137 ms P50**. Everything still open is
blocked on two credentials, not on engineering.

---

## Blockers

| blocker | blocks | effect |
|---|---|---|
| **Judge latency** (open question) | Phase 2 full run, and the 200 ms budget | judge p50 907 ms en / **12,048 ms hi**, p100 50–61 s |
| **`SARVAM_API_KEY`** | Phases 5–7, 10 | No STT, no TTS, no voice-loop latency |

`LLM_API_KEY` is now present and working. The Sarvam STT/TTS clients remain
verified against published contracts but never exercised live.

### Live-API findings (from the smoke test)

1. **`sarvam-m` is deprecated.** API offers `sarvam-105b` and
   `sarvam-105b-conversations`. Default updated everywhere.
2. **`sarvam-105b` is a REASONING model.** It emits `reasoning_content` before
   its answer. At `max_tokens=300` it returned `finish_reason="length"` and
   `content=None` — a well-formed 200 carrying no answer, which surfaced as a
   misleading "schema failure". At 1500 it used 901 tokens and returned valid
   JSON. Judge budget raised to 1500; `content=None` now raises a diagnosable
   error naming `max_tokens`. Regression-tested.
3. **Judge latency is the open problem.** ~900 ms typical, but Hindi p50 is
   12 s and some calls exceed the 20 s timeout. Needs a decision before the
   600-call run: raise timeout, add concurrency, or accept the cost.

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
| Latency | **TOTAL RAG P50 137.45 ms**, P70 149.84, P100 289.07 (360 requests) |
| Guardrails | adversarial **7/7**; injection blocked in all 3 languages |
| Answerability judge | Phase 1 built + 70 tests. **Smoke-tested live (n=5/class); full calibration pending.** |
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

## Phase 2 — RUN #1 INVALID. GATE NOT APPLIED. PHASE 3 NOT UNLOCKED.

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
