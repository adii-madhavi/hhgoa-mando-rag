# MANDO — Project State

**Status: FROZEN, awaiting `LLM_API_KEY`.**
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

## Blockers (both are credentials, not code)

| blocker | blocks | effect |
|---|---|---|
| **`LLM_API_KEY`** | Phase 2 calibration → Phases 3–11 | No answerability numbers, no Mando persona, no generation |
| **`SARVAM_API_KEY`** | Phases 5–7, 10 | No STT, no TTS, no voice-loop latency |

Both clients are written and verified against published API contracts. Neither
has been exercised against a live service.

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
| Answerability judge | **Phase 1 built + 66 tests. NOT YET MEASURED.** |

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
