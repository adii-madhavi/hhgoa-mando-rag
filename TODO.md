# MANDO — TODO

Source of truth alongside `PROJECT_STATE.md`. Update both after every major task.
Deadline: 2026-08-22 23:59.

Phases 1-3 COMPLETE. `LLM_API_KEY` present and working.

---

## 🔴 NEXT ACTION — the latency TAIL, and Sarvam credentials

Generation median is fixed (E17): switching to the non-reasoning
`sarvam-105b-conversations` cut English P50 **15.4 s -> 4.3 s** with quality
held (grounded 0.86-0.88, correct language, valid citations).

Remaining problems, in order:

- [ ] **Tail latency.** P100 hit **65 s** on a Marathi call; async judge P100
      127 s. Needs a hard client-side deadline + degradation path, not a better
      median.
- [ ] **Streaming** so first-audio starts before the full answer is ready --
      the single biggest perceived-latency win for voice.
- [ ] **`SARVAM_API_KEY`** to unblock Phases 5-7, 10 (STT, TTS, voice loop).

Honest position on the 200 ms target: only the **retrieval core** (22-50 ms
P50) is inside it. Generation is ~4.3 s, roughly 20x over, and the voice loop
adds two more network round trips on top. These are reported as separate
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
