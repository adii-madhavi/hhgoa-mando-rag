# MANDO — TODO

**FROZEN pending `LLM_API_KEY`.** See `PROJECT_STATE.md` for full status.
Deadline: 2026-08-22 23:59.

---

## 🔴 NEEDS A DECISION FROM YOU

- [ ] **Judge latency**: p50 907 ms en but **12 s hi**, p100 50–61 s (timeout is
      20 s). Before the 600-call run, pick one:
      (a) raise timeout to ~60 s and accept a 1–3 h run,
      (b) add concurrency to the calibration harness,
      (c) cap `--n-per-class` lower.
      Separately: at ~1 s+ per call the judge cannot sit in a 200 ms budget —
      it may be an offline/eval tool rather than a runtime stage.
- [ ] **Provide `SARVAM_API_KEY`** → unblocks Phases 5–7, 10 (voice)
- [ ] *(optional, later)* Konkani speaker to author the 25–50 product questions
      in `data/konkani_product_eval.jsonl` — currently template placeholders,
      and the script refuses to run until replaced

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
- [x] **Phase 2 smoke test (n=5/class) run live** — found and fixed: deprecated
      `sarvam-m` model, and reasoning-model `max_tokens` truncation returning
      `content=None`. 176 tests passing.

---

## ⏸ NEXT — the moment the LLM key lands

**Phase 2 — calibration (the gate)**

```bash
python evaluation/answerability_calibration.py --n-per-class 100
```

- [ ] Decide the latency question above FIRST
- [ ] Run A/B/C on en, hi, mr (Konkani → explicit N/A row)
- [ ] Apply the **pre-registered** rule: judge ships only if B or C beats A's
      balanced accuracy (0.6525 / 0.6325 / 0.6100) **and** the false-answer
      reduction justifies the latency
- [ ] Write measured results into `EXPERIMENTS.md` E13
- [ ] **If the gate fails → the judge does not ship.** Report that plainly and
      keep the cosine guard.

---

## Then, in order (each gated on the previous)

**Phase 3 — Mando generation** *(only if Phase 2 passes)*
- [ ] Migrate `generator.py` onto the shared `llm_client`
- [ ] Wire the persona; answer in the user's language, evidence-only
- [ ] Verify the persona never overrides factuality

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
- [ ] Keep all 172 passing; add live-API failure/timeout/retry cases

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
