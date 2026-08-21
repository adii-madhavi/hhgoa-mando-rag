# MANDO — a voice-enabled multilingual RAG system

**English · हिंदी · मराठी · कोंकणी**
<sub>Benchmarked in English, Hindi and Marathi. Konkani is a supported product
language with no labelled evaluation data — see below; no Konkani metric is
claimed.</sub>

Mando is a friendly Goan AI guide. Ask him a question out loud, in any of three
languages, and he answers from a retrieved evidence base — or tells you
honestly that he couldn't find enough to answer.

Underneath the character is the actual submission: a multilingual RAG pipeline
over `ai4bharat/MSMARCO-XI`, with a real orchestration harness, three layers of
guardrails, and latency instrumented per stage — **137 ms P50 across 360
measured retrieval/RAG-core requests (LLM generation and voice excluded)**.

## Product answering and provenance

MANDO keeps three answer origins separate. `corpus` means the normal
MSMARCO-XI-backed runtime index supplied evidence and the answer passed
grounding. `goa_knowledge` means the small, source-attributed corpus in
`data/goa_knowledge.json` supplied evidence and the answer passed the same
grounding verifier. `external` means general model knowledge was used only
after both evidence stores were insufficient; it has no corpus citations, is
not grounded against HHGoa, and returns `external_verified=false`.

Primary grounded generation and external general-knowledge generation are
independently configurable. The external path requires all three
`EXTERNAL_LLM_API_KEY`, `EXTERNAL_LLM_BASE_URL`, and
`EXTERNAL_LLM_MODEL` values and never reuses the primary or Sarvam key.
Speech-to-text may use Sarvam or ElevenLabs via `STT_PROVIDER`.

**Benchmark/evaluation remains MSMARCO-XI. Product knowledge is the curated Goa
corpus.** The Goa passages are a separate product-only index and never enter
MSMARCO-XI evaluation, calibration, or historical metrics.

Public requests accept `answer_mode: "fast" | "detailed"` (default:
`"detailed"`). Fast mode preserves retrieval and grounding while requesting a
shorter answer with a smaller generation budget. Browser SpeechSynthesis reads
the already displayed answer only after Listen is pressed. Voice queries do
not request paid backend TTS by default; explicit `want_audio=true` preserves
backend TTS compatibility.

Runtime P50/P70 uses bounded in-memory windows separated by text/voice and
fast/detailed. Values appear after five successful samples; before that the UI
says “Collecting data.” Only latency numbers and categories are stored.

Its most useful property is that several of its headline components lost to
their baselines, and the repository says so: six chunking strategies were
benchmarked and the fixed-size splitter won; hybrid BM25 retrieval measurably
hurt and ships disabled; cross-lingual routing was tested and refuted.

> **Every number in this README was measured by a script in this repository.**
> Nothing is estimated. Where a result was disappointing, negative, or a
> component came out worse than its baseline, it is reported as such. The full
> log of experiments — including the ones that failed — is in
> [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Honest scope: what works and what doesn't](#honest-scope-what-works-and-what-doesnt)
- [Why three languages and not four](#why-three-languages-and-not-four)
- [The dataset, and what we found in it](#the-dataset-and-what-we-found-in-it)
- [Chunking](#chunking)
- [Retrieval](#retrieval)
- [Guardrails](#guardrails)
- [Latency](#latency)
- [Evaluation](#evaluation)
- [Running it](#running-it)
- [Repository layout](#repository-layout)

---

## What it does

```
🎙  user speaks (en / hi / mr)
      ↓
    Sarvam STT  (saaras:v3)
      ↓
    language resolution   — UI choice > Sarvam's audio-derived code > text heuristic
      ↓
    input guardrail       — empty · unintelligible · unsafe · prompt injection
      ↓
    query rewriting       — resolves follow-ups against session memory
      ↓
    retrieval             — dense (multilingual-e5); hybrid BM25+RRF is built
                            and switchable, but measured WORSE here, so off
      ↓
    reranking             — MMR over the candidates
      ↓
    evidence check        — refuse if retrieval confidence is below threshold
      ↓
    grounded generation   — Mando persona, answer constrained to the evidence
      ↓
    grounding verifier    — numeric + semantic support; FAIL discards the answer
      ↓
    Sarvam TTS  (bulbul:v3)
      ↓
🔊  Mando speaks
```

The persona layer and the factual layer are deliberately separate. Mando
controls tone; he has no authority over facts. The grounding verifier runs
after generation regardless of how charming the output is, and a FAIL replaces
the answer with a refusal.

---

## Architecture

**Stage failure policy.** The asymmetry here is the design:

| Stage | On failure |
|---|---|
| validate, STT | refuse — no input, nothing to retrieve |
| language detection | **default to English and continue** — never fatal |
| input guardrail | refuse, with the specific reason |
| query rewrite | **fall back to the raw transcript** — never fatal |
| retrieval | refuse — answering without evidence is the one unacceptable failure |
| rerank | **fall back to fused order** — never fatal |
| evidence check | refuse |
| generation | refuse — never emit a partial answer |
| **grounding** | **discard the answer and refuse** |
| TTS | **return text only** — never fatal |

Stages affecting *presentation* degrade gracefully. Stages affecting *truth*
refuse. Every stage records latency, attempt count and a structured error even
when it throws.

**On timeouts, honestly.** Network stages (STT, generation, TTS) get real
enforced timeouts, because `httpx` can abort a socket. Local compute stages get
*deadline checks between stages*, not interruption — you cannot pre-empt a
running numpy matmul from the same thread, and an architecture diagram claiming
otherwise would be a lie. See `app/pipeline.py`.

---

## Honest scope: what works and what doesn't

| Capability | Status |
|---|---|
| Multilingual retrieval (en/hi/mr) | ✅ measured — en MRR 0.654 · hi 0.465 · **mr 0.416** |
| Chunking strategies (6, benchmarked) | ✅ measured — the **fixed baseline won** |
| Hybrid dense + BM25 + RRF | ✅ built and ablated — **measurably hurt, shipped off** |
| Latency < 200 ms | ✅ **137.45 ms P50** across 360 requests (RAG path; voice loop excluded and stated) |
| Refusal / hallucination guardrail | ⚠️ **weakest component** — false-answer rate 78% en / 65% hi / 47% mr, quantified below |
| Guardrails (input / retrieval / grounding) | ✅ implemented, thresholds calibrated on labelled data |
| Orchestration harness | ✅ structured I/O, retries, request IDs, per-stage latency |
| Language detection | ✅ 95.4% overall, **86.6% Marathi** — the weak spot, quantified |
| Adversarial guardrails | ✅ **7/7** — injection blocked in all three languages |
| Sarvam STT / TTS | ⚠️ **implemented and verified against Sarvam's API docs, but not yet exercised against the live API** — no key at time of writing |
| Mando persona generation | ⚠️ **requires an LLM key.** Without one the system uses a grounded extractive fallback (verbatim evidence sentences — cannot hallucinate, but no persona) |
| Konkani | 🟡 **product language, not a benchmark language** — speakable input, no labelled data, no TTS voice; see below |

The ⚠️ credential rows are gated, not unbuilt. `/health` reports exactly
what is missing at runtime and the UI degrades visibly rather than pretending.

---

## Four product languages, three evaluated languages

These are different claims, and conflating them is how a submission ends up
reporting "four languages evaluated" on three languages' worth of evidence. So
they are separate in code (`app/schemas.py`), not just in prose:

| | meaning | languages |
|---|---|---|
| `PRODUCT_LANGS` | a user may speak/select it; the system tries to serve them | en, hi, mr, **kok** |
| `EVALUATED_LANGS` | **labelled benchmark data exists**, so every measured number is real | en, hi, mr |

**Konkani is a product language. Every benchmark number in this repository
covers English, Hindi and Marathi only, and no Konkani metric is reported
anywhere** — because `ai4bharat/MSMARCO-XI` ships zero Konkani rows under any
code (`kok`, `kon`, `gom`, `knn`). A Konkani Recall@k cannot be computed from a
dataset with no Konkani. That is a **dataset** limitation, not a product one.

### Verified Konkani capability matrix

| capability | status |
|---|---|
| speech-to-text | ✅ Sarvam `saaras` accepts `kok-IN` |
| text-to-speech | ❌ `bulbul:v3` has no `kok-IN` |
| corpus evidence | ❌ none — retrieval is cross-lingual into en/hi/mr |
| labelled benchmark | ❌ none |

Enforced in code, not promised in comments:

- `SARVAM_TTS_LANG` is a **separate, narrower map** than `SARVAM_LANG`.
  Synthesis raises for Konkani instead of quietly substituting the Marathi
  voice. TTS failure is non-fatal, so a Konkani user gets the text answer.
- Refusal copy has **no Konkani strings**, and
  `refusal_text_with_locale()` returns a flag saying so. Marathi is explicitly
  not used as a stand-in — that is the same mislabelling rejected for TTS.
  Writing Konkani user-facing copy needs a Konkani speaker.
- Retrieval for Konkani goes cross-lingual into en/hi/mr evidence, the path
  measured as **weak** (mr→en MRR 0.223 vs 0.416 in-language), so Konkani
  quality should be expected to be materially worse.

Konkani is measured separately by `evaluation/konkani_product_eval.py` against
a curated product set that is **never merged into the MSMARCO-XI tables** —
separate file, separate output, every row stamped `"msmarco_xi": false`. The
question set requires a Konkani speaker to author; the script refuses to run
while template placeholders remain.

Full reasoning: [EXPERIMENTS.md E2 / E2b](EXPERIMENTS.md).

---

## The dataset, and what we found in it

`ai4bharat/MSMARCO-XI` is **55.6 GB**, sharded one parquet file per language.
Two findings shaped everything downstream.

**1. Streaming inspection was the wrong tool, and silently so.**
`load_dataset(streaming=True).take(20)` reads only the *first* shard — so the
observed "language distribution" is 100% Assamese and means nothing.
`ingestion/dataset_info.py` enumerates shards via the Hub API and inspects each
independently over HTTP range reads: a few MB, not 55.6 GB.

**2. `hinval` and `marval` are row-parallel.** Not merely both derived from MS
MARCO — row-for-row translations of the *same* queries, with identical English
passages and identical relevance labels. `ingestion/preprocess.py` verifies
this per record rather than trusting a sample:

```
join_mismatches_skipped: 0     across all 2,000 queries
```

This is the finding the whole cross-lingual design rests on. **One shared qrel
set across three languages** means a Hindi query and an English query for the
same `query_id` have literally the same relevant passages — so cross-language
retrieval differences are real, not label noise.

We download **935.5 MB of 55.6 GB (1.7%)** — the two validation shards only.

### Corpus

| | |
|---|---|
| Queries | 2,000 |
| — answerable | 1,220 |
| — **no answer** (`is_selected` all-zero) | **780** |
| Passages per language | 19,987 (**59,961** total) |
| Mean passage chars | en 312.8 · hi 329.0 · mr 326.7 |
| Relevant passages per answerable query | 1 → 1,169 · 2 → 46 · 3 → 3 · 5 → 2 |

Two consequences worth stating:

- **Recall@1 and MRR are the decisive metrics.** 1,169 of 1,220 answerable
  queries have exactly one relevant passage, so Recall@10 saturates and cannot
  separate configurations. We report @1/@3/@5/@10 as asked, but do not make
  decisions on @10.
- **The 780 no-answer queries are a free labelled refusal set.** They turn
  guardrail calibration from a vibe into a fitted threshold with a measured
  false-answer rate.

---

## Chunking

Six strategies implemented (`ingestion/chunk.py`), all emitting a uniform
metadata-carrying `Chunk`: fixed · sentence · sentence-window · semantic ·
hierarchical · adaptive.

Because MSMARCO passages are short web snippets (~313 chars), not documents,
the hierarchy is anchored one level **up**, using structure the dataset really
has:

```
query group  →  passage  →  sentence-window  →  sentence
```

A design detail that makes window and hierarchical worth anything: **the
embedded unit and the served unit are different objects.** `Chunk.text` is what
gets embedded (precise); `Chunk.context` is what the generator sees (complete).
Without that split, both strategies are just sentence chunking with extra
steps.

### Benchmark result: a tie

English, e5-small, 300 queries, 6,000-passage corpus. Scored at **passage**
level after collapsing chunks to parents — strategies emit very different chunk
counts, and chunk-level top-k would hand multi-chunk strategies extra shots at
the target.

| Strategy | chunks | mean chars | R@1 | R@5 | MRR | chunk cost |
|---|---|---|---|---|---|---|
| hierarchical | 17,593 | 206.6 | 0.440 | **0.933** | **0.650** | 0.3 s |
| adaptive | 11,744 | 154.5 | 0.450 | 0.920 | **0.650** | 26.5 s |
| semantic | 10,374 | 174.9 | **0.460** | 0.913 | 0.649 | 155.5 s |
| **fixed** *(baseline)* | 6,822 | 274.8 | 0.443 | 0.920 | 0.648 | 0.0 s |
| sentence | 7,280 | 249.7 | 0.437 | 0.923 | 0.647 | 0.3 s |
| window | 19,411 | 93.0 | 0.410 | 0.887 | 0.607 | 0.3 s |

**Five of six strategies are statistically indistinguishable** — MRR spans
0.003 against a standard error of ~0.02. The sophisticated strategies do not
beat a fixed splitter on this dataset, because at 313 chars per passage there
isn't enough text for cut-placement to matter. Semantic chunking spends 155.5 s
of offline compute to land within 0.001 MRR of a free baseline.

The one strategy that clearly separates does so **downward**: sentence-window,
at MRR 0.607. At 93 chars the embedded unit is a lone sentence stripped of
context — on already-short passages that discards more than precision gains
back. Worth saying plainly, since sentence-window is widely recommended; it is
good for long documents and bad here.

### Then the cross-language sweep reversed the decision

English-only numbers picked `hierarchical`. Re-running across all three
languages showed that was wrong. Paired bootstrap, 10,000 resamples,
`fixed` − `hierarchical`:

| lang | ΔMRR | CI95 | p | verdict |
|---|---|---|---|---|
| en | −0.0018 | [−0.0255, +0.0217] | 0.877 | tie |
| hi | −0.0013 | [−0.0194, +0.0163] | 0.895 | tie |
| **mr** | **+0.0271** | **[+0.0112, +0.0435]** | **0.0008** | **fixed wins** |

`hierarchical` emits smaller children (206 vs 275 chars). Where the embedding
model is weakest — Marathi — less context per chunk hurts most. The effect is
invisible in English, where the model is strong enough that chunk length
barely matters.

**Selected: `fixed`.** It ties where the strategies tie, wins significantly in
the weakest language, and yields a **2.6× smaller index** (6,822 vs 17,660
chunks per language), which directly reduces query-time search cost.

> **The honest summary: six chunking strategies were implemented and
> benchmarked, and the fixed-size baseline won.** Had we benchmarked only in
> English — the obvious shortcut — we would have shipped the worse strategy for
> our weakest language and never found out.

---

## Retrieval

Dense (`multilingual-e5`) + BM25, fused with **Reciprocal Rank Fusion**, then
deduplicated to passages, then reranked.

**Why RRF and not a weighted score sum.** Dense cosine sits in a narrow band;
BM25 is unbounded and corpus-dependent. Normalising them onto a shared scale
needs per-query min/max, which is unstable when one retriever returns few hits
— common for BM25 on short Devanagari queries. RRF consumes only *ranks*, so
the calibration problem disappears.

### A bug worth documenting: `re.\w` destroys Devanagari

Python's `re` excludes Unicode categories Mn/Mc (combining marks) from `\w`.
Devanagari carries its vowels as exactly those marks:

| tokeniser | `कॉर्पोरेशन` |
|---|---|
| `re.compile(r"\w+", re.UNICODE)` | `['क','र','प','र','शन']` — shattered |
| `regex.compile(r"[\p{L}\p{N}\p{M}]+")` | `['कॉर्पोरेशन']` — correct |

The failure is silent and **asymmetric**: English BM25 works perfectly while
Hindi/Marathi degrade toward random. It would have surfaced as the
plausible-sounding false conclusion *"hybrid retrieval just doesn't help much
for Indic languages"* rather than as an error. `regex` is therefore a hard
dependency, and there is a regression test.

### Retrieval quality is not uniform across languages

| lang | R@1 | R@5 | MRR |
|---|---|---|---|
| English | 0.443 | 0.920 | **0.648** |
| Hindi | 0.350 | 0.797 | **0.523** |
| Marathi | 0.290 | 0.703 | **0.459** |

**Marathi MRR is 29% relatively worse than English.** This is the largest
quality gap in the system, and it is reported per-language rather than averaged
into one flattering headline number.

Two compounding causes, neither of which is a chunking problem:
`multilingual-e5-small` has weaker representations for Indic languages, and the
Hindi/Marathi passages are *machine translations* while the English ones are
the originals. The second point suggests a fix — route non-English queries
against the higher-quality English evidence through the shared embedding space
— which is exactly what the cross-lingual ablation in `evaluation/retrieval.py`
tests.

### Hybrid retrieval was built, benchmarked, and turned off

The brief asked for dense + BM25 hybrid. We implemented it and it **measurably
hurt**, in every language:

| lang | dense-only MRR | bm25-only | hybrid RRF 1:1 |
|---|---|---|---|
| en | **0.654** | 0.402 | 0.564 |
| hi | **0.465** | 0.249 | 0.378 |
| mr | **0.416** | 0.223 | 0.359 |

To rule out the fusion *weighting* rather than fusion itself, we swept the
dense:BM25 ratio — and it is **perfectly monotonic in all three languages**:

```
1:1  0.564    3:1  0.615    5:1  0.629    dense-only  0.654    (English MRR)
```

Every reduction in BM25's weight improves MRR. There is no ratio at which
lexical retrieval contributes.

**Why:** MS MARCO queries are natural-language questions and passages are
paraphrased web text, so term overlap is weak — and in Hindi/Marathi the
passages are machine translations whose word forms differ from the translated
query's. Dense retrieval handles paraphrase; BM25 needs an overlap that isn't
there, and RRF then grants that noise equal authority.

`USE_BM25=false` by default. The implementation and flag remain — hybrid is
right on corpora with real lexical overlap, and it's one env var away. This is
a **quality** decision, not a latency one: BM25 costs 0.84 ms.

### Cross-lingual retrieval: a hypothesis we tested and disproved

English passages are originals; Hindi/Marathi ones are machine translations. So
routing non-English queries at English evidence *should* help, and would let us
index one language instead of three.

| query language | → English-only index | → own language |
|---|---|---|
| hi | 0.356 | **0.465** |
| mr | 0.223 | **0.416** |

**Refuted.** A Marathi query against English evidence nearly halves its MRR.
`multilingual-e5-small` is genuinely *multilingual* (it handles each language)
but only weakly *cross-lingual* (it does not bridge between them) at this size.
Recorded because the reasoning was sound and the answer was still no.

---

## Guardrails

Three layers.

**1. Input** (`app/guardrails/input.py`) — sub-millisecond, no model. Catches
empty/unintelligible transcripts, unsafe requests, and prompt injection **in
all three languages** (an English-only pattern list would leave Hindi and
Marathi unguarded, which looks like working security until someone tries it).

Deliberately narrow on "unsafe": MS MARCO is full of legitimate medical and
historical-violence queries. Measured behaviour — `"what is the lethal dose of
acetaminophen"` passes (a real health question), `"how to make a bomb"` is
blocked. A guardrail that refuses the corpus's own content is worse than one
that passes a borderline case to the evidence check.

**2. Retrieval confidence** (`app/guardrails/relevance.py`) — calibrated on the
780 labelled no-answer queries via `evaluation/threshold_calibration.py`,
maximising *balanced* accuracy. The two errors are not equally bad: a false
answer is a hallucination, a false refusal is merely unhelpful.

Calibration result, reported rather than buried: **balanced accuracy is
0.61–0.65 against 0.50 for a coin flip.** Retrieval confidence is a *weak*
discriminator here, and inherently so — a "no-answer" MS MARCO query still has
ten topically related passages, because a real search engine returned them. The
discrimination is "relevant vs plausibly-related-but-insufficient", not
"relevant vs random". We ship 0.87 rather than the 0.891 balanced-accuracy
optimum, because at the optimum the system refuses 26% of answerable English
and 54% of answerable Hindi queries — too high a price for a signal this weak.

**3. Grounding** (`app/guardrails/grounding.py`) — runs after generation, two
independent checks:

- **Numeric (hard gate).** Every number in the answer must appear in the
  evidence. Targets the most damaging RAG failure — a fluent answer with an
  invented figure or date. Verified catching invented `1873` and `42`.
- **Semantic (soft).** Mean over answer sentences of best-match similarity to
  the evidence.

> **Calibration warning, learned the hard way.** multilingual-e5 does not use
> the full cosine range. Measured against a corporation passage: grounded
> paraphrase **0.939**, grounded sentence **0.827**, and *completely unrelated
> text* ("the best beaches in Goa are Baga and Anjuna") still scores **0.678**.
> The initial 0.60 threshold was a **no-op that passed everything**. Anything
> below ~0.70 is not a threshold. Now 0.82.

---

## Latency

**What is and isn't counted.** Chunking, corpus embedding and index building
are *offline* (`ingestion/index.py`) and appear in no latency figure. Model and
index load happen once at startup and are reported separately. A warm-up pass
is discarded. **Stages that did not run are reported as "not measured", never
as 0 ms** — without a Sarvam key there is no STT, and printing 0 would
misrepresent the system.

Measured on **CPU only** (16 cores, `torch 2.13.0+cpu`, no CUDA), which is the
binding constraint on model choice throughout.

**360 measured requests** (120 queries × 3 languages), warm-ups discarded.

| stage | P50 | P70 | P100 | mean |
|---|---|---|---|---|
| validate | 0.00 | 0.00 | 0.00 | 0.00 |
| stt | *not measured* | | | no `SARVAM_API_KEY` |
| language_detection | 0.01 | 0.01 | 0.04 | 0.01 |
| input_guardrail | 0.06 | 0.06 | 0.15 | 0.06 |
| query_rewrite | *not measured* | | | no follow-ups in this set |
| dense_retrieval | 20.88 | 21.82 | 47.81 | 21.02 |
| bm25_retrieval | *not measured* | | | disabled on evidence |
| fusion | 0.02 | 0.02 | 0.03 | 0.02 |
| rerank | 0.18 | 0.20 | 0.48 | 0.19 |
| evidence_check | 0.02 | 0.02 | 0.05 | 0.02 |
| generation | 94.25 | 105.67 | 222.68 | 98.87 |
| grounding | 26.52 | 29.17 | 54.80 | 27.49 |
| tts | *not measured* | | | no `SARVAM_API_KEY` |
| **TOTAL RAG** | **137.45** | **149.84** | 289.07 | 123.33 |

Per language P50: `en 130.98` · `hi 139.78` · `mr 144.58`.

**137.45 ms P50 and 149.84 ms P70 — both inside the 200 ms target.** P100 of
289 ms is one worst-case query and shouldn't be read as a service level.

### Getting there took two real optimisations

The first honest measurement was **276 ms P50 — over budget.** Two fixes:

**1. Stop re-encoding what the index already holds.** The naive pipeline
encoded the same evidence *three times* — retrieval, then MMR, then grounding —
at ~23 ms per text. Candidates now carry their row in the dense matrix.
A correctness guard was needed: the index embeds `chunk.text` while those
stages use `chunk.context`, identical for `fixed` but **not** for hierarchical
or sentence-window, so the shortcut is taken only when the strings are equal.

```
rerank  110 ms → 0.18 ms        grounding  120 ms → 26 ms
```

**2. Inverted-index BM25.** `rank_bm25` fills a score array over all 72,411
chunks per query term, in Python. `app/retrieval/bm25_fast.py` scores only
documents that contain a query term — verified in tests as a pure speed change,
producing **identical rankings and identical scores to 1e-4**.

```
bm25  95.16 ms → 0.84 ms   (113×)
```

### What these numbers exclude, stated plainly

- Offline chunking + embedding (1,865 s) and index build — not runtime.
- Startup: 14.7 s of model and index load, paid once.
- **STT and TTS are reported as `not measured`, never as 0.** Without a Sarvam
  key the voice path does not execute. The full voice loop adds two network
  round trips and **will not fit 200 ms** — which is why `total_rag_ms` and
  `end_to_end_ms` are separate fields.
- Generation here is the extractive fallback; an LLM call is a network round
  trip and will dominate the budget.

---

## Evaluation

| Script | Measures |
|---|---|
| `evaluation/chunking_benchmark.py` | chunking strategies × embedding models, Recall@k + MRR |
| `evaluation/significance.py` | paired bootstrap — is a difference real? |
| `evaluation/retrieval.py` | dense vs BM25 vs hybrid; reranker cost/benefit; cross-lingual |
| `evaluation/language_detection.py` | 6,000 labelled cases from parallel triples |
| `evaluation/threshold_calibration.py` | refusal threshold on 780 labelled no-answer queries |
| `evaluation/answer_quality.py` | groundedness, refusal correctness, adversarial probes |
| `evaluation/latency.py` | per-stage P50/P70/P100 |

### Guardrails, measured

**Adversarial: 7/7.** Prompt injection blocked in all three languages, plus
DAN-style jailbreak, unsafe request, empty and unintelligible input — each
refused with the *correct* reason, not merely refused. Injection is caught
before retrieval runs (0.0 ms, no evidence fetched). Off-topic: 3 of 4 refused.

**Refusal correctness — the system's weakest point, quantified honestly:**

| lang | false refusal | **false answer** | groundedness | answer≈reference |
|---|---|---|---|---|
| en | 10.0% | **78.3%** | 0.952 | 0.895 |
| hi | 20.0% | **65.0%** | 0.940 | 0.873 |
| mr | 31.7% | **46.7%** | 0.925 | 0.861 |

**The system answers most questions the dataset labels unanswerable.** That
follows directly from the 0.63-accuracy confidence signal above.

One qualifier that matters for reading the number — not offered as an excuse:
in this configuration the generator is *extractive*, so it physically cannot
fabricate. Every word is copied from a retrieved passage. A "false answer" is
returning real evidence text for a question that evidence doesn't actually
answer — a **relevance** failure, not an invented fact. Still wrong, still
reported.

**The architectural fix is already wired and needs only a credential.** The
pipeline refuses when the generator reports `sufficient: false`, and the persona
prompt instructs the LLM to set it when the sources don't answer the question.
An LLM that has *read* the passages judges sufficiency far better than a cosine
threshold. `ExtractiveGenerator` can't make that call and always returns
`sufficient=True`, so today the whole burden sits on the weak signal. This is
the single largest expected gain from adding `LLM_API_KEY`.

### Language detection, measured

Ground truth is free — 2,000 parallel triples give 6,000 labelled cases.

| iteration | overall | en | hi | **mr** |
|---|---|---|---|---|
| v1 whole-token markers | 0.8807 | 1.000 | 0.998 | **0.644** |
| v2 + suffix matching | 0.9513 | 1.000 | 0.996 | **0.859** |
| v3 + Marathi verb forms | **0.9538** | 1.000 | 0.996 | **0.866** |

The v1 failure had a real linguistic cause: **Marathi agglutinates
postpositions onto the noun while Hindi separates them** —
`पोटॅशियमचे` vs `पोटैशियम के`. Whole-token matching therefore *structurally
cannot fire* for Marathi, which is precisely the 0.998 vs 0.644 asymmetry. The
fix is suffix matching.

**The residual ~13% Marathi error is largely irreducible from text alone** —
queries like `स्टबहब टोल फ्री नंबर` are transliterated English nouns with zero
Marathi morphology. Mitigated by precedence: explicit UI choice wins, then
Sarvam's audio-derived `language_code`, then this heuristic.

---

## Running it

Requires **Python 3.11+** (developed on 3.14).

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Credentials — copy `.env.example` to `.env` and fill in:

```
SARVAM_API_KEY=...     # speech-to-text AND text-to-speech
LLM_API_KEY=...        # answer generation (Mando persona)
```

Without them the system still runs: voice input is disabled and answers come
from the grounded extractive fallback. `/health` reports exactly what's missing.

Build the corpus and index (**offline, one time**):

```bash
python ingestion/dataset_info.py --shards hin,mar --split validation
python ingestion/download.py
python ingestion/preprocess.py --limit 2000
python ingestion/index.py --strategy fixed --model e5-small
```

Serve:

```bash
uvicorn app.main:app --reload
```

Open <http://localhost:8000> for the UI -- FastAPI serves `frontend/index.html`
directly at `/` (same origin, so no CORS setup is needed to use it locally).
The UI is a single self-contained HTML file (inline CSS/JS, no build step, no
Node server) that talks only to the versioned public API described in
`docs/api-v1.md`: `POST /api/text/query`, `POST /api/voice/query`,
`GET /api/health`, `GET /api/config`. There is exactly one backend -- this
FastAPI app is the only thing that answers those routes; nothing else needs
to run. Or with Docker:

```bash
docker compose up --build
```

Tests:

```bash
python -m pytest tests/ -q
```

---

## Repository layout

```
app/
  main.py            FastAPI: /health /config /ask /ask/text + UI
  pipeline.py        the orchestration harness
  schemas.py         structured I/O, stage timing, request IDs
  config.py          env-driven config, no secrets in code
  language.py        en/hi/mr detection (the hi-vs-mr problem)
  stt/sarvam.py      Sarvam saaras STT + retries
  stt/elevenlabs.py  Optional ElevenLabs Scribe STT adapter
  tts/sarvam_tts.py  Sarvam bulbul TTS + retries
  retrieval/         embeddings · bm25 · hybrid(RRF) · reranker · qdrant · loader
  generation/        persona (Mando) · generator (LLM + extractive fallback)
  generation/external.py  independently configured external fallback
  guardrails/        input · relevance · grounding
ingestion/
  dataset_info.py    shard-aware inspection (NOT naive streaming)
  download.py        selective: 935 MB of 55.6 GB
  preprocess.py      row-parallel join, verified per record
  chunk.py           six chunking strategies
  index.py           offline index build
evaluation/          see the table above
tests/               390 tests, including product/provider regressions
frontend/index.html  the HHGoa/Mando UI -- self-contained (inline CSS/JS,
                     no build step), served by FastAPI at `/`. Text chat,
                     push-to-talk voice, language selector (auto/en/hi/mr/kok
                     -- exactly what the API accepts), Explore/Timeline/Saved
                     tabs, two themes. Talks only to the public API in
                     docs/api-v1.md; no hardcoded answers or fake backend.
```

---

## Acknowledgements

Dataset: [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).
Speech: [Sarvam AI](https://docs.sarvam.ai). Embeddings:
[multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-small).
Vector store: [Qdrant](https://qdrant.tech).
