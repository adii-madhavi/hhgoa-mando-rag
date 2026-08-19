# MANDO — Engineering Experiment Log

Every number here was produced by a script in this repo and is reproducible via
the command shown. Nothing is estimated or hand-written. Where a result is
worse than expected, it is recorded as-is.

Hardware for all measurements: Windows 11, 16 CPU cores, **no CUDA**
(`torch 2.13.0+cpu`). This matters — it is the binding constraint on model
choice, and it is stated wherever a latency number appears.

---

## E1 — Dataset inventory: what MSMARCO-XI actually contains

**Command:** `python ingestion/dataset_info.py --shards hin,mar --split validation`

**Motivation.** An earlier naive `load_dataset(...)` call began pulling the
entire repo. We needed the language inventory before downloading anything.

**Method note — why streaming was the wrong tool.** The repo is sharded one
parquet file per language. `load_dataset(streaming=True).take(20)` reads only
the *first* shard, so the observed "language distribution" is 100% Assamese and
tells you nothing. The rewritten script enumerates shards via the Hub API and
inspects each independently over HTTP range reads (a few MB, not 55.6 GB).

**Result.**

| Property | Value |
|---|---|
| Total repo size | **55,619 MB** |
| Languages | as bn gu hi kn ml mr ne or pa sa ta te ur (**14**) |
| Code scheme | FLORES-style: `source_lang=eng_Latn`, `target_lang=hin_Deva` / `mar_Deva` |
| English | Not a shard — embedded in every record as `Eng_Query` / `English_passages` |
| **Konkani** | **Absent under every code tried** (`kok`, `kon`, `gom`, `knn`) |
| Telugu | `telval` exists but no `teltrain` (upstream inconsistency; not our concern) |
| Rows per validation shard | 97,941 |
| Passages per record | 10 (99.4% of rows) |

The README's documented `load_dataset("ai4bharat/MSMARCO-XI", "hi")` no longer
works — the loader script is disabled, which is why only the `default` config
is exposed.

**Decision.** Download `validation/hinval.parquet` + `validation/marval.parquet`
only: **935.5 MB, 1.7% of the repo**. The `train/` shards (~3.7 GB each) add
nothing a 3-day demo can index.

---

## E2 — Konkani dropped from scope

**Two independent blockers, both verified rather than assumed.**

1. **No data.** E1 above — MSMARCO-XI has no Konkani under any code.
2. **No voice.** Sarvam TTS `bulbul:v3` does not accept `kok-IN`. Its published
   enum is `bn/en/gu/hi/kn/ml/mr/od/pa/ta/te-IN` only.
   Sarvam **STT** *does* support `kok-IN` — the gap is output-side only.

**Decision.** Ship **English, Hindi, Marathi**. All three have dataset evidence
*and* a native Sarvam voice, so nothing in the demo carries an approximation
caveat. Considered and rejected: voicing Konkani through the Marathi voice
(would require a disclaimer on the flagship language) and machine-translating a
Konkani corpus (translated evidence is a weaker grounding source than the
English originals, and it still leaves no native voice).

---

## E3 — The shards are row-parallel (the key architectural finding)

**Command:** `python ingestion/preprocess.py --limit 2000`

`hinval` and `marval` are not merely both derived from MS MARCO — they are
row-for-row translations of the *same* queries. `preprocess.py` verifies this
per record rather than trusting a sample, asserting that marval's
`English_passages` and `is_selected` are identical to hinval's.

**Result:** `join_mismatches_skipped: 0` across all 2,000 queries.

| Metric | Value |
|---|---|
| Queries | 2,000 |
| — with answer | 1,220 |
| — **no answer** (`is_selected` all-zero) | **780** |
| Passages per language | 19,987 (**59,961** total evidence units) |
| Mean passage chars | en 312.8 · hi 329.0 · mr 326.7 |
| Relevant passages per answerable query | 1 → 1,169 · 2 → 46 · 3 → 3 · 5 → 2 |

**Consequences that shape the whole system.**

- **One shared qrel set across three languages.** A Hindi query and an English
  query for the same `query_id` have literally the same relevant passages, so
  cross-lingual retrieval differences are *real*, not label noise.
- **Recall@1 and MRR are the decisive metrics.** 1,169 of 1,220 answerable
  queries have exactly one relevant passage, so Recall@10 saturates and cannot
  separate configurations.
- **The 780 no-answer queries are a free, labelled refusal set** for
  calibrating the retrieval guardrail — a measured false-refusal rate instead
  of an anecdote.
- Passages are short web snippets (~313–329 chars), so chunking strategies are
  anchored one level up: `query group → passage → sentence-window → sentence`.

---

## E4 — BM25 tokenisation: `re.\w` silently destroys Devanagari

**Found while smoke-testing `app/retrieval/bm25.py`.**

Python's `re` module excludes Unicode categories Mn/Mc (combining marks) from
`\w`. Devanagari carries its vowels as exactly those marks, so every matra and
virama becomes a token boundary:

| Tokeniser | `कॉर्पोरेशन` |
|---|---|
| `re.compile(r"\w+", re.UNICODE)` | `['क', 'र', 'प', 'र', 'शन']` — shattered |
| `regex.compile(r"[\p{L}\p{N}\p{M}]+")` | `['कॉर्पोरेशन']` — correct |

**Why this matters more than it looks.** The failure is silent and
*asymmetric*: English BM25 works perfectly while Hindi/Marathi BM25 degrades
toward random. It would have surfaced as "hybrid retrieval just doesn't help
much for Indic languages" — a plausible-sounding false conclusion — rather than
as an error.

**Decision.** Use the `regex` module with explicit Unicode properties. Added
`regex` as a direct dependency rather than relying on it transitively via
`transformers`. No stemming: there is no reliable light stemmer for Marathi in
the dependency set, and stemming English but not Marathi would bias the
cross-language comparison this project exists to measure.

---

## E5 — Language detection: Hindi vs Marathi

**Command:** `python evaluation/language_detection.py`

Ground truth is free: 2,000 parallel triples → **6,000 labelled cases**, no
annotation. English is a control; hi-vs-mr is the real problem, since both use
Devanagari and script detection cannot separate them.

| Iteration | overall | en | hi | **mr** | hi↔mr confusion |
|---|---|---|---|---|---|
| v1 whole-token markers | 0.8807 | 1.000 | 0.998 | **0.644** | 17.88% |
| v2 + suffix matching, ambiguity fixes | 0.9513 | 1.000 | 0.996 | **0.859** | 7.28% |
| v3 + Marathi finite-verb forms | **0.9538** | 1.000 | 0.996 | **0.866** | 6.90% |

**Root cause of the v1 failure (the interesting part).** Marathi *agglutinates*
case and genitive postpositions onto the noun; Hindi keeps them as separate
words:

```
mr  पोटॅशियमचे    खाद्यपदार्थांचा     ← चे/चा fused onto the noun
hi  पोटैशियम के   खाद्य पदार्थों का    ← standalone tokens
```

Whole-token matching therefore *structurally cannot fire* for Marathi while
working perfectly for Hindi — which is exactly the 0.998 vs 0.644 asymmetry.
The fix is to match Marathi evidence as a **suffix**, not a token.

Two markers were also genuinely shared and were firing at confidence 1.0 in the
wrong direction: `ने` (Hindi ergative, listed as Marathi-only) and `का` (Hindi
genitive *and* Marathi interrogative particle). Both moved to an explicit
ambiguous set. Also added: `ळ`/`ॲ`/`ॅ` as Marathi character evidence vs `ॉ`/`ऑ`
for Hindi loanword vowels.

**Residual 13.4% Marathi error is largely irreducible from text alone** —
queries like `स्टबहब टोल फ्री नंबर` ("StubHub toll free number") are
transliterated English nouns with zero Marathi morphology. Mitigations, in
precedence order: an explicit UI language choice wins; then Sarvam STT's
`language_code` (derived from audio, which we cannot see); then this heuristic.

Detection cost: **p50 0.016 ms**, negligible against the latency budget.

---

## E6 — Chunking strategy sweep

**Command:**
`python evaluation/chunking_benchmark.py --langs en --strategies fixed,sentence,window,semantic,hierarchical,adaptive --n-queries 300 --corpus-queries 600 --models e5-small`

**Methodology (matters for whether these numbers mean anything).**

1. **Scored at passage level, not chunk level.** Strategies emit very different
   chunk counts (window ~3.2× fixed); a chunk-level top-k would hand
   multi-chunk strategies more shots at the target. We retrieve top-50 chunks,
   collapse to parent passages keeping each passage's best-scoring chunk, then
   rank. That is also what the generator actually consumes.
2. **Corpus is a superset of the eval queries.** An early smoke run capped
   recall at 0.05 because the corpus was truncated to the first 1,500 passages
   while eval queries were sampled from all 1,220 answerable ones — most gold
   passages were simply absent. The harness now asserts every eval query's
   passages are in-corpus.
3. **Corpus reduced to 600 queries (~6,000 passages), disclosed.** This box
   encodes ~44 texts/s, so the full 20k-passage corpus costs ~3.6 h for a
   6-strategy sweep. The distractor pool is smaller than production but
   identical across strategies, so the *comparison* is sound.
4. Offline chunking/embedding is excluded from all runtime latency figures.

**Results — English, e5-small, 300 queries, 6,000-passage corpus.**

| Strategy | chunks | chunks/passage | mean chars | R@1 | R@3 | R@5 | R@10 | MRR | chunk cost |
|---|---|---|---|---|---|---|---|---|---|
| hierarchical | 17,593 | 2.94 | 206.6 | 0.440 | 0.840 | **0.933** | 0.970 | **0.650** | 0.3 s |
| adaptive | 11,744 | 1.96 | 154.5 | 0.450 | 0.817 | 0.920 | 0.970 | **0.650** | 26.5 s |
| semantic | 10,374 | 1.73 | 174.9 | **0.460** | 0.823 | 0.913 | 0.967 | 0.649 | 155.5 s |
| fixed (baseline) | 6,822 | 1.14 | 274.8 | 0.443 | 0.827 | 0.920 | **0.973** | 0.648 | 0.0 s |
| sentence | 7,280 | 1.22 | 249.7 | 0.437 | 0.840 | 0.923 | **0.973** | 0.647 | 0.3 s |
| window | 19,411 | 3.24 | 93.0 | 0.410 | 0.780 | 0.887 | 0.920 | 0.607 | 0.3 s |

### The headline is a null result

**Five of the six strategies are tied.** MRR spans 0.647–0.650 — a range of
0.003, against a marginal standard error of roughly 0.02 at n=300. The
elaborate strategies do not beat a fixed-size splitter on this dataset.

This is not a disappointing outcome to be buried; it is the finding, and it has
a clear cause. **MSMARCO passages average 313 characters.** A "chunking
strategy" can only matter when there is enough text to make non-trivial
decisions about where to cut. At 1.14 chunks/passage the fixed baseline is
barely chunking at all — most passages pass through whole — and the
sophisticated strategies are mostly re-deriving the same units by a more
expensive route. Semantic chunking spends **155.5 s** of offline compute to
land within 0.001 MRR of a splitter that costs nothing.

The one strategy that clearly separates does so **downward**: `window` at MRR
0.607. At 93 chars the embedded unit is a lone sentence stripped of context,
which on already-short passages discards more signal than the precision gains
back. Worth stating plainly, because sentence-window is widely recommended —
it is a good technique for long documents and a bad one here.

### Selection

Because quality is tied, the choice is made on secondary criteria, which is a
legitimate engineering answer rather than a failure to find a winner. Paired
bootstrap testing (`evaluation/significance.py`, run on stage 2 where per-query
records are captured) is used to confirm the tie rather than eyeballing
aggregates — pairing matters because all configurations are scored on the same
queries, and between-query variance dominates.

**Chosen: `hierarchical`.**

- Tied-best MRR (0.650) and **best R@5 (0.933)**. R@5 is the decision-relevant
  metric, because `TOP_K=5` is what actually reaches the generator.
- Chunking cost 0.3 s vs semantic's 155.5 s for the same quality.
- The parent/child split is independently useful at generation time: children
  are embedded and retrieved for precision, while the **full parent passage is
  served as context**, so the LLM never receives a decapitated sentence. That
  benefit does not show up in a retrieval metric at all.
- Cost accepted: 2.6× the index size of `fixed` (17.6k vs 6.8k chunks), which
  is a larger brute-force matmul at query time. Quantified in E11.

**Honest caveat:** these are English-only numbers on a 6,000-passage corpus.
Stage 2 re-runs `hierarchical` vs `fixed` across en/hi/mr to confirm the tie
holds in Devanagari before the choice is locked.

> **The caveat was justified — stage 2 reversed this decision. See E7.**

---

## E7 — Cross-language sweep: the English-only choice was wrong

**Command:**
`python evaluation/chunking_benchmark.py --langs en,hi,mr --strategies hierarchical,fixed --n-queries 300 --corpus-queries 600 --models e5-small --tag stage2_models`
then `python evaluation/significance.py experiments/chunking_stage2_models.json --lang <L>`

### Result 1 — retrieval quality collapses across languages

| lang | strategy | R@1 | R@5 | MRR |
|---|---|---|---|---|
| en | hierarchical | 0.440 | 0.933 | **0.650** |
| en | fixed | 0.443 | 0.920 | 0.648 |
| hi | hierarchical | 0.347 | 0.790 | **0.524** |
| hi | fixed | 0.350 | 0.797 | 0.523 |
| mr | hierarchical | 0.263 | 0.653 | 0.432 |
| mr | fixed | 0.290 | 0.703 | **0.459** |

**English 0.650 → Hindi 0.524 → Marathi 0.459.** Marathi MRR is **29% relatively
worse** than English. This is the single largest quality gap in the system and
it is not a chunking problem — it is `multilingual-e5-small`'s representation
quality on Indic languages, compounded by the fact that the Hindi and Marathi
passages are *machine translations* while the English ones are originals.

Reported rather than averaged into a single headline number, because a mean
across three languages would hide the fact that one of the three is
substantially weaker.

### Result 2 — the strategy ranking FLIPS on Marathi

Paired bootstrap over per-query reciprocal ranks, 10,000 resamples,
`fixed` − `hierarchical`:

| lang | ΔMRR | CI95 | p | verdict |
|---|---|---|---|---|
| en | −0.0018 | [−0.0255, +0.0217] | 0.877 | not significant |
| hi | −0.0013 | [−0.0194, +0.0163] | 0.895 | not significant |
| **mr** | **+0.0271** | **[+0.0112, +0.0435]** | **0.0008** | **significant** |

The paired test does exactly the job it was added for. On English and Hindi it
**confirms the tie** rather than letting a 0.002 difference masquerade as a
result. On Marathi it detects a real effect that the marginal standard error
(~0.02) would have dismissed as noise.

**Mechanism.** `hierarchical` splits passages into smaller children (206 chars
vs 275). Where the embedding model is weakest, shorter text carries less signal
— so the strategy that trims context hurts most in the language that can least
afford it. The effect is invisible in English, where the model is strong enough
that context length barely matters.

### Decision — reversed to `fixed`

| criterion | fixed | hierarchical |
|---|---|---|
| English MRR | 0.648 | 0.650 (tie, p=0.88) |
| Hindi MRR | 0.523 | 0.524 (tie, p=0.90) |
| **Marathi MRR** | **0.459** | 0.432 (**fixed wins, p=0.0008**) |
| chunks/language | **6,822** | 17,660 (2.6×) |
| chunking cost | **0 s** | 0.3 s |

`fixed` loses nothing where the strategies tie, wins significantly where they
differ, and produces a **2.6× smaller index** — which directly reduces
brute-force search cost at query time.

**The honest summary of E6 + E7: six chunking strategies were implemented and
benchmarked, and the fixed-size baseline won.** The elaborate strategies are
not merely unnecessary on this dataset; the best of them is measurably *worse*
in Marathi. Had we benchmarked only in English — the obvious shortcut — we
would have shipped the worse strategy for our weakest language and never known.

**What would change this conclusion:** a corpus of long documents (where cut
placement actually matters), or a stronger multilingual encoder that removes
the Marathi context-sensitivity. Both are listed under Pending.

---

## E8 — Hybrid retrieval measurably HURTS (dense-only ships)

**Command:** `python evaluation/retrieval.py --n-queries 250 --cross-lingual`

The brief asked for hybrid dense + BM25 retrieval. It was built, benchmarked,
and **turned off on the evidence.**

| lang | retriever | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| en | **dense-only** | **0.472** | **0.916** | 0.980 | **0.654** |
| en | bm25-only | 0.208 | 0.696 | 0.872 | 0.402 |
| en | hybrid RRF 1:1 | 0.372 | 0.844 | 0.980 | 0.564 |
| hi | **dense-only** | **0.304** | **0.680** | 0.832 | **0.465** |
| hi | bm25-only | 0.148 | 0.408 | 0.496 | 0.249 |
| hi | hybrid RRF 1:1 | 0.220 | 0.580 | 0.776 | 0.378 |
| mr | **dense-only** | **0.268** | **0.612** | 0.748 | **0.416** |
| mr | bm25-only | 0.140 | 0.332 | 0.416 | 0.223 |
| mr | hybrid RRF 1:1 | 0.212 | 0.576 | 0.720 | 0.359 |

Fusing costs **−0.090 MRR in English**. To check this was the fusion weighting
rather than fusion itself, we swept the dense:BM25 weight ratio:

| weights | en | hi | mr |
|---|---|---|---|
| 1:1 | 0.564 | 0.378 | 0.359 |
| 2:1 | 0.592 | 0.414 | 0.379 |
| 3:1 | 0.615 | 0.430 | 0.393 |
| 5:1 | 0.629 | 0.446 | 0.402 |
| **dense-only** | **0.654** | **0.465** | **0.416** |

**Perfectly monotonic in all three languages: every reduction in BM25's weight
improves MRR.** There is no weighting at which lexical retrieval contributes.

**Why.** MS MARCO queries are natural-language questions; passages are
paraphrased web text. Term overlap between the two is weak, and in Hindi and
Marathi the passages are *machine translations* whose word forms differ from
the translated query's. Dense retrieval handles paraphrase; BM25 needs the
overlap that isn't there, so it contributes ranked noise that RRF then gives
equal authority to.

**Decision.** `USE_BM25=false` by default. The implementation and the flag are
kept — hybrid is the right default on corpora with real lexical overlap, and
turning it on is one env var. Note this is a quality decision, not a latency
one: BM25 costs only 0.84 ms after E11's rewrite.

---

## E9 — Cross-lingual retrieval: hypothesis refuted

Hypothesis: the English passages are *originals* while Hindi/Marathi ones are
machine translations, so routing a Marathi query at English evidence through
the shared multilingual space might beat searching its own language — and would
let us index one language instead of three.

| query language | → English-only index | → full index (own language) |
|---|---|---|
| en | 0.568 | 0.654 |
| hi | 0.356 | **0.465** |
| mr | **0.223** | **0.416** |

**Refuted, decisively.** A Marathi query against English evidence scores 0.223
versus 0.416 against Marathi evidence — it nearly halves. `multilingual-e5-small`
does not align Marathi queries with English passages closely enough for
cross-lingual retrieval to substitute for in-language evidence.

Worth recording because the reasoning was sound and the answer was still no.
It also sets a realistic expectation: the shared embedding space is *multilingual*
(it handles each language) but only weakly *cross-lingual* (it does not bridge
between them) at this model size. A larger encoder (bge-m3 is explicitly trained
for cross-lingual retrieval) is the obvious follow-up, untested here for lack of
CPU budget.

Incidental confirmation that some cross-lingual ability exists: in Demo 3, a
Marathi query retrieved a Hindi passage as its third piece of evidence.

---

## E10 — Refusal threshold: the signal is weak, and we say so

**Command:** `python evaluation/threshold_calibration.py --n-per-class 200`

Fitted on the labelled no-answer set (positives = answerable, negatives =
`is_selected` all-zero), sweeping three candidate features:

| lang | feature | separation | best threshold | balanced accuracy |
|---|---|---|---|---|
| en | top_score | +0.0183 | 0.8957 | **0.6525** |
| en | mean_top3 | +0.0177 | 0.8859 | 0.6475 |
| en | margin | +0.0004 | 0.0037 | 0.5450 |
| hi | top_score | +0.0130 | 0.9019 | **0.6325** |
| hi | margin | +0.0005 | 0.0197 | 0.5325 |
| mr | mean_top3 | +0.0081 | 0.8764 | **0.6125** |
| mr | margin | +0.0007 | 0.0024 | 0.5275 |

**The honest headline: balanced accuracy is 0.61–0.65 against 0.50 for a coin
flip.** Retrieval confidence is a weak discriminator on this dataset. `margin`
is worthless (≈0.53). This is not a tuning failure — it is inherent to the task:
a "no-answer" MS MARCO query still has ten *topically related* passages,
because a real search engine returned them. The discrimination is
"relevant vs plausibly-related-but-insufficient", not "relevant vs random".

Compounding it, e5 compresses cosine into a narrow band: positives average
0.909 and negatives 0.891 in English — an 0.018 gap.

**Operating point: 0.87, deliberately NOT the 0.891 balanced-accuracy optimum.**

| threshold | en | hi | mr |
|---|---|---|---|
| 0.86 | 98% / 12% | 84% / 24% | 88% / 18% |
| **0.87 (shipped)** | **94% / 24%** | **74% / 40%** | **76% / 40%** |
| 0.89 (optimum) | 74% / 57% | 62% / 64% | 35% / 82% |

*(answered-of-answerable / refused-of-unanswerable)*

At the balanced optimum the system refuses 26% of answerable English and 54% of
answerable Hindi queries. That is an enormous usability cost to extract slightly
more from a 0.63-accuracy signal. The grounding verifier inspects the actual
answer and is the stronger defence, so this stage ships as a cheap pre-filter.
Stated as the product decision it is; the full curve is in the output file so
the point can be moved with evidence rather than by feel.

**How fragile this is, concretely.** In the demo run, a question the corpus
cannot answer scored 0.867 (correctly refused) while a legitimate Hindi
question scored 0.869 (wrongly refused). Two thousandths apart, opposite
correct answers. That is what a separation of +0.013 means in practice.

---

## E11 — Latency, and the two optimisations that got us under budget

**Command:** `python evaluation/latency.py --n 120`
(120 queries × 3 languages = **360 measured requests**, 3 warm-ups discarded per
language, startup excluded.)

### First measurement: over budget

| stage | P50 | P70 | P100 |
|---|---|---|---|
| dense_retrieval | 21.43 | 22.13 | 43.40 |
| **bm25_retrieval** | **95.16** | 118.80 | 345.44 |
| **generation** (extractive) | **148.81** | 175.45 | 374.17 |
| grounding | 26.79 | 30.87 | 80.82 |
| rerank | 0.18 | 0.18 | 0.42 |
| **TOTAL RAG** | **276.12** | 322.05 | 601.18 |

### Optimisation 1 — stop re-encoding what the index already holds

The naive pipeline encoded the same evidence passages **three times**: once at
retrieval, again in MMR reranking, again in grounding. At ~23 ms per text on
this CPU that is most of the budget. Candidates now carry their row in the
dense matrix, so reranking and grounding look vectors up instead.

A correctness guard was required: the index stores embeddings of `chunk.text`,
while reranking and grounding operate on `chunk.context`. Those are identical
for `fixed` and `sentence` but **not** for hierarchical or sentence-window,
where context is deliberately wider. The shortcut is taken only when the two
strings are equal, otherwise the vectors would score against the wrong text.

    rerank    110 ms -> 0.18 ms
    grounding 120 ms -> 26 ms

### Optimisation 2 — inverted-index BM25

`rank_bm25.get_scores` fills a score array over all 72,411 chunks for every
query term, in Python. `app/retrieval/bm25_fast.py` replaces it with a posting-
list implementation that scores only documents containing a query term.

Verified in `tests/` as a **pure speed change**: identical rankings *and*
identical scores to 1e-4 across English and Devanagari queries.

    bm25  95.16 ms -> 0.84 ms   (113x)

### Final measurement: within budget

| stage | P50 | P70 | P100 | mean |
|---|---|---|---|---|
| validate | 0.00 | 0.00 | 0.00 | 0.00 |
| stt | *not measured* | | | no SARVAM_API_KEY |
| language_detection | 0.01 | 0.01 | 0.04 | 0.01 |
| input_guardrail | 0.06 | 0.06 | 0.15 | 0.06 |
| query_rewrite | *not measured* | | | no follow-ups in this set |
| dense_retrieval | 20.88 | 21.82 | 47.81 | 21.02 |
| bm25_retrieval | *not measured* | | | disabled per E8 |
| fusion | 0.02 | 0.02 | 0.03 | 0.02 |
| rerank | 0.18 | 0.20 | 0.48 | 0.19 |
| evidence_check | 0.02 | 0.02 | 0.05 | 0.02 |
| generation | 94.25 | 105.67 | 222.68 | 98.87 |
| grounding | 26.52 | 29.17 | 54.80 | 27.49 |
| tts | *not measured* | | | no SARVAM_API_KEY |
| **TOTAL RAG** | **137.45** | **149.84** | 289.07 | 123.33 |

Per language (P50 / P70 / P100):
`en 130.98 / 141.95 / 257.22` · `hi 139.78 / 156.00 / 261.11` ·
`mr 144.58 / 161.62 / 289.07`

**137.45 ms P50 and 149.84 ms P70, both within the 200 ms target.** P100 of
289 ms is a single worst-case query and should not be read as a service level.

**What is NOT in these numbers, stated plainly:**
- Offline chunking, embedding (1,865 s) and index build — not runtime.
- Startup: 14.7 s of model and index load, once.
- **STT and TTS are `not measured`, never 0.** Without a Sarvam key the voice
  path does not execute. The full voice loop adds two network round trips and
  will not fit 200 ms; `total_rag_ms` and `end_to_end_ms` are reported
  separately for exactly this reason.
- Generation here is the *extractive* fallback. An LLM call is a network round
  trip and will dominate the budget.

---

## E12 — Guardrails and answer quality

**Command:** `python evaluation/answer_quality.py --n-per-class 60`

### Adversarial: 7/7

Prompt injection blocked in **all three languages**, plus DAN-style jailbreak,
unsafe request, empty input and unintelligible input — each refused with the
*correct* reason, not merely refused. Injection is caught before retrieval
runs (Demo 6: 0.0 ms total, no evidence fetched).

Off-topic: 3 of 4 refused.

### Refusal correctness — the weak point, quantified

| lang | false refusal (of answerable) | **false answer** (of no-answer) | groundedness | answer≈reference |
|---|---|---|---|---|
| en | 10.0% | **78.3%** | 0.952 | 0.895 |
| hi | 20.0% | **65.0%** | 0.940 | 0.873 |
| mr | 31.7% | **46.7%** | 0.925 | 0.861 |

**The system answers most questions the dataset labels unanswerable.** This
follows directly from E10 — a 0.63-accuracy confidence signal cannot do better.

One qualifier that matters for interpreting the number, and which we are
careful not to use as an excuse: in this configuration the generator is
*extractive*, so it physically cannot fabricate — every word is copied from a
retrieved passage. A "false answer" here is returning real evidence text for a
question that evidence does not actually answer: a **relevance** failure, not
an invented fact. It is still wrong, and still reported.

**The architectural fix is already wired and needs a credential.** The pipeline
refuses when the generator reports `sufficient: false`, and the persona prompt
instructs the LLM to set it when the sources do not answer the question. An LLM
that has *read* the passages is a far better judge of sufficiency than a cosine
threshold. `ExtractiveGenerator` cannot make that judgement and always returns
`sufficient=True`, so today the entire burden falls on the weak signal. This is
the single largest expected improvement from adding `LLM_API_KEY`.

Marathi shows the trade-off cleanly: the highest false-refusal rate (31.7%) and
the lowest false-answer rate (46.7%), because its similarity scores sit lower
overall and more of them fall below one fixed threshold.

---

## E2b — Konkani: product language, not an evaluated language

E2 recorded Konkani as "out of scope". That framing was too blunt and has been
corrected. The two things it conflated are now separate, in code as well as in
prose:

| | meaning | languages |
|---|---|---|
| `PRODUCT_LANGS` | a user may speak/select it and the system will try to serve them | en, hi, mr, **kok** |
| `EVALUATED_LANGS` | we hold **labelled benchmark data**, so every number in this file is measured | en, hi, mr |

**Konkani is a product language. It is not a dataset-evaluated language.**
MSMARCO-XI contains zero Konkani rows under any code, so Recall@k, MRR,
false-answer rate and balanced accuracy **cannot be computed** for it. No
Konkani number appears anywhere in this file, and none will be manufactured.

### Verified capability matrix (`app/schemas.LANG_CAPABILITIES`)

| lang | STT | TTS | corpus | benchmark |
|---|---|---|---|---|
| en / hi / mr | ✅ | ✅ | ✅ | ✅ |
| **kok** | ✅ `kok-IN` | ❌ not in bulbul:v3 | ❌ none | ❌ none |

Three consequences, each enforced in code rather than trusted to a comment:

1. **`SARVAM_TTS_LANG` is a separate, narrower map than `SARVAM_LANG`.**
   `synthesize()` raises for Konkani instead of silently substituting the
   Marathi voice. TTS failure is non-fatal in the pipeline, so a Konkani user
   gets the text answer — the honest degradation.
2. **Refusal copy has no Konkani strings.** `refusal_text_with_locale()`
   returns `(english_text, in_requested_language=False)` so the caller can
   label it. Marathi is explicitly *not* used as a stand-in — that is the same
   mislabelling rejected for TTS. Writing Konkani user-facing copy requires a
   Konkani speaker.
3. **Retrieval is cross-lingual into en/hi/mr evidence**, the path E9 measured
   as weak (mr→en MRR 0.223 vs 0.416 in-language). Konkani answer quality
   should be expected to be materially worse, and this is stated rather than
   discovered by a judge.

### Observed behaviour (smoke test, not a benchmark)

Two Konkani queries through the real pipeline. Reported as observations, with
n=2, explicitly not as metrics:

| input | outcome |
|---|---|
| `गरुड कितल्या वेगान उडटा` (Devanagari) | answered in **122.8 ms** — but the reply came back in **Marathi**, because the extractive generator returns verbatim evidence and the evidence is Marathi |
| `garud kitlea vegan uddta` (Roman) | refused, insufficient evidence — Roman-script Konkani does not match Devanagari embeddings |

The Marathi-reply problem is a property of the extractive baseline, which can
only echo retrieved text. An LLM generator instructed to reply in the user's
language would fix it; that is Phase 3, gated behind Phase 2.

### The separate product evaluation

`evaluation/konkani_product_eval.py`, structurally incapable of contaminating
the benchmark: separate question file, separate output, and every row stamped
`"benchmark": "konkani_product_v1", "msmarco_xi": false`.

What it **can** measure: refusal behaviour, reply language, latency,
crash-freedom. What it **cannot**: Recall@k or answer correctness, because
there are no Konkani gold passages or reference answers.

`--init` writes a template covering native script / Roman script / factual /
unanswerable / code-switching. **The script refuses to run while PLACEHOLDER
rows remain** — machine-invented Konkani questions validated by nobody would be
exactly the fabricated evidence this project avoids. A Konkani speaker must
author the 25–50 items.

---

## E13 — LLM answerability judge (Phase 1 built, Phase 2 BLOCKED)

**Status: implemented and unit-tested. NOT YET MEASURED — no `LLM_API_KEY`.**
This section deliberately contains no accuracy numbers, because none have been
produced. It will be filled in by
`python evaluation/answerability_calibration.py` the moment a key exists.

### Motivation

E10 established that cosine similarity is a weak answerability signal
(balanced accuracy 0.61–0.65 vs 0.50 chance), and E12 showed the consequence: a
false-answer rate of 78% / 65% / 47%. The cause is structural — an MS MARCO
"no-answer" query still retrieves ten *topically related* passages, because a
real search engine returned them. Cosine measures topical relatedness, which is
precisely the axis on which answerable and unanswerable queries do **not**
differ.

The hypothesis: a model that has *read* the passages can separate "about the
right subject" from "contains the fact being asked for". **This is a hypothesis
until measured.**

### What was built

`app/guardrails/answerability.py` — a judge returning strict JSON:

```json
{"sufficient": false, "confidence": 0.94,
 "supporting_passage_ids": [],
 "reason": "The passages are topically related but do not contain enough information."}
```

`app/generation/llm_client.py` — a shared OpenAI-compatible client with retry
classification (429/5xx retried, 4xx fails fast), timeouts, and tolerant JSON
recovery. `generator.py` was left untouched; migrating it is Phase 3.

**Anti-fabrication is enforced in code, not entrusted to the prompt:**

1. Passages are relabelled `p1..pN` per request, so no corpus ID from training
   data can be cited.
2. Returned IDs are intersected with the offered set; invented ones are
   stripped and logged.
3. **A `sufficient: true` verdict whose citations are *all* invalid is
   downgraded to insufficient** — a claim of support we cannot verify is not
   support.

**Failure policy.** If the judge cannot produce a valid verdict (no key, API
down, repeated schema violations) it returns `available=False` and the cosine
decision stands, with a warning recorded. An LLM outage degrades the system to
its previous behaviour rather than refusing everything. `fail_closed=True`
inverts this for deployments preferring silence to a weaker guard.

**Placement.** The judge runs *after* the cosine guard, so the expensive check
is never spent on queries the free one already rejects. When the judge approves,
the evidence is narrowed to the passages it actually cited — a stronger
grounding constraint than "top k by cosine".

### A design flaw the tests caught

The first `confidence` validator treated any value > 1.0 as a percentage, so a
malformed `1.7` was silently coerced to `0.017` — inventing a low-confidence
verdict out of a schema violation. Now only values unambiguously on a
percentage scale (2–100) are coerced; `1.7` is rejected and triggers the
corrective retry. Regression-tested both ways.

### Phase 2 experiment design (pre-registered, before any numbers exist)

`evaluation/answerability_calibration.py` compares, per language:

| system | description |
|---|---|
| **A** | cosine threshold only — the incumbent |
| **B** | LLM judge only |
| **C** | cosine AND judge — the production composition |

Reporting false-answer rate, false-refusal rate, precision, recall and balanced
accuracy, plus `judge_ms`.

**Pre-registered decision rule, recorded now so it cannot be adjusted to fit
the result:** the judge ships only if B or C beats A's balanced accuracy
(0.6525 en / 0.6325 hi / 0.6100 mr) **and** the false-answer reduction
justifies the added latency. C can only ever *reduce* the answer rate relative
to A, since it adds a second gate — so it must be judged on whether fewer false
answers is worth more false refusals, not on accuracy alone.

The script **exits rather than emitting numbers** when no key is present.

### Konkani

Phase 2 asked for a Konkani breakdown. There is no labelled Konkani data:
MSMARCO-XI ships none under any code (E1), which is one of the two reasons
Konkani is out of scope (E2). `--langs` accepts it, and the script will find
zero rows. Results will cover en/hi/mr.

---

## E14 — Phase 2 calibration run #1: INVALID (methodology bug, documented)

**Command:** `python evaluation/answerability_calibration.py --n-per-class 100 --concurrency 6 --timeout 120`
**Raw:** `experiments/answerability_calibration.json` (600 per-query rows)

**Verdict: the pre-registered gate was NOT applied. This run is not valid
evidence.** The numbers below are recorded only so the failure is auditable.

### Measured (contaminated — do not cite)

| lang | system | falseAns | falseRef | precision | recall | balAcc |
|---|---|---|---|---|---|---|
| en | A cosine only | 0.8000 | 0.0600 | 0.5402 | 0.9400 | 0.5700 |
| en | B judge only | 0.5300 | 0.3800 | 0.5391 | 0.6200 | 0.5450 |
| en | C cosine + judge | 0.4300 | 0.3900 | 0.5865 | 0.6100 | 0.5900 |
| hi | A cosine only | 0.6400 | 0.2000 | 0.5556 | 0.8000 | 0.5800 |
| hi | B judge only | 0.2700 | 0.4200 | 0.6824 | 0.5800 | 0.6550 |
| hi | C cosine + judge | 0.2100 | 0.4600 | 0.7200 | 0.5400 | 0.6650 |
| mr | A cosine only | 0.5500 | 0.2400 | 0.5802 | 0.7600 | 0.6050 |
| mr | B judge only | 0.2400 | 0.4400 | 0.7000 | 0.5600 | 0.6600 |
| mr | C cosine + judge | 0.2200 | 0.4900 | 0.6986 | 0.5100 | 0.6450 |

Judge-unavailable counts: **en 41 · hi 27 · mr 35 = 103 of 600**.
Downgraded verdicts: 0. Invalid citations: 0 (the anti-fabrication rule never
had to fire).

Judge latency (ms), all 600 calls:

| lang | n | P50 | P70 | P100 | mean |
|---|---|---|---|---|---|
| en | 200 | 940 | 1,033 | 57,782 | 1,941 |
| hi | 200 | 8,638 | 14,346 | 96,801 | 12,085 |
| mr | 200 | 2,688 | 14,304 | 72,757 | 10,232 |
| **all** | 600 | **1,027** | **9,486** | **96,801** | 8,086 |

Successful calls only (n=497): P50 918 · P70 5,597 · P100 96,801.

### Why it is invalid

| cause | count | fix applied |
|---|---|---|
| HTTP 429 rate limit | 41 | backoff was 0.3 s base / 3 s cap / 3 attempts — too fast and too few to outlast a rate-limit window. Now **2 s / 30 s / 5 attempts**. |
| `content=None` (reasoning truncation) | 62 | `max_tokens=1500` still truncated the hard tail. Now **3000**. |

A failed verdict **fails open** (`sufficient=True`), so each is scored as
"answered". That alone would be tolerable noise. What invalidates the run is
that the failures were **not randomly distributed**:

| lang | failures | on negatives | on positives |
|---|---|---|---|
| **en** | 41 | **41** | **0** |
| hi | 27 | 11 | 16 |
| mr | 35 | 9 | 26 |

**Every English failure landed on a negative.** The harness built
`items = positives + negatives` and processed them in that order; rate-limit
pressure accumulated across the run and therefore struck the back half, which
for English is entirely negatives. Fail-open on a negative *is* a false answer,
so English B/C false-answer rates were inflated **by call ordering, not by
judge quality**.

Recomputing English on successful calls only: B false-answer
**0.5300 → 0.2034**, balanced accuracy **0.5450 → 0.7083**. A swing that large
can reverse the gate verdict, which is exactly why the run cannot be used in
either direction.

**Fix:** the two classes are now shuffled together
(`random.Random(seed + 1).shuffle(items)`), so any residual failure rate is
class-independent. The harness records `judge_failure_rate` and
`results_trustworthy` per language and warns loudly above 2%.

### Secondary observation (not a gate input)

System A here scores 0.5700 / 0.5800 / 0.6050, below the E10 pre-registered
baseline of 0.6525 / 0.6325 / 0.6100. The two are measured differently: E10
thresholds raw dense candidates, while this harness thresholds the top-5 list
*after* MMR reranking. Within a single run A/B/C are directly comparable
because all three consume identical candidates; across runs they are not. When
run #2 lands, the gate is applied against **A as measured in that same run**,
with the E10 figures as context rather than as the literal threshold.

---

## Pending / not done

- **e5-base and bge-m3 comparison.** Started, then abandoned: e5-base is ~3×
  slower on this CPU and the sweep would have taken ~2 h competing with the
  index build. e5-small's per-query encode is 15.3 ms; a 278M or 568M model
  would likely breach the latency budget on CPU regardless of quality. bge-m3
  is the obvious candidate for fixing E9's cross-lingual failure, since it is
  explicitly trained for it.
- **Cross-encoder reranking.** Implemented (`CrossEncoderReranker`) but not
  benchmarked — on this CPU it would add several hundred ms, and MMR already
  costs 0.18 ms. Would need a GPU to be a real option.
- **Phase 2 answerability calibration.** Blocked on `LLM_API_KEY`. This gates
  Phases 3-11 (generation, grounding, Sarvam, TTS, voice loop, UI), because the
  judge must be validated before the generation layer is built on top of it.
- **STT/TTS measurement.** Blocked on `SARVAM_API_KEY`. The clients are
  written and verified against Sarvam's published API contract, but they have
  not been exercised against the live service.
- **Mando persona output.** Blocked on `LLM_API_KEY`.
- Long-document corpus, which is where the chunking strategies of E6 would
  actually be expected to matter.
