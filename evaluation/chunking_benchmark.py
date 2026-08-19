"""
Chunking-strategy and embedding-model benchmark.

Methodology (read this before trusting the numbers)
---------------------------------------------------
1. Corpus = ALL passages for a language (19,987), so distractors are realistic.
   Queries = a sample of ANSWERABLE query_ids only; the 780 no-answer queries
   have no relevant passage and belong to the guardrail calibration set, not to
   a recall benchmark.

2. Ranking is scored at PASSAGE level, not chunk level. Strategies emit wildly
   different chunk counts (window ~4x fixed), and a chunk-level top-k would
   hand multi-chunk strategies more shots at the target. So we retrieve
   top-N chunks, collapse them to their parent passage keeping each passage's
   BEST-scoring chunk, and rank the deduplicated passages. That is what the
   downstream RAG actually consumes, and it is the only comparison that is fair
   across strategies.

3. Relevance comes from is_selected in the shared qrels. Because hinval/marval
   are row-parallel, the SAME qrels apply to en/hi/mr -- so cross-language
   differences here are real retrieval differences, not label noise.

4. Most answerable queries have exactly ONE relevant passage (1169/1220), so
   Recall@1 and MRR are the discriminating metrics. Recall@10 saturates and
   should not drive the decision.

Two-stage design, to keep CPU cost sane:
  stage 1  sweep 6 chunking strategies with one fast model  -> pick strategy
  stage 2  sweep embedding models on the winning strategy   -> pick model
A full cross product would be 6x3x3 = 54 index builds for no extra insight.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

torch.set_num_threads(8)  # measured marginally faster than 16 on this box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.embeddings import Embedder            # noqa: E402
from ingestion.chunk import chunk_records                # noqa: E402

K_VALUES = (1, 3, 5, 10)


def load_corpus(processed: str):
    passages, queries = [], []
    with open(os.path.join(processed, "passages.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            passages.append(json.loads(line))
    with open(os.path.join(processed, "queries.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            queries.append(json.loads(line))
    with open(os.path.join(processed, "qrels.json"), encoding="utf-8") as fh:
        qrels = {int(k): set(v) for k, v in json.load(fh).items()}
    return passages, queries, qrels


def evaluate(embedder: Embedder, chunks, eval_queries, qrels, lang: str,
             top_chunks: int = 50, batch_size: int = 128) -> dict:
    """Index `chunks`, run `eval_queries`, return passage-level metrics."""
    t0 = time.perf_counter()
    mat = embedder.encode_passages([c.text for c in chunks],
                                   batch_size=batch_size)
    index_seconds = time.perf_counter() - t0

    # chunk row -> (query_id, passage_index)
    owners = np.array([(c.query_id, c.passage_index) for c in chunks],
                      dtype=np.int64)

    qtexts = [q["queries"][lang] for q in eval_queries]
    t0 = time.perf_counter()
    qvecs = embedder.encode_queries(qtexts, batch_size=batch_size)
    query_encode_seconds = time.perf_counter() - t0

    hits = {k: 0 for k in K_VALUES}
    rr_total = 0.0
    # Per-query reciprocal rank, kept so configurations can be compared with a
    # PAIRED test. Comparing two aggregate MRRs with marginal standard errors
    # badly understates power when both configs are scored on the same queries.
    per_query: list[dict] = []
    t0 = time.perf_counter()
    for row, q in enumerate(qvecs):
        scores = mat @ q                      # vectors are L2-normalised
        n = min(top_chunks, scores.shape[0])
        cand = np.argpartition(-scores, n - 1)[:n]
        cand = cand[np.argsort(-scores[cand])]

        # Collapse chunks -> passages, keeping best-scoring chunk per passage.
        seen, ranked = set(), []
        for idx in cand:
            key = (int(owners[idx, 0]), int(owners[idx, 1]))
            if key in seen:
                continue
            seen.add(key)
            ranked.append(key)

        gold_qid = eval_queries[row]["query_id"]
        gold = qrels.get(gold_qid, set())
        rank = None
        for pos, (qid, pidx) in enumerate(ranked, start=1):
            if qid == gold_qid and pidx in gold:
                rank = pos
                break
        if rank is not None:
            rr_total += 1.0 / rank
            for k in K_VALUES:
                if rank <= k:
                    hits[k] += 1
        per_query.append({"query_id": gold_qid,
                          "rank": rank,
                          "rr": (1.0 / rank) if rank else 0.0})
    search_seconds = time.perf_counter() - t0

    n_q = len(eval_queries)
    return {
        "lang": lang,
        "n_chunks": len(chunks),
        "n_queries": n_q,
        "chunks_per_passage": round(len(chunks) / max(len(set(
            (c.query_id, c.passage_index) for c in chunks)), 1), 2),
        "mean_chunk_chars": round(
            sum(len(c.text) for c in chunks) / max(len(chunks), 1), 1),
        **{f"recall@{k}": round(hits[k] / n_q, 4) for k in K_VALUES},
        "mrr": round(rr_total / n_q, 4),
        "index_seconds": round(index_seconds, 1),
        "query_encode_ms_per_query": round(1000 * query_encode_seconds / n_q, 2),
        "brute_force_search_ms_per_query": round(
            1000 * search_seconds / n_q, 2),
        "per_query": per_query,
    }


def build_chunks(passages, strategy, embedder, semantic_batch=256):
    """Chunk a passage list, supplying an encoder for semantic/adaptive."""
    encode = None
    if strategy in ("semantic", "adaptive"):
        def encode(sents):
            return embedder.encode_passages(sents, batch_size=semantic_batch)
    return chunk_records(passages, strategy, encode=encode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--langs", default="en,hi,mr")
    ap.add_argument("--strategies",
                    default="fixed,sentence,window,hierarchical,adaptive")
    ap.add_argument("--models", default="e5-small",
                    help="stage 1 uses one fast model; stage 2 re-runs the "
                         "winning strategy across several")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--corpus-queries", type=int, default=600,
                    help="Corpus = the passages of this many sampled "
                         "answerable queries (a SUPERSET of the eval queries, "
                         "so every gold passage is guaranteed in-corpus). "
                         "0 = use every passage in the language. This box "
                         "encodes ~44 texts/s, so the full 20k-passage corpus "
                         "costs ~3.6h for a 6-strategy sweep; 600 queries "
                         "(~6k passages) keeps the distractor pool honest at "
                         "~1/6 the cost. Disclosed in the README.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="stage1")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    passages, queries, qrels = load_corpus(args.processed)

    answerable = [q for q in queries if qrels.get(q["query_id"])]
    random.Random(args.seed).shuffle(answerable)

    # The corpus pool must be a SUPERSET of the eval queries, otherwise gold
    # passages are missing from the index and recall is capped by construction
    # rather than by retrieval quality.
    if args.corpus_queries:
        corpus_qids = {q["query_id"] for q in answerable[:args.corpus_queries]}
    else:
        corpus_qids = None
    eval_queries = answerable[:min(args.n_queries, args.corpus_queries or
                                   len(answerable))]

    print(f"answerable={len(answerable)} eval queries={len(eval_queries)} "
          f"corpus queries={len(corpus_qids) if corpus_qids else 'ALL'}")
    missing = [q["query_id"] for q in eval_queries
               if corpus_qids and q["query_id"] not in corpus_qids]
    assert not missing, f"{len(missing)} eval queries absent from corpus"

    results = []
    for model_key in args.models.split(","):
        print(f"\n### loading {model_key}")
        emb = Embedder(model_key.strip(), batch_size=args.batch_size)
        print(f"    loaded in {emb.load_seconds:.1f}s, dim={emb.spec.dim}")
        latency = emb.time_single_query()
        print(f"    single-query encode: p50={latency['p50_ms']}ms "
              f"p100={latency['p100_ms']}ms")
        results.append({"kind": "query_encode_latency", **latency})

        for lang in args.langs.split(","):
            lang = lang.strip()
            pool = [p for p in passages if p["lang"] == lang
                    and (corpus_qids is None or p["query_id"] in corpus_qids)]
            print(f"  [{lang}] corpus pool = {len(pool)} passages")

            for strategy in args.strategies.split(","):
                strategy = strategy.strip()
                t0 = time.perf_counter()
                chunks = build_chunks(pool, strategy, emb)
                chunk_seconds = time.perf_counter() - t0

                row = evaluate(emb, chunks, eval_queries, qrels, lang,
                               batch_size=args.batch_size)
                row.update({
                    "kind": "retrieval",
                    "model": model_key.strip(),
                    "strategy": strategy,
                    "offline_chunk_seconds": round(chunk_seconds, 1),
                })
                results.append(row)
                print(f"  [{model_key}|{lang}|{strategy:12s}] "
                      f"chunks={row['n_chunks']:6d} "
                      f"R@1={row['recall@1']:.3f} R@5={row['recall@5']:.3f} "
                      f"R@10={row['recall@10']:.3f} MRR={row['mrr']:.3f} "
                      f"(index {row['index_seconds']}s)")

                out = os.path.join(args.out, f"chunking_{args.tag}.json")
                with open(out, "w", encoding="utf-8") as fh:
                    json.dump({"config": vars(args), "results": results},
                              fh, indent=2)

    print(f"\nWrote {args.out}/chunking_{args.tag}.json")


if __name__ == "__main__":
    main()
