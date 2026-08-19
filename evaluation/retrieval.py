"""
Retrieval evaluation against the built index.

Three questions, each an ablation rather than an assertion:

A. DOES HYBRID BEAT DENSE?  dense-only vs bm25-only vs RRF fusion, per
   language. Hybrid retrieval is standard advice, but on a corpus of
   machine-translated web snippets it is not obvious that lexical matching adds
   anything for Devanagari, so we measure the delta instead of assuming it.

B. DOES THE RERANKER EARN ITS LATENCY?  none vs MMR vs cross-encoder, scored
   with the SAME candidate list, reporting both the quality delta and the
   measured per-query cost. A reranker that adds 0.01 MRR for 400 ms is a bad
   trade at a 200 ms budget, and this is where that becomes visible.

C. DOES CROSS-LINGUAL RETRIEVAL WORK?  A Hindi query searching an
   ENGLISH-ONLY index. This is the interesting case: MSMARCO-XI's English
   passages are the originals, and the translated ones are machine output, so
   English evidence is the higher-quality source. If a Hindi query can reach it
   through the shared multilingual embedding space, we can index English once
   instead of three parallel corpora -- a 3x index saving. Because hinval and
   marval are row-parallel, the same qrels apply, so this comparison is exact.

Scored at passage level with per-query records, so configurations can be
compared with evaluation/significance.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG                              # noqa: E402
from app.retrieval.bm25_fast import FastBM25                # noqa: E402
from app.retrieval.hybrid import HybridRetriever           # noqa: E402
from app.retrieval.loader import load_index                # noqa: E402
from app.retrieval.reranker import (CrossEncoderReranker,  # noqa: E402
                                    MMRReranker, NoOpReranker)

K_VALUES = (1, 3, 5, 10)


def score(candidates, gold_qid: int, gold_idx: set) -> tuple[int | None, dict]:
    rank = None
    for pos, c in enumerate(candidates, start=1):
        if c.query_id == gold_qid and c.passage_index in gold_idx:
            rank = pos
            break
    hits = {k: int(rank is not None and rank <= k) for k in K_VALUES}
    return rank, hits


def run_config(retriever, queries, qrels, lang, *, use_dense, use_bm25,
               reranker=None, top_k=10, candidate_k=50, label=""):
    hits = {k: 0 for k in K_VALUES}
    rr = 0.0
    per_query, lat_dense, lat_bm25, lat_rr = [], [], [], []

    for q in queries:
        text = q["queries"][lang]
        cands, timing = retriever.retrieve(
            text, top_k=top_k, candidate_k=candidate_k,
            use_dense=use_dense, use_bm25=use_bm25)
        lat_dense.append(timing.dense_ms)
        lat_bm25.append(timing.bm25_ms)

        if reranker is not None:
            t0 = time.perf_counter()
            cands, _ = reranker.rerank(text, cands, top_k=top_k)
            lat_rr.append((time.perf_counter() - t0) * 1000)

        gold = qrels.get(q["query_id"], set())
        rank, h = score(cands, q["query_id"], gold)
        for k in K_VALUES:
            hits[k] += h[k]
        rr += (1.0 / rank) if rank else 0.0
        per_query.append({"query_id": q["query_id"], "rank": rank,
                          "rr": (1.0 / rank) if rank else 0.0})

    n = len(queries)
    out = {
        "label": label, "lang": lang, "n_queries": n,
        "use_dense": use_dense, "use_bm25": use_bm25,
        "reranker": getattr(reranker, "name", "none"),
        **{f"recall@{k}": round(hits[k] / n, 4) for k in K_VALUES},
        "mrr": round(rr / n, 4),
        "dense_ms_p50": round(float(np.percentile(lat_dense, 50)), 2),
        "bm25_ms_p50": round(float(np.percentile(lat_bm25, 50)), 2)
        if any(lat_bm25) else 0.0,
        "rerank_ms_p50": round(float(np.percentile(lat_rr, 50)), 2)
        if lat_rr else 0.0,
        "per_query": per_query,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--langs", default="en,hi,mr")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--with-cross-encoder", action="store_true",
                    help="include the cross-encoder reranker (slow on CPU)")
    ap.add_argument("--cross-lingual", action="store_true",
                    help="also test hi/mr queries against an English-only index")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", default="experiments/retrieval.json")
    args = ap.parse_args()

    loaded = load_index(args.index_dir or CONFIG.index_dir)
    print(f"index: {loaded.manifest['n_chunks']} chunks "
          f"strategy={loaded.manifest['strategy']} "
          f"model={loaded.manifest['model']} "
          f"loaded in {loaded.load_seconds:.1f}s")

    queries, qrels = [], {}
    with open(os.path.join(args.processed, "queries.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            queries.append(json.loads(line))
    with open(os.path.join(args.processed, "qrels.json"),
              encoding="utf-8") as fh:
        qrels = {int(k): set(v) for k, v in json.load(fh).items()}

    # Only queries whose gold passage is actually in this index.
    indexed_qids = {c.query_id for c in loaded.chunks}
    answerable = [q for q in queries
                  if qrels.get(q["query_id"]) and q["query_id"] in indexed_qids]
    random.Random(args.seed).shuffle(answerable)
    eval_queries = answerable[:args.n_queries]
    print(f"eval queries: {len(eval_queries)} "
          f"(answerable and present in index)")

    results = []
    mmr = MMRReranker(loaded.embedder, dense_matrix=loaded.retriever.matrix)

    for lang in [l.strip() for l in args.langs.split(",")]:
        print(f"\n=== {lang} ===")

        # --- A: retriever ablation ---------------------------------------
        for label, dense, bm25 in (("dense-only", True, False),
                                   ("bm25-only", False, True),
                                   ("hybrid-rrf", True, True)):
            r = run_config(loaded.retriever, eval_queries, qrels, lang,
                           use_dense=dense, use_bm25=bm25,
                           top_k=args.top_k, candidate_k=args.candidate_k,
                           label=label)
            results.append({"kind": "retriever_ablation", **r})
            print(f"  {label:12s} R@1={r['recall@1']:.3f} "
                  f"R@5={r['recall@5']:.3f} R@10={r['recall@10']:.3f} "
                  f"MRR={r['mrr']:.3f} "
                  f"(dense {r['dense_ms_p50']}ms bm25 {r['bm25_ms_p50']}ms)")

        # --- B: reranker ablation -----------------------------------------
        rerankers = [("none", NoOpReranker()), ("mmr", mmr)]
        if args.with_cross_encoder:
            try:
                rerankers.append(("cross-encoder", CrossEncoderReranker()))
            except Exception as exc:
                print(f"  cross-encoder unavailable: {exc}")

        for name, rr in rerankers:
            r = run_config(loaded.retriever, eval_queries, qrels, lang,
                           use_dense=True, use_bm25=True, reranker=rr,
                           top_k=5, candidate_k=args.candidate_k,
                           label=f"rerank:{name}")
            results.append({"kind": "reranker_ablation", **r})
            print(f"  rerank:{name:14s} R@1={r['recall@1']:.3f} "
                  f"R@5={r['recall@5']:.3f} MRR={r['mrr']:.3f} "
                  f"cost={r['rerank_ms_p50']}ms")

    # --- C: cross-lingual --------------------------------------------------
    if args.cross_lingual:
        print("\n=== cross-lingual: non-English query -> English-only index ===")
        en_rows = [i for i, c in enumerate(loaded.chunks) if c.lang == "en"]
        if not en_rows:
            print("  index contains no English chunks; skipping")
        else:
            en_chunks = [loaded.chunks[i] for i in en_rows]
            en_dense = np.load(
                os.path.join(args.index_dir or CONFIG.index_dir,
                             "dense.npy"))[en_rows]
            en_bm25 = FastBM25([c.text for c in en_chunks])
            en_retriever = HybridRetriever(en_chunks, en_dense, en_bm25,
                                           loaded.embedder, rrf_k=CONFIG.rrf_k)
            print(f"  english-only index: {len(en_chunks)} chunks")

            for lang in ("en", "hi", "mr"):
                r = run_config(en_retriever, eval_queries, qrels, lang,
                               use_dense=True, use_bm25=True,
                               top_k=args.top_k,
                               candidate_k=args.candidate_k,
                               label=f"crosslingual:{lang}->en")
                results.append({"kind": "cross_lingual", **r})
                print(f"  {lang} query -> en index: R@1={r['recall@1']:.3f} "
                      f"R@5={r['recall@5']:.3f} MRR={r['mrr']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "index": loaded.manifest,
                   "results": results}, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
