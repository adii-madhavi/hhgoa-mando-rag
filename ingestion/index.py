"""
Offline index build.

THIS IS NOT ON THE RUNTIME PATH. Chunking, embedding and index construction all
happen here, ahead of time, and none of their cost is counted in any latency
figure this project reports. The runtime path begins at a user's question.

Artifacts written to <index_dir>/:
    chunks.jsonl    every chunk with full metadata
    dense.npy       float32 (n_chunks, dim), L2-normalised
    manifest.json   model, strategy, dim, counts, build timings

The dense matrix is kept as a plain .npy alongside Qdrant on purpose: brute
force over ~60k x 384 floats is a single ~25 MB matmul, which on this box is
faster than a network round trip to a vector DB and lets the latency benchmark
isolate retrieval cost from transport cost. Qdrant is used for the served
deployment (--qdrant), where payload filtering and persistence matter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.embeddings import Embedder          # noqa: E402
from ingestion.chunk import chunk_records              # noqa: E402


def load_passages(processed: str, langs: list[str], limit_queries: int = 0):
    rows = []
    keep = None
    if limit_queries:
        qids = []
        with open(os.path.join(processed, "queries.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                qids.append(json.loads(line)["query_id"])
                if len(qids) >= limit_queries:
                    break
        keep = set(qids)

    with open(os.path.join(processed, "passages.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["lang"] not in langs:
                continue
            if keep is not None and rec["query_id"] not in keep:
                continue
            rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--langs", default="en,hi,mr")
    ap.add_argument("--strategy", default="semantic")
    ap.add_argument("--model", default="e5-small")
    ap.add_argument("--limit-queries", type=int, default=0,
                    help="0 = all 2000 queries")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--qdrant", action="store_true",
                    help="also upsert into a local Qdrant collection")
    args = ap.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    langs = [l.strip() for l in args.langs.split(",")]

    passages = load_passages(args.processed, langs, args.limit_queries)
    print(f"passages: {len(passages)} across {langs}")

    emb = Embedder(args.model, batch_size=args.batch_size)
    print(f"model {args.model} dim={emb.spec.dim} loaded in "
          f"{emb.load_seconds:.1f}s")

    encode = None
    if args.strategy in ("semantic", "adaptive"):
        def encode(sents):
            return emb.encode_passages(sents, batch_size=256)

    t0 = time.perf_counter()
    chunks = chunk_records(passages, args.strategy, encode=encode)
    chunk_s = time.perf_counter() - t0
    print(f"chunks: {len(chunks)} via {args.strategy} in {chunk_s:.1f}s")

    t0 = time.perf_counter()
    vectors = emb.encode_passages([c.text for c in chunks],
                                  batch_size=args.batch_size)
    embed_s = time.perf_counter() - t0
    print(f"embedded {vectors.shape} in {embed_s:.1f}s "
          f"({len(chunks) / max(embed_s, 1e-9):.0f} chunks/s)")

    with open(os.path.join(args.index_dir, "chunks.jsonl"), "w",
              encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.as_dict(), ensure_ascii=False) + "\n")
    np.save(os.path.join(args.index_dir, "dense.npy"), vectors)

    qdrant_s = None
    if args.qdrant:
        from app.retrieval.vector_store import QdrantStore
        store = QdrantStore(location=os.path.join(args.index_dir, "qdrant"),
                            dim=emb.spec.dim)
        store.recreate()
        qdrant_s = store.upsert_chunks(chunks, vectors)
        print(f"qdrant: {store.count()} points in {qdrant_s:.1f}s")

    manifest = {
        "model": args.model, "hf_id": emb.spec.hf_id, "dim": emb.spec.dim,
        "strategy": args.strategy, "langs": langs,
        "n_passages": len(passages), "n_chunks": len(chunks),
        "mean_chunk_chars": round(
            sum(len(c.text) for c in chunks) / max(len(chunks), 1), 1),
        "offline_seconds": {
            "chunking": round(chunk_s, 1), "embedding": round(embed_s, 1),
            "qdrant_upsert": round(qdrant_s, 1) if qdrant_s else None},
        "note": "offline build; not counted in runtime latency",
    }
    with open(os.path.join(args.index_dir, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("\n" + json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
