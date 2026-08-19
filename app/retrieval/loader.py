"""Load a prebuilt index and assemble a ready-to-query HybridRetriever."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np

from app.retrieval.bm25_fast import FastBM25
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid import HybridRetriever
from ingestion.chunk import Chunk


@dataclass
class LoadedIndex:
    retriever: HybridRetriever
    embedder: Embedder
    chunks: list
    manifest: dict
    load_seconds: float


def _chunk_from_dict(d: dict) -> Chunk:
    return Chunk(**d)


def load_index(index_dir: str = "data/index", build_bm25: bool = True,
               rrf_k: int = 60) -> LoadedIndex:
    t0 = time.perf_counter()

    manifest_path = os.path.join(index_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"no index at {index_dir}. Build one first:\n"
            f"  python ingestion/index.py --strategy semantic --model e5-small")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    chunks = []
    with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            chunks.append(_chunk_from_dict(json.loads(line)))

    dense = np.load(os.path.join(index_dir, "dense.npy"))
    if dense.shape[0] != len(chunks):
        raise ValueError(f"index corrupt: {len(chunks)} chunks but "
                         f"{dense.shape[0]} vectors")

    embedder = Embedder(manifest["model"])
    if embedder.spec.dim != dense.shape[1]:
        raise ValueError(
            f"manifest model {manifest['model']} has dim {embedder.spec.dim} "
            f"but index vectors are {dense.shape[1]}-dim -- the index was "
            f"built with a different model")

    # FastBM25, not rank_bm25: identical rankings and scores (verified in
    # tests) but the inverted index scores only documents containing a query
    # term instead of the whole 72k-chunk corpus. rank_bm25 was measured at
    # 95 ms P50, the largest single stage in the runtime path.
    bm25 = FastBM25([c.text for c in chunks]) if build_bm25 else None
    retriever = HybridRetriever(chunks, dense, bm25, embedder, rrf_k=rrf_k)

    return LoadedIndex(retriever=retriever, embedder=embedder, chunks=chunks,
                       manifest=manifest,
                       load_seconds=time.perf_counter() - t0)
