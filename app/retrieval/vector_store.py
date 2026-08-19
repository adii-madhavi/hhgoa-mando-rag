"""
Qdrant vector store.

Deployment note
---------------
qdrant-client runs in three modes with an identical API:

    QdrantClient(":memory:")          in-process, nothing persisted
    QdrantClient(path="data/qdrant")  in-process, persisted to disk
    QdrantClient(url="http://...")    the real server (docker-compose)

Development and the offline benchmarks use the local path so no daemon is
required; the container deployment flips one env var. Because the API is the
same, we are not benchmarking against a toy and then shipping something else.

Metadata
--------
Every chunk's metadata (query_id, passage_index, lang, chunk_type, parent_id,
is_selected, strategy) is stored in the payload, which is what makes
metadata-aware retrieval possible: filtering by `lang` at query time is a
server-side filter, not a post-hoc scan.
"""

from __future__ import annotations

import time
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

COLLECTION = "mando_chunks"


class QdrantStore:
    def __init__(self, location: str = "data/qdrant",
                 collection: str = COLLECTION, dim: int = 384,
                 url: str | None = None, prefer_grpc: bool = True):
        if url:
            self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc)
        elif location == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=location)
        self.collection = collection
        self.dim = dim

    def recreate(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dim, distance=models.Distance.COSINE),
        )
        # Indexed payload fields -> filters stay O(log n) instead of a scan.
        for field, schema in (("lang", models.PayloadSchemaType.KEYWORD),
                              ("query_id", models.PayloadSchemaType.INTEGER),
                              ("chunk_type", models.PayloadSchemaType.KEYWORD)):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field, field_schema=schema)

    def upsert_chunks(self, chunks, vectors: np.ndarray,
                      batch_size: int = 512) -> float:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"chunk/vector mismatch: {len(chunks)} vs "
                             f"{vectors.shape[0]}")
        t0 = time.perf_counter()
        for start in range(0, len(chunks), batch_size):
            sl = slice(start, start + batch_size)
            points = []
            for c, v in zip(chunks[sl], vectors[sl]):
                points.append(models.PointStruct(
                    # Qdrant ids must be uuid or int; our chunk_id is a string,
                    # so derive a stable uuid5 from it and keep the original.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                    vector=v.tolist(),
                    payload={
                        "chunk_id": c.chunk_id, "text": c.text,
                        "context": c.context, "chunk_type": c.chunk_type,
                        "strategy": c.strategy, "query_id": c.query_id,
                        "passage_index": c.passage_index, "lang": c.lang,
                        "is_selected": c.is_selected, "parent_id": c.parent_id,
                        **{f"meta_{k}": v2 for k, v2 in c.meta.items()},
                    }))
            self.client.upsert(collection_name=self.collection, points=points)
        return time.perf_counter() - t0

    def search(self, vector: np.ndarray, top_k: int = 50,
               lang: str | None = None):
        flt = None
        if lang:
            flt = models.Filter(must=[models.FieldCondition(
                key="lang", match=models.MatchValue(value=lang))])
        # qdrant-client >=1.16 removed the deprecated .search(); query_points
        # is the current API and returns a QueryResponse wrapping .points.
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        return [(h.payload, float(h.score)) for h in resp.points]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
