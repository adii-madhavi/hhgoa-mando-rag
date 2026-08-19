"""
Build the unified three-language corpus from the downloaded MSMARCO-XI shards.

Why this joins rather than concatenates
---------------------------------------
hinval.parquet and marval.parquet are row-parallel: same 97,941 query_ids,
same query_type histogram, same is_selected labels, same English passages.
So we can JOIN them on (query_id, passage_index) and emit, for each source
passage, three language views of the SAME evidence:

    en : passages.English_passages[i]      (identical in both shards)
    hi : hinval  passages.Translated_passages[i]
    mr : marval  passages.Translated_passages[i]

That gives one shared qrel set across all three languages, which is what makes
cross-lingual retrieval evaluation exact instead of approximate: a Hindi query
and an English query for the same query_id have literally the same relevant
passage indices.

Outputs (data/processed/)
-------------------------
queries.jsonl   one row per query_id, with the query and answer in all 3 langs
passages.jsonl  one row per (query_id, passage_index, lang) -- the raw evidence
                units that ingestion/chunk.py will later split
qrels.json      query_id -> list of relevant passage_index (from is_selected)
stats.json      corpus statistics, for the README

Note: ~50% of MSMARCO-XI queries have is_selected all-zero and
Answer == "No Answer Present". Those are kept and flagged has_answer=false --
they are the refusal test set for the retrieval guardrail.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import pyarrow.parquet as pq

COLUMNS = ["query_id", "query_type", "query", "Answer",
           "Eng_Query", "Eng_Answer", "passages"]

NO_ANSWER_EN = "no answer present."


def read_shard(path: str, limit: int) -> dict:
    """Read up to `limit` rows of a shard, keyed by query_id."""
    rows = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512, columns=COLUMNS):
        for row in batch.to_pylist():
            rows[row["query_id"]] = row
            if len(rows) >= limit:
                return rows
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--limit", type=int, default=2000,
                    help="number of query_ids to keep. 2000 queries ~ 20k "
                         "passages per language ~ 60k evidence units total, "
                         "which is a sane CPU embedding budget.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    hi_path = os.path.join(args.raw, "validation", "hinval.parquet")
    mr_path = os.path.join(args.raw, "validation", "marval.parquet")
    for p in (hi_path, mr_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run ingestion/download.py first")

    print(f"reading {hi_path} (limit {args.limit}) ...")
    hi_rows = read_shard(hi_path, args.limit)
    print(f"reading {mr_path} (limit {args.limit}) ...")
    mr_rows = read_shard(mr_path, args.limit)

    shared = [qid for qid in hi_rows if qid in mr_rows]
    print(f"hi rows={len(hi_rows)} mr rows={len(mr_rows)} shared={len(shared)}")
    if not shared:
        raise SystemExit("no shared query_ids -- shards are not row-parallel; "
                         "re-check the join assumption before proceeding")

    q_out = open(os.path.join(args.out, "queries.jsonl"), "w", encoding="utf-8")
    p_out = open(os.path.join(args.out, "passages.jsonl"), "w", encoding="utf-8")

    qrels: dict[str, list[int]] = {}
    stats = Counter()
    passage_len = Counter()
    eng_mismatch = 0

    for qid in shared:
        h, m = hi_rows[qid], mr_rows[qid]
        hp, mp = h["passages"], m["passages"]

        eng = hp["English_passages"]
        sel = hp["is_selected"]
        hi_p = hp["Translated_passages"]
        mr_p = mp["Translated_passages"]

        # Guard the join assumption instead of trusting it.
        if mp["English_passages"] != eng or mp["is_selected"] != sel:
            eng_mismatch += 1
            continue
        n = min(len(eng), len(hi_p), len(mr_p), len(sel))

        has_answer = bool(sum(sel)) and \
            str(h["Eng_Answer"]).strip().lower() != NO_ANSWER_EN

        q_out.write(json.dumps({
            "query_id": qid,
            "query_type": h["query_type"],
            "has_answer": has_answer,
            "queries": {"en": h["Eng_Query"], "hi": h["query"], "mr": m["query"]},
            "answers": {"en": h["Eng_Answer"], "hi": h["Answer"], "mr": m["Answer"]},
        }, ensure_ascii=False) + "\n")

        qrels[str(qid)] = [i for i in range(n) if sel[i]]
        stats["queries"] += 1
        stats["queries_with_answer" if has_answer else "queries_no_answer"] += 1

        for i in range(n):
            for lang, text in (("en", eng[i]), ("hi", hi_p[i]), ("mr", mr_p[i])):
                text = (text or "").strip()
                if not text:
                    stats[f"empty_{lang}"] += 1
                    continue
                p_out.write(json.dumps({
                    "doc_id": f"{qid}:{i}:{lang}",
                    "query_id": qid,
                    "passage_index": i,
                    "lang": lang,
                    "is_selected": bool(sel[i]),
                    "text": text,
                }, ensure_ascii=False) + "\n")
                stats[f"passages_{lang}"] += 1
                passage_len[lang] += len(text)

    q_out.close()
    p_out.close()

    with open(os.path.join(args.out, "qrels.json"), "w", encoding="utf-8") as fh:
        json.dump(qrels, fh)

    summary = {
        "source": "ai4bharat/MSMARCO-XI validation split (hinval + marval)",
        "languages": ["en", "hi", "mr"],
        "join_mismatches_skipped": eng_mismatch,
        "counts": dict(stats),
        "mean_passage_chars": {
            lang: round(passage_len[lang] / max(stats[f"passages_{lang}"], 1), 1)
            for lang in ("en", "hi", "mr")
        },
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}/{{queries,passages}}.jsonl, qrels.json, stats.json")


if __name__ == "__main__":
    main()
