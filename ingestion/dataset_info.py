"""
MSMARCO-XI dataset inspection.

Design notes
------------
The repo is sharded ONE PARQUET FILE PER LANGUAGE (train/<lang>train.parquet,
validation/<lang>val.parquet), ~55.6 GB in total. Two consequences:

1. Never call load_dataset() without restricting data_files -- it resolves the
   whole repo and downloads every language.
2. A naive `streaming=True` + `.take(20)` reads only the FIRST shard, so the
   "language distribution" you would observe is 100% Assamese. It tells you
   nothing. This script therefore inspects each shard independently.

Everything here is read over HTTP range requests (parquet footer + a few row
groups), so total bytes pulled is a few MB, not 55 GB.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field, asdict

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

DATASET_NAME = "ai4bharat/MSMARCO-XI"

# Languages we actually care about for MANDO, with every plausible ISO code.
TARGET_LANGUAGES = {
    "English": ("eng", "en"),
    "Hindi": ("hin", "hi"),
    "Marathi": ("mar", "mr"),
    "Konkani": ("kok", "kon", "gom", "knn", "gom-deva"),
}


@dataclass
class ShardReport:
    path: str
    split: str
    lang_hint: str
    size_mb: float
    num_rows: int = 0
    num_row_groups: int = 0
    source_langs: dict = field(default_factory=dict)
    target_langs: dict = field(default_factory=dict)
    query_types: dict = field(default_factory=dict)
    sampled_rows: int = 0
    passages_per_row: dict = field(default_factory=dict)
    selected_per_row: dict = field(default_factory=dict)
    aligned_lengths: int = 0


def list_shards(api: HfApi):
    """Return (path, split, lang_hint, size_mb) for every parquet shard."""
    info = api.repo_info(DATASET_NAME, repo_type="dataset", files_metadata=True)
    shards = []
    for sib in info.siblings:
        name = sib.rfilename
        if not name.endswith(".parquet"):
            continue
        split = name.split("/")[0]
        stem = name.split("/")[-1].removesuffix(".parquet")
        lang_hint = stem.removesuffix("train").removesuffix("val")
        shards.append((name, split, lang_hint, (sib.size or 0) / 1e6))
    return sorted(shards)


def inspect_shard(fs, path, split, lang_hint, size_mb, sample_rows):
    rep = ShardReport(path=path, split=split, lang_hint=lang_hint, size_mb=size_mb)

    with fs.open(f"datasets/{DATASET_NAME}/{path}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        rep.num_rows = pf.metadata.num_rows
        rep.num_row_groups = pf.metadata.num_row_groups

        cols = ["source_lang", "target_lang", "query_type", "passages"]
        src, tgt, qt = Counter(), Counter(), Counter()
        npass, nsel = Counter(), Counter()
        seen = 0
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            for row in batch.to_pylist():
                src[row["source_lang"]] += 1
                tgt[row["target_lang"]] += 1
                qt[row["query_type"]] += 1
                p = row["passages"]
                eng = p["English_passages"]
                tr = p["Translated_passages"]
                sel = p["is_selected"]
                npass[len(eng)] += 1
                nsel[sum(sel)] += 1
                if len(eng) == len(tr) == len(sel):
                    rep.aligned_lengths += 1
                seen += 1
                if seen >= sample_rows:
                    break
            if seen >= sample_rows:
                break

        rep.sampled_rows = seen
        rep.source_langs = dict(src)
        rep.target_langs = dict(tgt)
        rep.query_types = dict(qt.most_common())
        rep.passages_per_row = {str(k): v for k, v in sorted(npass.items())}
        rep.selected_per_row = {str(k): v for k, v in sorted(nsel.items())}
    return rep


def show_records(fs, path, n):
    print("\n" + "=" * 78)
    print(f"FIRST {n} RECORDS OF {path}")
    print("=" * 78)
    with fs.open(f"datasets/{DATASET_NAME}/{path}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        rows = next(pf.iter_batches(batch_size=n)).to_pylist()[:n]
    for i, r in enumerate(rows, 1):
        p = r["passages"]
        print(f"\n--- Record {i} | query_id={r['query_id']} "
              f"| {r['source_lang']} -> {r['target_lang']} "
              f"| type={r['query_type']}")
        print(f"  query      : {str(r['query'])[:180]}")
        print(f"  Eng_Query  : {str(r['Eng_Query'])[:180]}")
        print(f"  Answer     : {str(r['Answer'])[:180]}")
        print(f"  Eng_Answer : {str(r['Eng_Answer'])[:180]}")
        print(f"  passages   : eng={len(p['English_passages'])} "
              f"translated={len(p['Translated_passages'])} "
              f"is_selected={p['is_selected']}")
        if p["English_passages"]:
            print(f"  eng[0]     : {p['English_passages'][0][:160]}")
        if p["Translated_passages"]:
            print(f"  trans[0]   : {p['Translated_passages'][0][:160]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="",
                    help="comma-separated lang hints, e.g. hin,mar. Empty = all")
    ap.add_argument("--split", default="validation",
                    choices=["train", "validation", "both"])
    ap.add_argument("--sample-rows", type=int, default=512)
    ap.add_argument("--records", type=int, default=20)
    ap.add_argument("--out", default="data/dataset_info.json")
    args = ap.parse_args()

    api, fs = HfApi(), HfFileSystem()

    print("=" * 78)
    print(f"MSMARCO-XI INVENTORY  ({DATASET_NAME})")
    print("=" * 78)

    shards = list_shards(api)
    total = sum(s[3] for s in shards)
    langs_by_split = {}
    for path, split, lang, mb in shards:
        langs_by_split.setdefault(split, set()).add(lang)
        print(f"  {path:32s} {mb:9.1f} MB")
    print(f"  {'TOTAL':32s} {total:9.1f} MB  "
          f"({total / 1000:.1f} GB -- do NOT download blindly)")

    print("\nLanguage shards present per split:")
    for split, langs in sorted(langs_by_split.items()):
        print(f"  {split:11s}: {sorted(langs)}")
    all_langs = set().union(*langs_by_split.values())

    print("\nTARGET LANGUAGE AVAILABILITY (as dedicated shards):")
    for name, codes in TARGET_LANGUAGES.items():
        hit = sorted(all_langs & set(codes))
        status = f"PRESENT as {hit}" if hit else "NOT PRESENT as a shard"
        print(f"  {name:9s} tried {str(list(codes)):40s} -> {status}")

    wanted = [s for s in shards
              if (not args.shards or s[2] in args.shards.split(","))
              and (args.split == "both" or s[1] == args.split)]

    reports = []
    for path, split, lang, mb in wanted:
        print("\n" + "-" * 78)
        print(f"Inspecting {path} ({mb:.0f} MB on disk)")
        rep = inspect_shard(fs, path, split, lang, mb, args.sample_rows)
        reports.append(rep)
        print(f"  rows in shard      : {rep.num_rows:,} "
              f"({rep.num_row_groups} row groups)")
        print(f"  rows sampled       : {rep.sampled_rows}")
        print(f"  source_lang values : {rep.source_langs}")
        print(f"  target_lang values : {rep.target_langs}")
        print(f"  query_type values  : {rep.query_types}")
        print(f"  passages per row   : {rep.passages_per_row}")
        print(f"  is_selected sum    : {rep.selected_per_row}")
        print(f"  eng/trans/sel lengths aligned: "
              f"{rep.aligned_lengths}/{rep.sampled_rows}")

    if wanted and args.records:
        show_records(fs, wanted[0][0], args.records)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {"shards": [{"path": p, "split": s, "lang": l, "size_mb": mb}
                        for p, s, l, mb in shards],
             "reports": [asdict(r) for r in reports]},
            fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
