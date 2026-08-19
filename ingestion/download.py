"""
Selective MSMARCO-XI download.

We need exactly TWO files out of the 55.6 GB repo:

    validation/hinval.parquet   (462 MB)
    validation/marval.parquet   (474 MB)

Rationale (see EXPERIMENTS.md / dataset_info.py output):

* English is NOT a separate shard. It lives inside every record as
  Eng_Query / Eng_Answer / English_passages, so both shards carry it.
* hinval and marval are ROW-PARALLEL: both have 97,941 rows over the same
  query_ids with identical is_selected labels. Downloading these two gives us
  English + Hindi + Marathi over a single shared qrel set.
* The train/ shards are ~3.7 GB each and add nothing we need. A 98k-query
  validation split is already far more corpus than a 3-day RAG demo can index.

Konkani is absent from this dataset under every code (kok/kon/gom/knn) and is
out of scope for this project.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

DATASET_NAME = "ai4bharat/MSMARCO-XI"

# lang_key -> repo path. Keep this list short and explicit; never glob the repo.
SHARDS = {
    "hi": "validation/hinval.parquet",
    "mr": "validation/marval.parquet",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--langs", default="hi,mr")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    token = os.environ.get("HF_TOKEN")  # optional: higher rate limits
    if not token:
        print("[warn] HF_TOKEN not set -- downloads work but are rate limited.")

    wanted = [l.strip() for l in args.langs.split(",") if l.strip()]
    total = 0
    for lang in wanted:
        if lang not in SHARDS:
            raise SystemExit(f"unknown lang {lang!r}; known: {sorted(SHARDS)}")
        path = SHARDS[lang]
        print(f"[{lang}] downloading {path} ...")
        local = hf_hub_download(
            DATASET_NAME,
            path,
            repo_type="dataset",
            local_dir=args.out,
            token=token,
        )
        size = os.path.getsize(local)
        total += size
        print(f"[{lang}] -> {local}  ({size / 1e6:.1f} MB)")

    print(f"\nDone. {total / 1e6:.1f} MB downloaded "
          f"(repo total is 55,619 MB -- we took {100 * total / 55.6e9:.1f}%).")


if __name__ == "__main__":
    main()
