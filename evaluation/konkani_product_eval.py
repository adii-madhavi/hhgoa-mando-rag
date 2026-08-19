"""
Konkani PRODUCT evaluation — deliberately separate from the MSMARCO-XI benchmark.

Why this file exists at all
---------------------------
Konkani is a product language: a user may speak it and the system will try to
serve them. It is NOT a dataset-evaluated language: MSMARCO-XI contains zero
Konkani rows under any code (kok/kon/gom/knn), so Recall@k, MRR, false-answer
rate and every other benchmark number in EXPERIMENTS.md simply cannot be
computed for it.

Those two facts are easy to blur, and blurring them is how a submission ends up
claiming "four languages evaluated" on three languages' worth of evidence. So
this harness is separate by construction:

  * separate question file      data/konkani_product_eval.jsonl
  * separate output file        experiments/konkani_product_eval.json
  * separate report section     never merged into MSMARCO-XI tables
  * every output row carries    "benchmark": "konkani_product_v1"
                                "msmarco_xi": false

What can and cannot be measured here
------------------------------------
CAN: refusal behaviour, language of the reply, STT round-tripping, latency,
     whether the system stays inside its evidence, crash-freedom.
CANNOT: retrieval Recall@k or MRR against gold passages -- there are no Konkani
     gold passages. A Konkani query retrieves against en/hi/mr evidence
     cross-lingually, and EXPERIMENTS.md E9 measured that path as weak
     (mr->en MRR 0.223 vs 0.416 in-language). Expect Konkani answer quality to
     be materially worse than the three evaluated languages, and do not report
     otherwise without evidence from this set.

The question set
----------------
`data/konkani_product_eval.jsonl`, one JSON object per line:

    {"id": "kok-001",
     "script": "devanagari" | "roman" | "mixed",
     "kind": "factual" | "unanswerable" | "code_switching",
     "question": "...",
     "expect": "answer" | "refuse",
     "note": "why this item is here"}

`expect` is the ONLY label, and it is coarse on purpose: we can honestly say
whether the system should have answered or refused, but we cannot score answer
correctness without a Konkani reference corpus.

**The question set requires a Konkani speaker to author and review.** This
script does not ship one, and will not invent one -- machine-generated Konkani
questions validated by nobody would be exactly the fabricated evidence the rest
of this project avoids. `--init` writes a template with the required structure
and a handful of clearly-marked PLACEHOLDER rows for a reviewer to replace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG                              # noqa: E402
from app.pipeline import RAGPipeline                       # noqa: E402
from app.retrieval.loader import load_index                # noqa: E402
from app.retrieval.reranker import MMRReranker             # noqa: E402
from app.schemas import (LANG_CAPABILITIES, RAGRequest,    # noqa: E402
                         has_capability)

BENCHMARK_ID = "konkani_product_v1"
DEFAULT_SET = "data/konkani_product_eval.jsonl"

# Template rows. Every one is marked PLACEHOLDER and must be replaced by a
# Konkani speaker before any result from this file means anything. The
# structure shows the required coverage: native script, Roman script, factual,
# unanswerable, and code-switching.
TEMPLATE = [
    {"id": "kok-001", "script": "devanagari", "kind": "factual",
     "question": "PLACEHOLDER — native-script Konkani factual question whose "
                 "answer exists in the English/Hindi/Marathi corpus",
     "expect": "answer",
     "note": "tests cross-lingual retrieval from Konkani into en/hi/mr evidence"},
    {"id": "kok-002", "script": "roman", "kind": "factual",
     "question": "PLACEHOLDER — Romanized Konkani, e.g. 'garud kitlea vegan "
                 "uddta?' style transliteration",
     "expect": "answer",
     "note": "Roman-script Konkani is common in Goa; STT and detection must "
             "cope"},
    {"id": "kok-003", "script": "devanagari", "kind": "unanswerable",
     "question": "PLACEHOLDER — Konkani question with no support in the corpus",
     "expect": "refuse",
     "note": "the refusal path must work in Konkani too"},
    {"id": "kok-004", "script": "mixed", "kind": "code_switching",
     "question": "PLACEHOLDER — Konkani sentence with English technical terms "
                 "mixed in, which is how people actually speak in Goa",
     "expect": "answer",
     "note": "code-switching is the realistic input, not the clean case"},
]


def write_template(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists; refusing to overwrite")
    with open(path, "w", encoding="utf-8") as fh:
        for row in TEMPLATE:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote template to {path}\n")
    print("NEXT STEP, and it is a human one:")
    print("  A Konkani speaker must replace every PLACEHOLDER with a real")
    print("  question and confirm the `expect` label. Target 25-50 items with")
    print("  coverage across: devanagari / roman / mixed script, and")
    print("  factual / unanswerable / code_switching kinds.")
    print("\nThis script will refuse to run while PLACEHOLDER rows remain.")


def load_set(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"No Konkani product set at {path}.\n"
            f"Create the template with:  python {sys.argv[0]} --init\n"
            f"then have a Konkani speaker fill it in.")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    placeholders = [r["id"] for r in rows
                    if "PLACEHOLDER" in r.get("question", "")]
    if placeholders:
        raise SystemExit(
            f"{len(placeholders)} PLACEHOLDER rows remain ({', '.join(placeholders[:5])}...).\n"
            f"These are template stubs, not questions. A Konkani speaker must\n"
            f"replace them before this evaluation reports anything. Refusing\n"
            f"to produce numbers from placeholder text.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true",
                    help="write the template question file and exit")
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--out", default="experiments/konkani_product_eval.json")
    args = ap.parse_args()

    if args.init:
        write_template(args.set)
        return

    rows = load_set(args.set)
    print(f"Konkani product set: {len(rows)} questions from {args.set}")
    print(f"benchmark id: {BENCHMARK_ID}  (NOT MSMARCO-XI)")

    caps = LANG_CAPABILITIES["kok"]
    print(f"\nKonkani capability matrix (verified, not assumed):")
    for k, v in caps.items():
        print(f"  {k:10s} {v}")
    if not has_capability("kok", "tts"):
        print("  -> Konkani answers are TEXT ONLY; bulbul:v3 has no kok-IN voice")
    if not has_capability("kok", "corpus"):
        print("  -> retrieval is cross-lingual into en/hi/mr evidence, which "
              "E9 measured as weak; expect degraded quality")

    loaded = load_index(args.index_dir or CONFIG.index_dir)
    pipeline = RAGPipeline(
        retriever=loaded.retriever, embedder=loaded.embedder,
        reranker=MMRReranker(loaded.embedder,
                             dense_matrix=loaded.retriever.matrix),
        config=CONFIG, dense_matrix=loaded.retriever.matrix)

    results, counts = [], Counter()
    for row in rows:
        resp = pipeline.run(RAGRequest(text=row["question"], language="kok",
                                       want_audio=False))
        answered = not resp.refused
        expected_answer = row["expect"] == "answer"
        correct = answered == expected_answer
        counts[f"{row['kind']}_{'correct' if correct else 'wrong'}"] += 1
        counts["correct" if correct else "wrong"] += 1

        results.append({
            "benchmark": BENCHMARK_ID,
            "msmarco_xi": False,          # machine-readable guard against merging
            "id": row["id"], "script": row["script"], "kind": row["kind"],
            "expect": row["expect"],
            "answered": answered,
            "behaviour_correct": correct,
            "refusal_reason": (resp.refusal_reason.value
                               if resp.refusal_reason else None),
            "reply_language": resp.language.value if resp.language else None,
            "grounded": resp.grounded,
            "total_rag_ms": round(resp.latency.total_rag_ms, 1),
            "answer": (resp.answer or "")[:300],
        })
        print(f"  {row['id']} {row['kind']:14s} expect={row['expect']:7s} "
              f"{'OK ' if correct else 'BAD'} "
              f"lang={results[-1]['reply_language']} "
              f"{resp.latency.total_rag_ms:6.1f}ms")

    n = len(results)
    report = {
        "benchmark": BENCHMARK_ID,
        "msmarco_xi": False,
        "warning": ("PRODUCT evaluation on a curated Konkani set. These "
                    "numbers are NOT MSMARCO-XI benchmark metrics and must "
                    "never be merged into, averaged with, or reported "
                    "alongside them as if comparable."),
        "measurable": ["refusal behaviour", "reply language", "latency",
                       "crash-freedom"],
        "not_measurable": [
            "Recall@k / MRR -- no Konkani gold passages exist",
            "answer correctness -- no Konkani reference answers exist"],
        "capabilities": caps,
        "n_questions": n,
        "behaviour_accuracy": round(counts["correct"] / n, 4) if n else None,
        "by_kind": {k: v for k, v in sorted(counts.items())},
        "results": results,
    }

    print(f"\nbehaviour accuracy (answered-vs-refused matches expectation): "
          f"{report['behaviour_accuracy']}")
    print("NOTE: this measures BEHAVIOUR only. Answer correctness is not "
          "measurable without Konkani reference answers.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
