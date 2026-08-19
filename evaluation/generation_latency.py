"""
PHASE 3 — how long does grounded Mando generation actually take?

The 137 ms figure in EXPERIMENTS.md E11 is the RETRIEVAL core measured with the
extractive generator. It does not describe a response that involves an LLM. This
script measures what a real user waits for, split into:

    retrieval_critical_path_ms   everything except generation (the E11 path)
    generation_ms                the LLM call
    total_rag_ms                 the two together, text in -> answer out
    judge_async_ms               background verification, NEVER in total_rag_ms

Reported as P50/P70/P100 per language, never averaged into one number.

The judge runs in ASYNC mode here, exactly as in production, so its cost is
excluded from the critical path by construction rather than by subtraction.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG                              # noqa: E402
from app.generation.generator import LLMGenerator          # noqa: E402
from app.guardrails.answerability import AnswerabilityJudge  # noqa: E402
from app.pipeline import RAGPipeline                       # noqa: E402
from app.retrieval.loader import load_index                # noqa: E402
from app.retrieval.reranker import MMRReranker             # noqa: E402
from app.schemas import RAGRequest                         # noqa: E402


def pct(v, p):
    s = sorted(v)
    return s[min(int(len(s) * p / 100), len(s) - 1)] if s else float("nan")


def summarise(v):
    if not v:
        return {"measured": False}
    return {"measured": True, "n": len(v),
            "p50": round(pct(v, 50), 1), "p70": round(pct(v, 70), 1),
            "p100": round(pct(v, 100), 1),
            "mean": round(statistics.fmean(v), 1),
            "min": round(min(v), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--langs", default="en,hi,mr")
    ap.add_argument("--n", type=int, default=10, help="queries per language")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="experiments/generation_latency.json")
    args = ap.parse_args()

    gen = LLMGenerator(api_key=CONFIG.llm_api_key, model=CONFIG.llm_model,
                       base_url=CONFIG.llm_base_url, timeout_s=args.timeout)
    if not gen.available:
        raise SystemExit("No LLM key configured; refusing to emit numbers "
                         "that were not measured.")

    loaded = load_index(CONFIG.index_dir)
    pipeline = RAGPipeline(
        retriever=loaded.retriever, embedder=loaded.embedder,
        reranker=MMRReranker(loaded.embedder,
                             dense_matrix=loaded.retriever.matrix),
        config=CONFIG, dense_matrix=loaded.retriever.matrix, generator=gen,
        judge=AnswerabilityJudge(timeout_s=args.timeout), judge_mode="async",
        index_version=str(loaded.manifest["n_chunks"]))

    queries = []
    with open(os.path.join(args.processed, "queries.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            queries.append(json.loads(line))
    indexed = {c.query_id for c in loaded.chunks}
    pool = [q for q in queries if q["has_answer"] and q["query_id"] in indexed]
    random.Random(args.seed).shuffle(pool)

    report = {"config": {"model": gen.model, "n_per_language": args.n,
                         "judge_mode": "async",
                         "index": loaded.manifest},
              "note": ("generation is an LLM network round trip; it is NOT "
                       "part of the 137 ms retrieval core measured in E11"),
              "per_language": {}}

    for lang in [l.strip() for l in args.langs.split(",")]:
        print(f"\n=== {lang} ===", flush=True)
        crit, gen_ms, total, refused = [], [], [], 0
        for i, q in enumerate(pool[:args.n], 1):
            r = pipeline.run(RAGRequest(text=q["queries"][lang],
                                        language=lang, want_audio=False))
            f = r.latency.flat()
            g = f["generation_ms"]
            t = r.latency.total_rag_ms
            gen_ms.append(g)
            total.append(t)
            crit.append(t - g)              # everything except the LLM call
            refused += r.refused
            print(f"  {i}/{args.n} total={t:8.0f}ms gen={g:8.0f}ms "
                  f"crit={t - g:6.1f}ms refused={r.refused}", flush=True)

        report["per_language"][lang] = {
            "n": len(total), "refusals": refused,
            "retrieval_critical_path_ms": summarise(crit),
            "generation_ms": summarise(gen_ms),
            "total_rag_ms": summarise(total),
        }

    pipeline.shutdown()
    report["judge_async_ms"] = summarise(pipeline.judge_async_ms)
    report["judge_async_note"] = ("background verification; excluded from "
                                  "total_rag_ms by construction")

    print("\n" + "=" * 70)
    print(f"{'lang':5s} {'stage':26s} {'P50':>9s} {'P70':>9s} {'P100':>9s}")
    for lang, e in report["per_language"].items():
        for key, label in (("retrieval_critical_path_ms", "retrieval core"),
                           ("generation_ms", "generation (LLM)"),
                           ("total_rag_ms", "TOTAL text-in/answer-out")):
            m = e[key]
            print(f"{lang:5s} {label:26s} {m['p50']:9.1f} {m['p70']:9.1f} "
                  f"{m['p100']:9.1f}")
    j = report["judge_async_ms"]
    if j.get("measured"):
        print(f"{'-':5s} {'judge (async, excluded)':26s} {j['p50']:9.1f} "
              f"{j['p70']:9.1f} {j['p100']:9.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
