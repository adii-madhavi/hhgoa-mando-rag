"""
Streaming generation benchmark.

Measures what streaming actually buys, against the real degradation policy
(hard deadline, incomplete-answer marking), not a toy demo.

Fields per query, matching the requested instrumentation:
    retrieval_ms                  cosine-guard critical path
    time_to_first_token_ms        first raw delta off the wire (TTFT)
    time_to_first_visible_text_ms first user-visible prose (TTFV)
    generation_ms                 == stream_duration_ms, the LLM call
    stream_duration_ms
    total_rag_ms                  retrieval + generation, text in -> answer out
    end_to_end_ms                 == total_rag_ms here (no STT/TTS yet)

Failure/timeout accounting is explicit: a run that hits the deadline before any
visible text is a REFUSAL (correct behaviour, not a system failure) and is
counted separately from a transport/API failure.
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
from app.generation.generator import Generation, GenerationError  # noqa: E402
from app.generation.generator import LLMGenerator           # noqa: E402
from app.guardrails.relevance import assess_evidence        # noqa: E402
from app.retrieval.loader import load_index                 # noqa: E402
from app.retrieval.reranker import MMRReranker              # noqa: E402


def pct(v, p):
    s = sorted(v)
    return s[min(int(len(s) * p / 100), len(s) - 1)] if s else float("nan")


def summarise(v):
    if not v:
        return {"measured": False}
    return {"measured": True, "n": len(v), "p50": round(pct(v, 50), 1),
            "p70": round(pct(v, 70), 1), "p100": round(pct(v, 100), 1),
            "mean": round(statistics.fmean(v), 1), "min": round(min(v), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--langs", default="en,hi,mr")
    ap.add_argument("--n", type=int, default=10, help="queries per language")
    ap.add_argument("--deadline", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="experiments/streaming_latency.json")
    args = ap.parse_args()

    gen = LLMGenerator(api_key=CONFIG.llm_api_key, model=CONFIG.llm_model,
                       base_url=CONFIG.llm_base_url, timeout_s=args.timeout,
                       deadline_s=args.deadline)
    if not gen.available:
        raise SystemExit("No LLM key configured; refusing to emit numbers "
                         "that were not measured.")

    loaded = load_index(CONFIG.index_dir)
    reranker = MMRReranker(loaded.embedder,
                           dense_matrix=loaded.retriever.matrix)
    thresholds = {"min_top_score": CONFIG.evidence_min_top,
                  "min_mean_top_k": CONFIG.evidence_min_mean}

    queries = []
    with open(os.path.join(args.processed, "queries.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            queries.append(json.loads(line))
    indexed = {c.query_id for c in loaded.chunks}
    pool = [q for q in queries if q["has_answer"] and q["query_id"] in indexed]
    random.Random(args.seed).shuffle(pool)

    report = {"config": {"model": gen.model, "deadline_s": args.deadline,
                         "n_per_language": args.n},
              "note": ("time_to_first_token != time_to_first_visible_text: "
                       "the stream carries JSON, so the first bytes are "
                       "'{\"answer\": \"' and are not shown to the user. "
                       "TTFV is measured from the first decoded prose "
                       "character."),
              "per_language": {}}

    all_ttft, all_ttfv, all_gen, all_total = [], [], [], []
    total_attempts = total_deadline_refusals = total_failures = 0

    for lang in [l.strip() for l in args.langs.split(",")]:
        print(f"\n=== {lang} ===", flush=True)
        retrieval_ms, ttft, ttfv, gen_ms, total_ms = [], [], [], [], []
        refused_by_guard = deadline_refusals = incomplete = failures = 0

        for i, q in enumerate(pool[:args.n], 1):
            text = q["queries"][lang]
            t0 = time.perf_counter()

            cands, _ = loaded.retriever.retrieve(
                text, top_k=CONFIG.top_k, candidate_k=CONFIG.candidate_k,
                use_dense=True, use_bm25=CONFIG.use_bm25)
            cands, _ = reranker.rerank(text, cands, top_k=CONFIG.top_k)
            decision = assess_evidence(cands, thresholds)
            retrieval_elapsed = (time.perf_counter() - t0) * 1000
            retrieval_ms.append(retrieval_elapsed)
            total_attempts += 1

            if not decision.sufficient:
                refused_by_guard += 1
                print(f"  {i}/{args.n} [{lang}] cosine-refused "
                      f"retrieval={retrieval_elapsed:.0f}ms", flush=True)
                continue

            evidence_texts = [c.context for c in cands]
            final = None
            try:
                for item in gen.generate_stream(text, evidence_texts, lang):
                    if isinstance(item, Generation):
                        final = item
            except GenerationError as exc:
                if "deadline" in str(exc):
                    deadline_refusals += 1
                    total_deadline_refusals += 1
                    print(f"  {i}/{args.n} [{lang}] DEADLINE before any "
                          f"visible text", flush=True)
                else:
                    failures += 1
                    total_failures += 1
                    print(f"  {i}/{args.n} [{lang}] FAILURE: {exc}",
                         flush=True)
                continue

            if final.incomplete:
                incomplete += 1

            ttft.append(final.time_to_first_token_ms)
            all_ttft.append(final.time_to_first_token_ms)
            ttfv.append(final.time_to_first_visible_text_ms)
            all_ttfv.append(final.time_to_first_visible_text_ms)
            gen_ms.append(final.stream_duration_ms)
            all_gen.append(final.stream_duration_ms)
            t = retrieval_elapsed + final.stream_duration_ms
            total_ms.append(t)
            all_total.append(t)

            print(f"  {i}/{args.n} [{lang}] ttft={final.time_to_first_token_ms:6.0f}ms "
                  f"ttfv={final.time_to_first_visible_text_ms:6.0f}ms "
                  f"gen={final.stream_duration_ms:7.0f}ms "
                  f"incomplete={final.incomplete}", flush=True)

        report["per_language"][lang] = {
            "n_attempted": args.n,
            "cosine_refused": refused_by_guard,
            "deadline_refusals": deadline_refusals,
            "incomplete_answers": incomplete,
            "failures": failures,
            "retrieval_ms": summarise(retrieval_ms),
            "time_to_first_token_ms": summarise(ttft),
            "time_to_first_visible_text_ms": summarise(ttfv),
            "generation_ms": summarise(gen_ms),
            "total_rag_ms": summarise(total_ms),
        }

    report["overall"] = {
        "time_to_first_token_ms": summarise(all_ttft),
        "time_to_first_visible_text_ms": summarise(all_ttfv),
        "generation_ms": summarise(all_gen),
        "total_rag_ms": summarise(all_total),
        "n_attempted": total_attempts,
        "deadline_refusal_rate": round(
            total_deadline_refusals / max(total_attempts, 1), 4),
        "failure_rate": round(total_failures / max(total_attempts, 1), 4),
    }

    print("\n" + "=" * 74)
    print(f"{'metric':30s} {'P50':>9s} {'P70':>9s} {'P100':>9s}")
    for key, label in (("time_to_first_token_ms", "TTFT"),
                       ("time_to_first_visible_text_ms", "TTFV"),
                       ("generation_ms", "generation"),
                       ("total_rag_ms", "TOTAL (retrieval+generation)")):
        m = report["overall"][key]
        if m.get("measured"):
            print(f"{label:30s} {m['p50']:9.1f} {m['p70']:9.1f} {m['p100']:9.1f}")
    print(f"\ndeadline refusal rate: {report['overall']['deadline_refusal_rate']:.1%}")
    print(f"failure rate:          {report['overall']['failure_rate']:.1%}")
    print(f"n attempted:           {total_attempts}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
