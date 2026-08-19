"""
End-to-end latency benchmark with per-stage breakdown.

Reporting rules, enforced by this script rather than promised in prose
---------------------------------------------------------------------
1. OFFLINE WORK IS NOT LATENCY. Chunking, embedding the corpus and building the
   index happen in ingestion/index.py, ahead of time. None of it is counted
   here. The runtime path starts when a question arrives.

2. INDEX AND MODEL LOAD ARE NOT LATENCY EITHER. Both happen once at startup.
   The harness loads them before the timing loop and reports the load time
   separately, clearly labelled.

3. STAGES THAT DID NOT RUN ARE REPORTED AS "not measured", NEVER AS 0 ms.
   Without SARVAM_API_KEY there is no STT and no TTS, so end-to-end voice
   latency is genuinely unmeasured; printing 0 ms would understate the real
   figure and misrepresent the system. The output says so explicitly.

4. A WARM-UP PASS IS DISCARDED. The first query pays one-off lazy-init costs
   (BLAS thread pool, tokenizer caches) that no real user experiences after
   startup. It is excluded, and the count of discarded warm-ups is reported.

Percentiles reported: P50, P70, P100 (as required), plus mean/min and P90/P95
because P100 on a small sample is a single unlucky outlier and should never be
read as a service-level number on its own.
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
from app.pipeline import RAGPipeline                       # noqa: E402
from app.retrieval.loader import load_index                # noqa: E402
from app.retrieval.reranker import (CrossEncoderReranker,  # noqa: E402
                                    MMRReranker, NoOpReranker)
from app.schemas import RAGRequest, Stage                  # noqa: E402

STAGE_ORDER = [
    Stage.validate, Stage.stt, Stage.language_detection,
    Stage.input_guardrail, Stage.query_rewrite, Stage.dense_retrieval,
    Stage.bm25_retrieval, Stage.fusion, Stage.rerank, Stage.evidence_check,
    Stage.generation, Stage.grounding, Stage.tts,
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if p >= 100:
        return s[-1]
    idx = min(int(len(s) * p / 100.0), len(s) - 1)
    return s[idx]


def summarise(values: list[float]) -> dict:
    if not values:
        return {"measured": False, "note": "stage did not run"}
    return {
        "measured": True, "n": len(values),
        "p50": round(pct(values, 50), 2), "p70": round(pct(values, 70), 2),
        "p90": round(pct(values, 90), 2), "p95": round(pct(values, 95), 2),
        "p100": round(pct(values, 100), 2),
        "mean": round(statistics.fmean(values), 2),
        "min": round(min(values), 2),
        "stdev": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
    }


def build_reranker(name: str, embedder, dense_matrix=None):
    if name == "cross-encoder":
        return CrossEncoderReranker()
    if name == "none":
        return NoOpReranker()
    return MMRReranker(embedder, dense_matrix=dense_matrix)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--n", type=int, default=120,
                    help="queries per language (>=100 required by the brief)")
    ap.add_argument("--langs", default="en,hi,mr")
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="experiments/latency.json")
    args = ap.parse_args()

    index_dir = args.index_dir or CONFIG.index_dir
    reranker_name = args.reranker or CONFIG.reranker

    print("=" * 74)
    print("MANDO LATENCY BENCHMARK")
    print("=" * 74)

    # ---- startup (explicitly NOT part of any latency figure) -------------
    t0 = time.perf_counter()
    loaded = load_index(index_dir, rrf_k=CONFIG.rrf_k)
    reranker = build_reranker(reranker_name, loaded.embedder,
                              loaded.retriever.matrix)
    pipeline = RAGPipeline(retriever=loaded.retriever,
                           embedder=loaded.embedder, reranker=reranker,
                           config=CONFIG, stt=None, tts=None,
                           dense_matrix=loaded.retriever.matrix)
    startup_s = time.perf_counter() - t0

    generator_kind = type(pipeline.generator).__name__
    print(f"\nSTARTUP (excluded from all latency figures): {startup_s:.1f}s")
    print(f"  index      : {loaded.manifest['n_chunks']} chunks, "
          f"strategy={loaded.manifest['strategy']}, "
          f"model={loaded.manifest['model']}")
    print(f"  reranker   : {reranker_name}")
    print(f"  generator  : {generator_kind}")
    print(f"  STT/TTS    : {'ENABLED' if CONFIG.has_stt else 'DISABLED (no SARVAM_API_KEY)'}")
    if not CONFIG.has_stt:
        print("               -> stt_ms and tts_ms will be reported as "
              "'not measured', NOT as 0")

    # ---- queries ---------------------------------------------------------
    queries = []
    with open(os.path.join(args.processed, "queries.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            queries.append(json.loads(line))
    random.Random(args.seed).shuffle(queries)

    langs = [l.strip() for l in args.langs.split(",")]
    report = {
        "config": {
            "index": loaded.manifest, "reranker": reranker_name,
            "generator": generator_kind,
            "top_k": CONFIG.top_k, "candidate_k": CONFIG.candidate_k,
            "use_bm25": CONFIG.use_bm25,
            "stt_tts_enabled": CONFIG.has_stt,
            "n_per_language": args.n, "warmup_discarded": args.warmup,
        },
        "startup_seconds_excluded": round(startup_s, 2),
        "per_language": {}, "overall": {},
    }

    all_rag, all_e2e = [], []
    all_stages: dict[str, list[float]] = {s.value: [] for s in STAGE_ORDER}

    for lang in langs:
        print(f"\n--- {lang} ---")
        pool = [q for q in queries if q["queries"].get(lang)]
        sample = pool[:args.n + args.warmup]

        rag_ms, e2e_ms, refusals = [], [], 0
        stage_ms: dict[str, list[float]] = {s.value: [] for s in STAGE_ORDER}

        for i, q in enumerate(sample):
            req = RAGRequest(text=q["queries"][lang], language=lang,
                             want_audio=False)
            resp = pipeline.run(req)
            if i < args.warmup:
                continue                     # discard warm-up
            flat = resp.latency.flat()
            rag_ms.append(resp.latency.total_rag_ms)
            e2e_ms.append(resp.latency.end_to_end_ms)
            if resp.refused:
                refusals += 1
            for s in STAGE_ORDER:
                v = flat.get(f"{s.value}_ms", 0.0)
                # Only record stages that actually executed. A stage that was
                # skipped must not contribute a 0 and drag its percentiles down.
                if v > 0 or s in (Stage.validate, Stage.input_guardrail):
                    stage_ms[s.value].append(v)

        entry = {
            "n": len(rag_ms),
            "refusal_rate": round(refusals / max(len(rag_ms), 1), 4),
            "total_rag_ms": summarise(rag_ms),
            "end_to_end_ms": summarise(e2e_ms),
            "stages": {k: summarise(v) for k, v in stage_ms.items()},
        }
        report["per_language"][lang] = entry
        all_rag += rag_ms
        all_e2e += e2e_ms
        for k, v in stage_ms.items():
            all_stages[k] += v

        t = entry["total_rag_ms"]
        print(f"  total_rag  P50={t['p50']:7.2f}  P70={t['p70']:7.2f}  "
              f"P100={t['p100']:7.2f}  mean={t['mean']:7.2f}  n={t['n']}")
        print(f"  refusal rate: {entry['refusal_rate']:.1%}")

    report["overall"] = {
        "total_rag_ms": summarise(all_rag),
        "end_to_end_ms": summarise(all_e2e),
        "stages": {k: summarise(v) for k, v in all_stages.items()},
    }

    print("\n" + "=" * 74)
    print("OVERALL STAGE BREAKDOWN (ms)")
    print("=" * 74)
    print(f"{'stage':22s} {'P50':>9s} {'P70':>9s} {'P100':>9s} {'mean':>9s}")
    for s in STAGE_ORDER:
        d = report["overall"]["stages"][s.value]
        if not d.get("measured"):
            reason = ("no SARVAM_API_KEY" if s in (Stage.stt, Stage.tts)
                      else "did not run")
            print(f"{s.value:22s} {'not measured':>39s}  ({reason})")
        else:
            print(f"{s.value:22s} {d['p50']:9.2f} {d['p70']:9.2f} "
                  f"{d['p100']:9.2f} {d['mean']:9.2f}")

    t = report["overall"]["total_rag_ms"]
    print("-" * 74)
    print(f"{'TOTAL RAG':22s} {t['p50']:9.2f} {t['p70']:9.2f} "
          f"{t['p100']:9.2f} {t['mean']:9.2f}")
    if CONFIG.has_stt:
        e = report["overall"]["end_to_end_ms"]
        print(f"{'END-TO-END VOICE':22s} {e['p50']:9.2f} {e['p70']:9.2f} "
              f"{e['p100']:9.2f} {e['mean']:9.2f}")
    else:
        print(f"{'END-TO-END VOICE':22s} {'NOT MEASURED — STT/TTS disabled':>39s}")

    print(f"\n200 ms budget vs measured total RAG P50: "
          f"{t['p50']:.1f} ms -> "
          f"{'WITHIN' if t['p50'] <= 200 else 'OVER'} budget")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
