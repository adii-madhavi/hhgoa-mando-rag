"""
Language-detection accuracy on the real corpus.

Ground truth is free here: MSMARCO-XI gives us the SAME query in en/hi/mr for
every query_id, so 2,000 queries become 6,000 labelled detection cases with no
annotation effort. Hindi-vs-Marathi is the only hard pair; English is a control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.language import detect_language          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="experiments/language_detection.json")
    args = ap.parse_args()

    queries = []
    with open(os.path.join(args.processed, "queries.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            queries.append(json.loads(line))

    confusion = defaultdict(Counter)
    methods = defaultdict(Counter)
    low_conf = Counter()
    samples = defaultdict(list)
    latencies = []

    for q in queries:
        for gold in ("en", "hi", "mr"):
            text = q["queries"][gold]
            t0 = time.perf_counter()
            res = detect_language(text)
            latencies.append((time.perf_counter() - t0) * 1000)
            confusion[gold][res.lang] += 1
            methods[gold][res.method] += 1
            if res.confidence < 0.5:
                low_conf[gold] += 1
            if res.lang != gold and len(samples[gold]) < 5:
                samples[gold].append(
                    {"text": text[:90], "predicted": res.lang,
                     "confidence": res.confidence, "method": res.method,
                     "scores": res.scores})

    latencies.sort()
    n = len(queries)
    report = {"n_queries": n, "n_cases": n * 3, "per_language": {},
              "latency_ms": {
                  "p50": round(latencies[len(latencies) // 2], 4),
                  "p100": round(latencies[-1], 4),
                  "mean": round(sum(latencies) / len(latencies), 4)}}

    print(f"{'gold':5s} {'acc':>7s} {'n':>6s}   confusion")
    overall_correct = 0
    for gold in ("en", "hi", "mr"):
        c = confusion[gold]
        correct = c[gold]
        overall_correct += correct
        report["per_language"][gold] = {
            "accuracy": round(correct / n, 4),
            "confusion": dict(c),
            "methods": dict(methods[gold]),
            "low_confidence": low_conf[gold],
            "error_samples": samples[gold],
        }
        print(f"{gold:5s} {correct / n:7.3f} {n:6d}   {dict(c)}")
    report["overall_accuracy"] = round(overall_correct / (n * 3), 4)
    print(f"\noverall accuracy: {report['overall_accuracy']:.4f}")
    print(f"detection latency p50={report['latency_ms']['p50']}ms "
          f"p100={report['latency_ms']['p100']}ms")

    hi_mr = confusion["hi"]["mr"] + confusion["mr"]["hi"]
    print(f"hi<->mr confusions: {hi_mr} of {2 * n} "
          f"({100 * hi_mr / (2 * n):.2f}%)")

    for gold in ("hi", "mr"):
        if samples[gold]:
            print(f"\nsample {gold} errors:")
            for s in samples[gold][:3]:
                print(f"  {s['predicted']} ({s['confidence']}, {s['method']}) "
                      f"<- {s['text']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
