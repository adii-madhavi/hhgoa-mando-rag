"""
Paired significance testing between benchmark configurations.

Why this exists
---------------
Stage 1 produced MRRs of 0.650, 0.650, 0.649, 0.648, 0.647 across five chunking
strategies. Declaring a winner from a 0.003 gap would be noise-chasing. But the
naive check -- comparing each aggregate against its own marginal standard error
-- is too conservative here, because every configuration is scored on the SAME
queries. Pairing removes the between-query variance, which is by far the
largest term (some queries are just easy).

So we use a paired bootstrap over per-query reciprocal ranks:
    1. resample the query set with replacement, B times
    2. recompute the MRR *difference* on each resample
    3. report the 95% interval of that difference

If the interval spans zero, the configurations are indistinguishable on this
data and the choice must be made on secondary criteria (index size, build
cost, latency) -- which is a legitimate engineering answer, not a failure.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np


def paired_bootstrap(rr_a: np.ndarray, rr_b: np.ndarray, n_boot: int = 10000,
                     seed: int = 7) -> dict:
    if rr_a.shape != rr_b.shape:
        raise ValueError(f"unpaired inputs: {rr_a.shape} vs {rr_b.shape}")
    rng = np.random.default_rng(seed)
    n = len(rr_a)
    diff = rr_a - rr_b
    observed = float(diff.mean())

    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Two-sided p via the sign-flip null (H0: the difference is symmetric
    # about zero), which suits paired data without assuming normality.
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    null = (diff * signs).mean(axis=1)
    p = float((np.abs(null) >= abs(observed)).mean())

    return {"delta_mrr": round(observed, 5),
            "ci95_low": round(float(lo), 5),
            "ci95_high": round(float(hi), 5),
            "p_value": round(p, 4),
            "significant": bool(lo > 0 or hi < 0),
            "n_queries": n}


def load(path: str, lang: str | None = None) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for r in data["results"]:
        if r["kind"] != "retrieval" or "per_query" not in r:
            continue
        if lang and r["lang"] != lang:
            continue
        key = f"{r['model']}|{r['lang']}|{r['strategy']}"
        pq = sorted(r["per_query"], key=lambda x: x["query_id"])
        out[key] = {"rr": np.array([x["rr"] for x in pq], dtype="float64"),
                    "qids": [x["query_id"] for x in pq],
                    "mrr": r["mrr"], "recall@5": r["recall@5"],
                    "n_chunks": r["n_chunks"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="experiments/chunking_*.json")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configs = load(args.results, args.lang)
    if len(configs) < 2:
        raise SystemExit(
            f"need >=2 configs with per_query data, found {len(configs)}. "
            f"Re-run the benchmark after the per-query logging was added.")

    print(f"{len(configs)} configurations\n")
    for key, c in sorted(configs.items(), key=lambda kv: -kv[1]["mrr"]):
        print(f"  {key:40s} MRR={c['mrr']:.4f} R@5={c['recall@5']:.4f} "
              f"chunks={c['n_chunks']}")

    print(f"\nPaired bootstrap ({args.n_boot} resamples), A - B:\n")
    rows = []
    for a, b in itertools.combinations(sorted(configs), 2):
        ca, cb = configs[a], configs[b]
        if ca["qids"] != cb["qids"]:
            print(f"  SKIP {a} vs {b}: different query sets")
            continue
        res = paired_bootstrap(ca["rr"], cb["rr"], args.n_boot)
        rows.append({"a": a, "b": b, **res})
        flag = "SIGNIFICANT" if res["significant"] else "ns"
        print(f"  {a:38s}\n  vs {b:36s}\n"
              f"     dMRR={res['delta_mrr']:+.4f} "
              f"CI95=[{res['ci95_low']:+.4f}, {res['ci95_high']:+.4f}] "
              f"p={res['p_value']:.4f}  -> {flag}\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"configs": {k: {"mrr": v["mrr"],
                                       "recall@5": v["recall@5"],
                                       "n_chunks": v["n_chunks"]}
                                   for k, v in configs.items()},
                       "comparisons": rows}, fh, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
