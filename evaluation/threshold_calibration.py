"""
Calibrate the retrieval-confidence threshold on labelled data.

MSMARCO-XI hands us a labelled refusal set for free: 780 of our 2,000 queries
have `is_selected` all-zero and `Answer == "No Answer Present"`. So instead of
picking a similarity threshold that feels about right, we fit it:

    positives = answerable queries      -> the system SHOULD answer
    negatives = no-answer queries       -> the system SHOULD refuse

and choose the operating point maximising balanced accuracy.

Why balanced accuracy and not raw accuracy
------------------------------------------
The classes are 61/39, and the two errors are not equally bad. A false ANSWER
(confidently answering a question the corpus cannot support) is a hallucination
— the exact failure this system exists to prevent. A false REFUSAL is merely
unhelpful. Balanced accuracy weights the classes equally rather than letting
the majority class dominate; the full precision/recall curve is written to the
output so the operating point can be moved deliberately toward fewer false
answers if desired.

An important caveat that the report states explicitly: a "no-answer" query
still has 10 retrieved passages that are topically related — MS MARCO built
them from a real search. So this is not "relevant vs random", it is the much
harder "relevant vs plausibly-related-but-insufficient". Expect modest
separation, and treat any high score here with suspicion.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG                      # noqa: E402
from app.retrieval.loader import load_index        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--langs", default="en,hi,mr")
    ap.add_argument("--n-per-class", type=int, default=250)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="experiments/threshold_calibration.json")
    args = ap.parse_args()

    loaded = load_index(args.index_dir or CONFIG.index_dir)
    retriever = loaded.retriever

    queries = []
    with open(os.path.join(args.processed, "queries.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            queries.append(json.loads(line))

    pos = [q for q in queries if q["has_answer"]]
    neg = [q for q in queries if not q["has_answer"]]
    rng = random.Random(args.seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos, neg = pos[:args.n_per_class], neg[:args.n_per_class]
    print(f"positives(answerable)={len(pos)} negatives(no-answer)={len(neg)}")

    report = {"per_language": {}, "note": (
        "negatives are queries whose 10 retrieved passages are topically "
        "related but contain no answer -- a much harder discrimination than "
        "relevant-vs-random")}
    chosen_overall = None

    for lang in [l.strip() for l in args.langs.split(",")]:
        def features(qs):
            rows = []
            for q in qs:
                text = q["queries"].get(lang)
                if not text:
                    continue
                cands, _ = retriever.retrieve(
                    text, top_k=CONFIG.top_k,
                    candidate_k=CONFIG.candidate_k,
                    use_dense=True, use_bm25=False)
                scores = sorted(
                    [c.dense_score for c in cands if c.dense_score is not None],
                    reverse=True)
                if not scores:
                    rows.append((0.0, 0.0, 0.0))
                    continue
                top = scores[0]
                mean3 = float(np.mean(scores[:3]))
                margin = top - (scores[1] if len(scores) > 1 else 0.0)
                rows.append((top, mean3, margin))  # order matches feature_names
            return np.array(rows, dtype="float64")

        fp, fn = features(pos), features(neg)

        # Sweep EVERY feature, not just top_score. The top score is the obvious
        # choice and turns out to be a poor one: e5 compresses similarity into
        # a narrow band, so absolute values barely move between an answerable
        # and an unanswerable query. Margin (top - second) and mean@3 measure
        # the SHAPE of the score distribution instead of its level, and shape
        # may survive the compression better.
        feature_names = ("top_score", "mean_top3", "margin")
        per_feature = {}

        for fi, fname in enumerate(feature_names):
            a, b = fp[:, fi], fn[:, fi]
            sep = float(a.mean() - b.mean())
            best, curve = None, []
            lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
            for thr in np.linspace(lo, hi, 200):
                tp = int((a >= thr).sum())
                fn_ = len(a) - tp
                fpos = int((b >= thr).sum())
                tn = len(b) - fpos
                tpr = tp / max(len(a), 1)       # answered when it should
                tnr = tn / max(len(b), 1)       # refused when it should
                bal = (tpr + tnr) / 2
                row = {"threshold": round(float(thr), 4),
                       "answer_rate_on_answerable": round(tpr, 4),
                       "refusal_rate_on_noanswer": round(tnr, 4),
                       "balanced_accuracy": round(bal, 4),
                       "false_answers": fpos, "false_refusals": fn_}
                curve.append(row)
                if best is None or bal > best["balanced_accuracy"]:
                    best = row
            per_feature[fname] = {
                "mean_positive": round(float(a.mean()), 4),
                "mean_negative": round(float(b.mean()), 4),
                "separation": round(sep, 4),
                "best": best, "curve": curve[::20],
            }
            print(f"  [{lang}] {fname:10s} sep={sep:+.4f} "
                  f"best_thr={best['threshold']:.4f} "
                  f"bal_acc={best['balanced_accuracy']:.4f} "
                  f"(answers {best['answer_rate_on_answerable']:.0%} / "
                  f"refuses {best['refusal_rate_on_noanswer']:.0%})")

        winner = max(feature_names,
                     key=lambda f: per_feature[f]["best"]["balanced_accuracy"])
        best = per_feature[winner]["best"]
        print(f"  [{lang}] -> best feature: {winner} "
              f"(bal_acc {best['balanced_accuracy']:.4f})")

        report["per_language"][lang] = {
            "positives": len(fp), "negatives": len(fn),
            "features": per_feature, "best_feature": winner, "best": best,
            "mean_top_positive": per_feature["top_score"]["mean_positive"],
            "mean_top_negative": per_feature["top_score"]["mean_negative"],
            "separation": per_feature["top_score"]["separation"],
        }

    # A single operating point across languages: average the per-language
    # optima rather than tuning three thresholds, which would overfit 250
    # queries per class per language.
    thrs = [v["best"]["threshold"] for v in report["per_language"].values()]
    chosen_overall = round(float(np.mean(thrs)), 4)
    # The balanced-accuracy optimum is reported, but it is NOT what ships.
    # At ~0.89 the system refuses 26% of answerable English queries and 54% of
    # answerable Hindi ones -- an enormous usability cost to buy a little extra
    # coverage from a signal whose balanced accuracy is only ~0.63. The
    # grounding verifier inspects the ACTUAL ANSWER and is the stronger
    # hallucination defence, so this stage ships as a cheap pre-filter at 0.87:
    #   en 94% answered / 24% of no-answer refused
    #   hi 74% / 40%
    #   mr 76% / 40%
    # Stated as the deliberate product decision it is; the full curve is in
    # this file's output so the operating point can be moved with evidence.
    report["balanced_accuracy_optimum"] = {
        "min_top_score": chosen_overall,
        "note": "maximises balanced accuracy; rejected as the shipped default "
                "because the false-refusal cost is too high for a signal this "
                "weak",
    }
    report["chosen"] = {
        "min_top_score": 0.87,
        "min_mean_top_k": 0.85,
        "k": 3,
    }
    report["rationale"] = (
        "0.87 chosen deliberately over the 0.89 balanced-accuracy optimum: the "
        "retrieval-confidence signal is weak (balanced accuracy 0.61-0.65), so "
        "the grounding verifier carries the hallucination defence and this "
        "stage acts as a cheap pre-filter that preserves answer coverage")

    print(f"\nCHOSEN min_top_score={report['chosen']['min_top_score']} "
          f"min_mean_top_k={report['chosen']['min_mean_top_k']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
