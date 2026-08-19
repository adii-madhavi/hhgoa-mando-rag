"""
PHASE 2 — does the LLM answerability judge beat the cosine threshold?

Compares three systems on the labelled data, per language:

    A  cosine threshold only          (current production)
    B  LLM sufficiency judge only
    C  cosine AND judge               (production composition: cosine first,
                                       judge only on what cosine admits)

Labels come from MSMARCO-XI at no annotation cost:
    positive = answerable   (`is_selected` has a 1)  -> SHOULD answer
    negative = no-answer    (all-zero, "No Answer Present") -> SHOULD refuse

Metrics, all reported per language and never averaged into one number:

    false_answer_rate   answered / negatives      <- the hallucination risk
    false_refusal_rate  refused  / positives      <- the usability cost
    precision           TP / (TP + FP)            over "answered"
    recall              TP / (TP + FN)
    balanced_accuracy   (TPR + TNR) / 2

Honest framing, decided BEFORE seeing any numbers
-------------------------------------------------
The baseline to beat is A: balanced accuracy 0.6525 en / 0.6325 hi / 0.6100 mr
(EXPERIMENTS.md E10). The judge is a hypothesis. If it does not beat those
numbers it does not ship, regardless of how principled it sounds. The
composition C can only ever REDUCE the answer rate relative to A, since it adds
a second gate on top of the first -- so C must be judged on whether the
false-answer reduction is worth the extra false refusals, not on accuracy alone.

Cost is reported too: the judge is a network round trip per query, versus a
matrix multiply. A large accuracy gain may still be a bad trade at a 200 ms
budget, and `judge_ms` makes that visible.

Languages
---------
Evaluated here: en, hi, mr -- the languages MSMARCO-XI actually labels.

Konkani is a PRODUCT language but NOT a dataset-evaluated one. MSMARCO-XI
contains zero Konkani rows under any code, so a Konkani false-answer rate
cannot be computed from it. Passing --langs kok produces an explicit "N/A"
entry with the reason recorded; it never produces a number. Konkani quality is
measured separately by evaluation/konkani_product_eval.py against a curated
product set, and those results are never merged into these benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG                              # noqa: E402
from app.generation.llm_client import ChatClient           # noqa: E402
from app.guardrails.answerability import AnswerabilityJudge  # noqa: E402
from app.guardrails.relevance import assess_evidence       # noqa: E402
from app.retrieval.loader import load_index                # noqa: E402
from app.retrieval.reranker import MMRReranker             # noqa: E402
from app.schemas import EVALUATED_LANGS, PRODUCT_LANGS      # noqa: E402


@dataclass
class Counts:
    tp: int = 0        # answered a positive   (correct)
    fn: int = 0        # refused a positive    (false refusal)
    fp: int = 0        # answered a negative   (FALSE ANSWER)
    tn: int = 0        # refused a negative    (correct)
    judge_ms: list = field(default_factory=list)
    unavailable: int = 0
    downgraded: int = 0
    invalid_citations: int = 0

    def add(self, answered: bool, is_positive: bool) -> None:
        if is_positive:
            self.tp += answered
            self.fn += not answered
        else:
            self.fp += answered
            self.tn += not answered

    def metrics(self) -> dict:
        pos = self.tp + self.fn
        neg = self.fp + self.tn
        answered = self.tp + self.fp
        tpr = self.tp / pos if pos else 0.0
        tnr = self.tn / neg if neg else 0.0
        out = {
            "n_positive": pos, "n_negative": neg,
            "false_answer_rate": round(self.fp / neg, 4) if neg else None,
            "false_refusal_rate": round(self.fn / pos, 4) if pos else None,
            "precision": round(self.tp / answered, 4) if answered else None,
            "recall": round(tpr, 4),
            "specificity": round(tnr, 4),
            "balanced_accuracy": round((tpr + tnr) / 2, 4),
            "confusion": {"tp": self.tp, "fn": self.fn,
                          "fp": self.fp, "tn": self.tn},
        }
        if self.judge_ms:
            s = sorted(self.judge_ms)
            out["judge_ms"] = {
                "p50": round(s[len(s) // 2], 1),
                "p100": round(s[-1], 1),
                "mean": round(float(np.mean(s)), 1),
            }
            out["judge_unavailable"] = self.unavailable
            out["judge_downgraded_verdicts"] = self.downgraded
            out["judge_invalid_citations"] = self.invalid_citations
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--langs", default="en,hi,mr",
                    help="Benchmark languages. Konkani is a PRODUCT language "
                         "but not an evaluated one -- passing it yields an "
                         "explicit N/A row, never a fabricated metric.")
    ap.add_argument("--n-per-class", type=int, default=100,
                    help="queries per class PER LANGUAGE. Each one is an LLM "
                         "call for systems B/C, so this is the cost dial.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--model", default=None, help="override LLM_MODEL")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel judge calls; judge latency is network-bound")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-call LLM timeout (s). Smoke showed p100 50-61s "
                         "on Hindi, so 20s truncated slow calls into false "
                         "unavailable verdicts.")
    ap.add_argument("--out", default="experiments/answerability_calibration.json")
    args = ap.parse_args()

    client = ChatClient(model=args.model, timeout_s=args.timeout)
    if not client.available:
        raise SystemExit(
            "\nNo LLM key configured, so systems B and C cannot be measured.\n"
            "Set LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL) in .env\n"
            "and re-run. This script will not emit numbers it did not measure.\n")

    judge = AnswerabilityJudge(client=client)
    loaded = load_index(args.index_dir or CONFIG.index_dir)
    reranker = MMRReranker(loaded.embedder,
                           dense_matrix=loaded.retriever.matrix)
    print(f"index: {loaded.manifest['n_chunks']} chunks | "
          f"model: {client.model} @ {client.base_url}")

    queries = []
    with open(os.path.join(args.processed, "queries.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            queries.append(json.loads(line))
    indexed = {c.query_id for c in loaded.chunks}

    thresholds = {"min_top_score": CONFIG.evidence_min_top,
                  "min_mean_top_k": CONFIG.evidence_min_mean}

    report = {
        "config": {
            "llm_model": client.model, "base_url": client.base_url,
            "top_k": args.top_k, "n_per_class": args.n_per_class,
            "concurrency": args.concurrency, "timeout_s": args.timeout,
            "cosine_thresholds": thresholds,
            "index": loaded.manifest,
        },
        "baseline_note": (
            "System A reproduces the production cosine guard; its numbers "
            "should land near EXPERIMENTS.md E10 (balanced accuracy "
            "0.6525 en / 0.6325 hi / 0.6100 mr)."),
        "per_language": {},
        "per_query": [],
    }

    for lang in [l.strip() for l in args.langs.split(",")]:
        pos = [q for q in queries
               if q["has_answer"] and q["query_id"] in indexed]
        neg = [q for q in queries
               if not q["has_answer"] and q["query_id"] in indexed]
        rng = random.Random(args.seed)
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos, neg = pos[:args.n_per_class], neg[:args.n_per_class]
        if not pos and not neg:
            print(f"\n[{lang}] no labelled data -- skipping")
            report["per_language"][lang] = {"note": "no labelled data"}
            continue

        print(f"\n=== {lang}: {len(pos)} answerable + {len(neg)} no-answer ===")
        A, B, C = Counts(), Counts(), Counts()
        t_lang = time.perf_counter()

        # METHODOLOGY FIX: shuffle the two classes together.
        # The first full run processed all positives then all negatives.
        # Rate-limit failures accumulated over the run and therefore landed
        # almost entirely in the back half -- for English, 41 of 41 failures
        # hit negatives. Because an unavailable verdict fails OPEN, each one
        # was scored as a false answer, and B's English false-answer rate was
        # inflated 0.20 -> 0.53 purely by call ORDER. Interleaving makes any
        # residual failure rate class-independent.
        items = [(q, True) for q in pos] + [(q, False) for q in neg]
        random.Random(args.seed + 1).shuffle(items)

        # -- stage 1: retrieval + cosine, SERIAL ---------------------------
        # Deliberately not threaded. Retrieval is a numpy matmul against a
        # 72k x 384 matrix and reranking touches torch; putting those on a
        # thread pool contends for the GIL and for BLAS threads, slowing the
        # CPU work when the NETWORK calls are what actually need parallelism.
        # ~20 ms each, so serial costs nothing.
        prepared = []
        for q, is_pos in items:
            text = q["queries"].get(lang)
            if not text:
                continue
            cands, _ = loaded.retriever.retrieve(
                text, top_k=args.top_k, candidate_k=CONFIG.candidate_k,
                use_dense=True, use_bm25=CONFIG.use_bm25)
            cands, _ = reranker.rerank(text, cands, top_k=args.top_k)
            cosine = assess_evidence(cands, thresholds)
            prepared.append((q, is_pos, text,
                             [c.context for c in cands], cosine))

        # -- stage 2: judge calls, CONCURRENT ------------------------------
        # Network-bound, 1-12 s each. httpx.Client is thread-safe for
        # concurrent requests and tenacity still applies per call.
        done = {"n": 0}
        lock = threading.Lock()

        def run_judge(pair):
            i, (q, is_pos, text, contexts, cosine) = pair
            v = judge.judge(text, contexts, lang)
            with lock:
                done["n"] += 1
                if done["n"] % 25 == 0:
                    print(f"    {done['n']}/{len(prepared)} judged...",
                          flush=True)
            return i, v

        verdicts = [None] * len(prepared)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i, v in pool.map(run_judge, list(enumerate(prepared))):
                verdicts[i] = v

        # -- stage 3: score in the ORIGINAL order --------------------------
        # Reassembled by index so concurrency cannot reorder results and make
        # the run non-reproducible.
        for (q, is_pos, text, contexts, cosine), verdict in zip(prepared,
                                                                verdicts):
            A.add(cosine.sufficient, is_pos)

            B.add(verdict.sufficient, is_pos)
            B.judge_ms.append(verdict.latency_ms)
            B.unavailable += not verdict.available
            B.downgraded += verdict.downgraded
            B.invalid_citations += bool(verdict.invalid_ids)

            C.add(cosine.sufficient and verdict.sufficient, is_pos)

            report["per_query"].append({
                "lang": lang, "query_id": q["query_id"],
                "is_answerable": is_pos,
                "cosine_sufficient": cosine.sufficient,
                "cosine_top": cosine.top_score,
                "judge_sufficient": verdict.sufficient,
                "judge_confidence": verdict.confidence,
                "judge_available": verdict.available,
                "judge_downgraded": verdict.downgraded,
                "judge_ms": round(verdict.latency_ms, 1),
                "judge_error": verdict.error,
                "judge_reason": verdict.reason[:200],
            })

        fail_rate = B.unavailable / max(len(prepared), 1)
        if fail_rate > 0.02:
            print(f"  !! {B.unavailable}/{len(prepared)} judge calls FAILED "
                  f"({fail_rate:.1%}). Failed calls fail OPEN and are scored "
                  f"as 'answered', so B and C are NOT trustworthy above ~2%.")

        report["per_language"][lang] = {
            "judge_failure_rate": round(fail_rate, 4),
            "results_trustworthy": bool(fail_rate <= 0.02),
            "A_cosine_only": A.metrics(),
            "B_judge_only": B.metrics(),
            "C_cosine_and_judge": C.metrics(),
            "wall_seconds": round(time.perf_counter() - t_lang, 1),
        }

        print(f"\n  {'system':22s} {'falseAns':>9s} {'falseRef':>9s} "
              f"{'prec':>7s} {'recall':>7s} {'balAcc':>7s}")
        for name, c in (("A cosine only", A), ("B judge only", B),
                        ("C cosine + judge", C)):
            m = c.metrics()
            print(f"  {name:22s} {m['false_answer_rate']:9.3f} "
                  f"{m['false_refusal_rate']:9.3f} "
                  f"{(m['precision'] if m['precision'] is not None else 0):7.3f} "
                  f"{m['recall']:7.3f} {m['balanced_accuracy']:7.3f}")
        if B.judge_ms:
            jm = B.metrics()["judge_ms"]
            print(f"  judge latency p50={jm['p50']}ms p100={jm['p100']}ms | "
                  f"unavailable={B.unavailable} downgraded={B.downgraded}")

    # ---- verdict, stated against the pre-registered baseline -------------
    print("\n" + "=" * 74)
    print("SUMMARY (balanced accuracy; A is the incumbent to beat)")
    print("=" * 74)
    wins = []
    for lang, entry in report["per_language"].items():
        if "A_cosine_only" not in entry:
            continue
        a = entry["A_cosine_only"]["balanced_accuracy"]
        b = entry["B_judge_only"]["balanced_accuracy"]
        c = entry["C_cosine_and_judge"]["balanced_accuracy"]
        best = max((a, "A"), (b, "B"), (c, "C"))[1]
        wins.append(best)
        print(f"  {lang}:  A={a:.4f}  B={b:.4f}  C={c:.4f}   -> best: {best}")
    if wins:
        report["winner_per_language"] = wins
        print("\nThe judge ships only if B or C beats A on balanced accuracy "
              "AND the false-answer reduction justifies the added latency.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
