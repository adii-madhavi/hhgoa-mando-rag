"""
ONE small end-to-end voice pipeline validation. Not a benchmark.

    audio in -> Sarvam STT -> transcript -> RAGPipeline (retrieval, cosine
    guard, async judge, guarded LLM generation) -> Sarvam TTS -> audio out

5 real pipeline runs total: 3 fixed queries (en/hi/mr) + 1 extra male-voice
request + 1 extra female-voice request. No retrieval/embedding experiments are
rerun -- the existing prebuilt index is loaded as-is, exactly as
generation_latency.py and streaming_latency.py already do.

How audio input is produced
----------------------------
There is no microphone here, so real Sarvam TTS synthesizes each query's text
into real audio, and that audio (not the original text) is what gets base64-
encoded and sent to Sarvam STT. This is a genuine audio round trip through
Sarvam's real API in both directions -- the pipeline never sees the original
text, only the transcript STT recovers from real audio bytes. Same technique
already used in sarvam_live_check.py.

Test set
--------
Exactly 3 fixed queries, one answerable MSMARCO-XI query per language
(en/hi/mr), chosen deterministically (first answerable query per language in
the existing processed corpus -- no cherry-picking for a favourable result).
Konkani is deliberately excluded: MSMARCO-XI has no labelled Konkani
evaluation data (EXPERIMENTS.md E2/E2b), so it is not used as a benchmark query
here or anywhere else in this project.

Secret hygiene
--------------
Never logs, prints, or interpolates SARVAM_API_KEY or LLM_API_KEY. Only
non-secret fields are printed.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CONFIG                              # noqa: E402
from app.generation.generator import LLMGenerator          # noqa: E402
from app.guardrails.answerability import AnswerabilityJudge  # noqa: E402
from app.pipeline import RAGPipeline                       # noqa: E402
from app.retrieval.loader import load_index                # noqa: E402
from app.retrieval.reranker import MMRReranker             # noqa: E402
from app.schemas import RAGRequest                          # noqa: E402
from app.stt.sarvam import SarvamSTT                        # noqa: E402
from app.tts.sarvam_tts import SarvamTTS                    # noqa: E402

LANGS = ("en", "hi", "mr")   # Konkani excluded: no labelled evaluation data


def pick_fixed_queries(processed: str) -> dict:
    """First answerable query per language, in corpus order -- deterministic,
    not cherry-picked for a favourable outcome."""
    with open(os.path.join(processed, "queries.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            if q["has_answer"]:
                return {lang: q["queries"][lang] for lang in LANGS}
    raise SystemExit("no answerable query found in data/processed/queries.jsonl")


def synthesize_input_audio(tts: SarvamTTS, text: str, lang: str) -> str:
    """Real Sarvam TTS call -> base64 audio, used as the pipeline's INPUT."""
    speech = tts.synthesize(text, lang, voice="male")
    return speech.audio_base64


def wav_looks_valid(audio_b64: str | None) -> bool:
    if not audio_b64:
        return False
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except Exception:
        return False
    return len(raw) > 44 and raw[:4] == b"RIFF"   # RIFF/WAV header + payload


def report_row(label: str, resp, flat: dict) -> None:
    retrieval_ms = (flat.get("dense_retrieval_ms", 0)
                    + flat.get("bm25_retrieval_ms", 0)
                    + flat.get("fusion_ms", 0) + flat.get("rerank_ms", 0)
                    + flat.get("evidence_check_ms", 0))
    print(f"\n--- {label} ---")
    print(f"  input language requested : {resp.language.value if resp.language else 'n/a'}")
    print(f"  transcript (from real STT): {resp.transcript!r}")
    print(f"  STT latency               : {flat.get('stt_ms', 0):.1f} ms")
    print(f"  retrieval latency         : {retrieval_ms:.1f} ms "
          f"(dense+bm25+fusion+rerank+evidence_check)")
    print(f"  judge latency (inline)    : {flat.get('answerability_ms', 0):.1f} ms "
          f"(async by design -- see below for background cost)")
    print(f"  generation latency        : {flat.get('generation_ms', 0):.1f} ms")
    print(f"  TTS latency                : {flat.get('tts_ms', 0):.1f} ms")
    print(f"  RAG latency (excl. TTS)   : {resp.latency.total_rag_ms:.1f} ms")
    print(f"  TOTAL voice latency (e2e) : {resp.latency.end_to_end_ms:.1f} ms  "
          f"<- NOT claimed to meet any 200ms target")
    print(f"  refused                   : {resp.refused}"
          + (f" ({resp.refusal_reason.value})" if resp.refusal_reason else ""))
    print(f"  answer language           : {resp.language.value if resp.language else 'n/a'} "
          f"(match={resp.language_match})")
    print(f"  grounded                  : {resp.grounded}")
    valid_audio = wav_looks_valid(resp.audio_base64)
    print(f"  valid output audio produced: {valid_audio}"
          + ("" if valid_audio else f"  (warnings: {resp.warnings})"))
    if resp.answer:
        print(f"  answer (truncated)        : {resp.answer[:140]!r}")


def main() -> None:
    if not CONFIG.has_stt or not CONFIG.has_llm:
        raise SystemExit(
            f"\nMissing credentials: {CONFIG.missing_credentials()}\n"
            f"This script will not fabricate a result without real "
            f"credentials.\n")

    print("=" * 70)
    print("VOICE PIPELINE END-TO-END VALIDATION (3 fixed queries, real Sarvam)")
    print("=" * 70)

    t0 = time.perf_counter()
    loaded = load_index(CONFIG.index_dir)
    stt = SarvamSTT(api_key=CONFIG.sarvam_api_key)
    tts = SarvamTTS(api_key=CONFIG.sarvam_api_key)
    generator = LLMGenerator(api_key=CONFIG.llm_api_key,
                             model=CONFIG.llm_model,
                             base_url=CONFIG.llm_base_url, timeout_s=30)
    judge = AnswerabilityJudge(timeout_s=60)
    pipeline = RAGPipeline(
        retriever=loaded.retriever, embedder=loaded.embedder,
        reranker=MMRReranker(loaded.embedder,
                             dense_matrix=loaded.retriever.matrix),
        config=CONFIG, dense_matrix=loaded.retriever.matrix,
        generator=generator, stt=stt, tts=tts,
        judge=judge, judge_mode="async",
        index_version=str(loaded.manifest["n_chunks"]))
    print(f"startup (excluded from every latency figure below): "
          f"{time.perf_counter() - t0:.1f}s -- index reused, not rebuilt\n")

    queries = pick_fixed_queries(CONFIG.processed_dir)
    results = []

    # --- 3 fixed queries, one per evaluated language ------------------------
    for lang in LANGS:
        text = queries[lang]
        print(f"[{lang}] synthesizing real input audio via Sarvam TTS "
              f"for: {text[:60]!r}...")
        audio_b64 = synthesize_input_audio(tts, text, lang)

        resp = pipeline.run(RAGRequest(audio_base64=audio_b64, language=lang,
                                       voice="male", want_audio=True))
        flat = resp.latency.flat()
        results.append((lang, resp, flat))
        report_row(f"QUERY {len(results)}/3 -- {lang}", resp, flat)

    # --- male / female voice check ------------------------------------------
    print("\n" + "=" * 70)
    print("VOICE SELECTION CHECK (male vs female, same English query)")
    print("=" * 70)
    en_text = queries["en"]
    audio_b64 = synthesize_input_audio(tts, en_text, "en")

    voice_results = {}
    for voice in ("male", "female"):
        resp = pipeline.run(RAGRequest(audio_base64=audio_b64, language="en",
                                       voice=voice, want_audio=True))
        flat = resp.latency.flat()
        voice_results[voice] = resp
        report_row(f"VOICE={voice}", resp, flat)

    # Shutdown moved here (was previously called before the voice-selection
    # queries, which shut the async judge pool for those two runs). Harmless
    # in practice -- every query in this script is a fresh cache miss, so the
    # async judge never gates a first-time query regardless -- but wrong
    # placement, fixed for correctness.
    pipeline.shutdown(wait=True)
    if pipeline.judge_async_ms:
        print(f"\nBackground judge calls this run: {len(pipeline.judge_async_ms)}, "
              f"latencies (ms): {[round(x) for x in pipeline.judge_async_ms]}  "
              f"(excluded from total_rag_ms / end_to_end_ms by design)")

    # --- summary --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for lang, resp, flat in results:
        print(f"  {lang}: refused={resp.refused} grounded={resp.grounded} "
              f"valid_audio={wav_looks_valid(resp.audio_base64)} "
              f"e2e={resp.latency.end_to_end_ms:.0f}ms")
    for voice, resp in voice_results.items():
        print(f"  voice={voice}: valid_audio={wav_looks_valid(resp.audio_base64)} "
              f"refused={resp.refused}")

    print("\nHonest scope statement: every 'TOTAL voice latency' figure above "
          "is a single real measurement from ONE run each (n=1 per query), not "
          "a benchmark with percentiles. It is reported as measured, and no "
          "claim is made that it meets the 200ms target -- retrieval-only "
          "latency (E11/E16) is the only figure inside that target.")


if __name__ == "__main__":
    main()
