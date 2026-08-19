"""
The MANDO orchestration harness.

Every stage is wrapped so that it has: a measured latency, a structured error
on failure, a defined recovery, and a request id threaded through the logs.

On timeouts, honestly
---------------------
Two different mechanisms, because they are genuinely different problems:

* NETWORK stages (STT, generation, TTS) get real enforced timeouts, because
  httpx can actually abort a socket. Those are hard limits.
* LOCAL COMPUTE stages (retrieval, reranking, grounding) get DEADLINE CHECKS
  between stages, not interruption. You cannot pre-empt a running numpy matmul
  from the same thread without threads or signals, and pretending otherwise
  would be a lie in the architecture diagram. So we check the remaining budget
  before entering a stage and degrade -- skip the reranker, shrink candidate_k
  -- rather than claiming to abort mid-operation.

Failure policy per stage
------------------------
    validate      fail -> refuse (cannot proceed without input)
    stt           fail -> refuse (no transcript, nothing to retrieve)
    language      fail -> default to English, warn, CONTINUE (never fatal)
    input guard   block -> refuse with the specific reason
    rewrite       fail -> use the raw transcript, warn, CONTINUE
    retrieval     fail -> refuse (answering without evidence is the one thing
                          this system must never do)
    rerank        fail -> use fused order, warn, CONTINUE
    evidence      insufficient -> refuse
    generation    fail -> refuse (never emit a partial answer)
    grounding     FAIL -> discard the answer and refuse
    tts           fail -> return the text answer, warn, CONTINUE

The asymmetry is deliberate: stages that affect PRESENTATION degrade
gracefully; stages that affect TRUTH refuse.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.config import CONFIG, Config
from app.generation.generator import (ExtractiveGenerator, Generation,
                                      GenerationError, LLMGenerator)
from app.generation.persona import refusal_text_with_locale
from app.guardrails.answerability import AnswerabilityJudge
from app.guardrails.verdict_cache import VerdictCache, verdict_key
from app.guardrails.grounding import verify_grounding
from app.guardrails.input import check_input
from app.guardrails.relevance import assess_evidence
from app.language import detect_language, resolve_language
from app.schemas import (Evidence, Lang, Latency, RAGRequest, RAGResponse,
                         RefusalReason, Stage, Timer)

log = logging.getLogger("mando")

# Anaphora that signal a follow-up needing conversation context.
_FOLLOWUP_MARKERS = {
    "en": ("it", "its", "that", "this", "they", "them", "he", "she", "his",
           "her", "there", "those", "these"),
    "hi": ("इसका", "इसके", "उसका", "उसके", "यह", "वह", "इन", "उन", "वहाँ"),
    "mr": ("याचा", "याचे", "त्याचा", "त्याचे", "हे", "ते", "तिथे", "यांचा"),
}


class RAGPipeline:
    def __init__(self, retriever, embedder, reranker, config: Config = CONFIG,
                 stt=None, tts=None, generator=None, dense_matrix=None,
                 judge=None, judge_mode: str = "async",
                 verdict_cache=None, index_version: str = "",
                 judge_workers: int = 2):
        self.cfg = config
        self.retriever = retriever
        # Cached corpus embeddings, so reranking and grounding reuse what the
        # index already holds instead of re-encoding the same evidence.
        self.dense_matrix = (dense_matrix if dense_matrix is not None
                             else getattr(retriever, "matrix", None))
        self.embedder = embedder
        self.reranker = reranker
        self.stt = stt
        self.tts = tts
        # Answerability judge. None disables the stage entirely, which is the
        # configuration every pre-Phase-1 measurement used.
        #
        # judge_mode:
        #   "async" (default, PRODUCTION) -- never blocks the first response.
        #            A cache miss answers from the cosine guard and judges in
        #            the background so the next identical query is a hit.
        #   "sync"  -- blocks. For evaluation harnesses only; using it on the
        #            voice path would add P50 909 ms / P70 11.1 s (E15).
        self.judge = judge
        self.judge_mode = judge_mode
        self.index_version = index_version
        self.verdict_cache = (verdict_cache if verdict_cache is not None
                              else (VerdictCache() if judge is not None
                                    else None))
        self._judge_pool = (ThreadPoolExecutor(max_workers=judge_workers,
                                               thread_name_prefix="judge")
                            if judge is not None and judge_mode == "async"
                            else None)
        self.judge_async_ms: list[float] = []
        self._async_lock = threading.Lock()

        if generator is not None:
            self.generator = generator
        elif config.has_llm:
            self.generator = LLMGenerator(
                api_key=config.llm_api_key, model=config.llm_model,
                base_url=config.llm_base_url,
                timeout_s=config.timeout_generation_ms / 1000)
        else:
            self.generator = ExtractiveGenerator(embedder)

        self.evidence_thresholds = {
            "min_top_score": config.evidence_min_top,
            "min_mean_top_k": config.evidence_min_mean,
        }
        self.grounding_thresholds = {
            "min_semantic": config.grounding_min_semantic}

    # -- helpers -----------------------------------------------------------
    def _refuse(self, resp: RAGResponse, reason: RefusalReason,
                lang: str = "en", detail: str = "") -> RAGResponse:
        resp.ok = True          # a refusal is a SUCCESSFUL, correct outcome
        resp.refused = True
        resp.refusal_reason = reason
        resp.answer, localised = refusal_text_with_locale(lang, reason.value)
        if not localised:
            # Konkani has no localised refusal copy; the user is getting
            # English. Surfaced rather than silently substituted.
            resp.warnings.append(
                f"refusal copy not available in {lang!r}; replied in English")
        if detail:
            resp.warnings.append(detail)
        log.info("[%s] refused reason=%s detail=%s",
                 resp.request_id, reason.value, detail)
        return resp

    @staticmethod
    def _needs_rewrite(query: str, lang: str) -> bool:
        words = query.lower().split()
        markers = _FOLLOWUP_MARKERS.get(lang, _FOLLOWUP_MARKERS["en"])
        return len(words) <= 8 and any(w.strip("?.,") in markers for w in words)

    def _rewrite(self, query: str, lang: str, history) -> str:
        """
        Resolve a follow-up against session context.

        Heuristic by design: prepending the previous topic is cheap (<1 ms) and
        deterministic, where an LLM rewrite call would add a full round trip to
        every follow-up turn. "What about its history?" becomes
        "<previous query> — what about its history?", which is enough for the
        retriever to latch onto the right entity.
        """
        if not history or not self._needs_rewrite(query, lang):
            return query
        return f"{history[-1].query} — {query}"

    def _judge_in_background(self, key, query, evidence_texts, lang) -> None:
        """
        Fire one background judge call, then store the verdict.

        `claim()` guarantees exactly one call per key even under a burst of
        identical queries -- otherwise N concurrent askers would each pay for
        their own judgement of the same thing.
        """
        if self._judge_pool is None or self.verdict_cache is None:
            return
        if not self.verdict_cache.claim(key):
            return                      # already cached or already in flight

        def _work():
            t0 = time.perf_counter()
            try:
                verdict = self.judge.judge(query, evidence_texts, lang)
                self.verdict_cache.put(key, verdict)
            except Exception as exc:    # never let a background thread die loud
                log.warning("background judge failed: %s", exc)
                self.verdict_cache.release(key)
            finally:
                ms = (time.perf_counter() - t0) * 1000
                with self._async_lock:
                    self.judge_async_ms.append(ms)

        try:
            self._judge_pool.submit(_work)
        except RuntimeError:             # pool shut down
            self.verdict_cache.release(key)

    def shutdown(self, wait: bool = True) -> None:
        if self._judge_pool is not None:
            self._judge_pool.shutdown(wait=wait)

    # -- main --------------------------------------------------------------
    def run(self, request: RAGRequest) -> RAGResponse:
        started = time.perf_counter()
        lat = Latency()
        resp = RAGResponse(request_id=request.request_id, ok=False,
                           latency=lat)
        log.info("[%s] start lang=%s audio=%s text=%s",
                 request.request_id, request.language,
                 bool(request.audio_base64), bool(request.text))

        # 1. validate ------------------------------------------------------
        with Timer(lat, Stage.validate):
            if not request.has_input():
                return self._refuse(resp, RefusalReason.empty_input, "en",
                                    "request carried neither text nor audio")

        # 2. STT -----------------------------------------------------------
        stt_lang = None
        if request.text:
            transcript = request.text.strip()
            lat.add(Stage.stt, 0.0)      # text path bypasses STT entirely
        else:
            if self.stt is None:
                return self._refuse(resp, RefusalReason.internal_error, "en",
                                    "audio supplied but no STT client")
            try:
                with Timer(lat, Stage.stt) as t:
                    result = self.stt.transcribe_base64(
                        request.audio_base64, language=request.language)
                    t.attempts = getattr(result, "attempts", 1)
                if result.offline:
                    return self._refuse(resp, RefusalReason.internal_error,
                                        "en", "STT running without credentials")
                transcript, stt_lang = result.text, result.language_code
            except Exception as exc:
                return self._refuse(resp, RefusalReason.internal_error, "en",
                                    f"stt failed: {exc}")
        resp.transcript = transcript

        # 3. language detection -- never fatal ------------------------------
        lang = "en"
        try:
            with Timer(lat, Stage.language_detection):
                res = resolve_language(request.language, transcript, stt_lang)
                lang = res.lang
                resp.language = Lang(lang)
                resp.language_confidence = res.confidence
        except Exception as exc:
            resp.warnings.append(f"language detection failed, defaulted to en: {exc}")
            resp.language = Lang("en")

        # 4. input guardrail ------------------------------------------------
        with Timer(lat, Stage.input_guardrail):
            guard = check_input(transcript)
        if not guard.allowed:
            return self._refuse(resp, guard.reason, lang, guard.detail)

        # 5. query rewriting -- never fatal ---------------------------------
        query = transcript
        try:
            with Timer(lat, Stage.query_rewrite):
                query = self._rewrite(transcript, lang, request.context)
        except Exception as exc:
            resp.warnings.append(f"query rewrite failed, using transcript: {exc}")
        resp.rewritten_query = query

        # 6. retrieval ------------------------------------------------------
        try:
            candidates, timing = self.retriever.retrieve(
                query, top_k=self.cfg.top_k,
                candidate_k=self.cfg.candidate_k,
                use_dense=self.cfg.use_dense, use_bm25=self.cfg.use_bm25)
            lat.add(Stage.dense_retrieval, timing.dense_ms)
            lat.add(Stage.bm25_retrieval, timing.bm25_ms)
            lat.add(Stage.fusion, timing.fusion_ms)
        except Exception as exc:
            lat.add(Stage.dense_retrieval, 0.0, ok=False, error=str(exc))
            return self._refuse(resp, RefusalReason.internal_error, lang,
                                f"retrieval failed: {exc}")

        # 7. rerank -- degrade to fused order on failure --------------------
        # The query vector comes from the dense stage; the reranker's candidate
        # vectors come from the index. Neither is re-encoded.
        try:
            with Timer(lat, Stage.rerank):
                candidates, _ = self.reranker.rerank(
                    query, candidates, top_k=self.cfg.top_k,
                    query_vector=timing.query_vector)
        except Exception as exc:
            resp.warnings.append(f"rerank failed, using fused order: {exc}")
            candidates = candidates[:self.cfg.top_k]

        # 8. evidence sufficiency -------------------------------------------
        with Timer(lat, Stage.evidence_check):
            decision = assess_evidence(candidates, self.evidence_thresholds)
        resp.retrieval_confidence = decision.confidence
        resp.evidence = [
            Evidence(doc_id=c.doc_id, text=c.text, context=c.context,
                     lang=c.lang, query_id=c.query_id,
                     passage_index=c.passage_index,
                     dense_score=c.dense_score, bm25_score=c.bm25_score,
                     fused_score=c.fused_score, rerank_score=c.rerank_score)
            for c in candidates]

        if not decision.sufficient:
            resp.latency.total_rag_ms = (time.perf_counter() - started) * 1000
            resp.latency.end_to_end_ms = resp.latency.total_rag_ms
            return self._refuse(resp, RefusalReason.insufficient_evidence,
                                lang, decision.detail)

        evidence_texts = [c.context for c in candidates]
        verdict = None

        # 8b. LLM answerability judge ---------------------------------------
        # Runs AFTER the cosine guard, so it only sees candidates that already
        # passed the cheap filter -- the expensive check is not spent on
        # queries the free one can already reject. Composition is
        # "cosine AND judge" (system C in the calibration experiment).
        #
        # If the judge is unavailable (no key, API down, repeated schema
        # failures) the cosine decision stands and a warning is recorded. An
        # LLM outage degrades the system to its previous behaviour rather than
        # refusing everything; set judge.fail_closed=True to invert that.
        if self.judge is not None:
            # ASYNC BY DEFAULT. The judge's measured latency is P50 909 ms /
            # P70 11.1 s (E15) against a 137 ms RAG budget, so it MUST NOT
            # block the first user-visible response.
            #
            #   cache HIT  -> the verdict is already known; applying it costs
            #                 a dict lookup, so it gates synchronously
            #   cache MISS -> answer now from the cosine guard, and judge in
            #                 the BACKGROUND so the next identical query is a
            #                 hit. The background call is timed separately as
            #                 judge_async_ms and is NEVER counted in
            #                 total_rag_ms.
            #
            # Set judge_mode="sync" to block (evaluation harnesses do this;
            # the voice path never should).
            key = verdict_key(query, [c.doc_id for c in candidates],
                              self.index_version)
            verdict = self.verdict_cache.get(key) if self.verdict_cache else None
            resp.judge_cache_hit = verdict is not None

            if verdict is None and self.judge_mode == "sync":
                with Timer(lat, Stage.answerability):
                    verdict = self.judge.judge(query, evidence_texts, lang)
                if self.verdict_cache is not None:
                    self.verdict_cache.put(key, verdict)

            elif verdict is None:
                self._judge_in_background(key, query, evidence_texts, lang)
                resp.judge_async_dispatched = True

        if self.judge is not None and verdict is not None:
            resp.judge_available = verdict.available
            resp.judge_sufficient = verdict.sufficient
            resp.judge_confidence = verdict.confidence
            resp.judge_reason = verdict.reason
            resp.judge_supporting_ids = verdict.supporting_ids

            if not verdict.available and verdict.sufficient:
                # Fail-open: the judge could not be consulted, so the cosine
                # decision stands. `verdict.sufficient` already encodes the
                # judge's fail_closed policy, so it is checked rather than
                # assumed -- a fail_closed judge that is down falls through to
                # the refusal branch below instead of being waved past.
                resp.warnings.append(
                    f"answerability judge unavailable, cosine guard decided "
                    f"alone: {verdict.error}")
            elif not verdict.sufficient:
                detail = (f"judge: {verdict.reason}" if verdict.available
                          else f"judge unavailable and fail_closed: "
                               f"{verdict.error}")
                resp.latency.total_rag_ms = (
                    time.perf_counter() - started) * 1000
                resp.latency.end_to_end_ms = resp.latency.total_rag_ms
                return self._refuse(resp, RefusalReason.insufficient_evidence,
                                    lang, detail)
            elif verdict.supporting_indices:
                # Narrow the evidence to what the judge actually cited. The
                # generator then sees only passages a reader confirmed carry
                # the answer, which is a stronger grounding constraint than
                # "the top k by cosine".
                cited = [candidates[i] for i in verdict.supporting_indices
                         if i < len(candidates)]
                if cited:
                    candidates = cited
                    evidence_texts = [c.context for c in candidates]
                    resp.evidence = [e for e in resp.evidence
                                     if e.doc_id in {c.doc_id for c in cited}]

        # 9. generation ------------------------------------------------------
        try:
            with Timer(lat, Stage.generation) as t:
                gen: Generation = self.generator.generate(
                    query, evidence_texts, lang, history=request.context)
                t.attempts = gen.attempts
        except GenerationError as exc:
            return self._refuse(resp, RefusalReason.internal_error, lang,
                                f"generation failed: {exc}")
        except Exception as exc:
            return self._refuse(resp, RefusalReason.internal_error, lang,
                                f"generation error: {exc}")

        resp.generator = type(self.generator).__name__
        resp.persona = getattr(getattr(self.generator, "persona", None),
                               "name", None)
        resp.sources_used = list(getattr(gen, "sources_used", []) or [])
        resp.invalid_sources = list(getattr(gen, "invalid_sources", []) or [])
        if resp.invalid_sources:
            # The model cited a source number it was never shown. Already
            # dropped from sources_used; surfaced so fabricated citations are
            # visible rather than silently swallowed.
            resp.warnings.append(
                f"generator cited unknown sources {resp.invalid_sources}; "
                f"dropped")

        if not gen.answer.strip():
            return self._refuse(resp, RefusalReason.insufficient_evidence,
                                lang, "generator returned an empty answer")

        # SAME-LANGUAGE ENFORCEMENT. The persona is instructed to reply in the
        # user's language, but instruction is not verification. We detect the
        # answer's language and record a mismatch. It is a WARNING, not a
        # refusal: a correct answer in the wrong language is still useful, and
        # refusing it would trade a real answer for a cosmetic failure. The
        # flag lets the UI and evaluation see it.
        try:
            detected = detect_language(gen.answer)
            resp.language_match = (detected.lang == lang
                                   or detected.confidence < 0.5)
            if not resp.language_match:
                resp.warnings.append(
                    f"answer language {detected.lang!r} does not match "
                    f"requested {lang!r}")
        except Exception:
            resp.language_match = None

        # The model may itself report the sources were inadequate. Trust it --
        # it read them.
        if not gen.sufficient:
            return self._refuse(resp, RefusalReason.insufficient_evidence,
                                lang, "generator reported sufficient=false")

        # 10. grounding verification -----------------------------------------
        # Reuse the indexed evidence vectors rather than encoding them a third
        # time. Only the ANSWER's sentences genuinely need encoding here.
        # CORRECTNESS GUARD: the index stores embeddings of chunk.TEXT, but
        # grounding compares against chunk.CONTEXT. Those are the same object
        # for `fixed`/`sentence`, but NOT for hierarchical or sentence-window,
        # where context is deliberately wider than the embedded text. Reusing
        # the cached vector there would score the answer against the wrong
        # string, so we only take the shortcut when they are identical.
        ev_vectors = None
        if self.dense_matrix is not None and all(
                c.row is not None and c.text == c.context for c in candidates):
            ev_vectors = self.dense_matrix[[c.row for c in candidates]]

        with Timer(lat, Stage.grounding):
            grounding = verify_grounding(gen.answer, evidence_texts,
                                         self.embedder,
                                         self.grounding_thresholds,
                                         evidence_vectors=ev_vectors)
        resp.grounded = grounding.passed
        resp.grounding_score = grounding.semantic_score

        if self.cfg.enforce_grounding and not grounding.passed:
            resp.latency.total_rag_ms = (time.perf_counter() - started) * 1000
            resp.latency.end_to_end_ms = resp.latency.total_rag_ms
            return self._refuse(resp, RefusalReason.ungrounded_answer, lang,
                                grounding.detail)

        resp.answer = gen.answer
        resp.ok = True
        resp.latency.total_rag_ms = (time.perf_counter() - started) * 1000

        # 11. TTS -- presentation only, never fatal ---------------------------
        if request.want_audio and self.tts is not None:
            try:
                with Timer(lat, Stage.tts):
                    audio = self.tts.synthesize(gen.answer, lang,
                                                voice=request.voice)
                resp.audio_base64 = audio.audio_base64
            except Exception as exc:
                resp.warnings.append(f"tts failed, returning text only: {exc}")

        resp.latency.end_to_end_ms = (time.perf_counter() - started) * 1000
        log.info("[%s] ok lang=%s rag=%.1fms e2e=%.1fms grounded=%s",
                 resp.request_id, lang, resp.latency.total_rag_ms,
                 resp.latency.end_to_end_ms, resp.grounded)
        return resp
