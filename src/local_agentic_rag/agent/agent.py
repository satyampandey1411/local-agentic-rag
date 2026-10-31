"""
The agentic RAG loop.

Two modes (``RAGConfig.agent_mode``):

* ``adaptive`` (default): retrieve → generate → verify; escalate to the loop
  below only when the first answer is refused or fails verification
  (``_run_adaptive``).  This is what the benchmark favoured.
* ``full``: always run the loop below (``_run_agentic``):

    question (+ chat history)
        │
        ▼
    PLAN ── standalone question + 1-3 search queries          (1 LLM call, JSON)
        │
        ▼
    RETRIEVE ── hybrid dense+BM25 → RRF → cross-encoder        (no LLM)
        │
        ▼
    GRADE ── which passages are relevant? sufficient?          (1 LLM call, JSON)
        │   └─ insufficient & rounds left → add missing_query, RETRIEVE again
        ▼
    GENERATE ── answer with [n] citations or ANSWER_NOT_FOUND  (1 LLM call)
        │
        ▼
    VERIFY ── every claim supported by the passages?           (1 LLM call, JSON)
        │   └─ unsupported & attempts left → GENERATE with feedback
        ▼
    Answer (citations → pages, trace, grounded flag)

Every loop is bounded (``max_retrieval_rounds``, ``max_answer_attempts``) so
latency is predictable, and every step degrades gracefully: if a structured
step fails (bad JSON, timeout) the agent falls back to the simpler behaviour
instead of failing the question.

Design choice — no per-passage LLM grading (reference notebook) and no
LangGraph: grading 6 passages one call at a time costs 6 LLM calls per round
on a T4; one batched grading call gives the same signal.  A plain Python
state machine is easier to test and debug than a graph framework for five
nodes, and avoids a heavy dependency in Colab.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from local_agentic_rag import lang
from local_agentic_rag.agent import calculator, presupposition
from local_agentic_rag.agent import prompts as P
from local_agentic_rag.agent.citations import (
    build_citations,
    infer_citations,
    is_refusal,
    strip_invalid_markers,
    strip_trailing_sentinel,
    table_reference_suffix,
)
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.llm.base import LLM, LLMError, LLMOutputError, LLMTimeoutError
from local_agentic_rag.retrieval.hybrid import HybridRetriever
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.schema import REFUSAL_MESSAGE, Answer, ScoredChunk, TraceStep

logger = logging.getLogger(__name__)

History = list[tuple[str, str]]

#: Most translated search queries per question (one per document script it does not share).
MAX_TRANSLATIONS = 3
#: Documents considered when choosing which scripts to translate a question into.
LIKELY_DOCUMENTS = 3


class AgenticRAG:
    def __init__(self, config: RAGConfig, retriever: HybridRetriever, llm: LLM) -> None:
        self.config = config
        self.retriever = retriever
        self.llm = llm

    # ------------------------------------------------------------------ public

    def run(
        self,
        question: str,
        indexes: list[DocumentIndex],
        history: History | None = None,
        on_step: Callable[[TraceStep], None] | None = None,
        on_token: Callable[[dict[str, Any]], None] | None = None,
        answer_language: str | None = None,
    ) -> Answer:
        """Answer ``question``.

        ``answer_language`` is the user's explicit choice from the app's "Answer in" menu
        (one of ``lang.ANSWER_LANGUAGE_CHOICES``); None means: the question's own language,
        or the one the question asks for.

        ``on_token`` receives live answer text: ``{"reset": True}`` when a free-text answer
        generation starts (so a UI clears what an earlier generation showed), then
        ``{"text": piece}`` per streamed piece.  Only answer generations stream, never the
        JSON steps; citations and verification still run on the full text afterwards.
        """
        t0 = time.perf_counter()
        ans = Answer(question=question, answer="", refused=False)
        self._on_step = on_step
        self._on_token = on_token
        self._override = answer_language
        self._choice = lang.choose_answer_language(question, answer_language)
        self._pivot_of: dict[str, str] = {}  # answer text -> language it was written in
        try:
            if self.config.agentic and self.config.agent_mode == "adaptive":
                self._run_adaptive(ans, indexes, history)
            elif self.config.agentic:
                self._run_agentic(ans, indexes, history)
            else:
                self._run_single_pass(ans, indexes)
        except LLMError as exc:
            logger.error("LLM failure: %s", exc)
            ans.error = f"LLM error: {exc}"
            ans.answer = "The language model is unavailable, so no answer could be generated."
        except Exception as exc:  # never crash the chat / benchmark loop
            logger.exception("Agent failure")
            ans.error = f"{type(exc).__name__}: {exc}"
            ans.answer = "An internal error occurred while answering this question."
        self._presupposition_gate(ans)
        self._name_cited_table(ans)
        self._translate_answer(ans)
        ans.latency_sec = time.perf_counter() - t0
        return ans

    # ------------------------------------------------------------------ answer language

    def _refusal(self, ans: Answer) -> str:
        return lang.refusal_message(ans.question, REFUSAL_MESSAGE, getattr(self, "_override", None))

    def _pivot_language(self, passages: list[ScoredChunk]) -> str | None:
        """Language to write the answer in first, when it differs from the one the user wants.

        A 7B model reading Sanskrit and writing Telugu in one step was the weakest case of
        the 22-language benchmark (10/21 correct).  Writing the answer in the passages' own
        language, checking it there, and translating it in a second call splits that into
        two easier steps.  English answers stay direct (16/20 measured), and so do passages
        in several scripts (no single language to write in).
        """
        choice = getattr(self, "_choice", None)
        if not self.config.pivot_translation or not passages or choice is None:
            return None
        if choice.name in (None, "English"):
            return None
        top = passages[:4]
        if len({lang.script_of(p.chunk.text) for p in top}) != 1:
            return None
        source = lang.language_of(" ".join(p.chunk.text for p in top)[:800])
        if not source or source == choice.name or source.endswith("(Latin letters)"):
            return None
        return source

    def _translate_answer(self, ans: Answer) -> None:
        """Translate an answer written in the passages' language into the user's language."""
        if ans.refused or ans.error or not ans.answer:
            return
        # The calculator and table-name steps may have appended a suffix to the generated text.
        drafts = getattr(self, "_pivot_of", {})
        matches = [text for text in drafts if ans.answer.startswith(text)]
        source = drafts[max(matches, key=len)] if matches else None
        if not source:
            return
        msgs = [
            {"role": "system", "content": P.TRANSLATE_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": P.TRANSLATE_ANSWER_USER.format(
                    source=source, target=self._choice.instruction, answer=ans.answer
                ),
            },
        ]

        def call() -> str:
            ans.llm_calls += 1
            if getattr(self, "_on_token", None) is None:
                return self.llm.chat(
                    msgs, max_tokens=self.config.max_answer_tokens, temperature=0.0
                )
            return self._stream_answer(msgs)

        try:
            text = self._step(ans, "translate_answer", call)
        except LLMError as exc:  # keep the checked answer in the passages' language
            logger.warning("answer translation failed: %s", exc)
            return
        text = strip_invalid_markers(
            strip_trailing_sentinel(text or "").strip(), len(ans.retrieved)
        )
        if not text or is_refusal(text):
            return
        ans.citations = build_citations(text, ans.retrieved) or ans.citations
        ans.answer = text

    def _presupposition_gate(self, ans: Answer) -> None:
        """Presupposition gate (``RAGConfig.presupposition_check``, off by default): refuse a question whose
        year, proper-noun phrase or code is in none of the retrieved passages.

        Deliberately narrow: only when the flag is on, the verifier ran and did NOT find the
        answer supported (``grounded is False``; a missing verdict or a supported one never
        qualifies), and the answer is not already a refusal.  See ``agent.presupposition``.
        """
        if not self.config.presupposition_check:
            return
        if ans.refused or ans.error or ans.grounded is not False or not ans.retrieved:
            return
        try:
            missing = presupposition.find_missing_presupposition(
                ans.question, (sc.chunk.text for sc in ans.retrieved)
            )
            if missing is None:
                return
            unsupported = ans.unsupported_claims
            ans.refused = True
            ans.answer = self._refusal(ans)
            ans.citations = []
            # A refusal has no verdict (as after "no passages retrieved") and no claims.
            ans.grounded, ans.unsupported_claims = None, []
            self._record(
                ans,
                "presupposition_refusal",
                {
                    "entity": missing.text,
                    "kind": missing.kind,
                    "unsupported": unsupported,
                },
                0.0,
            )
        except Exception:  # an optional gate must never lose the answer
            logger.debug("presupposition gate failed", exc_info=True)

    def _name_cited_table(self, ans: Answer) -> None:
        """ "Which table ...?" answers: append the cited table's printed number if missing.

        Runs after the mode has finished (so verification and escalation never see it) and
        only for a non-refused, error-free answer to a question that asks which
        table/figure/exhibit.  Deterministic; see ``citations.table_reference_suffix``.
        """
        if ans.refused or ans.error or not ans.answer or not ans.citations:
            return
        try:
            asked = ans.question
            if ans.standalone_question and ans.standalone_question != ans.question:
                asked = f"{ans.question}\n{ans.standalone_question}"
            found = table_reference_suffix(asked, ans.answer, ans.citations, ans.retrieved)
            if found:
                suffix, detail = found
                ans.answer = ans.answer.rstrip() + suffix
                self._record(ans, "table_reference_added", detail, 0.0)
        except Exception:  # an optional nicety must never lose the answer
            logger.debug("table reference step failed", exc_info=True)

    # ------------------------------------------------------------------ modes

    def _run_single_pass(self, ans: Answer, indexes: list[DocumentIndex]) -> None:
        ans.standalone_question = ans.question
        ans.search_queries = [ans.question]
        passages = self._step(ans, "retrieve", lambda: self._retrieve([ans.question], indexes))
        ans.retrieved = passages
        self._generate(ans, passages, feedback="")

    def _run_adaptive(
        self, ans: Answer, indexes: list[DocumentIndex], history: History | None
    ) -> None:
        """Single pass first; escalate to the agent loop only when it is needed.

        Measured on the benchmark: the always-on loop was *worse* than a single pass
        (extra queries and grader reordering diluted good first-round evidence), so the
        single-pass answer is the floor and the loop is spent only on questions whose
        first answer is refused or fails verification.
        """
        cfg = self.config
        title = indexes[0].title if len(indexes) == 1 else "multiple documents"
        standalone, queries = ans.question, [ans.question]
        planned = False
        if history and cfg.use_planner:  # follow-ups must be made standalone before searching
            plan = self._step(ans, "plan", lambda: self._plan(ans.question, title, history))
            planned = True
            if plan:
                standalone = plan["standalone_question"] or ans.question
                queries = [standalone] + [q for q in plan["search_queries"] if q]
        ans.standalone_question = standalone
        lead = self._lead_queries(ans, standalone, indexes)
        ans.search_queries = list(lead)

        # 1. Single pass on the (standalone) question (+ its translation if cross-lingual).
        first = self._step(ans, "retrieve", lambda: self._retrieve(lead, indexes), round=1)
        ans.retrieved = first
        if not first:
            ans.refused = True
            ans.answer = self._refusal(ans)
            self._record(ans, "refuse", {"reason": "no passages retrieved"}, 0.0)
            return
        calc_hint, first = self._calculate(ans, standalone, first, indexes)
        ans.retrieved = first
        self._generate(ans, first, feedback=calc_hint, question=standalone)
        self._ensure_calc_in_answer(ans)
        if not ans.refused and cfg.use_verifier:
            verdict = self._step(ans, "verify", lambda: self._verify(standalone, ans.answer, first))
            if verdict is not None:
                ans.grounded = bool(verdict["supported"])
                ans.unsupported_claims = [c for c in verdict.get("unsupported_claims", []) if c][:5]
        calc_failed = getattr(self, "_calc_failed", False)
        if (
            not (ans.refused or ans.grounded is False or calc_failed)
            or cfg.max_retrieval_rounds < 2
        ):
            return

        # 2. Escalate: broader search, filtered evidence, regenerate with feedback.
        snapshot = (
            ans.answer,
            ans.refused,
            list(ans.citations),
            ans.grounded,
            list(ans.unsupported_claims),
            ans.raw_response,
            ans.retrieved,
        )
        reason = "refused" if ans.refused else "unsupported claims"
        if not ans.refused and ans.grounded is not False:
            reason = "no grounded calculation"
        self._record(ans, "escalate", {"reason": reason}, 0.0)
        if cfg.use_planner and not planned:
            plan = self._step(ans, "plan", lambda: self._plan(standalone, title, history))
            if plan:
                queries = [standalone] + [q for q in plan["search_queries"] if q]
        ans.retrieval_rounds = 2
        ans.search_queries = list(dict.fromkeys(lead + queries[1:]))
        second = self._step(
            ans, "retrieve", lambda: self._retrieve(ans.search_queries, indexes), round=2
        )
        context = _merge(second, first, cfg.top_k)
        if cfg.use_grader:
            grade = self._step(ans, "grade", lambda: self._grade(standalone, context), round=2)
            if grade is not None:
                keep = {
                    context[i - 1].chunk.chunk_id
                    for i in grade["relevant"]
                    if 1 <= i <= len(context)
                }
                context = [
                    p
                    for p in context
                    if p.chunk.chunk_id in keep
                    or p.rerank_score is None
                    or p.rerank_score >= cfg.grader_override_score
                ] or context
        feedback = ""
        if ans.unsupported_claims:
            feedback = P.REGENERATE_FEEDBACK.format(claims="; ".join(ans.unsupported_claims)[:500])
        if calc_hint or calculator.is_quantitative(standalone):
            calc_hint, context = self._calculate(ans, standalone, context, indexes)
            feedback = "\n".join(f for f in (feedback, calc_hint) if f)
        ans.answer_attempts = 2
        ans.retrieved = context
        ans.grounded, ans.unsupported_claims = None, []
        self._generate(ans, context, feedback, question=standalone)
        self._ensure_calc_in_answer(ans)
        if not ans.refused and cfg.use_verifier:
            verdict = self._step(
                ans, "verify", lambda: self._verify(standalone, ans.answer, context)
            )
            if verdict is not None:
                ans.grounded = bool(verdict["supported"])
                ans.unsupported_claims = [c for c in verdict.get("unsupported_claims", []) if c][:5]

        # Keep the first answer if escalation only produced a refusal: a flagged answer
        # with citations is more useful than "not found" when the first pass found evidence.
        first_answer, first_refused = snapshot[0], snapshot[1]
        if ans.refused and not first_refused:
            (
                ans.answer,
                ans.refused,
                ans.citations,
                ans.grounded,
                ans.unsupported_claims,
                ans.raw_response,
                ans.retrieved,
            ) = snapshot
            self._record(ans, "keep_first_answer", {"answer": first_answer[:80]}, 0.0)

    def _run_agentic(
        self, ans: Answer, indexes: list[DocumentIndex], history: History | None
    ) -> None:
        cfg = self.config
        title = indexes[0].title if len(indexes) == 1 else "multiple documents"

        # 1. PLAN -----------------------------------------------------------
        standalone, queries = ans.question, [ans.question]
        if cfg.use_planner:
            plan = self._step(ans, "plan", lambda: self._plan(ans.question, title, history))
            if plan:
                standalone = plan["standalone_question"] or ans.question
                # The literal question is always searched too; planner queries add recall.
                queries = [standalone] + [q for q in plan["search_queries"] if q]
                if history and standalone != ans.question:
                    queries.append(ans.question)
        ans.standalone_question = standalone
        lead = self._lead_queries(ans, standalone, indexes)
        ans.search_queries = list(dict.fromkeys(lead + queries[1:]))

        # 2. RETRIEVE ⇄ GRADE ---------------------------------------------
        passages: list[ScoredChunk] = []
        relevant: list[ScoredChunk] = []
        for round_no in range(1, cfg.max_retrieval_rounds + 1):
            ans.retrieval_rounds = round_no
            fresh = self._step(
                ans, "retrieve", lambda: self._retrieve(ans.search_queries, indexes), round=round_no
            )
            passages = _merge(relevant, fresh, cfg.top_k)
            if not cfg.use_grader:
                relevant = passages
                break
            grade = self._step(
                ans, "grade", lambda ps=passages: self._grade(standalone, ps), round=round_no
            )
            if grade is None:  # grader failed → trust the reranker order
                relevant = passages
                break
            relevant = [passages[i - 1] for i in grade["relevant"] if 1 <= i <= len(passages)]
            if grade["sufficient"] and relevant:
                break
            missing = (grade.get("missing_query") or "").strip()
            if round_no < cfg.max_retrieval_rounds:
                if not missing or missing in ans.search_queries:
                    missing = f"{standalone} details"
                ans.search_queries.append(missing)

        # Context for generation keeps the reranker's order (measured: letting the grader
        # reorder passages pushed the evidence down and hurt answers).  The grader only
        # filters: a passage is dropped when the grader rejected it AND the reranker is
        # not confident about it either (small models are conservative graders).
        keep_ids = {p.chunk.chunk_id for p in relevant}
        context = [
            p
            for p in _merge(passages, relevant, cfg.top_k + len(relevant))
            if p.chunk.chunk_id in keep_ids
            or p.rerank_score is None
            or p.rerank_score >= cfg.grader_override_score
        ][: cfg.top_k]
        ans.retrieved = context
        if not context:
            ans.refused = True
            ans.answer = self._refusal(ans)
            self._record(ans, "refuse", {"reason": "no relevant passages"}, 0.0)
            return

        # 3. GENERATE ⇄ VERIFY ----------------------------------------------
        calc_hint, context = self._calculate(ans, standalone, context, indexes)
        ans.retrieved = context
        feedback = ""
        for attempt in range(1, cfg.max_answer_attempts + 1):
            ans.answer_attempts = attempt
            ans.grounded, ans.unsupported_claims = None, []
            self._generate(
                ans, context, "\n".join(f for f in (feedback, calc_hint) if f), question=standalone
            )
            self._ensure_calc_in_answer(ans)
            if ans.refused or not cfg.use_verifier:
                return
            verdict = self._step(
                ans, "verify", lambda: self._verify(standalone, ans.answer, context)
            )
            if verdict is None:
                return
            ans.grounded = bool(verdict["supported"])
            ans.unsupported_claims = [c for c in verdict.get("unsupported_claims", []) if c][:5]
            if ans.grounded:
                return
            claims = "; ".join(ans.unsupported_claims) or "unspecified claims"
            feedback = P.REGENERATE_FEEDBACK.format(claims=claims[:500])

    # ------------------------------------------------------------------ steps

    def _retrieve(self, queries: list[str], indexes: list[DocumentIndex]) -> list[ScoredChunk]:
        return self.retriever.retrieve(queries, indexes, n_primary=getattr(self, "_n_primary", 1))

    def _lead_queries(self, ans: Answer, question: str, indexes: list[DocumentIndex]) -> list[str]:
        """[question] or, for a cross-lingual question, [question, translation-into-doc-language].

        BGE-M3 already matches across languages, but keyword search (BM25) and the
        reranker do better with a same-language query, and Hinglish is out of
        distribution for both models — one cheap translation call fixes that.  When the
        searched documents use several scripts, the question is translated into each one
        it does not share (at most ``MAX_TRANSLATIONS``, one call each).
        """
        self._n_primary = 1
        if not self.config.translate_queries:
            return [question]
        first_of_script: dict[str, DocumentIndex] = {}
        for ix in self._likely_indexes(question, indexes):
            first_of_script.setdefault(getattr(ix, "script", "unknown"), ix)
        targets = [ix for s, ix in first_of_script.items() if lang.needs_translation(question, s)]
        queries = [question]
        for ix in targets[:MAX_TRANSLATIONS]:
            sample = " ".join(c.text for c in ix.chunks[:3])[:400]
            msgs = [
                {"role": "system", "content": P.TRANSLATE_SYSTEM},
                {
                    "role": "user",
                    "content": P.TRANSLATE_USER.format(sample=sample, question=question),
                },
            ]
            data = self._step(
                ans,
                "translate",
                lambda m=msgs: self._json_call(m, P.TRANSLATE_SCHEMA, max_tokens=120),
            )
            translation = str((data or {}).get("translation", "")).strip()
            if translation and translation not in queries:
                queries.append(translation)
        self._n_primary = len(queries)
        return queries

    def _likely_indexes(self, question: str, indexes: list[DocumentIndex]) -> list[DocumentIndex]:
        """The documents a question most likely concerns, best first.

        With one script in the library every document qualifies.  With several, a cheap
        fused search with the untranslated question (no reranker) ranks the documents, and
        only the top ``LIKELY_DOCUMENTS`` are kept, so a Telugu question over 22 books is
        translated into the scripts of the books it is about, not the first three found.
        """
        if len({getattr(ix, "script", "unknown") for ix in indexes}) <= 1:
            return indexes
        try:
            ranked = self.retriever.fuse([question], indexes)
        except Exception:  # ranking is an optimisation; fall back to every document
            return indexes
        by_id = {ix.doc_id: ix for ix in indexes}
        order: list[DocumentIndex] = []
        for sc in ranked:
            ix = by_id.get(sc.chunk.doc_id)
            if ix is not None and ix not in order:
                order.append(ix)
            if len(order) >= LIKELY_DOCUMENTS:
                break
        return order or indexes

    def _plan(self, question: str, title: str, history: History | None) -> dict[str, Any] | None:
        msgs = [
            {"role": "system", "content": P.PLAN_SYSTEM},
            {
                "role": "user",
                "content": P.PLAN_USER.format(
                    title=title,
                    history=P.format_history(history, self.config.history_turns),
                    question=question,
                ),
            },
        ]
        data = self._json_call(msgs, P.PLAN_SCHEMA, max_tokens=200)
        if data is None:
            return None
        queries = [str(q).strip() for q in data.get("search_queries", []) if str(q).strip()][:3]
        return {
            "standalone_question": str(data.get("standalone_question", "")).strip(),
            "search_queries": queries,
        }

    def _grade(self, question: str, passages: list[ScoredChunk]) -> dict[str, Any] | None:
        if not passages:
            return {"relevant": [], "sufficient": False, "missing_query": ""}
        msgs = [
            {"role": "system", "content": P.GRADE_SYSTEM},
            {
                "role": "user",
                "content": P.GRADE_USER.format(
                    question=question, passages=P.format_passages(passages)
                ),
            },
        ]
        data = self._json_call(msgs, P.GRADE_SCHEMA, max_tokens=120)
        if data is None:
            return None
        relevant = []
        for x in data.get("relevant", []):
            try:
                relevant.append(int(x))
            except (TypeError, ValueError):
                continue
        return {
            "relevant": list(dict.fromkeys(relevant)),
            "sufficient": bool(data.get("sufficient", False)),
            "missing_query": str(data.get("missing_query", "") or ""),
        }

    def _calculate(
        self,
        ans: Answer,
        question: str,
        passages: list[ScoredChunk],
        indexes: list[DocumentIndex] | None = None,
    ) -> tuple[str, list[ScoredChunk]]:
        """Run the calculator tool for quantitative questions: (generator hint or "", passages).

        Passages are returned because counting identifiers may add the chunks that hold
        the rest of a sequence (NIST q10: MS-2.7-001..008 on p37, MS-2.7-009 on p38).
        The result is *forced* into the answer (``_ensure_calc_in_answer``) only when the
        question needs arithmetic by its wording and every operand was checked.
        """
        self._calc = None
        self._calc_failed = False
        if (
            not self.config.use_calculator
            or not passages
            or not calculator.is_quantitative(question)
        ):
            return "", passages
        forced = calculator.needs_arithmetic(question)

        prefix = calculator.id_prefix(question, [p.chunk.text for p in passages])
        if prefix:
            passages = self._complete_id_sequence(ans, prefix, passages, indexes or [])
            calc = calculator.count_ids(prefix, [p.chunk.text for p in passages])
            if calc:
                calc["forced"] = True
                self._calc = calc
                self._record(ans, "calculate", {"output": calc, "deterministic": True}, 0.0)
                return calculator.hint(calc), passages

        if not forced:
            stated = calculator.stated_change(question, [p.chunk.text for p in passages])
            if stated:  # BEA q13: the release says "increased $6.8 billion" — nothing to compute
                self._record(ans, "calc_skipped", {"stated": stated}, 0.0)
                return "", passages

        texts = [p.chunk.text for p in passages]
        cfg = self.config
        dates = ""
        if cfg.calc_date_candidates and calculator.asks_time_span(question):
            dates = calculator.date_candidates([(p.chunk.page, p.chunk.text) for p in passages])
            if dates:
                self._record(ans, "date_candidates", {"dates": dates.count("\n- ")}, 0.0)
        feedback = dates
        calc = None
        rejected = False
        for attempt in (1, 2):  # one retry, told why the first attempt was rejected or empty
            msgs = [
                {"role": "system", "content": calculator.CALC_SYSTEM},
                {
                    "role": "user",
                    "content": calculator.CALC_USER.format(
                        passages=P.format_passages(passages), question=question, feedback=feedback
                    ),
                },
            ]
            outcome = self._step(
                ans,
                "calculate",
                lambda m=msgs: calculator.evaluate(
                    self._json_call(m, calculator.CALC_SCHEMA, max_tokens=450),
                    texts,
                    align=cfg.calc_cell_alignment,
                    question=question if cfg.calc_measure_check else "",
                ),
                attempt=attempt,
            )
            if outcome is None and forced and attempt == 1:
                # Kaggle trace (NIST q11, "how many months between ..."): the first call gave
                # no calculation (needed=false or invalid JSON) and the generator then refused.
                feedback = dates + calculator.CALC_MISSING
                continue
            if not outcome or "rejected" not in outcome:
                calc = outcome
                break
            rejected = True
            feedback = dates + calculator.CALC_FEEDBACK.format(
                problems="\n".join(f"- {p}" for p in outcome["rejected"])
            )
        if not calc:
            # An arithmetic question whose operands were rejected has no usable evidence in
            # these passages; with the measure check on, the adaptive agent then searches
            # wider instead of trusting a verifier that accepts the wrong table's numbers.
            self._calc_failed = cfg.calc_measure_check and forced and rejected
            return "", passages
        calc["forced"] = forced and calc.get("verified", False)
        self._calc = calc
        return calculator.hint(calc, forced=calc["forced"]), passages

    def _complete_id_sequence(
        self,
        ans: Answer,
        prefix: str,
        passages: list[ScoredChunk],
        indexes: list[DocumentIndex],
    ) -> list[ScoredChunk]:
        """Append the chunks that list further ``prefix-NNN`` IDs missing from the passages.

        A list that spans a page break is split over several chunks and retrieval rarely
        returns all of them, so a count over the passages alone comes out short.
        """
        have = {p.chunk.chunk_id for p in passages}
        seen = set().union(*(calculator.ids_with_prefix(p.chunk.text, prefix) for p in passages))
        extra: list[ScoredChunk] = []
        for ix in indexes:
            for chunk in ix.chunks:
                new = calculator.ids_with_prefix(chunk.text, prefix) - seen
                if chunk.chunk_id not in have and new and len(extra) < self.config.top_k:
                    extra.append(ScoredChunk(chunk=chunk, score=0.0))
                    seen |= new
        if extra:
            self._record(
                ans, "complete_ids", {"prefix": prefix, "pages": [p.chunk.page for p in extra]}, 0.0
            )
        return passages + extra

    def _ensure_calc_in_answer(self, ans: Answer) -> None:
        """If the generator ignored the computed result, state it (with citations) explicitly.

        Observed on Colab: given "computed 0.59 - 0.5 = 0.09" the model still wrote its own
        number.  The citations refer to the same passage list the calculator saw.  Only
        forced results are appended: a hint for a lookup question (BEA q11 got an
        unrequested difference appended) or an unchecked operand never overrides the answer.
        """
        calc = getattr(self, "_calc", None)
        if (
            not calc
            or not calc.get("forced")
            or ans.refused
            or calculator.result_in_answer(ans.answer, calc)
        ):
            return
        ans.answer = ans.answer.rstrip() + calculator.answer_suffix(calc)
        ans.citations = build_citations(ans.answer, ans.retrieved)
        self._record(ans, "calc_appended", {"result": calc["result"]}, 0.0)

    def _verify(
        self, question: str, answer: str, passages: list[ScoredChunk]
    ) -> dict[str, Any] | None:
        msgs = [
            {"role": "system", "content": P.VERIFY_SYSTEM},
            {
                "role": "user",
                "content": P.VERIFY_USER.format(
                    passages=P.format_passages(passages), question=question, answer=answer
                ),
            },
        ]
        data = self._json_call(msgs, P.VERIFY_SCHEMA, max_tokens=200)
        if data is None:
            return None
        return {
            "supported": bool(data.get("supported", False)),
            "unsupported_claims": [str(c) for c in data.get("unsupported_claims", [])],
        }

    def _generate(
        self, ans: Answer, passages: list[ScoredChunk], feedback: str, question: str | None = None
    ) -> None:
        question = question or ans.question
        title = passages[0].chunk.doc_title if passages else ""
        pivot = self._pivot_language(passages)
        if pivot:
            answer_instruction = f"{pivot}, the language of the passages"
        elif getattr(self, "_choice", None) is not None:
            answer_instruction = self._choice.instruction
        else:
            answer_instruction = lang.answer_language(ans.question)
        msgs = [
            {"role": "system", "content": P.ANSWER_SYSTEM},
            {
                "role": "user",
                "content": P.ANSWER_USER.format(
                    title=title,
                    passages=P.format_passages(passages),
                    question=question,
                    sentinel=P.ANSWER_NOT_FOUND,
                    feedback=feedback,
                    answer_language=answer_instruction,
                ),
            },
        ]

        def call() -> str:
            ans.llm_calls += 1
            if getattr(self, "_on_token", None) is None:
                return self.llm.chat(
                    msgs,
                    max_tokens=self.config.max_answer_tokens,
                    temperature=self.config.temperature,
                )
            return self._stream_answer(msgs)

        raw = self._step(ans, "generate", call)
        ans.raw_response = raw
        raw = strip_trailing_sentinel(raw)
        if is_refusal(raw):
            ans.refused = True
            ans.answer = self._refusal(ans)
            ans.citations = []
            return
        ans.refused = False
        text = strip_invalid_markers(raw, len(passages))
        citations = build_citations(text, passages)
        if not citations:
            citations = infer_citations(text, passages)
            if citations:
                self._record(ans, "infer_citations", {"pages": [c.page for c in citations]}, 0.0)
        ans.answer = text
        ans.citations = citations
        if pivot:
            self._pivot_of[text] = pivot

    def _stream_answer(self, msgs: list[dict[str, str]]) -> str:
        """Run one free-text generation through ``chat_stream``, forwarding pieces to the UI.

        The joined pieces are stripped, matching what ``chat`` returns, so everything after
        this point (refusal check, citations, verification) sees identical text.
        """
        self._emit_token({"reset": True})
        parts: list[str] = []
        try:
            for piece in self.llm.chat_stream(
                msgs, max_tokens=self.config.max_answer_tokens, temperature=self.config.temperature
            ):
                if piece:
                    parts.append(piece)
                    self._emit_token({"text": piece})
        except LLMTimeoutError:
            raise  # chat() never retries a read timeout either; a second wait is pointless
        except LLMError as exc:
            # A stream that breaks part-way (e.g. the runner was restarted) cannot be
            # resumed; fall back to the non-streaming call, which has its own retries, and
            # tell the UI to drop the partial text.
            logger.warning("answer stream failed, retrying without streaming: %s", exc)
            self._emit_token({"reset": True})
            text = self.llm.chat(
                msgs, max_tokens=self.config.max_answer_tokens, temperature=self.config.temperature
            )
            if text:
                self._emit_token({"text": text})
            return text
        return "".join(parts).strip()

    def _emit_token(self, event: dict[str, Any]) -> None:
        try:
            self._on_token(event)
        except Exception:  # UI callbacks must never break answering
            logger.debug("on_token callback failed", exc_info=True)

    # ------------------------------------------------------------------ plumbing

    def _json_call(
        self, msgs: list[dict[str, str]], schema: dict[str, Any], max_tokens: int
    ) -> dict[str, Any] | None:
        try:
            self._calls += 1
            return self.llm.chat_json(msgs, schema, max_tokens=max_tokens)
        except LLMOutputError as exc:  # bad JSON → degrade; backend failures propagate
            logger.warning("Structured step failed, falling back: %s", exc)
            return None

    _calls = 0

    def _step(self, ans: Answer, name: str, fn: Callable[[], Any], **info: Any) -> Any:
        before = self._calls
        t = time.perf_counter()
        result = fn()
        ans.llm_calls += self._calls - before
        self._record(ans, name, {**info, **_summarise(name, result)}, time.perf_counter() - t)
        return result

    def _record(self, ans: Answer, name: str, detail: dict[str, Any], seconds: float) -> None:
        step = TraceStep(step=name, detail=detail, seconds=round(seconds, 3))
        ans.trace.append(step)
        if getattr(self, "_on_step", None):
            try:
                self._on_step(step)
            except Exception:  # UI callbacks must never break answering
                logger.debug("on_step callback failed", exc_info=True)


def _merge(first: list[ScoredChunk], second: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
    out: list[ScoredChunk] = []
    seen: set[str] = set()
    for sc in [*first, *second]:
        if sc.chunk.chunk_id not in seen:
            seen.add(sc.chunk.chunk_id)
            out.append(sc)
        if len(out) >= limit:
            break
    return out


def _summarise(name: str, result: Any) -> dict[str, Any]:
    if name == "retrieve" and isinstance(result, list):
        return {
            "pages": [sc.chunk.page for sc in result],
            "top_score": round(result[0].score, 3) if result else None,
        }
    if name == "generate" and isinstance(result, str):
        return {"chars": len(result), "preview": result[:120]}
    if isinstance(result, dict):
        return {"output": result}
    if result is None:
        return {"output": None, "fallback": True}
    return {}
