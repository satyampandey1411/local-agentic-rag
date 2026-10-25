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
from local_agentic_rag.agent import calculator
from local_agentic_rag.agent import prompts as P
from local_agentic_rag.agent.citations import (
    build_citations,
    infer_citations,
    is_refusal,
    strip_invalid_markers,
)
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.llm.base import LLM, LLMError, LLMOutputError
from local_agentic_rag.retrieval.hybrid import HybridRetriever
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.schema import REFUSAL_MESSAGE, Answer, ScoredChunk, TraceStep

logger = logging.getLogger(__name__)

History = list[tuple[str, str]]


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
    ) -> Answer:
        t0 = time.perf_counter()
        ans = Answer(question=question, answer="", refused=False)
        self._on_step = on_step
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
        ans.latency_sec = time.perf_counter() - t0
        return ans

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
            ans.answer = lang.refusal_message(ans.question, REFUSAL_MESSAGE)
            self._record(ans, "refuse", {"reason": "no passages retrieved"}, 0.0)
            return
        calc_hint = self._calculate(ans, standalone, first)
        self._generate(ans, first, feedback=calc_hint, question=standalone)
        self._ensure_calc_in_answer(ans)
        if not ans.refused and cfg.use_verifier:
            verdict = self._step(ans, "verify", lambda: self._verify(standalone, ans.answer, first))
            if verdict is not None:
                ans.grounded = bool(verdict["supported"])
                ans.unsupported_claims = [c for c in verdict.get("unsupported_claims", []) if c][:5]
        if not (ans.refused or ans.grounded is False) or cfg.max_retrieval_rounds < 2:
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
        self._record(
            ans, "escalate", {"reason": "refused" if ans.refused else "unsupported claims"}, 0.0
        )
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
            calc_hint = self._calculate(ans, standalone, context)
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
            ans.answer = lang.refusal_message(ans.question, REFUSAL_MESSAGE)
            self._record(ans, "refuse", {"reason": "no relevant passages"}, 0.0)
            return

        # 3. GENERATE ⇄ VERIFY ----------------------------------------------
        calc_hint = self._calculate(ans, standalone, context)
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
        distribution for both models — one cheap translation call fixes that.
        """
        self._n_primary = 1
        scripts = {getattr(ix, "script", "unknown") for ix in indexes}
        doc_script = scripts.pop() if len(scripts) == 1 else "unknown"
        if not (self.config.translate_queries and lang.needs_translation(question, doc_script)):
            return [question]
        sample = " ".join(c.text for c in indexes[0].chunks[:3])[:400]
        msgs = [
            {"role": "system", "content": P.TRANSLATE_SYSTEM},
            {"role": "user", "content": P.TRANSLATE_USER.format(sample=sample, question=question)},
        ]
        data = self._step(
            ans, "translate", lambda: self._json_call(msgs, P.TRANSLATE_SCHEMA, max_tokens=120)
        )
        translation = str((data or {}).get("translation", "")).strip()
        if not translation or translation == question:
            return [question]
        self._n_primary = 2
        return [question, translation]

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

    def _calculate(self, ans: Answer, question: str, passages: list[ScoredChunk]) -> str:
        """Run the calculator tool for quantitative questions; returns a generator hint or ""."""
        self._calc = None
        if (
            not self.config.use_calculator
            or not passages
            or not calculator.is_quantitative(question)
        ):
            return ""
        msgs = [
            {"role": "system", "content": calculator.CALC_SYSTEM},
            {
                "role": "user",
                "content": calculator.CALC_USER.format(
                    passages=P.format_passages(passages), question=question
                ),
            },
        ]
        result = self._step(
            ans,
            "calculate",
            lambda: calculator.run(self._json_call(msgs, calculator.CALC_SCHEMA, max_tokens=350)),
        )
        self._calc = result
        return calculator.hint(result) if result else ""

    def _ensure_calc_in_answer(self, ans: Answer) -> None:
        """If the generator ignored the computed result, state it (with citations) explicitly.

        Observed on Colab: given "computed 0.59 - 0.5 = 0.09" the model still wrote its own
        number.  The citations refer to the same passage list the calculator saw.
        """
        calc = getattr(self, "_calc", None)
        if not calc or ans.refused or calculator.result_in_answer(ans.answer, calc):
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
                    answer_language=lang.answer_language(ans.question),
                ),
            },
        ]

        def call() -> str:
            ans.llm_calls += 1
            return self.llm.chat(
                msgs, max_tokens=self.config.max_answer_tokens, temperature=self.config.temperature
            )

        raw = self._step(ans, "generate", call)
        ans.raw_response = raw
        if is_refusal(raw):
            ans.refused = True
            ans.answer = lang.refusal_message(ans.question, REFUSAL_MESSAGE)
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
