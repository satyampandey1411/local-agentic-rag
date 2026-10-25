"""Unit tests for local_agentic_rag.agent.agent.AgenticRAG with scripted LLMs and fake models."""

from __future__ import annotations

import pytest

from local_agentic_rag.agent import prompts as P
from local_agentic_rag.agent.agent import AgenticRAG
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.llm.base import LLMError
from local_agentic_rag.llm.fake import ScriptedLLM
from local_agentic_rag.retrieval.hybrid import HybridRetriever
from local_agentic_rag.schema import ANSWER_NOT_FOUND, REFUSAL_MESSAGE
from tests.helpers import (
    FakeEmbedder,
    FakeReranker,
    make_index,
    passage_number,
    role_of,
    scripted,
)

CORPUS = [
    (3, "Revenue increased to 4.2 billion dollars in fiscal 2023, driven by cloud subscriptions."),
    (5, "The company employs 12,000 people across 40 countries worldwide."),
    (7, "Operating margin improved to 18 percent after restructuring charges ended."),
    (9, "The board approved a dividend of 0.50 dollars per share in March."),
    (11, "Headquarters relocated to Austin, Texas during the summer."),
    (12, "Research spending reached 600 million dollars, mostly on software."),
]

QUESTION = "What was revenue in fiscal 2023?"


@pytest.fixture
def index():
    return make_index(CORPUS, FakeEmbedder(), doc_id="rep", title="Annual Report")


def make_agent(responder, **cfg) -> tuple[AgenticRAG, ScriptedLLM]:
    cfg.setdefault("agent_mode", "full")  # these tests exercise the always-on loop
    config = RAGConfig(**cfg)
    llm = ScriptedLLM(responder)
    return AgenticRAG(config, HybridRetriever(config, FakeEmbedder(), FakeReranker()), llm), llm


def cite(needle: str, template: str):
    """Generator reply that cites the passage containing ``needle``: template has {n}."""
    return lambda messages: template.format(n=passage_number(messages, needle))


def grade_relevant(*needles: str, sufficient: bool = True, missing: str = ""):
    return lambda messages: {
        "relevant": [passage_number(messages, n) for n in needles],
        "sufficient": sufficient,
        "missing_query": missing,
    }


PLAN_OK = {"standalone_question": QUESTION, "search_queries": ["revenue fiscal 2023"]}
VERIFY_OK = {"supported": True, "unsupported_claims": []}


def steps(ans) -> list[str]:
    return [t.step for t in ans.trace]


def calls_by_role(llm: ScriptedLLM) -> list[str]:
    return [role_of(c["messages"], c["json_schema"]) for c in llm.calls]


def user_prompts(llm: ScriptedLLM, role: str) -> list[str]:
    return [
        c["messages"][-1]["content"]
        for c in llm.calls
        if role_of(c["messages"], c["json_schema"]) == role
    ]


# ---------------------------------------------------------------------------
# (1) happy path
# ---------------------------------------------------------------------------


def test_happy_path_agentic(index):
    agent, llm = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite(
                "Revenue increased", "Revenue was 4.2 billion dollars in fiscal 2023 [{n}]."
            ),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])

    assert ans.error is None
    assert steps(ans) == ["plan", "retrieve", "grade", "generate", "verify"]
    assert ans.llm_calls == 4 == len(llm.calls)
    assert calls_by_role(llm) == ["plan", "grade", "generate", "verify"]
    assert ans.refused is False
    assert ans.grounded is True
    assert ans.unsupported_claims == []
    assert ans.cited_pages == [3]
    assert ans.citations[0].chunk_id == "rep-p3-c0"
    assert ans.answer == "Revenue was 4.2 billion dollars in fiscal 2023 [1]."
    assert ans.raw_response == ans.answer
    assert ans.retrieval_rounds == 1
    assert ans.answer_attempts == 1
    assert ans.standalone_question == QUESTION
    assert ans.search_queries == [QUESTION, "revenue fiscal 2023"]
    assert ans.retrieved[0].chunk.page == 3
    assert ans.latency_sec > 0


def test_trace_details_and_on_step_callback(index):
    seen = []
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index], on_step=seen.append)
    assert [s.step for s in seen] == steps(ans)
    retrieve = ans.trace[1].detail
    assert retrieve["round"] == 1
    assert retrieve["pages"][0] == 3
    assert ans.trace[0].detail["output"]["standalone_question"] == QUESTION
    assert ans.trace[3].detail["preview"].startswith("Revenue was 4.2 billion")


def test_on_step_callback_errors_are_swallowed(index):
    def boom(step):
        raise RuntimeError("UI crashed")

    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index], on_step=boom)
    assert ans.error is None
    assert ans.grounded is True


def test_invalid_markers_stripped_and_citations_built(index):
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}][42]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert "[42]" not in ans.answer
    assert [c.marker for c in ans.citations] == [1]
    assert "[42]" in ans.raw_response


def test_uncited_answer_gets_inferred_citations(index):
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate="Revenue increased to 4.2 billion dollars in fiscal 2023.",
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert "infer_citations" in steps(ans)
    assert ans.cited_pages == [3]
    assert ans.refused is False


# ---------------------------------------------------------------------------
# (2) corrective retrieval
# ---------------------------------------------------------------------------


def test_grader_insufficient_triggers_second_retrieval(index):
    question = "What was revenue and the dividend per share?"
    agent, llm = make_agent(
        scripted(
            plan={"standalone_question": question, "search_queries": ["revenue fiscal 2023"]},
            grade=[
                grade_relevant(
                    "Revenue increased", sufficient=False, missing="board dividend per share"
                ),
                grade_relevant("Revenue increased", "dividend of 0.50"),
            ],
            generate=lambda m: (
                f"Revenue was 4.2 billion [{passage_number(m, 'Revenue increased')}] and the dividend "
                f"was 0.50 dollars [{passage_number(m, 'dividend of 0.50')}]."
            ),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(question, [index])
    assert ans.error is None
    assert ans.retrieval_rounds == 2
    assert steps(ans) == ["plan", "retrieve", "grade", "retrieve", "grade", "generate", "verify"]
    assert ans.search_queries[-1] == "board dividend per share"
    assert [t.detail["round"] for t in ans.trace if t.step == "retrieve"] == [1, 2]
    assert ans.llm_calls == 5 == len(llm.calls)
    assert ans.cited_pages == [3, 9]


def test_missing_query_defaults_when_grader_gives_none(index):
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=[
                grade_relevant("Revenue increased", sufficient=False),
                grade_relevant("Revenue increased"),
            ],
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.retrieval_rounds == 2
    assert ans.search_queries[-1] == f"{QUESTION} details"


def test_retrieval_rounds_are_bounded(index):
    agent, llm = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased", sufficient=False, missing="more revenue"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        ),
        max_retrieval_rounds=1,
    )
    ans = agent.run(QUESTION, [index])
    assert ans.retrieval_rounds == 1
    assert steps(ans).count("retrieve") == 1
    assert "more revenue" not in ans.search_queries
    assert ans.llm_calls == 4


def test_no_relevant_passages_refuses_without_generating(index):
    question = "Which penguin species nest on glaciers?"
    agent, llm = make_agent(
        scripted(
            plan={"standalone_question": question, "search_queries": ["penguin glacier nesting"]},
            grade={"relevant": [], "sufficient": False, "missing_query": "penguin nesting"},
        )
    )
    ans = agent.run(question, [index])
    assert ans.error is None
    assert ans.refused is True
    assert ans.answer == REFUSAL_MESSAGE
    assert steps(ans)[-1] == "refuse"
    assert "generate" not in calls_by_role(llm)
    assert ans.retrieved == []
    assert ans.llm_calls == 3  # plan + 2 grades


# ---------------------------------------------------------------------------
# (3) verification + regeneration
# ---------------------------------------------------------------------------


def test_unsupported_answer_is_regenerated_with_feedback(index):
    agent, llm = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=[
                cite("Revenue increased", "Revenue was 9 billion dollars [{n}]."),
                cite("Revenue increased", "Revenue was 4.2 billion dollars [{n}]."),
            ],
            verify=[
                {"supported": False, "unsupported_claims": ["Revenue was 9 billion dollars"]},
                VERIFY_OK,
            ],
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.answer_attempts == 2
    assert ans.grounded is True
    assert ans.answer == "Revenue was 4.2 billion dollars [1]."
    assert steps(ans) == ["plan", "retrieve", "grade", "generate", "verify", "generate", "verify"]
    assert ans.llm_calls == 6
    first, second = user_prompts(llm, "generate")
    assert "previous draft" not in first
    assert "Revenue was 9 billion dollars" in second
    assert P.REGENERATE_FEEDBACK.format(claims="Revenue was 9 billion dollars") in second


def test_answer_attempts_bounded_and_ungrounded_reported(index):
    agent, llm = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 9 billion dollars [{n}]."),
            verify={"supported": False, "unsupported_claims": ["Revenue was 9 billion", ""]},
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.answer_attempts == 2
    assert ans.grounded is False
    assert ans.unsupported_claims == ["Revenue was 9 billion"]
    assert calls_by_role(llm).count("generate") == 2
    assert ans.error is None


def test_verifier_disabled(index):
    agent, llm = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
        ),
        use_verifier=False,
    )
    ans = agent.run(QUESTION, [index])
    assert ans.grounded is None
    assert ans.llm_calls == 3
    assert "verify" not in steps(ans)


# ---------------------------------------------------------------------------
# (4) refusal
# ---------------------------------------------------------------------------


def test_generator_sentinel_is_refusal(index):
    agent, llm = make_agent(
        scripted(plan=PLAN_OK, grade=grade_relevant("Revenue increased"), generate=ANSWER_NOT_FOUND)
    )
    ans = agent.run(QUESTION, [index])
    assert ans.refused is True
    assert ans.answer == REFUSAL_MESSAGE
    assert ans.citations == []
    assert ans.raw_response == ANSWER_NOT_FOUND
    assert ans.grounded is None
    assert "verify" not in calls_by_role(llm)
    assert ans.llm_calls == 3
    assert ans.error is None


def test_uncited_does_not_mention_is_refusal(index):
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate="The document does not mention revenue for 2019.",
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.refused is True
    assert ans.answer == REFUSAL_MESSAGE


# ---------------------------------------------------------------------------
# (5) graceful fallback on bad structured output
# ---------------------------------------------------------------------------


def test_planner_invalid_json_falls_back_to_question(index):
    agent, llm = make_agent(
        scripted(
            plan="I would search for revenue figures.",
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.error is None
    assert ans.standalone_question == QUESTION
    assert ans.search_queries == [QUESTION]
    plan_step = ans.trace[0]
    assert plan_step.step == "plan" and plan_step.detail.get("fallback") is True
    assert ans.llm_calls == 4
    assert QUESTION in user_prompts(llm, "generate")[0]
    assert ans.cited_pages == [3]


def test_planner_json_in_code_fence_is_parsed(index):
    agent, _ = make_agent(
        scripted(
            plan='```json\n{"standalone_question": "Q2", "search_queries": ["a", "", "b", "c", "d"]}\n```',
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.standalone_question == "Q2"
    assert ans.search_queries == ["Q2", "a", "b", "c"]  # max 3 planner queries, blanks dropped


def test_invalid_json_mentioning_http_still_falls_back(index):
    agent, _ = make_agent(
        scripted(
            plan="Search the HTTP API section for revenue.",
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.error is None


def test_grader_invalid_json_trusts_reranker(index):
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade="relevant: 1",
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.error is None
    assert ans.retrieval_rounds == 1
    assert ans.retrieved[0].chunk.page == 3
    assert ans.grounded is True


def test_grader_non_integer_indices_ignored(index):
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=lambda m: {
                "relevant": ["x", None, passage_number(m, "Revenue increased"), 99],
                "sufficient": True,
                "missing_query": "",
            },
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.error is None
    assert ans.retrieved[0].chunk.page == 3


def test_verifier_invalid_json_leaves_grounded_unknown(index):
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify="looks fine to me",
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.error is None
    assert ans.grounded is None
    assert ans.answer_attempts == 1


# ---------------------------------------------------------------------------
# (6) backend failures
# ---------------------------------------------------------------------------


def test_llm_unreachable_sets_error_without_raising(index):
    agent, _ = make_agent(scripted(plan=LLMError("Ollama unreachable at http://localhost:11434")))
    ans = agent.run(QUESTION, [index])
    assert ans.error is not None
    assert ans.error.startswith("LLM error:")
    assert "unreachable" in ans.error
    assert ans.answer == "The language model is unavailable, so no answer could be generated."


def test_llm_unreachable_during_generation(index):
    agent, _ = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=LLMError("Ollama unreachable: connection refused"),
        )
    )
    ans = agent.run(QUESTION, [index])
    assert ans.error.startswith("LLM error:")
    assert ans.citations == []


def test_unexpected_exception_is_caught(index):
    agent, _ = make_agent(scripted(plan=ValueError("boom")))
    ans = agent.run(QUESTION, [index])
    assert ans.error == "ValueError: boom"
    assert ans.answer == "An internal error occurred while answering this question."


# ---------------------------------------------------------------------------
# (7) single-pass mode
# ---------------------------------------------------------------------------


def test_non_agentic_makes_exactly_one_llm_call(index):
    agent, llm = make_agent(
        scripted(generate=cite("Revenue increased", "Revenue was 4.2 billion dollars [{n}].")),
        agentic=False,
    )
    ans = agent.run(QUESTION, [index])
    assert len(llm.calls) == 1
    assert ans.llm_calls == 1
    assert steps(ans) == ["retrieve", "generate"]
    assert ans.standalone_question == QUESTION
    assert ans.search_queries == [QUESTION]
    assert ans.cited_pages == [3]
    assert ans.grounded is None


def test_planner_and_grader_can_be_disabled(index):
    agent, llm = make_agent(
        scripted(
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."), verify=VERIFY_OK
        ),
        use_planner=False,
        use_grader=False,
    )
    ans = agent.run(QUESTION, [index])
    assert calls_by_role(llm) == ["generate", "verify"]
    assert steps(ans) == ["retrieve", "generate", "verify"]
    assert ans.search_queries == [QUESTION]


# ---------------------------------------------------------------------------
# (8) conversation history
# ---------------------------------------------------------------------------


def test_history_is_given_to_planner_and_standalone_question_used(index):
    history = [
        ("Who is the auditor?", "The auditor is not named."),
        ("What was revenue in fiscal 2023?", "Revenue was 4.2 billion dollars [1]."),
    ]
    follow_up = "What drove it?"
    standalone = "What drove revenue growth in fiscal 2023?"
    agent, llm = make_agent(
        scripted(
            plan={"standalone_question": standalone, "search_queries": ["revenue growth driver"]},
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Growth was driven by cloud subscriptions [{n}]."),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(follow_up, [index], history=history)
    plan_prompt = user_prompts(llm, "plan")[0]
    assert "User: What was revenue in fiscal 2023?" in plan_prompt
    assert "Assistant: Revenue was 4.2 billion dollars [1]." in plan_prompt
    assert f"User question: {follow_up}" in plan_prompt
    assert ans.question == follow_up
    assert ans.standalone_question == standalone
    assert ans.search_queries == [standalone, "revenue growth driver", follow_up]
    assert f"Question: {standalone}" in user_prompts(llm, "generate")[0]
    assert f"Question: {standalone}" in user_prompts(llm, "grade")[0]
    assert f"Question: {standalone}" in user_prompts(llm, "verify")[0]
    assert ans.cited_pages == [3]


def test_history_truncated_to_history_turns(index):
    history = [(f"question number {i}", f"answer {i}") for i in range(6)]
    agent, llm = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        ),
        history_turns=2,
    )
    agent.run(QUESTION, [index], history=history)
    prompt = user_prompts(llm, "plan")[0]
    assert "question number 5" in prompt and "question number 4" in prompt
    assert "question number 3" not in prompt


def test_no_history_shows_none(index):
    agent, llm = make_agent(
        scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            verify=VERIFY_OK,
        )
    )
    agent.run(QUESTION, [index])
    assert "Conversation so far:\n(none)" in user_prompts(llm, "plan")[0]
    assert "Document title: Annual Report" in user_prompts(llm, "plan")[0]
