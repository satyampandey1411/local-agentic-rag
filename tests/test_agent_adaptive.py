"""Adaptive agent mode: single pass first, escalate only on refusal or failed verification."""

from __future__ import annotations

from local_agentic_rag.schema import ANSWER_NOT_FOUND, REFUSAL_MESSAGE
from tests.test_agent import (
    PLAN_OK,
    QUESTION,
    VERIFY_OK,
    calls_by_role,
    cite,
    grade_relevant,
    index,  # noqa: F401 - pytest fixture
    make_agent,
    steps,
    user_prompts,
)

UNSUPPORTED = {"supported": False, "unsupported_claims": ["revenue doubled"]}


def adaptive(responder, **cfg):
    return make_agent(responder, agent_mode="adaptive", **cfg)


def test_good_first_answer_does_not_escalate(index):  # noqa: F811
    agent, llm = adaptive(
        _scripted(
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."), verify=VERIFY_OK
        )
    )
    ans = agent.run(QUESTION, [index])
    assert steps(ans) == ["retrieve", "generate", "verify"]
    assert calls_by_role(llm) == ["generate", "verify"]
    assert ans.grounded is True and ans.retrieval_rounds == 1 and ans.cited_pages == [3]


def _scripted(**kw):
    from tests.helpers import scripted

    return scripted(**kw)


def test_unsupported_first_answer_escalates_with_feedback(index):  # noqa: F811
    agent, llm = adaptive(
        _scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=[
                cite("Revenue increased", "Revenue doubled to 4.2 billion [{n}]."),
                cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            ],
            verify=[UNSUPPORTED, VERIFY_OK],
        )
    )
    ans = agent.run(QUESTION, [index])
    assert steps(ans) == [
        "retrieve",
        "generate",
        "verify",
        "escalate",
        "plan",
        "retrieve",
        "grade",
        "generate",
        "verify",
    ]
    assert ans.retrieval_rounds == 2 and ans.answer_attempts == 2
    assert ans.grounded is True
    assert "revenue doubled" in user_prompts(llm, "generate")[1]
    assert ans.answer == "Revenue was 4.2 billion [1]."


def test_refusal_escalates_and_can_recover(index):  # noqa: F811
    agent, _ = adaptive(
        _scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=[
                ANSWER_NOT_FOUND,
                cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
            ],
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert "escalate" in steps(ans)
    assert ans.refused is False and ans.cited_pages == [3]


def test_escalation_ending_in_refusal_keeps_first_answer(index):  # noqa: F811
    agent, _ = adaptive(
        _scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=[cite("Revenue increased", "Revenue doubled [{n}]."), ANSWER_NOT_FOUND],
            verify=UNSUPPORTED,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert steps(ans)[-1] == "keep_first_answer"
    assert ans.refused is False
    assert ans.answer == "Revenue doubled [1]."
    assert ans.grounded is False and ans.unsupported_claims == ["revenue doubled"]


def test_refused_twice_stays_refused(index):  # noqa: F811
    agent, _ = adaptive(
        _scripted(plan=PLAN_OK, grade=grade_relevant(), generate=ANSWER_NOT_FOUND, verify=VERIFY_OK)
    )
    ans = agent.run(QUESTION, [index])
    assert ans.refused is True and ans.answer == REFUSAL_MESSAGE
    assert ans.retrieval_rounds == 2


def test_history_condenses_before_first_retrieval(index):  # noqa: F811
    plan = {"standalone_question": QUESTION, "search_queries": ["revenue 2023"]}
    agent, llm = adaptive(
        _scripted(
            plan=plan, generate=cite("Revenue increased", "4.2 billion [{n}]."), verify=VERIFY_OK
        )
    )
    ans = agent.run("And revenue?", [index], history=[("How many employees?", "12,000 [1].")])
    assert calls_by_role(llm)[0] == "plan"
    assert ans.standalone_question == QUESTION
    assert steps(ans) == ["plan", "retrieve", "generate", "verify"]


def test_cited_answer_with_trailing_sentinel_is_answered_not_refused(index):  # noqa: F811
    # Kaggle trace (Hinglish two-part question): a correct cited answer, then the sentinel.
    agent, llm = adaptive(
        _scripted(
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]. " + ANSWER_NOT_FOUND),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert not ans.refused and "escalate" not in steps(ans)
    assert ans.answer == "Revenue was 4.2 billion [1]." and ans.cited_pages == [3]
    assert ANSWER_NOT_FOUND in ans.raw_response  # the raw model output is kept for the trace
