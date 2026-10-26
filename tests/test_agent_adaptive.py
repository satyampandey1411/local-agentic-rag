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
            generate=cite(
                "Revenue increased", "Revenue was 4.2 billion [{n}]. " + ANSWER_NOT_FOUND
            ),
            verify=VERIFY_OK,
        )
    )
    ans = agent.run(QUESTION, [index])
    assert not ans.refused and "escalate" not in steps(ans)
    assert ans.answer == "Revenue was 4.2 billion [1]." and ans.cited_pages == [3]
    assert ANSWER_NOT_FOUND in ans.raw_response  # the raw model output is kept for the trace


# ---------------------------------------------------------------------------
# token streaming (on_token)
# ---------------------------------------------------------------------------


def _generations(events):
    """Group on_token events into the texts of each generation (split at every reset)."""
    out: list[str] = []
    for ev in events:
        if ev.get("reset"):
            out.append("")
        elif "text" in ev:
            out[-1] += ev["text"]
    return out


def test_on_token_receives_exactly_the_generated_text(index):  # noqa: F811
    script = {
        "generate": cite("Revenue increased", "  Revenue was 4.2 billion in fiscal 2023 [{n}].  "),
        "verify": VERIFY_OK,
    }
    events: list[dict] = []
    agent, llm = adaptive(_scripted(**script))
    ans = agent.run(QUESTION, [index], on_token=events.append)

    assert events[0] == {"reset": True}
    assert len([e for e in events if "text" in e]) > 1  # really streamed in several pieces
    assert _generations(events) == ["  Revenue was 4.2 billion in fiscal 2023 [1].  "]
    assert _generations(events)[0].strip() == ans.raw_response
    # only the free-text generation streams; the JSON verify call does not
    assert llm.stream_calls == 1
    assert calls_by_role(llm) == ["generate", "verify"]
    # post-processing is unchanged: citations and verification run on the full text
    assert ans.answer == "Revenue was 4.2 billion in fiscal 2023 [1]." and ans.cited_pages == [3]
    assert ans.grounded is True


def test_on_token_none_makes_the_same_calls_as_streaming(index):  # noqa: F811
    script = {
        "generate": cite("Revenue increased", "Revenue was 4.2 billion [{n}]."),
        "verify": VERIFY_OK,
    }
    plain_agent, plain_llm = adaptive(_scripted(**script))
    plain = plain_agent.run(QUESTION, [index])
    stream_agent, stream_llm = adaptive(_scripted(**script))
    streamed = stream_agent.run(QUESTION, [index], on_token=lambda e: None)

    assert plain_llm.stream_calls == 0
    assert plain_llm.calls == stream_llm.calls  # identical prompts, schemas and order
    assert plain.answer == streamed.answer and plain.raw_response == streamed.raw_response
    assert plain.llm_calls == streamed.llm_calls
    assert steps(plain) == steps(streamed)


def test_on_token_resets_when_the_agent_regenerates(index):  # noqa: F811
    first = "Revenue doubled to 4.2 billion [1]."
    second = "Revenue was 4.2 billion [1]."
    events: list[dict] = []
    agent, _ = adaptive(
        _scripted(
            plan=PLAN_OK,
            grade=grade_relevant("Revenue increased"),
            generate=[
                cite("Revenue increased", first.replace("[1]", "[{n}]")),
                cite("Revenue increased", second.replace("[1]", "[{n}]")),
            ],
            verify=[UNSUPPORTED, VERIFY_OK],
        )
    )
    ans = agent.run(QUESTION, [index], on_token=events.append)
    assert sum(1 for e in events if e.get("reset")) == 2
    assert _generations(events) == [first, second]
    assert _generations(events)[-1] == ans.raw_response == ans.answer


def test_on_token_errors_are_swallowed(index):  # noqa: F811
    def boom(event):
        raise RuntimeError("UI crashed")

    agent, _ = adaptive(
        _scripted(
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."), verify=VERIFY_OK
        )
    )
    ans = agent.run(QUESTION, [index], on_token=boom)
    assert ans.error is None and ans.grounded is True


def test_broken_stream_falls_back_to_a_plain_call(index):  # noqa: F811
    from local_agentic_rag.llm.base import LLMError

    agent, llm = adaptive(
        _scripted(
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."), verify=VERIFY_OK
        )
    )

    def broken_stream(*args, **kwargs):
        yield "Revenue was"
        raise LLMError("Ollama connection lost mid-stream")

    llm.chat_stream = broken_stream
    events: list[dict] = []
    ans = agent.run(QUESTION, [index], on_token=events.append)
    # the partial text is discarded and the full answer comes from the non-streaming call
    assert _generations(events)[-1] == ans.raw_response == "Revenue was 4.2 billion [1]."
    assert ans.error is None and ans.grounded is True and ans.cited_pages == [3]


def test_stream_timeout_is_not_retried_with_a_plain_call(index):  # noqa: F811
    from local_agentic_rag.llm.base import LLMTimeoutError

    agent, llm = adaptive(
        _scripted(
            generate=cite("Revenue increased", "Revenue was 4.2 billion [{n}]."), verify=VERIFY_OK
        )
    )

    def timed_out_stream(*args, **kwargs):
        yield "Revenue was"
        raise LLMTimeoutError("Ollama timed out after 600s")

    llm.chat_stream = timed_out_stream
    ans = agent.run(QUESTION, [index], on_token=lambda e: None)
    # chat() never retries a read timeout, so a second full wait would only double the delay
    assert "generate" not in calls_by_role(llm)
    assert ans.error and "timed out" in ans.error and ans.refused is False
