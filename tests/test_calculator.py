"""Calculator tool: safe evaluation, question detection, result formatting, agent wiring."""

from __future__ import annotations

import json

import pytest

from local_agentic_rag.agent import calculator as C
from tests.helpers import scripted
from tests.test_agent import VERIFY_OK, cite, index, make_agent, steps, user_prompts  # noqa: F401


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("30485.7 - 30042.1", 443.6),
        ("3.8 - (-0.6)", 4.4),
        ("0.71 - 0.41", 0.30),
        ("(64.8 - 30.8)", 34.0),
        ("1,234 + 66", 1300),
        ("12.5% * 2", 25.0),
        ("10 / 4", 2.5),
        ("-3 + +5", 2),
    ],
)
def test_safe_eval_arithmetic(expr, expected):
    assert C.safe_eval(expr) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('x')",
        "abs(-1)",
        "2 ** 10",
        "a + 1",
        "1 / 0",
        "[1, 2]",
        "'1' + '2'",
        "True + 1",
    ],
)
def test_safe_eval_rejects_non_arithmetic(expr):
    with pytest.raises(C.CalculatorError):
        C.safe_eval(expr)


@pytest.mark.parametrize(
    "question",
    [
        "On NQ, how much higher was the Recall@10 of the MS MARCO fine-tuned Contriever than the unsupervised one?",
        "By how much did current-dollar GDP increase from Q1 2025 to Q2 2025, in billions?",
        "How many percentage points did the annualized real GDP growth rate swing between Q1 and Q2 2025?",
        "How many suggested actions are listed under MEASURE 2.7?",
        "How many months passed between the release of the AI RMF 1.0 and the profile?",
        "What is the difference between the two accuracies?",
    ],
)
def test_quantitative_questions_detected(question):
    assert C.is_quantitative(question)


@pytest.mark.parametrize(
    "question",
    [
        "Which existing dataset did the authors repurpose?",
        "What do the authors mean by brevity bias?",
        "What was the personal saving rate in Q2 2025?",
    ],
)
def test_lookup_questions_not_quantitative(question):
    assert not C.is_quantitative(question)


def test_run_expression_and_formatting():
    calc = C.run(
        {
            "needed": True,
            "values": [
                {"label": "Q2", "value": 30485.7, "passage": 2},
                {"label": "Q1", "value": 30042.1, "passage": 2},
            ],
            "expression": "30485.7 - 30042.1",
            "unit": "billion dollars",
        }
    )
    assert calc["result"] == "443.6"
    hint = C.hint(calc)
    assert "30485.7 - 30042.1 = 443.6 billion dollars" in hint and "[2]" in hint


def test_run_count_deduplicates_items():
    calc = C.run(
        {"needed": True, "expression": "count", "items": ["MS-2.7-001", "MS-2.7-002", "MS-2.7-001"]}
    )
    assert calc["result"] == "2"
    assert "counted 2 distinct items" in C.hint(calc)


@pytest.mark.parametrize(
    "data",
    [
        None,
        {"needed": False},
        {"needed": True, "expression": "x + 1"},
        {"needed": True, "expression": "count"},
    ],
)
def test_run_unusable_returns_none(data):
    assert C.run(data) is None


def test_format_number_precision():
    assert C.format_number(4.4000000001, [3.8, -0.6]) == "4.4"
    assert C.format_number(0.3, [0.71, 0.41]) == "0.30"
    assert C.format_number(18.0) == "18"


def test_agent_uses_calculator_hint_for_quantitative_question(index):  # noqa: F811
    question = "By how much did revenue increase from fiscal 2022 to fiscal 2023?"
    calc_reply = json.dumps(
        {
            "needed": True,
            "values": [
                {"label": "2023", "value": 4.2, "passage": 1},
                {"label": "2022", "value": 3.5, "passage": 1},
            ],
            "expression": "4.2 - 3.5",
            "unit": "billion dollars",
        }
    )
    agent, llm = make_agent(
        scripted(
            calculate=calc_reply,
            generate=cite("Revenue increased", "Revenue rose by 0.7 billion dollars [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run(question, [index])
    assert steps(ans) == ["retrieve", "calculate", "generate", "verify"]
    assert "4.2 - 3.5 = 0.7 billion dollars" in user_prompts(llm, "generate")[0]


def test_calculator_disabled_or_non_quantitative_makes_no_call(index):  # noqa: F811
    agent, llm = make_agent(
        scripted(generate=cite("Revenue increased", "4.2 billion [{n}]."), verify=VERIFY_OK),
        agent_mode="adaptive",
        use_calculator=False,
    )
    agent.run("By how much did revenue increase?", [index])
    assert all(c["json_schema"] is not C.CALC_SCHEMA for c in llm.calls)
    agent2, llm2 = make_agent(
        scripted(generate=cite("Revenue increased", "4.2 billion [{n}]."), verify=VERIFY_OK),
        agent_mode="adaptive",
    )
    agent2.run("What was revenue in fiscal 2023?", [index])
    assert all(c["json_schema"] is not C.CALC_SCHEMA for c in llm2.calls)
