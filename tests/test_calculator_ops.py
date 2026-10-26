"""Structured calculator operations and the 'result must appear in the answer' guarantee."""

from __future__ import annotations

import json

import pytest

from local_agentic_rag.agent import calculator as C
from local_agentic_rag.schema import ANSWER_NOT_FOUND
from tests.helpers import scripted
from tests.test_agent import VERIFY_OK, cite, index, make_agent, steps  # noqa: F401
from tests.test_calculator import revenue_index  # noqa: F401


def v(*nums, passage=1):
    return [{"label": f"v{i}", "value": n, "passage": passage} for i, n in enumerate(nums)]


@pytest.mark.parametrize(
    "op, values, expected",
    [
        ("difference", v(30485.7, 30042.1), "443.6"),  # bea q09
        ("difference", v(-0.6, 3.8), "4.4"),  # bea q10 swing
        ("change", v(64.8, 30.8), "-34"),  # signed, second minus first
        ("product", v(16, 16), "256"),  # Marathi Hazari Karanje: 16 petals x 16 jets
        ("sum", v(1.5, 2.5, 3), "7"),
        ("ratio", v(3, 4), "0.75"),
        ("percent_change", v(200, 250), "25"),
    ],
)
def test_operations(op, values, expected):
    calc = C.run({"needed": True, "operation": op, "values": values})
    assert calc["result"] == expected


def test_months_between_dates():
    calc = C.run(
        {
            "needed": True,
            "operation": "months_between",
            "dates": [
                {"label": "AI RMF 1.0", "year": 2023, "month": 1, "passage": 2},
                {"year": 2024, "month": 7, "passage": 1},
            ],
        }
    )
    assert calc["result"] == "18" and calc["unit"] == "months"


@pytest.mark.parametrize(
    "data",
    [
        {"needed": True, "operation": "difference", "values": v(1, 2, 3)},
        {"needed": True, "operation": "ratio", "values": v(1, 0)},
        {
            "needed": True,
            "operation": "months_between",
            "dates": [{"year": 2023, "month": 13}, {"year": 2024, "month": 1}],
        },
        {"needed": True, "operation": "count", "items": []},
        {"needed": True, "operation": "product", "values": []},
    ],
)
def test_unusable_operations_return_none(data):
    assert C.run(data) is None


def test_string_numbers_with_commas_are_accepted():
    calc = C.run(
        {
            "needed": True,
            "operation": "difference",
            "values": [{"value": "30,485.7"}, {"value": "30,042.1"}],
        }
    )
    assert calc["result"] == "443.6"


@pytest.mark.parametrize(
    "answer, present",
    [
        ("It rose by $443.6 billion", True),
        ("It rose by 1443.6", False),
        ("Total: 30,485.7", True),
        ("4.41 points", False),
    ],
)
def test_result_in_answer(answer, present):
    target = "30485.7" if "30" in answer else ("4.4" if "4.41" in answer else "443.6")
    assert C.result_in_answer(answer, {"result": target}) is present


CALC_REPLY = json.dumps(
    {"needed": True, "operation": "difference", "values": v(4.2, 3.5), "unit": "billion dollars"}
)


def test_ignored_calculation_is_appended_with_citation(revenue_index):  # noqa: F811
    agent, _ = make_agent(
        scripted(
            calculate=CALC_REPLY,
            generate=cite("Revenue increased", "Revenue grew by about 1 billion dollars [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run("By how much did revenue increase from 2022 to 2023?", [revenue_index])
    assert "(Calculated: |4.2 - 3.5| = 0.7 billion dollars. [1])" in ans.answer
    assert "calc_appended" in steps(ans)
    assert 1 in [c.marker for c in ans.citations]


def test_no_append_when_answer_states_result_or_refuses(revenue_index):  # noqa: F811
    agent, _ = make_agent(
        scripted(
            calculate=CALC_REPLY,
            generate=cite("Revenue increased", "It rose 0.7 billion [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    assert (
        "Calculated" not in agent.run("By how much did revenue increase?", [revenue_index]).answer
    )
    agent2, _ = make_agent(
        scripted(
            calculate=CALC_REPLY,
            generate=ANSWER_NOT_FOUND,
            plan={"standalone_question": "q", "search_queries": []},
            grade={"relevant": [], "sufficient": False, "missing_query": ""},
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    refused = agent2.run("By how much did revenue increase?", [revenue_index])
    assert refused.refused and "Calculated" not in refused.answer
