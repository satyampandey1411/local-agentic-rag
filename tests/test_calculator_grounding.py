"""Calculator safety: grounded operands, no override of stated values, operation choice, ID counts.

Each test reproduces a failure from an early Colab run (the passages are trimmed copies of
the real chunks: ``ingest/tables.py`` rows and Markdown grids).
"""

from __future__ import annotations

import json
import re

import pytest

from local_agentic_rag.agent import calculator as C
from tests.helpers import FakeEmbedder, make_index, scripted
from tests.test_agent import VERIFY_OK, cite, make_agent, steps, user_prompts

# --------------------------------------------------------------------------- passages

BEA_P1 = (
    "Real gross domestic product (GDP) increased at an annual rate of 3.8 percent in the second "
    "quarter of 2025 (April, May, and June), according to the third estimate released by the U.S. "
    "Bureau of Economic Analysis. In the first quarter, real GDP decreased 0.6 percent (revised)."
)
BEA_GDP_ROW = (
    "September 25, 2025 Table 1. Real Gross Domestic Product and Related Measures\n"
    "Percent change at annual rate | Gross domestic product | 2024 = 2.8; "
    "Seasonally adjusted at annual rates 2024 Q4 = 1.9; "
    "Seasonally adjusted at annual rates 2025 Q1 = -0.6; "
    "Seasonally adjusted at annual rates 2025 Q2 = 3.8"
)
BEA_NONFARM_ROW = (
    "September 25, 2025 Table 1. Real Gross Domestic Product and Related Measures\n"
    "Percentage points at annual rates | Nonfarm | 2024 = 0.02; "
    "Seasonally adjusted at annual rates 2024 Q4 = -0.92; "
    "Seasonally adjusted at annual rates 2025 Q1 = 2.61; "
    "Seasonally adjusted at annual rates 2025 Q2 = -3.52"
)
BEA_PROFITS = (
    "Profits from current production (corporate profits with inventory valuation and capital "
    "consumption adjustments) increased $6.8 billion in the second quarter, a downward revision "
    "of $58.7 billion."
)
BEA_MANUFACTURING_ROW = (
    "September 25, 2025 Table 10. Corporate Profits by Industry [Billions of dollars]\n"
    "Manufacturing | Level Seasonally adjusted at annual rates 2025 Q1 = 591.1; "
    "Level Seasonally adjusted at annual rates 2025 Q2 = 648.6; "
    "Change from preceding period 2025 Q1 = -136.6; Change from preceding period 2025 Q2 = 57.4"
)
ARXIV_TABLE5 = (
    "|  | Poison | Foil | No | Evidence |\n"
    "|---|---|---|---|---|\n"
    "| MODEL | Doc | Doc | Doc | Doc |\n"
    "| gpt-4o-mini | 32.0% | 44.0% | 52.0% | 88.0% |\n"
    "| gpt-4o | 30.8% | 62.8% | 64.8% | 93.6% |"
)
ARXIV_PROSE = (
    "D2 := 4 × S Table 5 reports the RAG accuracy, showing that the evidence document helps."
)
ARXIV_TABLE2 = (
    "| Model | pooling | nDCG@10 | Recall@10 |\n"
    "|---|---|---|---|\n"
    "| Dragon RoBERTa Dragon+ COCO-DR Base MSMARCO | cls cls cls | 0.55 0.54 0.50 | 0.75 0.74 0.71 |\n"
    "| Contriever MSMARCO RetroMAE MSMARCO FT | avg cls | 0.50 0.48 | 0.71 0.68 |\n"
    "| Contriever | avg | 0.25 | 0.41 |"
)
ARXIV_TABLE_A3 = (  # a different dataset (refined Re-DocRED), not NQ
    "| Model | Pooling | nDCG@10 | Recall@10 |\n"
    "|---|---|---|---|\n"
    "| Dragon+ Dragon RoBERTa | cls cls | 0.55 0.53 | 0.63 0.59 |\n"
    "| Contriever MSMARCO | avg | 0.52 | 0.59 |"
)
MARATHI = (
    "हजारी कारंजे: कमळाच्या आकाराचे हे कारंजे होते. त्यात सोळा पाकळ्या असून प्रत्येक पाकळीत सोळा "
    "याप्रमाणे सर्व पाकळ्यांत मिळून दोनशे छप्पन्न कारंजी होती."
)


def vals(*items):
    return [{"label": label, "value": value, "passage": p} for label, value, p in items]


def extraction(op, *items, unit=""):
    return {"needed": True, "operation": op, "values": vals(*items), "unit": unit}


# --------------------------------------------------------------------------- failure mode 1: grounding


def test_wrong_table_row_is_rejected_with_the_row_named():
    # bea q10: 2.61 / -3.52 are the 2025 Q1/Q2 cells of the "Nonfarm" row, not GDP.
    out = C.evaluate(
        extraction(
            "difference",
            ("Seasonally adjusted at annual rates 2025 Q1", 2.61, 1),
            ("Seasonally adjusted at annual rates 2025 Q2", -3.52, 1),
        ),
        [BEA_NONFARM_ROW],
    )
    assert "rejected" in out and len(out["rejected"]) == 2
    assert '"Nonfarm"' in out["rejected"][0]


def test_right_row_and_columns_are_accepted():
    out = C.evaluate(
        extraction("difference", ("Real GDP 2025 Q1", -0.6, 1), ("Real GDP 2025 Q2", 3.8, 1)),
        [BEA_GDP_ROW],
    )
    assert out["result"] == "4.4" and out["verified"] is True


def test_wrong_period_column_is_rejected():
    out = C.evaluate(
        extraction("difference", ("Real GDP 2025 Q2", -0.6, 1), ("Real GDP 2025 Q2", 3.8, 1)),
        [BEA_GDP_ROW],
    )
    assert "rejected" in out
    assert 'column "Seasonally adjusted at annual rates 2025 Q1"' in out["rejected"][0]


def test_prose_value_with_sign_word_is_grounded():
    # "decreased 0.6 percent" is the operand -0.6.
    out = C.evaluate(
        extraction("difference", ("real GDP Q1 2025", -0.6, 1), ("real GDP Q2 2025", 3.8, 1)),
        [BEA_P1],
    )
    assert out["result"] == "4.4"


def test_markdown_column_and_name_digits_are_checked():
    # arxiv q10: "4" came from "gpt-4o"/"4 ×", 93.6 is the "Evidence Doc" column, not "No Doc".
    out = C.evaluate(
        extraction(
            "difference",
            ("Table A.9 | gpt-4o | accuracy", 4, 1),
            ("Table 4 | gpt-4o | no document", 93.6, 2),
        ),
        [ARXIV_PROSE, ARXIV_TABLE5],
    )
    reasons = " ".join(out["rejected"])
    assert "part of a name" in reasons
    assert 'column "Evidence Doc"' in reasons and '"No Doc"' in reasons

    good = C.evaluate(
        extraction("difference", ("gpt-4o | No Doc", 64.8, 2), ("gpt-4o | Poison Doc", 30.8, 2)),
        [ARXIV_PROSE, ARXIV_TABLE5],
    )
    assert good["result"] == "34"


def test_label_without_row_is_rejected_and_stacked_cells_are_matched_by_row():
    # arxiv q11: label "Recall@10" names no model; 0.59 is on another table's rows.
    out = C.evaluate(
        extraction("difference", ("Recall@10", 0.59, 1), ("Recall@10", 0.5, 2)),
        [ARXIV_TABLE_A3, ARXIV_TABLE2],
    )
    assert "rejected" in out
    good = C.evaluate(
        extraction(
            "difference",
            ("Contriever MSMARCO | Recall@10", 0.71, 2),
            ("Contriever (unsupervised) | Recall@10", 0.41, 2),
        ),
        [ARXIV_TABLE_A3, ARXIV_TABLE2],
    )
    assert good["result"] == "0.30"


def test_value_absent_from_passages_is_rejected_and_citation_slips_are_fixed():
    out = C.evaluate(
        extraction("difference", ("real GDP Q2", 3.8, 2), ("real GDP Q1", 9.9, 1)),
        [BEA_P1, "Other text."],
    )
    assert out["rejected"] == ['9.9 ("real GDP Q1") does not appear in passage 1']
    fixed = C.evaluate(  # both values cite passage 1 but are in passage 2
        extraction("difference", ("real GDP Q2", 3.8, 1), ("real GDP Q1", -0.6, 1)),
        ["Other text.", BEA_P1],
    )
    assert fixed["result"] == "4.4" and {v["passage"] for v in fixed["values"]} == {2}


def test_free_form_expression_may_only_use_listed_operands():
    data = {
        "needed": True,
        "values": vals(("real GDP Q2", 3.8, 1)),
        "expression": "3.8 - 1.2",
    }
    assert "rejected" in C.evaluate(data, [BEA_P1])


def test_number_words_in_non_latin_passage_are_unverified_not_rejected():
    out = C.evaluate(
        extraction("product", ("petals", 16, 1), ("per petal: jets", 16, 1)), [MARATHI]
    )
    assert out["result"] == "256" and out["verified"] is False


def test_not_needed_returns_none():
    assert C.evaluate({"needed": False}, [BEA_P1]) is None


# --------------------------------------------------------------------------- failure mode 2: stated values


@pytest.mark.parametrize(
    "question, forced",
    [
        (
            "How many percentage points did the annualized real GDP growth rate swing from Q1 2025 to Q2 2025?",
            True,
        ),
        (
            "For gpt-4o, how many percentage points lower was accuracy with the poisoned document than with no document?",
            True,
        ),
        ("How many suggested actions are listed under MEASURE 2.7?", True),
        ("How many months passed between the release of the AI RMF 1.0 and the profile?", True),
        ("How many fountain jets did the Hazari Karanje fountain have in total?", True),
        (
            "Profits from current production rose by how much in Q2 2025, and what level did they reach?",
            False,
        ),
        (
            "How many percentage points did retail trade contribute to the change in real GDP in Q2 2025?",
            False,
        ),
        (
            "Did real value added grow faster for goods or for services industries, and by how much each?",
            False,
        ),
    ],
)
def test_only_arithmetic_wording_forces_the_result(question, forced):
    assert C.is_quantitative(question)
    assert C.needs_arithmetic(question) is forced


def test_stated_change_is_found_but_levels_are_not():
    q13 = "Profits from current production rose by how much in Q2 2025, and what level did they reach?"
    assert "increased $6.8 billion" in C.stated_change(q13, [BEA_MANUFACTURING_ROW, BEA_PROFITS])
    assert C.stated_change(q13, [BEA_MANUFACTURING_ROW]) == ""
    level = "Profits from current production increased to $3,929.7 billion in the quarter."
    moved = "Profits from current production rose from 3,922.9 to 3,929.7 billion."
    assert C.stated_change(q13, [level, moved]) == ""


# --------------------------------------------------------------------------- failure mode 3: operation, counts


def test_per_group_sum_becomes_product():
    calc = C.run(extraction("sum", ("petals", 16, 1), ("per petal: jets", 16, 1)))
    assert calc["result"] == "256" and calc["expression"] == "16 x 16"
    calc = C.run(extraction("sum", ("jets in each petal", 16, 1), ("petals", 16, 1)))
    assert calc["result"] == "256"
    # Two per-unit values (or none) are a genuine sum.
    assert (
        C.run(
            extraction("sum", ("Q1 dividend per share", 0.5, 1), ("Q2 dividend per share", 0.5, 1))
        )["result"]
        == "1"
    )
    assert C.run(extraction("sum", ("north", 12, 1), ("south", 8, 1)))["result"] == "20"


def test_id_prefix_needs_the_number_named_in_the_question():
    texts = ["| MS-2.7-001 | Apply security measures |", "| MS-2.6-001 | Other |"]
    assert (
        C.id_prefix("How many suggested actions are listed under MEASURE 2.7?", texts) == "MS-2.7"
    )
    assert C.id_prefix("How many suggested actions are listed under MEASURE 3.1?", texts) == ""
    assert C.id_prefix("How many percentage points did MEASURE 2.7 add?", texts) == ""


def test_count_ids_deduplicates_and_orders_ids():
    calc = C.count_ids("MS-2.7", ["MS-2.7-002 and MS-2.7-001", "MS-2.7-002 again; MS-2.70-001 no"])
    assert calc["result"] == "2" and calc["items"] == ["MS-2.7-001", "MS-2.7-002"]
    assert calc["values"] == [{"passage": 1}]


# --------------------------------------------------------------------------- agent wiring

Q10 = "How many percentage points did the annualized real GDP growth rate swing from Q1 2025 to Q2 2025?"


@pytest.fixture
def bea_index():
    return make_index(
        [(1, BEA_P1), (9, BEA_NONFARM_ROW), (3, BEA_PROFITS), (18, BEA_MANUFACTURING_ROW)],
        FakeEmbedder(),
        doc_id="bea",
        title="GDP release",
    )


def passage_of(messages, needle: str) -> int:
    """Number of the (possibly multi-line) prompt passage containing ``needle``."""
    blocks = re.split(r"^\[(\d+)\] \(page ", messages[-1]["content"], flags=re.MULTILINE)
    for number, body in zip(blocks[1::2], blocks[2::2], strict=False):
        if needle in body.split("\n\nQuestion:")[0]:
            return int(number)
    raise AssertionError(f"{needle!r} not among passages")


def calc_reply(op, *items):
    """Scripted calculator reply whose passage numbers follow the prompt's passage order."""

    def reply(messages):
        cited = [(label, value, passage_of(messages, needle)) for label, value, needle in items]
        return json.dumps(extraction(op, *cited))

    return reply


def test_rejected_extraction_is_retried_with_feedback(bea_index):
    agent, llm = make_agent(
        scripted(
            calculate=[
                calc_reply(
                    "difference",
                    ("Seasonally adjusted at annual rates 2025 Q1", 2.61, "Nonfarm"),
                    ("Seasonally adjusted at annual rates 2025 Q2", -3.52, "Nonfarm"),
                ),
                calc_reply(
                    "difference",
                    ("real GDP 2025 Q1", -0.6, "Real gross domestic product"),
                    ("real GDP 2025 Q2", 3.8, "Real gross domestic product"),
                ),
            ],
            generate=cite("Real gross domestic product", "It swung 2.7 points [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run(Q10, [bea_index])
    assert steps(ans)[:3] == ["retrieve", "calculate", "calculate"]
    retry_prompt = user_prompts(llm, "calculate")[1]
    assert (
        "previous answer to this task was rejected" in retry_prompt and '"Nonfarm"' in retry_prompt
    )
    assert "|-0.6 - 3.8| = 4.4" in user_prompts(llm, "generate")[0]
    assert re.search(r"\(Calculated: \|-0\.6 - 3\.8\| = 4\.4\. \[\d\]\)$", ans.answer)


def test_twice_ungrounded_calculation_is_dropped(bea_index):
    bad = calc_reply("difference", ("2025 Q1", 2.61, "Nonfarm"), ("2025 Q2", -3.52, "Nonfarm"))
    agent, llm = make_agent(
        scripted(
            calculate=bad,
            generate=cite("Real gross domestic product", "It swung 4.4 points [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run(Q10, [bea_index])
    assert steps(ans).count("calculate") == 2
    assert "calculator" not in user_prompts(llm, "generate")[0]
    assert re.fullmatch(r"It swung 4\.4 points \[\d\]\.", ans.answer)
    assert "calc_appended" not in steps(ans)


def test_lookup_with_stated_change_skips_the_calculator(bea_index):
    # bea q13: the calculator computed -136.6 - 648.6 from the Manufacturing row.
    question = "Profits from current production rose by how much in Q2 2025, and what level did they reach?"
    agent, llm = make_agent(
        scripted(
            generate=cite("Profits from current production", "They rose $6.8 billion [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run(question, [bea_index])
    assert "calc_skipped" in steps(ans) and "calculate" not in steps(ans)
    assert re.fullmatch(r"They rose \$6\.8 billion \[\d\]\.", ans.answer)


def test_lookup_result_is_a_soft_hint_and_never_appended():
    # bea q11: a contribution is stated in a table; an unrequested difference was appended.
    row = (
        "Table 13. Contributions to Percent Change in Real GDP by Industry\n"
        "Percentage points at annual rates | Retail trade | "
        "Seasonally adjusted at annual rates 2025 Q1 = -0.17; "
        "Seasonally adjusted at annual rates 2025 Q2 = -0.54"
    )
    index = make_index([(21, row)], FakeEmbedder(), doc_id="bea", title="GDP release")
    question = "How many percentage points did retail trade contribute to the change in real GDP in Q2 2025?"
    agent, llm = make_agent(
        scripted(
            calculate=calc_reply(
                "change",
                ("Retail trade 2025 Q1", -0.17, "Retail trade"),
                ("Retail trade 2025 Q2", -0.54, "Retail trade"),
            ),
            generate=cite("Table 13", "Retail trade contributed -0.54 percentage point [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run(question, [index])
    prompt = user_prompts(llm, "generate")[0]
    assert "-0.54 - -0.17 = -0.37" in prompt and "only if the question asks for exactly" in prompt
    assert ans.answer == "Retail trade contributed -0.54 percentage point [1]."


def test_unverified_product_is_hinted_not_forced():
    # Marathi q05: 16 petals x 16 jets each, numbers written as words.
    index = make_index(
        [(2, MARATHI + " Hazari Karanje fountain jets")], FakeEmbedder(), doc_id="mr"
    )
    question = "How many fountain jets did the Hazari Karanje fountain have in total?"
    agent, llm = make_agent(
        scripted(
            calculate=calc_reply("sum", ("petals", 16, "हजारी"), ("per petal: jets", 16, "हजारी")),
            generate=cite("हजारी", "It had 256 jets [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
        translate_queries=False,
    )
    ans = agent.run(question, [index])
    prompt = user_prompts(llm, "generate")[0]
    assert "16 x 16 = 256" in prompt and "only if the question asks for exactly" in prompt
    assert ans.answer == "It had 256 jets [1]."


def test_id_sequence_split_over_pages_is_counted_deterministically():
    # nist q10: MS-2.7-001..008 on p37, MS-2.7-009 on p38 (not retrieved); the answer said 8.
    p37 = "MEASURE 2.7: AI system security and resilience suggested actions. " + " ".join(
        f"| MS-2.7-00{i} | action {i} |" for i in range(1, 9)
    )
    p38 = "| MS-2.7-009 | Regularly assess that controls remain effective. |"
    other = "| MS-2.6-001 | A different subcategory. |"
    index = make_index([(37, p37), (38, p38), (36, other)], FakeEmbedder(), doc_id="nist")
    question = "How many suggested actions are listed under MEASURE 2.7 (security and resilience)?"
    agent, llm = make_agent(
        scripted(  # no "calculate" reply: the count must not need the LLM
            generate=cite("MEASURE 2.7", "There are 8 suggested actions [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
        top_k=1,
    )
    ans = agent.run(question, [index])
    assert "complete_ids" in steps(ans)
    assert [p.chunk.page for p in ans.retrieved] == [37, 38]
    assert "counted 9 distinct items" in user_prompts(llm, "generate")[0]
    assert ans.answer.endswith("(Count: 9. [1][2])")
    assert 38 in ans.cited_pages


def test_empty_calculation_for_an_arithmetic_question_is_retried_once(bea_index):
    # Kaggle trace, NIST q11: the first call returned no calculation and the generator refused.
    agent, llm = make_agent(
        scripted(
            calculate=[
                json.dumps({"needed": False}),
                calc_reply(
                    "difference",
                    ("real GDP 2025 Q1", -0.6, "Real gross domestic product"),
                    ("real GDP 2025 Q2", 3.8, "Real gross domestic product"),
                ),
            ],
            generate=cite("Real gross domestic product", "It swung 4.4 points [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run(Q10, [bea_index])
    assert steps(ans)[:3] == ["retrieve", "calculate", "calculate"]
    assert "gave no calculation" in user_prompts(llm, "calculate")[1]
    assert "|-0.6 - 3.8| = 4.4" in user_prompts(llm, "generate")[0]


def test_lookup_question_with_no_calculation_is_not_retried(bea_index):
    question = "What was real GDP growth in Q2 2025?"  # no arithmetic wording
    agent, llm = make_agent(
        scripted(
            calculate=json.dumps({"needed": False}),
            generate=cite("Real gross domestic product", "Real GDP grew 3.8 percent [{n}]."),
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
    )
    ans = agent.run(question, [bea_index])
    assert steps(ans).count("calculate") <= 1
