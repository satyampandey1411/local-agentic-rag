"""Numerical-question fixes behind flags (all off by default): cell alignment, measure check
with escalation, and date candidates for time spans.

Passages are trimmed copies of the real chunks from a Kaggle run (English dev set).
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from local_agentic_rag.agent import calculator as C
from local_agentic_rag.config import PRESETS, RAGConfig
from local_agentic_rag.retrieval.hybrid import HybridRetriever, measure_order
from local_agentic_rag.retrieval.index import DocumentIndex
from tests.helpers import FakeEmbedder, TableReranker, make_index, scripted
from tests.test_agent import VERIFY_OK, cite, make_agent, steps, user_prompts
from tests.test_calculator_grounding import calc_reply, extraction

ARXIV_TABLE5 = (
    "Table 5: RAG accuracy when using different document versions as references.\n"
    "|  | Poison | Foil | No | Evidence |\n"
    "|---|---|---|---|---|\n"
    "| MODEL | Doc | Doc | Doc | Doc |\n"
    "| gpt-4o-mini | 32.0% | 44.0% | 52.0% | 88.0% |\n"
    "| gpt-4o | 30.8% | 62.8% | 64.8% | 93.6% |"
)
ARXIV_TABLE2 = (
    "Table 2: Models' performance on NQ dataset with test set queries.\n"
    "| Model | pooling | nDCG@10 | Recall@10 |\n"
    "|---|---|---|---|\n"
    "| Dragon RoBERTa | cls | 0.55 | 0.75 |\n"
    "| COCO-DR Base MSMARCO | cls | 0.50 | 0.71 |\n"
    "| Contriever MSMARCO | avg | 0.50 | 0.71 |\n"
    "| Contriever | avg | 0.25 | 0.41 |"
)
STACKED = (  # merged columns: several numbers per cell, positions cannot be trusted
    "| Model | nDCG@10 | Recall@10 |\n"
    "|---|---|---|\n"
    "| Contriever MSMARCO Contriever | 0.50 0.25 | 0.71 0.41 |"
)
BEA_T6 = (
    "September 25, 2025 Table 6. Real Gross Domestic Product: Percent Change from Quarter One "
    "Year Ago\nGross domestic product (GDP) | 2024 Q4 = 2.4; 2025 Q1 = 2.0; 2025 Q2 = 2.1"
)
BEA_T1 = (
    "September 25, 2025 Table 1. Real Gross Domestic Product and Related Measures: Percent "
    "Change from Preceding Period\nCurrent-dollar measures | GDP | "
    "Seasonally adjusted at annual rates 2025 Q1 = 2.9; Seasonally adjusted at annual rates 2025 Q2 = 6.0"
)
BEA_T3 = (
    "September 25, 2025 Table 3. Gross Domestic Product: Level and Change from Preceding Period\n"
    "Gross domestic product (GDP) | Billions of dollars Seasonally adjusted at annual rates 2025 Q1 = "
    "30,042.1; Billions of dollars Seasonally adjusted at annual rates 2025 Q2 = 30,485.7"
)
BEA_T13 = (
    "September 25, 2025 Table 13. Contributions to Percent Change in Real Gross Domestic Product "
    "by Industry Group\nPercentage points at annual rates | Mining | "
    "Seasonally adjusted at annual rates 2025 Q1 = -0.27; Seasonally adjusted at annual rates 2025 Q2 = 0.33"
)
BEA_T12 = (
    "September 25, 2025 Table 12. Real Gross Domestic Product by Industry Group: Percent Change "
    "from Preceding Period\nMining | Seasonally adjusted at annual rates 2025 Q1 = -18.3; "
    "Seasonally adjusted at annual rates 2025 Q2 = 29.0"
)
BEA_P1 = (
    "Real gross domestic product (GDP) increased at an annual rate of 3.8 percent in the second "
    "quarter of 2025 (April, May, and June). In the first quarter, real GDP decreased 0.6 percent (revised)."
)
Q_SWING = "How many percentage points did the annualized real GDP growth rate swing from Q1 2025 to Q2 2025?"
Q_DOLLARS = "By how much did current-dollar GDP increase from Q1 2025 to Q2 2025, in billions of dollars at annual rates?"


FLAGS = (
    "calc_cell_alignment",
    "calc_measure_check",
    "calc_date_candidates",
    "measure_aware_retrieval",
)


def test_default_turns_on_the_measured_fixes_only():
    # Kaggle runs 5-6: alignment + dates + measure-aware retrieval are the default; the
    # measure check (latency cost, no extra gain) stays opt-in.
    cfg = RAGConfig()
    assert cfg.calc_cell_alignment and cfg.calc_date_candidates and cfg.measure_aware_retrieval
    assert not cfg.calc_measure_check
    assert RAGConfig().with_overrides(**PRESETS["agentic_numeric_lite"]) == cfg
    # the old default and the starting design reproduce earlier results: no fix on
    for name in ("agentic_pre_numeric", "baseline"):
        old = RAGConfig().with_overrides(**PRESETS[name])
        assert not any(getattr(old, f) for f in FLAGS), name
    # each single-flag ablation preset turns on exactly its own flag
    single = {
        "agentic_align": "calc_cell_alignment",
        "agentic_measure": "calc_measure_check",
        "agentic_dates": "calc_date_candidates",
        "agentic_measure_retrieval": "measure_aware_retrieval",
    }
    for name, flag in single.items():
        c = RAGConfig().with_overrides(**PRESETS[name])
        assert [f for f in FLAGS if getattr(c, f)] == [flag], name
    assert all(PRESETS["agentic_numeric"][f] for f in FLAGS)
    # agent-only flags: the ingestion fingerprint (chunk text, embeddings) is unchanged
    numeric = RAGConfig().with_overrides(**PRESETS["agentic_numeric"])
    assert numeric.ingestion_fingerprint() == RAGConfig().ingestion_fingerprint()
    old = RAGConfig().with_overrides(**PRESETS["agentic_pre_numeric"])
    assert old.ingestion_fingerprint() == RAGConfig().ingestion_fingerprint()


# --------------------------------------------------------------------------- cell alignment


def _gpt4o_extraction(no_doc_value):
    return extraction(
        "difference", ("gpt-4o | Poison Doc", 30.8, 1), ("gpt-4o | No Doc", no_doc_value, 1)
    )


def test_value_from_a_lookalike_row_is_realigned_to_the_labelled_cell():
    # arXiv q10 (Kaggle run): 52.0 is gpt-4o-mini's "No Doc"; the label says gpt-4o.
    assert C.evaluate(_gpt4o_extraction(52.0), [ARXIV_TABLE5])["result"] == "21.2"  # flag off
    out = C.evaluate(_gpt4o_extraction(52.0), [ARXIV_TABLE5], align=True)
    assert out["result"] == "34"
    assert out["values"][1]["value"] == 64.8 and out["values"][1]["aligned_from"] == 52.0
    assert out["aligned"] == ['52 -> 64.8 ("gpt-4o | No Doc")']


def test_correct_cells_and_the_mini_row_are_left_alone():
    out = C.evaluate(_gpt4o_extraction(64.8), [ARXIV_TABLE5], align=True)
    assert out["result"] == "34" and "aligned" not in out
    mini = extraction(
        "difference", ("gpt-4o-mini | Poison Doc", 32.0, 1), ("gpt-4o-mini | No Doc", 52.0, 1)
    )
    out = C.evaluate(mini, [ARXIV_TABLE5], align=True)
    assert out["result"] == "20" and "aligned" not in out


def test_wrong_metric_column_is_realigned_by_row_and_column():
    # RESULTS.md: the model once took nDCG@10 (0.50/0.25) for a Recall@10 question.
    data = extraction(
        "difference",
        ("MS MARCO fine-tuned Contriever | Recall@10", 0.50, 1),
        ("unsupervised Contriever | Recall@10", 0.25, 1),
    )
    out = C.evaluate(data, [ARXIV_TABLE2], align=True)
    assert out["result"] == "0.30"
    assert [v["value"] for v in out["values"]] == [0.71, 0.41]


@pytest.mark.parametrize(
    "label, value, text",
    [
        ("Contriever | Recall@10", 0.71, STACKED),  # merged cells: no alignment
        ("Recall@10", 0.71, ARXIV_TABLE2),  # names no row
        ("Dragon | Recall@10", 0.71, ARXIV_TABLE2 + "\n| Dragon+ | cls | 0.54 | 0.74 |"),  # tie
        ("gpt-4o | Doc", 52.0, ARXIV_TABLE5),  # "Doc" fits four columns equally
        ("gpt-4o | No Doc", 52.0, ARXIV_TABLE5 + "\nWith no document, accuracy was 52.0%."),
        ("Contriever | Recall@10", 0.99, ARXIV_TABLE2),  # not in a cell at all
    ],
)
def test_alignment_needs_an_unambiguous_cell(label, value, text):
    assert C.align_cell(value, label, text) is None


def test_row_record_tables_are_aligned_within_a_chunk():
    two_rows = BEA_T6 + "\nPersonal consumption expenditures (PCE) | 2025 Q1 = 3.1; 2025 Q2 = 2.7"
    assert C.align_cell(2.7, "Gross domestic product | 2025 Q2", two_rows)[0] == 2.1
    negative = (
        "Table 3\nNonfarm | 2025 Q1 = 207.8; 2025 Q2 = -56.0\nFarm | 2025 Q1 = 1.0; 2025 Q2 = 2.0"
    )
    assert C.align_cell(2.0, "Nonfarm | 2025 Q2", negative)[0] == -56.0


# --------------------------------------------------------------------------- measure check


def test_question_measure_classification():
    assert C.question_measure(Q_SWING) == {
        "dollars": False,
        "rate": True,
        "annualized": True,
        "nominal": False,
    }
    assert C.question_measure(Q_DOLLARS)["dollars"] is True
    assert C.question_measure(Q_DOLLARS)["nominal"] is True  # "current-dollar"
    assert C.question_measure(Q_DOLLARS)["rate"] is False
    assert not any(C.question_measure("How many queries per setting?").values())
    # both a dollar amount and a rate: left unclassified
    both = C.question_measure("What was the level of GDP in dollars and its growth rate?")
    assert not both["dollars"] and not both["rate"]
    # BEA q11: a contribution and a rate in one question -> no rate filter (only annualized)
    q11 = (
        "How many percentage points did retail trade contribute to the change in real GDP in "
        "Q2 2025, and at what annual rate did retail trade's real value added change that quarter?"
    )
    assert C.question_measure(q11) == {
        "dollars": False,
        "rate": False,
        "annualized": True,
        "nominal": False,
    }
    assert not C.table_conflicts(q11, BEA_T13)


def test_year_over_year_operands_are_rejected_for_an_annualized_question():
    # BEA q10 (Kaggle): |2.0 - 2.1| = 0.1 from Table 6, verified and accepted.
    data = extraction("difference", ("Real GDP | 2025 Q1", 2.0, 1), ("Real GDP | 2025 Q2", 2.1, 1))
    assert C.evaluate(data, [BEA_T6])["result"] == "0.1"  # flag off
    out = C.evaluate(data, [BEA_T6], question=Q_SWING)
    assert "rejected" in out and "from a year earlier" in out["rejected"][0]
    prose = extraction("difference", ("real GDP Q1", -0.6, 1), ("real GDP Q2", 3.8, 1))
    assert C.evaluate(prose, [BEA_P1], question=Q_SWING)["result"] == "4.4"


def test_percent_operands_are_rejected_for_a_dollar_question_and_levels_pass():
    pct = extraction("change", ("GDP | 2025 Q1", 2.9, 1), ("GDP | 2025 Q2", 6.0, 1))
    out = C.evaluate(pct, [BEA_T1], question=Q_DOLLARS)
    assert "rejected" in out and "not an amount in dollars" in out["rejected"][0]
    level = extraction("change", ("GDP | 2025 Q1", 30042.1, 1), ("GDP | 2025 Q2", 30485.7, 1))
    assert C.evaluate(level, [BEA_T3], question=Q_DOLLARS)["result"] == "443.6"


def test_chained_dollars_are_not_current_dollars():
    # RESULTS.md: for this question the model once answered 222.8, the chained-dollar change
    # from the same Table 3 "Addenda | GDP" row that holds the current-dollar levels.
    addenda = (
        "September 25, 2025 Table 3. Gross Domestic Product: Level and Change from Preceding "
        "Period--Table Ends\nAddenda | GDP | Billions of dollars Seasonally adjusted at annual "
        "rates 2025 Q1 = 30,042.1; Billions of dollars Seasonally adjusted at annual rates 2025 "
        "Q2 = 30,485.7; Billions of chained (2017) dollars Change from preceding period 2025 Q2 = 222.8"
    )
    assert "chained" in C.measure_conflict(Q_DOLLARS, 222.8, addenda)
    assert C.measure_conflict(Q_DOLLARS, 30485.7, addenda) == ""
    assert (
        C.measure_conflict(
            "By how much did real GDP increase, in billions of dollars?", 222.8, addenda
        )
        == ""
    )


def test_contribution_is_not_a_growth_rate():
    q = "At what annual rate did real value added in the mining industry grow in Q2 2025?"
    assert C.measure_conflict(q, 0.33, BEA_T13).startswith("it is a contribution")
    assert C.measure_conflict(q, 29.0, BEA_T12) == ""
    # unclassified questions never conflict
    assert C.measure_conflict("What did mining contribute in Q2 2025?", 0.33, BEA_T13) == ""


@pytest.fixture
def bea_index():
    return make_index([(14, BEA_T6), (1, BEA_P1)], FakeEmbedder(), doc_id="bea", title="GDP")


def _swing_agent(**cfg):
    bad = calc_reply(
        "difference", ("Real GDP | 2025 Q1", 2.0, "Table 6"), ("Real GDP | 2025 Q2", 2.1, "Table 6")
    )
    good = calc_reply(
        "difference", ("real GDP Q1", -0.6, "decreased 0.6"), ("real GDP Q2", 3.8, "decreased 0.6")
    )
    return make_agent(
        scripted(
            calculate=[bad, bad, good],
            plan={"standalone_question": Q_SWING, "search_queries": ["real GDP annual rate Q1 Q2"]},
            grade=lambda m: {"relevant": [1, 2], "sufficient": True, "missing_query": ""},
            generate=[
                cite("Table 6", "The rate swung 0.1 percentage points [{n}]."),
                cite("decreased 0.6", "Real GDP swung 4.4 percentage points [{n}]."),
            ],
            verify=VERIFY_OK,
        ),
        agent_mode="adaptive",
        **cfg,
    )


def test_rejected_calculation_escalates_with_the_measure_check(bea_index):
    agent, llm = _swing_agent(calc_measure_check=True)
    ans = agent.run(Q_SWING, [bea_index])
    escalations = [t.detail for t in ans.trace if t.step == "escalate"]
    assert escalations == [{"reason": "no grounded calculation"}]
    assert "|-0.6 - 3.8| = 4.4" in user_prompts(llm, "generate")[1]
    assert "4.4" in ans.answer


def test_without_the_flag_the_wrong_table_answer_stands(bea_index):
    agent, _ = _swing_agent()
    ans = agent.run(Q_SWING, [bea_index])
    assert "escalate" not in steps(ans) and "0.1" in ans.answer


# --------------------------------------------------------------------------- dates

NIST_P5 = (
    "This document is a cross-sectoral profile of and companion resource for the AI Risk "
    "Management Framework (AI RMF 1.0) for Generative AI. The AI RMF was released in January "
    "2023, and is intended for voluntary use."
)
NIST_P2 = (
    "This publication is available free of charge from: https://doi.org/10.6028/NIST.AI.600-1 "
    "July 2024 U.S. Department of Commerce"
)
Q_MONTHS = "How many months passed between the release of the AI RMF 1.0 and the publication of this profile?"


def test_date_candidates_lists_month_years_and_marks_the_title_page():
    note = C.date_candidates([(5, NIST_P5), (2, NIST_P2), (6, "Signed on October 30, 2023.")])
    assert "January 2023 (month 1): passage 1, page 5" in note
    assert "July 2024 (month 7): passage 2, title page" in note
    assert "October 2023 (month 10): passage 3" in note
    assert "publication date" in note
    assert C.date_candidates([(3, "No dates here, only 2023 numbers.")]) == ""
    assert C.asks_time_span(Q_MONTHS) and not C.asks_time_span("How many actions are listed?")


def test_date_candidates_reach_the_calculator_prompt_only_with_the_flag():
    index = make_index([(5, NIST_P5), (2, NIST_P2)], FakeEmbedder(), doc_id="nist")
    reply = json.dumps(
        {
            "needed": True,
            "operation": "months_between",
            "dates": [
                {"label": "AI RMF release", "year": 2023, "month": 1, "passage": 1},
                {"label": "profile publication", "year": 2024, "month": 7, "passage": 2},
            ],
            "unit": "months",
        }
    )
    for flag in (False, True):
        agent, llm = make_agent(
            scripted(
                calculate=reply,
                generate=cite("July 2024", "18 months passed [{n}]."),
                verify=VERIFY_OK,
            ),
            agent_mode="adaptive",
            calc_date_candidates=flag,
        )
        ans = agent.run(Q_MONTHS, [index])
        prompt = user_prompts(llm, "calculate")[0]
        assert ("Dates found in the passages" in prompt) is flag
        assert ("date_candidates" in steps(ans)) is flag
        assert "18" in ans.answer


# --------------------------------------------------------------------------- measure-aware retrieval

Q_MINING = "At what annual rate did real value added in the mining industry grow in Q2 2025?"


def _table_index():
    embedder = FakeEmbedder()
    base = make_index([(21, BEA_T13), (20, BEA_T12), (2, BEA_P1)], embedder, doc_id="bea")
    chunks = [replace(c, kind="table") if c.page in (20, 21) else c for c in base.chunks]
    return DocumentIndex("bea", "", chunks, base.embeddings)


def _retriever(**cfg):
    reranker = TableReranker(
        {(Q_MINING, "Table 13"): 0.99, (Q_MINING, "Table 12"): 0.98, (Q_MINING, "Real gross"): 0.5}
    )
    return HybridRetriever(RAGConfig(**cfg), FakeEmbedder(), reranker)


def test_contribution_table_is_moved_behind_the_rate_table_only_with_the_flag():
    index = _table_index()
    off = [
        sc.chunk.page
        for sc in _retriever(measure_aware_retrieval=False).retrieve([Q_MINING], [index], top_k=3)
    ]
    assert off == [21, 20, 2]  # BEA q08: the contributions table outranked Table 12
    on = _retriever(measure_aware_retrieval=True).retrieve([Q_MINING], [index], top_k=3)
    assert [sc.chunk.page for sc in on] == [20, 2, 21]
    # with top_k=2 the contributions table drops out of the context entirely
    on2 = _retriever(measure_aware_retrieval=True).retrieve([Q_MINING], [index], top_k=2)
    assert [sc.chunk.page for sc in on2] == [20, 2]


def test_measure_order_leaves_unclassified_questions_alone():
    ranked = _retriever(measure_aware_retrieval=False).retrieve(
        [Q_MINING], [_table_index()], top_k=3
    )
    assert measure_order("What did mining contribute in Q2 2025?", ranked) == ranked
    assert [sc.chunk.page for sc in measure_order(Q_MINING, ranked)] == [20, 2, 21]
