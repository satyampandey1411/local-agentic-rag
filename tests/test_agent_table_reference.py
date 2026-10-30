""""Which table shows X?" answers name the cited table (deterministic, no prompt change)."""

from __future__ import annotations

import json

import pytest

from local_agentic_rag.agent import prompts as P
from local_agentic_rag.agent.citations import (
    asks_which_table,
    find_caption,
    has_table_identifier,
)
from local_agentic_rag.schema import ANSWER_NOT_FOUND, REFUSAL_MESSAGE
from tests.helpers import FakeEmbedder, make_index
from tests.test_agent import VERIFY_OK, cite, make_agent, steps
from tests.test_agent import index as _unused  # noqa: F401

TABLE = (
    "Table 12.4: Average monthly rainfall by region in millimetres\n"
    "| Region | Rainfall |\n|---|---|\n| North | 120 |\n| South | 80 |"
)
CORPUS = [
    (4, TABLE),
    (5, "Rainfall is highest in the northern region during the monsoon season."),
    (6, "Figure 2: Quarterly revenue growth trend for the whole company\nRevenue grew steadily."),
    (7, "Table A.3: Employees by country of operation\n| Country | Staff |\n|---|---|\n| US | 9 |"),
]
WHICH = "Which table shows the average monthly rainfall by region?"
CAPTION = "Table 12.4: Average monthly rainfall by region in millimetres"


@pytest.fixture
def doc():
    ix = make_index(CORPUS, FakeEmbedder(), doc_id="rep", title="Climate Report")
    ix.chunks[0].kind = "table"
    ix.chunks[3].kind = "table"
    return ix


def agent_for(generate):
    from tests.helpers import scripted

    # One retrieval round, verifier off: the answer is final after the first generation.
    agent, llm = make_agent(
        scripted(generate=generate, verify=VERIFY_OK),
        agent_mode="adaptive",
        max_retrieval_rounds=1,
        use_verifier=False,
    )
    original = llm.responder

    def responder(messages, schema):  # a non-English question adds one translate call
        if schema is P.TRANSLATE_SCHEMA:
            return json.dumps({"translation": WHICH})
        return original(messages, schema)

    llm.responder = responder
    return agent


def test_identifier_appended_when_missing(doc):
    agent = agent_for(cite("Table 12.4", "The table with rainfall figures for each region [{n}]."))
    ans = agent.run(WHICH, [doc])
    n = ans.citations[0].marker
    assert ans.answer == (
        f"The table with rainfall figures for each region [{n}]. "
        f"(Table 12.4: Average monthly rainfall by region in millimetres [{n}])"
    )
    assert [c.page for c in ans.citations] == [4]
    assert not ans.refused


def test_trace_records_the_addition(doc):
    seen = []
    agent = agent_for(cite("Table 12.4", "It lists rainfall by region [{n}]."))
    ans = agent.run(WHICH, [doc], on_step=seen.append)
    assert steps(ans)[-1] == "table_reference_added"
    detail = ans.trace[-1].detail
    assert detail["identifier"] == "Table 12.4" and detail["page"] == 4
    assert detail["marker"] == ans.citations[0].marker
    assert seen[-1].step == "table_reference_added"


def test_not_appended_when_answer_already_names_a_table(doc):
    agent = agent_for(cite("Table 12.4", "Table 12.4 shows rainfall by region [{n}]."))
    ans = agent.run(WHICH, [doc])
    assert ans.answer.count("12.4") == 1
    assert "table_reference_added" not in steps(ans)


def test_any_identifier_in_the_answer_counts(doc):
    agent = agent_for(cite("Table 12.4", "That is table 3 in the report [{n}]."))
    ans = agent.run(WHICH, [doc])
    assert "table_reference_added" not in steps(ans)


def test_not_appended_for_non_table_questions(doc):
    agent = agent_for(cite("Table 12.4", "Rainfall in the north is 120 mm [{n}]."))
    ans = agent.run("What is the average monthly rainfall in the north region?", [doc])
    assert ans.answer == f"Rainfall in the north is 120 mm [{ans.citations[0].marker}]."
    assert "table_reference_added" not in steps(ans)


def test_not_appended_on_refusal(doc):
    agent = agent_for(ANSWER_NOT_FOUND)
    ans = agent.run(WHICH, [doc])
    assert ans.refused and ans.answer == REFUSAL_MESSAGE
    assert "table_reference_added" not in steps(ans)


def test_uses_the_number_of_the_cited_table_chunk(doc):
    # The text chunk is cited first but has no caption; the table is cited second.
    def reply(messages):
        text_n = _n(messages, "Rainfall is highest")
        table_n = _n(messages, "Table 12.4")
        return f"Rainfall is highest in the north [{text_n}]; see the table [{table_n}]."

    ans = agent_for(reply).run(WHICH, [doc])
    table_n = ans.citations[1].marker
    assert ans.citations[1].page == 4
    assert ans.answer.endswith(
        f"(Table 12.4: Average monthly rainfall by region in millimetres [{table_n}])"
    )


def test_identifier_only_from_cited_chunks(doc):
    # The captioned table was retrieved but the answer cites only the uncaptioned text chunk.
    agent = agent_for(cite("Rainfall is highest", "Rainfall is highest in the north [{n}]."))
    ans = agent.run(WHICH, [doc])
    assert [c.page for c in ans.citations] == [5]
    assert "Table" not in ans.answer
    assert "table_reference_added" not in steps(ans)


def test_first_cited_table_wins_and_other_tables_are_ignored(doc):
    def reply(messages):
        return f"[{_n(messages, 'Table A.3')}] [{_n(messages, 'Table 12.4')}] by region."

    ans = agent_for(reply).run("Which table lists the employees by country?", [doc])
    first = ans.citations[0].marker
    assert ans.answer.endswith(f"(Table A.3: Employees by country of operation [{first}])")
    assert "12.4" not in ans.answer


def test_figure_question_uses_caption_in_a_text_chunk(doc):
    agent = agent_for(cite("Figure 2", "The chart of quarterly revenue growth [{n}]."))
    ans = agent.run("Which figure shows the quarterly revenue growth trend?", [doc])
    n = ans.citations[0].marker
    assert ans.answer.endswith(
        f"(Figure 2: Quarterly revenue growth trend for the whole company [{n}])"
    )


def test_hindi_question_triggers(doc):
    agent = agent_for(cite("Table 12.4", "क्षेत्र के अनुसार वर्षा की तालिका [{n}]।"))
    ans = agent.run("कौन सी तालिका क्षेत्र के अनुसार औसत मासिक वर्षा दिखाती है?", [doc])
    assert "table_reference_added" in steps(ans)
    assert ans.answer.endswith(f"millimetres [{ans.citations[0].marker}])")


def test_works_in_the_full_agent_mode(doc):
    from tests.helpers import scripted
    from tests.test_agent import PLAN_OK, grade_relevant

    agent, _ = make_agent(
        scripted(
            plan={"standalone_question": WHICH, "search_queries": ["rainfall table"]},
            grade=grade_relevant("Table 12.4"),
            generate=cite("Table 12.4", "It gives rainfall by region [{n}]."),
            verify=VERIFY_OK,
        )
    )
    assert PLAN_OK  # shared fixture imported for parity with the other agent tests
    ans = agent.run(WHICH, [doc])
    assert steps(ans)[-1] == "table_reference_added"
    assert "Table 12.4" in ans.answer


def _n(messages, needle):
    from tests.helpers import passage_number

    return passage_number(messages, needle)


# ---------------------------------------------------------------------------
# trigger and caption parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        "Which table shows the budget by ministry?",
        "which tables list the revenue?",
        "In which table can I find the dividend?",
        "What table gives the unemployment rate?",
        "Which figure shows the growth trend?",
        "Which exhibit presents the segment results?",
        "कौन सी तालिका में बजट दिया गया है?",
        "किस तालिका में आय दी गई है?",
        "कौन सा चित्र वृद्धि दिखाता है?",
    ],
)
def test_trigger_matches(q):
    assert asks_which_table(q)


@pytest.mark.parametrize(
    "q",
    [
        "What does Table 3 show?",
        "What was revenue in fiscal 2023?",
        "Which year had the highest revenue?",
        "Which figure is higher, 2022 or 2023?",
        "What is the total in Table 2?",
        "Which table 3 row has the lowest value?",
        "Summarise the tables in this report.",
        "तालिका 3 में क्या दिया गया है?",
    ],
)
def test_trigger_does_not_match(q):
    assert not asks_which_table(q)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("It is Table 12.4.", True),
        ("see table 2", True),
        ("Table A.3 lists it", True),
        ("Figure 2 shows", True),
        ("Exhibit 5", True),
        ("तालिका 5 में", True),
        ("The table below lists it", False),
        ("a table is shown", False),
        ("Tablet 3 sales", False),
    ],
)
def test_has_table_identifier(text, expected):
    assert has_table_identifier(text) is expected


def test_find_caption_variants():
    assert find_caption("Table 12.4: Rainfall\n| a | b |") == ("Table 12.4", "Rainfall")
    assert find_caption("TABLE IV. Staff by grade\n| a | b |") == ("TABLE IV", "Staff by grade")
    assert find_caption("Table A.3: Staff") == ("Table A.3", "Staff")
    assert find_caption("Exhibit 5 - Segment results") == ("Exhibit 5", "Segment results")
    assert find_caption("Table 2 shows the results in detail.") is None
    assert find_caption("no caption here\n| a | b |") is None


def test_find_caption_wrapped_truncated_and_footnote_free():
    assert find_caption("Table 2: Results on the\nnatural questions set\n| a |") == (
        "Table 2",
        "Results on the natural questions set",
    )
    ident, title = find_caption("Table 9: " + "word " * 60)
    assert ident == "Table 9" and len(title) <= 121 and title.endswith("…")
    assert find_caption("Table 3: Revenue [1] by year [2, 3]") == ("Table 3", "Revenue by year")
