"""Unit tests for local_agentic_rag.agent.citations."""

from __future__ import annotations

import pytest

from local_agentic_rag.agent.citations import (
    build_citations,
    infer_citations,
    is_refusal,
    parse_markers,
    strip_invalid_markers,
)
from local_agentic_rag.schema import ANSWER_NOT_FOUND, Chunk, ScoredChunk


def passages(*items: tuple[int, str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"doc-p{page}-c{i}", doc_id="doc", page=page, section="", text=text
            ),
            score=1.0,
        )
        for i, (page, text) in enumerate(items)
    ]


PASSAGES = passages(
    (3, "Revenue increased to 4.2 billion dollars in fiscal 2023, driven by cloud subscriptions."),
    (7, "The company employs 12,000 people across 40 countries worldwide."),
    (7, "Operating margin improved to 18 percent after restructuring charges ended."),
    (12, "The board approved a dividend of 0.50 dollars per share in March."),
)


# ---------------------------------------------------------------------------
# parse_markers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue rose [1][3].", [1, 3]),
        ("Revenue rose [2, 4].", [2, 4]),
        ("A [2;3] b [ 1 ]", [2, 3]),  # "[ 1 ]" is not a marker
        ("first [3] then [1] then [3] again", [3, 1]),
        ("no markers here", []),
        ("[10] and [2,10]", [10, 2]),
    ],
)
def test_parse_markers(text, expected):
    assert parse_markers(text) == expected


# ---------------------------------------------------------------------------
# strip_invalid_markers
# ---------------------------------------------------------------------------


def test_strip_removes_out_of_range_marker():
    assert strip_invalid_markers("Revenue rose [5].", 4) == "Revenue rose."


def test_strip_keeps_valid_and_filters_groups():
    assert (
        strip_invalid_markers("Revenue rose [2, 9] and fell [1][7].", 4)
        == "Revenue rose [2] and fell [1]."
    )


def test_strip_zero_is_invalid():
    out = strip_invalid_markers("Text [0] here [1].", 2)
    assert " ".join(out.split()) == "Text here [1]."  # mid-sentence removal leaves a double space


def test_strip_no_passages_removes_all():
    out = strip_invalid_markers("A [1]. B [2, 3].", 0)
    assert parse_markers(out) == []
    assert out == "A. B."


def test_strip_untouched_when_all_valid():
    text = "Revenue rose [1]. Margin improved [2, 3]."
    assert strip_invalid_markers(text, 3) == text


# ---------------------------------------------------------------------------
# build_citations
# ---------------------------------------------------------------------------


def test_build_citations_maps_markers_to_pages():
    cites = build_citations("Revenue was 4.2B [1]. Margin was 18% [3]. Staff 12,000 [2].", PASSAGES)
    assert [c.marker for c in cites] == [1, 3, 2]
    assert [c.page for c in cites] == [3, 7, 7]
    assert cites[0].chunk_id == "doc-p3-c0"
    assert cites[0].doc_id == "doc"
    assert cites[0].snippet == PASSAGES[0].chunk.text[:200]


def test_build_citations_ignores_out_of_range():
    cites = build_citations("Dividend [4]. Nonsense [9].", PASSAGES)
    assert [(c.marker, c.page) for c in cites] == [(4, 12)]


def test_build_citations_snippet_truncated():
    long = passages((1, "x" * 500))
    assert len(build_citations("[1]", long)[0].snippet) == 200


def test_build_citations_none():
    assert build_citations("no citations", PASSAGES) == []


# ---------------------------------------------------------------------------
# infer_citations
# ---------------------------------------------------------------------------


def test_infer_citations_attributes_sentences_by_overlap():
    answer = (
        "Revenue increased to 4.2 billion dollars in fiscal 2023. "
        "The board approved a dividend of 0.50 dollars per share."
    )
    cites = infer_citations(answer, PASSAGES)
    assert sorted(c.page for c in cites) == [3, 12]
    assert {c.marker for c in cites} == {1, 4}


def test_infer_citations_requires_min_overlap():
    assert infer_citations("Penguins migrate across Antarctic ice shelves yearly.", PASSAGES) == []


def test_infer_citations_skips_short_sentences():
    assert infer_citations("Revenue increased.", PASSAGES) == []


def test_infer_citations_dedupes_passages():
    answer = "Operating margin improved to 18 percent. Operating margin improved after restructuring charges ended."
    cites = infer_citations(answer, PASSAGES)
    assert [c.marker for c in cites] == [3]


# ---------------------------------------------------------------------------
# is_refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        ANSWER_NOT_FOUND,
        f"  {ANSWER_NOT_FOUND}  ",
        f"Sorry. {ANSWER_NOT_FOUND}",
        "",
        "   ",
        "The document does not mention the CEO's salary.",
        "The passages do not provide information about headcount in 2019.",
        "This is not mentioned in the provided passages.",
        "There is no information about the merger.",
        "The answer cannot be determined from the passages.",
    ],
)
def test_is_refusal_true(text):
    assert is_refusal(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Revenue was 4.2 billion dollars [1].",
        "Revenue was 4.2 billion dollars [1]. The reason for the increase is not mentioned in the document.",
        "The document does not mention a dividend cut; it says a dividend of 0.50 was approved [4].",
        "Revenue was 4.2 billion dollars.",  # uncited but substantive
    ],
)
def test_is_refusal_false(text):
    assert is_refusal(text) is False


def test_long_uncited_answer_with_refusal_phrase_is_not_refusal():
    text = (
        "Revenue grew strongly across all segments. " * 12
        + "The document does not mention guidance."
    )
    assert len(text) >= 400
    assert is_refusal(text) is False
