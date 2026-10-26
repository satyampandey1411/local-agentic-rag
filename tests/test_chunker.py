"""Unit tests for local_agentic_rag.ingest.chunker (structure-aware chunking)."""

from __future__ import annotations

import re

import pytest

from local_agentic_rag.ingest.chunker import _split_table, approx_token_count, chunk_document
from local_agentic_rag.schema import PageContent, ParsedDocument, TextBlock


def make_doc(
    pages: list[tuple[list[tuple[str, str]], list[str]]], doc_id: str = "d"
) -> ParsedDocument:
    """pages: [([(kind, text), ...], [table_md, ...]), ...]"""
    return ParsedDocument(
        doc_id=doc_id,
        title="Test Title",
        source_path="/x.pdf",
        sha256="0" * 64,
        num_pages=len(pages),
        pages=[
            PageContent(page=i + 1, blocks=[TextBlock(t, k) for k, t in blocks], tables=tables)
            for i, (blocks, tables) in enumerate(pages)
        ],
    )


def sentence(page: int, i: int) -> str:
    return f"Sentence p{page}s{i} describes the quarterly results in some detail."


def paragraph(page: int, n: int, start: int = 0) -> str:
    return " ".join(sentence(page, i) for i in range(start, start + n))


def ids_in(text: str) -> list[str]:
    return re.findall(r"p\d+s\d+", text)


@pytest.fixture
def two_page_doc() -> ParsedDocument:
    return make_doc(
        [
            ([("heading", "Introduction"), ("paragraph", paragraph(1, 30))], []),
            (
                [
                    ("paragraph", paragraph(2, 30)),
                    ("heading", "Methods"),
                    ("paragraph", paragraph(2, 10, 30)),
                ],
                [],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# approx_token_count
# ---------------------------------------------------------------------------


def test_approx_token_count():
    assert approx_token_count("") == 1
    assert approx_token_count("one two three") == 3
    assert approx_token_count("Hello, world!") == 4  # 4 pieces * 1.1 → 4


# ---------------------------------------------------------------------------
# Page boundaries
# ---------------------------------------------------------------------------


def test_chunks_never_cross_pages(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        for sid in ids_in(c.text):
            assert sid.startswith(f"p{c.page}s"), (c.chunk_id, sid)


def test_chunk_ids_and_metadata(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    page1 = [c for c in chunks if c.page == 1]
    assert [c.chunk_id for c in page1] == [f"d-p1-c{i}" for i in range(len(page1))]
    for c in chunks:
        assert c.doc_id == "d"
        assert c.doc_title == "Test Title"
        assert c.kind == "text"
        assert c.n_tokens == approx_token_count(c.text)


def test_all_text_is_covered(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    seen = {sid for c in chunks for sid in ids_in(c.text)}
    expected = {f"p1s{i}" for i in range(30)} | {f"p2s{i}" for i in range(40)}
    assert seen == expected


# ---------------------------------------------------------------------------
# Size limits and overlap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_tokens", [40, 60, 120])
def test_chunks_respect_chunk_tokens_without_overlap(two_page_doc, chunk_tokens):
    chunks = chunk_document(
        two_page_doc, chunk_tokens=chunk_tokens, overlap_tokens=0, min_chunk_tokens=10
    )
    assert len(chunks) > 2
    assert all(c.n_tokens <= chunk_tokens for c in chunks)


def test_chunks_respect_chunk_tokens_with_sentence_overlap(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    assert all(c.n_tokens <= 60 for c in chunks)


def test_overlap_never_pushes_chunk_over_limit():
    # Each paragraph is a single 49-token unit; observed chunk sizes are [49, 66, 66, 66].
    paras = [
        ("paragraph", paragraph(1, 4, start=4 * k) + " Extra words follow here.") for k in range(4)
    ]
    doc = make_doc([(paras, [])])
    chunks = chunk_document(doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    assert all(c.n_tokens <= 60 for c in chunks), [c.n_tokens for c in chunks]


def test_overlap_between_consecutive_chunks_on_same_page(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    page1 = [c for c in chunks if c.page == 1]
    assert len(page1) >= 3
    for prev, nxt in zip(page1, page1[1:], strict=False):
        prev_ids, next_ids = ids_in(prev.text), ids_in(nxt.text)
        # The last sentence of a chunk opens the next one.
        assert next_ids[0] == prev_ids[-1]


def test_no_overlap_when_disabled(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=0, min_chunk_tokens=10)
    page1 = [c for c in chunks if c.page == 1]
    for prev, nxt in zip(page1, page1[1:], strict=False):
        assert not set(ids_in(prev.text)) & set(ids_in(nxt.text))


def test_no_overlap_across_pages(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    first_p2 = next(c for c in chunks if c.page == 2)
    assert ids_in(first_p2.text)[0] == "p2s0"


def test_huge_sentence_split_into_word_windows():
    words = " ".join(f"word{i}" for i in range(200))  # one 200-token "sentence"
    doc = make_doc([([("paragraph", words)], [])])
    chunks = chunk_document(doc, chunk_tokens=50, overlap_tokens=0, min_chunk_tokens=5)
    assert len(chunks) >= 4
    assert all(
        c.n_tokens <= 55 for c in chunks
    )  # windows stop at the first word reaching the limit
    rebuilt = " ".join(c.text for c in chunks)
    assert rebuilt.split() == words.split()


def test_small_trailing_fragment_merged_into_previous_chunk():
    doc = make_doc([([("paragraph", paragraph(1, 5)), ("paragraph", "Tiny tail here ok.")], [])])
    chunks = chunk_document(doc, chunk_tokens=200, overlap_tokens=0, min_chunk_tokens=24)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("Tiny tail here ok.")


def test_contentless_fragments_dropped():
    doc = make_doc([([("paragraph", "x y")], [])])
    assert chunk_document(doc) == []


# ---------------------------------------------------------------------------
# Section headings
# ---------------------------------------------------------------------------


def test_chunks_carry_section_headings(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    for c in chunks:
        ids = ids_in(c.text)
        if c.page == 1 or int(ids[-1][3:]) < 30:
            assert c.section == "Introduction", c.chunk_id
        else:
            assert c.section == "Methods", c.chunk_id


def test_section_carries_over_page_boundary(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    assert next(c for c in chunks if c.page == 2).section == "Introduction"


def test_headings_not_in_chunk_text_but_in_embed_text(two_page_doc):
    chunks = chunk_document(two_page_doc, chunk_tokens=60, overlap_tokens=20, min_chunk_tokens=10)
    assert not any("Introduction" in c.text for c in chunks)
    assert chunks[0].embed_text.startswith("Test Title > Introduction\n")


def test_section_after_small_fragment_gets_new_heading():
    doc = make_doc(
        [
            (
                [
                    ("heading", "H1"),
                    ("paragraph", paragraph(1, 30)),
                    ("heading", "H2"),
                    ("paragraph", "Short bit here."),
                    ("heading", "H3"),
                    ("paragraph", paragraph(1, 3, start=100)),
                ],
                [],
            )
        ]
    )
    chunks = chunk_document(doc, chunk_tokens=60, overlap_tokens=0, min_chunk_tokens=10)
    h3 = [c for c in chunks if "p1s100" in c.text]
    assert h3 and h3[0].section == "H3"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def markdown_table(n_rows: int) -> str:
    rows = [f"|Region {i}|{1000 + i}|{i}.5%|" for i in range(n_rows)]
    return "\n".join(["|Region|Revenue|Growth|", "|---|---|---|", *rows])


def test_small_table_is_single_chunk():
    doc = make_doc([([("heading", "Sales"), ("paragraph", paragraph(1, 3))], [markdown_table(3)])])
    tables = [c for c in chunk_document(doc) if c.kind == "table"]
    assert len(tables) == 1
    assert tables[0].text == markdown_table(3)
    assert tables[0].chunk_id == "d-p1-t0-0"
    assert tables[0].page == 1


def test_table_split_repeats_markdown_header():
    md = markdown_table(40)
    doc = make_doc([([("paragraph", paragraph(1, 3))], [md])])
    tables = [
        c for c in chunk_document(doc, chunk_tokens=80, overlap_tokens=0) if c.kind == "table"
    ]
    assert len(tables) > 2
    for part in tables:
        lines = part.text.splitlines()
        assert lines[:2] == ["|Region|Revenue|Growth|", "|---|---|---|"]
        assert len(lines) > 2
        assert part.n_tokens <= 80
    body_rows = [ln for part in tables for ln in part.text.splitlines()[2:]]
    assert body_rows == md.splitlines()[2:]
    assert [t.chunk_id for t in tables] == [f"d-p1-t0-{i}" for i in range(len(tables))]


def test_captioned_table_split_repeats_caption_and_header():
    md = "Table 2: Revenue by region.\n" + markdown_table(40)
    parts = _split_table(md, 80, approx_token_count)
    assert len(parts) > 2
    for part in parts:
        lines = part.splitlines()
        assert lines[:3] == [
            "Table 2: Revenue by region.",
            "|Region|Revenue|Growth|",
            "|---|---|---|",
        ]
        assert len(lines) > 3
    assert [ln for p in parts for ln in p.splitlines()[3:]] == md.splitlines()[3:]


def test_serialised_table_split_repeats_title_line():
    lines = ["Table 3. Prices"] + [
        f"Section | Item {i} | 2023 = {i}.0; 2024 = {i}.5" for i in range(30)
    ]
    parts = _split_table("\n".join(lines), 60, approx_token_count)
    assert len(parts) > 2
    for part in parts:
        assert part.splitlines()[0] == "Table 3. Prices"
    assert [ln for p in parts for ln in p.splitlines()[1:]] == lines[1:]


def test_table_section_is_section_at_page_start():
    doc = make_doc(
        [
            ([("heading", "Before"), ("paragraph", paragraph(1, 3))], []),
            ([("heading", "After"), ("paragraph", paragraph(2, 3))], [markdown_table(5)]),
        ]
    )
    table = next(c for c in chunk_document(doc) if c.kind == "table")
    assert table.page == 2
    assert table.section == "Before"


def test_table_section_is_its_own_title_not_the_previous_tables():
    # One table per page, titled at the top: the section at the page start is the
    # previous page's table, which must not label this table's rows.
    t12 = "Table 12. Real Gross Domestic Product by Industry Group: Percent Change"
    serialised = f"September 25, 2025 {t12}\nMining | 2025 Q1 = -18.3; 2025 Q2 = 29.0"
    doc = make_doc(
        [
            ([("heading", "Table 10. Corporate Profits by Industry: Level")], []),
            ([("heading", t12)], [serialised]),
            # Continued on the next page: the inherited section already names Table 12.
            ([("paragraph", paragraph(3, 2))], [f"{t12}--Continues\nUtilities | 2025 Q2 = 1.0"]),
            # Markdown tables (no title line) keep the section at the page start.
            ([("heading", "Appendix")], [markdown_table(3)]),
        ]
    )
    tables = [c for c in chunk_document(doc) if c.kind == "table"]
    assert [t.section for t in tables] == [t12, t12, t12]
