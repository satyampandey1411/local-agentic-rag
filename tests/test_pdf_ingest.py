"""Unit tests for local_agentic_rag.ingest.pdf (PyMuPDF parsing of reportlab PDFs)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("reportlab")

from local_agentic_rag.ingest.chunker import chunk_document  # noqa: E402
from local_agentic_rag.ingest.pdf import PDFParseError, parse_pdf, sha256_file  # noqa: E402
from tests.helpers import (  # noqa: E402
    make_blank_pdf,
    make_table_pdf,
    make_text_pdf,
    make_two_column_pdf,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pdf_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("pdf_ingest")


@pytest.fixture(scope="module")
def text_pdf(pdf_dir):
    path = pdf_dir / "annual_report.pdf"
    pages = make_text_pdf(path, n_pages=4)
    return path, pages


@pytest.fixture(scope="module")
def text_doc(text_pdf):
    path, _ = text_pdf
    return parse_pdf(path)


@pytest.fixture(scope="module")
def table_pdf(pdf_dir) -> Path:
    path = pdf_dir / "sales.pdf"
    make_table_pdf(path)
    return path


@pytest.fixture(scope="module")
def table_doc(table_pdf):
    return parse_pdf(table_pdf)


# ---------------------------------------------------------------------------
# Simple multi-page document
# ---------------------------------------------------------------------------


def test_document_metadata(text_pdf, text_doc):
    path, pages = text_pdf
    assert text_doc.num_pages == len(pages) == 4
    assert [p.page for p in text_doc.pages] == [1, 2, 3, 4]
    assert text_doc.doc_id == "annual-report"
    assert text_doc.sha256 == sha256_file(path)
    assert text_doc.source_path == str(path)
    assert text_doc.warnings == []


def test_explicit_doc_id_is_used(text_pdf):
    path, _ = text_pdf
    assert parse_pdf(path, "my-doc").doc_id == "my-doc"


def test_title_falls_back_to_largest_font_text(text_doc):
    # The PDF has no metadata title, so the largest-font text on page 1 wins.
    assert text_doc.title == "Section 1 Overview of Topic1"


def test_headings_detected_from_larger_bold_font(text_pdf, text_doc):
    _, pages = text_pdf
    for page, (heading, _) in zip(text_doc.pages, pages, strict=True):
        headings = [b.text for b in page.blocks if b.kind == "heading"]
        assert headings == [heading]
        paragraphs = [b for b in page.blocks if b.kind == "paragraph"]
        assert len(paragraphs) == 3


def test_body_text_is_extracted_per_page(text_pdf, text_doc):
    _, pages = text_pdf
    for page, (_, marker) in zip(text_doc.pages, pages, strict=True):
        assert marker in page.text
        # No other page's unique marker leaks onto this page.
        others = [m for _, m in pages if m != marker]
        assert not any(o in page.text for o in others)


def test_running_header_footer_and_page_numbers_are_stripped(text_doc):
    for page in text_doc.pages:
        text = page.text
        assert "Running Header" not in text
        assert "Confidential draft footer" not in text
        assert f"Page {page.page}" not in text
        assert all(b.text.strip() != f"Page {page.page}" for b in page.blocks)


def test_header_footer_kept_when_stripping_disabled(text_pdf):
    path, _ = text_pdf
    doc = parse_pdf(path, strip_headers_footers=False)
    joined = "\n".join(p.text for p in doc.pages)
    assert "Running Header" in joined
    assert "Confidential draft footer" in joined


def test_blocks_are_in_top_to_bottom_order(text_doc):
    for page in text_doc.pages:
        tops = [b.bbox[1] for b in page.blocks]
        assert tops == sorted(tops)
        assert page.blocks[0].kind == "heading"


# ---------------------------------------------------------------------------
# Two-column reading order
# ---------------------------------------------------------------------------


def test_two_column_reading_order_left_then_right(pdf_dir):
    path = pdf_dir / "two_column.pdf"
    make_two_column_pdf(path)
    doc = parse_pdf(path)
    labels = [b.text.split()[0] for b in doc.pages[0].blocks]
    assert labels == ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"]
    text = doc.pages[0].text
    last_left = max(text.index(f"A{i}") for i in range(1, 6))
    first_right = min(text.index(f"B{i}") for i in range(1, 6))
    assert last_left < first_right


# ---------------------------------------------------------------------------
# Ruled tables
# ---------------------------------------------------------------------------


def test_ruled_table_is_extracted(table_doc):
    page = table_doc.pages[0]
    assert len(page.tables) == 1
    table = page.tables[0]
    for value in (
        "Northwind",
        "Southgate",
        "Eastvale",
        "Westport",
        "1250",
        "980",
        "1430",
        "402",
        "6.8%",
    ):
        assert value in table


def test_table_cells_not_duplicated_in_body_text(table_doc):
    body = table_doc.pages[0].text
    assert "Regional Sales Summary" in body
    assert "Northwind" not in body
    assert "1430" not in body


def test_table_chunk_is_produced(table_doc):
    chunks = chunk_document(table_doc)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t.page == 1
    assert t.chunk_id == f"{table_doc.doc_id}-p1-t0-0"
    for value in ("Northwind", "1250", "Westport", "198"):
        assert value in t.text
    assert any(c.kind == "text" for c in chunks)


def test_tables_skipped_when_disabled(table_pdf):
    doc = parse_pdf(table_pdf, extract_tables=False)
    assert doc.pages[0].tables == []
    assert "Northwind" in doc.pages[0].text


def test_negative_number_in_table_is_preserved(table_doc):
    table = table_doc.pages[0].tables[0]
    assert "-1.2%" in table
    assert "&amp;" not in table and "&#45;" not in table


def test_table_header_row_is_the_first_grid_row(table_doc):
    table = table_doc.pages[0].tables[0]
    first_row = table.splitlines()[0]
    assert "Region" in first_row and "Revenue" in first_row
    assert "following table" not in table


def test_table_caption_is_attached_to_its_table(pdf_dir):
    path = pdf_dir / "captioned.pdf"
    make_table_pdf(path, caption="Table 1: Revenue by region in fiscal 2024.")
    page = parse_pdf(path).pages[0]
    lines = page.tables[0].splitlines()
    assert lines[0] == "Table 1: Revenue by region in fiscal 2024."
    assert "Region" in lines[1] and "Revenue" in lines[1]
    # The caption moves out of the running text; the paragraph after it stays.
    assert "Table 1:" not in page.text
    assert "Growth was strongest in Eastvale" in page.text


def test_table_reference_in_text_is_not_a_caption(pdf_dir):
    path = pdf_dir / "referenced.pdf"
    make_table_pdf(path, caption="Table 1 shows that Eastvale grew fastest.")
    page = parse_pdf(path).pages[0]
    assert page.tables[0].splitlines()[0].startswith("|")
    assert "Table 1 shows that Eastvale grew fastest." in page.text


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_blank_pdf_raises_parse_error(tmp_path):
    path = tmp_path / "blank.pdf"
    make_blank_pdf(path)
    with pytest.raises(PDFParseError, match="No text could be extracted"):
        parse_pdf(path)


def test_non_pdf_bytes_raise_parse_error(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_text("this is plain text pretending to be a PDF " * 10)
    with pytest.raises(PDFParseError, match="Could not open"):
        parse_pdf(path)


def test_non_pdf_text_file_raises_parse_error(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello this is not a pdf at all, just some text " * 5)
    with pytest.raises(PDFParseError):
        parse_pdf(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "does_not_exist.pdf")


def test_directory_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path)


def test_pdf_parse_error_is_runtime_error():
    assert issubclass(PDFParseError, RuntimeError)


def test_per_chapter_running_header_lines_are_stripped(tmp_path):
    """A header block = fixed title line + a chapter line that changes every few pages."""
    import pymupdf

    path = tmp_path / "chapters.pdf"
    doc = pymupdf.open()
    for p in range(20):  # 5 chapters x 4 pages: each chapter line is on 20% of pages
        page = doc.new_page()
        chapter = p // 4 + 1
        page.insert_text(
            (72, 30),
            f"Statistical Compendium 2024\nChapter {chapter}: Region {'ABCDE'[p // 4]}",
            fontsize=8,
        )
        page.insert_text(
            (72, 120), f"Body text on page {p + 1} about district {p} totals.", fontsize=11
        )
        page.insert_text((280, 810), f"Page {p + 1}", fontsize=8)
    doc.save(path)
    parsed = parse_pdf(path, ocr="off")
    texts = [b.text for page in parsed.pages for b in page.blocks]
    assert not any("Compendium" in t or t.startswith("Chapter ") for t in texts), texts[:6]
    assert sum("Body text" in t for t in texts) == 20


def test_numbered_headings_at_page_top_are_not_running_headers(tmp_path):
    """'3.1 ...' .. '3.5 ...' at the top of 5 consecutive pages differ only in digits; the
    consecutive-page rule compares exact text, so they are kept (they are under 40%)."""
    import pymupdf

    path = tmp_path / "sections.pdf"
    doc = pymupdf.open()
    for p in range(20):
        page = doc.new_page()
        if p < 5:
            page.insert_text((72, 30), f"Section 3.{p + 1} Results", fontsize=8)
        page.insert_text(
            (72, 120), f"Body text on page {p + 1} about district {p} totals.", fontsize=11
        )
    doc.save(path)
    texts = [b.text for page in parse_pdf(path, ocr="off").pages for b in page.blocks]
    assert sum(t.startswith("Section 3.") for t in texts) == 5
