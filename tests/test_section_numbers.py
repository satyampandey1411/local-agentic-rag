"""Chunks record the numbered section they sit in (``Chunk.section_no``) so that questions
naming "section 164" find it even when a bold sub-heading directly follows "§ 164. Taxes"."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from local_agentic_rag.config import RAGConfig
from local_agentic_rag.ingest.chunker import PageChunker, chunk_document
from local_agentic_rag.retrieval.hybrid import HybridRetriever
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.schema import Chunk, PageContent, ParsedDocument, TextBlock
from local_agentic_rag.sections import STRUCTURAL_HEAD_RE
from tests.helpers import FakeEmbedder, FakeReranker, make_chunks

# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

BODY = "The taxpayer shall be allowed a deduction for amounts paid during the taxable year. "


def _doc(pages: list[list[tuple[str, str]]]) -> ParsedDocument:
    return ParsedDocument(
        doc_id="d",
        title="Code",
        source_path="/x.pdf",
        sha256="0" * 64,
        num_pages=len(pages),
        pages=[
            PageContent(page=i + 1, blocks=[TextBlock(t, k) for k, t in blocks], tables=[])
            for i, blocks in enumerate(pages)
        ],
    )


def _code_doc() -> ParsedDocument:
    return _doc(
        [
            [
                ("heading", "§ 164. Taxes"),
                ("heading", "(a) General rule"),
                ("paragraph", BODY * 3),
            ],
            [
                ("paragraph", BODY * 3 + "Interest accrues."),
                ("heading", "Editorial Notes"),
                ("paragraph", "Amendments and history of the section. " * 4),
            ],
            [
                ("heading", "CHAPTER 79—DEFINITIONS"),
                ("paragraph", "Sec. 7701 Definitions. Sec. 7702 Life insurance contracts. " * 3),
                ("heading", "§ 165. Losses"),
                ("paragraph", BODY * 3 + "Losses of individuals."),
            ],
        ]
    )


def _numbers(chunks: list[Chunk]) -> list[tuple[int, str, str | None]]:
    return [(c.page, c.section, c.section_no) for c in chunks]


def test_sub_headings_keep_the_numbered_section():
    chunks = chunk_document(_code_doc(), chunk_tokens=80, overlap_tokens=0, min_chunk_tokens=5)
    by_page = {}
    for page, section, no in _numbers(chunks):
        by_page.setdefault(page, []).append((section, no))
    assert ("(a) General rule", "164") in by_page[1]
    assert ("Editorial Notes", "164") in by_page[2]
    assert {no for _, no in by_page[1] + by_page[2]} == {"164"}
    assert ("§ 165. Losses", "165") in by_page[3]


def test_structural_heading_resets_the_number():
    chunks = chunk_document(_code_doc(), chunk_tokens=80, overlap_tokens=0, min_chunk_tokens=5)
    toc = [c for c in chunks if c.page == 3 and c.section.startswith("CHAPTER")]
    assert toc and all(c.section_no is None for c in toc)  # not attributed to section 164
    assert all(c.section_no == "165" for c in chunks if c.section == "§ 165. Losses")


@pytest.mark.parametrize(
    ("heading", "structural"),
    [
        ("CHAPTER 79—DEFINITIONS", True),
        ("Subchapter A—Crimes", True),
        ("PART I—GENERAL", True),
        ("ARTICLE 3.", True),
        ("TITLE 26", True),
        ("Subtitle F—Procedure", True),
        ("Part of the problem", False),
        ("Title page", False),
        ("(a) General rule", False),
        ("Editorial Notes", False),
    ],
)
def test_structural_heading_pattern_is_conservative(heading, structural):
    assert bool(STRUCTURAL_HEAD_RE.match(heading)) is structural


def test_section_starting_inside_text_sets_the_number():
    doc = _doc(
        [
            [
                ("paragraph", "Earlier text. " * 4),
                ("paragraph", "§ 164. Taxes (a) General rule " + BODY * 2),
            ]
        ]
    )
    chunks = chunk_document(doc, chunk_tokens=40, overlap_tokens=0, min_chunk_tokens=5)
    assert chunks[0].section_no is None
    assert chunks[-1].section_no == "164"


def test_overlap_seeded_chunk_takes_the_section_of_its_new_text():
    """The unit after "§ 165." overflows the buffer; the chunk re-seeded with the overlap tail
    holds § 165 text, so it is a § 165 chunk, not a second § 164 one."""
    text = (
        "§ 164. Taxes " + BODY * 2 + "Also this. "
        "§ 165. Losses There shall be allowed a deduction. " + BODY
    )
    doc = _doc([[("paragraph", text)]])
    chunks = chunk_document(doc, chunk_tokens=48, overlap_tokens=10, min_chunk_tokens=5)
    assert len(chunks) == 2 and chunks[1].text.startswith("Also this. § 165. Losses")  # the tail
    assert [c.section_no for c in chunks] == ["164", "165"]


def test_chunk_text_does_not_depend_on_section_numbers():
    chunks = chunk_document(_code_doc(), chunk_tokens=80, overlap_tokens=0, min_chunk_tokens=5)
    assert all("164" not in c.embed_text for c in chunks if c.section_no == "165")
    assert chunks[0].embed_text == f"Code > {chunks[0].section}\n{chunks[0].text}"


def test_section_no_is_serialised_only_when_set():
    plain = make_chunks([(1, "Some text about nothing in particular.")])[0]
    assert "section_no" not in plain.to_dict()
    numbered = replace(plain, section_no="164")
    assert numbered.to_dict()["section_no"] == "164"
    assert Chunk.from_dict(numbered.to_dict()) == numbered
    assert Chunk.from_dict(plain.to_dict()).section_no is None  # an old chunks.jsonl row


def test_number_survives_a_checkpoint_between_pages():
    doc = _code_doc()
    kw = {"chunk_tokens": 80, "overlap_tokens": 0, "min_chunk_tokens": 5}
    whole = chunk_document(doc, **kw)

    first = PageChunker("d", "Code", **kw)
    head = first.add_page(doc.pages[0])
    assert first.section_no == "164" and first.section == "(a) General rule"
    resumed = PageChunker("d", "Code", section=first.section, section_no=first.section_no, **kw)
    tail = [c for p in doc.pages[1:] for c in resumed.add_page(p)]
    assert head + tail == whole
    # resuming WITHOUT the number loses the section: that is what the checkpoint prevents
    lost = PageChunker("d", "Code", section=first.section, **kw).add_page(doc.pages[1])
    assert lost[0].section_no is None


# ---------------------------------------------------------------------------
# Real PDF through streaming ingest (bold "§ 164. Taxes" then bold "(a) General rule")
# ---------------------------------------------------------------------------


def _make_code_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    from tests.helpers import _wrap

    _, height = LETTER
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle("")

    def para(y: float, body: str) -> float:
        text = c.beginText(72, y)
        text.setFont("Helvetica", 10)
        text.setLeading(13)
        for line in _wrap(body, 95):
            text.textLine(line)
        c.drawText(text)
        return y - 13 * len(_wrap(body, 95)) - 12

    def bold(y: float, label: str) -> float:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y, label)
        return y - 26

    body = BODY * 14
    y = bold(height - 100, "§ 164. Taxes")
    y = bold(y, "(a) General rule")
    para(y, body)
    c.showPage()
    y = para(height - 100, body + "Interest accrues until paid.")
    y = bold(y - 10, "§ 165. Losses")
    para(y, "Losses of individuals are deductible as follows. " + body)
    c.showPage()
    c.save()


@pytest.fixture(scope="module")
def code_index(tmp_path_factory):
    pytest.importorskip("pymupdf")
    pytest.importorskip("reportlab")
    from local_agentic_rag.ingest.streaming import ingest_pdf

    root = tmp_path_factory.mktemp("code")
    pdf = root / "code.pdf"
    _make_code_pdf(pdf)
    index, _ = ingest_pdf(pdf, root / "store", RAGConfig(ocr="off"), FakeEmbedder(), workers=1)
    return index


def test_pdf_section_chunks_return_only_that_section(code_index):
    chunks = code_index.chunks
    sec164 = code_index.section_chunks("164")
    sec165 = code_index.section_chunks("165")
    assert sec164 and sec165 and not set(sec164) & set(sec165)
    assert any(chunks[i].section == "(a) General rule" for i in sec164)  # the failing shape
    assert all("Losses of individuals" not in chunks[i].text for i in sec164)
    assert any("Losses of individuals" in chunks[i].text for i in sec165)
    assert code_index.section_chunks("16") == [] and code_index.section_chunks("1640") == []


def test_pdf_resume_mid_document_keeps_the_number(code_index, tmp_path):
    from local_agentic_rag.ingest import streaming
    from local_agentic_rag.ingest.streaming import ingest_pdf

    pdf = tmp_path / "code.pdf"
    _make_code_pdf(pdf)

    class Stop(Exception):
        pass

    def crash(p):
        if p.stage == "indexing" and p.pages_done >= 1:
            raise Stop  # dies right after the checkpoint that ends page 1 (inside §164)

    with pytest.raises(Stop):
        ingest_pdf(
            pdf, tmp_path / "s", RAGConfig(ocr="off"), FakeEmbedder(), progress=crash, workers=1
        )
    index, report = ingest_pdf(pdf, tmp_path / "s", RAGConfig(ocr="off"), FakeEmbedder(), workers=1)
    assert report.resumed_from_page >= 1
    assert [c.to_dict() for c in index.chunks] == [c.to_dict() for c in code_index.chunks]
    assert index.section_chunks("164") == code_index.section_chunks("164")
    assert streaming.INGEST_VERSION  # sanity: store dir is keyed by it


# ---------------------------------------------------------------------------
# DocumentIndex.section_chunks and routing
# ---------------------------------------------------------------------------


def _index(sections: list[tuple[str, str | None]], texts: list[str]) -> DocumentIndex:
    chunks = [
        replace(c, section=s, section_no=n)
        for c, (s, n) in zip(
            make_chunks([(i + 1, t) for i, t in enumerate(texts)]), sections, strict=True
        )
    ]
    emb = FakeEmbedder()
    return DocumentIndex("doc", "", chunks, emb.encode_passages([c.embed_text for c in chunks]))


def test_old_index_without_section_no_uses_the_heading_fallback():
    index = _index(
        [("§ 163. Interest", None), ("(b) Limits", None), ("§ 164. Taxes", None), ("Notes", None)],
        ["interest", "limits", "taxes", "notes"],
    )
    assert index.section_chunks("163") == [0, 1]
    assert index.section_chunks("164") == [2, 3]


def test_section_no_wins_over_headings_and_none_means_no_section():
    index = _index(
        [("(a) General rule", "164"), ("Editorial Notes", "164"), ("CHAPTER 79", None)],
        ["taxes", "notes", "table of contents; see section 164 and § 221(a)(2)"],
    )
    assert index.section_chunks("164") == [0, 1]  # the TOC chunk is not section 164
    assert index.section_chunks("221") == []


def test_section_starting_mid_chunk_is_still_found():
    index = _index(
        [("§ 163. Interest", "163"), ("Notes", "164")],
        ["tail of 163. § 164. Taxes (a) General rule", "more of 164"],
    )
    assert index.section_chunks("164") == [0, 1]
    assert index.section_chunks("163") == [0]


def test_hybrid_routing_pins_the_numbered_section(code_index):
    emb = FakeEmbedder()
    # a long code: filler sections that all look alike, one of them numbered 164
    items = [(f"§ {200 + i}. Filler", str(200 + i)) for i in range(30)]
    texts = [f"The taxpayer shall be allowed a deduction number {i}." for i in range(30)]
    items[17] = ("(a) General rule", "164")
    texts[17] = "The taxpayer shall be allowed a deduction number 17."
    index = _index(items, texts)
    cfg = RAGConfig(retrieval_mode="hybrid", candidates_per_retriever=3, top_k=3)
    out = HybridRetriever(cfg, emb, FakeReranker()).retrieve(
        ["What does section 164 allow?"], [index]
    )
    assert index.chunks[17].chunk_id in {sc.chunk.chunk_id for sc in out}
    assert np.isfinite(code_index.embeddings).all()
