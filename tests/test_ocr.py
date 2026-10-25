"""OCR path: image-only pages go through the tesseract CLI (mocked here) with language fallback."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pymupdf
import pytest

from local_agentic_rag.ingest import pdf as pdfmod
from local_agentic_rag.ingest.pdf import _ocr_text_to_paragraphs, parse_pdf


@pytest.fixture
def scanned_pdf(tmp_path):
    """A 2-page PDF whose pages are images only (no text layer)."""
    src = pymupdf.open()
    page = src.new_page()
    page.insert_text((72, 100), "Scanned page text", fontsize=24)
    pix = page.get_pixmap(dpi=100)
    out = pymupdf.open()
    for _ in range(2):
        p = out.new_page()
        p.insert_image(p.rect, pixmap=pix)
    path = tmp_path / "scan.pdf"
    out.save(path)
    return path


def _fake_tesseract(outputs: dict[str, str], calls: list):
    def run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        lang = cmd[cmd.index("-l") + 1]
        if lang in outputs:
            return SimpleNamespace(returncode=0, stdout=outputs[lang], stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr=f"Failed loading language '{lang}'")

    return run


def test_paragraph_splitting_and_nfc():
    text = "ताजमहल आगरा में\nस्थित है।\n\n\nदूसरा अनुच्छेद 35 मीटर\n"
    assert _ocr_text_to_paragraphs(text) == ["ताजमहल आगरा में स्थित है।", "दूसरा अनुच्छेद 35 मीटर"]


def test_image_only_pages_use_tesseract_cli(scanned_pdf, monkeypatch):
    calls: list = []
    monkeypatch.setattr(pdfmod.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_tesseract({"eng+hin": "मुख्य गुम्बद की ऊँचाई लगभग 35 मीटर है।\n\nSecond para"}, calls),
    )
    doc = parse_pdf(scanned_pdf, ocr="auto", ocr_language="eng+hin")
    assert all(p.ocr_applied for p in doc.pages)
    assert "35 मीटर" in doc.pages[0].text
    assert all(
        b.kind == "paragraph" for p in doc.pages for b in p.blocks
    )  # OCR noise never becomes headings
    assert calls and calls[0][-1] == "eng+hin"


def test_missing_language_pack_falls_back_to_english(scanned_pdf, monkeypatch):
    calls: list = []
    monkeypatch.setattr(pdfmod.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        subprocess, "run", _fake_tesseract({"eng": "English only text here"}, calls)
    )
    doc = parse_pdf(scanned_pdf, ocr="auto", ocr_language="eng+xyz")
    assert doc.pages[0].text == "English only text here"
    assert [c[-1] for c in calls[:2]] == ["eng+xyz", "eng"]


def test_ocr_off_leaves_scanned_pdf_unreadable(scanned_pdf):
    with pytest.raises(pdfmod.PDFParseError, match="scanned"):
        parse_pdf(scanned_pdf, ocr="off")
