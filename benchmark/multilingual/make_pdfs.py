#!/usr/bin/env python3
"""Build the multilingual benchmark PDFs from the saved Wikipedia article text.

    .venv/bin/python benchmark/multilingual/make_pdfs.py

Inputs : benchmark/multilingual/sources/*.txt  (plain-text Wikipedia extracts, committed)
Outputs: benchmark/multilingual/pdfs/hi-tajmahal.pdf
         benchmark/multilingual/pdfs/mr-shaniwarwada.pdf
         benchmark/multilingual/pdfs/hi-scanned.pdf   (image-only, no text layer)

Why generate instead of downloading Wikipedia's own PDF render
(``/api/rest_v1/page/pdf/<title>``)? That endpoint re-renders on every request
(new timestamps -> new SHA-256), the article changes over time, and its Skia
text layer duplicates Devanagari vowel signs on extraction (e.g. "सम्रााट").
PyMuPDF's own Story/insert_htmlbox shapes Devanagari correctly on screen but
(PyMuPDF 1.26/2.0) writes no usable ToUnicode for the shaped glyphs, so
``get_text()`` returns garbage ("उǴर Śदेश"). LibreOffice headless export does
round-trip, so it renders the text layer; PyMuPDF only adds footers/metadata
and builds the image-only copy.

Requires: LibreOffice (``soffice`` on PATH) and the Noto Serif/Sans Devanagari
fonts (Debian/Ubuntu: ``fonts-noto-core``). Different LibreOffice/font versions
can change line breaks, page breaks and therefore the SHA-256; validate.py
reports a hash mismatch and the evidence-page check tells you whether the
ground truth still holds.

Text is CC BY-SA 4.0 (Wikipedia contributors); see documents.json for the
exact article revision each source file was taken from.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
OUT = HERE / "pdfs"

DOCS = {
    # doc_id: (source file, lang, footer attribution)
    "hi-tajmahal": ("hi-tajmahal.txt", "hi",
                    "Text: Hindi Wikipedia, 'Taj Mahal' (page id 5804, rev 6620327), "
                    "Wikipedia contributors, CC BY-SA 4.0"),
    "mr-shaniwarwada": ("mr-shaniwarwada.txt", "mr",
                        "Text: Marathi Wikipedia, 'Shaniwar Wada' (page id 4954, rev 2705147), "
                        "Wikipedia contributors, CC BY-SA 4.0"),
}
TITLES = {"hi-tajmahal": "ताजमहल", "mr-shaniwarwada": "शनिवार वाडा"}
SOFFICE = shutil.which("soffice") or shutil.which("libreoffice") or "soffice"
SCANNED_ID = "hi-scanned"
SCANNED_FROM = "hi-tajmahal"
SCANNED_PAGES = [1, 2]        # 1-indexed pages of the source PDF to rasterise
SCANNED_DPI = 200

MARGIN = 57                    # ~2 cm, matches the @page margin
CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: "Noto Serif Devanagari", "Noto Sans Devanagari", serif; font-size: 12pt; }
h1 { font-size: 20pt; margin-bottom: 6pt; }
h2 { font-size: 15pt; margin-top: 10pt; margin-bottom: 4pt; }
h3 { font-size: 13pt; margin-top: 8pt; margin-bottom: 3pt; }
h4 { font-size: 12pt; margin-top: 6pt; margin-bottom: 2pt; }
p  { margin-top: 0pt; margin-bottom: 6pt; text-align: justify; }
"""
HEADING = re.compile(r"^(={2,5})\s*(.*?)\s*\1$")
FIXED_META = {"producer": "PyMuPDF", "creator": "benchmark/multilingual/make_pdfs.py",
              "creationDate": "D:20261025000000Z", "modDate": "D:20261025000000Z"}


def text_to_html(title: str, text: str, lang: str) -> str:
    parts = [f'<body lang="{lang}"><h1>{html.escape(title)}</h1>']
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = HEADING.match(line)
        if m:
            level = min(len(m.group(1)), 4)          # == -> h2, === -> h3, ====+ -> h4
            parts.append(f"<h{level}>{html.escape(m.group(2))}</h{level}>")
        else:
            parts.append(f"<p>{html.escape(line)}</p>")
    parts.append("</body>")
    return "\n".join(parts)


def build_text_pdf(doc_id: str, src: str, lang: str, footer: str) -> Path:
    """Saved text -> HTML -> LibreOffice (headless) PDF -> footers + fixed metadata."""
    text = (SRC / src).read_text(encoding="utf-8")
    title = TITLES[doc_id]
    body = text_to_html(title, text, lang)
    page_html = ('<!doctype html><html><head><meta charset="utf-8">'
                 f'<title>{html.escape(title)}</title><style>{CSS}</style></head>{body}</html>')
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / f"{doc_id}.html"
        h.write_text(page_html, encoding="utf-8")
        subprocess.run([SOFFICE, "--headless", f"-env:UserInstallation=file://{td}/lo-profile",
                        "--convert-to", "pdf:writer_pdf_Export", "--outdir", td, str(h)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        doc = pymupdf.open(Path(td) / f"{doc_id}.pdf")
        n = doc.page_count
        for i, page in enumerate(doc, start=1):
            # Latin-only footer (Helvetica): never interferes with Devanagari text extraction.
            page.insert_text((MARGIN, page.rect.height - 20), f"{footer}  -  page {i}/{n}",
                             fontsize=7, color=(0.35, 0.35, 0.35))
        _freeze(doc, title)
        out = OUT / f"{doc_id}.pdf"
        doc.save(out, garbage=4, deflate=True, no_new_id=True)
        doc.close()
    return out


def _freeze(doc: pymupdf.Document, title: str) -> None:
    """Fixed metadata + fixed trailer /ID so rebuilding gives the same bytes where possible."""
    doc.set_metadata({**FIXED_META, "title": title})
    doc.del_xml_metadata()
    fid = "<" + "0" * 32 + ">"
    doc.xref_set_key(-1, "ID", f"[{fid}{fid}]")


def build_scanned_pdf() -> Path:
    """Image-only PDF: rasterise pages of the Hindi PDF; no text layer at all."""
    src = pymupdf.open(OUT / f"{SCANNED_FROM}.pdf")
    out_doc = pymupdf.open()
    for pno in SCANNED_PAGES:
        pix = src[pno - 1].get_pixmap(dpi=SCANNED_DPI, colorspace=pymupdf.csGRAY)
        page = out_doc.new_page(width=src[pno - 1].rect.width, height=src[pno - 1].rect.height)
        page.insert_image(page.rect, stream=pix.tobytes("png"))
    _freeze(out_doc, "ताजमहल (scanned, image-only)")
    out = OUT / f"{SCANNED_ID}.pdf"
    out_doc.save(out, garbage=4, deflate=True, no_new_id=True)
    out_doc.close()
    src.close()
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for doc_id, (src, lang, footer) in DOCS.items():
        p = build_text_pdf(doc_id, src, lang, footer)
        d = pymupdf.open(p)
        print(f"{p.name}: {d.page_count} pages")
        d.close()
    p = build_scanned_pdf()
    d = pymupdf.open(p)
    chars = sum(len(pg.get_text().strip()) for pg in d)
    print(f"{p.name}: {d.page_count} pages, text-layer chars = {chars}")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
