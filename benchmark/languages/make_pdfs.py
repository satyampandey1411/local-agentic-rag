#!/usr/bin/env python3
"""Build the many-language benchmark PDFs from the saved Wikipedia article text.

    .venv/bin/python benchmark/languages/make_pdfs.py                  # dev set
    .venv/bin/python benchmark/languages/make_pdfs.py --heldout        # sealed held-out set
    .venv/bin/python benchmark/languages/make_pdfs.py --set heldout2   # second sealed held-out set
    .venv/bin/python benchmark/languages/make_pdfs.py --set special    # mixed-language + Chromium Arabic

Inputs : <set>/config.json            (one entry per document: language, article, page/revision id)
         <set>/sources/<doc_id>.txt   (plain-text Wikipedia extract, cut at paragraph boundaries)
Outputs: <set>/pdfs/<doc_id>.pdf       (LibreOffice text PDF + Latin footer + frozen metadata)
         <set>/pdfs/<doc_id>-scan.pdf  (image-only copy, 200 dpi grayscale) when scanned_copy is true
         <set>/documents.json          (manifest: sha256, num_pages, licence, round-trip score)

Same method as benchmark/multilingual/make_pdfs.py: text -> HTML (font per script via CSS,
``lang``/``dir`` set) -> LibreOffice headless PDF -> PyMuPDF adds footers and fixed metadata.
PyMuPDF's own insert_htmlbox writes unusable text layers for shaped Indic text, so it is
not used for text.

Every text PDF is round-tripped: ``page.get_text()`` of all pages, NFC, whitespace removed,
compared with the source (also NFC, whitespace and heading markers removed). The score is the
share of source characters found in order (difflib matching blocks). A document below
ROUNDTRIP_MIN is still written, but is flagged ``roundtrip_ok: false`` in documents.json and
validate.py refuses questions on it.

Optional config keys (used only by special/; the dev and held-out sets do not set them, so their
output is unchanged):
  para_langs     one language per paragraph (non-heading line), for a mixed-language document;
                 paragraphs whose language differs from ``lang`` get their own ``lang`` and font.
  parts          the source articles of a mixed document (lang, wiki, article_title, page_id,
                 rev_id, rev_timestamp); written to documents.json with their revision URLs.
  title, footer  override the manifest title / page footer.
  renderer       "chromium": print the HTML with headless Chromium instead of LibreOffice.
  roundtrip_min  a stricter round-trip threshold for this document.

Text is CC BY-SA 4.0 (Wikipedia contributors).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
SOFFICE = shutil.which("soffice") or shutil.which("libreoffice") or "soffice"
SCAN_DPI = 200
ROUNDTRIP_MIN = 0.95
MARGIN = 57  # ~2 cm, matches the @page margin

# lang -> (English name, HTML lang tag, direction, CSS font-family list)
DEVA = '"Noto Serif Devanagari", "Noto Sans Devanagari", serif'
LATIN = '"DejaVu Serif", "Liberation Serif", serif'
SCRIPTS = {
    "hi": ("Hindi", "hi", "ltr", DEVA),
    "mr": ("Marathi", "mr", "ltr", DEVA),
    "sa": ("Sanskrit", "sa", "ltr", DEVA),
    "ne": ("Nepali", "ne", "ltr", DEVA),
    "bh": ("Bhojpuri", "bho", "ltr", DEVA),
    "awa": ("Awadhi", "awa", "ltr", DEVA),
    "mai": ("Maithili", "mai", "ltr", DEVA),
    # Noto Nastaliq Urdu draws correctly but its text layer reads back as garbage (round-trip
    # 0.44: doubled letters, broken order), so Urdu is set in Naskh, which round-trips (0.99).
    "ur": ("Urdu", "ur", "rtl", '"Noto Naskh Arabic", serif'),
    "bn": ("Bengali", "bn", "ltr", '"Noto Serif Bengali", "Noto Sans Bengali", serif'),
    "te": ("Telugu", "te", "ltr", '"Noto Serif Telugu", "Noto Sans Telugu", serif'),
    "ta": ("Tamil", "ta", "ltr", '"Noto Serif Tamil", "Noto Sans Tamil", serif'),
    "kn": ("Kannada", "kn", "ltr", '"Noto Serif Kannada", "Noto Sans Kannada", serif'),
    "ml": ("Malayalam", "ml", "ltr", '"Noto Serif Malayalam", "Noto Sans Malayalam", serif'),
    "gu": ("Gujarati", "gu", "ltr", '"Noto Serif Gujarati", "Noto Sans Gujarati", serif'),
    "pa": ("Punjabi", "pa", "ltr", '"Noto Serif Gurmukhi", "Noto Sans Gurmukhi", serif'),
    "or": ("Odia", "or", "ltr", '"Noto Sans Oriya", serif'),
    "en": ("English", "en", "ltr", LATIN),
    "fr": ("French", "fr", "ltr", LATIN),
    "ru": ("Russian", "ru", "ltr", LATIN),
    # Arabic round-trips at 0.94 with Naskh or Sans: lines that mix Arabic with digits come back
    # in the wrong order. The doc is built and flagged roundtrip_ok=false; no questions use it.
    "ar": ("Arabic", "ar", "rtl", '"Noto Naskh Arabic", "Noto Sans Arabic", serif'),
    "zh": ("Chinese", "zh-CN", "ltr", '"Noto Serif CJK SC", "Noto Sans CJK SC", serif'),
    "ja": ("Japanese", "ja", "ltr", '"Noto Serif CJK JP", "Noto Sans CJK JP", serif'),
    # special/: alternating Hindi and English paragraphs (English paragraphs get LATIN, see para_langs)
    "hi-en": ("Hindi-English", "hi", "ltr", DEVA),
}
SETS = {"dev": ".", "heldout": "heldout", "heldout2": "heldout2", "special": "special"}
CHROMIUM = str(Path.home() / ".cache/ms-playwright/chromium-1248/chrome-linux64/chrome")
CSS = """
@page {{ size: A4; margin: 2cm; }}
body {{ font-family: {font}; font-size: 12pt; }}
h1 {{ font-size: 20pt; margin-bottom: 6pt; }}
h2 {{ font-size: 15pt; margin-top: 10pt; margin-bottom: 4pt; }}
h3 {{ font-size: 13pt; margin-top: 8pt; margin-bottom: 3pt; }}
h4 {{ font-size: 12pt; margin-top: 6pt; margin-bottom: 2pt; }}
p  {{ margin-top: 0pt; margin-bottom: 6pt; text-align: {align}; }}
"""
HEADING = re.compile(r"^(={2,5})\s*(.*?)\s*\1$")
LICENCE = ("Text: Wikipedia contributors, licensed CC BY-SA 4.0 "
           "(https://creativecommons.org/licenses/by-sa/4.0/). PDF generated by make_pdfs.py from the "
           "saved plain-text extract in sources/ (Wikipedia's own PDF renderer output is not "
           "byte-stable). Changes: plain text only, first ~2-3k characters cut at a paragraph "
           "boundary, no images/tables/infobox/references; headings kept. Any redistribution of "
           "the PDF must keep this attribution and licence.")


def fixed_meta(set_name: str) -> dict:
    return {"producer": "PyMuPDF", "creator": f"benchmark/{set_name}/make_pdfs.py",
            "creationDate": "D:20261031000000Z", "modDate": "D:20261031000000Z"}


def text_to_html(title: str, text: str, lang_tag: str, direction: str,
                 para_langs: list[str] | None = None) -> str:
    parts = [f'<body lang="{lang_tag}" dir="{direction}"><h1>{html.escape(title)}</h1>']
    langs = iter(para_langs or [])
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = HEADING.match(line)
        if m:
            level = min(len(m.group(1)), 4)
            parts.append(f"<h{level}>{html.escape(m.group(2))}</h{level}>")
        elif para_langs is not None and (pl := next(langs)) != lang_tag:
            _, pl_tag, pl_dir, pl_font = SCRIPTS[pl]
            font = pl_font.replace('"', "'")
            parts.append(f'<p lang="{pl_tag}" dir="{pl_dir}" style="font-family: {font}">{html.escape(line)}</p>')
        else:
            parts.append(f"<p>{html.escape(line)}</p>")
    parts.append("</body>")
    return "\n".join(parts)


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def roundtrip_score(source_text: str, pdf_text: str) -> float:
    """Share of source characters (whitespace and '=' heading markers removed) recovered in order."""
    src = _squash("\n".join(HEADING.sub(r"\2", ln.strip()) for ln in source_text.splitlines()))
    got = _squash(pdf_text)
    if not src:
        return 0.0
    sm = difflib.SequenceMatcher(None, src, got, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(src)


def _freeze(doc: pymupdf.Document, title: str, meta: dict) -> None:
    doc.set_metadata({**meta, "title": title})
    doc.del_xml_metadata()
    fid = "<" + "0" * 32 + ">"
    doc.xref_set_key(-1, "ID", f"[{fid}{fid}]")


def footer_for(d: dict) -> str:
    if "footer" in d:
        return d["footer"]
    name = SCRIPTS[d["lang"]][0]
    return (f"Text: {name} Wikipedia, '{d['topic_en']}' (page id {d['page_id']}, rev {d['rev_id']}), "
            "Wikipedia contributors, CC BY-SA 4.0")


def build_text_pdf(base: Path, d: dict, out_dir: Path, meta: dict) -> tuple[Path, float]:
    _, lang_tag, direction, font = SCRIPTS[d["lang"]]
    text = (base / d["source_text"]).read_text(encoding="utf-8")
    title = d["article_title"]
    css = CSS.format(font=font, align="right" if direction == "rtl" else "justify")
    page_html = (f'<!doctype html><html lang="{lang_tag}" dir="{direction}"><head><meta charset="utf-8">'
                 f"<title>{html.escape(title)}</title><style>{css}</style></head>"
                 f"{text_to_html(title, text, lang_tag, direction, d.get('para_langs'))}</html>")
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / f"{d['doc_id']}.html"
        h.write_text(page_html, encoding="utf-8")
        if d.get("renderer") == "chromium":
            # --no-sandbox: Chromium's namespace sandbox aborts (signal 6) on this machine
            subprocess.run([CHROMIUM, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                            f"--user-data-dir={td}/chromium-profile", f"--print-to-pdf={td}/{d['doc_id']}.pdf",
                            h.as_uri()],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        else:
            subprocess.run([SOFFICE, "--headless", f"-env:UserInstallation=file://{td}/lo-profile",
                            "--convert-to", "pdf:writer_pdf_Export", "--outdir", td, str(h)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        doc = pymupdf.open(Path(td) / f"{d['doc_id']}.pdf")
        body_text = "".join(pg.get_text() for pg in doc)   # before the footer is added
        n = doc.page_count
        footer = footer_for(d)
        for i, page in enumerate(doc, start=1):
            page.insert_text((MARGIN, page.rect.height - 20), f"{footer}  -  page {i}/{n}",
                             fontsize=7, color=(0.35, 0.35, 0.35))
        _freeze(doc, title, meta)
        out = out_dir / f"{d['doc_id']}.pdf"
        doc.save(out, garbage=4, deflate=True, no_new_id=True)
        doc.close()
    return out, roundtrip_score(title + "\n" + text, body_text)


def build_scanned_pdf(src_pdf: Path, out: Path, title: str, meta: dict) -> int:
    """Image-only copy of every page of ``src_pdf``; returns the page count."""
    src = pymupdf.open(src_pdf)
    out_doc = pymupdf.open()
    for page_in in src:
        pix = page_in.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
        page = out_doc.new_page(width=page_in.rect.width, height=page_in.rect.height)
        page.insert_image(page.rect, stream=pix.tobytes("png"))
    _freeze(out_doc, f"{title} (scanned, image-only)", meta)
    n = out_doc.page_count
    out_doc.save(out, garbage=4, deflate=True, no_new_id=True)
    out_doc.close()
    src.close()
    return n


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heldout", action="store_true", help="build heldout/ instead of the dev set (= --set heldout)")
    ap.add_argument("--set", choices=sorted(SETS), default="dev", help="which set to build (default: dev)")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    which = "heldout" if args.heldout else args.set
    base = HERE / SETS[which] if which != "dev" else HERE
    set_name = f"languages/{SETS[which]}" if which != "dev" else "languages"
    meta = fixed_meta("languages")
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    out_dir = base / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        built = list(ex.map(lambda d: build_text_pdf(base, d, out_dir, meta), cfg))

    manifest = []
    bad = []
    for d, (pdf, score) in zip(cfg, built, strict=True):
        name = SCRIPTS[d["lang"]][0]
        n = pymupdf.open(pdf).page_count
        ok = score >= d.get("roundtrip_min", ROUNDTRIP_MIN)
        if not ok:
            bad.append(d["doc_id"])
        url = f"https://{d['wiki']}/w/index.php?curid={d['page_id']}&oldid={d['rev_id']}"
        entry = {
            "doc_id": d["doc_id"],
            "title": d.get("title") or f"{d['article_title']} ({d['topic_en']}) - {name} Wikipedia article, "
                                       f"revision {d['rev_id']} ({d['rev_timestamp'][:10]})",
            "url": url, "sha256": sha256(pdf), "num_pages": n,
            "doc_type": f"encyclopedia_article_{name.lower()}",
            "source_license_note": LICENCE, "language": d["lang"], "source_text": d["source_text"],
            "page_id": d["page_id"], "rev_id": d["rev_id"],
            "roundtrip_score": round(score, 4), "roundtrip_ok": ok,
        }
        for k in ("roundtrip_min", "renderer", "para_langs"):
            if k in d:
                entry[k] = d[k]
        if "parts" in d:
            entry["parts"] = [{**p, "url": f"https://{p['wiki']}/w/index.php?curid={p['page_id']}&oldid={p['rev_id']}"}
                              for p in d["parts"]]
        manifest.append(entry)
        print(f"{pdf.name:32s} {n} page(s)  round-trip {score:.3f} {'OK' if ok else 'FAIL'}")
        if d.get("scanned_copy"):
            sid = f"{d['doc_id']}-scan"
            spdf = out_dir / f"{sid}.pdf"
            sn = build_scanned_pdf(pdf, spdf, d["article_title"], meta)
            chars = sum(len(pg.get_text().strip()) for pg in pymupdf.open(spdf))
            manifest.append({
                "doc_id": sid,
                "title": f"{d['article_title']} - image-only copy of {d['doc_id']} "
                         f"(all {sn} page(s), {SCAN_DPI} dpi grayscale, no text layer)",
                "url": url, "sha256": sha256(spdf), "num_pages": sn,
                "doc_type": f"scanned_image_only_{name.lower()}",
                "source_license_note": f"Derived from {d['doc_id']} (same CC BY-SA 4.0 attribution). "
                                       "Built by make_pdfs.py; for OCR testing - get_text() returns nothing.",
                "language": d["lang"], "derived_from": d["doc_id"], "derived_pages": list(range(1, sn + 1)),
            })
            print(f"{spdf.name:32s} {sn} page(s)  text-layer chars = {chars}")
    (base / "documents.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"{set_name}: {len(manifest)} documents; round-trip failures: {bad or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
