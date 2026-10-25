"""
PDF parsing with PyMuPDF.

Why PyMuPDF instead of pdfminer (original prototype) or unstructured (reference):
    * 10-50x faster than pdfminer, no ML layout model to download (unlike
      unstructured hi_res / Marker), so ingestion of a 100-page PDF takes
      seconds on the free Colab CPU.
    * Exposes span-level font sizes and bounding boxes, which we use for
      heading detection and two-column reading order.
    * Built-in ruled-table detection (``page.find_tables``) → Markdown, so
      table cells keep their row/column structure instead of being flattened.
    * Optional Tesseract OCR for image-only (scanned) pages.

Pipeline per page:
    raw blocks → drop blocks inside tables → order (column-aware) → dehyphenate
    → classify heading/paragraph → (document-level) strip repeated headers/footers
"""

from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from local_agentic_rag.ingest.tables import (
    is_garbled_rows,
    serialize_table_words,
    split_stacked_cells,
    vertical_rules,
)
from local_agentic_rag.schema import PageContent, ParsedDocument, TextBlock

logger = logging.getLogger(__name__)

#: Top/bottom fraction of the page treated as header/footer territory.
MARGIN_BAND = 0.08
#: A margin line is boilerplate if it repeats on at least this share of pages.
REPEAT_FRACTION = 0.4
#: Pages with fewer visible characters than this are considered image-only.
MIN_TEXT_CHARS = 25

_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?[\divxlc]+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
_SECTION_NUMBER_RE = re.compile(r"^(\d+(\.\d+)*|[A-Z]\.|[IVX]+\.)\s+\S")


class PDFParseError(RuntimeError):
    """Raised for unreadable, encrypted or empty PDFs (user-facing message)."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_pdf(
    pdf_path: str | Path,
    doc_id: str | None = None,
    *,
    extract_tables: bool = True,
    strip_headers_footers: bool = True,
    ocr: str = "auto",
    ocr_language: str = "eng",
) -> ParsedDocument:
    """Parse a PDF into cleaned, reading-ordered page content.

    Raises
    ------
    FileNotFoundError
        Path does not exist.
    PDFParseError
        File is not a PDF, is password-protected, or has no pages.
    """
    import pymupdf

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {str(path)!r}")

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # pymupdf.FileDataError and friends
        raise PDFParseError(f"Could not open {path.name!r} as a PDF: {exc}") from exc

    try:
        if not doc.is_pdf:
            raise PDFParseError(f"{path.name!r} is not a PDF file.")
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFParseError(
                f"{path.name!r} is password-protected; please upload an unlocked copy."
            )
        if doc.page_count == 0:
            raise PDFParseError(f"{path.name!r} contains no pages.")

        ocr_available = ocr == "auto" and shutil.which("tesseract") is not None
        warnings: list[str] = []
        tables_by_page: dict[int, list[tuple[str, tuple]]] = {}
        if extract_tables:
            candidates = [i for i in range(doc.page_count) if _may_have_table(doc[i])]
            tables_by_page = _find_tables(str(path), candidates)
        raw_pages: list[_RawPage] = []
        for index in range(doc.page_count):
            page = doc[index]
            try:
                raw_pages.append(
                    _extract_page(
                        page, index + 1, tables_by_page.get(index, []), ocr_available, ocr_language
                    )
                )
            except Exception as exc:  # one bad page must not kill the document
                logger.warning("Page %d failed to parse: %s", index + 1, exc)
                warnings.append(f"page {index + 1}: parse error ({exc})")
                raw_pages.append(_RawPage(page=index + 1, blocks=[], tables=[], height=1.0))

        if strip_headers_footers:
            _strip_repeated_margins(raw_pages)

        font_title = _title_from_fonts(raw_pages[0]) if raw_pages else ""
        body_size = _body_font_size(raw_pages)
        pages = [_finalise_page(rp, body_size) for rp in raw_pages]

        needs_ocr = [p.page for p in pages if p.needs_ocr]
        if needs_ocr:
            warnings.append(
                f"{len(needs_ocr)} page(s) look scanned/image-only and have no extractable text"
                + ("" if ocr_available else " (install tesseract-ocr to enable OCR)")
                + f": {needs_ocr[:20]}"
            )
        if all(not p.blocks and not p.tables for p in pages):
            raise PDFParseError(
                f"No text could be extracted from {path.name!r}. It is probably a scanned "
                "document; install tesseract-ocr (apt-get install tesseract-ocr) to enable OCR."
            )

        title = _guess_title(doc.metadata or {}, font_title, path)
        return ParsedDocument(
            doc_id=doc_id or _slugify(path.stem),
            title=title,
            source_path=str(path),
            sha256=sha256_file(path),
            num_pages=doc.page_count,
            pages=pages,
            warnings=warnings,
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------


class _RawBlock:
    __slots__ = ("text", "bbox", "max_size", "bold", "n_lines")

    def __init__(self, text, bbox, max_size, bold, n_lines):
        self.text: str = text
        self.bbox: tuple[float, float, float, float] = bbox
        self.max_size: float = max_size
        self.bold: bool = bold
        self.n_lines: int = n_lines


class _RawPage:
    __slots__ = ("page", "blocks", "tables", "height", "ocr_applied", "needs_ocr", "sizes")

    def __init__(self, page, blocks, tables, height, ocr_applied=False, needs_ocr=False):
        self.page: int = page
        self.blocks: list[_RawBlock] = blocks
        self.tables: list[str] = tables
        self.height: float = height
        self.ocr_applied: bool = ocr_applied
        self.needs_ocr: bool = needs_ocr
        self.sizes: list[tuple[float, int]] = []  # (font size, char count) samples


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------
#
# ``find_tables`` costs ~1-2 s per page, so it only runs on pages that have
# ruling lines (pages with none cannot contain a ruled table; pages with
# thousands are vector figures), and in parallel processes for long documents.

MIN_TABLE_LINES = 4
MAX_TABLE_LINES = 3000


def _may_have_table(page) -> bool:
    try:
        n = sum(
            1 for d in page.get_cdrawings() for it in d.get("items", []) if it[0] in ("l", "re")
        )
    except Exception:
        return True
    return MIN_TABLE_LINES <= n <= MAX_TABLE_LINES


def _table_rows(page, tab) -> list[list[str]]:
    """Cell text rebuilt from the words whose centre lies inside each cell box.

    We avoid ``tab.to_markdown()`` — it HTML-escapes cell text ("-1.2%" →
    "&#45;1.2%") and can pull the paragraph *above* the table in as an
    "external" header — and ``tab.extract()``, which splits glyphs whose boxes
    sit slightly off the baseline ("4.5%" → "45%\n.").
    """
    words = page.get_text("words", clip=tab.bbox)
    rows: list[list[str]] = []
    for row in tab.rows:
        cell_words: list[list[tuple]] = []
        for box in row.cells:
            if box is None:
                cell_words.append([])
                continue
            x0, y0, x1, y1 = box
            cell_words.append(
                [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
            )
        # Booktabs tables (no rules between body rows) come back with several visual
        # rows stacked in one detected row; split them so each value keeps its label.
        for part in split_stacked_cells(cell_words):
            cells = []
            for inside in part:
                inside.sort(key=lambda w: (w[5], w[6], w[7]))  # block, line, word order
                cells.append(html.unescape(" ".join(w[4] for w in inside)).strip())
            if any(cells):
                rows.append(cells)
    return rows


def _rows_to_markdown(rows: list[list[str]]) -> str:
    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(re.sub(r"\s+", " ", c).replace("|", "/") for c in cells) + " |"

    header, body = rows[0], rows[1:]
    return "\n".join([fmt(header), "|" + "---|" * len(header), *(fmt(r) for r in body)])


def _tables_on_page(page) -> list[tuple[str, tuple]]:
    found = []
    for tab in page.find_tables().tables:
        rows = _table_rows(page, tab)
        # Tiny "tables" are usually boxed callouts; keep them as text instead.
        if (
            len(rows) < 2
            or max(len(r) for r in rows) < 2
            or sum(len("".join(r)) for r in rows) <= 30
        ):
            continue
        bbox = tuple(tab.bbox)
        if is_garbled_rows(rows):
            # Whitespace-aligned statistical table: rebuild rows from word positions.
            words = page.get_text("words", clip=bbox)
            above = page.get_text("text", clip=(0, max(0, bbox[1] - 60), page.rect.width, bbox[1]))
            title = " ".join(ln.strip() for ln in above.splitlines() if ln.strip())[-200:]
            lines = serialize_table_words(words, title=title, vrules=vertical_rules(page, bbox))
            if not lines:
                continue  # leave the region to normal text extraction
            found.append(("\n".join(lines), bbox))
        else:
            found.append((_rows_to_markdown(rows), bbox))
    return found


def _tables_worker(args: tuple[str, list[int]]) -> dict[int, list[tuple[str, tuple]]]:
    import pymupdf

    path, indices = args
    out: dict[int, list[tuple[str, tuple]]] = {}
    with pymupdf.open(path) as doc:
        for i in indices:
            try:
                out[i] = _tables_on_page(doc[i])
            except Exception as exc:  # a broken page must not lose the others
                logger.debug("find_tables failed on page %d: %s", i + 1, exc)
                out[i] = []
    return out


def _find_tables(path: str, indices: list[int]) -> dict[int, list[tuple[str, tuple]]]:
    if not indices:
        return {}
    workers = min(len(indices) // 4, os.cpu_count() or 1, 8)
    if workers < 2:
        return _tables_worker((path, indices))
    shards = [indices[w::workers] for w in range(workers)]
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        result: dict[int, list[tuple[str, tuple]]] = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for part in pool.map(_tables_worker, [(path, s) for s in shards]):
                result.update(part)
        return result
    except Exception as exc:  # e.g. restricted environments without fork
        logger.debug("parallel table detection unavailable (%s); running serially", exc)
        return _tables_worker((path, indices))


def _extract_page(
    page, page_no: int, tables: list[tuple[str, tuple]], ocr_available: bool, lang: str
) -> _RawPage:
    width, height = page.rect.width, page.rect.height
    tables_md = [md for md, _ in tables]
    table_bboxes = [bbox for _, bbox in tables]

    textpage = None
    ocr_applied = False
    plain = page.get_text("text").strip()
    image_only = len(plain) < MIN_TEXT_CHARS and bool(page.get_images(full=False))
    if image_only and ocr_available:
        # Preferred: Tesseract CLI on a 300-dpi render.  Measured on a scanned Hindi page:
        # 99% word recall vs 70% for PyMuPDF's OCR text layer, which splits Devanagari
        # words at vowel signs and merges neighbours.
        paragraphs = _ocr_cli_paragraphs(page, page_no, lang)
        if paragraphs:
            raw = _RawPage(
                page=page_no, blocks=[], tables=tables_md, height=height, ocr_applied=True
            )
            mid = (0.0, height * 0.5, width, height * 0.5)  # outside header/footer bands
            raw.blocks = [
                _RawBlock(
                    text=p, bbox=mid, max_size=0.0, bold=False, n_lines=p.count(" ") // 12 + 1
                )
                for p in paragraphs
            ]
            return raw
        textpage = _ocr_textpage(page, page_no, lang)
        ocr_applied = textpage is not None

    data = page.get_text("dict", textpage=textpage) if textpage else page.get_text("dict")
    raw = _RawPage(
        page=page_no, blocks=[], tables=tables_md, height=height, ocr_applied=ocr_applied
    )

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = tuple(block["bbox"])
        if any(_overlap_ratio(bbox, tb) > 0.5 for tb in table_bboxes):
            continue
        lines_text: list[str] = []
        sizes: list[float] = []
        bold_chars = 0
        total_chars = 0
        for line in block.get("lines", []):
            if line.get("dir", (1, 0))[0] < 0.5:  # skip rotated / vertical text
                continue
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            lines_text.append(text)
            for s in spans:
                n = len(s.get("text", "").strip())
                if n == 0:
                    continue
                sizes.append(float(s.get("size", 0)))
                raw.sizes.append((round(float(s.get("size", 0)), 1), n))
                total_chars += n
                if s.get("flags", 0) & 16 or "bold" in s.get("font", "").lower():
                    bold_chars += n
        if not lines_text:
            continue
        raw.blocks.append(
            _RawBlock(
                text=_join_lines(lines_text),
                bbox=bbox,
                max_size=max(sizes) if sizes else 0.0,
                bold=total_chars > 0 and bold_chars / total_chars > 0.8,
                n_lines=len(lines_text),
            )
        )

    raw.blocks = _reading_order(raw.blocks, width)
    if image_only and not raw.blocks and not ocr_applied:
        raw.needs_ocr = True
    return raw


def _tessdata() -> str | None:
    """Tesseract's language-data folder (PyMuPDF needs it explicitly on many systems)."""
    env = os.environ.get("TESSDATA_PREFIX")
    if env:
        return env
    try:
        import pymupdf

        return pymupdf.get_tessdata()
    except Exception:
        for candidate in (
            "/usr/share/tesseract-ocr/5/tessdata",
            "/usr/share/tesseract-ocr/4.00/tessdata",
        ):
            if os.path.isdir(candidate):
                return candidate
    return None


OCR_DPI = 300


def _ocr_cli_paragraphs(page, page_no: int, lang: str) -> list[str]:
    """OCR a page with the ``tesseract`` CLI; returns paragraphs ([] if unavailable/failed).

    Retries with English only when a requested language pack is missing.
    """
    import subprocess
    import tempfile

    if shutil.which("tesseract") is None:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        image = os.path.join(tmp, f"page{page_no}.png")
        page.get_pixmap(dpi=OCR_DPI).save(image)
        for language in dict.fromkeys([lang, "eng"]):
            try:
                result = subprocess.run(
                    ["tesseract", image, "stdout", "-l", language],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("tesseract failed on page %d: %s", page_no, exc)
                return []
            if result.returncode == 0 and result.stdout.strip():
                return _ocr_text_to_paragraphs(result.stdout)
            logger.warning(
                "tesseract (%s) failed on page %d: %s", language, page_no, result.stderr[-200:]
            )
    return []


def _ocr_text_to_paragraphs(text: str) -> list[str]:
    """Blank-line separated OCR output → whitespace-normalised paragraphs."""
    paragraphs = []
    for chunk in re.split(r"\n\s*\n", unicodedata.normalize("NFC", text)):
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if lines:
            paragraphs.append(_join_lines(lines))
    return [p for p in paragraphs if len(p) >= 2]


def _ocr_textpage(page, page_no: int, lang: str):
    """OCR one page; if a requested language pack is missing, retry with English only."""
    tessdata = _tessdata()
    for language in dict.fromkeys([lang, "eng"]):
        try:
            return page.get_textpage_ocr(language=language, dpi=200, full=True, tessdata=tessdata)
        except Exception as exc:
            logger.warning("OCR (%s) failed on page %d: %s", language, page_no, exc)
    return None


def _join_lines(lines: list[str]) -> str:
    """Join PDF lines into a paragraph, repairing end-of-line hyphenation."""
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return re.sub(r"\s+", " ", out).strip()


def _overlap_ratio(a, b) -> float:
    """Fraction of box ``a`` covered by box ``b``."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return (x1 - x0) * (y1 - y0) / area


def _reading_order(blocks: list[_RawBlock], page_width: float) -> list[_RawBlock]:
    """Column-aware reading order.

    PyMuPDF's naive ``sort=True`` orders by y then x, which interleaves the two
    columns of an academic paper line by line.  Instead: classify blocks as
    left-column, right-column or full-width.  Full-width blocks (title,
    abstract, wide figures) act as separators; between two separators we read
    the whole left column top-to-bottom, then the right column.
    """
    if len(blocks) < 4:
        return sorted(blocks, key=lambda b: (round(b.bbox[1]), b.bbox[0]))

    mid = page_width / 2
    tol = page_width * 0.04
    left, right, full = [], [], []
    for b in blocks:
        x0, _, x1, _ = b.bbox
        if x1 <= mid + tol:
            left.append(b)
        elif x0 >= mid - tol:
            right.append(b)
        else:
            full.append(b)

    if len(left) < 2 or len(right) < 2:
        return sorted(blocks, key=lambda b: (round(b.bbox[1]), b.bbox[0]))

    full.sort(key=lambda b: b.bbox[1])
    bounds = [float("-inf")] + [b.bbox[1] for b in full] + [float("inf")]
    ordered: list[_RawBlock] = []
    for i in range(len(full) + 1):
        lo, hi = bounds[i], bounds[i + 1]

        def in_band(b: _RawBlock, lo: float = lo, hi: float = hi) -> bool:
            cy = (b.bbox[1] + b.bbox[3]) / 2
            return lo <= cy < hi

        ordered += sorted(filter(in_band, left), key=lambda b: b.bbox[1])
        ordered += sorted(filter(in_band, right), key=lambda b: b.bbox[1])
        if i < len(full):
            ordered.append(full[i])
    return ordered


# ---------------------------------------------------------------------------
# Document-level cleaning
# ---------------------------------------------------------------------------


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _strip_repeated_margins(pages: list[_RawPage]) -> None:
    """Remove running headers/footers and bare page numbers."""

    def in_margin(b: _RawBlock, height: float) -> bool:
        return b.bbox[3] <= height * MARGIN_BAND or b.bbox[1] >= height * (1 - MARGIN_BAND)

    counts: Counter[str] = Counter()
    for p in pages:
        keys = {
            _margin_key(b.text) for b in p.blocks if in_margin(b, p.height) and len(b.text) < 200
        }
        counts.update(keys)

    threshold = max(3, int(REPEAT_FRACTION * len(pages)))
    repeated = {k for k, c in counts.items() if c >= threshold}
    for p in pages:
        p.blocks = [
            b
            for b in p.blocks
            if not (
                in_margin(b, p.height)
                and (_margin_key(b.text) in repeated or _PAGE_NUMBER_RE.match(b.text.strip()))
            )
        ]


def _body_font_size(pages: list[_RawPage]) -> float:
    counter: Counter[float] = Counter()
    for p in pages:
        for size, n in p.sizes:
            counter[size] += n
    return counter.most_common(1)[0][0] if counter else 10.0


def _is_heading(b: _RawBlock, body_size: float) -> bool:
    text = b.text.strip()
    if len(text) > 140 or b.n_lines > 3 or len(text) < 2:
        return False
    if text.endswith((".", ",", ";")) and not _SECTION_NUMBER_RE.match(text):
        return False
    if not any(c.isalpha() for c in text):
        return False
    larger = b.max_size >= body_size * 1.15
    return larger or (b.bold and len(text) < 100)


def _finalise_page(rp: _RawPage, body_size: float) -> PageContent:
    blocks = [
        TextBlock(
            text=b.text,
            kind="heading" if _is_heading(b, body_size) else "paragraph",
            bbox=b.bbox,
        )
        for b in rp.blocks
    ]
    return PageContent(
        page=rp.page,
        blocks=blocks,
        tables=rp.tables,
        ocr_applied=rp.ocr_applied,
        needs_ocr=rp.needs_ocr,
    )


def _title_from_fonts(first_page: _RawPage) -> str:
    """Largest-font text on page 1 (consecutive blocks of that size are joined)."""
    blocks = [b for b in first_page.blocks if len(b.text) > 3 and any(c.isalpha() for c in b.text)]
    if not blocks:
        return ""
    top = max(b.max_size for b in blocks)
    parts = [b.text for b in blocks if b.max_size >= top - 0.5]
    return " ".join(parts)[:200]


def _guess_title(metadata: dict, font_title: str, path: Path) -> str:
    meta_title = (metadata.get("title") or "").strip()
    junk = ("untitled", "microsoft word", ".doc", ".pdf", "slide 1")
    if len(meta_title) > 4 and not any(j in meta_title.lower() for j in junk):
        return meta_title
    if len(font_title) > 6:
        return font_title
    return path.stem.replace("_", " ").replace("-", " ")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:60] or "document"
