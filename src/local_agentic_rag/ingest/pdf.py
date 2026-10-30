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
#: A margin line is boilerplate if it repeats on at least this share of pages...
REPEAT_FRACTION = 0.4
#: ...or on at least this many consecutive pages (per-chapter running headers).
LOCAL_REPEAT_RUN = 3
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


def open_pdf(pdf_path: str | Path):
    """Open and validate a PDF; the caller closes the returned ``pymupdf.Document``.

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
        # The library's message names the stored file's full path; keep it out of the UI.
        logger.warning("pymupdf could not open %s: %s", path, exc)
        raise PDFParseError(f"Could not open {path.name!r} as a PDF; the file may be damaged.") from exc
    try:
        if not doc.is_pdf:
            raise PDFParseError(f"{path.name!r} is not a PDF file.")
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFParseError(
                f"{path.name!r} is password-protected; please upload an unlocked copy."
            )
        if doc.page_count == 0:
            raise PDFParseError(f"{path.name!r} contains no pages.")
    except BaseException:
        doc.close()
        raise
    return doc


def ocr_available(ocr: str) -> bool:
    return ocr == "auto" and shutil.which("tesseract") is not None


def parse_pdf(
    pdf_path: str | Path,
    doc_id: str | None = None,
    *,
    extract_tables: bool = True,
    strip_headers_footers: bool = True,
    ocr: str = "auto",
    ocr_language: str = "eng",
    workers: int | None = None,
) -> ParsedDocument:
    """Parse a PDF into cleaned, reading-ordered page content, all in memory.

    Long documents go through :mod:`local_agentic_rag.ingest.streaming` instead, which
    runs the same two passes page batch by page batch with checkpoints on disk.

    Raises
    ------
    FileNotFoundError
        Path does not exist.
    PDFParseError
        File is not a PDF, is password-protected, has no pages, or has no text.
    """
    path = Path(pdf_path)
    with open_pdf(path) as doc:
        num_pages, metadata = doc.page_count, doc.metadata or {}
    use_ocr = ocr_available(ocr)
    raw_pages: list[RawPage] = []
    for batch in iter_raw_batches(
        path,
        0,
        num_pages,
        extract_tables=extract_tables,
        use_ocr=use_ocr,
        ocr_language=ocr_language,
        workers=workers,
    ):
        raw_pages.extend(RawPage.from_dict(d) for d in batch)
    stats = PageStats()
    for rp in raw_pages:
        stats.add(rp)
    finaliser = PageFinaliser(stats, strip_headers_footers)
    pages = [finaliser(rp) for rp in raw_pages]
    warnings = [rp.warning for rp in raw_pages if rp.warning]
    check_parse_failures(warnings, num_pages, path.name)
    warnings.extend(finaliser.summary_warnings(use_ocr))
    if not finaliser.has_content:
        raise PDFParseError(no_text_message(path.name))
    return ParsedDocument(
        doc_id=doc_id or _slugify(path.stem),
        title=_guess_title(metadata, finaliser.font_title, path),
        source_path=str(path),
        sha256=sha256_file(path),
        num_pages=num_pages,
        pages=pages,
        warnings=warnings,
    )


def check_parse_failures(page_errors: list[str], num_pages: int, name: str) -> None:
    """Most pages failing is a broken environment or file, not "a scanned PDF": say so."""
    if num_pages >= 4 and len(page_errors) > num_pages // 2:
        raise PDFParseError(
            f"{len(page_errors)} of {num_pages} pages of {name!r} could not be parsed; "
            f"first error: {page_errors[0]}"
        )


def no_text_message(name: str) -> str:
    return (
        f"No text could be extracted from {name!r}. It is probably a scanned "
        "document; install tesseract-ocr (apt-get install tesseract-ocr) to enable OCR."
    )


# ---------------------------------------------------------------------------
# Pass 1: raw page extraction (parallel, page batches)
# ---------------------------------------------------------------------------
#
# Each page is extracted independently (tables, OCR, text blocks), so page ranges
# run in separate processes.  Results come back in page order with a bounded
# number of batches in flight, so memory does not grow with the document.

#: Pages per worker task.  Small enough for frequent checkpoints and even load.
BATCH_PAGES = 16
#: Below this many pages a process pool costs more than it saves.
MIN_PARALLEL_PAGES = 24
MAX_WORKERS = 8


def default_workers() -> int:
    env = os.environ.get("LOCAL_RAG_PARSE_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(os.cpu_count() or 1, MAX_WORKERS))


def extract_raw_pages(
    path: str, start: int, stop: int, extract_tables: bool, use_ocr: bool, ocr_language: str
) -> list[dict]:
    """Raw pages ``start..stop-1`` (0-based) as JSON-serialisable dicts (worker entry point)."""
    import pymupdf

    with pymupdf.open(path) as doc:
        return [
            extract_raw_page(doc, index, extract_tables, use_ocr, ocr_language)
            for index in range(start, stop)
        ]


def extract_raw_page(
    doc, index: int, extract_tables: bool, use_ocr: bool, ocr_language: str
) -> dict:
    """One raw page (0-based ``index``) of an open document, as a JSON-serialisable dict."""
    tables: list[tuple[str, tuple]] = []
    if extract_tables:
        try:
            if _may_have_table(doc[index]):
                tables = _tables_on_page(doc[index])
        except Exception as exc:  # a broken page must not lose the others
            logger.debug("find_tables failed on page %d: %s", index + 1, exc)
    try:
        raw = _extract_page(doc[index], index + 1, tables, use_ocr, ocr_language)
    except Exception as exc:  # one bad page must not kill the document
        logger.warning("Page %d failed to parse: %s", index + 1, exc)
        raw = RawPage(page=index + 1, blocks=[], tables=[], height=1.0)
        raw.warning = f"page {index + 1}: parse error ({exc})"
    return raw.to_dict()


def iter_raw_batches(
    path: str | Path,
    start: int,
    stop: int,
    *,
    extract_tables: bool,
    use_ocr: bool,
    ocr_language: str,
    workers: int | None = None,
    batch_pages: int = BATCH_PAGES,
    isolate: bool = False,
):
    """Yield lists of raw-page dicts for pages ``start..stop-1``, in page order.

    Long documents are parsed by crash-isolated worker processes (see ``workers.py``);
    short ones in this process.
    """
    path = str(Path(path).resolve())
    ranges = [(i, min(i + batch_pages, stop)) for i in range(start, stop, batch_pages)]
    workers = max(1, min(workers or default_workers(), len(ranges)))
    args = (extract_tables, use_ocr, ocr_language)
    if not isolate and (workers < 2 or stop - start < MIN_PARALLEL_PAGES):
        for a, b in ranges:
            yield extract_raw_pages(path, a, b, *args)
        return

    from local_agentic_rag.ingest.workers import ParsePool

    with ParsePool(
        workers,
        path=path,
        extract_tables=extract_tables,
        use_ocr=use_ocr,
        ocr_language=ocr_language,
    ) as pool:
        yield from pool.imap(ranges)


# ---------------------------------------------------------------------------
# Pass 2: document-level cleanup (needs statistics over all raw pages)
# ---------------------------------------------------------------------------


class PageStats:
    """Running headers/footers and body font size, accumulated one raw page at a time."""

    def __init__(self) -> None:
        self.margin_counts: Counter[str] = Counter()
        self.size_counts: Counter[float] = Counter()
        self.n_pages = 0
        # Running headers that change per chapter ("Chapter 3: Region") never reach the
        # 40% threshold; they do repeat verbatim on consecutive pages, which body text in
        # the margin bands does not.  Keyed by the exact text (digits kept), so numbered
        # headings at the top of consecutive pages ("3.1 ...", "3.2 ...") are not stripped.
        # exact text -> (last page seen, current run, longest run).
        self._runs: dict[str, tuple[int, int, int]] = {}

    def add(self, rp: RawPage) -> None:
        self.n_pages += 1
        keys: set[str] = set()
        exact: set[str] = set()
        for b in rp.blocks:
            if _in_margin(b, rp.height) and len(b.text) < 200:
                for text in (b.text, *(b.lines or ())):
                    keys.add(_margin_key(text))
                    exact.add(_exact_key(text))
        self.margin_counts.update(keys)
        for k in exact:
            last, run, best = self._runs.get(k, (-1, 0, 0))
            run = run + 1 if last == rp.page - 1 else 1
            self._runs[k] = (rp.page, run, max(best, run))
        for size, n in rp.sizes:
            self.size_counts[size] += n

    def repeated_margins(self) -> set[str]:
        threshold = max(3, int(REPEAT_FRACTION * self.n_pages))
        repeated = {k for k, c in self.margin_counts.items() if c >= threshold}
        repeated.update(
            "=" + k for k, (_, _, best) in self._runs.items() if best >= LOCAL_REPEAT_RUN
        )
        return repeated

    def body_size(self) -> float:
        return self.size_counts.most_common(1)[0][0] if self.size_counts else 10.0


class PageFinaliser:
    """Raw page → :class:`PageContent`, in page order; also tracks the title and OCR state."""

    def __init__(self, stats: PageStats, strip_headers_footers: bool) -> None:
        self.repeated = stats.repeated_margins() if strip_headers_footers else None
        self.body_size = stats.body_size()
        self.font_title = ""
        self.has_content = False
        self.needs_ocr: list[int] = []
        self._first = True

    def __call__(self, rp: RawPage) -> PageContent:
        if self.repeated is not None:
            _strip_margins(rp, self.repeated)
        if self._first:  # the first page's largest text is the title candidate
            self.font_title = _title_from_fonts(rp)
            self._first = False
        page = _finalise_page(rp, self.body_size)
        if page.blocks or page.tables:
            self.has_content = True
        if page.needs_ocr:
            self.needs_ocr.append(page.page)
        return page

    def summary_warnings(self, use_ocr: bool) -> list[str]:
        if not self.needs_ocr:
            return []
        return [
            f"{len(self.needs_ocr)} page(s) look scanned/image-only and have no extractable text"
            + ("" if use_ocr else " (install tesseract-ocr to enable OCR)")
            + f": {self.needs_ocr[:20]}"
        ]


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------


class _RawBlock:
    __slots__ = ("text", "bbox", "max_size", "bold", "n_lines", "lines")

    def __init__(self, text, bbox, max_size, bold, n_lines, lines=None):
        self.text: str = text
        self.bbox: tuple[float, float, float, float] = bbox
        self.max_size: float = max_size
        self.bold: bool = bold
        self.n_lines: int = n_lines
        # Kept only for multi-line header/footer blocks, so running-header lines can be
        # removed one by one ("Report title" + "Chapter 3: Region" in one block).
        self.lines: list[str] | None = lines


class RawPage:
    __slots__ = (
        "page",
        "blocks",
        "tables",
        "height",
        "ocr_applied",
        "needs_ocr",
        "sizes",
        "warning",
    )

    def __init__(self, page, blocks, tables, height, ocr_applied=False, needs_ocr=False):
        self.page: int = page
        self.blocks: list[_RawBlock] = blocks
        self.tables: list[str] = tables
        self.height: float = height
        self.ocr_applied: bool = ocr_applied
        self.needs_ocr: bool = needs_ocr
        self.sizes: list[tuple[float, int]] = []  # (font size, char count) samples
        self.warning: str | None = None

    def to_dict(self) -> dict:
        sizes: Counter[float] = Counter()
        for size, n in self.sizes:  # aggregated per page; first-seen order is kept
            sizes[size] += n
        return {
            "page": self.page,
            "height": self.height,
            "ocr_applied": self.ocr_applied,
            "needs_ocr": self.needs_ocr,
            "tables": self.tables,
            "blocks": [
                [b.text, b.bbox, b.max_size, b.bold, b.n_lines, *([b.lines] if b.lines else [])]
                for b in self.blocks
            ],
            "sizes": list(sizes.items()),
            "warning": self.warning,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RawPage:
        rp = cls(
            page=d["page"],
            blocks=[_RawBlock(b[0], tuple(b[1]), *b[2:]) for b in d["blocks"]],
            tables=list(d["tables"]),
            height=d["height"],
            ocr_applied=d["ocr_applied"],
            needs_ocr=d["needs_ocr"],
        )
        rp.sizes = [(float(s), int(n)) for s, n in d["sizes"]]
        rp.warning = d.get("warning")
        return rp


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
                [
                    w
                    for w in words
                    if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1
                ]
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


def _extract_page(
    page, page_no: int, tables: list[tuple[str, tuple]], ocr_available: bool, lang: str
) -> RawPage:
    width, height = page.rect.width, page.rect.height
    tables_md = [md for md, _ in tables]
    table_bboxes = [bbox for _, bbox in tables]

    textpage = None
    ocr_applied = False
    plain = page.get_text("text").strip()
    image_only = len(plain) < MIN_TEXT_CHARS and bool(page.get_images(full=False))
    # A text layer that is mostly garbage (unmapped glyphs from legacy or subset fonts)
    # is read like a scan: OCR of the rendered page recovers what the eye sees.
    garbled = not image_only and garbled_fraction(plain) > GARBLED_FRACTION
    if (image_only or garbled) and ocr_available:
        # Preferred: Tesseract CLI on a 300-dpi render.  Measured on a scanned Hindi page:
        # 99% word recall vs 70% for PyMuPDF's OCR text layer, which splits Devanagari
        # words at vowel signs and merges neighbours.
        paragraphs = _ocr_cli_paragraphs(page, page_no, lang)
        if paragraphs:
            raw = RawPage(
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
    raw = RawPage(page=page_no, blocks=[], tables=tables_md, height=height, ocr_applied=ocr_applied)

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
        block_obj = _RawBlock(
            text=_join_lines(lines_text),
            bbox=bbox,
            max_size=max(sizes) if sizes else 0.0,
            bold=total_chars > 0 and bold_chars / total_chars > 0.8,
            n_lines=len(lines_text),
        )
        if len(lines_text) > 1 and _in_margin(block_obj, height):
            block_obj.lines = lines_text
        raw.blocks.append(block_obj)

    raw.blocks, raw.tables = _attach_captions(raw.blocks, tables)
    raw.blocks = _reading_order(raw.blocks, width)
    if image_only and not raw.blocks and not ocr_applied:
        raw.needs_ocr = True
    return raw


#: "Table 2: ...", "Table A.3: ...", "TABLE IV. ..." at the start of a text block.
_CAPTION_RE = re.compile(r"^(Table|TABLE|Tab\.)\s+([A-Z]\.?)?([0-9]+|[IVXL]+)(\.[0-9]+)*\s*[:.]")
#: Largest vertical gap (points) between a table's edge and its caption block.
CAPTION_GAP = 30.0
MAX_CAPTION_CHARS = 400


def _attach_captions(
    blocks: list[_RawBlock], tables: list[tuple[str, tuple]]
) -> tuple[list[_RawBlock], list[str]]:
    """Prefix each Markdown table with its caption block ("Table 2: ... on NQ ...").

    ``find_tables`` returns only the grid, so a table chunk would not say *which*
    table it is: an arXiv paper with the same columns in Table 2 (NQ) and Table A.3
    (another dataset) gave the reader two indistinguishable chunks.  The caption is
    the nearest "Table N" block directly above or below the table and overlapping it
    horizontally; it moves out of the running text, where it otherwise lands in the
    middle of the paragraph that flows around the table.  Serialised statistical
    tables already start with the title read above them and are left alone.
    """
    texts = [md for md, _ in tables]
    pairs = []
    for t, (md, bbox) in enumerate(tables):
        if not md.lstrip().startswith("|"):
            continue
        for b, block in enumerate(blocks):
            if len(block.text) > MAX_CAPTION_CHARS or not _CAPTION_RE.match(block.text):
                continue
            x0, y0, x1, y1 = block.bbox
            overlap = min(x1, bbox[2]) - max(x0, bbox[0])
            if overlap < 0.5 * min(x1 - x0, bbox[2] - bbox[0]):
                continue
            gap = y0 - bbox[3] if y0 >= bbox[1] else bbox[1] - y1
            if -2.0 <= gap <= CAPTION_GAP:
                pairs.append((gap, t, b))
    used_tables: set[int] = set()
    used_blocks: set[int] = set()
    for _, t, b in sorted(pairs):  # nearest pairs first; one caption per table
        if t in used_tables or b in used_blocks:
            continue
        used_tables.add(t)
        used_blocks.add(b)
        texts[t] = f"{blocks[b].text}\n{texts[t]}"
    return [blk for b, blk in enumerate(blocks) if b not in used_blocks], texts


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


#: Share of unreadable characters above which a page's text layer is replaced by OCR.
GARBLED_FRACTION = 0.3


def garbled_fraction(text: str) -> float:
    """Share of characters that cannot be real text: U+FFFD, private-use or unassigned code
    points, and ``(cid:NN)`` placeholders that some extractors emit for unmapped glyphs."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = sum(1 for c in chars if c == "\ufffd" or unicodedata.category(c) in ("Co", "Cn", "Cs"))
    bad += 7 * len(re.findall(r"\(cid:\d+\)", text))  # "(cid:12)" is 8 chars, one glyph
    return min(1.0, bad / len(chars))


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


def _in_margin(b: _RawBlock, height: float) -> bool:
    return b.bbox[3] <= height * MARGIN_BAND or b.bbox[1] >= height * (1 - MARGIN_BAND)


def _exact_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_boilerplate(text: str, repeated: set[str]) -> bool:
    """``repeated`` holds digit-masked keys (global rule) and "=exact" keys (local rule)."""
    return (
        _margin_key(text) in repeated
        or "=" + _exact_key(text) in repeated
        or bool(_PAGE_NUMBER_RE.match(text.strip()))
    )


def _strip_margins(p: RawPage, repeated: set[str]) -> None:
    """Remove running headers/footers (``repeated`` margin keys) and bare page numbers."""
    kept = []
    for b in p.blocks:
        if not _in_margin(b, p.height):
            kept.append(b)
            continue
        if _is_boilerplate(b.text, repeated):
            continue
        if b.lines:  # drop the boilerplate lines of a mixed header block, keep the rest
            rest = [line for line in b.lines if not _is_boilerplate(line, repeated)]
            if not rest:
                continue
            if len(rest) < len(b.lines):
                b.text, b.n_lines, b.lines = _join_lines(rest), len(rest), rest
        kept.append(b)
    p.blocks = kept


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


def _finalise_page(rp: RawPage, body_size: float) -> PageContent:
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


def _title_from_fonts(first_page: RawPage) -> str:
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
