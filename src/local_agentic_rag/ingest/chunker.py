"""
Structure-aware chunking.

Rules (each one fixes a concrete failure of the original prototype or the reference notebook):
    * Sizes are measured in *embedder tokens*, not characters — the original
      2048-char chunks were silently truncated to 256 tokens by MiniLM.
    * Chunks break at headings and paragraph boundaries first, sentences
      second, words last — never mid-sentence unless a sentence is enormous.
    * Every chunk remembers its section heading, which is prepended to the
      embedding text (see ``Chunk.embed_text``).
    * Chunks never span pages, so a citation always points to one page.
    * Every chunk also records the number of the numbered section it sits in
      (``Chunk.section_no``): the last "§ 164." / "Section 2.1" heading, which unnumbered
      sub-headings ("(a) General rule", "Editorial Notes") do not displace and which a
      CHAPTER/PART/ARTICLE-level heading ends.  The number never enters the chunk text.
    * Tables become their own chunks (Markdown), split by rows with the
      header row repeated so each piece is self-describing.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from local_agentic_rag.schema import Chunk, PageContent, ParsedDocument
from local_agentic_rag.sections import INLINE_SECTION_RE, STRUCTURAL_HEAD_RE, section_number

TokenCounter = Callable[[str], int]

# INLINE_SECTION_RE for a single unit: sentence splitting can cut right after "§ 164.", leaving
# the unit ending in the number, so the capital letter that follows may be in the next unit.
_UNIT_SECTION_RE = re.compile(r"(?:^|\s)§\s*(\d+[A-Za-z]?)\.(?=\s+[A-Z]|$)")
# English-style stops need a capital (or digit/quote) next, so "e.g. foo" stays whole. The
# danda (।, ॥), Urdu full stop (۔), Arabic question mark (؟) and CJK stops always end a sentence.
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?;:])\s+(?=[A-Z0-9\"'(\[•\-])|(?<=[।॥۔؟])\s*(?=\S)|(?<=[。！？])(?=\S)"
)
# "Table 7.", "Table A.3:", "Appendix Table B." — the label that names a table.
_TABLE_LABEL_RE = re.compile(r"\b(?:Appendix\s+)?Table\s+[A-Z]?(?:\d+(?:\.\d+)*)?(?=[.:]\s)")


def approx_token_count(text: str) -> int:
    """Cheap tokenizer-free estimate (~1.3 tokens per word for BERT-style vocabularies)."""
    return max(1, int(len(re.findall(r"\w+|[^\w\s]", text)) * 1.1))


class _PageBuffer:
    """Accumulates paragraph units for one page and emits (section, text, section_no) chunks."""

    def __init__(self, section: str, min_tokens: int) -> None:
        self.chunks: list[tuple[str, str, str | None]] = []
        self.units: list[str] = []
        self.tokens = 0
        self.section = section
        self.section_no: str | None = None
        self.min_tokens = min_tokens

    def add(self, unit: str, n_tokens: int, section: str, section_no: str | None) -> None:
        if not self.units:
            self.section = section
            self.section_no = section_no
        self.units.append(unit)
        self.tokens += n_tokens

    def reset(self, units: list[str] | None = None, tokens: int = 0) -> None:
        self.units, self.tokens = units or [], tokens

    def flush(self, force: bool = False) -> None:
        text = " ".join(self.units).strip()
        if not text:
            self.reset()
            return
        if force or self.tokens >= self.min_tokens or not self.chunks:
            self.chunks.append((self.section, text, self.section_no))
            self.reset()
        elif self.chunks[-1][0] == self.section:
            # Too small to stand alone: append to the previous chunk of the same section.
            prev_section, prev_text, prev_no = self.chunks[-1]
            self.chunks[-1] = (prev_section, f"{prev_text} {text}", prev_no)
            self.reset()


def chunk_document(
    doc: ParsedDocument,
    *,
    chunk_tokens: int = 320,
    overlap_tokens: int = 48,
    min_chunk_tokens: int = 24,
    count_tokens: TokenCounter = approx_token_count,
) -> list[Chunk]:
    chunker = PageChunker(
        doc.doc_id,
        doc.title,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
        count_tokens=count_tokens,
    )
    return [c for page in doc.pages for c in chunker.add_page(page)]


class PageChunker:
    """Chunks pages one at a time, in order.

    The only state carried between pages is the current section heading and the current
    section number, so a streaming ingest can checkpoint them (``section``,
    ``section_no``) and resume mid-document.
    """

    def __init__(
        self,
        doc_id: str,
        doc_title: str,
        *,
        chunk_tokens: int = 320,
        overlap_tokens: int = 48,
        min_chunk_tokens: int = 24,
        count_tokens: TokenCounter = approx_token_count,
        section: str = "",
        section_no: str | None = None,
    ) -> None:
        self.doc_id, self.doc_title = doc_id, doc_title
        self.chunk_tokens, self.overlap_tokens = chunk_tokens, overlap_tokens
        self.min_chunk_tokens, self.count_tokens = min_chunk_tokens, count_tokens
        self.section = section
        self.section_no = section_no

    def add_page(self, page: PageContent) -> list[Chunk]:
        chunk_tokens, overlap_tokens = self.chunk_tokens, self.overlap_tokens
        min_chunk_tokens, count_tokens = self.min_chunk_tokens, self.count_tokens
        chunks: list[Chunk] = []
        section = self.section
        page_section_start = section
        section_no = page_no_start = self.section_no
        buf = _PageBuffer(section, min_chunk_tokens)
        for block in page.blocks:
            if block.kind == "heading":
                buf.flush()
                if buf.units:  # fragment that could not merge: keep it under its own heading
                    buf.flush(force=True)
                section = block.text.strip()
                section_no = _heading_section_no(section, section_no)
                continue
            for unit in _split_units(block.text, chunk_tokens, count_tokens):
                n = count_tokens(unit)
                opens = _UNIT_SECTION_RE.match(unit.lstrip())
                if opens:  # a unit that opens a section belongs to it
                    section_no = opens.group(1).lower()
                if buf.units and buf.tokens + n > chunk_tokens:
                    buf.flush(force=True)
                    tail = _overlap_tail(buf.chunks[-1][1], overlap_tokens, count_tokens)
                    if tail and count_tokens(tail) + n <= chunk_tokens:
                        buf.reset([tail], count_tokens(tail))
                        # The tail only repeats earlier text; the chunk is about its new unit.
                        buf.section, buf.section_no = section, section_no
                buf.add(unit, n, section, section_no)
                for m in _UNIT_SECTION_RE.finditer(unit):  # later units follow the last start
                    section_no = m.group(1).lower()
        buf.flush()
        buf.flush(force=True)  # a fragment that could not be merged
        self.section = section
        self.section_no = section_no

        # Drop fragments with no real content (stray labels, figure axis ticks).
        page_chunks = [(sec, text, no) for sec, text, no in buf.chunks if count_tokens(text) >= 4]
        for i, (sec, text, no) in enumerate(page_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"{self.doc_id}-p{page.page}-c{i}",
                    doc_id=self.doc_id,
                    page=page.page,
                    section=sec,
                    text=text,
                    kind="text",
                    n_tokens=count_tokens(text),
                    doc_title=self.doc_title,
                    section_no=no,
                )
            )

        for t_index, table in enumerate(page.tables):
            for part_index, part in enumerate(_split_table(table, chunk_tokens, count_tokens)):
                if count_tokens(part) < min_chunk_tokens // 2:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{self.doc_id}-p{page.page}-t{t_index}-{part_index}",
                        doc_id=self.doc_id,
                        page=page.page,
                        section=_table_section(table, page_section_start or section),
                        text=part,
                        kind="table",
                        n_tokens=count_tokens(part),
                        doc_title=self.doc_title,
                        section_no=page_no_start,
                    )
                )
        return chunks


def _heading_section_no(heading: str, current: str | None) -> str | None:
    """Section number in force after ``heading``.

    A numbered heading ("§ 164. Taxes", "Section 2.1 Scope") starts that section; a
    CHAPTER/SUBCHAPTER/PART/ARTICLE/TITLE heading ends the current one (None); any other
    heading ("(a) General rule", "Editorial Notes") is a sub-heading and keeps it."""
    number = section_number(heading)
    if number is not None:
        return number
    m = INLINE_SECTION_RE.search(heading)
    if m:
        return m.group(1).lower()
    if STRUCTURAL_HEAD_RE.match(heading):
        return None
    return current


def _table_section(table: str, inherited: str) -> str:
    """Section of a table's chunks: the one at its page start, unless that names another table.

    Statistical releases print one table per page with its title as the page's
    first heading, so the section at the page start is the previous page's table
    ("Table 10. Corporate Profits ..." over the rows of Table 12).  A serialised
    table starts with its own title line; when that names another table than the
    inherited section, the title line is the section.
    """
    title = table.strip().splitlines()[0] if table.strip() else ""
    own = _TABLE_LABEL_RE.search(title)
    old = _TABLE_LABEL_RE.search(inherited)
    if own is None or old is None or own.group(0) == old.group(0):
        return inherited
    return title[own.start() :].strip()[:200]


def _split_units(text: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """Paragraph → sentences → word windows, only as far as needed to fit."""
    if count_tokens(text) <= max_tokens:
        return [text]
    units: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if count_tokens(sentence) <= max_tokens:
            units.append(sentence)
            continue
        window: list[str] = []
        for word in sentence.split():
            pieces = _char_windows(word, max_tokens, count_tokens)
            if len(pieces) > 1:  # one space-less run longer than a chunk: emit its pieces as is
                if window:
                    units.append(" ".join(window))
                    window = []
                units.extend(pieces)
                continue
            window.append(word)
            if count_tokens(" ".join(window)) >= max_tokens:
                units.append(" ".join(window))
                window = []
        if window:
            units.append(" ".join(window))
    return units


def _char_windows(word: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """A "word" longer than a chunk (Chinese, Japanese or Thai text has no spaces) → pieces."""
    n = count_tokens(word)
    if n <= max_tokens:
        return [word]
    step = max(1, int(len(word) * max_tokens / n * 0.9))
    return [word[i : i + step] for i in range(0, len(word), step)]


def _overlap_tail(text: str, overlap_tokens: int, count_tokens: TokenCounter) -> str:
    """Last whole sentences of ``text`` totalling at most ``overlap_tokens``."""
    if overlap_tokens <= 0:
        return ""
    tail: list[str] = []
    total = 0
    for sentence in reversed(_SENTENCE_SPLIT.split(text)):
        n = count_tokens(sentence)
        if total + n > overlap_tokens:
            break
        tail.insert(0, sentence)
        total += n
    return " ".join(tail).strip()


def _split_table(markdown: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    lines = [ln for ln in markdown.splitlines() if ln.strip()]
    # Markdown tables repeat caption (if any) + header + separator; serialised tables
    # repeat their title line.
    first_row = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")), None)
    n_header = 1 if first_row is None else first_row + 2
    if count_tokens(markdown) <= max_tokens or len(lines) <= max(3, n_header + 1):
        return [markdown]
    header, rows = lines[:n_header], lines[n_header:]
    parts: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join(header + current + [row])
        if current and count_tokens(candidate) > max_tokens:
            parts.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(header + current))
    return parts
