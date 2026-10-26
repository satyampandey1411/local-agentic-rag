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
    * Tables become their own chunks (Markdown), split by rows with the
      header row repeated so each piece is self-describing.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from local_agentic_rag.schema import Chunk, PageContent, ParsedDocument

TokenCounter = Callable[[str], int]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9\"'(\[•\-])")
# "Table 7.", "Table A.3:", "Appendix Table B." — the label that names a table.
_TABLE_LABEL_RE = re.compile(r"\b(?:Appendix\s+)?Table\s+[A-Z]?(?:\d+(?:\.\d+)*)?(?=[.:]\s)")


def approx_token_count(text: str) -> int:
    """Cheap tokenizer-free estimate (~1.3 tokens per word for BERT-style vocabularies)."""
    return max(1, int(len(re.findall(r"\w+|[^\w\s]", text)) * 1.1))


class _PageBuffer:
    """Accumulates paragraph units for one page and emits (section, text) chunks."""

    def __init__(self, section: str, min_tokens: int) -> None:
        self.chunks: list[tuple[str, str]] = []
        self.units: list[str] = []
        self.tokens = 0
        self.section = section
        self.min_tokens = min_tokens

    def add(self, unit: str, n_tokens: int, section: str) -> None:
        if not self.units:
            self.section = section
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
            self.chunks.append((self.section, text))
            self.reset()
        elif self.chunks[-1][0] == self.section:
            # Too small to stand alone: append to the previous chunk of the same section.
            prev_section, prev_text = self.chunks[-1]
            self.chunks[-1] = (prev_section, f"{prev_text} {text}")
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

    The only state carried between pages is the current section heading, so a
    streaming ingest can checkpoint it (``section``) and resume mid-document.
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
    ) -> None:
        self.doc_id, self.doc_title = doc_id, doc_title
        self.chunk_tokens, self.overlap_tokens = chunk_tokens, overlap_tokens
        self.min_chunk_tokens, self.count_tokens = min_chunk_tokens, count_tokens
        self.section = section

    def add_page(self, page: PageContent) -> list[Chunk]:
        chunk_tokens, overlap_tokens = self.chunk_tokens, self.overlap_tokens
        min_chunk_tokens, count_tokens = self.min_chunk_tokens, self.count_tokens
        chunks: list[Chunk] = []
        section = self.section
        page_section_start = section
        buf = _PageBuffer(section, min_chunk_tokens)
        for block in page.blocks:
            if block.kind == "heading":
                buf.flush()
                if buf.units:  # fragment that could not merge: keep it under its own heading
                    buf.flush(force=True)
                section = block.text.strip()
                continue
            for unit in _split_units(block.text, chunk_tokens, count_tokens):
                n = count_tokens(unit)
                if buf.units and buf.tokens + n > chunk_tokens:
                    buf.flush(force=True)
                    tail = _overlap_tail(buf.chunks[-1][1], overlap_tokens, count_tokens)
                    if tail and count_tokens(tail) + n <= chunk_tokens:
                        buf.reset([tail], count_tokens(tail))
                buf.add(unit, n, section)
        buf.flush()
        buf.flush(force=True)  # a fragment that could not be merged
        self.section = section

        # Drop fragments with no real content (stray labels, figure axis ticks).
        page_chunks = [(sec, text) for sec, text in buf.chunks if count_tokens(text) >= 4]
        for i, (sec, text) in enumerate(page_chunks):
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
                    )
                )
        return chunks


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
        words = sentence.split()
        window: list[str] = []
        for word in words:
            window.append(word)
            if count_tokens(" ".join(window)) >= max_tokens:
                units.append(" ".join(window))
                window = []
        if window:
            units.append(" ".join(window))
    return units


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
