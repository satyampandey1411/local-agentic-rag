"""
Shared offline fakes and PDF builders for the v2 (agentic) test-suite.

Nothing here downloads a model:
    FakeEmbedder  hashed bag-of-words vectors, L2-normalised
    FakeReranker  token-overlap score in [0, 1]
    make_*_pdf    reportlab generators for synthetic PDFs
    make_index    DocumentIndex from plain (page, text) tuples
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from local_agentic_rag.agent import prompts as P
from local_agentic_rag.retrieval.bm25 import tokenize
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.schema import Chunk

# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder with the real Embedder's interface."""

    def __init__(self, dim: int = 256, max_tokens: int = 512) -> None:
        self.dim = dim
        self.max_tokens = max_tokens
        self.loaded = False
        self.passage_calls = 0
        self.query_calls = 0

    def load(self) -> None:
        self.loaded = True

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        self.passage_calls += 1
        return self._encode(texts)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        self.query_calls += 1
        return self._encode(queries)


class FakeReranker:
    """Fraction of the query's content tokens present in the passage (in [0, 1])."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def load(self) -> None:
        pass

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        self.calls.append((query, len(passages)))
        q = set(tokenize(query))
        out = []
        for p in passages:
            toks = set(tokenize(p))
            out.append(len(q & toks) / len(q) if q else 0.0)
        return np.asarray(out, dtype=np.float32)


class TableReranker:
    """Reranker whose scores come from a lookup ``{(query, substring): score}``."""

    def __init__(self, table: dict[tuple[str, str], float]) -> None:
        self.table = table

    def load(self) -> None:
        pass

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        out = []
        for p in passages:
            s = 0.0
            for (q, needle), value in self.table.items():
                if q == query and needle in p:
                    s = max(s, value)
            out.append(s)
        return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def make_chunks(items: list[tuple[int, str]], doc_id: str = "doc", title: str = "") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}-p{page}-c{i}",
            doc_id=doc_id,
            page=page,
            section="",
            text=text,
            doc_title=title,
        )
        for i, (page, text) in enumerate(items)
    ]


def make_index(
    items: list[tuple[int, str]],
    embedder: FakeEmbedder | None = None,
    doc_id: str = "doc",
    title: str = "",
) -> DocumentIndex:
    embedder = embedder or FakeEmbedder()
    chunks = make_chunks(items, doc_id=doc_id, title=title)
    return DocumentIndex(
        doc_id, title, chunks, embedder.encode_passages([c.embed_text for c in chunks])
    )


# ---------------------------------------------------------------------------
# Scripted LLM responder
# ---------------------------------------------------------------------------


def passage_number(messages: list[dict[str, str]], needle: str) -> int:
    """Number [n] of the passage containing ``needle`` in a grader/generator/verifier prompt."""
    user = messages[-1]["content"]
    for match in re.finditer(r"^\[(\d+)\] \(page \d+[^)]*\)\n(.*)$", user, flags=re.MULTILINE):
        if needle in match.group(2):
            return int(match.group(1))
    raise AssertionError(f"{needle!r} not among passages")


def role_of(messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
    """Which agent step issued this LLM call."""
    if schema is P.PLAN_SCHEMA:
        return "plan"
    if schema is P.GRADE_SCHEMA:
        return "grade"
    if schema is P.VERIFY_SCHEMA:
        return "verify"
    from local_agentic_rag.agent.calculator import CALC_SCHEMA

    if schema is CALC_SCHEMA:
        return "calculate"
    system = messages[0]["content"] if messages else ""
    if system == P.ANSWER_SYSTEM:
        return "generate"
    raise AssertionError(f"unrecognised LLM call: {system[:80]!r}")


Reply = str | dict | Callable[[list[dict[str, str]]], str | dict]


def scripted(**replies: Reply | list[Reply]) -> Callable[[list[dict[str, str]], dict | None], str]:
    """Responder for ``ScriptedLLM`` keyed by step name (plan/grade/generate/verify).

    A list value is consumed one item per call (the last item repeats); dict
    replies are JSON-encoded; callables receive the messages.
    """
    counters: dict[str, int] = {}

    def responder(messages: list[dict[str, str]], schema: dict | None) -> str:
        role = role_of(messages, schema)
        if role not in replies:
            raise AssertionError(f"no scripted reply for step {role!r}")
        value = replies[role]
        if isinstance(value, list):
            i = counters.get(role, 0)
            counters[role] = i + 1
            value = value[min(i, len(value) - 1)]
        if callable(value):
            value = value(messages)
        if isinstance(value, BaseException):
            raise value
        return json.dumps(value) if isinstance(value, dict) else value

    return responder


# ---------------------------------------------------------------------------
# PDF builders (reportlab)
# ---------------------------------------------------------------------------

LOREM = (
    "The committee reviewed the operating results for the fiscal year and noted steady growth "
    "in recurring revenue. Costs were contained through supplier consolidation and lower "
    "freight rates. Management expects similar conditions to continue next year."
)


def make_text_pdf(path: Path, n_pages: int = 4) -> list[tuple[str, str]]:
    """Multi-page PDF with a heading + paragraphs per page, running header/footer and page numbers.

    Returns the (heading, unique_marker) for each page.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    width, height = LETTER
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle("")
    pages = []
    for p in range(1, n_pages + 1):
        heading = f"Section {p} Overview of Topic{p}"
        marker = f"uniqueterm{p}"
        # running header + footer + page number
        c.setFont("Helvetica", 8)
        c.drawString(72, height - 30, "ACME Corp Annual Report Running Header")
        c.drawString(72, 25, "Confidential draft footer text")
        c.drawRightString(width - 72, 25, f"Page {p}")
        # heading
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 100, heading)
        # paragraphs
        y = height - 140
        for para in range(3):
            text = c.beginText(72, y)
            text.setFont("Helvetica", 10)
            text.setLeading(13)
            body = f"Paragraph {para} on page {p} mentions {marker}. " + LOREM
            for line in _wrap(body, 95):
                text.textLine(line)
            c.drawText(text)
            y -= 90
        c.showPage()
        pages.append((heading, marker))
    c.save()
    return pages


def make_two_column_pdf(path: Path) -> None:
    """One page: left column A1..A5 and right column B1..B5 at the same heights."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    width, height = LETTER
    c = canvas.Canvas(str(path), pagesize=LETTER)
    y = height - 100
    for i in range(1, 6):
        for x, col in ((60, "A"), (330, "B")):
            text = c.beginText(x, y)
            text.setFont("Helvetica", 10)
            text.setLeading(13)
            body = f"{col}{i} column {col} paragraph number {i} with filler words for wrapping across lines here."
            for line in _wrap(body, 40):
                text.textLine(line)
            c.drawText(text)
        y -= 110
    c.showPage()
    c.save()


def make_table_pdf(path: Path) -> None:
    """One page with a heading, a paragraph and a ruled (grid) table."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    data = [
        ["Region", "Revenue", "Units", "Growth"],
        ["Northwind", "1250", "310", "4.5%"],
        ["Southgate", "980", "275", "2.1%"],
        ["Eastvale", "1430", "402", "6.8%"],
        ["Westport", "760", "198", "-1.2%"],
    ]
    table = Table(data, colWidths=[120, 90, 90, 90])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story = [
        Paragraph("Regional Sales Summary", styles["Heading1"]),
        Paragraph(
            "The following table lists revenue, units and growth for each sales region in the "
            "reporting period. Figures are in thousands of dollars.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        table,
    ]
    SimpleDocTemplate(str(path), pagesize=LETTER).build(story)


def make_blank_pdf(path: Path, n_pages: int = 2) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)
    for _ in range(n_pages):
        c.showPage()
    c.save()


def _wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
