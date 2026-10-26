"""
Core data structures shared by ingestion, retrieval, the agent and evaluation.

Plain dataclasses with ``to_dict()`` so every result is JSON-serialisable for
benchmark logs and can be shown in the notebook UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

#: Sentinel the generator is instructed to emit when the evidence does not
#: contain the answer.  Kept identical to the original prototype for comparability.
ANSWER_NOT_FOUND = "ANSWER_NOT_FOUND"

#: User-facing text shown instead of the raw sentinel.
REFUSAL_MESSAGE = "I could not find the answer to this question in the document."


@dataclass
class PageContent:
    """Cleaned content of one physical PDF page."""

    page: int  # 1-indexed physical page number
    blocks: list[TextBlock]  # in reading order (headers/footers removed)
    tables: list[str]  # markdown tables found on the page
    ocr_applied: bool = False
    needs_ocr: bool = False  # image-only page that could not be OCR'd

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


@dataclass
class TextBlock:
    text: str
    kind: Literal["heading", "paragraph"] = "paragraph"
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    source_path: str
    sha256: str
    num_pages: int
    pages: list[PageContent]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """One retrieval unit.  Never crosses a page boundary so citations stay exact."""

    chunk_id: str
    doc_id: str
    page: int
    section: str
    text: str
    kind: Literal["text", "table"] = "text"
    n_tokens: int = 0
    doc_title: str = ""

    @property
    def embed_text(self) -> str:
        """Text given to the embedder/BM25: contextual header + body.

        Prefixing the document title and section heading lets a chunk that
        says "it increased by 4%" still match "inflation in Q2" queries.
        """
        header = " > ".join(p for p in (self.doc_title, self.section) if p)
        return f"{header}\n{self.text}" if header else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # final score used for ordering
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None  # cross-encoder probability in [0, 1]
    matched_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "page": self.chunk.page,
            "section": self.chunk.section,
            "kind": self.chunk.kind,
            "text": self.chunk.text,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rerank_score": self.rerank_score,
        }


@dataclass
class Citation:
    marker: int  # the [n] number used in the answer
    chunk_id: str
    doc_id: str
    page: int
    snippet: str  # first ~200 chars of the cited passage


@dataclass
class TraceStep:
    """One node execution in the agent loop — shown to users for transparency."""

    step: str
    detail: dict[str, Any]
    seconds: float


@dataclass
class IngestReport:
    doc_id: str
    title: str
    num_pages: int
    num_chunks: int
    num_table_chunks: int
    ocr_pages: list[int]
    empty_pages: list[int]
    seconds: float
    cached: bool
    warnings: list[str]
    resumed_from_page: int = 0  # >0: an interrupted ingest continued after this page

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    standalone_question: str = ""
    search_queries: list[str] = field(default_factory=list)
    grounded: bool | None = None  # verifier verdict (None = not verified)
    unsupported_claims: list[str] = field(default_factory=list)
    retrieval_rounds: int = 1
    answer_attempts: int = 1
    llm_calls: int = 0
    trace: list[TraceStep] = field(default_factory=list)
    latency_sec: float = 0.0
    raw_response: str = ""
    error: str | None = None

    @property
    def cited_pages(self) -> list[int]:
        return sorted({c.page for c in self.citations})

    @property
    def retrieved_pages(self) -> list[int]:
        """Ranked pages of retrieved passages (duplicates removed, order kept)."""
        seen: list[int] = []
        for sc in self.retrieved:
            if sc.chunk.page not in seen:
                seen.append(sc.chunk.page)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "citations": [asdict(c) for c in self.citations],
            "cited_pages": self.cited_pages,
            "retrieved_pages": self.retrieved_pages,
            "retrieved": [sc.to_dict() for sc in self.retrieved],
            "standalone_question": self.standalone_question,
            "search_queries": self.search_queries,
            "grounded": self.grounded,
            "unsupported_claims": self.unsupported_claims,
            "retrieval_rounds": self.retrieval_rounds,
            "answer_attempts": self.answer_attempts,
            "iterations": self.retrieval_rounds,
            "llm_calls": self.llm_calls,
            "trace": [asdict(t) for t in self.trace],
            "latency_sec": self.latency_sec,
            "raw_response": self.raw_response,
            "error": self.error,
        }
