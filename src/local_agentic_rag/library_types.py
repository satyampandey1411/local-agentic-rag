"""
The contract between the HTTP server (``local_agentic_rag.server``) and the
document library backend (``local_agentic_rag.library.Library``).

The server only ever talks to a :class:`LibraryProtocol`, so it can be developed
and tested against an in-memory fake while the real, GPU-backed library is built.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from local_agentic_rag.schema import Answer, TraceStep

#: Life-cycle of an uploaded document: queued -> processing -> ready | failed.
DocStatus = Literal["queued", "processing", "ready", "failed"]


@dataclass
class DocRecord:
    """Everything the UI needs to show one document in the library."""

    doc_id: str
    title: str
    filename: str
    status: DocStatus
    stage: str = ""  # human-readable, e.g. "Reading pages 1,201-1,264 of 10,000"
    num_pages: int = 0
    pages_done: int = 0  # pages parsed+indexed so far (progress = pages_done / num_pages)
    num_chunks: int = 0
    size_bytes: int = 0
    ocr_pages: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    ingest_seconds: float | None = None
    eta_seconds: float | None = None
    #: Ask-while-processing: True while the document is still processing but a checkpointed
    #: prefix can already be searched; ``pages_indexed`` pages are covered by it.
    askable: bool = False
    pages_indexed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LibraryProtocol(Protocol):
    """What the server needs from a document library.  All methods are called from
    worker threads, possibly concurrently, so implementations must be thread-safe."""

    def submit(
        self, src_path: str, filename: str
    ) -> DocRecord: ...  # takes ownership of a temp file (moves it), queues background ingestion; the same file twice returns the existing record
    def list_documents(self) -> list[DocRecord]: ...
    def get(self, doc_id: str) -> DocRecord: ...  # KeyError if unknown
    def delete(self, doc_id: str) -> None: ...  # cancels a running job; KeyError if unknown
    def retry(self, doc_id: str) -> DocRecord: ...  # re-queue a failed document
    def ask(
        self,
        question: str,
        doc_ids: list[str] | None,
        history: list[tuple[str, str]],
        on_step: Callable[[TraceStep], None] | None = None,
        on_token: Callable[[dict[str, Any]], None] | None = None,
        answer_language: str | None = None,
    ) -> (
        Answer
    ): ...  # blocking, thread-safe; doc_ids None = all ready docs; ValueError if a doc is not ready
    def render_page(
        self, doc_id: str, page: int, zoom: float = 1.5, chunk_id: str | None = None
    ) -> bytes: ...  # PNG; highlights the passage of chunk_id if given; KeyError / IndexError
    # Highlight boxes [x0, y0, x1, y1] as 0-1 page fractions, top to bottom; [] if none;
    # KeyError / IndexError. Optional: the server falls back to [] when a library lacks it.
    def highlight_boxes(
        self, doc_id: str, page: int, chunk_id: str | None
    ) -> list[list[float]]: ...
    def page_count(self, doc_id: str) -> int: ...
    def health(
        self,
    ) -> dict[
        str, Any
    ]: ...  # {"llm": "ok"|"unavailable"|..., "device": "cuda"|"cpu", "queue": int, "busy": bool, ...}
