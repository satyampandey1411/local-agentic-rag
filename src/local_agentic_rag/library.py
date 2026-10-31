"""
Document library: a persistent catalog of PDFs, a background ingestion queue, and
question answering over any subset of ready documents.

    lib = Library("~/rag-library", llm=OllamaLLM("qwen2.5:7b-instruct"))
    rec = lib.submit("/tmp/upload.pdf", "Budget Appendix 2026.pdf")   # returns at once
    lib.get(rec.doc_id).status          # queued -> processing -> ready
    answer = lib.ask("What was the total outlay?", [rec.doc_id], history=[])

Layout under ``root``:
    catalog.sqlite      one row per document (status, progress, errors)
    files/<doc_id>.pdf  the uploaded PDF
    store/              parsed pages, chunks, vectors, BM25 (see ingest/streaming.py)

Design choices:
    * One ingestion worker thread: the GPU embeds one document at a time, and the
      parse workers already use every CPU core.  Jobs survive a restart: documents
      left "queued" or "processing" are re-queued and resume from their checkpoint.
    * Questions are answered while a document is being ingested.  The embedder and
      reranker are shared, behind one lock held only for a single encode/score call,
      so a question waits at most one ingestion batch (48 pages), never the whole PDF.
    * Indexes are loaded on first use and kept in a small LRU cache, bounded by the
      total number of passages, so a library of many large PDFs fits in RAM.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import re
import shutil
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

from local_agentic_rag.agent.agent import AgenticRAG
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.ingest.pdf import PDFParseError, ocr_available, open_pdf, sha256_file
from local_agentic_rag.ingest.streaming import (
    IngestCancelled,
    IngestProgress,
    ingest_pdf,
    load_index,
    load_partial_index,
    partial_checkpoint,
    store_dirs,
)
from local_agentic_rag.library_types import DocRecord
from local_agentic_rag.llm.base import LLM
from local_agentic_rag.pipeline import RAGPipeline, _rename
from local_agentic_rag.retrieval.hybrid import HybridRetriever
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.schema import Answer, TraceStep

#: File names that say nothing about the document ("scan_0012", "document (3)", a hash).
_GENERIC_NAME_RE = re.compile(
    r"^(?:document|doc|file|scan|scanned|download|untitled|new|output|print|image|img|pdf)?"
    r"[\s_()-]*\d*[\s_()-]*$|^[0-9a-f]{12,}$",
    re.IGNORECASE,
)


def display_title(filename_title: str, parsed_title: str | None) -> str:
    """The name shown in the library and on citations.

    The uploaded file name is what the user recognises ("9-11_Commission_Report"), so it is
    kept unless it is generic; the title read from the PDF is often junk ("201-635.job" in
    the 9/11 Report's metadata, or a garbled first-page heading).
    """
    name = (filename_title or "").replace("_", " ").strip()
    if name and not _GENERIC_NAME_RE.match(name):
        return name
    return (parsed_title or "").strip() or name or "Untitled"


logger = logging.getLogger(__name__)

#: Upper bound on passages held in memory across loaded indexes (~10,000 pages is ~40k).
MAX_LOADED_CHUNKS = 250_000
#: Progress is written to the catalog at most this often.
PROGRESS_INTERVAL_S = 0.5
#: A partial index (document still being ingested) is rebuilt at most this often.
PARTIAL_REBUILD_S = 5.0

_COLUMNS = [f.name for f in fields(DocRecord)]


class _Locked:
    """Proxy that serialises the listed model calls on a shared lock."""

    # count_tokens too: the fast tokenizer is shared with encode_*, and two threads using it
    # at once fail with "Already borrowed".
    _GUARDED = frozenset({"encode_queries", "encode_passages", "score", "count_tokens"})

    def __init__(self, obj: Any, lock: threading.RLock) -> None:
        self._obj, self._lock = obj, lock

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name)
        if name not in self._GUARDED:
            return attr

        def call(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)

        return call


class Library:
    def __init__(
        self,
        root: str | Path,
        llm: LLM | None = None,
        config: RAGConfig | None = None,
        *,
        pipeline: RAGPipeline | None = None,
        start_worker: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        (self.root / "files").mkdir(parents=True, exist_ok=True)
        self.store = self.root / "store"
        self.pipe = pipeline or RAGPipeline(config or RAGConfig(), llm, cache_dir=None)
        self.config = self.pipe.config
        self.model_lock = threading.RLock()
        self._db_lock = threading.Lock()
        # Serialises id allocation + file move + catalog insert, deletes, and job start/end,
        # so a delete racing a running job or a re-upload never touches the wrong files.
        self._job_lock = threading.RLock()
        self._db = sqlite3.connect(self.root / "catalog.sqlite", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, "
            "index_dir TEXT, record TEXT NOT NULL)"
        )
        self._db.commit()
        self._loaded: OrderedDict[str, DocumentIndex] = OrderedDict()
        self._loaded_lock = threading.Lock()
        # Partial indexes of documents still being ingested: doc_id -> (checkpoint key,
        # built at, index).  Kept apart from ``_loaded`` so they never pass for a final index.
        self._partial: dict[str, tuple[tuple, float, DocumentIndex]] = {}
        # One lock per document so concurrent asks share a single partial-index build.
        self._partial_locks: dict[str, threading.Lock] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel: set[str] = set()
        self._active: str | None = None
        self._stopping = threading.Event()
        self._worker: threading.Thread | None = None
        # Jobs interrupted by a restart resume from their checkpoint.
        for rec in self.list_documents():
            if rec.status in ("queued", "processing"):
                self._update(
                    rec.doc_id,
                    status="queued",
                    stage="Waiting to resume",
                    askable=False,
                    pages_indexed=0,
                )
                self._queue.put(rec.doc_id)
        if start_worker:
            self.start()

    # ------------------------------------------------------------------ catalog

    def _row(self, doc_id: str) -> tuple[DocRecord, str | None, str]:
        with self._db_lock:
            row = self._db.execute(
                "SELECT record, index_dir, sha256 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            raise KeyError(doc_id)
        return DocRecord(**json.loads(row[0])), row[1], row[2]

    def _save(self, rec: DocRecord, sha: str | None = None, index_dir: str | None = None) -> None:
        rec.updated_at = time.time()
        with self._db_lock:
            if sha is not None:
                self._db.execute(
                    "INSERT INTO documents (doc_id, sha256, index_dir, record) VALUES (?, ?, ?, ?)",
                    (rec.doc_id, sha, index_dir, json.dumps(rec.to_dict())),
                )
            else:
                self._db.execute(
                    "UPDATE documents SET record = ? WHERE doc_id = ?",
                    (json.dumps(rec.to_dict()), rec.doc_id),
                )
            self._db.commit()

    def _update(self, doc_id: str, **changes: Any) -> DocRecord:
        rec, _, _ = self._row(doc_id)
        for key, value in changes.items():
            setattr(rec, key, value)
        self._save(rec)
        return rec

    def get(self, doc_id: str) -> DocRecord:
        return self._row(doc_id)[0]

    def list_documents(self) -> list[DocRecord]:
        with self._db_lock:
            rows = self._db.execute("SELECT record FROM documents").fetchall()
        recs = [DocRecord(**json.loads(r[0])) for r in rows]
        return sorted(recs, key=lambda r: r.created_at, reverse=True)

    def page_count(self, doc_id: str) -> int:
        return self.get(doc_id).num_pages

    def _pdf_path(self, doc_id: str) -> Path:
        return self.root / "files" / f"{doc_id}.pdf"

    def _unique_id(self, filename: str) -> str:
        """A fresh id: not in the catalog, not awaiting cleanup after a delete, no file."""
        base = re.sub(r"[^a-zA-Z0-9]+", "-", Path(filename).stem).strip("-").lower()[:50]
        base = base or "document"
        with self._db_lock:
            taken = {r[0] for r in self._db.execute("SELECT doc_id FROM documents")}
        taken |= self._cancel
        doc_id, n = base, 2
        while doc_id in taken or self._pdf_path(doc_id).exists():
            doc_id, n = f"{base}-{n}", n + 1
        return doc_id

    # ------------------------------------------------------------------ jobs

    def submit(self, src_path: str, filename: str) -> DocRecord:
        """Take ownership of ``src_path`` (moved into the library) and queue it."""
        src = Path(src_path)
        sha = sha256_file(src)
        with self._job_lock:
            with self._db_lock:
                row = self._db.execute(
                    "SELECT doc_id FROM documents WHERE sha256 = ?", (sha,)
                ).fetchone()
            if row is not None:  # the same file again: one copy, one index
                src.unlink(missing_ok=True)
                return self.get(row[0])
            doc_id = self._unique_id(filename)
            dest = self._pdf_path(doc_id)
            shutil.move(str(src), dest)
            rec = DocRecord(
                doc_id=doc_id,
                title=Path(filename).stem,
                filename=Path(filename).name,
                status="queued",
                stage="Waiting in queue",
                size_bytes=dest.stat().st_size,
                created_at=time.time(),
            )
            try:
                with open_pdf(dest) as pdf:
                    rec.num_pages = pdf.page_count
            except PDFParseError as exc:
                rec.status, rec.stage, rec.error = "failed", "", str(exc)
            self._save(rec, sha=sha, index_dir=None)
        if rec.status == "queued":
            self._queue.put(doc_id)
        return rec

    def retry(self, doc_id: str) -> DocRecord:
        rec = self.get(doc_id)
        if rec.status != "failed":
            raise ValueError(f"{rec.filename} has not failed ({rec.status})")
        rec = self._update(doc_id, status="queued", stage="Waiting in queue", error=None)
        self._queue.put(doc_id)
        return rec

    def delete(self, doc_id: str) -> None:
        with self._job_lock:
            _, _, sha = self._row(doc_id)
            with self._db_lock:
                self._db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                self._db.commit()
            with self._loaded_lock:
                self._loaded.pop(doc_id, None)
                self._partial.pop(doc_id, None)
            if self._active == doc_id:
                self._cancel.add(doc_id)  # the running job stops and cleans up at its end
            else:
                self._remove_files(doc_id, sha)

    def _remove_files(self, doc_id: str, sha: str) -> None:
        """The PDF plus every parsed/index directory built from it (any configuration)."""
        sha16 = sha[:16]
        for path in [*self.store.glob(f"{sha16}-*"), *(self.store / "parsed").glob(f"{sha16}-*")]:
            shutil.rmtree(path, ignore_errors=True)
        self._pdf_path(doc_id).unlink(missing_ok=True)

    def start(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="ingest", daemon=True)
            self._worker.start()

    def close(self) -> None:
        self._stopping.set()
        self._queue.put("")
        if self._worker is not None:
            self._worker.join(timeout=60)

    def _work(self) -> None:
        while not self._stopping.is_set():
            doc_id = self._queue.get()
            if not doc_id or self._stopping.is_set():
                continue
            try:
                self._ingest(doc_id)
            except Exception:  # the worker must survive any single document
                logger.exception("ingest worker error for %s", doc_id)

    def _ingest(self, doc_id: str) -> None:
        with self._job_lock:
            try:
                rec, _, sha = self._row(doc_id)
            except KeyError:
                return  # deleted while queued
            if rec.status != "queued":
                return
            self._active = doc_id
        last = [0.0]
        first_checkpoint = [True]

        def progress(p: IngestProgress) -> None:
            now = time.perf_counter()
            # The first pass-2 checkpoint is always recorded: it makes the document askable.
            first = p.stage == "indexing" and p.chunks > 0 and first_checkpoint[0]
            if (
                p.stage not in ("done", "finalising")
                and now - last[0] < PROGRESS_INTERVAL_S
                and not first
            ):
                return
            if first:
                first_checkpoint[0] = False
            last[0] = now
            partial = {}
            if p.stage in ("indexing", "finalising") and p.chunks > 0:
                partial = {"askable": True, "pages_indexed": min(p.pages_done, p.num_pages)}
            pages_done = p.pages_done if p.stage != "reading" else p.pages_done // 2
            if p.stage == "indexing":
                pages_done = (p.num_pages + p.pages_done) // 2
            with contextlib.suppress(KeyError):
                self._update(
                    doc_id,
                    stage=p.message,
                    pages_done=min(pages_done, p.num_pages),
                    num_chunks=p.chunks,
                    eta_seconds=p.eta,
                    **partial,
                )

        def should_stop() -> bool:
            return doc_id in self._cancel or self._stopping.is_set()

        outcome: tuple = ("stopped",)
        try:
            with contextlib.suppress(KeyError):
                self._update(
                    doc_id, status="processing", stage="Starting", askable=False, pages_indexed=0
                )
            self.pipe.embedder.load()
            index, report = ingest_pdf(
                self._pdf_path(doc_id),
                self.store,
                self.config,
                _Locked(self.pipe.embedder, self.model_lock),
                doc_id=doc_id,
                display_name=rec.filename,
                progress=progress,
                should_stop=should_stop,
                isolate=True,  # parse in worker processes even for short documents
            )
            outcome = ("ready", index, report)
        except IngestCancelled:
            pass  # deleted (cleaned up below) or stopping (resumes on the next start)
        except Exception as exc:
            logger.exception("ingest failed for %s", doc_id)
            message = str(exc) if isinstance(exc, PDFParseError) else f"{type(exc).__name__}: {exc}"
            outcome = ("failed", message)
        finally:
            with self._job_lock:
                self._active = None
                deleted = doc_id in self._cancel
                if deleted:  # deleted while running, at any stage: remove what was written
                    self._remove_files(doc_id, sha)
                    self._cancel.discard(doc_id)
        if deleted or outcome[0] == "stopped":
            return
        if outcome[0] == "failed":
            with contextlib.suppress(KeyError):
                self._update(
                    doc_id,
                    status="failed",
                    stage="",
                    error=outcome[1],
                    eta_seconds=None,
                    askable=False,
                )
            with self._loaded_lock:
                self._partial.pop(doc_id, None)
            return
        _, index, report = outcome
        if index.doc_id != doc_id:
            index = _rename(index, doc_id)
        with self._db_lock:
            self._db.execute(
                "UPDATE documents SET index_dir = ? WHERE doc_id = ?",
                (index.meta.get("index_dir"), doc_id),
            )
            self._db.commit()
        try:
            self._update(
                doc_id,
                status="ready",
                stage="",
                title=display_title(rec.title, report.title),
                num_pages=report.num_pages,
                pages_done=report.num_pages,
                askable=False,
                pages_indexed=report.num_pages,
                num_chunks=report.num_chunks,
                ocr_pages=len(report.ocr_pages),
                warnings=report.warnings,
                ingest_seconds=report.seconds,
                eta_seconds=None,
                error=None,
            )
        except KeyError:
            return
        self._remember(doc_id, index)
        with self._loaded_lock:
            self._partial.pop(doc_id, None)

    # ------------------------------------------------------------------ indexes

    def _remember(self, doc_id: str, index: DocumentIndex) -> None:
        with self._loaded_lock:
            self._loaded[doc_id] = index
            self._loaded.move_to_end(doc_id)
            while len(self._loaded) > 1 and (
                sum(len(i) for i in self._loaded.values()) > MAX_LOADED_CHUNKS
            ):
                self._loaded.popitem(last=False)

    def index(self, doc_id: str) -> DocumentIndex:
        with self._loaded_lock:
            if doc_id in self._loaded:
                self._loaded.move_to_end(doc_id)
                return self._loaded[doc_id]
        rec, index_dir, _ = self._row(doc_id)
        if rec.status != "ready" or not index_dir:
            raise ValueError(f"{rec.filename} is not ready yet ({rec.status})")
        index = load_index(index_dir)
        if index.doc_id != doc_id:
            index = _rename(index, doc_id)
        self._remember(doc_id, index)
        return index

    def _index_dir(self, doc_id: str) -> Path | None:
        """Where pass 2 writes this document's checkpointed index (same path as ingest_pdf)."""
        _, _, sha = self._row(doc_id)
        _, index_dir = store_dirs(self.store, sha, self.config, ocr_available(self.config.ocr))
        return index_dir

    def partial_index(self, doc_id: str) -> DocumentIndex | None:
        """Index over the checkpointed prefix of a document still being ingested, or None.

        Rebuilt when the ingest thread has committed a newer checkpoint, but at most every
        ``PARTIAL_REBUILD_S`` seconds; in between the previous one is reused.  Read-only.
        """
        index_dir = self._index_dir(doc_id)  # KeyError if the document is gone
        state = partial_checkpoint(index_dir)
        if state is None:
            return None
        key = (state["chunk_bytes"], state["vector_bytes"], state["pages_done"])
        cached = self._cached_partial(doc_id, key)
        if cached is not None:
            return cached
        with self._loaded_lock:
            build_lock = self._partial_locks.setdefault(doc_id, threading.Lock())
        with build_lock:  # one build per document: the next waiter reuses its result
            cached = self._cached_partial(doc_id, key)
            if cached is not None:
                return cached
            index = load_partial_index(index_dir)
            if index is None:
                return None
            if index.doc_id != doc_id:
                pages = (index.pages_indexed, index.pages_total)
                index = _rename(index, doc_id)
                index.pages_indexed, index.pages_total = pages
            with self._loaded_lock:
                # KeyError if deleted meanwhile: do not cache a ghost.  ``_ingest`` marks the
                # record ready *before* it drops the cache entry, so a record still
                # 'processing' here means that drop has not happened yet.
                still_processing = self._row(doc_id)[0].status == "processing"
                if still_processing:
                    self._partial[doc_id] = (key, time.monotonic(), index)
        if not still_processing:  # finished while we were building: answer from the final index
            return self.index(doc_id)
        return index

    def _cached_partial(self, doc_id: str, key: tuple) -> DocumentIndex | None:
        with self._loaded_lock:
            cached = self._partial.get(doc_id)
        if cached is not None and (
            cached[0] == key or time.monotonic() - cached[1] < PARTIAL_REBUILD_S
        ):
            return cached[2]
        return None

    def _ask_index(self, doc_id: str) -> DocumentIndex:
        """The final index of a ready document, else the partial one of a processing document."""
        rec = self.get(doc_id)
        if rec.status == "processing":
            index = self.partial_index(doc_id)
            if index is not None:
                return index
        return self.index(doc_id)  # ValueError if not ready (or it just finished: final index)

    # ------------------------------------------------------------------ questions

    def ask(
        self,
        question: str,
        doc_ids: list[str] | None,
        history: list[tuple[str, str]],
        on_step: Callable[[TraceStep], None] | None = None,
        on_token: Callable[[dict[str, Any]], None] | None = None,
        answer_language: str | None = None,
    ) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer(question="", answer="Please enter a question.", refused=True)
        if self.pipe.llm is None:
            raise RuntimeError("No language model configured.")
        if doc_ids is None:
            doc_ids = [r.doc_id for r in self.list_documents() if r.status == "ready"]
        if not doc_ids:
            raise ValueError("No ready documents to search. Upload a PDF first.")
        indexes = [self._ask_index(d) for d in doc_ids]
        partial = next((i for i in indexes if i.pages_indexed is not None), None)
        embedder = _Locked(self.pipe.embedder, self.model_lock)
        reranker = self.pipe.reranker
        retriever = HybridRetriever(
            self.config, embedder, _Locked(reranker, self.model_lock) if reranker else None
        )
        agent = AgenticRAG(self.config, retriever, self.pipe.llm)
        answer = agent.run(
            question[:2000],
            indexes,
            history=history,
            on_step=on_step,
            on_token=on_token,
            answer_language=answer_language,
        )
        if partial is not None:  # at most one document is being ingested at a time
            answer.partial = {
                "doc_id": partial.doc_id,
                "pages_indexed": partial.pages_indexed,
                "pages_total": partial.pages_total,
            }
        return answer

    # ------------------------------------------------------------------ pages

    def render_page(
        self, doc_id: str, page: int, zoom: float = 1.5, chunk_id: str | None = None
    ) -> bytes:
        rec = self.get(doc_id)
        if not 1 <= page <= rec.num_pages:
            raise IndexError(f"page {page} out of range 1..{rec.num_pages}")
        zoom = min(max(zoom, 0.5), 3.0)
        segments = self._highlight_segments(doc_id, page, chunk_id) if chunk_id else []
        with open_pdf(self._pdf_path(doc_id)) as pdf:
            pg = pdf[page - 1]
            for quad in _highlight_quads(pg, segments):
                annot = pg.add_highlight_annot(quad)
                annot.set_colors(stroke=(1.0, 0.85, 0.2))
                annot.update()
            return pg.get_pixmap(matrix=_matrix(zoom), annots=True).tobytes("png")

    def highlight_boxes(self, doc_id: str, page: int, chunk_id: str | None) -> list[list[float]]:
        """Where ``render_page`` paints the highlight: ``[x0, y0, x1, y1]`` boxes as 0-1
        fractions of the rendered page, top to bottom. Same segments, same search."""
        rec = self.get(doc_id)
        if not 1 <= page <= rec.num_pages:
            raise IndexError(f"page {page} out of range 1..{rec.num_pages}")
        segments = self._highlight_segments(doc_id, page, chunk_id) if chunk_id else []
        if not segments:
            return []
        with open_pdf(self._pdf_path(doc_id)) as pdf:
            pg = pdf[page - 1]
            width, height = pg.rect.width, pg.rect.height  # rotated size, as rendered
            boxes = []
            for quad in _highlight_quads(pg, segments):
                r = quad.rect * pg.rotation_matrix
                boxes.append(
                    [
                        round(min(max(v, 0.0), 1.0), 4)
                        for v in (r.x0 / width, r.y0 / height, r.x1 / width, r.y1 / height)
                    ]
                )
        return sorted(boxes, key=lambda b: (b[1], b[0]))

    def _highlight_segments(self, doc_id: str, page: int, chunk_id: str) -> list[str]:
        try:
            index = self._ask_index(doc_id)
        except (ValueError, FileNotFoundError, KeyError):
            return []
        chunk = next((c for c in index.chunks if c.chunk_id == chunk_id), None)
        if chunk is None or chunk.page != page:
            return []
        return highlight_segments(chunk.text, chunk.kind)

    # ------------------------------------------------------------------ status

    def health(self) -> dict[str, Any]:
        llm = self.pipe.llm
        status = "unavailable"
        if llm is not None:
            try:
                status = "ok" if getattr(llm, "is_server_up", lambda: True)() else "unavailable"
            except Exception:
                status = "unavailable"
        docs = self.list_documents()
        return {
            "llm": status,
            "model": (llm.describe() if llm is not None else {}).get("model"),
            "device": getattr(self.pipe.embedder, "device", "cpu"),
            "queue": self._queue.qsize(),
            "busy": self._active is not None,
            "processing": self._active,
            "documents": len(docs),
            "ready": sum(r.status == "ready" for r in docs),
        }


def _highlight_quads(pg, segments: list[str]):
    """Quads of every segment found on ``pg`` (text extracted once, searched many times)."""
    textpage = pg.get_textpage() if segments else None
    for seg in segments:
        yield from pg.search_for(seg, quads=True, textpage=textpage)


def _matrix(zoom: float):
    import pymupdf

    return pymupdf.Matrix(zoom, zoom)


def highlight_segments(text: str, kind: str = "text", max_len: int = 60) -> list[str]:
    """Short exact-text pieces of a passage to search for on its page.

    ``page.search_for`` matches text on one line or across line breaks, but a whole
    320-token passage rarely matches verbatim (dehyphenation, joined lines), so the
    passage is cut into sentence fragments; table passages into their cells.
    """
    if kind == "table":
        # Body rows only: header cells ("Action ID", "Year") repeat in every table on the
        # page and would light up all of them.
        rows = [ln for ln in text.splitlines() if ln.lstrip().startswith("|")]
        body = (
            rows[2:]
            if len(rows) > 2 and set(rows[1].replace("|", "").strip()) <= set("-: ")
            else rows
        )
        cells = [c.strip() for line in body for c in line.split("|")]
        pieces = [c for c in cells if not set(c) <= set("-: ")]
    else:
        pieces = []
        for sentence in re.split(r"(?<=[.!?;:])\s+", text):
            words = sentence.split()
            while words:
                piece, words = words[:8], words[8:]
                pieces.append(" ".join(piece))
    out = []
    min_len = 3 if kind == "table" else 4  # table cells: short numbers like "4.5" count
    for p in pieces:
        p = p[:max_len].strip()
        if len(p) >= min_len and p not in out:
            out.append(p)
    return out[:80]
