"""
Streaming, resumable ingestion for documents of any length.

    PDF ──pass 1──▶ raw pages        parsed/<sha>-<parse settings>/raw.jsonl
        ──pass 2──▶ chunks + vectors <sha>-<ingest settings>/chunks.jsonl, vectors.f16
        ──final───▶ keyword index    bm25.npz, bm25_vocab.json, meta.json

Pass 1 extracts pages in parallel, crash-isolated worker processes.  Pass 2 removes
running headers/footers, chunks and embeds page batches.  It needs two document-wide
statistics from pass 1 (margin lines repeating on 40% of pages, the body font size),
which is why there are two passes.  The output is identical to parsing the whole PDF
in memory (checked against the benchmark PDFs chunk by chunk).

Both passes append to their files and checkpoint after every batch (data fsynced
first, then an atomic ``state.json`` replace).  A crash, a cancelled job or a closed
notebook resumes from the last checkpoint; files are truncated to the checkpointed
byte offsets, which discards a half-written batch.

Memory stays bounded: pass 1 keeps a few page batches in flight, pass 2 one batch of
pages; vectors go to disk as float16 as soon as they are computed.  The parsed pages
are shared by every embedder/chunking variant of the same PDF.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from local_agentic_rag.config import INGEST_VERSION, RAGConfig
from local_agentic_rag.ingest.chunker import PageChunker
from local_agentic_rag.ingest.pdf import (
    PageFinaliser,
    PageStats,
    PDFParseError,
    RawPage,
    _guess_title,
    _slugify,
    check_parse_failures,
    iter_raw_batches,
    no_text_message,
    ocr_available,
    open_pdf,
    sha256_file,
)
from local_agentic_rag.retrieval.bm25 import BM25Index
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.schema import Chunk, IngestReport

logger = logging.getLogger(__name__)

FORMAT = 1
#: Pages per pass-2 batch: one embedding call and one checkpoint each.
INDEX_BATCH_PAGES = 48


@dataclass
class IngestProgress:
    stage: str  # "reading" | "indexing" | "finalising" | "done"
    pages_done: int
    num_pages: int
    chunks: int = 0
    elapsed: float = 0.0
    eta: float | None = None  # seconds left in the whole job (estimate)

    @property
    def message(self) -> str:
        if self.stage == "reading":
            return f"Reading page {self.pages_done:,} of {self.num_pages:,}"
        if self.stage == "indexing":
            return f"Indexing page {self.pages_done:,} of {self.num_pages:,}"
        if self.stage == "finalising":
            return f"Building the keyword index ({self.chunks:,} passages)"
        return f"Ready: {self.num_pages:,} pages, {self.chunks:,} passages"


ProgressFn = Callable[[IngestProgress], None]


class IngestCancelled(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Small file helpers
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    """Make a new or renamed directory entry durable (POSIX; a no-op elsewhere)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic: readers see the old or the new state, never half
    _fsync_dir(path.parent)


def _jsonl(records) -> bytes:
    # errors="replace": a lone surrogate in extracted text must not fail the document.
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode(
        "utf-8", errors="replace"
    )


def _file_holds(path: Path, size: int) -> bool:
    """The data file a checkpoint refers to still exists with at least that many bytes."""
    return size == 0 or (path.is_file() and path.stat().st_size >= size)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class _AppendFile:
    """Binary append-only file, truncated on open to the last checkpointed size."""

    def __init__(self, path: Path, size: int) -> None:
        new = not path.exists()
        self.f = open(path, "w+b" if new else "r+b")  # noqa: SIM115
        self.f.truncate(size)
        self.f.seek(size)
        if new:
            _fsync_dir(path.parent)

    def write(self, data: bytes) -> None:
        self.f.write(data)

    def sync(self) -> int:
        self.f.flush()
        os.fsync(self.f.fileno())
        return self.f.tell()

    def close(self) -> None:
        self.f.close()


def _iter_jsonl(path: Path, size: int, skip: int = 0) -> Iterator[dict]:
    """Records of a checkpointed JSONL file (only the first ``size`` bytes count)."""
    with open(path, "rb") as f:
        read = 0
        for i, line in enumerate(f):
            read += len(line)
            if read > size:
                return
            if i >= skip:
                yield json.loads(line)


def _eta(done: int, total: int, started: float, done_at_start: int) -> float | None:
    rate_done = done - done_at_start
    elapsed = time.perf_counter() - started
    if rate_done <= 0 or elapsed <= 0:
        return None
    return (total - done) * elapsed / rate_done


# ---------------------------------------------------------------------------
# Store locations
# ---------------------------------------------------------------------------


def parse_fingerprint(config: RAGConfig, use_ocr: bool) -> str:
    return config.fingerprint(("extract_tables", "ocr", "ocr_language")) + (
        "-ocr" if use_ocr else "-noocr"
    )


def store_dirs(root: Path, sha: str, config: RAGConfig, use_ocr: bool) -> tuple[Path, Path]:
    """(parsed-pages dir, index dir) for one PDF under one configuration."""
    parse_dir = (
        root / "parsed" / f"{sha[:16]}-{INGEST_VERSION}-{parse_fingerprint(config, use_ocr)}"
    )
    suffix = "" if use_ocr or config.ocr == "off" else "-noocr"
    return parse_dir, root / f"{sha[:16]}-{config.ingestion_fingerprint()}{suffix}"


def index_complete(index_dir: Path) -> bool:
    state = _read_json(index_dir / "state.json")
    return bool(state and state.get("complete") and state.get("format") == FORMAT)


def load_index(index_dir: str | Path) -> DocumentIndex:
    index_dir = Path(index_dir)
    state = _read_json(index_dir / "state.json")
    if not state or not state.get("complete"):
        raise FileNotFoundError(f"No complete index in {index_dir}")
    meta = _read_json(index_dir / "meta.json") or {}
    chunks = [
        Chunk.from_dict(c) for c in _iter_jsonl(index_dir / "chunks.jsonl", state["chunk_bytes"])
    ]
    n, dim = state["n_chunks"], state.get("dim") or 0
    if n:
        vectors = np.fromfile(index_dir / "vectors.f16", dtype=np.float16, count=n * dim)
        embeddings = vectors.reshape(n, dim).astype(np.float32)
    else:
        embeddings = np.zeros((0, dim), dtype=np.float32)
    bm25 = BM25Index.load(index_dir) if BM25Index.exists(index_dir) else None
    meta["index_dir"] = str(index_dir)
    return DocumentIndex(state["doc_id"], state["title"], chunks, embeddings, meta, bm25=bm25)


# ---------------------------------------------------------------------------
# Pass 1
# ---------------------------------------------------------------------------


def _parse_pass(
    pdf_path: Path,
    parse_dir: Path,
    num_pages: int,
    config: RAGConfig,
    use_ocr: bool,
    workers: int | None,
    progress: ProgressFn | None,
    should_stop: Callable[[], bool] | None,
    t0: float,
    isolate: bool = False,
) -> dict:
    parse_dir.mkdir(parents=True, exist_ok=True)
    state_path = parse_dir / "state.json"
    state = _read_json(state_path)
    if (
        not state
        or state.get("format") != FORMAT
        or state.get("num_pages") != num_pages
        or not _file_holds(parse_dir / "raw.jsonl", state.get("bytes", 0))  # lost on power loss
    ):
        state = {"format": FORMAT, "num_pages": num_pages, "pages_done": 0, "bytes": 0}
        state.update(complete=False, seconds=0.0)
    if state["complete"]:
        return state
    out = _AppendFile(parse_dir / "raw.jsonl", state["bytes"])
    started, done_at_start = time.perf_counter(), state["pages_done"]
    state["_counted"] = 0.0  # seconds of this session already added to state["seconds"]
    if progress:  # show the stage at once; the first batch can take a while
        progress(IngestProgress("reading", state["pages_done"], num_pages, elapsed=0.0))
    try:
        for batch in iter_raw_batches(
            pdf_path,
            state["pages_done"],
            num_pages,
            extract_tables=config.extract_tables,
            use_ocr=use_ocr,
            ocr_language=config.ocr_language,
            workers=workers,
            isolate=isolate,
        ):
            out.write(_jsonl(batch))
            state["bytes"] = out.sync()
            state["pages_done"] += len(batch)
            state["seconds"] += time.perf_counter() - started - state.get("_counted", 0.0)
            state["_counted"] = time.perf_counter() - started
            _write_json(state_path, state)
            if progress:
                eta = _eta(state["pages_done"], num_pages, started, done_at_start)
                # Indexing is measured at roughly the same pace again (embedding on GPU).
                progress(
                    IngestProgress(
                        "reading",
                        state["pages_done"],
                        num_pages,
                        elapsed=time.perf_counter() - t0,
                        eta=None if eta is None else eta * 2,
                    )
                )
            if should_stop and should_stop():
                raise IngestCancelled("cancelled")
    finally:
        out.close()
    state.pop("_counted", None)
    state["complete"] = True
    _write_json(state_path, state)
    return state


# ---------------------------------------------------------------------------
# Pass 2 + finalise
# ---------------------------------------------------------------------------


def _index_pass(
    pdf_path: Path,
    parse_state: dict,
    parse_dir: Path,
    index_dir: Path,
    *,
    doc_id: str,
    title_path: Path,
    metadata: dict,
    config: RAGConfig,
    use_ocr: bool,
    embedder,
    embed_lock,
    progress: ProgressFn | None,
    should_stop: Callable[[], bool] | None,
    t0: float,
) -> dict:
    index_dir.mkdir(parents=True, exist_ok=True)
    state_path = index_dir / "state.json"
    num_pages = parse_state["num_pages"]
    raw_path, raw_size = parse_dir / "raw.jsonl", parse_state["bytes"]

    # Document-wide statistics (one streaming read of the parsed pages).
    stats, parse_warnings = PageStats(), []
    for d in _iter_jsonl(raw_path, raw_size):
        rp = RawPage.from_dict(d)
        stats.add(rp)
        if rp.warning:
            parse_warnings.append(rp.warning)

    state = _read_json(state_path)
    if (
        not state
        or state.get("format") != FORMAT
        or not _file_holds(index_dir / "chunks.jsonl", state.get("chunk_bytes", 0))
        or not _file_holds(index_dir / "vectors.f16", state.get("vector_bytes", 0))
    ):
        state = {
            "format": FORMAT,
            "doc_id": doc_id,
            "title": None,
            "num_pages": num_pages,
            "pages_done": 0,
            "n_chunks": 0,
            "n_table_chunks": 0,
            "chunk_bytes": 0,
            "vector_bytes": 0,
            "dim": None,
            "section": "",
            "has_content": False,
            "empty_pages": [],
            "ocr_pages": [],
            "needs_ocr": [],
            "embed_seconds": 0.0,
            "seconds": 0.0,
            "complete": False,
        }
    if state["complete"]:
        return state

    finaliser = PageFinaliser(stats, config.strip_headers_footers)
    finaliser._first = state["pages_done"] == 0
    finaliser.has_content = state["has_content"]
    finaliser.needs_ocr = list(state["needs_ocr"])
    chunk_tokens = min(config.chunk_tokens, embedder.max_tokens - 32)  # room for header/prefix
    chunker = PageChunker(
        state["doc_id"],
        state["title"] or "",
        chunk_tokens=chunk_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
        min_chunk_tokens=config.min_chunk_tokens,
        count_tokens=embedder.count_tokens,
        section=state["section"],
    )
    chunks_out = _AppendFile(index_dir / "chunks.jsonl", state["chunk_bytes"])
    vectors_out = _AppendFile(index_dir / "vectors.f16", state["vector_bytes"])
    started, done_at_start = time.perf_counter(), state["pages_done"]
    state["_counted"] = 0.0  # seconds of this session already added to state["seconds"]
    pending: list[Chunk] = []
    pages_in_batch = 0

    def flush() -> None:
        nonlocal pending, pages_in_batch
        if pending:
            t = time.perf_counter()
            with embed_lock or contextlib.nullcontext():
                vectors = embedder.encode_passages([c.embed_text for c in pending])
            state["embed_seconds"] += time.perf_counter() - t
            state["dim"] = int(vectors.shape[1])
            vectors_out.write(np.ascontiguousarray(vectors, dtype=np.float16).tobytes())
            chunks_out.write(_jsonl(c.to_dict() for c in pending))
        state["vector_bytes"] = vectors_out.sync()
        state["chunk_bytes"] = chunks_out.sync()
        state["n_chunks"] += len(pending)
        state["n_table_chunks"] += sum(c.kind == "table" for c in pending)
        state["pages_done"] += pages_in_batch
        state["section"] = chunker.section
        state["has_content"] = finaliser.has_content
        state["needs_ocr"] = finaliser.needs_ocr
        state["seconds"] += time.perf_counter() - started - state.get("_counted", 0.0)
        state["_counted"] = time.perf_counter() - started
        _write_json(state_path, state)
        pending, pages_in_batch = [], 0
        if progress:
            progress(
                IngestProgress(
                    "indexing",
                    state["pages_done"],
                    num_pages,
                    chunks=state["n_chunks"],
                    elapsed=time.perf_counter() - t0,
                    eta=_eta(state["pages_done"], num_pages, started, done_at_start),
                )
            )
        if should_stop and should_stop():
            raise IngestCancelled("cancelled")

    try:
        for d in _iter_jsonl(raw_path, raw_size, skip=state["pages_done"]):
            rp = RawPage.from_dict(d)
            page = finaliser(rp)
            if state["title"] is None:  # page 1: the title is part of every chunk's header
                state["title"] = _guess_title(metadata, finaliser.font_title, title_path)
                chunker.doc_title = state["title"]
            if not page.blocks and not page.tables:
                state["empty_pages"].append(page.page)
            if page.ocr_applied:
                state["ocr_pages"].append(page.page)
            pending.extend(chunker.add_page(page))
            pages_in_batch += 1
            if pages_in_batch >= INDEX_BATCH_PAGES:
                flush()
        flush()
    finally:
        chunks_out.close()
        vectors_out.close()

    check_parse_failures(parse_warnings, num_pages, title_path.name)
    if not state["has_content"]:
        raise PDFParseError(no_text_message(title_path.name))

    if progress:
        progress(
            IngestProgress(
                "finalising",
                num_pages,
                num_pages,
                chunks=state["n_chunks"],
                elapsed=time.perf_counter() - t0,
                eta=None,
            )
        )
    t = time.perf_counter()
    texts = [
        Chunk.from_dict(c).embed_text
        for c in _iter_jsonl(index_dir / "chunks.jsonl", state["chunk_bytes"])
    ]
    BM25Index(texts).save(index_dir)
    state["bm25_seconds"] = time.perf_counter() - t
    state.pop("_counted", None)
    warnings = parse_warnings + finaliser.summary_warnings(use_ocr)
    meta = {
        "num_pages": num_pages,
        "sha256": sha256_file(pdf_path),
        "source_path": str(pdf_path),
        "empty_pages": state["empty_pages"],
        "ocr_pages": state["ocr_pages"],
        "warnings": warnings,
        "config": config.ingestion_fingerprint(),
        "parse_seconds": parse_state["seconds"],
        "index_seconds": state["seconds"],
        "embed_seconds": state["embed_seconds"],
        "bm25_seconds": state["bm25_seconds"],
    }
    _write_json(index_dir / "meta.json", meta)
    state["complete"] = True
    _write_json(state_path, state)
    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def ingest_pdf(
    pdf_path: str | Path,
    store_root: str | Path | None,
    config: RAGConfig,
    embedder,
    *,
    doc_id: str | None = None,
    display_name: str | None = None,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
    workers: int | None = None,
    embed_lock=None,
    isolate: bool = False,
) -> tuple[DocumentIndex, IngestReport]:
    """Parse, chunk, embed and index one PDF, resuming any earlier partial run.

    ``store_root=None`` builds in a temporary directory that is deleted afterwards
    (nothing is cached).  ``display_name`` is the original file name, used for the
    document id and title fallback when the PDF is stored under another name.
    ``isolate=True`` parses even short documents in worker processes, so a page that
    crashes MuPDF cannot take down a long-running server.
    """
    t0 = time.perf_counter()
    pdf_path = Path(pdf_path)
    with open_pdf(pdf_path) as pdf:
        num_pages, metadata = pdf.page_count, pdf.metadata or {}
    sha = sha256_file(pdf_path)
    title_path = Path(display_name) if display_name else pdf_path
    doc_id = doc_id or _slugify(title_path.stem)
    use_ocr = ocr_available(config.ocr)

    tmp = None
    if store_root is None:
        tmp = tempfile.TemporaryDirectory(prefix="local-rag-")
        store_root = tmp.name
    try:
        parse_dir, index_dir = store_dirs(Path(store_root), sha, config, use_ocr)
        if not use_cache:
            shutil.rmtree(index_dir, ignore_errors=True)
            shutil.rmtree(parse_dir, ignore_errors=True)
        cached = index_complete(index_dir)
        resumed_from = 0
        if not cached:
            parse_state = _read_json(parse_dir / "state.json") or {}
            index_state = _read_json(index_dir / "state.json") or {}
            resumed_from = index_state.get("pages_done") or parse_state.get("pages_done") or 0
            if resumed_from:
                logger.info("Resuming %s from page %d", pdf_path.name, resumed_from + 1)
            parse_state = _parse_pass(
                pdf_path,
                parse_dir,
                num_pages,
                config,
                use_ocr,
                workers,
                progress,
                should_stop,
                t0,
                isolate=isolate,
            )
            _index_pass(
                pdf_path,
                parse_state,
                parse_dir,
                index_dir,
                doc_id=doc_id,
                title_path=title_path,
                metadata=metadata,
                config=config,
                use_ocr=use_ocr,
                embedder=embedder,
                embed_lock=embed_lock,
                progress=progress,
                should_stop=should_stop,
                t0=t0,
            )
        index = load_index(index_dir)
    finally:
        if tmp is not None:
            tmp.cleanup()

    meta = index.meta
    report = IngestReport(
        doc_id=index.doc_id,
        title=index.title,
        num_pages=meta.get("num_pages", num_pages),
        num_chunks=len(index),
        num_table_chunks=sum(c.kind == "table" for c in index.chunks),
        ocr_pages=meta.get("ocr_pages", []),
        empty_pages=meta.get("empty_pages", []),
        seconds=time.perf_counter() - t0,
        cached=cached,
        warnings=meta.get("warnings", []),
        resumed_from_page=resumed_from,
    )
    if progress:
        progress(
            IngestProgress(
                "done", num_pages, num_pages, chunks=len(index), elapsed=report.seconds, eta=0.0
            )
        )
    return index, report
