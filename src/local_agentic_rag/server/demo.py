"""
``DemoLibrary``: a dependency-light stand-in for the real ``Library`` so the web UI
can be developed and demoed without a GPU, embeddings or Ollama.

* Ingestion is real PDF text extraction with PyMuPDF, slowed down to show progress
  (one background worker, like the single-GPU real thing).
* "Answers" are keyword matches over the extracted text blocks with real citations,
  so the page viewer and highlight are exercised end to end.
* Page images are real renders, with the cited block highlighted.

Magic words for UI QA: a filename containing ``fail`` fails half-way (retry then
succeeds); a question containing ``slow`` takes 60 s, ``crash`` raises, and
``unverified`` returns ``grounded=False``.
"""

from __future__ import annotations

import hashlib
import os
import queue
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from local_agentic_rag.library_types import DocRecord
from local_agentic_rag.schema import (
    REFUSAL_MESSAGE,
    Answer,
    Chunk,
    Citation,
    ScoredChunk,
    TraceStep,
)

_MUPDF_LOCK = threading.Lock()  # PyMuPDF is not thread-safe
_WORD = re.compile(r"[\w']{3,}", re.UNICODE)
_STOP = {
    "the",
    "and",
    "for",
    "are",
    "was",
    "what",
    "which",
    "who",
    "how",
    "does",
    "did",
    "this",
    "that",
    "with",
    "from",
    "about",
    "document",
    "according",
    "there",
    "their",
}


@dataclass
class _Block:
    chunk_id: str
    page: int
    text: str
    bbox: tuple[float, float, float, float]


class _Cancelled(Exception):
    pass


class DemoLibrary:
    def __init__(self, root: str | None = None, seconds_per_doc: float = 8.0) -> None:
        self.root = Path(root or tempfile.mkdtemp(prefix="lar-demo-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.seconds_per_doc = seconds_per_doc
        self._lock = threading.RLock()
        self._docs: dict[str, DocRecord] = {}
        self._blocks: dict[str, list[_Block]] = {}
        self._attempts: dict[str, int] = {}
        self._cancel: set[str] = set()
        self._queue: queue.Queue[str] = queue.Queue()
        self._busy = False
        threading.Thread(target=self._worker, name="demo-ingest", daemon=True).start()

    # ------------------------------------------------------------------ protocol
    def submit(self, src_path: str, filename: str) -> DocRecord:
        sha = hashlib.sha256()
        with open(src_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                sha.update(block)
        doc_id = sha.hexdigest()[:12]
        with self._lock:
            if doc_id in self._docs:
                os.unlink(src_path)
                return replace(self._docs[doc_id])
            dest = self.root / f"{doc_id}.pdf"
            shutil.move(src_path, dest)
            now = time.time()
            rec = DocRecord(
                doc_id=doc_id,
                title=Path(filename).stem.replace("_", " "),
                filename=filename,
                status="queued",
                stage="Waiting in line",
                size_bytes=dest.stat().st_size,
                created_at=now,
                updated_at=now,
            )
            self._docs[doc_id] = rec
            self._queue.put(doc_id)
            return replace(rec)

    def list_documents(self) -> list[DocRecord]:
        with self._lock:
            return [replace(d) for d in self._docs.values()]

    def get(self, doc_id: str) -> DocRecord:
        with self._lock:
            return replace(self._docs[doc_id])

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id)  # KeyError if unknown
            self._blocks.pop(doc_id, None)
            self._cancel.add(doc_id)
        (self.root / f"{doc_id}.pdf").unlink(missing_ok=True)

    def retry(self, doc_id: str) -> DocRecord:
        with self._lock:
            rec = self._docs[doc_id]
            if rec.status != "failed":
                raise ValueError("Only failed documents can be retried.")
            self._update(
                doc_id,
                status="queued",
                stage="Waiting in line",
                error=None,
                pages_done=0,
                eta_seconds=None,
            )
            self._queue.put(doc_id)
            return replace(self._docs[doc_id])

    def page_count(self, doc_id: str) -> int:
        import pymupdf

        self.get(doc_id)
        with _MUPDF_LOCK, pymupdf.open(self.root / f"{doc_id}.pdf") as pdf:
            return pdf.page_count

    def render_page(
        self, doc_id: str, page: int, zoom: float = 1.5, chunk_id: str | None = None
    ) -> bytes:
        import pymupdf

        self.get(doc_id)
        with self._lock:
            block = next((b for b in self._blocks.get(doc_id, []) if b.chunk_id == chunk_id), None)
        with _MUPDF_LOCK, pymupdf.open(self.root / f"{doc_id}.pdf") as pdf:
            if not 1 <= page <= pdf.page_count:
                raise IndexError(page)
            pg = pdf[page - 1]
            if block is not None and block.page == page:
                r = pymupdf.Rect(block.bbox) + (-3, -2, 3, 2)
                pg.draw_rect(
                    r, color=(0.95, 0.6, 0.0), fill=(1.0, 0.85, 0.1), fill_opacity=0.35, width=1.2
                )
            return pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png")

    def ask(
        self,
        question: str,
        doc_ids: list[str] | None,
        history: list[tuple[str, str]],
        on_step: Callable[[TraceStep], None] | None = None,
    ) -> Answer:
        t_start = time.perf_counter()
        with self._lock:
            ready = {d.doc_id for d in self._docs.values() if d.status == "ready"}
            if doc_ids is None:
                doc_ids = sorted(ready)
            for d in doc_ids:
                if d not in self._docs:
                    raise KeyError(d)
                if d not in ready:
                    raise ValueError(f"'{self._docs[d].title}' is not ready yet.")
            titles = {d: self._docs[d].title for d in doc_ids}
            blocks = [b for d in doc_ids for b in self._blocks.get(d, [])]
        ans = Answer(question=question, answer="", refused=False)
        lowered = question.lower()

        def step(name: str, seconds: float, **detail: Any) -> None:
            time.sleep(seconds)
            ts = TraceStep(step=name, detail=detail, seconds=seconds)
            ans.trace.append(ts)
            if on_step:
                on_step(ts)

        if "crash" in lowered:
            raise RuntimeError("simulated failure (the question contained 'crash')")
        terms = [w for w in _WORD.findall(lowered) if w not in _STOP]
        ans.standalone_question = question
        ans.search_queries = [question, " ".join(terms)] if terms else [question]
        step("plan", 0.5, search_queries=ans.search_queries)
        if "slow" in lowered:
            time.sleep(60)
        scored = []
        for b in blocks:
            text = b.text.lower()
            score = sum(text.count(t) for t in terms)
            if score:
                scored.append((score, b))
        scored.sort(key=lambda x: (-x[0], x[1].page))
        top = scored[:4]
        ans.retrieved = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=b.chunk_id,
                    doc_id=b.chunk_id.split("-p")[0],
                    page=b.page,
                    section="",
                    text=b.text,
                    doc_title=titles.get(b.chunk_id[:12], ""),
                ),
                score=float(s),
                dense_rank=i + 1,
                bm25_rank=i + 1,
            )
            for i, (s, b) in enumerate(top)
        ]
        step(
            "retrieve",
            0.7,
            round=1,
            pages=[b.page for _, b in top],
            top_score=float(top[0][0]) if top else None,
        )
        if not top:
            ans.refused = True
            ans.answer = REFUSAL_MESSAGE
            step("refuse", 0.0, reason="no relevant passages")
        else:
            cited = top[:2]
            ans.citations = [
                Citation(
                    marker=i + 1,
                    chunk_id=b.chunk_id,
                    doc_id=b.chunk_id[:12],
                    page=b.page,
                    snippet=b.text[:200],
                )
                for i, (_, b) in enumerate(cited)
            ]
            parts = [f"“{_shorten(b.text, 160)}” [{i + 1}]" for i, (_, b) in enumerate(cited)]
            ans.answer = (
                "**Demo mode** - no language model is running, so this reply quotes the "
                "passages that best match your question:\n\n- " + "\n- ".join(parts)
            )
            step("generate", 1.0, chars=len(ans.answer), preview=ans.answer[:120])
            ans.grounded = "unverified" not in lowered
            if not ans.grounded:
                ans.unsupported_claims = ["(simulated) a statement the verifier could not find"]
            step("verify", 0.5, output={"grounded": ans.grounded})
        ans.llm_calls = 0
        ans.latency_sec = round(time.perf_counter() - t_start, 3)
        return ans

    def health(self) -> dict[str, Any]:
        with self._lock:
            queued = sum(1 for d in self._docs.values() if d.status == "queued")
            return {
                "llm": "ok",
                "device": "cpu",
                "queue": queued,
                "busy": self._busy,
                "mode": "demo",
                "model": "demo (no LLM)",
            }

    # ------------------------------------------------------------------ worker
    def _update(self, doc_id: str, **fields: Any) -> None:
        with self._lock:
            rec = self._docs.get(doc_id)
            if rec is None:
                raise _Cancelled
            for k, v in fields.items():
                setattr(rec, k, v)
            rec.updated_at = time.time()

    def _worker(self) -> None:
        while True:
            doc_id = self._queue.get()
            with self._lock:
                if doc_id in self._cancel or doc_id not in self._docs:
                    self._cancel.discard(doc_id)
                    continue
                self._busy = True
            try:
                self._ingest(doc_id)
            except _Cancelled:
                pass
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    if doc_id in self._docs:
                        self._update(
                            doc_id,
                            status="failed",
                            stage="Failed",
                            error=f"{type(exc).__name__}: {exc}",
                            eta_seconds=None,
                        )
            finally:
                with self._lock:
                    self._busy = False
                    self._cancel.discard(doc_id)

    def _ingest(self, doc_id: str) -> None:
        import pymupdf

        t0 = time.time()
        self._attempts[doc_id] = self._attempts.get(doc_id, 0) + 1
        rec = self.get(doc_id)
        simulate_failure = "fail" in rec.filename.lower() and self._attempts[doc_id] == 1
        self._update(doc_id, status="processing", stage="Opening PDF", pages_done=0)
        path = str(self.root / f"{doc_id}.pdf")
        with _MUPDF_LOCK:
            pdf = pymupdf.open(path)
            if pdf.needs_pass:
                pdf.close()
                raise ValueError("This PDF is password-protected.")
            n = pdf.page_count
            title = (pdf.metadata or {}).get("title") or rec.title
        self._update(doc_id, num_pages=n, title=title.strip() or rec.title)
        blocks: list[_Block] = []
        batch = max(1, n // 40)
        delay = self.seconds_per_doc / max(1, -(-n // batch))
        try:
            for start in range(0, n, batch):
                if doc_id in self._cancel:
                    raise _Cancelled
                end = min(n, start + batch)
                self._update(doc_id, stage=f"Reading pages {start + 1:,}-{end:,} of {n:,}")
                with _MUPDF_LOCK:
                    for pno in range(start, end):
                        for i, b in enumerate(pdf[pno].get_text("blocks")):
                            text = " ".join(str(b[4]).split())
                            if b[6] == 0 and len(text) > 20:
                                blocks.append(
                                    _Block(
                                        f"{doc_id}-p{pno + 1}-b{i}",
                                        pno + 1,
                                        text,
                                        (b[0], b[1], b[2], b[3]),
                                    )
                                )
                time.sleep(delay)
                if simulate_failure and end >= n / 2:
                    raise RuntimeError("simulated failure (filename contains 'fail') - press Retry")
                elapsed = time.time() - t0
                eta = elapsed / end * (n - end)
                self._update(
                    doc_id, pages_done=end, num_chunks=len(blocks), eta_seconds=round(eta, 1)
                )
        finally:
            with _MUPDF_LOCK:
                pdf.close()
        warnings = [] if blocks else ["No text found - this PDF may be scanned images."]
        with self._lock:
            self._blocks[doc_id] = blocks
        self._update(
            doc_id,
            status="ready",
            stage="Ready",
            pages_done=n,
            num_chunks=len(blocks),
            ingest_seconds=round(time.time() - t0, 2),
            eta_seconds=None,
            warnings=warnings,
        )


def _shorten(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"
