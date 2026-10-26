"""
Per-document index: chunks + dense vectors + BM25, with an on-disk cache.

A flat NumPy matrix is the right tool here, even for 10,000 pages: ~40k chunks
x 1024 dims is 160 MB in float32 and one exact dot product per query (FAISS's own
guidance is that only flat indexes are exact; IVF needs more training vectors than
a single document has).  Streaming ingestion (``ingest/streaming.py``) stores the
vectors as float16 and the BM25 matrix on disk; this class is the in-memory view.

The cache key is (PDF sha256, ingestion config fingerprint), so re-running a
Colab cell or re-uploading the same file skips parsing and embedding.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from local_agentic_rag.lang import dominant_script
from local_agentic_rag.retrieval.bm25 import BM25Index
from local_agentic_rag.schema import Chunk

logger = logging.getLogger(__name__)


#: A numbered section heading: "§ 164. Taxes", "Sec. 7203.", "Section 2.1 Scope", "4.3 Results".
#: A bare integer does not count ("2010 (I) In general" is a year, not section 2010).
_SECTION_HEAD_RE = re.compile(
    r"^\s*(?:(?:§+\s*|(?:section|sec\.)\s+)(\d+[A-Za-z]?(?:\.\d+)*)|(\d+(?:\.\d+)+))"
    r"(?![\d])(?!\.\d)[.:\s]",
    re.IGNORECASE,
)
#: A section start inside running text, as codes print it: "§ 164. Taxes".  Cross-references
#: are written "section 164" and amendment notes "§ 221(a)(2)", so neither matches.
_INLINE_SECTION_RE = re.compile(r"(?:^|\s)§\s*(\d+[A-Za-z]?)\.\s+[A-Z]")


class DocumentIndex:
    def __init__(
        self,
        doc_id: str,
        title: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        meta: dict | None = None,
        bm25: BM25Index | None = None,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        self.doc_id = doc_id
        self.title = title
        self.chunks = chunks
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.meta = meta or {}
        self.bm25 = bm25 if bm25 is not None else BM25Index([c.embed_text for c in chunks])
        self.script = self.meta.get("script") or dominant_script([c.text for c in chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def section_chunks(self, number: str) -> list[int]:
        """Indices of the chunks inside numbered section ``number`` ("164", "2.1").

        A chunk belongs to the last numbered heading seen before it, in document order
        ("§164. Taxes", "Section 2.1 Scope", "4.3 Results"), including chunks under
        unnumbered sub-headings such as "(b) Definitions"."""
        if not hasattr(self, "_by_section"):
            by_section: dict[str, list[int]] = {}
            current = None
            heading = None
            for i, c in enumerate(self.chunks):
                if c.section != heading:  # a new heading: does it start a numbered section?
                    heading = c.section
                    m = _SECTION_HEAD_RE.match(heading)
                    if m:
                        current = (m.group(1) or m.group(2)).lower()
                starts = [x.lower() for x in _INLINE_SECTION_RE.findall(c.text)]
                if current is not None:
                    by_section.setdefault(current, []).append(i)
                for key in starts:  # sections that begin inside this chunk's text
                    if key != current:
                        by_section.setdefault(key, []).append(i)
                    current = key
            self._by_section = by_section
        return self._by_section.get(number.lower(), [])

    def page_chunks(self, page: int) -> list[int]:
        """Indices of the chunks on a physical page (built on first use)."""
        if not hasattr(self, "_by_page"):
            by_page: dict[int, list[int]] = {}
            for i, c in enumerate(self.chunks):
                by_page.setdefault(c.page, []).append(i)
            self._by_page = by_page
        return self._by_page.get(page, [])

    def dense_search(self, query_vec: np.ndarray, n: int) -> list[tuple[int, float]]:
        if len(self.chunks) == 0:
            return []
        scores = self.embeddings @ query_vec.astype(np.float32)
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def bm25_search(self, query: str, n: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, n)

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        payload = {
            "doc_id": self.doc_id,
            "title": self.title,
            "meta": self.meta,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        tmp = path / "index.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path / "index.json")  # atomic: a crash never leaves a half-written cache

    @classmethod
    def load(cls, directory: str | Path) -> DocumentIndex:
        path = Path(directory)
        payload = json.loads((path / "index.json").read_text(encoding="utf-8"))
        embeddings = np.load(path / "embeddings.npy")
        chunks = [Chunk.from_dict(c) for c in payload["chunks"]]
        return cls(payload["doc_id"], payload["title"], chunks, embeddings, payload.get("meta"))

    @staticmethod
    def exists(directory: str | Path) -> bool:
        path = Path(directory)
        return (path / "index.json").is_file() and (path / "embeddings.npy").is_file()
