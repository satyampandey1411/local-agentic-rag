"""
Per-document index: chunks + dense vectors + BM25, with an on-disk cache.

A flat NumPy matrix is the right tool here: a 300-page PDF yields ~2-3k
chunks, and a (3000 x 768) float32 dot product takes well under a
millisecond.  A vector database would add a dependency and a service with
no measurable benefit at this scale.

The cache key is (PDF sha256, ingestion config fingerprint), so re-running a
Colab cell or re-uploading the same file skips parsing and embedding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from local_agentic_rag.lang import dominant_script
from local_agentic_rag.retrieval.bm25 import BM25Index
from local_agentic_rag.schema import Chunk

logger = logging.getLogger(__name__)


class DocumentIndex:
    def __init__(
        self,
        doc_id: str,
        title: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        meta: dict | None = None,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        self.doc_id = doc_id
        self.title = title
        self.chunks = chunks
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.meta = meta or {}
        self.bm25 = BM25Index([c.embed_text for c in chunks])
        self.script = self.meta.get("script") or dominant_script([c.text for c in chunks])

    def __len__(self) -> int:
        return len(self.chunks)

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
