"""
High-level facade: ``RAGPipeline.ingest(pdf)`` then ``RAGPipeline.ask(question)``.

    pipe = RAGPipeline(RAGConfig(), llm=OllamaLLM("qwen2.5:7b-instruct"))
    report = pipe.ingest("paper.pdf")
    answer = pipe.ask("What dataset did they use?")
    print(answer.answer, answer.cited_pages)

Models are loaded lazily and shared between pipelines created with
``pipe.variant(**overrides)``, so ablation runs do not reload weights.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from local_agentic_rag.agent.agent import AgenticRAG, History
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.ingest.streaming import ProgressFn, ingest_pdf
from local_agentic_rag.llm.base import LLM
from local_agentic_rag.retrieval.embedder import Embedder
from local_agentic_rag.retrieval.hybrid import HybridRetriever
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.retrieval.reranker import Reranker
from local_agentic_rag.schema import Answer, IngestReport, ScoredChunk, TraceStep

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
DEFAULT_CACHE_DIR = os.environ.get(
    "LOCAL_RAG_CACHE", str(Path.home() / ".cache" / "local_agentic_rag" / "indexes")
)


class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig | None = None,
        llm: LLM | None = None,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        *,
        _shared: dict | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.llm = llm
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # Model + index registry shared across variants (keyed by model name).
        self._shared = (
            _shared if _shared is not None else {"embedders": {}, "rerankers": {}, "indexes": {}}
        )

    # ------------------------------------------------------------------ components

    @property
    def embedder(self) -> Embedder:
        name = self.config.embedder_model
        if name not in self._shared["embedders"]:
            self._shared["embedders"][name] = Embedder(name, device=self.config.device)
        return self._shared["embedders"][name]

    @property
    def reranker(self) -> Reranker | None:
        name = self.config.reranker_model
        if not name:
            return None
        if name not in self._shared["rerankers"]:
            self._shared["rerankers"][name] = Reranker(name, device=self.config.device)
        return self._shared["rerankers"][name]

    @property
    def indexes(self) -> dict[str, DocumentIndex]:
        """doc_id → index for the current embedder *and* chunking settings."""
        return self._shared["indexes"].setdefault(self.config.ingestion_fingerprint(), {})

    def load_models(self) -> None:
        self.embedder.load()
        if self.reranker is not None:
            self.reranker.load()

    def variant(self, **overrides) -> RAGPipeline:
        """New pipeline with config overrides that reuses loaded models and indexes."""
        return RAGPipeline(
            self.config.with_overrides(**overrides), self.llm, self.cache_dir, _shared=self._shared
        )

    # ------------------------------------------------------------------ ingestion

    def ingest(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
        use_cache: bool = True,
        progress: ProgressFn | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> IngestReport:
        """Parse, chunk, embed and index a PDF (streaming, resumable; see ``streaming.py``).

        With a cache directory an interrupted ingest resumes where it stopped, and the
        parsed pages are shared by variants that only change chunking or the embedder.
        """
        embedder = self.embedder
        embedder.load()
        index, report = ingest_pdf(
            pdf_path,
            self.cache_dir,
            self.config,
            embedder,
            doc_id=doc_id,
            use_cache=use_cache,
            progress=progress,
            should_stop=should_stop,
        )
        if doc_id and doc_id != index.doc_id:
            index = _rename(index, doc_id)
            report = replace(report, doc_id=doc_id)
        self.indexes[index.doc_id] = index
        return report

    # ------------------------------------------------------------------ querying

    def _select(self, doc_id: str | list[str] | None) -> list[DocumentIndex]:
        if not self.indexes:
            raise RuntimeError("No document ingested yet — call ingest(pdf_path) first.")
        if doc_id is None:
            return list(self.indexes.values())
        ids = [doc_id] if isinstance(doc_id, str) else doc_id
        missing = [d for d in ids if d not in self.indexes]
        if missing:
            raise KeyError(f"Unknown doc_id(s) {missing}; ingested: {sorted(self.indexes)}")
        return [self.indexes[d] for d in ids]

    def retriever(self) -> HybridRetriever:
        return HybridRetriever(self.config, self.embedder, self.reranker)

    def retrieve(
        self, question: str, doc_id: str | list[str] | None = None, top_k: int | None = None
    ) -> list[ScoredChunk]:
        return self.retriever().retrieve([question], self._select(doc_id), top_k=top_k)

    def ask(
        self,
        question: str,
        doc_id: str | list[str] | None = None,
        history: History | None = None,
        on_step: Callable[[TraceStep], None] | None = None,
        on_token: Callable[[dict[str, Any]], None] | None = None,
    ) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer(
                question="", answer="Please enter a question.", refused=True, error="empty question"
            )
        if len(question) > MAX_QUESTION_CHARS:
            question = question[:MAX_QUESTION_CHARS]
        if self.llm is None:
            raise RuntimeError("No LLM configured — pass llm=OllamaLLM(...) to RAGPipeline.")
        indexes = self._select(doc_id)
        agent = AgenticRAG(self.config, self.retriever(), self.llm)
        return agent.run(question, indexes, history=history, on_step=on_step, on_token=on_token)


def _rename(index: DocumentIndex, doc_id: str) -> DocumentIndex:
    chunks = [
        replace(c, doc_id=doc_id, chunk_id=c.chunk_id.replace(c.doc_id, doc_id, 1))
        for c in index.chunks
    ]
    return DocumentIndex(doc_id, index.title, chunks, index.embeddings, index.meta, index.bm25)
