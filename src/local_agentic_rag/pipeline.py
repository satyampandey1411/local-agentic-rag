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
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from local_agentic_rag.agent.agent import AgenticRAG, History
from local_agentic_rag.config import INGEST_VERSION, RAGConfig
from local_agentic_rag.ingest.chunker import chunk_document
from local_agentic_rag.ingest.pdf import parse_pdf, sha256_file
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
        self, pdf_path: str | Path, doc_id: str | None = None, use_cache: bool = True
    ) -> IngestReport:
        t0 = time.perf_counter()
        cfg = self.config
        sha = sha256_file(pdf_path)
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / f"{sha[:16]}-{cfg.ingestion_fingerprint()}"

        if use_cache and cache is not None and DocumentIndex.exists(cache):
            index = DocumentIndex.load(cache)
            if doc_id and doc_id != index.doc_id:
                index = _rename(index, doc_id)
            self.indexes[index.doc_id] = index
            meta = index.meta
            return IngestReport(
                doc_id=index.doc_id,
                title=index.title,
                num_pages=meta.get("num_pages", 0),
                num_chunks=len(index),
                num_table_chunks=sum(c.kind == "table" for c in index.chunks),
                ocr_pages=meta.get("ocr_pages", []),
                empty_pages=meta.get("empty_pages", []),
                seconds=time.perf_counter() - t0,
                cached=True,
                warnings=meta.get("warnings", []),
            )

        # Parsing does not depend on the embedder, so variants (e.g. the retrieval ablation's
        # MiniLM / BGE-en / BGE-M3 indexes) share one parse per PDF.  Measured on Colab's
        # 2 vCPUs: the table-heavy BEA report alone takes ~37 s to parse.
        parse_key = (
            sha,
            str(Path(pdf_path).resolve()),
            cfg.extract_tables,
            cfg.strip_headers_footers,
            cfg.ocr,
            cfg.ocr_language,
            INGEST_VERSION,
        )
        parsed = self._shared.setdefault("parsed", {})
        doc = parsed.get(parse_key)
        if doc is None:
            doc = parse_pdf(
                pdf_path,
                doc_id,
                extract_tables=cfg.extract_tables,
                strip_headers_footers=cfg.strip_headers_footers,
                ocr=cfg.ocr,
                ocr_language=cfg.ocr_language,
            )
            parsed[parse_key] = doc
        if doc_id and doc.doc_id != doc_id:
            doc = replace(doc, doc_id=doc_id)
        embedder = self.embedder
        embedder.load()
        chunk_tokens = min(cfg.chunk_tokens, embedder.max_tokens - 32)  # room for header + prefix
        chunks = chunk_document(
            doc,
            chunk_tokens=chunk_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
            min_chunk_tokens=cfg.min_chunk_tokens,
            count_tokens=embedder.count_tokens,
        )
        embeddings = embedder.encode_passages([c.embed_text for c in chunks])
        empty_pages = [p.page for p in doc.pages if not p.blocks and not p.tables]
        ocr_pages = [p.page for p in doc.pages if p.ocr_applied]
        meta = {
            "num_pages": doc.num_pages,
            "sha256": doc.sha256,
            "source_path": doc.source_path,
            "empty_pages": empty_pages,
            "ocr_pages": ocr_pages,
            "warnings": doc.warnings,
            "config": cfg.ingestion_fingerprint(),
        }
        index = DocumentIndex(doc.doc_id, doc.title, chunks, embeddings, meta)
        if cache is not None:
            try:
                index.save(cache)
            except OSError as exc:
                logger.warning("Could not write index cache: %s", exc)
        self.indexes[index.doc_id] = index
        return IngestReport(
            doc_id=doc.doc_id,
            title=doc.title,
            num_pages=doc.num_pages,
            num_chunks=len(chunks),
            num_table_chunks=sum(c.kind == "table" for c in chunks),
            ocr_pages=ocr_pages,
            empty_pages=empty_pages,
            seconds=time.perf_counter() - t0,
            cached=False,
            warnings=doc.warnings,
        )

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
        return agent.run(question, indexes, history=history, on_step=on_step)


def _rename(index: DocumentIndex, doc_id: str) -> DocumentIndex:
    from dataclasses import replace

    chunks = [
        replace(c, doc_id=doc_id, chunk_id=c.chunk_id.replace(c.doc_id, doc_id, 1))
        for c in index.chunks
    ]
    return DocumentIndex(doc_id, index.title, chunks, index.embeddings, index.meta)
