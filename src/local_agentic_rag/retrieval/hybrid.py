"""
Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion, then reranked.

Why RRF instead of the reference notebook's weighted min-max fusion:
min-max rescales each query's scores to [0, 1] using that query's own best
and worst hit, so a query with *no* good match still produces a top result
with normalised score 1.0, and the 0.7/0.3 weights mix incomparable scales.
RRF only uses ranks — ``score = Σ 1 / (k + rank)`` — so it needs no tuning
and is robust to score-scale differences.

Multi-query: the agent may issue several search queries (decomposed
sub-questions, rewrites).  Each (query, retriever) ranking contributes to the
same RRF candidate pool.  The cross-encoder ranks the pool against the user's
question; each extra query can only add its single best passage (so both
halves of a comparison are present) without reordering the rest.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from local_agentic_rag.config import RAGConfig
from local_agentic_rag.retrieval.embedder import Embedder
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.retrieval.reranker import Reranker
from local_agentic_rag.schema import Chunk, ScoredChunk


class HybridRetriever:
    def __init__(self, config: RAGConfig, embedder: Embedder, reranker: Reranker | None) -> None:
        self.config = config
        self.embedder = embedder
        self.reranker = reranker

    def fuse(self, queries: list[str], indexes: list[DocumentIndex]) -> list[ScoredChunk]:
        """RRF-fused candidates (no reranking), best first."""
        cfg = self.config
        n = cfg.candidates_per_retriever
        fused: dict[str, float] = defaultdict(float)
        by_id: dict[str, ScoredChunk] = {}
        query_vecs = self.embedder.encode_queries(queries) if cfg.retrieval_mode != "bm25" else None

        def add(chunk: Chunk, rank: int, kind: str, query: str) -> None:
            fused[chunk.chunk_id] += 1.0 / (cfg.rrf_k + rank)
            sc = by_id.get(chunk.chunk_id)
            if sc is None:
                sc = by_id[chunk.chunk_id] = ScoredChunk(chunk=chunk, score=0.0)
            if kind == "dense":
                sc.dense_rank = rank if sc.dense_rank is None else min(sc.dense_rank, rank)
            else:
                sc.bm25_rank = rank if sc.bm25_rank is None else min(sc.bm25_rank, rank)
            if query not in sc.matched_queries:
                sc.matched_queries.append(query)

        for qi, query in enumerate(queries):
            for index in indexes:
                if query_vecs is not None:
                    for rank, (i, _) in enumerate(index.dense_search(query_vecs[qi], n), start=1):
                        add(index.chunks[i], rank, "dense", query)
                if cfg.retrieval_mode != "dense":
                    for rank, (i, _) in enumerate(index.bm25_search(query, n), start=1):
                        add(index.chunks[i], rank, "bm25", query)

        for chunk_id, score in fused.items():
            by_id[chunk_id].score = score
        return sorted(by_id.values(), key=lambda s: s.score, reverse=True)

    def retrieve(
        self,
        queries: list[str],
        indexes: list[DocumentIndex],
        top_k: int | None = None,
        n_primary: int = 1,
    ) -> list[ScoredChunk]:
        """Fuse → rerank (if configured) → top_k.  ``queries[0]`` should be the user question."""
        top_k = top_k or self.config.top_k
        queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        if not queries or not indexes:
            return []
        candidates = self.fuse(queries, indexes)
        if self.reranker is None:
            return candidates[:top_k]

        pool = candidates[: self.config.rerank_candidates]
        if not pool:
            return []
        texts = [sc.chunk.embed_text for sc in pool]
        # The first ``n_primary`` queries are the user's question and its translation into
        # the document's language (cross-lingual questions); a passage's score is its best match.
        n_primary = max(1, min(n_primary, len(queries)))
        primary = np.max([self.reranker.score(q, texts) for q in queries[:n_primary]], axis=0)
        for sc, s in zip(pool, primary, strict=True):
            sc.rerank_score = float(s)
            sc.score = float(s)
        ranked = sorted(pool, key=lambda s: s.score, reverse=True)
        # Extra queries (decomposed sub-questions, rewrites) only ADD coverage: each one's
        # best passage gets a slot if it scores well, replacing the weakest *primary* hit
        # (never another sub-query's slot).  Ranking by the max over all queries instead was
        # measured to let keyword-ish sub-queries crowd out passages answering the question.
        extras: list[ScoredChunk] = []
        for query in queries[n_primary:]:
            scores = self.reranker.score(query, texts)
            best = int(np.argmax(scores))
            candidate = pool[best]
            if scores[best] >= self.SUBQUERY_MIN_SCORE and candidate not in extras:
                extras.append(candidate)
        extras = extras[:top_k]
        primaries = [sc for sc in ranked if sc not in extras][: max(top_k - len(extras), 0)]
        chosen = {id(sc) for sc in primaries} | {id(sc) for sc in extras}
        return [sc for sc in ranked if id(sc) in chosen]

    #: A sub-query's best passage must clear this reranker probability to claim a slot.
    SUBQUERY_MIN_SCORE = 0.3
