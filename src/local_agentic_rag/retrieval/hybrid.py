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

import re
from collections import defaultdict

import numpy as np

from local_agentic_rag.config import RAGConfig
from local_agentic_rag.retrieval.embedder import Embedder
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.retrieval.reranker import Reranker
from local_agentic_rag.schema import Chunk, ScoredChunk

_NUM = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+|[०-९]+)"  # "4,512" is one number, "3,4" is two
_ITEM = rf"{_NUM}(?:\s*(?:-|–|to|से|ते)\s*{_NUM})?"  # a page or a range
_PAGE_REF_RE = re.compile(
    r"(?:\bpages?|\bpp?\.|\bpg\.?|पृष्ठ|पेज|पान)\s*(?:no\.?|number|संख्या|क्रमांक)?\s*"
    rf"({_ITEM}(?:\s*(?:,|\band\b|&|और|व)\s*{_ITEM})*)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(rf"({_NUM})\s*(?:-|–|to|से|ते)\s*({_NUM})", re.IGNORECASE)
#: Longest page range, and most pages, a question can pull in directly.
MAX_PAGE_RANGE = 5
MAX_NAMED_PAGES = 8


def _to_int(text: str) -> int:
    return int(text.replace(",", "").translate(str.maketrans("०१२३४५६७८९", "0123456789")))


def page_refs(question: str) -> list[int]:
    """Physical page numbers a question names ("page 12", "pp. 4-6", "pages 3, 4 and 9",
    "पृष्ठ 7").  "and" and commas list pages; only "-", "to" (and Hindi/Marathi से/ते) make
    a range."""
    pages: list[int] = []
    for m in _PAGE_REF_RE.finditer(question or ""):
        items = re.split(r"\s*(?:,(?![0-9]{3}\b)|\band\b|&|और|व)\s*", m.group(1))
        for item in items:
            rng = _RANGE_RE.fullmatch(item.strip())
            if rng:
                first, last = _to_int(rng.group(1)), _to_int(rng.group(2))
            elif re.fullmatch(_NUM, item.strip()):
                first = last = _to_int(item.strip())
            else:
                continue
            if first < 1 or last < first:
                continue
            for page in range(first, min(last, first + MAX_PAGE_RANGE - 1) + 1):
                if page not in pages:
                    pages.append(page)
    return pages[:MAX_NAMED_PAGES]


_SECTION_REF_RE = re.compile(
    r"(?:§+\s*|\b(?:section|sec\.)\s+)(\d+[A-Za-z]?(?:\.\d+)*)", re.IGNORECASE
)


def section_refs(question: str) -> list[str]:
    """Numbered sections a question names: "section 164(b)(6)" → "164", "§ 7203", "Section 2.1".

    Only the number is kept; the subsection ("(b)(6)") is left to the reranker."""
    refs: list[str] = []
    for m in _SECTION_REF_RE.finditer(question or ""):
        ref = m.group(1).lower()
        if ref not in refs:
            refs.append(ref)
    return refs[:4]


class HybridRetriever:
    def __init__(self, config: RAGConfig, embedder: Embedder, reranker: Reranker | None) -> None:
        self.config = config
        self.embedder = embedder
        self.reranker = reranker

    def fuse(self, queries: list[str], indexes: list[DocumentIndex]) -> list[ScoredChunk]:
        """RRF-fused candidates (no reranking), best first."""
        return self._fuse(queries, indexes)[0]

    def _fuse(
        self, queries: list[str], indexes: list[DocumentIndex]
    ) -> tuple[list[ScoredChunk], set[str]]:
        """RRF-fused candidates and the ids of chunks routed from a named page/section."""
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
            elif kind == "bm25":
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

        # "What does page 4,512 say about ...": page numbers are stripped from the text as
        # running footers, so neither retriever can find the page; add its chunks directly
        # and let the reranker judge them.
        routed: set[str] = set()
        for page in page_refs(queries[0]):
            for index in indexes:
                for i in index.page_chunks(page):
                    add(index.chunks[i], 1, "page", queries[0])
                    routed.add(index.chunks[i].chunk_id)
        # "What does section 7203 impose ...": in a 4,000-page code every section looks
        # alike, and the number is in the heading, not the text.  Search inside the named
        # section with the question's vector (bounded even for huge sections).
        refs = section_refs(queries[0])
        if refs and query_vecs is not None:
            for index in indexes:
                for ref in refs:
                    idx = index.section_chunks(ref)
                    if not idx:
                        continue
                    scores = index.embeddings[idx] @ query_vecs[0].astype(np.float32)
                    best = np.argsort(-scores)[: self.SECTION_CANDIDATES]
                    for rank, j in enumerate(best, start=1):
                        chunk = index.chunks[idx[int(j)]]
                        add(chunk, rank, "section", queries[0])
                        routed.add(chunk.chunk_id)

        for chunk_id, score in fused.items():
            by_id[chunk_id].score = score
        return sorted(by_id.values(), key=lambda s: s.score, reverse=True), routed

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
        candidates, routed = self._fuse(queries, indexes)
        # Chunks of a page or section the question names explicitly always reach the
        # reranker, and the best of them are kept (first) whatever their score: the
        # reranker cannot see page or section numbers, so "the table on page 64" would
        # otherwise lose to look-alike tables.
        on_named = [sc for sc in candidates if sc.chunk.chunk_id in routed]
        if self.reranker is None:
            pinned = on_named[: min(self.MAX_PINNED, top_k)]
            return pinned + [sc for sc in candidates if sc not in pinned][: top_k - len(pinned)]

        rest = [sc for sc in candidates if sc.chunk.chunk_id not in routed]
        pool = on_named[: self.config.rerank_candidates] + rest[: self.config.rerank_candidates]
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
        pinned = [sc for sc in ranked if sc.chunk.chunk_id in routed][: min(self.MAX_PINNED, top_k)]
        # Extra queries (decomposed sub-questions, rewrites) only ADD coverage: each one's
        # best passage gets a slot if it scores well, replacing the weakest *primary* hit
        # (never another sub-query's slot).  Ranking by the max over all queries instead was
        # measured to let keyword-ish sub-queries crowd out passages answering the question.
        extras: list[ScoredChunk] = []
        for query in queries[n_primary:]:
            scores = self.reranker.score(query, texts)
            best = int(np.argmax(scores))
            candidate = pool[best]
            if scores[best] >= self.SUBQUERY_MIN_SCORE and candidate not in extras + pinned:
                extras.append(candidate)
        extras = extras[: top_k - len(pinned)]
        taken = pinned + extras
        primaries = [sc for sc in ranked if sc not in taken][: max(top_k - len(taken), 0)]
        chosen = {id(sc) for sc in primaries} | {id(sc) for sc in extras}
        return pinned + [sc for sc in ranked if id(sc) in chosen]

    #: Passages kept from a page or section the question names ("page 64", "section 7203").
    MAX_PINNED = 3
    #: Passages taken from inside a named section (best by dense score) for the reranker.
    SECTION_CANDIDATES = 8

    #: A sub-query's best passage must clear this reranker probability to claim a slot.
    SUBQUERY_MIN_SCORE = 0.3
