"""Unit tests for retrieval: bm25.tokenize/BM25Index, DocumentIndex and HybridRetriever."""

from __future__ import annotations

import numpy as np
import pytest

from local_agentic_rag.config import RAGConfig
from local_agentic_rag.retrieval.bm25 import BM25Index, tokenize
from local_agentic_rag.retrieval.hybrid import HybridRetriever
from local_agentic_rag.retrieval.index import DocumentIndex
from local_agentic_rag.schema import Chunk
from tests.helpers import FakeEmbedder, FakeReranker, TableReranker, make_chunks, make_index

CORPUS = [
    (1, "Inflation rose to 3.5 percent in the second quarter of 2023."),
    (1, "The SP-800 controller supports encrypted firmware updates."),
    (2, "Employee headcount grew across all regional offices."),
    (2, "Interest policies were reviewed by the central bank committee."),
    (3, "Shipping costs fell because freight contracts were renegotiated."),
    (3, "The cafeteria menu now includes vegetarian options daily."),
]


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_tokenize_keeps_decimals_intact():
    toks = tokenize("GDP grew 3.5 percent")
    assert "3.5" in toks
    assert "3" not in toks and "5" not in toks


def test_tokenize_hyphen_code_kept_and_split():
    assert tokenize("SP-800 standard") == ["sp-800", "sp", "800", "standard"]


def test_tokenize_version_like_token_split_into_parts():
    toks = tokenize("model v1.5")
    assert toks[:1] == ["model"]
    assert {"v1.5", "v1", "5"} <= set(toks)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is THE Revenue of the Company") == ["revenue", "company"]
    assert tokenize("the and of") == []


def test_tokenize_plural_stripping():
    assert tokenize("policies") == tokenize("policy") == ["policy"]
    assert tokenize("reports") == tokenize("report") == ["report"]
    assert tokenize("classes") == ["class"]
    assert tokenize("process") == ["process"]  # "ss" is not a plural
    assert tokenize("1990s") == ["1990s"]  # numbers are not stemmed
    assert tokenize("bus") == ["bus"]  # short words untouched


def test_tokenize_plural_of_e_ending_word_matches_singular():
    assert tokenize("rates") == tokenize("rate")
    assert tokenize("tables") == tokenize("table")


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


def test_bm25_search_finds_exact_token():
    bm = BM25Index([t for _, t in CORPUS])
    hits = bm.search("SP-800 firmware", 3)
    assert hits[0][0] == 1
    assert all(score > 0 for _, score in hits)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_bm25_search_decimal_query():
    bm = BM25Index([t for _, t in CORPUS])
    assert bm.search("3.5", 5)[0][0] == 0


def test_bm25_search_plural_query_matches_singular_text():
    bm = BM25Index([t for _, t in CORPUS])
    assert bm.search("policy", 5)[0][0] == 3


def test_bm25_search_limits_and_filters_zero_scores():
    bm = BM25Index([t for _, t in CORPUS])
    assert len(bm.search("inflation freight cafeteria headcount", 2)) == 2
    assert bm.search("zebra", 5) == []
    assert bm.search("the of and", 5) == []


def test_bm25_empty_corpus():
    assert BM25Index([]).search("anything", 5) == []


# ---------------------------------------------------------------------------
# DocumentIndex
# ---------------------------------------------------------------------------


def test_index_rejects_mismatched_embeddings():
    chunks = make_chunks(CORPUS[:2])
    with pytest.raises(ValueError, match="2 chunks but 3 embeddings"):
        DocumentIndex("doc", "", chunks, np.zeros((3, 4), dtype=np.float32))


def test_dense_search_ranks_by_cosine():
    emb = FakeEmbedder()
    index = make_index(CORPUS, emb)
    q = emb.encode_queries(["freight shipping costs renegotiated"])[0]
    hits = index.dense_search(q, 3)
    assert hits[0][0] == 4
    assert len(hits) == 3
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert len(index.dense_search(q, 100)) == len(CORPUS)


def test_dense_search_empty_index():
    index = DocumentIndex("doc", "", [], np.zeros((0, 8), dtype=np.float32))
    assert len(index) == 0
    assert index.dense_search(np.ones(8, dtype=np.float32), 5) == []
    assert index.bm25_search("x", 5) == []


def test_index_save_load_roundtrip(tmp_path):
    emb = FakeEmbedder()
    chunks = make_chunks(CORPUS, doc_id="rep", title="Annual Report")
    chunks[0].kind = "table"
    chunks[0].section = "Prices"
    embeddings = emb.encode_passages([c.embed_text for c in chunks])
    index = DocumentIndex(
        "rep", "Annual Report", chunks, embeddings, {"num_pages": 3, "warnings": ["w"]}
    )
    target = tmp_path / "cache" / "rep"
    assert not DocumentIndex.exists(target)
    index.save(target)
    assert DocumentIndex.exists(target)
    assert not (target / "index.json.tmp").exists()

    loaded = DocumentIndex.load(target)
    assert loaded.doc_id == "rep"
    assert loaded.title == "Annual Report"
    assert loaded.meta == {"num_pages": 3, "warnings": ["w"]}
    assert [c.to_dict() for c in loaded.chunks] == [c.to_dict() for c in index.chunks]
    assert loaded.embeddings.dtype == np.float32
    np.testing.assert_array_equal(loaded.embeddings, index.embeddings)
    assert loaded.bm25_search("SP-800", 1) == index.bm25_search("SP-800", 1)


def test_exists_requires_both_files(tmp_path):
    (tmp_path / "index.json").write_text("{}")
    assert not DocumentIndex.exists(tmp_path)


# ---------------------------------------------------------------------------
# HybridRetriever — RRF with controlled rankings
# ---------------------------------------------------------------------------


class StubIndex:
    """Index whose dense/BM25 rankings are fixed per query.

    HybridRetriever hands dense_search a query *vector*; IndexEmbedder encodes the
    i-th query as [i], so the stub recovers which query it is answering.
    """

    def __init__(
        self, doc_id: str, dense: dict[str, list[int]], bm25: dict[str, list[int]], n: int = 6
    ):
        self.doc_id = doc_id
        self.title = doc_id
        self.chunks = [
            Chunk(
                chunk_id=f"{doc_id}-c{i}",
                doc_id=doc_id,
                page=i + 1,
                section="",
                text=f"{doc_id} text {i}",
            )
            for i in range(n)
        ]
        self.dense, self.bm25 = dense, bm25
        self.queries: list[str] = []

    def dense_search(self, query_vec, n):
        query = self.queries[int(query_vec[0])]
        return [(i, 1.0) for i in self.dense.get(query, [])][:n]

    def bm25_search(self, query, n):
        return [(i, 1.0) for i in self.bm25.get(query, [])][:n]


class IndexEmbedder(FakeEmbedder):
    """Encodes query i as the 1-d vector [i]."""

    def __init__(self, stubs: list[StubIndex] | None = None):
        super().__init__()
        self.stubs = stubs or []

    def encode_queries(self, queries):
        self.query_calls += 1
        for s in self.stubs:
            s.queries = list(queries)
        return np.arange(len(queries), dtype=np.float32).reshape(-1, 1)


def _stub(dense, bm25, doc_id="doc", n=6):
    return StubIndex(doc_id, dense, bm25, n)


def _emb(*stubs: StubIndex) -> IndexEmbedder:
    return IndexEmbedder(list(stubs))


def _cfg(**kw) -> RAGConfig:
    return RAGConfig(reranker_model=None, rrf_k=60, **kw)


def test_rrf_fusion_scores_and_order():
    idx = _stub({"q": [0, 1, 2]}, {"q": [2, 0, 3]})
    retriever = HybridRetriever(_cfg(), _emb(idx), None)
    fused = retriever.fuse(["q"], [idx])
    ids = [sc.chunk.chunk_id for sc in fused]
    assert ids == ["doc-c0", "doc-c2", "doc-c1", "doc-c3"]
    by_id = {sc.chunk.chunk_id: sc for sc in fused}
    assert by_id["doc-c0"].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["doc-c2"].score == pytest.approx(1 / 63 + 1 / 61)
    assert by_id["doc-c1"].score == pytest.approx(1 / 62)
    assert by_id["doc-c3"].score == pytest.approx(1 / 63)
    assert (by_id["doc-c0"].dense_rank, by_id["doc-c0"].bm25_rank) == (1, 2)
    assert (by_id["doc-c1"].dense_rank, by_id["doc-c1"].bm25_rank) == (2, None)
    assert (by_id["doc-c3"].dense_rank, by_id["doc-c3"].bm25_rank) == (None, 3)
    assert by_id["doc-c0"].matched_queries == ["q"]


def test_rrf_k_changes_scores():
    idx = _stub({"q": [0]}, {"q": []})
    fused = HybridRetriever(RAGConfig(reranker_model=None, rrf_k=10), _emb(idx), None).fuse(
        ["q"], [idx]
    )
    assert fused[0].score == pytest.approx(1 / 11)


def test_dense_mode_ignores_bm25():
    idx = _stub({"q": [0, 1]}, {"q": [5, 4]})
    emb = _emb(idx)
    out = HybridRetriever(_cfg(retrieval_mode="dense"), emb, None).retrieve(["q"], [idx])
    assert [sc.chunk.chunk_id for sc in out] == ["doc-c0", "doc-c1"]
    assert all(sc.bm25_rank is None for sc in out)
    assert emb.query_calls == 1


def test_bm25_mode_does_not_embed_queries():
    idx = _stub({"q": [0, 1]}, {"q": [5, 4]})
    emb = _emb(idx)
    out = HybridRetriever(_cfg(retrieval_mode="bm25"), emb, None).retrieve(["q"], [idx])
    assert [sc.chunk.chunk_id for sc in out] == ["doc-c5", "doc-c4"]
    assert all(sc.dense_rank is None for sc in out)
    assert emb.query_calls == 0


def test_hybrid_mode_uses_both():
    idx = _stub({"q": [0, 1]}, {"q": [5, 4]})
    out = HybridRetriever(_cfg(retrieval_mode="hybrid"), _emb(idx), None).retrieve(["q"], [idx])
    assert {sc.chunk.chunk_id for sc in out} == {"doc-c0", "doc-c1", "doc-c4", "doc-c5"}


def test_multi_query_rankings_accumulate():
    idx = _stub({"q1": [0, 1], "q2": [1, 2]}, {})
    fused = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(idx), None).fuse(["q1", "q2"], [idx])
    by_id = {sc.chunk.chunk_id: sc for sc in fused}
    assert fused[0].chunk.chunk_id == "doc-c1"
    assert by_id["doc-c1"].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["doc-c1"].dense_rank == 1  # best rank over queries
    assert by_id["doc-c1"].matched_queries == ["q1", "q2"]
    assert by_id["doc-c0"].matched_queries == ["q1"]


def test_candidates_per_retriever_limits_each_list():
    idx = _stub({"q": [0, 1, 2, 3]}, {"q": [3, 2, 1, 0]})
    fused = HybridRetriever(_cfg(candidates_per_retriever=1), _emb(idx), None).fuse(["q"], [idx])
    assert {sc.chunk.chunk_id for sc in fused} == {"doc-c0", "doc-c3"}


def test_retrieve_top_k_without_reranker():
    idx = _stub({"q": [0, 1, 2, 3, 4, 5]}, {})
    retriever = HybridRetriever(_cfg(retrieval_mode="dense", top_k=3), _emb(idx), None)
    assert len(retriever.retrieve(["q"], [idx])) == 3
    assert len(retriever.retrieve(["q"], [idx], top_k=5)) == 5
    assert all(sc.rerank_score is None for sc in retriever.retrieve(["q"], [idx]))


def test_retrieve_dedupes_queries_and_drops_blank():
    idx = _stub({"q": [0, 1]}, {"q": [0, 1]})
    reranker = FakeReranker()
    retriever = HybridRetriever(_cfg(), _emb(idx), reranker)
    out = retriever.retrieve(["q", " q ", "", "   ", "q"], [idx])
    assert [q for q, _ in reranker.calls] == ["q"]
    ids = [sc.chunk.chunk_id for sc in out]
    assert len(ids) == len(set(ids)) == 2


def test_retrieve_empty_inputs():
    idx = _stub({"q": [0]}, {})
    retriever = HybridRetriever(_cfg(), _emb(idx), None)
    assert retriever.retrieve([], [idx]) == []
    assert retriever.retrieve(["", "  "], [idx]) == []
    assert retriever.retrieve(["q"], []) == []


def test_multiple_indexes_are_searched():
    a = _stub({"q": [0]}, {}, doc_id="a")
    b = _stub({"q": [1]}, {}, doc_id="b")
    out = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(a, b), None).retrieve(["q"], [a, b])
    assert {sc.chunk.doc_id for sc in out} == {"a", "b"}


# ---------------------------------------------------------------------------
# HybridRetriever — reranking
# ---------------------------------------------------------------------------


def test_reranker_reorders_and_sets_scores():
    idx = _stub({"q": [0, 1, 2]}, {})
    reranker = TableReranker(
        {("q", "doc text 2"): 0.9, ("q", "doc text 0"): 0.2, ("q", "doc text 1"): 0.5}
    )
    out = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(idx), reranker).retrieve(["q"], [idx])
    assert [sc.chunk.chunk_id for sc in out] == ["doc-c2", "doc-c1", "doc-c0"]
    assert [sc.rerank_score for sc in out] == pytest.approx([0.9, 0.5, 0.2])
    assert all(sc.score == sc.rerank_score for sc in out)
    assert out[0].dense_rank == 3  # fusion ranks are kept for diagnostics


def test_multi_query_ranks_by_primary_question():
    # Order follows the user's question (q1); the sub-query cannot reorder it.
    idx = _stub({"q1": [1, 0], "q2": [1, 0]}, {})
    reranker = TableReranker(
        {
            ("q1", "doc text 0"): 0.05,
            ("q2", "doc text 0"): 0.95,
            ("q1", "doc text 1"): 0.6,
            ("q2", "doc text 1"): 0.4,
        }
    )
    out = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(idx), reranker).retrieve(
        ["q1", "q2"], [idx]
    )
    assert [sc.chunk.chunk_id for sc in out] == ["doc-c1", "doc-c0"]
    assert [sc.rerank_score for sc in out] == pytest.approx([0.6, 0.05])


def test_subquery_best_passage_claims_a_slot():
    # top_k=1: the primary winner (c1) would fill it, but q2's strong best passage (c0)
    # replaces the weakest primary hit so both halves of a comparison are present.
    idx = _stub({"q1": [1, 0], "q2": [0, 1]}, {})
    reranker = TableReranker({("q1", "doc text 1"): 0.6, ("q2", "doc text 0"): 0.9})
    out = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(idx), reranker).retrieve(
        ["q1", "q2"], [idx], top_k=1
    )
    assert [sc.chunk.chunk_id for sc in out] == ["doc-c0"]


def test_weak_subquery_does_not_displace_primary_hits():
    idx = _stub({"q1": [1, 0], "q2": [0, 1]}, {})
    reranker = TableReranker({("q1", "doc text 1"): 0.6, ("q2", "doc text 0"): 0.1})
    out = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(idx), reranker).retrieve(
        ["q1", "q2"], [idx], top_k=1
    )
    assert [sc.chunk.chunk_id for sc in out] == ["doc-c1"]


def test_rerank_pool_limited_to_rerank_candidates():
    idx = _stub({"q": [0, 1, 2, 3, 4, 5]}, {})
    reranker = TableReranker({("q", "doc text 5"): 1.0})
    cfg = _cfg(retrieval_mode="dense", rerank_candidates=3, top_k=6)
    out = HybridRetriever(cfg, _emb(idx), reranker).retrieve(["q"], [idx])
    # c5 is fused rank 6, outside the pool of 3, so it can never be promoted.
    assert [sc.chunk.chunk_id for sc in out] == ["doc-c0", "doc-c1", "doc-c2"]


def test_reranker_scores_embed_text():
    idx = _stub({"q": [0]}, {})
    idx.chunks[0].doc_title = "Report"
    idx.chunks[0].section = "Prices"
    reranker = TableReranker({("q", "Report > Prices\ndoc text 0"): 0.7})
    out = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(idx), reranker).retrieve(["q"], [idx])
    assert out[0].rerank_score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# HybridRetriever over a real DocumentIndex with fake models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_end_to_end_modes_find_relevant_chunk(mode):
    emb = FakeEmbedder()
    index = make_index(CORPUS, emb)
    cfg = RAGConfig(retrieval_mode=mode, top_k=2)
    out = HybridRetriever(cfg, emb, FakeReranker()).retrieve(
        ["SP-800 controller firmware"], [index]
    )
    assert out[0].chunk.chunk_id == "doc-p1-c1"
    assert out[0].rerank_score == pytest.approx(1.0)
    assert len(out) <= 2


# ---------------------------------------------------------------------------
# Explicit page references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "pages"),
    [
        ("What does page 4,512 say about rail?", [4512]),
        ("Summarise pp. 10-12", [10, 11, 12]),
        ("पृष्ठ 7 पर क्या लिखा है", [7]),
        ("In 2024, pages 3 and 4", [3, 4]),
        ("page 5000 to 5100", [5000, 5001, 5002, 5003, 5004]),
        ("What happened in 2019?", []),
        ("the 3-page summary", []),
        ("homepage 12", []),
        ("Compare page 12 and 40", [12, 40]),
        ("pages 3,4 and 5", [3, 4, 5]),
        ("page 12 and the summary", [12]),
        ("pages 1,200-1,202", [1200, 1201, 1202]),
        ("page-level citations", []),
        ("Table 7.3 on p. 88", [88]),
    ],
)
def test_page_refs(question, pages):
    from local_agentic_rag.retrieval.hybrid import page_refs

    assert page_refs(question) == pages


def test_named_page_joins_the_candidate_pool():
    emb = FakeEmbedder()
    items = [(p, f"Filler paragraph number {p} about routine matters.") for p in range(1, 40)]
    index = make_index(items, emb)
    cfg = RAGConfig(retrieval_mode="hybrid", candidates_per_retriever=3, reranker_model=None)
    fused = HybridRetriever(cfg, emb, None).fuse(["What does page 27 say?"], [index])
    assert 27 in [sc.chunk.page for sc in fused[:5]]
    for page in (13, 38):
        fused = HybridRetriever(cfg, emb, None).fuse([f"Summarise page {page}"], [index])
        assert page in [sc.chunk.page for sc in fused[:5]]


# ---------------------------------------------------------------------------
# Named sections ("section 7203")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "refs"),
    [
        ("What does section 164(b)(6) limit?", ["164"]),
        ("Under § 7203, what penalties apply?", ["7203"]),
        ("see Section 2.1 of the profile", ["2.1"]),
        ("What does sec. 6501(a) say?", ["6501"]),
        ("the Population section of the abstract", []),
        ("sections of the report", []),
    ],
)
def test_section_refs(question, refs):
    from local_agentic_rag.retrieval.hybrid import section_refs

    assert section_refs(question) == refs


def _code_index(emb):
    """A tax-code-like document: many look-alike penalty sections, numbers only in headings."""
    from dataclasses import replace

    items, sections = [], []
    for n, sec in enumerate(
        [
            "§7201. Attempt to evade",
            "§7202. Willful failure to collect",
            "§7203. Willful failure to file",
            "§7204. Fraudulent statements",
        ]
    ):
        for k in range(4):
            items.append(
                (
                    n * 4 + k + 1,
                    f"Any person who willfully fails shall be fined not more than ${(n + 1) * 25},000 or imprisoned, clause {k}.",
                )
            )
            sections.append(sec if k == 0 else "(b) Penalties")
    chunks = [replace(c, section=s) for c, s in zip(make_chunks(items), sections, strict=True)]
    return DocumentIndex("doc", "", chunks, emb.encode_passages([c.embed_text for c in chunks]))


def test_section_chunks_follow_the_last_numbered_heading():
    index = _code_index(FakeEmbedder())
    assert [index.chunks[i].page for i in index.section_chunks("7203")] == [9, 10, 11, 12]
    assert index.section_chunks("720") == [] and index.section_chunks("9999") == []


def test_named_section_passages_are_pinned_first():
    emb = FakeEmbedder()
    index = _code_index(emb)
    cfg = RAGConfig(retrieval_mode="hybrid", candidates_per_retriever=3, top_k=4)
    out = HybridRetriever(cfg, emb, FakeReranker()).retrieve(
        ["What criminal penalties does section 7203 impose for willful failure?"], [index]
    )
    assert {sc.chunk.page for sc in out[:3]} <= {9, 10, 11, 12}


def test_inline_section_starts_and_year_headings():
    """US Code PDFs merge "§ 164. Taxes" into the paragraph; "2010 (I)" is a year, not §2010."""
    from dataclasses import replace

    items = [
        (1, "Earlier text. § 164. Taxes (a) General rule Except as otherwise provided"),
        (1, "(b)(6) Limitation for taxable years 2018 through 2025: $10,000"),
        (2, "see section 164 and § 221(a)(2)(A), Dec. 19, 2014, 128 Stat. 4037]"),
        (3, "Interest on bonds"),
    ]
    sections = ["§ 163. Interest", "§ 163. Interest", "Editorial Notes", "2010 (I) In general"]
    chunks = [replace(c, section=s) for c, s in zip(make_chunks(items), sections, strict=True)]
    emb = FakeEmbedder()
    index = DocumentIndex("doc", "", chunks, emb.encode_passages([c.embed_text for c in chunks]))
    assert index.section_chunks("164") == [0, 1, 2, 3]
    assert index.section_chunks("163") == [0]
    assert index.section_chunks("221") == [] and index.section_chunks("2010") == []


def test_second_document_gets_a_slot_when_clearly_relevant():
    # Book A fills the top 3 by score; book B's best passage (0.6) still gets a slot.
    a = _stub({"q": [0, 1, 2, 3]}, {}, doc_id="a")
    b = _stub({"q": [0, 1]}, {}, doc_id="b")
    reranker = TableReranker(
        {
            ("q", "a text 0"): 0.95,
            ("q", "a text 1"): 0.9,
            ("q", "a text 2"): 0.85,
            ("q", "a text 3"): 0.8,
            ("q", "b text 0"): 0.6,
            ("q", "b text 1"): 0.1,
        }
    )
    out = HybridRetriever(_cfg(retrieval_mode="dense", top_k=3), _emb(a, b), reranker).retrieve(
        ["q"], [a, b]
    )
    assert [sc.chunk.chunk_id for sc in out] == ["a-c0", "a-c1", "b-c0"]


def test_weak_second_document_does_not_displace_passages():
    a = _stub({"q": [0, 1, 2]}, {}, doc_id="a")
    b = _stub({"q": [0]}, {}, doc_id="b")
    reranker = TableReranker(
        {
            ("q", "a text 0"): 0.9,
            ("q", "a text 1"): 0.8,
            ("q", "a text 2"): 0.7,
            ("q", "b text 0"): 0.2,
        }
    )
    out = HybridRetriever(_cfg(retrieval_mode="dense", top_k=3), _emb(a, b), reranker).retrieve(
        ["q"], [a, b]
    )
    assert [sc.chunk.chunk_id for sc in out] == ["a-c0", "a-c1", "a-c2"]
