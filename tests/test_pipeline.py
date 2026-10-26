"""Unit tests for local_agentic_rag.pipeline.RAGPipeline with fake embedder/reranker/LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("reportlab")

from local_agentic_rag.config import RAGConfig  # noqa: E402
from local_agentic_rag.ingest.pdf import PDFParseError  # noqa: E402
from local_agentic_rag.llm.fake import ScriptedLLM  # noqa: E402
from local_agentic_rag.pipeline import MAX_QUESTION_CHARS, RAGPipeline  # noqa: E402
from tests.helpers import (  # noqa: E402
    FakeEmbedder,
    FakeReranker,
    make_blank_pdf,
    make_table_pdf,
    make_text_pdf,
    passage_number,
    role_of,
    scripted,
)


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("pipeline_pdfs")
    text, table = d / "annual_report.pdf", d / "sales.pdf"
    make_text_pdf(text, n_pages=4)
    make_table_pdf(table)
    return {"text": text, "table": table}


def default_responder():
    return scripted(
        plan=lambda m: {
            "standalone_question": "Which page mentions uniqueterm3?",
            "search_queries": ["uniqueterm3"],
        },
        grade=lambda m: {
            "relevant": [passage_number(m, "uniqueterm3")],
            "sufficient": True,
            "missing_query": "",
        },
        generate=lambda m: f"It is mentioned in paragraph 0 [{passage_number(m, 'uniqueterm3')}].",
        verify={"supported": True, "unsupported_claims": []},
    )


def make_pipeline(
    tmp_path, responder=None, embedder=None, **cfg
) -> tuple[RAGPipeline, FakeEmbedder]:
    config = RAGConfig(**cfg)
    pipe = RAGPipeline(
        config, llm=ScriptedLLM(responder or default_responder()), cache_dir=tmp_path / "cache"
    )
    embedder = embedder or FakeEmbedder()
    pipe._shared["embedders"][config.embedder_model] = embedder
    if config.reranker_model:
        pipe._shared["rerankers"][config.reranker_model] = FakeReranker()
    return pipe, embedder


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_ingest_report(tmp_path, pdfs):
    pipe, emb = make_pipeline(tmp_path)
    report = pipe.ingest(pdfs["text"])
    assert report.cached is False
    assert report.doc_id == "annual-report"
    assert report.num_pages == 4
    assert report.num_chunks >= 4
    assert report.num_table_chunks == 0
    assert report.empty_pages == []
    assert report.ocr_pages == []
    assert report.warnings == []
    assert emb.loaded is True
    assert emb.passage_calls == 1
    assert "annual-report" in pipe.indexes
    index = pipe.indexes["annual-report"]
    assert len(index) == report.num_chunks
    assert {c.page for c in index.chunks} == {1, 2, 3, 4}
    assert index.meta["num_pages"] == 4


def test_ingest_table_pdf_counts_table_chunks(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    report = pipe.ingest(pdfs["table"])
    assert report.num_table_chunks == 1
    table = next(c for c in pipe.indexes[report.doc_id].chunks if c.kind == "table")
    assert "Northwind" in table.text


def test_second_ingest_is_cache_hit(tmp_path, pdfs):
    pipe, emb = make_pipeline(tmp_path)
    first = pipe.ingest(pdfs["text"])
    second = pipe.ingest(pdfs["text"])
    assert first.cached is False
    assert second.cached is True
    assert emb.passage_calls == 1  # no re-embedding
    assert second.num_chunks == first.num_chunks
    assert second.num_pages == first.num_pages
    assert second.doc_id == first.doc_id


def test_cache_survives_new_pipeline(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    fresh, emb = make_pipeline(tmp_path)
    report = fresh.ingest(pdfs["text"])
    assert report.cached is True
    assert emb.passage_calls == 0
    assert list(fresh.indexes) == ["annual-report"]


def test_cache_hit_with_new_doc_id_renames_chunks(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    report = pipe.ingest(pdfs["text"], doc_id="renamed")
    assert report.cached is True
    assert report.doc_id == "renamed"
    index = pipe.indexes["renamed"]
    assert all(c.doc_id == "renamed" for c in index.chunks)
    assert all(c.chunk_id.startswith("renamed-p") for c in index.chunks)


def test_use_cache_false_reingests(tmp_path, pdfs):
    pipe, emb = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    report = pipe.ingest(pdfs["text"], use_cache=False)
    assert report.cached is False
    assert emb.passage_calls == 2


def test_ingest_config_change_misses_cache(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    other = pipe.variant(chunk_tokens=100)
    assert other.ingest(pdfs["text"]).cached is False


def test_no_cache_dir_writes_nothing(tmp_path, pdfs):
    config = RAGConfig()
    pipe = RAGPipeline(config, llm=None, cache_dir=None)
    pipe._shared["embedders"][config.embedder_model] = FakeEmbedder()
    assert pipe.ingest(pdfs["text"]).cached is False
    assert pipe.ingest(pdfs["text"]).cached is False
    assert not (tmp_path / "cache").exists()


def test_chunk_size_clamped_to_embedder_window(tmp_path, pdfs):
    pipe, emb = make_pipeline(
        tmp_path, embedder=FakeEmbedder(max_tokens=72), chunk_overlap_tokens=0
    )
    report = pipe.ingest(pdfs["text"])
    chunks = pipe.indexes[report.doc_id].chunks
    assert all(c.n_tokens <= 72 - 32 for c in chunks)
    assert all(c.n_tokens == emb.count_tokens(c.text) for c in chunks)


def test_ingest_missing_file(tmp_path):
    pipe, _ = make_pipeline(tmp_path)
    with pytest.raises(FileNotFoundError):
        pipe.ingest(tmp_path / "nope.pdf")


def test_ingest_blank_pdf_raises(tmp_path):
    path = tmp_path / "blank.pdf"
    make_blank_pdf(path)
    pipe, _ = make_pipeline(tmp_path)
    with pytest.raises(PDFParseError):
        pipe.ingest(path)
    assert pipe.indexes == {}


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def test_ask_end_to_end(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    ans = pipe.ask("Which page mentions uniqueterm3?")
    assert ans.error is None
    assert ans.refused is False
    assert ans.cited_pages == [3]
    assert ans.grounded is True
    # Default adaptive mode: a verified first answer needs no escalation.
    assert ans.llm_calls == 2
    assert [t.step for t in ans.trace] == ["retrieve", "generate", "verify"]


def test_ask_non_agentic_variant(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    single = pipe.variant(agentic=False)
    ans = single.ask("Which page mentions uniqueterm3?")
    assert ans.llm_calls == 1
    assert ans.cited_pages == [3]
    assert [role_of(c["messages"], c["json_schema"]) for c in pipe.llm.calls] == ["generate"]


def test_variant_shares_models_and_indexes(tmp_path, pdfs):
    pipe, emb = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    v = pipe.variant(top_k=2)
    assert v.config.top_k == 2
    assert v.embedder is emb
    assert v.indexes is pipe.indexes
    assert v.llm is pipe.llm
    with pytest.raises(ValueError, match="Unknown config keys"):
        pipe.variant(not_a_key=1)


def test_retrieve_returns_ranked_chunks(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    hits = pipe.retrieve("uniqueterm2", top_k=3)
    assert 1 <= len(hits) <= 3
    assert hits[0].chunk.page == 2
    assert hits[0].rerank_score is not None


def test_ask_empty_question(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    for q in ("", "   ", None):
        ans = pipe.ask(q)
        assert ans.refused is True
        assert ans.error == "empty question"
        assert ans.answer == "Please enter a question."
    assert pipe.llm.calls == []


def test_ask_long_question_truncated(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    ans = pipe.ask("uniqueterm3 " + "x" * 5000)
    assert len(ans.question) == MAX_QUESTION_CHARS


def test_ask_unknown_doc_id_raises_key_error(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    with pytest.raises(KeyError, match="nope"):
        pipe.ask("Which page mentions uniqueterm3?", doc_id="nope")
    with pytest.raises(KeyError):
        pipe.ask("Which page mentions uniqueterm3?", doc_id=["annual-report", "nope"])


def test_ask_specific_doc_ids(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    pipe.ingest(pdfs["table"])
    ans = pipe.ask("Which page mentions uniqueterm3?", doc_id="annual-report")
    assert ans.error is None
    assert all(sc.chunk.doc_id == "annual-report" for sc in ans.retrieved)
    ans = pipe.ask("Which page mentions uniqueterm3?", doc_id=["annual-report", "sales"])
    assert ans.error is None


def test_ask_before_ingest_raises_runtime_error(tmp_path):
    pipe, _ = make_pipeline(tmp_path)
    with pytest.raises(RuntimeError, match="No document ingested"):
        pipe.ask("anything?")


def test_ask_without_llm_raises_runtime_error(tmp_path, pdfs):
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"])
    pipe.llm = None
    with pytest.raises(RuntimeError, match="No LLM configured"):
        pipe.ask("anything?")


def test_reranker_none_when_disabled(tmp_path):
    pipe, _ = make_pipeline(tmp_path, reranker_model=None)
    assert pipe.reranker is None


def test_variants_share_one_parse_per_pdf(tmp_path, pdfs, monkeypatch):
    from local_agentic_rag.ingest import streaming

    calls = []
    real_parse = streaming.iter_raw_batches

    def counting_parse(*args, **kwargs):
        calls.append(args[0])
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(streaming, "iter_raw_batches", counting_parse)
    pipe, _ = make_pipeline(tmp_path)
    pipe.ingest(pdfs["text"], doc_id="a")
    variant = pipe.variant(chunk_tokens=128)  # different chunking/index, same parse
    variant._shared["embedders"][variant.config.embedder_model] = pipe.embedder
    variant.ingest(pdfs["text"], doc_id="b")
    assert len(calls) == 1
    assert variant.indexes["b"].doc_id == "b"
