"""Regression tests for bugs found by the independent code review (2026-10-25)."""

from __future__ import annotations

import pytest

from local_agentic_rag.agent import calculator as C
from local_agentic_rag.agent.prompts import format_history
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.retrieval.hybrid import HybridRetriever
from tests.helpers import TableReranker, scripted
from tests.test_agent import VERIFY_OK, make_agent
from tests.test_retrieval import _cfg, _emb, _stub


def test_empty_candidate_pool_returns_nothing_instead_of_crashing():
    idx = _stub({"q1": [], "q2": []}, {})
    out = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(idx), TableReranker({})).retrieve(
        ["q1", "q2"], [idx]
    )
    assert out == []


def test_each_subquery_keeps_its_own_slot():
    # top_k=3, primary order c0 > c1 > c2 > c3; two sub-queries each want a different passage.
    idx = _stub({"alpha": [0, 1, 2, 3], "beta": [0, 1, 2, 3], "gamma": [0, 1, 2, 3]}, {})
    reranker = TableReranker(
        {
            ("alpha", "doc text 0"): 0.9,
            ("alpha", "doc text 1"): 0.8,
            ("alpha", "doc text 2"): 0.1,
            ("alpha", "doc text 3"): 0.05,
            ("beta", "doc text 2"): 0.9,
            ("gamma", "doc text 3"): 0.9,
        }
    )
    out = HybridRetriever(_cfg(retrieval_mode="dense"), _emb(idx), reranker).retrieve(
        ["alpha", "beta", "gamma"], [idx], top_k=3
    )
    ids = {sc.chunk.chunk_id for sc in out}
    assert {"doc-c2", "doc-c3"} <= ids and len(out) == 3


@pytest.mark.parametrize("expr", ["1e999", "1e300*1e300-1e300*1e300"])
def test_calculator_non_finite_results_are_dropped(expr):
    assert C.run({"needed": True, "expression": expr}) is None


def test_history_turns_zero_sends_no_history():
    assert format_history([("q", "a")], 0) == "(none)"


def test_adaptive_with_no_passages_refuses_without_generating():
    from tests.helpers import FakeEmbedder, make_index

    empty = make_index([], FakeEmbedder(), doc_id="e", title="Empty")
    agent, llm = make_agent(scripted(verify=VERIFY_OK), agent_mode="adaptive")
    ans = agent.run("What is X?", [empty])
    assert ans.refused is True and ans.error is None
    assert llm.calls == []


def test_ingestion_fingerprint_includes_ocr_language():
    assert (
        RAGConfig(ocr_language="eng").ingestion_fingerprint()
        != RAGConfig(ocr_language="deu").ingestion_fingerprint()
    )


def test_answer_ablation_keeps_base_settings(monkeypatch, tmp_path):
    from local_agentic_rag import bench
    from local_agentic_rag.pipeline import RAGPipeline

    seen = {}

    def fake_ingest(pipe, *a, **k):
        seen["top_k"] = pipe.config.top_k
        seen["agentic"] = pipe.config.agentic
        return {}

    monkeypatch.setattr(bench, "ingest_benchmark", fake_ingest)
    monkeypatch.setattr(bench, "load_benchmark", lambda d: ({}, []))
    monkeypatch.setattr(bench, "run_benchmark", lambda *a, **k: [])
    monkeypatch.setattr(bench, "aggregate", lambda r: {})
    base = RAGPipeline(RAGConfig(top_k=9), llm=object(), cache_dir=None)
    bench.answer_ablation(
        object(), "benchmark", configs=["hybrid_rerank"], out_dir=tmp_path, base=base
    )
    assert seen == {"top_k": 9, "agentic": False}
