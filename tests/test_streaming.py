"""Streaming ingestion: equivalence with in-memory parsing, resume, cancellation, crash isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("reportlab")

from local_agentic_rag.config import RAGConfig  # noqa: E402
from local_agentic_rag.ingest import streaming, workers  # noqa: E402
from local_agentic_rag.ingest.chunker import chunk_document  # noqa: E402
from local_agentic_rag.ingest.pdf import parse_pdf  # noqa: E402
from local_agentic_rag.ingest.streaming import (  # noqa: E402
    IngestCancelled,
    ingest_pdf,
    load_index,
    store_dirs,
)
from local_agentic_rag.retrieval.bm25 import BM25Index  # noqa: E402
from tests.helpers import FakeEmbedder, make_text_pdf  # noqa: E402

N_PAGES = 60
CFG = RAGConfig(ocr="off")


@pytest.fixture(scope="module")
def long_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("streaming") / "long_report.pdf"
    make_text_pdf(path, n_pages=N_PAGES)
    return path


@pytest.fixture(scope="module")
def reference(long_pdf, tmp_path_factory):
    """A clean, uninterrupted streaming ingest (parallel workers)."""
    index, report = ingest_pdf(
        long_pdf, tmp_path_factory.mktemp("ref"), CFG, FakeEmbedder(), workers=2
    )
    return index, report


def _same(a, b) -> None:
    assert [c.to_dict() for c in a.chunks] == [c.to_dict() for c in b.chunks]
    np.testing.assert_array_equal(a.embeddings, b.embeddings)
    assert a.title == b.title


def test_streaming_matches_in_memory_parse(long_pdf, reference):
    index, report = reference
    emb = FakeEmbedder()
    doc = parse_pdf(long_pdf, ocr="off")
    chunks = chunk_document(
        doc,
        chunk_tokens=min(CFG.chunk_tokens, emb.max_tokens - 32),
        overlap_tokens=CFG.chunk_overlap_tokens,
        min_chunk_tokens=CFG.min_chunk_tokens,
        count_tokens=emb.count_tokens,
    )
    assert [c.to_dict() for c in index.chunks] == [c.to_dict() for c in chunks]
    assert report.num_pages == N_PAGES and not report.cached
    # running header/footer removed even though pages were parsed in separate processes
    assert not any("Running Header" in c.text for c in index.chunks)


def test_serial_and_parallel_workers_agree(long_pdf, reference, tmp_path):
    index, _ = ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), workers=1)
    _same(index, reference[0])


def test_second_ingest_is_cached(long_pdf, tmp_path):
    ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder())
    _, report = ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder())
    assert report.cached


class _Crash(Exception):
    pass


@pytest.mark.parametrize("stage", ["reading", "indexing"])
def test_resume_after_crash_gives_identical_index(long_pdf, reference, tmp_path, stage):
    seen = []

    def crash(p):
        seen.append(p)
        if p.stage == stage and p.pages_done >= 16:
            raise _Crash  # simulates the process dying right after a checkpoint

    with pytest.raises(_Crash):
        ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), progress=crash, workers=2)
    index, report = ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), workers=2)
    assert report.resumed_from_page >= 16
    _same(index, reference[0])


def test_torn_writes_are_discarded_on_resume(long_pdf, reference, tmp_path):
    calls = {"n": 0}

    def stop_after_two_batches():
        calls["n"] += 1
        return calls["n"] == 2

    with pytest.raises(IngestCancelled):
        ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), should_stop=stop_after_two_batches)
    parse_dir, _ = store_dirs(tmp_path, reference[0].meta["sha256"], CFG, use_ocr=False)
    with open(parse_dir / "raw.jsonl", "ab") as f:
        f.write(b'{"page": 999, "half-written')  # a batch interrupted mid-write
    index, _ = ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder())
    _same(index, reference[0])


def test_cancelled_ingest_leaves_no_usable_index(long_pdf, tmp_path):
    with pytest.raises(IngestCancelled):
        ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), should_stop=lambda: True)
    _, index_dir = store_dirs(tmp_path, load_sha(long_pdf), CFG, use_ocr=False)
    with pytest.raises(FileNotFoundError):
        load_index(index_dir)


def load_sha(path: Path) -> str:
    from local_agentic_rag.ingest.pdf import sha256_file

    return sha256_file(path)


def test_worker_crash_only_loses_the_crashing_page(long_pdf, reference, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RAG_FAULT_PAGES", "crash:20")
    index, report = ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), workers=2)
    assert any("page 20: parse error" in w for w in report.warnings)
    assert 20 in report.empty_pages
    pages = {c.page for c in index.chunks}
    assert 20 not in pages and {19, 21} <= pages
    ref_other = [c.text for c in reference[0].chunks if c.page != 20]
    assert [c.text for c in index.chunks if c.page != 20] == ref_other


def test_hung_page_times_out(long_pdf, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RAG_FAULT_PAGES", "hang:5")
    monkeypatch.setattr(workers, "PAGE_TIMEOUT_S", 3.0)
    _, report = ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), workers=2)
    assert any("page 5: parse error (timed out" in w for w in report.warnings)
    assert report.num_pages == N_PAGES


def test_variant_reuses_parsed_pages(long_pdf, tmp_path, monkeypatch):
    ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder())
    calls = []
    real = streaming.iter_raw_batches
    monkeypatch.setattr(
        streaming, "iter_raw_batches", lambda *a, **k: calls.append(a) or real(*a, **k)
    )
    index, report = ingest_pdf(
        long_pdf, tmp_path, CFG.with_overrides(chunk_tokens=100), FakeEmbedder()
    )
    assert not calls and not report.cached and len(index) > 0


def test_bm25_roundtrip(tmp_path):
    texts = ["gross domestic product rose 3.8%", "corporate profits fell", "ताजमहल आगरा में है"]
    bm = BM25Index(texts)
    bm.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    for q in ["product 3.8%", "profits", "ताजमहल", "nothing matches"]:
        assert loaded.search(q, 3) == bm.search(q, 3)


def test_stage_timings_stay_positive_after_resume(long_pdf, tmp_path):
    calls = {"n": 0}

    def stop_once():
        calls["n"] += 1
        return calls["n"] == 3

    with pytest.raises(IngestCancelled):
        ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), should_stop=stop_once)
    index, _ = ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder())
    for key in ("parse_seconds", "index_seconds", "embed_seconds", "bm25_seconds"):
        assert index.meta[key] >= 0, key


def test_systematic_parse_failure_is_reported_not_called_scanned(long_pdf, tmp_path, monkeypatch):
    from local_agentic_rag.ingest.pdf import PDFParseError

    monkeypatch.setenv("LOCAL_RAG_FAULT_PAGES", "crash:" + ",".join(str(p) for p in range(1, 41)))
    with pytest.raises(PDFParseError, match="40 of 60 pages .* could not be parsed"):
        ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), workers=2)


def test_onnx_thread_cap_keeps_inference_session_a_class(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "InferenceSession", ort.InferenceSession)  # restored afterwards
    workers._limit_onnx_threads(2)
    # PyMuPDF evaluates ``ort.InferenceSession | None``: a function there broke every page.
    assert isinstance(ort.InferenceSession, type)
    assert ort.InferenceSession | None is not None


def test_checkpoint_without_its_data_file_restarts_cleanly(long_pdf, reference, tmp_path):
    """Power loss can keep state.json but lose a just-created data file: start the pass over."""
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] == 2

    with pytest.raises(IngestCancelled):
        ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), should_stop=stop_after_two)
    parse_dir, _ = store_dirs(tmp_path, reference[0].meta["sha256"], CFG, use_ocr=False)
    (parse_dir / "raw.jsonl").unlink()
    index, _ = ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder())
    _same(index, reference[0])
