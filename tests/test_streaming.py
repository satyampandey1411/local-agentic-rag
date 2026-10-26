"""Streaming ingestion: equivalence with in-memory parsing, resume, cancellation, crash isolation."""

from __future__ import annotations

import json
import threading
import time
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
    load_partial_index,
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


CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"


def test_provider_selection_uses_gpu_only_when_asked_and_available():
    pick = workers.select_onnx_providers
    assert pick([CPU], True, [CUDA, CPU]) == [CUDA, CPU]
    assert pick(None, True, [CUDA, CPU]) == [CUDA, CPU]
    assert pick([CPU], False, [CUDA, CPU]) is None  # flag off: unchanged
    assert pick([CPU], True, [CPU]) is None  # no CUDA build: unchanged
    assert pick(["TensorrtExecutionProvider"], True, [CUDA, CPU]) is None  # caller's choice wins


def test_gpu_flag_parsing():
    from local_agentic_rag.config import onnx_gpu_requested

    assert not onnx_gpu_requested({})
    assert not onnx_gpu_requested({"LOCAL_RAG_ONNX_GPU": "0"})
    assert onnx_gpu_requested({"LOCAL_RAG_ONNX_GPU": "1"})
    assert onnx_gpu_requested({"LOCAL_RAG_ONNX_GPU": " True "})


def test_gpu_session_failure_falls_back_to_cpu_silently(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    seen: list[list[str] | None] = []

    class FakeSession:
        def __init__(self, path_or_bytes, sess_options=None, providers=None, **kw):
            seen.append(providers)
            if providers and CUDA in providers:
                raise RuntimeError("CUDA driver not found")

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CUDA, CPU])
    workers._limit_onnx_threads(2, use_gpu=True)
    ort.InferenceSession("model.onnx", providers=[CPU])
    assert seen == [[CUDA, CPU], [CPU]]  # tried CUDA, then the caller's CPU request
    seen.clear()
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    workers._limit_onnx_threads(2, use_gpu=False)
    ort.InferenceSession("model.onnx", providers=[CPU])
    assert seen == [[CPU]]


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


# ---------------------------------------------------------------------------
# Partial index: ask while a document is still being ingested
# ---------------------------------------------------------------------------


def _file_digests(index_dir: Path) -> dict[str, tuple[bytes, int]]:
    return {
        name: ((index_dir / name).read_bytes(), (index_dir / name).stat().st_mtime_ns)
        for name in ("chunks.jsonl", "vectors.f16", "state.json")
    }


def test_partial_index_is_none_before_the_first_checkpoint(long_pdf, tmp_path, monkeypatch):
    assert load_partial_index(tmp_path / "missing") is None
    monkeypatch.setattr(streaming, "INDEX_BATCH_PAGES", 12)
    seen = []

    def watch(p):
        _, index_dir = store_dirs(tmp_path, load_sha(long_pdf), CFG, use_ocr=False)
        seen.append((p.stage, load_partial_index(index_dir)))

    ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), progress=watch, workers=1)
    assert all(idx is None for stage, idx in seen if stage == "reading")  # pass 1 only
    assert any(idx is not None for stage, idx in seen if stage == "indexing")


def test_partial_index_holds_exactly_the_checkpointed_prefix(
    long_pdf, reference, tmp_path, monkeypatch
):
    monkeypatch.setattr(streaming, "INDEX_BATCH_PAGES", 12)
    ref, snapshots = reference[0], []

    def watch(p):
        if p.stage != "indexing":
            return
        _, index_dir = store_dirs(tmp_path, ref.meta["sha256"], CFG, use_ocr=False)
        snapshots.append((p.pages_done, load_partial_index(index_dir)))

    ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), progress=watch, workers=1)
    assert sorted({pages for pages, _ in snapshots}) == [12, 24, 36, 48, 60]
    for pages, partial in snapshots:
        assert partial.pages_indexed == pages and partial.pages_total == N_PAGES
        n = len(partial)
        assert n > 0 and max(c.page for c in partial.chunks) <= pages
        assert [c.to_dict() for c in partial.chunks] == [c.to_dict() for c in ref.chunks[:n]]
        np.testing.assert_array_equal(partial.embeddings, ref.embeddings[:n])
        assert partial.bm25.n_docs == n and partial.bm25_search("uniqueterm3", 3)
        # no chunk beyond the pages covered by the checkpoint
        assert n == sum(c.page <= pages for c in ref.chunks)


def test_partial_load_never_reads_past_the_checkpoint_and_never_writes(
    long_pdf, reference, tmp_path, monkeypatch
):
    monkeypatch.setattr(streaming, "INDEX_BATCH_PAGES", 12)
    states = []

    def grab(p):
        if p.stage == "indexing" and p.pages_done == 24:
            _, d = store_dirs(tmp_path, reference[0].meta["sha256"], CFG, use_ocr=False)
            states.append(json.loads((d / "state.json").read_text()))

    ingest_pdf(long_pdf, tmp_path, CFG, FakeEmbedder(), progress=grab, workers=1)
    _, index_dir = store_dirs(tmp_path, reference[0].meta["sha256"], CFG, use_ocr=False)
    # Rewind state.json to the page-24 checkpoint while the files already hold later batches
    # (exactly what a reader sees while the writer is mid-batch), plus a torn half record.
    state = states[0]
    (index_dir / "state.json").write_text(json.dumps(state))
    with open(index_dir / "chunks.jsonl", "ab") as f:
        f.write(b'{"chunk_id": "half-written')
    with open(index_dir / "vectors.f16", "ab") as f:
        f.write(b"\x01\x02\x03")
    before = _file_digests(index_dir)
    listing = sorted(p.name for p in index_dir.iterdir())

    partial = load_partial_index(index_dir)

    assert partial.pages_indexed == 24 and len(partial) == state["n_chunks"]
    assert [c.to_dict() for c in partial.chunks] == [
        c.to_dict() for c in reference[0].chunks[: state["n_chunks"]]
    ]
    assert _file_digests(index_dir) == before  # bytes and mtimes: nothing written or truncated
    assert sorted(p.name for p in index_dir.iterdir()) == listing  # no files created


def test_partial_load_rejects_files_that_do_not_back_the_checkpoint(tmp_path):
    state = {
        "format": streaming.FORMAT, "doc_id": "d", "title": "T", "num_pages": 5, "pages_done": 3,
        "n_chunks": 2, "chunk_bytes": 10_000, "vector_bytes": 2 * 4 * 2, "dim": 4, "complete": False,
    }  # fmt: skip
    (tmp_path / "state.json").write_text(json.dumps(state))
    assert load_partial_index(tmp_path) is None  # data files missing
    (tmp_path / "chunks.jsonl").write_bytes(b"{}\n")
    (tmp_path / "vectors.f16").write_bytes(b"\0" * 16)
    assert load_partial_index(tmp_path) is None  # chunks.jsonl shorter than the checkpoint


class _SlowEmbedder(FakeEmbedder):
    def encode_passages(self, texts):
        time.sleep(0.02)  # keep the writer busy while the reader polls
        return super().encode_passages(texts)


def test_reading_while_the_ingest_thread_appends_is_safe_and_changes_nothing(
    long_pdf, reference, tmp_path, monkeypatch
):
    monkeypatch.setattr(streaming, "INDEX_BATCH_PAGES", 6)
    ref = reference[0]
    _, index_dir = store_dirs(tmp_path, ref.meta["sha256"], CFG, use_ocr=False)
    stop, errors, loads = threading.Event(), [], []

    def reader():
        last = 0
        while not stop.is_set():
            try:
                idx = load_partial_index(index_dir)
                if idx is None:
                    continue
                n = len(idx)
                assert [c.to_dict() for c in idx.chunks] == [c.to_dict() for c in ref.chunks[:n]]
                np.testing.assert_array_equal(idx.embeddings, ref.embeddings[:n])
                assert idx.pages_indexed >= last
                last = idx.pages_indexed
                loads.append(n)
            except BaseException as exc:  # noqa: BLE001 - reported on the main thread
                errors.append(exc)
                return

    t = threading.Thread(target=reader)
    t.start()
    try:
        index, _ = ingest_pdf(long_pdf, tmp_path, CFG, _SlowEmbedder(), workers=1)
    finally:
        stop.set()
        t.join(30)
    assert not errors, errors
    assert len(set(loads)) > 2  # the reader really saw the index grow
    # The finished index is byte-identical to a run nobody queried.
    _same(index, ref)
    ref_dir = Path(ref.meta["index_dir"])
    for name in ("chunks.jsonl", "vectors.f16", "bm25_vocab.json"):
        assert (index_dir / name).read_bytes() == (ref_dir / name).read_bytes(), name
    a, b = BM25Index.load(index_dir), BM25Index.load(ref_dir)
    assert (a._weights != b._weights).nnz == 0 and a.vocab == b.vocab
