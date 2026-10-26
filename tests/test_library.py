"""Document library: background ingestion, dedupe, delete/cancel, restart-resume, ask, pages."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")
pytest.importorskip("reportlab")

from local_agentic_rag.config import RAGConfig  # noqa: E402
from local_agentic_rag.library import Library, highlight_segments  # noqa: E402
from local_agentic_rag.llm.fake import ScriptedLLM  # noqa: E402
from local_agentic_rag.pipeline import RAGPipeline  # noqa: E402
from tests.helpers import (  # noqa: E402
    FakeEmbedder,
    FakeReranker,
    make_blank_pdf,
    make_text_pdf,
    passage_number,
    scripted,
)


def _pipe() -> RAGPipeline:
    cfg = RAGConfig(ocr="off")
    llm = ScriptedLLM(
        scripted(
            plan=lambda m: {"standalone_question": "q", "search_queries": ["uniqueterm3"]},
            grade=lambda m: {
                "relevant": [passage_number(m, "uniqueterm3")],
                "sufficient": True,
                "missing_query": "",
            },
            generate=lambda m: f"Page three says so [{passage_number(m, 'uniqueterm3')}].",
            verify={"supported": True, "unsupported_claims": []},
        )
    )
    pipe = RAGPipeline(cfg, llm=llm, cache_dir=None)
    pipe._shared["embedders"][cfg.embedder_model] = FakeEmbedder()
    pipe._shared["rerankers"][cfg.reranker_model] = FakeReranker()
    return pipe


def _wait(lib: Library, doc_id: str, status: str = "ready", timeout: float = 60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = lib.get(doc_id)
        if rec.status == status:
            return rec
        if rec.status == "failed" and status != "failed":
            raise AssertionError(rec.error)
        time.sleep(0.05)
    raise AssertionError(f"{doc_id} stuck in {lib.get(doc_id).status}")


@pytest.fixture()
def upload(tmp_path):
    """Returns a fresh temp copy of a generated PDF (submit moves the file)."""
    src = tmp_path / "src.pdf"
    make_text_pdf(src, n_pages=30)

    def make(name: str = "Annual Report 2025.pdf") -> str:
        dst = tmp_path / f"up-{time.perf_counter_ns()}.pdf"
        shutil.copy(src, dst)
        return str(dst)

    return make


def test_submit_ingests_in_background_and_answers(tmp_path, upload):
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    rec = lib.submit(upload(), "Annual Report 2025.pdf")
    assert rec.status in ("queued", "processing") and rec.num_pages == 30
    rec = _wait(lib, rec.doc_id)
    assert rec.doc_id == "annual-report-2025" and rec.num_chunks > 0 and rec.pages_done == 30
    steps = []
    ans = lib.ask("Where is uniqueterm3?", [rec.doc_id], history=[], on_step=steps.append)
    assert not ans.refused and ans.cited_pages == [3]
    assert steps and lib.health()["ready"] == 1
    png = lib.render_page(rec.doc_id, 3, chunk_id=ans.citations[0].chunk_id)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(IndexError):
        lib.render_page(rec.doc_id, 31)
    lib.close()


def test_same_file_twice_is_one_document(tmp_path, upload):
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    a = lib.submit(upload(), "report.pdf")
    second = upload()
    b = lib.submit(second, "copy of report.pdf")
    assert a.doc_id == b.doc_id and not Path(second).exists()
    assert len(lib.list_documents()) == 1
    lib.close()


def test_unreadable_pdf_fails_with_message_and_retry_requeues(tmp_path):
    blank = tmp_path / "scan.pdf"
    make_blank_pdf(blank)
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    rec = _wait(lib, lib.submit(str(blank), "scan.pdf").doc_id, status="failed")
    assert "No text could be extracted" in rec.error
    assert lib.retry(rec.doc_id).status == "queued"
    _wait(lib, rec.doc_id, status="failed")
    with pytest.raises(ValueError):
        lib.ask("anything", [rec.doc_id], history=[])
    lib.close()


def test_delete_removes_files_and_unknown_ids_raise(tmp_path, upload):
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    rec = _wait(lib, lib.submit(upload(), "report.pdf").doc_id)
    lib.delete(rec.doc_id)
    assert lib.list_documents() == []
    assert not any((tmp_path / "lib" / "files").iterdir())
    assert not [p for p in (tmp_path / "lib" / "store").glob("*") if p.name != "parsed"]
    with pytest.raises(KeyError):
        lib.get(rec.doc_id)
    with pytest.raises(KeyError):
        lib.delete(rec.doc_id)
    lib.close()


def test_restart_resumes_interrupted_job(tmp_path, upload, monkeypatch):
    from local_agentic_rag import library as library_mod

    calls = {"n": 0}
    real = library_mod.ingest_pdf

    def stop_midway(*args, **kwargs):
        user_stop = kwargs["should_stop"]

        def stop():
            calls["n"] += 1
            return calls["n"] >= 2 or user_stop()

        return real(*args, **{**kwargs, "should_stop": stop})

    monkeypatch.setattr(library_mod, "ingest_pdf", stop_midway)
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    rec = lib.submit(upload(), "report.pdf")
    deadline = time.time() + 30
    while lib._active is None and lib.get(rec.doc_id).status != "processing":
        assert time.time() < deadline
        time.sleep(0.02)
    while lib._active is not None:  # the "crash": the job stopped after one checkpoint
        assert time.time() < deadline
        time.sleep(0.02)
    lib.close()
    assert lib.get(rec.doc_id).status == "processing"

    monkeypatch.setattr(library_mod, "ingest_pdf", real)
    lib2 = Library(tmp_path / "lib", pipeline=_pipe())  # a new process after a restart
    done = _wait(lib2, rec.doc_id)
    assert done.num_chunks > 0
    lib2.close()


def test_highlight_segments():
    segs = highlight_segments("Real GDP increased 3.8 percent in the second quarter. It rose.")
    assert segs[0] == "Real GDP increased 3.8 percent in the second"
    table = "| Line | 2024 |\n|---|---|\n| Exports | 4.5 |"
    assert highlight_segments(table, "table") == ["Exports", "4.5"]


def _other_pdf(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    make_text_pdf(path, n_pages=5)
    import pymupdf  # different bytes than the 30-page upload

    with pymupdf.open(path) as d:
        d.set_metadata({"title": "Corrected report"})
        d.save(tmp_path / f"m-{name}")
    return str(tmp_path / f"m-{name}")


def test_delete_while_processing_then_reupload_same_name_keeps_new_file(
    tmp_path, upload, monkeypatch
):
    """Review finding: the old job's cleanup must not delete the re-uploaded file."""
    import threading

    from local_agentic_rag import library as library_mod
    from local_agentic_rag.ingest.streaming import IngestCancelled

    real = library_mod.ingest_pdf
    started = threading.Event()

    def slow_first(path, *args, **kwargs):
        if kwargs["doc_id"] == "report":  # the first upload: run until cancelled
            started.set()
            while not kwargs["should_stop"]():
                time.sleep(0.02)
            raise IngestCancelled("cancelled")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(library_mod, "ingest_pdf", slow_first)
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    first = lib.submit(upload(), "report.pdf")
    assert started.wait(30)
    lib.delete(first.doc_id)
    second = lib.submit(_other_pdf(tmp_path, "report.pdf"), "report.pdf")
    assert second.doc_id != first.doc_id  # the id is not reused while cleanup is pending
    done = _wait(lib, second.doc_id)
    assert done.status == "ready"
    assert (tmp_path / "lib" / "files" / f"{second.doc_id}.pdf").exists()
    assert not (tmp_path / "lib" / "files" / "report.pdf").exists()
    lib.close()


def test_concurrent_uploads_with_the_same_name_get_distinct_ids(tmp_path, upload):
    import threading

    from local_agentic_rag.ingest.pdf import sha256_file

    lib = Library(tmp_path / "lib", pipeline=_pipe(), start_worker=False)
    paths = [upload(), _other_pdf(tmp_path, "x.pdf")]
    shas = [sha256_file(p) for p in paths]
    barrier, recs = threading.Barrier(2), []

    def go(path):
        barrier.wait()
        recs.append(lib.submit(path, "same name.pdf"))

    threads = [threading.Thread(target=go, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({r.doc_id for r in recs}) == 2
    for r in recs:  # each catalog row describes the file stored under its id
        _, _, sha = lib._row(r.doc_id)
        assert sha256_file(tmp_path / "lib" / "files" / f"{r.doc_id}.pdf") == sha
    assert sorted(lib._row(r.doc_id)[2] for r in recs) == sorted(shas)
    lib.close()


def test_token_counting_shares_the_model_lock():
    """Review finding: the fast tokenizer is not thread-safe, so count_tokens is serialised."""
    import threading

    from local_agentic_rag.library import _Locked

    lock = threading.RLock()
    locked = _Locked(FakeEmbedder(), lock)
    done = threading.Event()
    with lock:
        t = threading.Thread(target=lambda: (locked.count_tokens("a b c"), done.set()))
        t.start()
        assert not done.wait(0.3)  # blocked while another thread holds the lock
    assert done.wait(5)
    t.join()
