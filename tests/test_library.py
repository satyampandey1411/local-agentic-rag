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


def _pipe(on_generate=None, nothing_relevant=False) -> RAGPipeline:
    cfg = RAGConfig(ocr="off")
    llm = ScriptedLLM(
        scripted(
            plan=lambda m: {"standalone_question": "q", "search_queries": ["uniqueterm3"]},
            grade=lambda m: {
                "relevant": [] if nothing_relevant else [passage_number(m, "uniqueterm3")],
                "sufficient": not nothing_relevant,
                "missing_query": "",
            },
            generate=lambda m: (
                on_generate() if on_generate else None,
                "ANSWER_NOT_FOUND"
                if nothing_relevant
                else f"Page three says so [{passage_number(m, 'uniqueterm3')}].",
            )[1],
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
    boxes = lib.highlight_boxes(rec.doc_id, 3, ans.citations[0].chunk_id)
    assert boxes and all(0 <= v <= 1 and b[0] < b[2] and b[1] < b[3] for b in boxes for v in b)
    assert lib.highlight_boxes(rec.doc_id, 3, None) == []
    assert lib.highlight_boxes(rec.doc_id, 4, ans.citations[0].chunk_id) == []  # chunk is on p.3
    with pytest.raises(IndexError):
        lib.highlight_boxes(rec.doc_id, 31, None)
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
    resubmitted = threading.Event()  # keeps the cancelled job's cleanup pending until then

    def slow_first(path, *args, **kwargs):
        if kwargs["doc_id"] == "report":  # the first upload: run until cancelled
            started.set()
            while not kwargs["should_stop"]():
                time.sleep(0.02)
            resubmitted.wait(30)
            raise IngestCancelled("cancelled")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(library_mod, "ingest_pdf", slow_first)
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    first = lib.submit(upload(), "report.pdf")
    assert started.wait(30)
    lib.delete(first.doc_id)
    second = lib.submit(_other_pdf(tmp_path, "report.pdf"), "report.pdf")
    assert second.doc_id != first.doc_id  # the id is not reused while cleanup is pending
    resubmitted.set()
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


# ---------------------------------------------------------------------------
# Ask while a document is still being ingested
# ---------------------------------------------------------------------------


@pytest.fixture()
def paused_ingest(monkeypatch):
    """Pause the ingest thread right after its checkpoint at page ``PAUSE_AT`` of pass 2."""
    import threading

    from local_agentic_rag import library as library_mod
    from local_agentic_rag.ingest import streaming

    monkeypatch.setattr(streaming, "INDEX_BATCH_PAGES", 10)
    real = library_mod.ingest_pdf
    reached, proceed = threading.Event(), threading.Event()

    def pausing(path, *args, **kwargs):
        real_progress = kwargs["progress"]

        def progress(p):
            real_progress(p)  # the library records the checkpoint first ...
            if p.stage == "indexing" and p.pages_done >= 10 and not reached.is_set():
                reached.set()
                proceed.wait(60)  # ... then the ingest thread waits here

        return real(path, *args, **{**kwargs, "progress": progress})

    monkeypatch.setattr(library_mod, "ingest_pdf", pausing)
    yield reached, proceed
    proceed.set()


def _store_bytes(lib: Library, doc_id: str) -> dict[str, bytes]:
    _, index_dir, _ = lib._row(doc_id)
    return {n: (Path(index_dir) / n).read_bytes() for n in ("chunks.jsonl", "vectors.f16")}


def test_processing_document_with_a_checkpoint_is_askable_and_labelled(
    tmp_path, upload, paused_ingest, monkeypatch
):
    reached, proceed = paused_ingest
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    rec = lib.submit(upload(), "Annual Report 2025.pdf")
    assert reached.wait(60)
    live = lib.get(rec.doc_id)
    assert live.status == "processing" and live.askable and live.pages_indexed == 10
    assert live.to_dict()["askable"] is True and live.to_dict()["pages_indexed"] == 10

    ans = lib.ask("Where is uniqueterm3?", [rec.doc_id], history=[])
    assert not ans.refused and ans.cited_pages == [3]
    assert ans.partial == {"doc_id": rec.doc_id, "pages_indexed": 10, "pages_total": 30}
    assert ans.to_dict()["partial"] == ans.partial
    assert rec.doc_id not in lib._loaded  # a partial index is never cached as the final one
    first = lib._partial[rec.doc_id][2]
    lib.ask("Where is uniqueterm3?", [rec.doc_id], history=[])
    assert lib._partial[rec.doc_id][2] is first  # same checkpoint: not rebuilt
    assert lib.render_page(rec.doc_id, 3, chunk_id=ans.citations[0].chunk_id)[:4] == b"\x89PNG"
    with pytest.raises(ValueError):  # the default selection is ready documents only
        lib.ask("Where is uniqueterm3?", None, history=[])

    proceed.set()
    done = _wait(lib, rec.doc_id)
    assert not done.askable and done.pages_indexed == 30
    final = lib.ask("Where is uniqueterm3?", [rec.doc_id], history=[])
    assert final.partial is None and "partial" not in final.to_dict()
    assert rec.doc_id not in lib._partial and rec.doc_id in lib._loaded
    with_queries = _store_bytes(lib, rec.doc_id)
    lib.close()

    # A library nobody asked while it ingested (and with other batch boundaries) produces
    # byte-identical index files.
    monkeypatch.undo()
    lib2 = Library(tmp_path / "lib2", pipeline=_pipe())
    rec2 = lib2.submit(upload(), "Annual Report 2025.pdf")
    _wait(lib2, rec2.doc_id)
    assert _store_bytes(lib2, rec2.doc_id) == with_queries
    lib2.close()


def test_refusal_on_a_partial_document_still_carries_the_banner_fields(
    tmp_path, upload, paused_ingest
):
    reached, proceed = paused_ingest
    lib = Library(tmp_path / "lib", pipeline=_pipe(nothing_relevant=True))
    rec = lib.submit(upload(), "report.pdf")
    assert reached.wait(60)
    # a question whose answer lies beyond the indexed prefix finds nothing relevant
    ans = lib.ask("What does page 25 say?", [rec.doc_id], history=[])
    assert ans.refused
    assert ans.partial == {"doc_id": rec.doc_id, "pages_indexed": 10, "pages_total": 30}
    proceed.set()
    lib.close()


def test_processing_document_without_a_checkpoint_is_not_askable(tmp_path, upload):
    lib = Library(tmp_path / "lib", pipeline=_pipe(), start_worker=False)
    rec = lib.submit(upload(), "report.pdf")
    assert not rec.askable and rec.pages_indexed == 0
    lib._update(rec.doc_id, status="processing")
    assert lib.partial_index(rec.doc_id) is None
    with pytest.raises(ValueError, match="not ready"):
        lib.ask("Where is uniqueterm3?", [rec.doc_id], history=[])
    lib.close()


def test_delete_during_a_partial_ask_is_safe(tmp_path, upload, paused_ingest):
    import threading

    reached, proceed = paused_ingest
    asking, release, result = threading.Event(), threading.Event(), {}

    def block_once():
        if not asking.is_set():
            asking.set()  # the question is now inside the model call
            release.wait(60)

    lib = Library(tmp_path / "lib", pipeline=_pipe(on_generate=block_once))
    rec = lib.submit(upload(), "report.pdf")
    assert reached.wait(60)

    def ask():
        try:
            result["answer"] = lib.ask("Where is uniqueterm3?", [rec.doc_id], history=[])
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    t = threading.Thread(target=ask)
    t.start()
    assert asking.wait(60)
    lib.delete(rec.doc_id)  # mid-question, mid-ingest
    release.set()
    proceed.set()
    t.join(60)
    deadline = time.time() + 60
    while lib._active is not None:
        assert time.time() < deadline
        time.sleep(0.02)
    with lib._job_lock:  # the job clears _active and removes the files under this lock
        pass

    assert "error" not in result, result.get("error")  # the ask held its own in-memory copy
    assert result["answer"].partial["doc_id"] == rec.doc_id and not result["answer"].error
    with pytest.raises(KeyError):
        lib.get(rec.doc_id)
    with pytest.raises(KeyError):  # a later partial ask finds nothing, and caches nothing
        lib.ask("Where is uniqueterm3?", [rec.doc_id], history=[])
    assert rec.doc_id not in lib._partial and rec.doc_id not in lib._loaded
    assert not [p for p in (tmp_path / "lib" / "store").glob("*-*") if p.name != "parsed"]
    assert not (tmp_path / "lib" / "files" / f"{rec.doc_id}.pdf").exists()
    lib.close()


def test_partial_index_built_while_ingest_finishes_is_not_cached(
    tmp_path, upload, paused_ingest, monkeypatch
):
    """An ask that saw 'processing' and then lost the race with the end of ingest must not
    leave a partial index in the cache (nothing would ever drop it) nor answer from it."""
    from local_agentic_rag import library as library_mod

    reached, proceed = paused_ingest
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    rec = lib.submit(upload(), "report.pdf")
    assert reached.wait(60)
    real = library_mod.load_partial_index

    def load_then_ingest_finishes(index_dir):
        index = real(index_dir)  # built from the checkpoint ...
        proceed.set()  # ... and the ingest completes before the caller can cache it
        _wait(lib, rec.doc_id)
        deadline = time.time() + 30
        while rec.doc_id in lib._partial or rec.doc_id not in lib._loaded:
            assert time.time() < deadline
            time.sleep(0.01)
        time.sleep(0.2)  # let _ingest finish its own cleanup
        return index

    monkeypatch.setattr(library_mod, "load_partial_index", load_then_ingest_finishes)
    got = lib.partial_index(rec.doc_id)
    assert rec.doc_id not in lib._partial
    assert got is not None and got.pages_indexed is None  # the final index: no partial banner
    assert got is lib._loaded[rec.doc_id]
    lib.close()


def test_concurrent_partial_asks_share_one_build(tmp_path, upload, paused_ingest, monkeypatch):
    import threading

    from local_agentic_rag import library as library_mod

    reached, proceed = paused_ingest
    lib = Library(tmp_path / "lib", pipeline=_pipe())
    rec = lib.submit(upload(), "report.pdf")
    assert reached.wait(60)
    real = library_mod.load_partial_index
    builds, results = [], []

    def slow_load(index_dir):
        builds.append(1)
        time.sleep(0.4)  # long enough for the second ask to arrive and wait
        return real(index_dir)

    monkeypatch.setattr(library_mod, "load_partial_index", slow_load)
    threads = [
        threading.Thread(target=lambda: results.append(lib.partial_index(rec.doc_id)))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert len(builds) == 1
    assert len(results) == 2 and results[0] is results[1] and results[0].pages_indexed == 10
    proceed.set()
    lib.close()


def test_display_title_prefers_a_meaningful_file_name():
    from local_agentic_rag.library import display_title

    assert display_title("9-11_Commission_Report", "201-635.job") == "9-11 Commission Report"
    assert display_title("scan_0012", "Annual Report 2024") == "Annual Report 2024"
    assert display_title("document (3)", "Budget") == "Budget"
    assert display_title("a3f9c2b1d4e5f6a7", "") == "a3f9c2b1d4e5f6a7"
