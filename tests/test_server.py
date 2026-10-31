"""HTTP server tests against an in-memory FakeLibrary (no GPU, no PDFs parsed, < 10 s)."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_agentic_rag.library_types import DocRecord, LibraryProtocol
from local_agentic_rag.schema import Answer, Chunk, Citation, ScoredChunk, TraceStep
from local_agentic_rag.server import create_app

PDF = b"%PDF-1.4\n%fake but good enough for the fake library\n" + b"x" * 2000


# ============================================================================ fake library


class FakeLibrary:
    """Thread-safe in-memory LibraryProtocol.

    * Uploaded docs advance one tick per ``get``: queued -> processing (0, 25, 50, 75 %)
      -> ready.  A filename containing ``bad`` fails at 50 %.
    * ``ask`` emits plan/retrieve/generate/verify steps; questions containing ``slow``
      block until ``release`` is set; ``boom`` raises; ``refuse`` returns a refusal.
    """

    ANSWER = "Revenue grew 4% [1]."

    def __init__(self, store: Path) -> None:
        self.store = store
        self.lock = threading.Lock()
        self.docs: dict[str, DocRecord] = {}
        self.auto: set[str] = set()
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0
        self.asks: list[dict[str, Any]] = []
        self.health_error = False

    def add_ready(self, doc_id: str, title: str = "Ready doc", pages: int = 10) -> None:
        self.docs[doc_id] = DocRecord(
            doc_id=doc_id,
            title=title,
            filename=f"{doc_id}.pdf",
            status="ready",
            num_pages=pages,
            pages_done=pages,
            num_chunks=40,
            created_at=time.time(),
        )

    def add_processing(
        self, doc_id: str, *, askable: bool, pages_indexed: int = 0, pages: int = 100
    ) -> None:
        self.docs[doc_id] = DocRecord(
            doc_id=doc_id,
            title=f"Processing {doc_id}",
            filename=f"{doc_id}.pdf",
            status="processing",
            num_pages=pages,
            pages_done=pages // 2,
            askable=askable,
            pages_indexed=pages_indexed,
            created_at=time.time(),
        )

    def submit(self, src_path: str, filename: str) -> DocRecord:
        data = Path(src_path).read_bytes()
        doc_id = hashlib.sha256(data).hexdigest()[:10]
        with self.lock:
            if doc_id in self.docs:
                Path(src_path).unlink()
                return replace(self.docs[doc_id])
            shutil.move(src_path, self.store / f"{doc_id}.pdf")
            now = time.time()
            rec = DocRecord(
                doc_id=doc_id,
                title=Path(filename).stem,
                filename=filename,
                status="queued",
                size_bytes=len(data),
                created_at=now,
                updated_at=now,
            )
            self.docs[doc_id] = rec
            self.auto.add(doc_id)
            return replace(rec)

    def _tick(self, rec: DocRecord) -> None:
        if rec.status == "queued":
            rec.status, rec.num_pages, rec.stage = "processing", 100, "Reading pages"
        elif rec.status == "processing":
            rec.pages_done += 25
            rec.stage = f"Reading pages {rec.pages_done - 24}-{rec.pages_done} of 100"
            rec.eta_seconds = float(100 - rec.pages_done)
            if "bad" in rec.filename and rec.pages_done >= 50:
                rec.status, rec.error, rec.eta_seconds = "failed", "simulated parser crash", None
            elif rec.pages_done >= 100:
                rec.status, rec.eta_seconds, rec.ingest_seconds = "ready", None, 1.5
        rec.updated_at = time.time()

    def list_documents(self) -> list[DocRecord]:
        with self.lock:
            return [replace(d) for d in self.docs.values()]

    def get(self, doc_id: str) -> DocRecord:
        with self.lock:
            rec = self.docs[doc_id]
            if doc_id in self.auto and rec.status in ("queued", "processing"):
                self._tick(rec)
            return replace(rec)

    def delete(self, doc_id: str) -> None:
        with self.lock:
            del self.docs[doc_id]

    def retry(self, doc_id: str) -> DocRecord:
        with self.lock:
            rec = self.docs[doc_id]
            if rec.status != "failed":
                raise ValueError("Only failed documents can be retried.")
            rec.status, rec.error, rec.pages_done = "queued", None, 0
            rec.filename = rec.filename.replace("bad", "ok")
            return replace(rec)

    def ask(self, question, doc_ids, history, on_step=None, on_token=None) -> Answer:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.asks.append({"question": question, "doc_ids": doc_ids, "history": history})
        try:
            if "boom" in question:
                raise RuntimeError("kaboom")
            if "slow" in question:
                self.release.wait(5)
            if "wait" in question:
                time.sleep(0.15)
            ans = Answer(question=question, answer="", refused=False)
            for name in ("plan", "retrieve", "generate", "verify"):
                if name == "generate" and on_token and "refuse" not in question:
                    on_token({"reset": True})
                    for i in range(0, len(self.ANSWER), 4):  # many tiny pieces, like a model
                        on_token({"text": self.ANSWER[i : i + 4]})
                step = TraceStep(
                    step=name, detail={"pages": [3]} if name == "retrieve" else {}, seconds=0.01
                )
                ans.trace.append(step)
                if on_step:
                    on_step(step)
            if "refuse" in question:
                ans.refused, ans.answer = True, "I could not find the answer."
                return ans
            doc = (doc_ids or ["d1"])[0]
            chunk = Chunk(chunk_id=f"{doc}-c1", doc_id=doc, page=3, section="", text="Revenue")
            ans.answer = self.ANSWER
            ans.citations = [
                Citation(
                    marker=1, chunk_id=chunk.chunk_id, doc_id=doc, page=3, snippet="Revenue grew"
                )
            ]
            ans.retrieved = [
                ScoredChunk(chunk=chunk, score=float("nan"))
            ]  # NaN must not break JSON
            ans.grounded = True
            rec = self.docs.get(doc)
            if rec is not None and rec.status == "processing":
                ans.partial = {
                    "doc_id": doc,
                    "pages_indexed": rec.pages_indexed,
                    "pages_total": rec.num_pages,
                }
            return ans
        finally:
            with self.lock:
                self.active -= 1

    def render_page(self, doc_id, page, zoom=1.5, chunk_id=None) -> bytes:
        rec = self.get(doc_id)
        if not 1 <= page <= rec.num_pages:
            raise IndexError(page)
        return b"\x89PNG\r\n\x1a\n" + f"{doc_id}:{page}:{zoom}:{chunk_id}".encode()

    def highlight_boxes(self, doc_id, page, chunk_id):
        rec = self.get(doc_id)
        if not 1 <= page <= rec.num_pages:
            raise IndexError(page)
        return [[0.1, 0.4, 0.9, 0.45], [0.1, 0.46, 0.5, 0.5]] if chunk_id else []

    def page_count(self, doc_id: str) -> int:
        return self.get(doc_id).num_pages

    def health(self) -> dict[str, Any]:
        if self.health_error:
            raise RuntimeError("ollama down")
        return {"llm": "ok", "device": "cpu", "queue": 0, "busy": False}


# ============================================================================ helpers


def parse_sse(text: str) -> list[tuple[str, Any]]:
    events = []
    for block in text.split("\n\n"):
        name, data = "message", []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
        if data:
            events.append((name, json.loads("\n".join(data))))
    return events


@pytest.fixture
def lib(tmp_path: Path) -> FakeLibrary:
    store = tmp_path / "store"
    store.mkdir()
    fake = FakeLibrary(store)
    fake.add_ready("d1", "Annual report")
    yield fake
    fake.release.set()


def make_client(lib: FakeLibrary, tmp_path: Path, **kw: Any) -> TestClient:
    kw.setdefault("events_poll_s", 0.01)
    kw.setdefault("heartbeat_s", 0.05)
    app = create_app(lib, upload_dir=str(tmp_path / "incoming"), **kw)
    return TestClient(app)


@pytest.fixture
def client(lib: FakeLibrary, tmp_path: Path):
    with make_client(lib, tmp_path) as c:
        yield c


def upload(client: TestClient, data: bytes = PDF, name: str = "report.pdf"):
    return client.post("/api/documents", files={"file": (name, data, "application/pdf")})


def test_fake_library_satisfies_protocol(lib: FakeLibrary) -> None:
    _: LibraryProtocol = lib  # static check; also documents the intent


# ============================================================================ uploads


def test_upload_returns_record_and_moves_temp_file(client, lib, tmp_path) -> None:
    r = upload(client)
    assert r.status_code == 201, r.text
    rec = r.json()
    assert rec["status"] == "queued" and rec["filename"] == "report.pdf"
    assert rec["size_bytes"] == len(PDF)
    assert (lib.store / f"{rec['doc_id']}.pdf").read_bytes() == PDF
    assert list((tmp_path / "incoming").iterdir()) == []  # library took ownership


def test_same_file_twice_returns_existing_record(client) -> None:
    a, b = upload(client).json(), upload(client, name="copy.pdf").json()
    assert a["doc_id"] == b["doc_id"] and b["filename"] == "report.pdf"


def test_upload_filename_is_sanitised(client) -> None:
    assert upload(client, name="../../etc/passwd").json()["filename"] == "passwd.pdf"
    other = PDF + b"2"
    assert upload(client, data=other, name="..\\..\\C:\\Win  report.PDF").json()["filename"] == (
        "Win report.PDF"
    )


@pytest.mark.parametrize(
    ("data", "status", "fragment"),
    [
        (b"PK\x03\x04 this is a zip" + b"z" * 2000, 415, "not a PDF"),
        (b"hello", 415, "not a PDF"),
        (b"", 415, "empty"),
    ],
)
def test_non_pdf_uploads_are_rejected(client, tmp_path, data, status, fragment) -> None:
    r = upload(client, data=data, name="x.pdf")
    assert r.status_code == status
    assert fragment in r.json()["detail"]
    assert list((tmp_path / "incoming").iterdir()) == []


def test_pdf_magic_may_follow_leading_junk(client) -> None:
    assert upload(client, data=b"\r\n\r\n junk " + PDF).status_code == 201


def test_oversize_upload_rejected_by_content_length(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, max_upload_mb=1) as c:
        r = upload(c, data=PDF + b"0" * (2 * 1024 * 1024))
    assert r.status_code == 413 and "1 MB" in r.json()["detail"]


def test_oversize_upload_rejected_while_streaming(lib, tmp_path) -> None:
    """No Content-Length (chunked): the limit is enforced as bytes arrive."""
    boundary = "XyZ123"
    sent = {"chunks": 0}

    def body():
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="big.pdf"\r\nContent-Type: application/pdf\r\n\r\n'
        ).encode()
        yield PDF
        for _ in range(40):  # 40 x 64 KiB = 2.5 MiB > 1 MiB limit
            sent["chunks"] += 1
            yield b"0" * 65536
        yield f"\r\n--{boundary}--\r\n".encode()

    with make_client(lib, tmp_path, max_upload_mb=1) as c:
        r = c.post(
            "/api/documents",
            content=body(),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
    assert r.status_code == 413
    assert list((tmp_path / "incoming").iterdir()) == []
    assert lib.list_documents()[0].doc_id == "d1" and len(lib.list_documents()) == 1


def test_upload_requires_multipart_and_file_field(client) -> None:
    assert (
        client.post(
            "/api/documents", content=PDF, headers={"content-type": "application/pdf"}
        ).status_code
        == 415
    )
    r = client.post("/api/documents", data={"note": "no file"})
    assert r.status_code in (400, 415)
    assert isinstance(r.json()["detail"], str)


def test_truncated_upload_is_rejected(client, tmp_path) -> None:
    boundary = "B0und"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="t.pdf"\r\n\r\n'
    ).encode() + PDF  # no closing boundary
    r = client.post(
        "/api/documents",
        content=body,
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert r.status_code == 400 and "incomplete" in r.json()["detail"]
    assert list((tmp_path / "incoming").iterdir()) == []


# ============================================================================ documents


def test_list_get_delete_and_unknown_ids(client) -> None:
    upload(client)
    docs = client.get("/api/documents").json()["documents"]
    assert {d["doc_id"] for d in docs} >= {"d1"} and len(docs) == 2
    assert client.get("/api/documents/d1").json()["title"] == "Annual report"
    for method, url in [
        ("GET", "/api/documents/nope"),
        ("DELETE", "/api/documents/nope"),
        ("POST", "/api/documents/nope/retry"),
        ("GET", "/api/documents/nope/events"),
        ("GET", "/api/documents/nope/pages/1.png"),
    ]:
        r = client.request(method, url)
        assert r.status_code == 404, (method, url)
        assert r.json() == {"detail": "Document not found."}
    assert client.delete("/api/documents/d1").status_code == 204
    assert client.get("/api/documents/d1").status_code == 404


def test_progress_events_stream_until_ready(client) -> None:
    doc_id = upload(client).json()["doc_id"]
    r = client.get(f"/api/documents/{doc_id}/events")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    docs = [d for name, d in events if name == "document"]
    statuses = [d["status"] for d in docs]
    assert statuses[0] in ("queued", "processing") and statuses[-1] == "ready"
    assert "processing" in statuses
    progress = [d["pages_done"] for d in docs if d["status"] == "processing"]
    assert progress == sorted(progress) and len(progress) >= 3
    assert len({json.dumps(d, sort_keys=True) for d in docs}) == len(docs)  # only changes sent
    assert events[-1] == ("done", {"doc_id": doc_id, "status": "ready"})


def test_failed_document_stream_and_retry(client) -> None:
    doc_id = upload(client, name="bad scan.pdf").json()["doc_id"]
    events = parse_sse(client.get(f"/api/documents/{doc_id}/events").text)
    last = [d for n, d in events if n == "document"][-1]
    assert last["status"] == "failed" and last["error"] == "simulated parser crash"
    assert events[-1][1]["status"] == "failed"
    r = client.post(f"/api/documents/{doc_id}/retry")
    assert r.status_code == 200 and r.json()["status"] == "queued"
    events = parse_sse(client.get(f"/api/documents/{doc_id}/events").text)
    assert events[-1][1]["status"] == "ready"
    assert client.post(f"/api/documents/{doc_id}/retry").status_code == 409  # not failed


def test_events_for_ready_document_end_immediately(client) -> None:
    events = parse_sse(client.get("/api/documents/d1/events").text)
    assert [n for n, _ in events] == ["document", "done"]


def test_events_report_deletion(lib, tmp_path) -> None:
    lib.docs["slow"] = DocRecord(doc_id="slow", title="t", filename="t.pdf", status="processing")
    with make_client(lib, tmp_path) as c:
        threading.Timer(0.1, lambda: lib.delete("slow")).start()
        r = c.get("/api/documents/slow/events")
    events = parse_sse(r.text)
    assert events[-1] == ("deleted", {"doc_id": "slow"})
    assert ": ping" in r.text  # heartbeat comments while nothing changes


# ============================================================================ pages


def test_page_png_with_cache_headers(client) -> None:
    r = client.get("/api/documents/d1/pages/3.png", params={"zoom": 2, "chunk_id": "d1-c1"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content.endswith(b"d1:3:2.0:d1-c1")
    assert "max-age" in r.headers["cache-control"]
    etag = r.headers["etag"]
    r2 = client.get(
        "/api/documents/d1/pages/3.png",
        params={"zoom": 2, "chunk_id": "d1-c1"},
        headers={"If-None-Match": etag},
    )
    assert r2.status_code == 304 and r2.content == b""
    other = client.get("/api/documents/d1/pages/3.png", params={"zoom": 2})
    assert other.headers["etag"] != etag


def test_page_errors(client) -> None:
    assert client.get("/api/documents/d1/pages/11.png").status_code == 404
    assert client.get("/api/documents/d1/pages/0.png").status_code == 404
    r = client.get("/api/documents/d1/pages/1.png", params={"zoom": 99})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)
    assert client.get("/api/documents/d1/pages/abc.png").status_code == 422


def test_page_highlight_boxes(client) -> None:
    r = client.get("/api/documents/d1/pages/3/highlight", params={"chunk_id": "d1-c1"})
    assert r.status_code == 200 and "max-age" in r.headers["cache-control"]
    boxes = r.json()["boxes"]
    assert boxes and all(len(b) == 4 and all(0 <= v <= 1 for v in b) for b in boxes)


def test_page_highlight_without_chunk_id_is_empty(client) -> None:
    r = client.get("/api/documents/d1/pages/3/highlight")
    assert r.status_code == 200 and r.json() == {"boxes": []}


def test_page_highlight_errors(client) -> None:
    for url in (
        "/api/documents/nope/pages/1/highlight",
        "/api/documents/d1/pages/11/highlight",
        "/api/documents/d1/pages/0/highlight",
    ):
        assert client.get(url, params={"chunk_id": "c"}).status_code == 404
    assert client.get("/api/documents/d1/pages/abc/highlight").status_code == 422
    r = client.get("/api/documents/d1/pages/1/highlight", params={"chunk_id": "x" * 513})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


def test_page_highlight_falls_back_for_libraries_without_it(lib, tmp_path, monkeypatch) -> None:
    monkeypatch.delattr(FakeLibrary, "highlight_boxes")
    with make_client(lib, tmp_path) as c:
        r = c.get("/api/documents/d1/pages/1/highlight", params={"chunk_id": "d1-c1"})
    assert r.status_code == 200 and r.json() == {"boxes": []}


# ============================================================================ asking


def test_ask_json(client, lib) -> None:
    r = client.post(
        "/api/ask",
        json={
            "question": "  How did revenue change?  ",
            "doc_ids": ["d1"],
            "history": [["hi", "hello"]],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Revenue grew 4% [1]." and body["citations"][0]["page"] == 3
    assert body["retrieved"][0]["score"] is None  # NaN sanitised
    assert body["queue_wait_s"] >= 0
    assert lib.asks[-1] == {
        "question": "How did revenue change?",
        "doc_ids": ["d1"],
        "history": [("hi", "hello")],
    }


def test_ask_validation(client, lib) -> None:
    def ask(**body):
        return client.post("/api/ask", json={"question": "q", **body})

    assert ask(question="   ").status_code == 422
    r = ask(question="x" * 2001)
    assert r.status_code == 422 and "2000" in r.json()["detail"]
    assert ask(doc_ids=["missing"]).status_code == 404
    lib.docs["p1"] = DocRecord(doc_id="p1", title="Big book", filename="b.pdf", status="processing")
    r = ask(doc_ids=["p1"])
    assert r.status_code == 409 and "Big book" in r.json()["detail"]
    r = client.post("/api/ask", json={"doc_ids": ["d1"]})
    assert r.status_code == 422 and r.json()["detail"].startswith("Invalid request")
    r = ask(history=[["only one"]])
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)
    assert ask(doc_ids=[]).status_code == 200  # empty list = all ready documents
    assert lib.asks[-1]["doc_ids"] is None


def test_ask_without_ready_documents(lib, tmp_path) -> None:
    lib.docs.clear()
    with make_client(lib, tmp_path) as c:
        r = c.post("/api/ask", json={"question": "anything?"})
    assert r.status_code == 409 and "ready" in r.json()["detail"]


def test_ask_library_error(client) -> None:
    r = client.post("/api/ask", json={"question": "boom"})
    assert r.status_code == 500 and "kaboom" in r.json()["detail"]


def test_ask_stream_emits_queued_steps_answer(client) -> None:
    r = client.post("/api/ask/stream", json={"question": "revenue?", "doc_ids": ["d1"]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [n for n, _ in events if n != "token"]
    assert names == ["queued", "started", "step", "step", "step", "step", "answer"]
    assert events[0][1]["position"] == 0
    assert [d["step"] for n, d in events if n == "step"] == [
        "plan",
        "retrieve",
        "generate",
        "verify",
    ]
    steps = [d for n, d in events if n == "step"]
    assert steps[0]["detail"] == {} and steps[1]["detail"] == {"pages": [3]}
    answer = events[-1][1]
    assert answer["grounded"] is True and "queue_wait_s" in answer
    assert len(answer["trace"]) == 4


def test_ask_stream_validation_and_errors(client) -> None:
    assert client.post("/api/ask/stream", json={"question": ""}).status_code == 422
    assert (
        client.post("/api/ask/stream", json={"question": "q", "doc_ids": ["nope"]}).status_code
        == 404
    )
    events = parse_sse(client.post("/api/ask/stream", json={"question": "boom"}).text)
    assert events[-1][0] == "error"
    assert events[-1][1]["status"] == 500 and "kaboom" in events[-1][1]["detail"]
    refusal = parse_sse(client.post("/api/ask/stream", json={"question": "refuse"}).text)[-1]
    assert refusal[0] == "answer" and refusal[1]["refused"] is True


def test_ask_timeout(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, ask_timeout_s=0.3) as c:
        t0 = time.monotonic()
        r = c.post("/api/ask", json={"question": "slow one"})
        assert r.status_code == 504 and "0.3 seconds" in r.json()["detail"]
        events = parse_sse(c.post("/api/ask/stream", json={"question": "slow two"}).text)
        assert events[-1][0] == "error" and events[-1][1]["code"] == "timeout"
        assert time.monotonic() - t0 < 3
        # Timed-out threads still hold their slots (default max_concurrent_asks=2), so a
        # third question waits in line and then gets a "busy" error...
        r = c.post("/api/ask", json={"question": "fast"})
        assert r.status_code == 503 and r.headers["retry-after"]
        assert c.get("/api/health").json()["server"]["asks_active"] == 2
        # ...until the stuck calls finish and release them.
        lib.release.set()
        for _ in range(100):
            if c.get("/api/health").json()["server"]["asks_active"] == 0:
                break
            time.sleep(0.02)
        assert c.post("/api/ask", json={"question": "fast"}).status_code == 200


def test_semaphore_limits_concurrent_asks(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, max_concurrent_asks=1) as c, ThreadPoolExecutor(4) as pool:
        futures = [
            pool.submit(c.post, "/api/ask", json={"question": f"wait {i}"}) for i in range(4)
        ]
        results = [f.result() for f in futures]
    assert [r.status_code for r in results] == [200] * 4
    assert lib.max_active == 1
    assert max(r.json()["queue_wait_s"] for r in results) > 0.1


def test_stream_reports_queue_position(lib, tmp_path) -> None:
    with make_client(lib, tmp_path, max_concurrent_asks=1) as c, ThreadPoolExecutor(2) as pool:
        blocker = pool.submit(c.post, "/api/ask", json={"question": "slow blocker"})
        for _ in range(200):
            if lib.active:
                break
            time.sleep(0.01)
        threading.Timer(0.2, lib.release.set).start()
        events = parse_sse(c.post("/api/ask/stream", json={"question": "after"}).text)
        assert blocker.result().status_code == 200
    queued = [d["position"] for n, d in events if n == "queued"]
    assert queued[0] == 1 and queued[-1] == 0
    assert events[-1][0] == "answer"
    assert next(d for n, d in events if n == "started")["queue_wait_s"] >= 0.1


# ============================================================================ misc


def test_health(client, lib) -> None:
    body = client.get("/api/health").json()
    assert body["llm"] == "ok" and body["server"]["max_concurrent_asks"] == 2
    assert body["server"]["max_question_chars"] == 2000
    lib.health_error = True
    body = client.get("/api/health").json()
    assert body["llm"] == "unknown" and "ollama down" in body["error"]


def test_frontend_is_served(client) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "<title>PDF Chat</title>" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    for asset, ctype in (("app.js", "javascript"), ("styles.css", "text/css")):
        res = client.get(f"/{asset}")
        assert res.status_code == 200 and ctype in res.headers["content-type"]
    assert "innerHTML" not in client.get("/app.js").text.split("*/", 1)[1]


def test_frontend_accessibility_and_phone_hooks(client) -> None:
    """Static guards (no browser here): the pieces the keyboard/phone behaviour relies on."""
    js, css = client.get("/app.js").text, client.get("/styles.css").text
    assert "type: 'button', className: `chip" in js  # chips are real buttons: Tab, Enter, Space
    assert "/highlight?chunk_id=" in js and "prefers-reduced-motion" in js and "scrollTo" in js
    assert "aria-busy" in js and 'aria-live="polite"' in client.get("/").text
    assert ".page-stage:focus-visible" in css and ".chip:focus-visible" in css
    phone = css.split("@media (max-width: 819px)", 1)[1].split("@media", 1)[0]
    assert 'body[data-view="viewer"] .viewer { display: flex; }' in phone
    assert "#viewer-close { display: none; }" not in phone  # the drawer can be closed on phones
    assert "100dvh" in phone  # composer stays within the visible area


def test_unknown_api_route_is_json_404(client) -> None:
    r = client.get("/api/nothing/here")
    assert r.status_code == 404 and r.json()["detail"].startswith("No such API endpoint")


# ============================================================================ token streaming


def test_stream_sequence_queued_started_tokens_answer(client) -> None:
    r = client.post("/api/ask/stream", json={"question": "revenue?", "doc_ids": ["d1"]})
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[:2] == ["queued", "started"] and names[-1] == "answer"
    assert names.count("token") >= 2  # the reset marker plus at least one batch of text
    assert names.index("token") < names.index("answer")
    # tokens belong to the generation, so they arrive after retrieval and before verification
    first_token = names.index("token")
    steps_before = [d["step"] for n, d in events[:first_token] if n == "step"]
    assert steps_before == ["plan", "retrieve"]
    tokens = [d for n, d in events if n == "token"]
    assert tokens[0] == {"text": "", "reset": True}
    assert all(set(t) == {"text", "reset"} for t in tokens)
    streamed = "".join(t["text"] for t in tokens)
    assert streamed == FakeLibrary.ANSWER == events[-1][1]["answer"]


def test_stream_batches_tiny_tokens(client) -> None:
    events = parse_sse(client.post("/api/ask/stream", json={"question": "revenue?"}).text)
    pieces = [d for n, d in events if n == "token" and not d["reset"]]
    # FakeLibrary emits 5 pieces of 4 characters back to back; they must be merged
    assert 1 <= len(pieces) < 5
    assert "".join(p["text"] for p in pieces) == FakeLibrary.ANSWER


def test_stream_final_answer_equals_plain_ask(client) -> None:
    body = {"question": "revenue?", "doc_ids": ["d1"]}
    plain = client.post("/api/ask", json=body).json()
    events = parse_sse(client.post("/api/ask/stream", json=body).text)
    final = events[-1][1]
    for key in ("answer", "citations", "refused", "grounded", "cited_pages", "trace"):
        assert final[key] == plain[key], key


def test_stream_refusal_has_no_tokens(client) -> None:
    events = parse_sse(client.post("/api/ask/stream", json={"question": "refuse"}).text)
    assert not any(n == "token" for n, _ in events) and events[-1][1]["refused"] is True


def test_library_without_on_token_still_streams_steps(lib, tmp_path) -> None:
    """A LibraryProtocol written before token streaming (no ``on_token``) keeps working."""

    class Legacy(FakeLibrary):
        def ask(self, question, doc_ids, history, on_step=None) -> Answer:
            return super().ask(question, doc_ids, history, on_step)

    old = Legacy(lib.store)
    old.add_ready("d1")
    with make_client(old, tmp_path) as c:
        events = parse_sse(c.post("/api/ask/stream", json={"question": "revenue?"}).text)
    assert [n for n, _ in events if n == "token"] == [] and events[-1][0] == "answer"


def test_real_library_stream_matches_plain_ask(tmp_path) -> None:
    """End to end with the real Library + agent + scripted LLM: streamed == non-streamed."""
    pytest.importorskip("pymupdf")
    pytest.importorskip("reportlab")
    from local_agentic_rag.library import Library
    from tests.helpers import make_text_pdf
    from tests.test_library import _pipe, _wait

    src = tmp_path / "src.pdf"
    make_text_pdf(src, n_pages=30)
    library = Library(tmp_path / "lib", pipeline=_pipe())
    try:
        rec = library.submit(str(src), "Annual Report 2025.pdf")
        _wait(library, rec.doc_id)
        with TestClient(
            create_app(library, upload_dir=str(tmp_path / "in"), heartbeat_s=0.05)
        ) as c:
            body = {"question": "Where is uniqueterm3?", "doc_ids": [rec.doc_id]}
            plain = c.post("/api/ask", json=body).json()
            events = parse_sse(c.post("/api/ask/stream", json=body).text)
        names = [n for n, _ in events]
        assert names[:2] == ["queued", "started"] and names[-1] == "answer"
        assert "token" in names and names.index("token") < names.index("answer")
        final = events[-1][1]
        assert final["answer"] == plain["answer"] and final["cited_pages"] == plain["cited_pages"]
        generated = "".join(d["text"] for n, d in events if n == "token")
        assert generated.strip() == final["answer"] == plain["answer"]
        # only the free-text generation streams: plan/grade/verify JSON never reaches the UI
        assert "{" not in generated
    finally:
        library.close()


def _asgi_call(app, method: str, path: str, body: bytes = b"", *, disconnect_on: bytes = b""):
    """Drive the ASGI app by hand (TestClient cannot hang up mid-response).

    Returns the list of body chunks seen.  When ``disconnect_on`` is set the client
    "closes the connection" as soon as a chunk containing it is sent.
    """
    import asyncio

    async def run() -> list[bytes]:
        chunks: list[bytes] = []
        hung_up = asyncio.Event()
        sent_request = False

        async def receive():
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": body, "more_body": False}
            await hung_up.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                chunks.append(chunk)
                if disconnect_on and disconnect_on in chunk:
                    hung_up.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 5000),
            "server": ("test", 80),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=10)
        return chunks

    return run


def test_client_disconnect_mid_stream_releases_gate_slot(lib, tmp_path) -> None:
    import asyncio

    app = create_app(
        lib, upload_dir=str(tmp_path / "incoming"), max_concurrent_asks=1, heartbeat_s=0.05
    )
    ask = json.dumps({"question": "slow hang up", "doc_ids": ["d1"]}).encode()

    async def scenario() -> tuple[int, int, bool]:
        # 1. the browser hangs up as soon as the answer starts (the worker is still running)
        await _asgi_call(app, "POST", "/api/ask/stream", ask, disconnect_on=b"event: started")()
        during = json.loads(b"".join(await _asgi_call(app, "GET", "/api/health")()))
        # 2. the thread finishes later: only then is the slot free again
        lib.release.set()
        for _ in range(200):
            health = json.loads(b"".join(await _asgi_call(app, "GET", "/api/health")()))
            if health["server"]["asks_active"] == 0:
                break
            await asyncio.sleep(0.02)
        # 3. a new user is served immediately
        chunks = await _asgi_call(app, "POST", "/api/ask/stream", ask)()
        return (
            during["server"]["asks_active"],
            health["server"]["asks_active"],
            (b"event: answer" in b"".join(chunks)),
        )

    held, after, served = asyncio.run(scenario())
    assert held == 1  # a disconnect never frees the GPU slot while the thread still runs
    assert after == 0  # ...and it is released when the thread ends
    assert served is True
    assert lib.max_active == 1


# ============================================================================ ask while processing


def test_document_records_carry_the_askable_fields(lib, client) -> None:
    lib.add_processing("p1", askable=True, pages_indexed=12)
    docs = {d["doc_id"]: d for d in client.get("/api/documents").json()["documents"]}
    assert docs["p1"]["askable"] is True and docs["p1"]["pages_indexed"] == 12
    assert docs["d1"]["askable"] is False
    assert client.get("/api/documents/p1").json()["askable"] is True


def test_processing_document_with_a_partial_index_can_be_asked(lib, client) -> None:
    lib.add_processing("p1", askable=True, pages_indexed=12, pages=100)
    r = client.post("/api/ask", json={"question": "What grew?", "doc_ids": ["p1"]})
    assert r.status_code == 200, r.text
    assert r.json()["partial"] == {"doc_id": "p1", "pages_indexed": 12, "pages_total": 100}
    assert lib.asks[-1]["doc_ids"] == ["p1"]

    r = client.post("/api/ask/stream", json={"question": "What grew?", "doc_ids": ["p1"]})
    assert r.status_code == 200, r.text
    answer = [data for name, data in parse_sse(r.text) if name == "answer"][0]
    assert answer["partial"] == {"doc_id": "p1", "pages_indexed": 12, "pages_total": 100}
    assert any(name == "token" for name, _ in parse_sse(r.text))  # streaming still works


def test_mixed_selection_of_ready_and_partial_documents(lib, client) -> None:
    lib.add_processing("p1", askable=True, pages_indexed=12)
    r = client.post("/api/ask", json={"question": "What grew?", "doc_ids": ["p1", "d1"]})
    assert r.status_code == 200 and r.json()["partial"]["pages_indexed"] == 12


def test_ready_answers_have_no_partial_field(client) -> None:
    r = client.post("/api/ask", json={"question": "What grew?", "doc_ids": ["d1"]})
    assert r.status_code == 200 and "partial" not in r.json()


def test_processing_document_without_a_checkpoint_is_rejected(lib, client) -> None:
    lib.add_processing("p0", askable=False)
    for path in ("/api/ask", "/api/ask/stream"):
        r = client.post(path, json={"question": "What grew?", "doc_ids": ["p0"]})
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == "'Processing p0' is not ready yet (processing)."
    assert lib.asks == []


def test_default_selection_stays_ready_only(lib, client) -> None:
    lib.add_processing("p1", askable=True, pages_indexed=12)
    r = client.post("/api/ask", json={"question": "What grew?"})
    assert r.status_code == 200 and lib.asks[-1]["doc_ids"] is None  # library picks ready docs
    del lib.docs["d1"]  # only a partially indexed document is left: nothing is "ready"
    r = client.post("/api/ask", json={"question": "What grew?"})
    assert r.status_code == 409 and "No document is ready yet" in r.json()["detail"]


def test_step_event_never_overtakes_held_back_tokens() -> None:
    """Token text is batched for 50 ms; a step that follows it must queue behind that text."""
    import asyncio

    from local_agentic_rag.server.asks import AskGate, AskJob

    class Lib:
        def ask(self, question, doc_ids, history, on_step=None, on_token=None) -> Answer:
            on_token({"text": "Hello"})  # idle: goes out at once
            on_token({"text": " world"})  # inside the 50 ms window: held back
            on_step(TraceStep(step="verify", detail={}, seconds=0.0))
            return Answer(question=question, answer="Hello world", refused=False)

    async def run() -> list[tuple[str, Any]]:
        with ThreadPoolExecutor(1) as pool:
            job = AskJob(Lib(), pool, AskGate(1), "q", None, [], stream_steps=True)
            await job.future
            await asyncio.sleep(0.2)  # let the batching timer fire if it is still armed
            out = []
            while not job.events.empty():
                out.append(job.events.get_nowait())
            return out

    events = asyncio.run(run())
    kinds = [(n, d.get("text", d.get("step"))) for n, d in events]
    assert kinds == [("token", "Hello"), ("token", " world"), ("step", "verify")]


class _LanguageLibrary(FakeLibrary):
    """A library whose ``ask`` takes the "Answer in" choice, as the real one does."""

    def ask(self, question, doc_ids, history, on_step=None, on_token=None, answer_language=None):
        self.answer_languages = getattr(self, "answer_languages", []) + [answer_language]
        return super().ask(question, doc_ids, history, on_step, on_token)


def test_ask_passes_the_answer_language_choice(tmp_path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    lib = _LanguageLibrary(store)
    lib.add_ready("d1", "Annual report")
    with make_client(lib, tmp_path) as c:
        assert (
            c.post("/api/ask", json={"question": "q", "answer_language": "Urdu"}).status_code == 200
        )
        assert (
            c.post("/api/ask", json={"question": "q", "answer_language": "auto"}).status_code == 200
        )
        assert c.post("/api/ask", json={"question": "q"}).status_code == 200
        r = c.post("/api/ask", json={"question": "q", "answer_language": "Klingon"})
        assert r.status_code == 422 and "Klingon" in r.json()["detail"]
    lib.release.set()
    assert lib.answer_languages == ["Urdu", None, None]


def test_libraries_without_the_choice_keep_working(client, lib) -> None:
    r = client.post("/api/ask", json={"question": "q", "answer_language": "Hindi"})
    assert r.status_code == 200 and lib.asks[-1]["question"] == "q"
