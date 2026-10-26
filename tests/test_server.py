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

    def ask(self, question, doc_ids, history, on_step=None) -> Answer:
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
            ans.answer = "Revenue grew 4% [1]."
            ans.citations = [
                Citation(
                    marker=1, chunk_id=chunk.chunk_id, doc_id=doc, page=3, snippet="Revenue grew"
                )
            ]
            ans.retrieved = [
                ScoredChunk(chunk=chunk, score=float("nan"))
            ]  # NaN must not break JSON
            ans.grounded = True
            return ans
        finally:
            with self.lock:
                self.active -= 1

    def render_page(self, doc_id, page, zoom=1.5, chunk_id=None) -> bytes:
        rec = self.get(doc_id)
        if not 1 <= page <= rec.num_pages:
            raise IndexError(page)
        return b"\x89PNG\r\n\x1a\n" + f"{doc_id}:{page}:{zoom}:{chunk_id}".encode()

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
    names = [n for n, _ in events]
    assert names == ["queued", "started", "step", "step", "step", "step", "answer"]
    assert events[0][1]["position"] == 0
    assert [d["step"] for n, d in events if n == "step"] == [
        "plan",
        "retrieve",
        "generate",
        "verify",
    ]
    assert events[2][1]["detail"] == {} and events[3][1]["detail"] == {"pages": [3]}
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


def test_unknown_api_route_is_json_404(client) -> None:
    r = client.get("/api/nothing/here")
    assert r.status_code == 404 and r.json()["detail"].startswith("No such API endpoint")
