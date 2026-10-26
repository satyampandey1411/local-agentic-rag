"""
HTTP API + web UI for the document library.

    from local_agentic_rag.server import create_app
    app = create_app(library)                    # any LibraryProtocol implementation
    uvicorn.run(app, host="127.0.0.1", port=8000)  # ONE worker process: the library owns the GPU

Request flow::

    browser ──upload (multipart, streamed)──► POST /api/documents ──► library.submit()  (returns at once)
            ◄──SSE progress every change──── GET  /api/documents/{id}/events  (polls library.get)
            ──question──────────────────────► POST /api/ask/stream
                                                 │  AskGate (FIFO, max_concurrent_asks)  ── "queued" events
                                                 │  worker: library.ask(on_step=…, on_token=…) ── "step" / "token" events
            ◄──"answer" / "error" event──────────┘
            ──page image────────────────────► GET  /api/documents/{id}/pages/{n}.png
            ──highlight boxes (scroll target)► GET  /api/documents/{id}/pages/{n}/highlight?chunk_id=

Every blocking library call runs off the event loop (sync endpoints run in
Starlette's thread pool; ``ask`` runs in its own pool), so one slow request never
freezes the server for other users.  Errors are always ``{"detail": "<message>"}``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import hmac
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from local_agentic_rag.library_types import DocRecord, LibraryProtocol
from local_agentic_rag.server import sse
from local_agentic_rag.server.asks import AskGate, AskJob
from local_agentic_rag.server.uploads import receive_pdf, sweep_stale_uploads

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).with_name("static")
MAX_QUESTION_CHARS = 2000
MAX_HISTORY_TURNS = 20
MAX_HISTORY_CHARS = 8000  # per message; older/longer context is truncated, not rejected
MAX_DOC_IDS = 500
TERMINAL = ("ready", "failed")

SESSION_COOKIE = "local_rag_session"
_PUBLIC_API = ("/api/health",)  # reachable without a token (liveness probes)

_CSP = (
    "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'"
)


class JSONBody(Response):
    """JSON response that tolerates numpy values / NaN in library output."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return sse.dumps(content).encode("utf-8")


class AskRequest(BaseModel):
    question: str = Field(..., description=f"1-{MAX_QUESTION_CHARS} characters")
    doc_ids: list[str] | None = Field(None, description="None = every ready document")
    history: list[tuple[str, str]] = Field(default_factory=list, description="(user, assistant)")


class _SecurityHeaders:
    """Pure-ASGI middleware (BaseHTTPMiddleware would interfere with streaming)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_api = scope["path"].startswith("/api/")

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"referrer-policy", b"no-referrer"))
                if not is_api:
                    headers.append((b"content-security-policy", _CSP.encode()))
                    if scope["path"] in ("/", "/index.html"):
                        headers.append((b"cache-control", b"no-cache"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


def _session_value(token: str) -> str:
    """What the session cookie holds: a keyed hash of the token, never the token itself."""
    return hmac.new(token.encode(), b"local-rag-session", hashlib.sha256).hexdigest()


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8", "replace"), b.encode("utf-8", "replace"))


class _TokenAuth:
    """Pure-ASGI gate: with a token configured, /api/* needs a Bearer header or the session cookie.

    Static files stay public so the page can load and show its login form;
    ``/api/health`` stays public for liveness probes; ``/api/session`` does its own check.
    """

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token
        self.cookie_value = _session_value(token)

    def _authorised(self, scope: dict) -> bool:
        cookies: dict[str, str] = {}
        for name, value in scope["headers"]:
            if name == b"authorization":
                scheme, _, credential = value.decode("latin-1").partition(" ")
                if scheme.lower() == "bearer" and _same(credential.strip(), self.token):
                    return True
            elif name == b"cookie":
                jar: SimpleCookie = SimpleCookie()
                try:
                    jar.load(value.decode("latin-1"))
                except Exception:  # noqa: BLE001 - malformed Cookie header = no cookie
                    continue
                cookies.update({k: m.value for k, m in jar.items()})
        return _same(cookies.get(SESSION_COOKIE, ""), self.cookie_value)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        path = scope.get("path", "")  # lifespan scopes have none
        if (
            scope["type"] != "http"
            or not path.startswith("/api/")
            or path in _PUBLIC_API
            or path == "/api/session"
            or self._authorised(scope)
        ):
            await self.app(scope, receive, send)
            return
        response = JSONResponse(
            {"detail": "Authentication required. Send the access token."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)


class SessionRequest(BaseModel):
    token: str = Field(..., max_length=1024)


class UploadLimiter:
    """Per-client-IP upload limits: one upload at a time, optional daily byte quota.

    In memory only: counters reset when the server restarts.  The "day" is the UTC
    calendar day.  Event-loop only (no locks needed: no ``await`` between check and update).
    """

    def __init__(self, quota_bytes: int | None) -> None:
        self.quota_bytes = quota_bytes if quota_bytes and quota_bytes > 0 else None
        self.in_flight: set[str] = set()
        self._used: dict[str, tuple[str, int]] = {}  # ip -> (utc day, bytes)

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def used(self, ip: str) -> int:
        day, n = self._used.get(ip, ("", 0))
        return n if day == self._today() else 0

    def begin(self, ip: str, declared_bytes: int | None) -> None:
        """Raise 429 / 413 if this client may not start an upload now; else mark it in flight."""
        if ip in self.in_flight:
            raise HTTPException(
                429,
                "Another upload from your address is still in progress. Wait for it to finish.",
                headers={"Retry-After": "10"},
            )
        if self.quota_bytes is not None:
            used = self.used(ip)
            if used >= self.quota_bytes or used + (declared_bytes or 0) > self.quota_bytes:
                raise HTTPException(413, self.quota_message())
        self.in_flight.add(ip)

    def quota_message(self) -> str:
        return (
            "Daily upload quota exceeded "
            f"({(self.quota_bytes or 0) // (1024 * 1024)} MB per address per day). "
            "Try again tomorrow (UTC)."
        )

    def remaining(self, ip: str) -> int | None:
        """Bytes this client may still upload today (None = no quota)."""
        if self.quota_bytes is None:
            return None
        return max(0, self.quota_bytes - self.used(ip))

    def finish(self, ip: str, stored_bytes: int = 0) -> None:
        self.in_flight.discard(ip)
        if stored_bytes:
            today = self._today()
            self._used = {k: v for k, v in self._used.items() if v[0] == today}
            self._used[ip] = (today, self.used(ip) + stored_bytes)


def create_app(
    library: LibraryProtocol,
    *,
    max_upload_mb: int = 2048,
    ask_timeout_s: float = 300,
    max_concurrent_asks: int = 2,
    upload_dir: str | None = None,
    events_poll_s: float = 1.0,
    heartbeat_s: float = 15.0,
    token: str | None = None,
    upload_quota_bytes: int | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``upload_dir`` is where uploads are streamed before ``library.submit`` moves
    them; put it on the same filesystem as the library so the move is a rename
    (default: the system temp dir).  ``events_poll_s`` / ``heartbeat_s`` are
    exposed for tests.  ``token`` (None = no auth) gates every ``/api`` route but
    ``/api/health``; ``upload_quota_bytes`` is the per-IP daily upload quota (None = off).
    """
    token = token or None
    limiter = UploadLimiter(upload_quota_bytes)
    max_bytes = int(max_upload_mb) * 1024 * 1024
    if upload_dir:
        os.makedirs(upload_dir, exist_ok=True)
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_concurrent_asks, thread_name_prefix="ask"
    )
    gates: dict[int, AskGate] = {}

    def gate() -> AskGate:
        # asyncio objects belong to one event loop; normally there is exactly one.
        loop_id = id(asyncio.get_running_loop())
        if loop_id not in gates:
            gates.clear()
            gates[loop_id] = AskGate(max_concurrent_asks)
        return gates[loop_id]

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if upload_dir:
            removed = sweep_stale_uploads(upload_dir)
            if removed:
                logger.info("removed %d stale partial uploads from %s", removed, upload_dir)
        try:
            yield
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            close = getattr(library, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    await run_in_threadpool(close)

    app = FastAPI(
        title="Local Agentic RAG",
        description="Chat with your PDFs, locally. Answers cite the pages they come from.",
        version="0.1.0",
        lifespan=lifespan,
        # The API docs are served outside /api, so the token check would not cover them.
        docs_url=None if token else "/docs",
        redoc_url=None if token else "/redoc",
        openapi_url=None if token else "/openapi.json",
    )
    if token:
        app.add_middleware(_TokenAuth, token=token)  # added first = inside the header middleware
    app.add_middleware(_SecurityHeaders)
    app.state.library = library

    # ------------------------------------------------------------------ errors

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_req: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc.detail)}, status_code=exc.status_code, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors()[:3]:
            loc = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path"))
            parts.append(f"{loc}: {err.get('msg', 'invalid')}" if loc else err.get("msg", ""))
        return JSONResponse({"detail": "Invalid request - " + "; ".join(parts)}, status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_req: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "Internal server error."}, status_code=500)

    def _record(doc_id: str) -> DocRecord:
        if len(doc_id) > 256:
            raise HTTPException(404, "Document not found.")
        try:
            return library.get(doc_id)
        except KeyError:
            raise HTTPException(404, "Document not found.") from None

    # ------------------------------------------------------------------ documents

    @app.post("/api/documents", status_code=201)
    async def upload_document(request: Request) -> JSONBody:
        ip = request.client.host if request.client else "unknown"
        declared = request.headers.get("content-length", "")
        limiter.begin(ip, int(declared) if declared.isdigit() else None)
        stored = 0
        try:
            # The quota also caps what is read, so a chunked upload cannot exceed it.
            limit, detail = max_bytes, None
            left = limiter.remaining(ip)
            if left is not None and left < max_bytes:
                limit, detail = left, limiter.quota_message()
            received = await receive_pdf(request, limit, upload_dir, detail)
            try:
                submitted_at = time.time()
                record = await run_in_threadpool(library.submit, received.path, received.filename)
                if record.created_at >= submitted_at:  # a duplicate returns the older record
                    stored = received.size
            except ValueError as exc:
                raise HTTPException(400, str(exc) or "The PDF could not be added.") from None
            except BaseException:
                Path(received.path).unlink(missing_ok=True)
                raise
        finally:
            limiter.finish(ip, stored)
        return JSONBody(record.to_dict(), status_code=201)

    @app.get("/api/documents")
    def list_documents() -> JSONBody:
        docs = sorted(library.list_documents(), key=lambda d: d.created_at, reverse=True)
        return JSONBody({"documents": [d.to_dict() for d in docs]})

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str) -> JSONBody:
        return JSONBody(_record(doc_id).to_dict())

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> Response:
        try:
            library.delete(doc_id)
        except KeyError:
            raise HTTPException(404, "Document not found.") from None
        return Response(status_code=204)

    @app.post("/api/documents/{doc_id}/retry")
    def retry_document(doc_id: str) -> JSONBody:
        try:
            record = library.retry(doc_id)
        except KeyError:
            raise HTTPException(404, "Document not found.") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc) or "Only failed documents can be retried.") from None
        return JSONBody(record.to_dict())

    @app.get("/api/documents/{doc_id}/events")
    async def document_events(doc_id: str) -> StreamingResponse:
        first = await run_in_threadpool(_record, doc_id)  # 404 before the stream starts

        async def stream() -> AsyncIterator[str]:
            yield sse.PADDING + "retry: 3000\n\n"
            last: dict[str, Any] | None = None
            record: DocRecord | None = first
            last_sent = time.monotonic()
            while True:
                if record is not None:
                    current = record.to_dict()
                    if current != last:
                        yield sse.event("document", current)
                        last, last_sent = current, time.monotonic()
                    if record.status in TERMINAL:
                        yield sse.event("done", {"doc_id": doc_id, "status": record.status})
                        return
                if time.monotonic() - last_sent >= heartbeat_s:
                    yield sse.comment()
                    last_sent = time.monotonic()
                await asyncio.sleep(events_poll_s)
                try:
                    record = await run_in_threadpool(library.get, doc_id)
                except KeyError:
                    yield sse.event("deleted", {"doc_id": doc_id})
                    return
                except Exception:  # keep the stream alive across a transient library error
                    logger.exception("library.get(%s) failed during event stream", doc_id)
                    record = None

        return StreamingResponse(stream(), media_type="text/event-stream", headers=sse.SSE_HEADERS)

    @app.get("/api/documents/{doc_id}/pages/{page}.png")
    def page_image(
        request: Request,
        doc_id: str,
        page: int,
        zoom: float = Query(1.5, ge=0.25, le=4.0),
        chunk_id: str | None = Query(None, max_length=512),
    ) -> Response:
        record = _record(doc_id)
        if page < 1 or (record.num_pages and page > record.num_pages):
            raise HTTPException(404, f"Page {page} does not exist in this document.")
        zoom = round(zoom * 4) / 4  # quantise so caches are reused
        key = f"{doc_id}|{record.created_at}|{page}|{zoom}|{chunk_id or ''}"
        etag = '"' + hashlib.sha1(key.encode()).hexdigest()[:20] + '"'
        headers = {"Cache-Control": "private, max-age=86400", "ETag": etag}
        if etag in request.headers.get("if-none-match", ""):
            return Response(status_code=304, headers=headers)
        try:
            png = library.render_page(doc_id, page, zoom=zoom, chunk_id=chunk_id)
        except KeyError:
            raise HTTPException(404, "Document not found.") from None
        except IndexError:
            raise HTTPException(404, f"Page {page} does not exist in this document.") from None
        return Response(png, media_type="image/png", headers=headers)

    @app.get("/api/documents/{doc_id}/pages/{page}/highlight")
    def page_highlight(
        doc_id: str,
        page: int,
        chunk_id: str | None = Query(None, max_length=512),
    ) -> Response:
        """Boxes (0-1 page fractions) of the highlight ``pages/{page}.png?chunk_id=`` paints."""
        record = _record(doc_id)
        if page < 1 or (record.num_pages and page > record.num_pages):
            raise HTTPException(404, f"Page {page} does not exist in this document.")
        boxes: list[list[float]] = []
        finder = getattr(library, "highlight_boxes", None)  # optional on older libraries
        if chunk_id and finder is not None:
            try:
                boxes = finder(doc_id, page, chunk_id)
            except KeyError:
                raise HTTPException(404, "Document not found.") from None
            except IndexError:
                raise HTTPException(404, f"Page {page} does not exist in this document.") from None
        headers = {"Cache-Control": "private, max-age=86400"}
        return JSONResponse({"boxes": boxes}, headers=headers)

    # ------------------------------------------------------------------ asking

    def _validate(body: AskRequest) -> tuple[str, list[str] | None, list[tuple[str, str]]]:
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "Please type a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise HTTPException(
                422,
                f"Questions are limited to {MAX_QUESTION_CHARS} characters "
                f"(this one has {len(question)}).",
            )
        doc_ids = body.doc_ids
        if doc_ids is not None:
            doc_ids = list(dict.fromkeys(d for d in doc_ids if d))
            if not doc_ids:
                doc_ids = None
            elif len(doc_ids) > MAX_DOC_IDS:
                raise HTTPException(422, "Too many documents selected.")
        history = [
            (u[-MAX_HISTORY_CHARS:], a[-MAX_HISTORY_CHARS:])
            for u, a in body.history[-MAX_HISTORY_TURNS:]
        ]
        return question, doc_ids, history

    def _check_docs(doc_ids: list[str] | None) -> None:
        """Clear 404/409 messages before the question waits in line."""
        if doc_ids is None:
            if not any(d.status == "ready" for d in library.list_documents()):
                raise HTTPException(
                    409,
                    "No document is ready yet. Upload a PDF and wait for it to finish processing.",
                )
            return
        for doc_id in doc_ids:
            record = _record(doc_id)
            # A processing document with a checkpointed prefix can be asked (partial answer).
            if record.status == "ready" or (record.status == "processing" and record.askable):
                continue
            raise HTTPException(409, f"'{record.title}' is not ready yet ({record.status}).")

    def _ask_error(exc: BaseException) -> tuple[int, str]:
        if isinstance(exc, ValueError):
            return 409, str(exc) or "A selected document is not ready."
        if isinstance(exc, KeyError):
            return 404, "A selected document no longer exists."
        logger.error("library.ask failed", exc_info=exc)
        return 500, f"Answering failed: {type(exc).__name__}: {exc}"

    def _timeout_message() -> str:
        return (
            f"No answer after {ask_timeout_s:g} seconds - the model may be overloaded. "
            "Please try again."
        )

    def _busy_message() -> str:
        return "The server is busy with other questions. Please try again in a minute."

    @app.post("/api/ask")
    async def ask(body: AskRequest) -> JSONBody:
        question, doc_ids, history = _validate(body)
        await run_in_threadpool(_check_docs, doc_ids)
        g = gate()
        t0 = time.monotonic()
        ticket = g.enqueue()
        try:
            await asyncio.wait_for(asyncio.shield(ticket), timeout=ask_timeout_s)
        except asyncio.TimeoutError:
            g.abandon(ticket)
            raise HTTPException(503, _busy_message(), headers={"Retry-After": "30"}) from None
        except BaseException:
            g.abandon(ticket)
            raise
        queue_wait = time.monotonic() - t0
        try:
            job = AskJob(library, executor, g, question, doc_ids, history, stream_steps=False)
        except BaseException:
            g.release()
            raise
        try:
            answer = await asyncio.wait_for(asyncio.shield(job.future), timeout=ask_timeout_s)
        except asyncio.TimeoutError:
            raise HTTPException(504, _timeout_message()) from None
        except Exception as exc:  # noqa: BLE001
            status, detail = _ask_error(exc)
            raise HTTPException(status, detail) from None
        return JSONBody({**answer.to_dict(), "queue_wait_s": round(queue_wait, 3)})

    @app.post("/api/ask/stream")
    async def ask_stream(body: AskRequest) -> StreamingResponse:
        question, doc_ids, history = _validate(body)
        await run_in_threadpool(_check_docs, doc_ids)

        async def stream() -> AsyncIterator[str]:
            g = gate()
            yield sse.PADDING
            t0 = time.monotonic()
            ticket = g.enqueue()
            try:
                last_pos = -1
                last_sent = time.monotonic()
                while True:
                    pos = g.position(ticket)
                    if pos != last_pos:
                        yield sse.event(
                            "queued", {"position": pos, "active": g.active, "waiting": g.waiting}
                        )
                        last_pos, last_sent = pos, time.monotonic()
                    if ticket.done():
                        break
                    if time.monotonic() - t0 > ask_timeout_s:
                        g.abandon(ticket)
                        yield sse.event(
                            "error", {"status": 503, "code": "busy", "detail": _busy_message()}
                        )
                        return
                    if time.monotonic() - last_sent >= heartbeat_s:
                        yield sse.comment()
                        last_sent = time.monotonic()
                    await asyncio.wait({ticket}, timeout=min(0.25, heartbeat_s))
            except BaseException:
                g.abandon(ticket)
                raise

            queue_wait = time.monotonic() - t0
            try:
                job = AskJob(library, executor, g, question, doc_ids, history, stream_steps=True)
            except BaseException:
                g.release()
                raise
            yield sse.event("started", {"queue_wait_s": round(queue_wait, 3)})
            deadline = time.monotonic() + ask_timeout_s
            getter: asyncio.Future[tuple[str, dict[str, Any]]] | None = None
            try:
                while not job.future.done():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # The worker thread cannot be killed; it keeps its gate slot
                        # until library.ask returns (see asks.py).
                        yield sse.event(
                            "error",
                            {"status": 504, "code": "timeout", "detail": _timeout_message()},
                        )
                        return
                    if getter is None:
                        getter = asyncio.ensure_future(job.events.get())
                    done, _ = await asyncio.wait(
                        {getter, job.future},
                        timeout=min(remaining, heartbeat_s),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if getter in done:
                        name, payload = getter.result()
                        yield sse.event(name, payload)
                        getter = None
                    elif not done:
                        yield sse.comment()
            finally:
                if getter is not None and not getter.done():
                    getter.cancel()
            if getter is not None and getter.done() and not getter.cancelled():
                name, payload = getter.result()  # fetched but not yet sent
                yield sse.event(name, payload)
            job.flush_tokens()
            while not job.events.empty():  # events scheduled just before the thread returned
                name, payload = job.events.get_nowait()
                yield sse.event(name, payload)
            exc = job.future.exception()
            if exc is not None:
                status, detail = _ask_error(exc)
                yield sse.event("error", {"status": status, "code": "failed", "detail": detail})
                return
            answer = job.future.result()
            yield sse.event("answer", {**answer.to_dict(), "queue_wait_s": round(queue_wait, 3)})

        return StreamingResponse(stream(), media_type="text/event-stream", headers=sse.SSE_HEADERS)

    # ------------------------------------------------------------------ health

    @app.get("/api/health")
    def health() -> JSONBody:
        try:
            info: dict[str, Any] = dict(library.health())
        except Exception as exc:  # noqa: BLE001 - health must answer even when degraded
            logger.exception("library.health failed")
            info = {"llm": "unknown", "error": f"{type(exc).__name__}: {exc}"}
        g = next(iter(gates.values()), None)
        info["server"] = {
            "asks_active": g.active if g else 0,
            "asks_waiting": g.waiting if g else 0,
            "max_concurrent_asks": max_concurrent_asks,
            "ask_timeout_s": ask_timeout_s,
            "max_upload_mb": max_upload_mb,
            "max_question_chars": MAX_QUESTION_CHARS,
        }
        return JSONBody(info)

    # ------------------------------------------------------------------ session (token mode)

    if token:
        cookie_value = _session_value(token)

        @app.post("/api/session")
        async def login(body: SessionRequest, request: Request) -> Response:
            if not _same(body.token, token):
                raise HTTPException(
                    401, "Wrong access token.", headers={"WWW-Authenticate": "Bearer"}
                )
            secure = (
                request.url.scheme == "https"
                or request.headers.get("x-forwarded-proto", "").lower() == "https"
            )
            response = Response(status_code=204)
            # Session cookie (no Max-Age): gone when the browser closes.
            response.set_cookie(
                SESSION_COOKIE, cookie_value, httponly=True, samesite="strict", secure=secure
            )
            return response

        @app.delete("/api/session", status_code=204)
        async def logout() -> Response:
            response = Response(status_code=204)
            response.delete_cookie(SESSION_COOKIE, httponly=True, samesite="strict")
            return response

    # ------------------------------------------------------------------ frontend

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def api_not_found(rest: str) -> None:
        raise HTTPException(404, f"No such API endpoint: /api/{rest}")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
