"""
Streaming PDF upload: request body -> temp file, without ever holding the file in RAM.

FastAPI's ``UploadFile`` would first spool the whole multipart body to its own
temporary file and only then call the endpoint, so a 2 GB non-PDF would be fully
received before it could be rejected, and every upload would be written to disk
twice.  Here the multipart body is parsed incrementally with ``python-multipart``
as it arrives, so the server can

* reject an oversize upload from ``Content-Length`` before reading anything,
* reject a non-PDF after its first KB (``%PDF-`` magic bytes),
* stop reading the moment the size limit is crossed (HTTP 413),
* write the file part straight to its final temp file in 1 MiB batches, with the
  blocking ``write`` calls run in a worker thread so the event loop stays responsive.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import anyio
from fastapi import HTTPException, Request
from python_multipart.exceptions import ParseError
from python_multipart.multipart import MultipartParser, parse_options_header

PDF_MAGIC = b"%PDF-"
#: The PDF spec puts the header at byte 0, but readers (Acrobat included) accept
#: it anywhere in the first 1024 bytes; some generators prepend junk.
MAGIC_WINDOW = 1024
_FLUSH_BYTES = 1 << 20
_MULTIPART_OVERHEAD = 64 * 1024  # boundaries + part headers + small form fields
_MAX_FIELD_BYTES = 64 * 1024
TEMP_PREFIX = "upload-"
TEMP_SUFFIX = ".part"


@dataclass
class ReceivedFile:
    path: str
    filename: str
    size: int


def safe_filename(raw: str | None) -> str:
    """Basename only, no control characters, bounded length, always ending in .pdf."""
    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip(" .")
    name = re.sub(r"\s+", " ", name)[:200]
    if not name:
        name = "document.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


class _Receiver:
    """python-multipart callbacks that route the ``file`` part into a temp file."""

    def __init__(self, max_bytes: int, upload_dir: str | None) -> None:
        self.max_bytes = max_bytes
        self.upload_dir = upload_dir
        self.fh: BinaryIO | None = None
        self.path: str | None = None
        self.filename: str | None = None
        self.size = 0
        self.head = bytearray()  # first MAGIC_WINDOW bytes, for the magic check
        self.magic_ok = False
        self.pending = bytearray()  # bytes waiting to be flushed to disk
        self.files_seen = 0
        self.field_bytes = 0
        self.error: HTTPException | None = None
        self.ended = False  # saw the closing boundary
        # per-part header parsing state
        self._hname = bytearray()
        self._hvalue = bytearray()
        self._headers: dict[str, bytes] = {}
        self._in_file = False

    # ---- python-multipart callbacks (synchronous, called from parser.write) ----
    def on_part_begin(self) -> None:
        self._headers = {}
        self._in_file = False

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._hname += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._hvalue += data[start:end]

    def on_header_end(self) -> None:
        self._headers[self._hname.decode("latin-1").lower()] = bytes(self._hvalue)
        self._hname.clear()
        self._hvalue.clear()

    def on_headers_finished(self) -> None:
        _, params = parse_options_header(self._headers.get("content-disposition", b""))
        filename = params.get(b"filename")
        if filename is None:
            return  # an ordinary form field - counted but ignored
        if params.get(b"name") != b"file":
            self._fail(400, "The PDF must be sent in a form field called 'file'.")
            return
        self.files_seen += 1
        if self.files_seen > 1:
            self._fail(400, "Upload one PDF per request.")
            return
        raw = filename.decode("utf-8", errors="replace")
        self.filename = safe_filename(raw)
        fd, self.path = tempfile.mkstemp(
            prefix=TEMP_PREFIX, suffix=TEMP_SUFFIX, dir=self.upload_dir
        )
        self.fh = os.fdopen(fd, "wb")
        self._in_file = True

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self.error is not None:
            return
        chunk = data[start:end]
        if not self._in_file:
            self.field_bytes += len(chunk)
            if self.field_bytes > _MAX_FIELD_BYTES:
                self._fail(400, "Form fields are too large.")
            return
        self.size += len(chunk)
        if self.size > self.max_bytes:
            self._fail(413, _too_large(self.max_bytes))
            return
        if not self.magic_ok:
            self.head += chunk[: MAGIC_WINDOW - len(self.head)]
            if PDF_MAGIC in self.head:
                self.magic_ok = True
            elif len(self.head) >= MAGIC_WINDOW:
                self._fail(415, "This file is not a PDF.")
                return
        self.pending += chunk

    def on_part_end(self) -> None:
        if self._in_file and self.error is None and not self.magic_ok:
            self._fail(415, "This file is not a PDF." if self.size else "The file is empty.")
        self._in_file = False

    def on_end(self) -> None:
        self.ended = True

    def callbacks(self) -> dict:
        return {
            "on_end": self.on_end,
            "on_part_begin": self.on_part_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
        }

    def _fail(self, status: int, detail: str) -> None:
        if self.error is None:
            self.error = HTTPException(status_code=status, detail=detail)

    # ---- async side ----
    async def flush(self, force: bool = False) -> None:
        if self.fh is None or not self.pending:
            return
        if force or len(self.pending) >= _FLUSH_BYTES:
            data = bytes(self.pending)
            self.pending.clear()
            await anyio.to_thread.run_sync(self.fh.write, data)

    def discard(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None
        if self.path:
            Path(self.path).unlink(missing_ok=True)
            self.path = None


def _too_large(max_bytes: int) -> str:
    return f"This file is larger than the {max_bytes // (1024 * 1024)} MB upload limit."


async def receive_pdf(request: Request, max_bytes: int, upload_dir: str | None) -> ReceivedFile:
    """Stream the ``file`` field of a multipart request into a temp file.

    Raises ``HTTPException`` (400/411/413/415/507) and leaves no temp file behind on
    failure.  On success the caller owns ``ReceivedFile.path``.
    """
    ctype, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise HTTPException(415, "Send the PDF as multipart/form-data with a 'file' field.")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length header.") from None
        if declared > max_bytes + _MULTIPART_OVERHEAD:
            raise HTTPException(413, _too_large(max_bytes))

    rx = _Receiver(max_bytes, upload_dir)
    parser = MultipartParser(boundary, rx.callbacks())
    try:
        async for chunk in request.stream():
            parser.write(chunk)
            if rx.error is not None:
                raise rx.error
            await rx.flush()
        parser.finalize()
        if rx.error is not None:
            raise rx.error
        if not rx.ended:
            raise HTTPException(400, "The upload was incomplete. Please try again.")
        if rx.path is None or rx.fh is None:
            raise HTTPException(400, "No file was uploaded (expected a form field called 'file').")
        await rx.flush(force=True)
        fh, rx.fh = rx.fh, None
        await anyio.to_thread.run_sync(fh.close)
        return ReceivedFile(path=rx.path, filename=rx.filename or "document.pdf", size=rx.size)
    except HTTPException:
        rx.discard()
        raise
    except ParseError:
        rx.discard()
        raise HTTPException(
            400, "The upload could not be read (malformed multipart body)."
        ) from None
    except OSError as exc:
        rx.discard()
        if exc.errno == 28:  # ENOSPC
            raise HTTPException(507, "The server is out of disk space.") from None
        raise
    except BaseException:  # client disconnect, cancellation, parser errors
        rx.discard()
        raise


def sweep_stale_uploads(upload_dir: str, older_than_s: float = 6 * 3600) -> int:
    """Delete half-written uploads left behind by a crash.  Only touches our own files."""
    removed = 0
    cutoff = time.time() - older_than_s
    for p in Path(upload_dir).glob(f"{TEMP_PREFIX}*{TEMP_SUFFIX}"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed
