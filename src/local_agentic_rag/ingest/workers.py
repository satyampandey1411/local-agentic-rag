"""
Crash-isolated parse workers: subprocesses that extract raw pages and speak JSON lines.

Why subprocesses instead of ``multiprocessing``:
    * ``spawn``/``forkserver`` re-import the caller's ``__main__``, so a plain script
      without an ``if __name__ == "__main__"`` guard re-runs itself in every worker
      (and reloads the embedding model there).  ``fork`` copies a process that may hold
      CUDA state and server threads, which can deadlock.
    * A malformed page can crash or hang MuPDF.  Here that only kills one worker: the
      batch is retried page by page in a fresh worker, and a page that fails again
      becomes an empty page with a warning instead of failing a 10,000-page document.

Protocol: one JSON task per stdin line ``{"path", "start", "stop", "extract_tables",
"use_ocr", "ocr_language"}``; the worker replies with one stdout line per page
(``{"page": {...}}``) and then ``{"done": true}``, or ``{"error": "..."}``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

logger = logging.getLogger(__name__)

#: Seconds one page may take (OCR at 300 dpi on a slow CPU is the worst case).
PAGE_TIMEOUT_S = float(os.environ.get("LOCAL_RAG_PAGE_TIMEOUT", "180"))


class WorkerFailed(RuntimeError):
    pass


class _Worker:
    def __init__(self, env: dict[str, str]) -> None:
        self._env = env
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "local_agentic_rag.ingest.workers"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=self._env,
            text=True,
            encoding="utf-8",
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc, self.lines), daemon=True).start()

    @staticmethod
    def _read(proc: subprocess.Popen, lines: queue.Queue) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # EOF: the worker exited

    def run(self, task: dict, timeout: float, out: list[dict]) -> None:
        """Run one page-range task, appending each page to ``out`` as it arrives.

        ``timeout`` applies per page, so a hang is detected after one page's budget
        and the pages finished before it are kept.
        """
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(task) + "\n")
            self.proc.stdin.flush()
        except OSError as exc:  # broken pipe: the worker already died
            raise WorkerFailed(f"worker exited ({exc})") from exc
        while True:
            try:
                line = self.lines.get(timeout=timeout)
            except queue.Empty as exc:
                raise WorkerFailed(f"timed out after {timeout:.0f}s") from exc
            if line is None:
                raise WorkerFailed(f"worker crashed (exit code {self.proc.wait()})")
            try:
                reply = json.loads(line)
            except ValueError as exc:  # stray output on the protocol stream: fail this page
                raise WorkerFailed(f"unreadable worker reply: {line[:80]!r}") from exc
            if "error" in reply:
                raise WorkerFailed(reply["error"])
            if reply.get("done"):
                return
            out.append(reply["page"])

    def restart(self) -> None:
        self.close()
        self._start()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def _failed_page(index: int, reason: str) -> dict:
    from local_agentic_rag.ingest.pdf import RawPage

    raw = RawPage(page=index + 1, blocks=[], tables=[], height=1.0)
    raw.warning = f"page {index + 1}: parse error ({reason})"
    return raw.to_dict()


class ParsePool:
    """``n`` parse workers; :meth:`imap` yields page-range results in order."""

    def __init__(self, n: int, **task_args) -> None:
        env = dict(os.environ)
        if n > 1:
            env.setdefault("OMP_THREAD_LIMIT", "1")  # one Tesseract thread per worker
            env.setdefault("LOCAL_RAG_ONNX_THREADS", str(max(1, (os.cpu_count() or n) // n)))
        self.task_args = task_args
        self._closed = False
        self.workers: list[_Worker] = []
        try:
            for _ in range(n):
                self.workers.append(_Worker(env))
        except BaseException:  # do not leak the workers already started
            self.close()
            raise

    def _run_range(self, worker: _Worker, start: int, stop: int) -> list[dict]:
        """Pages ``start..stop-1``; a page that crashes or hangs its worker is replaced by
        an empty page with a warning, the worker restarts and the range continues."""
        pages: list[dict] = []
        while start + len(pages) < stop:
            first = start + len(pages)
            try:
                worker.run({**self.task_args, "start": first, "stop": stop}, PAGE_TIMEOUT_S, pages)
            except WorkerFailed as exc:
                if self._closed:
                    raise
                bad = start + len(pages)  # pages are extracted in order: the next one failed
                logger.warning("page %d: %s; restarting the worker", bad + 1, exc)
                worker.restart()
                pages.append(_failed_page(bad, str(exc)))
        return pages

    def imap(self, ranges: list[tuple[int, int]]) -> Iterator[list[dict]]:
        results: dict[int, list[dict] | BaseException] = {}
        cond = threading.Condition()
        state = {"next_task": 0, "next_yield": 0, "stop": False}
        window = 2 * len(self.workers)  # bounded number of finished-but-unread batches

        def drive(worker: _Worker) -> None:
            while True:
                with cond:
                    while (
                        not state["stop"]
                        and state["next_task"] < len(ranges)
                        and state["next_task"] >= state["next_yield"] + window
                    ):
                        cond.wait()
                    if state["stop"] or state["next_task"] >= len(ranges):
                        return
                    i = state["next_task"]
                    state["next_task"] += 1
                try:
                    res: list[dict] | BaseException = self._run_range(worker, *ranges[i])
                except BaseException as exc:  # surfaced to the consumer, never swallowed
                    res = exc
                with cond:
                    results[i] = res
                    cond.notify_all()

        threads = [threading.Thread(target=drive, args=(w,), daemon=True) for w in self.workers]
        for t in threads:
            t.start()
        try:
            for i in range(len(ranges)):
                with cond:
                    while i not in results:
                        cond.wait()
                    res = results.pop(i)
                    state["next_yield"] = i + 1
                    cond.notify_all()
                if isinstance(res, BaseException):
                    raise res
                yield res
        finally:
            with cond:
                state["stop"] = True
                cond.notify_all()

    def close(self) -> None:
        self._closed = True
        for w in self.workers:
            w.close()

    def __enter__(self) -> ParsePool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _limit_onnx_threads(threads: int) -> None:
    """Cap ONNX Runtime's threads in this worker.

    PyMuPDF 2's table finder runs ONNX layout models, and every ONNX session starts one
    thread per core by default.  With several parse workers that oversubscribes the CPU
    (load average 370 measured on 16 cores with 8 workers); each worker gets its share.
    """
    if threads <= 0:
        return
    try:
        import onnxruntime as ort
    except ImportError:
        return

    class Limited(ort.InferenceSession):  # a class: PyMuPDF uses it in type expressions
        def __init__(self, path_or_bytes, sess_options=None, **kwargs):
            options = sess_options or ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options=options, **kwargs)

    ort.InferenceSession = Limited


def _serve() -> None:
    # Keep the protocol stream clean: anything else written to stdout (C libraries,
    # stray prints) goes to stderr instead.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    logging.basicConfig(level=logging.WARNING)
    _limit_onnx_threads(int(os.environ.get("LOCAL_RAG_ONNX_THREADS", "0")))
    import pymupdf

    from local_agentic_rag.ingest.pdf import extract_raw_page

    # Fault injection for failure-recovery tests and benchmarks: 1-based page numbers
    # whose extraction kills ("crash:") or hangs ("hang:") this worker.
    faults = os.environ.get("LOCAL_RAG_FAULT_PAGES", "")
    crash = (
        {int(x) for x in faults.removeprefix("crash:").split(",") if x.isdigit()}
        if faults.startswith("crash:")
        else set()
    )
    hang = (
        {int(x) for x in faults.removeprefix("hang:").split(",") if x.isdigit()}
        if faults.startswith("hang:")
        else set()
    )

    def send(reply: dict) -> None:
        # ensure_ascii: lone surrogates or other odd code points in page text cannot break
        # the UTF-8 protocol stream.
        protocol.write(json.dumps(reply) + "\n")
        protocol.flush()

    open_doc: list = [None, None]  # (path, document): reopening per page costs ~150 ms

    def document(path: str):
        if open_doc[0] != path:
            if open_doc[1] is not None:
                open_doc[1].close()
            open_doc[:] = [path, pymupdf.open(path)]
        return open_doc[1]

    for line in sys.stdin:
        task = json.loads(line)
        for index in range(task["start"], task["stop"]):
            if index + 1 in crash:
                os._exit(3)
            if index + 1 in hang:
                time.sleep(3600)
            try:
                page = extract_raw_page(
                    document(task["path"]),
                    index,
                    task["extract_tables"],
                    task["use_ocr"],
                    task["ocr_language"],
                )
            except Exception as exc:  # reported to the parent, which skips this page
                send({"error": f"{type(exc).__name__}: {exc}"})
                break
            send({"page": page})
        else:
            send({"done": True})


if __name__ == "__main__":
    _serve()
