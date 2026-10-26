"""
Running blocking ``library.ask`` calls from async request handlers.

Two problems are solved here:

1. **Admission control.**  One GPU can only run a couple of answers at a time.
   :class:`AskGate` is a first-come-first-served counting semaphore that also
   knows each waiter's place in line, so the UI can say "you are #3 in line".

2. **Thread <-> event-loop bridging.**  ``library.ask`` blocks for seconds to
   minutes, so it runs in a dedicated thread pool.  Its ``on_step`` callback fires
   in that worker thread; :class:`AskJob` forwards each step into an
   ``asyncio.Queue`` with ``loop.call_soon_threadsafe`` so an async generator can
   stream it to the browser while the answer is still being produced.

Timeouts: Python cannot kill a running thread.  When ``ask_timeout_s`` expires the
*client* gets a 504 / ``error`` event immediately, but the worker thread keeps
running until ``library.ask`` returns on its own.  Its gate slot is only released
when the thread really finishes, so a stuck question can never cause more than
``max_concurrent_asks`` answers to run on the GPU at once.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from local_agentic_rag.library_types import LibraryProtocol
from local_agentic_rag.schema import Answer, TraceStep

logger = logging.getLogger(__name__)


class AskGate:
    """FIFO admission control.  Must only be used from one event loop thread.

    ``enqueue()`` returns a future that resolves once the caller holds a slot.
    A caller that gives up while waiting calls ``abandon(ticket)``; a caller that
    got a slot calls ``release()`` exactly once when its work is done.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.limit = limit
        self.active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    def enqueue(self) -> asyncio.Future[None]:
        ticket: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if self.active < self.limit and not self._waiters:
            self.active += 1
            ticket.set_result(None)
        else:
            self._waiters.append(ticket)
        return ticket

    def position(self, ticket: asyncio.Future[None]) -> int:
        """1-based place in line, or 0 once the ticket holds a slot."""
        try:
            return self._waiters.index(ticket) + 1
        except ValueError:
            return 0

    def abandon(self, ticket: asyncio.Future[None]) -> None:
        if ticket in self._waiters:
            self._waiters.remove(ticket)
            ticket.cancel()
        elif ticket.done() and not ticket.cancelled():
            self.release()  # the slot was granted but never used

    def release(self) -> None:
        self.active -= 1
        while self._waiters and self.active < self.limit:
            nxt = self._waiters.popleft()
            if not nxt.done():
                self.active += 1
                nxt.set_result(None)


class AskJob:
    """One ``library.ask`` call running in a worker thread, with its steps queued."""

    def __init__(
        self,
        library: LibraryProtocol,
        executor: concurrent.futures.Executor,
        gate: AskGate,
        question: str,
        doc_ids: list[str] | None,
        history: list[tuple[str, str]],
        stream_steps: bool,
    ) -> None:
        self.loop = asyncio.get_running_loop()
        self.steps: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        on_step: Callable[[TraceStep], None] | None = self._on_step if stream_steps else None
        cf = executor.submit(library.ask, question, doc_ids, history, on_step)
        # The gate slot is held until the *thread* finishes, even if the client
        # already got a timeout - see the module docstring.
        cf.add_done_callback(lambda _f: self._threadsafe(gate.release))
        self.future: asyncio.Future[Answer] = asyncio.wrap_future(cf, loop=self.loop)

    def _threadsafe(self, fn: Callable[..., Any], *args: Any) -> None:
        with contextlib.suppress(RuntimeError):  # loop already closed (server shutdown)
            self.loop.call_soon_threadsafe(fn, *args)

    def _on_step(self, step: TraceStep) -> None:  # runs in the worker thread
        try:
            payload = asdict(step)
        except TypeError:
            payload = {"step": str(getattr(step, "step", "step")), "detail": {}, "seconds": 0.0}
        self._threadsafe(self.steps.put_nowait, payload)
