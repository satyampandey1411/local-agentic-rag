"""
Server-Sent Events (SSE) and JSON helpers.

SSE is a one-way, line-based protocol over a normal HTTP response: each event is
``event: <name>`` + ``data: <payload>`` lines followed by a blank line, and lines
starting with ``:`` are comments (used here as heartbeats so proxies do not close
an idle connection).  ``sse-starlette`` is not a dependency, so the few bytes of
framing are produced here and sent through a plain ``StreamingResponse``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

#: Headers that stop browsers, Starlette and reverse proxies (nginx, the Colab
#: port proxy) from caching or buffering the stream.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

#: Some proxies hold back the first few KB of a response before forwarding
#: anything; a padding comment at the start forces the first flush.
PADDING = ":" + " " * 2048 + "\n\n"


def _default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, "item"):  # numpy scalars
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    return str(obj)


def _finite(obj: Any) -> Any:
    """Replace NaN/Infinity (invalid JSON that breaks ``JSON.parse``) with null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def dumps(data: Any) -> str:
    """JSON-encode anything the library may hand back (numpy values, sets, dataclasses)."""
    try:
        return json.dumps(data, default=_default, ensure_ascii=False, allow_nan=False)
    except ValueError:  # NaN/Infinity somewhere in the payload
        plain = json.loads(json.dumps(data, default=_default, ensure_ascii=False))
        return json.dumps(_finite(plain), ensure_ascii=False, allow_nan=False)


def event(name: str, data: Any) -> str:
    """One SSE frame.  ``json.dumps`` never emits raw newlines, so one data line suffices."""
    return f"event: {name}\ndata: {dumps(data)}\n\n"


def comment(text: str = "ping") -> str:
    return f": {text}\n\n"
