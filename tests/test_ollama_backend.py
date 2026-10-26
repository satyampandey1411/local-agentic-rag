"""OllamaLLM request handling: retries on runner crashes, no retry on client errors."""

from __future__ import annotations

import json

import pytest

from local_agentic_rag.llm import ollama
from local_agentic_rag.llm.base import LLMError, LLMTimeoutError
from local_agentic_rag.llm.ollama import OllamaLLM


class _Resp:
    def __init__(self, status: int, content: str = "ok") -> None:
        self.status_code = status
        self.text = f"status {status}"
        self._content = content

    def json(self) -> dict:
        return {"message": {"content": self._content}}


def _llm(monkeypatch, responses: list[_Resp]) -> tuple[OllamaLLM, list[dict]]:
    sent: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002 - mirrors requests.post
        sent.append(json)
        return responses.pop(0)

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(ollama.time, "sleep", lambda s: None)
    return OllamaLLM("m", retries=2, backoff=0.0), sent


def test_retries_runner_crash_then_succeeds(monkeypatch):
    llm, sent = _llm(monkeypatch, [_Resp(500), _Resp(200, " answer ")])
    assert llm.chat([{"role": "user", "content": "q"}]) == "answer"
    assert len(sent) == 2
    assert ollama.inflight_requests() == 0


def test_client_error_is_not_retried(monkeypatch):
    llm, sent = _llm(monkeypatch, [_Resp(404), _Resp(200)])
    with pytest.raises(LLMError, match="404"):
        llm.chat([{"role": "user", "content": "q"}])
    assert len(sent) == 1


def test_gives_up_after_retries(monkeypatch):
    llm, sent = _llm(monkeypatch, [_Resp(500), _Resp(500), _Resp(500)])
    with pytest.raises(LLMError, match="unreachable after 3 attempts"):
        llm.chat([{"role": "user", "content": "q"}])
    assert len(sent) == 3


def test_payload_forces_mmap_and_schema(monkeypatch):
    llm, sent = _llm(monkeypatch, [_Resp(200, "{}")])
    llm.chat([{"role": "user", "content": "q"}], json_schema={"type": "object"})
    assert sent[0]["options"]["use_mmap"] is True
    assert sent[0]["format"] == {"type": "object"}


# ---------------------------------------------------------------------------- streaming


class _StreamResp:
    """Mimics a streaming ``requests.Response``: NDJSON lines, optionally breaking mid-way."""

    def __init__(self, lines: list, status: int = 200, break_with: Exception | None = None):
        self.status_code = status
        self.text = f"status {status}"
        self._lines = lines
        self._break_with = break_with
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            yield line if isinstance(line, bytes) else json.dumps(line).encode()
        if self._break_with is not None:
            raise self._break_with

    def close(self) -> None:
        self.closed = True


def _chunk(text: str, done: bool = False) -> dict:
    return {"message": {"role": "assistant", "content": text}, "done": done}


def _stream_llm(monkeypatch, responses: list[_StreamResp]):
    sent: list[dict] = []

    def fake_post(url, json, timeout, stream=False):  # noqa: A002 - mirrors requests.post
        sent.append({"payload": json, "stream": stream})
        return responses.pop(0)

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(ollama.time, "sleep", lambda s: None)
    return OllamaLLM("m", retries=2, backoff=0.0), sent


def test_stream_reassembles_chunks(monkeypatch):
    resp = _StreamResp(
        [_chunk(" Rev"), b"", _chunk("enue grew"), _chunk(" 4% [1]. "), _chunk("", done=True)]
    )
    llm, sent = _stream_llm(monkeypatch, [resp])
    pieces = list(llm.chat_stream([{"role": "user", "content": "q"}], max_tokens=64))
    assert pieces == [" Rev", "enue grew", " 4% [1]. "]
    assert "".join(pieces).strip() == "Revenue grew 4% [1]."
    assert sent[0]["stream"] is True
    assert sent[0]["payload"]["stream"] is True
    assert sent[0]["payload"]["options"]["num_predict"] == 64
    assert resp.closed
    assert ollama.inflight_requests() == 0


def test_stream_error_line_raises(monkeypatch):
    resp = _StreamResp([_chunk("partial"), {"error": "model runner crashed"}])
    llm, _ = _stream_llm(monkeypatch, [resp])
    got = []
    with pytest.raises(LLMError, match="model runner crashed"):
        for piece in llm.chat_stream([{"role": "user", "content": "q"}]):
            got.append(piece)
    assert got == ["partial"]
    assert resp.closed
    assert ollama.inflight_requests() == 0


def test_stream_connection_drop_raises(monkeypatch):
    import requests

    resp = _StreamResp([_chunk("part")], break_with=requests.ConnectionError("reset"))
    llm, sent = _stream_llm(monkeypatch, [resp, _StreamResp([_chunk("x", done=True)])])
    with pytest.raises(LLMError, match="mid-stream"):
        list(llm.chat_stream([{"role": "user", "content": "q"}]))
    assert len(sent) == 1  # text was already shown, so no silent retry


def test_stream_ending_without_done_raises(monkeypatch):
    llm, _ = _stream_llm(monkeypatch, [_StreamResp([_chunk("cut off")])])
    with pytest.raises(LLMError, match="before the reply was complete"):
        list(llm.chat_stream([{"role": "user", "content": "q"}]))


def test_stream_retries_runner_crash_before_first_piece(monkeypatch):
    llm, sent = _stream_llm(
        monkeypatch, [_StreamResp([], status=500), _StreamResp([_chunk("ok", done=True)])]
    )
    assert list(llm.chat_stream([{"role": "user", "content": "q"}])) == ["ok"]
    assert len(sent) == 2


def test_stream_client_error_not_retried(monkeypatch):
    llm, sent = _stream_llm(monkeypatch, [_StreamResp([], status=404)])
    with pytest.raises(LLMError, match="404"):
        list(llm.chat_stream([{"role": "user", "content": "q"}]))
    assert len(sent) == 1


def test_default_chat_stream_yields_whole_reply():
    from local_agentic_rag.llm.base import LLM

    class Whole(LLM):
        def chat(self, messages, **kw):
            return "full reply"

    assert list(Whole().chat_stream([])) == ["full reply"]


def test_read_timeout_raises_llm_timeout_error_and_is_not_retried(monkeypatch):
    import requests

    llm, sent = _llm(monkeypatch, [])

    def slow_post(url, json, timeout):  # noqa: A002
        sent.append(json)
        raise requests.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "post", slow_post)
    with pytest.raises(LLMTimeoutError, match="timed out after"):
        llm.chat([{"role": "user", "content": "q"}])
    assert len(sent) == 1
    assert issubclass(LLMTimeoutError, LLMError)  # existing handlers keep catching it


def test_stream_read_timeout_raises_llm_timeout_error(monkeypatch):
    import requests

    llm, sent = _stream_llm(monkeypatch, [])

    def slow_post(url, json, timeout, stream=False):  # noqa: A002
        sent.append(json)
        raise requests.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "post", slow_post)
    with pytest.raises(LLMTimeoutError, match="timed out after"):
        list(llm.chat_stream([{"role": "user", "content": "q"}]))
    assert len(sent) == 1
