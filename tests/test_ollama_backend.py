"""OllamaLLM request handling: retries on runner crashes, no retry on client errors."""

from __future__ import annotations

import pytest

from local_agentic_rag.llm import ollama
from local_agentic_rag.llm.base import LLMError
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
