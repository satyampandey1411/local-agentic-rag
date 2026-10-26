"""Deterministic scripted LLM for unit tests (no model download needed)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from local_agentic_rag.llm.base import LLM, Message

Responder = Callable[[list[Message], dict[str, Any] | None], str]


class ScriptedLLM(LLM):
    """Replies via ``responder(messages, json_schema)``; records every call."""

    def __init__(self, responder: Responder) -> None:
        self.name = "scripted"
        self.responder = responder
        self.calls: list[dict[str, Any]] = []
        self.stream_calls = 0

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> str:
        self.calls.append({"messages": messages, "json_schema": json_schema})
        return self.responder(messages, json_schema)

    def chat_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Scripted streaming: the same reply as ``chat``, split into several chunks."""
        text = self.chat(messages, max_tokens=max_tokens, temperature=temperature, stop=stop)
        self.stream_calls += 1
        yield from split_chunks(text)


def split_chunks(text: str, size: int = 12) -> list[str]:
    """Cut ``text`` into pieces of about ``size`` characters; joining them restores it."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]
