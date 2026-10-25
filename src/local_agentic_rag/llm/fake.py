"""Deterministic scripted LLM for unit tests (no model download needed)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from local_agentic_rag.llm.base import LLM, Message

Responder = Callable[[list[Message], dict[str, Any] | None], str]


class ScriptedLLM(LLM):
    """Replies via ``responder(messages, json_schema)``; records every call."""

    def __init__(self, responder: Responder) -> None:
        self.name = "scripted"
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

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
