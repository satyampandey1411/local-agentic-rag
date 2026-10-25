"""
Minimal chat-LLM interface.

Every agent step talks to the model through ``LLM.chat``.  Backends only need
to turn OpenAI-style ``messages`` into text, optionally constrained to a JSON
schema (grammar-constrained decoding in Ollama / llama.cpp), which is what
makes small local models reliable at structured steps like grading.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

Message = dict[str, str]


class LLMError(RuntimeError):
    """Backend failure (server down, model missing, timeout)."""


class LLMOutputError(LLMError):
    """The backend worked but the model's output was unusable (e.g. invalid JSON)."""


class LLM(ABC):
    name: str = "llm"

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> str:
        """Return the assistant reply text."""

    def chat_json(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Chat with schema-constrained output and parse it; raises ``LLMError`` if unparseable."""
        text = self.chat(
            messages, max_tokens=max_tokens, temperature=temperature, json_schema=schema
        )
        data = parse_json_object(text)
        if data is None:
            raise LLMOutputError(f"Model did not return valid JSON: {text[:200]!r}")
        return data

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object in ``text``."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None
