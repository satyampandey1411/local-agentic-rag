"""
llama-cpp-python backend (in-process GGUF inference).

Useful for CPU-only development and for environments where running a
separate server is inconvenient.  The chat template is read from the GGUF
metadata, so any instruct model works without hand-written prompt tokens
(the original prototype hard-coded Llama-3 special tokens).
"""

from __future__ import annotations

from typing import Any

from local_agentic_rag.llm.base import LLM, LLMError, Message


class LlamaCppLLM(LLM):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        seed: int = 42,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise LLMError("llama-cpp-python is not installed") from exc
        self.model_path = model_path
        self.name = f"llamacpp:{model_path.rsplit('/', 1)[-1]}"
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_threads=n_threads,
            verbose=verbose,
        )

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
        }
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object", "schema": json_schema}
        try:
            out = self._llm.create_chat_completion(**kwargs)
        except Exception as exc:
            raise LLMError(f"llama.cpp generation failed: {exc}") from exc
        return (out["choices"][0]["message"].get("content") or "").strip()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "backend": "llama_cpp", "model_path": self.model_path}
