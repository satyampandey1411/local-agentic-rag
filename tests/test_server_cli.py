"""Command-line option defaults of ``python -m local_agentic_rag.server``."""

from __future__ import annotations

import pytest

from local_agentic_rag.server.__main__ import _ollama_url, _parse_args


def test_defaults_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCAL_RAG_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("LOCAL_RAG_MODEL", raising=False)
    args = _parse_args([])
    assert args.ollama_host == "127.0.0.1:11434"
    assert args.model == "qwen2.5:7b-instruct"
    assert args.host == "127.0.0.1"  # localhost only unless asked otherwise


def test_environment_sets_ollama_host_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RAG_OLLAMA_HOST", "ollama:11434")
    monkeypatch.setenv("LOCAL_RAG_MODEL", "qwen2.5:3b-instruct")
    args = _parse_args([])
    assert args.ollama_host == "ollama:11434"
    assert args.model == "qwen2.5:3b-instruct"
    assert _ollama_url(args.ollama_host) == "http://ollama:11434"


def test_explicit_flags_beat_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RAG_OLLAMA_HOST", "ollama:11434")
    monkeypatch.setenv("LOCAL_RAG_MODEL", "from-env")
    args = _parse_args(["--ollama-host", "http://gpu-box:11434", "--model", "from-flag"])
    assert args.ollama_host == "http://gpu-box:11434"
    assert args.model == "from-flag"
    assert _ollama_url(args.ollama_host) == "http://gpu-box:11434"
