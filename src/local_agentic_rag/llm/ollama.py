"""
Ollama backend (default on Colab).

Why Ollama on Colab: it ships prebuilt CUDA binaries (no 15-minute
llama.cpp source build, no fragile wheel index), downloads quantised GGUF
models from its registry quickly, supports JSON-schema constrained output,
and keeps the model resident between calls.  The server is installed and
started by ``local_agentic_rag.colab.start_ollama``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from local_agentic_rag.llm.base import LLM, LLMError, Message

logger = logging.getLogger(__name__)

_inflight = 0
_inflight_lock = threading.Lock()


def inflight_requests() -> int:
    """Number of chat requests this process currently has open against Ollama."""
    return _inflight


class OllamaLLM(LLM):
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        host: str = "http://127.0.0.1:11434",
        num_ctx: int = 8192,
        timeout: float = 300.0,
        keep_alive: str = "60m",
        seed: int = 42,
        think: bool | None = None,
        retries: int = 3,
        backoff: float = 5.0,
        use_mmap: bool = True,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.seed = seed
        self.think = think
        self.retries = retries
        self.backoff = backoff
        # Colab has only ~12.7 GB host RAM.  Under memory pressure Ollama disables mmap and
        # copies the whole GGUF into anonymous RAM (observed: 8.7 GB RSS) even though every
        # layer is on the GPU, which got the runner OOM-killed.  Forcing mmap keeps the
        # weights file-backed and reclaimable.
        self.use_mmap = use_mmap

    # ------------------------------------------------------------------ server helpers

    def is_server_up(self) -> bool:
        import requests

        try:
            return requests.get(f"{self.host}/api/version", timeout=3).ok
        except requests.RequestException:
            return False

    def has_model(self) -> bool:
        import requests

        resp = requests.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names

    def pull(self, progress: bool = True) -> None:
        import requests

        with requests.post(
            f"{self.host}/api/pull",
            json={"model": self.model, "stream": True},
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            last = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise LLMError(f"ollama pull {self.model} failed: {event['error']}")
                status = event.get("status", "")
                if progress and status != last:
                    print(f"  ollama pull {self.model}: {status}", flush=True)
                    last = status

    def ensure_ready(self, pull: bool = True) -> None:
        if not self.is_server_up():
            raise LLMError(
                f"Ollama server not reachable at {self.host}. Start it with "
                "local_agentic_rag.colab.start_ollama() or `ollama serve`."
            )
        if not self.has_model():
            if not pull:
                raise LLMError(f"Model {self.model!r} not present; run `ollama pull {self.model}`.")
            self.pull()
        # Warm-up loads the weights into VRAM so the first question is not penalised.
        self.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=4)

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> str:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "seed": self.seed,
                "use_mmap": self.use_mmap,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        if json_schema is not None:
            payload["format"] = json_schema
        if self.think is not None:
            payload["think"] = self.think

        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._post(payload)
                if resp.status_code == 200:
                    return resp.json().get("message", {}).get("content", "").strip()
                error = LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
                if resp.status_code < 500:
                    raise error  # bad request / unknown model: retrying will not help
                # 5xx: the model runner crashed or was restarted (observed on Colab when it
                # was OOM-killed); Ollama respawns it on the next request, so retry.
                last_exc = error
            except requests.ReadTimeout as exc:
                # The server accepted the request but generation exceeded the timeout:
                # re-sending would just repeat the same long generation.
                raise LLMError(f"Ollama timed out after {self.timeout:.0f}s") from exc
            except (requests.ConnectionError, requests.ConnectTimeout) as exc:
                last_exc = exc
            logger.warning("Ollama call failed (attempt %d): %s", attempt + 1, last_exc)
            if attempt < self.retries:
                time.sleep(self.backoff * (attempt + 1))
        raise LLMError(f"Ollama unreachable after {self.retries + 1} attempts: {last_exc}")

    def _post(self, payload: dict[str, Any]):
        import requests

        global _inflight
        with _inflight_lock:
            _inflight += 1
        try:
            return requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        finally:
            with _inflight_lock:
                _inflight -= 1

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": "ollama",
            "model": self.model,
            "num_ctx": self.num_ctx,
        }
