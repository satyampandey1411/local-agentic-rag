"""
Command line entry point.

    python -m local_agentic_rag.server --library ./library --model qwen2.5:7b-instruct \
        --ollama-host 127.0.0.1:11434 --host 127.0.0.1 --port 8000

``--model`` and ``--ollama-host`` default to the ``LOCAL_RAG_MODEL`` and
``LOCAL_RAG_OLLAMA_HOST`` environment variables (used by the Docker image); an explicit
flag always wins.

    python -m local_agentic_rag.server --demo --port 8765   # UI demo: no GPU, no LLM

Always runs ONE uvicorn worker process: the library owns the GPU models and the
background ingestion queue, which cannot be shared between processes.  Concurrency
comes from threads and the async event loop inside that one process.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from local_agentic_rag.library_types import LibraryProtocol


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m local_agentic_rag.server",
        description="Serve the 'chat with your PDFs' web app.",
    )
    p.add_argument(
        "--library",
        default="./library",
        help="directory holding uploaded PDFs and their indexes (default: ./library)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("LOCAL_RAG_MODEL", "qwen2.5:7b-instruct"),
        help="Ollama model name (default: $LOCAL_RAG_MODEL, else qwen2.5:7b-instruct)",
    )
    p.add_argument(
        "--ollama-host",
        default=os.environ.get("LOCAL_RAG_OLLAMA_HOST", "127.0.0.1:11434"),
        help="Ollama host[:port] or URL (default: $LOCAL_RAG_OLLAMA_HOST, else 127.0.0.1:11434)",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to listen on (use 0.0.0.0 to expose on the network)",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--token",
        default=os.environ.get("LOCAL_RAG_TOKEN") or None,
        help="require this access token on every /api route except /api/health "
        "(default: $LOCAL_RAG_TOKEN; unset = no authentication)",
    )
    p.add_argument("--max-upload-mb", type=int, default=2048)
    p.add_argument(
        "--max-concurrent-asks",
        type=int,
        default=2,
        help="questions answered at the same time (they share one GPU)",
    )
    p.add_argument("--ask-timeout", type=float, default=300.0, help="seconds per answer")
    p.add_argument(
        "--upload-dir",
        default=None,
        help="where uploads are streamed before ingestion "
        "(default: <library>/.incoming, same filesystem => cheap move)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="run with the built-in DemoLibrary (fake ingestion and answers, "
        "real page rendering) for UI work without a GPU or Ollama",
    )
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def _is_loopback(host: str) -> bool:
    return host in ("localhost", "::1") or host.startswith("127.")


def _upload_quota_from_env() -> int | None:
    """``LOCAL_RAG_UPLOAD_QUOTA_BYTES``: per-IP daily upload quota; unset/0 = off."""
    raw = os.environ.get("LOCAL_RAG_UPLOAD_QUOTA_BYTES", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f"LOCAL_RAG_UPLOAD_QUOTA_BYTES must be a whole number of bytes, got {raw!r}."
        ) from None
    return value if value > 0 else None


def _ollama_url(host: str) -> str:
    return host if "://" in host else f"http://{host}"


# =====================================================================================
# TODO(lead engineer): finish once ``local_agentic_rag.library`` lands.
# This is the ONLY place the server touches the concrete backend.  Adjust the
# Library(...) / OllamaLLM(...) arguments to the final constructor signatures.
# =====================================================================================
def _build_library(args: argparse.Namespace) -> LibraryProtocol:
    from local_agentic_rag.library import Library  # lazy: heavy imports (torch, models)
    from local_agentic_rag.llm.ollama import OllamaLLM

    llm = OllamaLLM(model=args.model, host=_ollama_url(args.ollama_host))
    return Library(args.library, llm=llm)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from local_agentic_rag.server.app import create_app

    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    library: Any
    if args.demo:
        from local_agentic_rag.server.demo import DemoLibrary

        library = DemoLibrary(args.library if args.library != "./library" else None)
        upload_dir = args.upload_dir
    else:
        os.makedirs(args.library, exist_ok=True)
        try:
            library = _build_library(args)
        except ModuleNotFoundError as exc:
            if exc.name != "local_agentic_rag.library":
                raise
            raise SystemExit(
                "local_agentic_rag.library is not available yet. "
                "Run with --demo to try the UI without it."
            ) from None
        upload_dir = args.upload_dir or os.path.join(args.library, ".incoming")

    app = create_app(
        library,
        max_upload_mb=args.max_upload_mb,
        ask_timeout_s=args.ask_timeout,
        max_concurrent_asks=args.max_concurrent_asks,
        upload_dir=upload_dir,
        token=args.token,
        upload_quota_bytes=_upload_quota_from_env(),
    )
    if not args.token and not _is_loopback(args.host):
        logging.getLogger(__name__).warning(
            "listening on %s with NO access token: anyone who can reach this port can read, "
            "upload and delete every document. Set --token (or LOCAL_RAG_TOKEN) and put TLS "
            "in front, or bind to 127.0.0.1.",
            args.host,
        )
    print(f"\n  Open http://{args.host}:{args.port}/ in your browser\n", flush=True)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
