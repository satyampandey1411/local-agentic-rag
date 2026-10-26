"""
Google Colab helpers: GPU checks, the Ollama server, and notebook rendering.

Everything here also imports outside Colab (on a laptop, in CI): heavy or
optional modules (torch, IPython, requests) are imported inside functions, and
every probe "fails soft", returning a dict with an ``error`` key instead of
raising when there is no GPU or no ``nvidia-smi``.

Typical notebook use::

    from local_agentic_rag.colab import setup_llm, display_answer, print_gpu_report
    print_gpu_report()
    llm = setup_llm("qwen2.5:7b-instruct")          # install + start + pull + warm-up
    ...
    display_answer(pipe.ask("What dataset did they use?"))

Why Ollama runs as a separate process: it is a Go server holding the model in
GPU memory.  Python only talks to it over HTTP (``OllamaLLM``), so the model's
VRAM never shows up in ``torch.cuda`` counters, and that is why
:func:`vram_snapshot` also asks ``nvidia-smi``.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_agentic_rag.llm.ollama import OllamaLLM
    from local_agentic_rag.schema import Answer, Citation, IngestReport, TraceStep

DEFAULT_HOST = "127.0.0.1:11434"
OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"

#: Wall-clock timings recorded by :func:`timed` (and :func:`setup_llm`); shown by
#: :func:`print_resource_report`.
TIMINGS: dict[str, float] = {}


# --------------------------------------------------------------------------- environment


def is_colab() -> bool:
    """True when running inside a Google Colab kernel."""
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    """Run a command and return stdout, or None if it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout


@contextmanager
def timed(label: str, verbose: bool = True) -> Iterator[None]:
    """``with timed("ingest"):`` records the block's duration in :data:`TIMINGS`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        TIMINGS[label] = time.perf_counter() - t0
        if verbose:
            print(f"[{label}] {TIMINGS[label]:.1f}s")


# --------------------------------------------------------------------------- GPU


def _nvidia_smi_gpus() -> list[dict[str, Any]] | None:
    fields = "name,memory.total,memory.used,driver_version,compute_cap"
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if out is None:
        # Older drivers do not know ``compute_cap``; retry without it.
        fields = "name,memory.total,memory.used,driver_version"
        out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if out is None:
        return None
    keys = fields.split(",")
    gpus = []
    for line in out.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        row = dict(zip(keys, values, strict=False))
        gpus.append(
            {
                "name": row.get("name"),
                "memory_total_mb": _to_int(row.get("memory.total")),
                "memory_used_mb": _to_int(row.get("memory.used")),
                "driver": row.get("driver_version"),
                "compute_capability": row.get("compute_cap"),
            }
        )
    return gpus


def _to_int(value: str | None) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_version": None, "torch_cuda_available": False, "torch_cuda_version": None}
    available = bool(torch.cuda.is_available())
    return {
        "torch_version": torch.__version__,
        "torch_cuda_available": available,
        "torch_cuda_version": torch.version.cuda,
        "torch_device_name": torch.cuda.get_device_name(0) if available else None,
    }


def gpu_report() -> dict[str, Any]:
    """GPU name, memory, driver and CUDA versions, plus what torch sees.

    Never raises: without ``nvidia-smi`` (CPU runtime, laptop) it returns
    ``gpus=[]`` and an ``error`` explaining what to do.
    """
    report: dict[str, Any] = {"gpus": [], "driver": None, "cuda_version": None, "error": None}
    gpus = _nvidia_smi_gpus()
    if gpus is None:
        report["error"] = (
            "nvidia-smi not found or failed: no NVIDIA GPU is visible. "
            "In Colab: Runtime > Change runtime type > T4 GPU."
        )
    else:
        report["gpus"] = gpus
        report["driver"] = gpus[0]["driver"] if gpus else None
        header = _run(["nvidia-smi"]) or ""
        match = re.search(r"CUDA Version:\s*([\d.]+)", header)
        report["cuda_version"] = match.group(1) if match else None
    report.update(_torch_info())
    return report


def print_gpu_report() -> dict[str, Any]:
    report = gpu_report()
    if report["error"]:
        print(f"WARNING: {report['error']}")
    for i, gpu in enumerate(report["gpus"]):
        cc = f", compute {gpu['compute_capability']}" if gpu.get("compute_capability") else ""
        print(
            f"GPU {i}: {gpu['name']}{cc} | {gpu['memory_used_mb']} / {gpu['memory_total_mb']} MiB used"
        )
    if report["gpus"]:
        print(f"Driver {report['driver']} | CUDA (driver) {report['cuda_version']}")
    print(
        f"torch {report['torch_version']} | CUDA build {report['torch_cuda_version']} | "
        f"cuda available: {report['torch_cuda_available']}"
    )
    return report


def vram_snapshot() -> dict[str, Any]:
    """Memory used by torch (embedder, reranker) and by the whole GPU (incl. Ollama).

    ``torch_*`` numbers only cover tensors allocated by this Python process.
    The LLM lives in the separate ``ollama serve`` process, so ``gpu_used_mb``
    from ``nvidia-smi`` is the real total.
    """
    snap: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            mb = 1024 * 1024
            snap["torch_allocated_mb"] = round(torch.cuda.memory_allocated() / mb, 1)
            snap["torch_reserved_mb"] = round(torch.cuda.memory_reserved() / mb, 1)
            snap["torch_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / mb, 1)
    except ImportError:
        pass
    gpus = _nvidia_smi_gpus()
    if gpus:
        snap["gpu_used_mb"] = gpus[0]["memory_used_mb"]
        snap["gpu_total_mb"] = gpus[0]["memory_total_mb"]
        if snap.get("gpu_used_mb") is not None and "torch_reserved_mb" in snap:
            # Rough: everything not reserved by torch (CUDA context ~0.3 GB + Ollama).
            snap["other_processes_mb"] = round(snap["gpu_used_mb"] - snap["torch_reserved_mb"], 1)
    if not snap:
        snap["error"] = "no CUDA device visible"
    return snap


def print_resource_report() -> None:
    """VRAM snapshot plus the timings recorded with :func:`timed` / :func:`setup_llm`."""
    snap = vram_snapshot()
    print("VRAM")
    for key, value in snap.items():
        print(f"  {key:<24} {value}")
    if TIMINGS:
        print("Timings")
        for label, seconds in TIMINGS.items():
            print(f"  {label:<24} {seconds:8.1f}s")


# --------------------------------------------------------------------------- Ollama


INSTALL_ATTEMPTS = 3


def install_ollama(verbose: bool = True) -> str:
    """Install the Ollama binary (no-op if ``ollama`` is already on PATH).

    The official installer unpacks a ``.tar.zst`` archive (needs ``zstd``) and
    uses ``lspci`` (``pciutils``) to detect the GPU; neither ships with Colab,
    so they are apt-installed first when missing.
    """
    existing = shutil.which("ollama")
    if existing:
        if verbose:
            print(f"ollama already installed: {existing}")
        return existing

    missing = [
        pkg for pkg, binary in (("zstd", "zstd"), ("pciutils", "lspci")) if not shutil.which(binary)
    ]
    if missing and shutil.which("apt-get"):
        if verbose:
            print(f"apt-get install {' '.join(missing)} ...")
        is_root = getattr(os, "geteuid", lambda: 0)() == 0
        prefix = [] if is_root or not shutil.which("sudo") else ["sudo"]
        subprocess.run([*prefix, "apt-get", "update", "-qq"], check=False, capture_output=True)
        result = subprocess.run(
            [*prefix, "apt-get", "install", "-y", "-qq", *missing], capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"WARNING: apt-get install {missing} failed:\n{result.stderr[-1000:]}")

    # The installer downloads ~1.5 GB; a connection reset half-way ("curl: (56)") was
    # observed on Kaggle, so transient network failures are retried.
    for attempt in range(1, INSTALL_ATTEMPTS + 1):
        if verbose:
            print(f"Installing Ollama (official install script, attempt {attempt}) ...")
        result = subprocess.run(
            f"curl -fsSL {OLLAMA_INSTALL_URL} | sh",
            shell=True,
            capture_output=True,
            text=True,
        )
        path = shutil.which("ollama")
        if result.returncode == 0 and path is not None:
            break
        if attempt < INSTALL_ATTEMPTS:
            time.sleep(10 * attempt)
    if result.returncode != 0 or path is None:
        tail = (result.stdout + result.stderr)[-2000:]
        raise RuntimeError(f"Ollama installation failed (exit {result.returncode}):\n{tail}")
    if verbose:
        print(f"ollama installed: {path}")
    return path


def _server_up(host: str, timeout: float = 2.0) -> bool:
    url = f"http://{_strip_scheme(host)}/api/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _strip_scheme(host: str) -> str:
    return re.sub(r"^https?://", "", host).rstrip("/")


def _default_log_path() -> str:
    if Path("/content").is_dir():
        return "/content/ollama.log"
    return str(Path(tempfile.gettempdir()) / "ollama.log")


def _tail(path: str | Path, lines: int = 30) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return "(no log file)"
    return "\n".join(text.splitlines()[-lines:]) or "(log is empty)"


def start_ollama(
    host: str = DEFAULT_HOST,
    log_path: str | None = None,
    timeout: float = 60.0,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen | None:
    """Launch ``ollama serve`` in the background and wait until it answers.

    Returns the ``Popen`` handle, or ``None`` if a server was already running
    at ``host``.  Raises ``RuntimeError`` with the log tail if it does not come
    up within ``timeout`` seconds.

    Environment:
      * ``OLLAMA_KEEP_ALIVE=-1``: keep the model in VRAM forever (no reload between questions).
      * ``OLLAMA_FLASH_ATTENTION=1``: smaller KV-cache footprint and faster long prompts.
        The T4 is compute capability 7.5 (Turing); llama.cpp's flash-attention
        kernels support it, and Ollama silently falls back where they do not.
      * ``OLLAMA_NUM_PARALLEL=1`` / ``OLLAMA_MAX_LOADED_MODELS=1``: one KV cache and one
        model resident.  Colab has ~12.7 GB host RAM; extra parallel slots and models
        contributed to an observed OOM-kill of the runner.
    """
    host = _strip_scheme(host)
    if _server_up(host):
        print(f"Ollama already running at http://{host}")
        return None
    binary = shutil.which("ollama")
    if binary is None:
        raise RuntimeError("`ollama` binary not found; call install_ollama() first.")

    log_path = log_path or _default_log_path()
    env = {
        **os.environ,
        "OLLAMA_HOST": host,
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        **(extra_env or {}),
    }
    log = open(log_path, "ab")  # noqa: SIM115 - handle must outlive this function (server child)
    proc = subprocess.Popen(
        [binary, "serve"],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,  # detached like nohup: survives cell interrupts
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_up(host):
            print(f"Ollama server up at http://{host} (pid {proc.pid}, log {log_path})")
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    status = "exited with code " + str(proc.returncode) if proc.poll() is not None else "timed out"
    log.close()  # the server never came up, so nothing else will write to it
    raise RuntimeError(
        f"Ollama server failed to start ({status}) at http://{host}.\n"
        f"Last lines of {log_path}:\n{_tail(log_path)}"
    )


def _runner_anon_rss(
    server_pid: int | None = None, proc_root: str = "/proc"
) -> list[tuple[int, int]]:
    """(pid, anonymous RSS bytes) for Ollama model-runner processes (Linux /proc).

    With ``server_pid``, only runners spawned by that ``ollama serve`` are returned, so two
    servers (one per GPU) never restart each other's runner mid-request.
    """
    found = []
    for status in Path(proc_root).glob("[0-9]*/status"):
        try:
            text = status.read_text()
        except OSError:
            continue
        name = text.split("\n", 1)[0].removeprefix("Name:\t")
        if name not in ("llama-server", "ollama_llama_se"):
            if name != "ollama":
                continue
            try:  # newer Ollama runs the model as `ollama runner ...`
                cmdline = (status.parent / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if b"runner" not in cmdline[1:2]:
                continue
        if server_pid is not None:
            parent = re.search(r"PPid:\s+(\d+)", text)
            if not parent or int(parent.group(1)) != server_pid:
                continue
        match = re.search(r"RssAnon:\s+(\d+) kB", text)
        if match:
            found.append((int(status.parent.name), int(match.group(1)) * 1024))
    return found


def start_runner_watchdog(
    max_anon_gb: float = 4.0, interval: float = 5.0, server_pid: int | None = None
) -> threading.Thread:
    """Restart Ollama's model runner when its anonymous memory grows past ``max_anon_gb``.

    On Colab (12.7 GB host RAM) we measured the runner's anonymous memory growing by
    ~100 MB per request until the kernel OOM-killer took down the notebook.  Killing the
    runner is safe: Ollama respawns it on the next request, and because the GGUF is
    page-cached and mmap'd the reload takes seconds.  The watchdog only acts while this
    process has no request in flight, so no answer is interrupted.
    """
    from local_agentic_rag.llm.ollama import inflight_requests

    limit = int(max_anon_gb * 1024**3)

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                for pid, anon in _runner_anon_rss(server_pid):
                    if anon > limit and inflight_requests() == 0:
                        os.kill(pid, signal.SIGTERM)
                        print(
                            f"[watchdog] restarted Ollama runner (anon RSS {anon / 1024**3:.1f} GB)"
                        )
            except Exception as exc:  # the watchdog must never crash the notebook
                print(f"[watchdog] error: {exc}")

    thread = threading.Thread(target=loop, name="ollama-runner-watchdog", daemon=True)
    thread.start()
    return thread


def setup_llm(
    model: str = "qwen2.5:7b-instruct",
    num_ctx: int = 8192,
    host: str = DEFAULT_HOST,
    watchdog: bool = True,
    **llm_kwargs: Any,
) -> OllamaLLM:
    """Install Ollama, start the server, pull ``model`` and load it into VRAM.

    Idempotent: re-running the cell skips whatever is already done.
    """
    from local_agentic_rag.llm.ollama import OllamaLLM

    with timed("ollama_install"):
        install_ollama()
    with timed("ollama_start"):
        server = start_ollama(host)
    llm = OllamaLLM(model, host=f"http://{_strip_scheme(host)}", num_ctx=num_ctx, **llm_kwargs)
    with timed("model_pull_and_warmup"):
        llm.ensure_ready()
    if watchdog and Path("/proc").is_dir():
        # Scope the watchdog to our own server when we started it (None: any runner).
        start_runner_watchdog(server_pid=server.pid if server else None)
    print(f"LLM ready: {model} (num_ctx={num_ctx})")
    return llm


# --------------------------------------------------------------------------- rendering

# Explicit colours everywhere: Colab's dark theme would otherwise inherit light
# text onto our light card (or vice versa).  The card is a fixed neutral light
# surface that reads well on both themes.
_C = {
    "bg": "#f8f9fb",
    "fg": "#1f2328",
    "muted": "#57606a",
    "border": "#d0d7de",
    "panel": "#ffffff",
    "link": "#0b57d0",
    "code_bg": "#eef1f4",
}
_BADGES = {
    "verified": ("#e6f4ea", "#137333", "#a8dab5", "Verified: every claim supported"),
    "partial": ("#fef7e0", "#8a5300", "#f6d58a", "Partially unverified"),
    "unverified": ("#eef1f4", "#57606a", "#d0d7de", "Not verified"),
    "refused": ("#e8f0fe", "#174ea6", "#aecbfa", "Not found in document"),
    "error": ("#fce8e6", "#a50e0e", "#f6aea9", "Error"),
}
_MARKER_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def _badge(kind: str, label: str | None = None) -> str:
    bg, fg, border, default = _BADGES[kind]
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f"background:{bg};color:{fg};border:1px solid {border};font-size:12px;"
        f'font-weight:600;margin-right:6px">{html.escape(label or default)}</span>'
    )


def _answer_status(answer: Answer) -> str:
    if answer.error:
        return "error"
    if answer.refused:
        return "refused"
    if answer.grounded is True:
        return "verified"
    if answer.grounded is False:
        return "partial"
    return "unverified"


def _linkify_markers(escaped_text: str, valid: set[int], uid: str) -> str:
    """Turn ``[1]`` / ``[1, 3]`` in already-escaped text into superscript links."""

    def repl(match: re.Match[str]) -> str:
        parts = []
        for token in re.split(r"[,;\s]+", match.group(1)):
            if not token.isdigit():
                continue
            n = int(token)
            if n in valid:
                parts.append(
                    f'<a href="#cite-{uid}-{n}" style="color:{_C["link"]};text-decoration:none">{n}</a>'
                )
            else:
                parts.append(str(n))
        return f'<sup style="font-size:0.75em">[{",".join(parts)}]</sup>'

    return _MARKER_RE.sub(repl, escaped_text)


def render_citations_html(citations: list[Citation], uid: str = "") -> str:
    if not citations:
        return f'<div style="color:{_C["muted"]};font-size:13px">No citations.</div>'
    seen: set[int] = set()
    items = []
    for c in sorted(citations, key=lambda c: c.marker):
        if c.marker in seen:
            continue
        seen.add(c.marker)
        snippet = html.escape(" ".join(c.snippet.split()))
        items.append(
            f'<li id="cite-{uid}-{c.marker}" style="margin:4px 0;color:{_C["fg"]}">'
            f'<b>[{c.marker}]</b> <span style="color:{_C["muted"]}">p.&nbsp;{c.page}'
            f"{' · ' + html.escape(c.doc_id) if c.doc_id else ''}</span><br>"
            f'<span style="font-size:13px">&ldquo;{snippet}&hellip;&rdquo;</span></li>'
        )
    return f'<ol style="list-style:none;padding-left:0;margin:4px 0">{"".join(items)}</ol>'


def _compact_json(detail: dict[str, Any], limit: int = 240) -> str:
    text = json.dumps(detail, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_trace_html(trace: list[TraceStep], open_: bool = False) -> str:
    if not trace:
        return ""
    cell = f"padding:4px 8px;border-bottom:1px solid {_C['border']};vertical-align:top"
    rows = "".join(
        f'<tr><td style="{cell}">{i}</td><td style="{cell}"><b>{html.escape(t.step)}</b></td>'
        f'<td style="{cell};text-align:right">{t.seconds:.2f}s</td>'
        f'<td style="{cell};font-family:monospace;font-size:12px;word-break:break-word">'
        f"{html.escape(_compact_json(t.detail))}</td></tr>"
        for i, t in enumerate(trace, 1)
    )
    head = f"{cell};text-align:left;color:{_C['muted']}"
    return (
        f'<details{" open" if open_ else ""} style="margin-top:10px">'
        f'<summary style="cursor:pointer;color:{_C["fg"]};font-weight:600">'
        f"Agent trace ({len(trace)} steps)</summary>"
        f'<table style="border-collapse:collapse;width:100%;margin-top:6px;font-size:13px;'
        f'background:{_C["panel"]};color:{_C["fg"]}">'
        f'<tr><th style="{head}">#</th><th style="{head}">step</th>'
        f'<th style="{head};text-align:right">time</th><th style="{head}">detail</th></tr>'
        f"{rows}</table></details>"
    )


def render_answer_html(answer: Answer, uid: str | None = None) -> str:
    """Self-contained HTML card for one :class:`~local_agentic_rag.schema.Answer`.

    All document-derived text (answer, snippets, claims, trace) is HTML-escaped.
    """
    uid = uid or uuid.uuid4().hex[:8]
    status = _answer_status(answer)
    valid = {c.marker for c in answer.citations}

    body = html.escape(answer.answer or "").replace("\n", "<br>")
    if status != "refused":
        body = _linkify_markers(body, valid, uid)
    text_style = f"font-size:15px;line-height:1.55;color:{_C['fg']}"
    if status == "refused":
        text_style += f";font-style:italic;color:{_BADGES['refused'][1]}"

    badges = _badge(status)
    if answer.cited_pages:
        badges += _badge("unverified", "Pages " + ", ".join(map(str, answer.cited_pages)))

    extra = ""
    if status == "partial" and answer.unsupported_claims:
        claims = "".join(f"<li>{html.escape(c)}</li>" for c in answer.unsupported_claims)
        extra += (
            f'<div style="margin-top:8px;padding:6px 10px;background:{_BADGES["partial"][0]};'
            f'color:{_BADGES["partial"][1]};border-radius:6px;font-size:13px">'
            f"The verifier could not find support for:<ul style='margin:4px 0'>{claims}</ul></div>"
        )
    if answer.error:
        extra += (
            f'<div style="margin-top:8px;padding:6px 10px;background:{_BADGES["error"][0]};'
            f'color:{_BADGES["error"][1]};border-radius:6px;font-size:13px;font-family:monospace">'
            f"{html.escape(answer.error)}</div>"
        )

    citations = ""
    if answer.citations:
        citations = (
            f'<div style="margin-top:12px;padding-top:8px;border-top:1px solid {_C["border"]}">'
            f'<div style="font-weight:600;color:{_C["fg"]};margin-bottom:2px">Sources</div>'
            f"{render_citations_html(answer.citations, uid)}</div>"
        )

    meta_parts = [f"{answer.latency_sec:.1f}s", f"{answer.llm_calls} LLM calls"]
    if answer.retrieval_rounds > 1:
        meta_parts.append(f"{answer.retrieval_rounds} retrieval rounds")
    if answer.answer_attempts > 1:
        meta_parts.append(f"{answer.answer_attempts} answer attempts")
    meta = (
        f'<div style="margin-top:10px;font-size:12px;color:{_C["muted"]}">'
        f"{html.escape(' · '.join(meta_parts))}</div>"
    )
    question = (
        f'<div style="font-size:13px;color:{_C["muted"]};margin-bottom:6px">'
        f"Q: {html.escape(answer.question)}</div>"
        if answer.question
        else ""
    )
    return (
        f'<div class="lar-answer" style="background:{_C["bg"]};color:{_C["fg"]};'
        f"border:1px solid {_C['border']};border-radius:10px;padding:14px 16px;margin:6px 0;"
        f'font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:960px">'
        f"{question}<div style='margin-bottom:8px'>{badges}</div>"
        f'<div style="{text_style}">{body}</div>{extra}{citations}'
        f"{render_trace_html(answer.trace)}{meta}</div>"
    )


def display_answer(answer: Answer) -> None:
    """Render an answer card in a notebook (plain text fallback outside IPython)."""
    try:
        from IPython.display import HTML, display
    except ImportError:
        print(answer.answer)
        if answer.cited_pages:
            print(f"(pages {answer.cited_pages})")
        return
    display(HTML(render_answer_html(answer)))


def format_ingest_report(report: IngestReport) -> str:
    """One readable block for an :class:`~local_agentic_rag.schema.IngestReport`."""
    lines = [
        f"{report.title or report.doc_id}  (doc_id={report.doc_id})",
        f"  pages {report.num_pages} | chunks {report.num_chunks} (tables {report.num_table_chunks}) | "
        f"{report.seconds:.1f}s{' (cached)' if report.cached else ''}",
    ]
    if report.ocr_pages:
        lines.append(f"  OCR pages: {report.ocr_pages}")
    if report.empty_pages:
        lines.append(f"  empty pages (no text found): {report.empty_pages}")
    lines.extend(f"  warning: {w}" for w in report.warnings)
    return "\n".join(lines)


def print_ingest_report(report: IngestReport) -> None:
    print(format_ingest_report(report))


def ingest_progress(min_interval: float = 2.0):
    """``pipe.ingest(path, progress=ingest_progress())``: one updating status line.

    Long PDFs take minutes; this shows the stage, the page reached and an ETA.
    """
    last = [0.0]

    def show(p) -> None:
        now = time.perf_counter()
        if p.stage not in ("done", "finalising") and now - last[0] < min_interval:
            return
        last[0] = now
        eta = f", about {p.eta / 60:.0f} min left" if p.eta and p.eta >= 90 else ""
        end = "\n" if p.stage == "done" else ""
        print(f"\r{p.message}{eta}" + " " * 20, end=end, flush=True)

    return show


def launch_web_app(
    pipe,
    library_dir: str = "/content/library",
    port: int = 8000,
    max_concurrent_asks: int = 1,
):
    """Start the document-library web app in a background thread, sharing ``pipe``'s models.

    In Colab the page opens inside the notebook through Colab's private port proxy (only you
    can reach it; nothing is public).
    Returns ``(library, server)``; ``server.should_exit = True`` stops it.
    """
    import uvicorn

    from local_agentic_rag.library import Library
    from local_agentic_rag.server.app import create_app

    library = Library(library_dir, pipeline=pipe)
    app = create_app(
        library,
        max_concurrent_asks=max_concurrent_asks,
        upload_dir=os.path.join(library_dir, ".incoming"),
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="web-app", daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    if is_colab():
        from google.colab import output  # type: ignore[import-not-found]

        # The iframe form is Colab's recommended one; "as_window" now warns that browser
        # security changes may break it.
        output.serve_kernel_port_as_iframe(port, height=820)
    else:
        print(f"Web app running at http://127.0.0.1:{port}/")
    return library, server
