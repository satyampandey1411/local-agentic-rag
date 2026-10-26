"""Tests for the Colab helpers, the Gradio app helpers and the generated notebook."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from local_agentic_rag import colab
from local_agentic_rag.schema import Answer, Citation, IngestReport, TraceStep

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "notebooks" / "build_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "Local_Agentic_RAG.ipynb"


def make_answer(**overrides) -> Answer:
    base: dict = {
        "question": "What is <b>bias</b>?",
        "answer": "Brevity bias <script>alert(1)</script> favours short text [1]. Also [2, 3] and [9].",
        "refused": False,
        "citations": [
            Citation(
                marker=1, chunk_id="d-p3-0", doc_id="d", page=3, snippet="Short & <early> text"
            ),
            Citation(marker=2, chunk_id="d-p5-1", doc_id="d", page=5, snippet="Second passage"),
            Citation(marker=3, chunk_id="d-p7-0", doc_id="d", page=7, snippet="Third passage"),
        ],
        "grounded": True,
        "llm_calls": 4,
        "latency_sec": 7.25,
        "trace": [
            TraceStep(step="plan", detail={"output": {"q": "<x>"}}, seconds=1.2),
            TraceStep(step="retrieve", detail={"pages": [3, 5]}, seconds=0.3),
        ],
    }
    base.update(overrides)
    return Answer(**base)


# --------------------------------------------------------------------------- render_answer_html


def test_render_escapes_document_text():
    out = colab.render_answer_html(make_answer(), uid="t")
    assert "<script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert "Short &amp; &lt;early&gt; text" in out
    assert "What is &lt;b&gt;bias&lt;/b&gt;?" in out
    assert "&lt;x&gt;" in out  # trace detail is escaped too


def test_render_turns_markers_into_superscript_links():
    out = colab.render_answer_html(make_answer(), uid="t")
    assert '<a href="#cite-t-1"' in out
    assert '<a href="#cite-t-2"' in out and '<a href="#cite-t-3"' in out
    assert 'id="cite-t-1"' in out and 'id="cite-t-3"' in out
    # [9] has no citation: shown as a superscript but not linked
    assert "#cite-t-9" not in out
    assert re.search(r"<sup[^>]*>\[9\]</sup>", out)
    assert "p.&nbsp;3" in out and "p.&nbsp;7" in out


@pytest.mark.parametrize(
    ("overrides", "expected", "absent"),
    [
        ({"grounded": True}, "Verified", "Partially unverified"),
        (
            {"grounded": False, "unsupported_claims": ["GDP rose <10%>"]},
            "Partially unverified",
            "Verified:",
        ),
        ({"grounded": None}, "Not verified", "Partially unverified"),
        (
            {"refused": True, "citations": [], "answer": "I could not find it."},
            "Not found in document",
            "Verified",
        ),
        ({"error": "LLM error: down"}, "LLM error: down", "Verified:"),
    ],
)
def test_render_badges(overrides, expected, absent):
    out = colab.render_answer_html(make_answer(**overrides), uid="t")
    assert expected in out
    assert absent not in out


def test_render_lists_unsupported_claims_escaped():
    out = colab.render_answer_html(
        make_answer(grounded=False, unsupported_claims=["GDP rose <10%>"]), uid="t"
    )
    assert "GDP rose &lt;10%&gt;" in out


def test_render_trace_and_meta():
    out = colab.render_answer_html(make_answer(), uid="t")
    assert "<details" in out and "Agent trace (2 steps)" in out
    assert "7.2s" in out or "7.3s" in out
    assert "4 LLM calls" in out
    # explicit colours so dark/light Colab themes both render the card readably
    assert "background:#f8f9fb" in out and "color:#1f2328" in out


def test_render_unique_ids_by_default():
    a = make_answer()
    first, second = colab.render_answer_html(a), colab.render_answer_html(a)
    ids = [re.search(r'id="(cite-[^"]+)"', h).group(1) for h in (first, second)]
    assert ids[0] != ids[1]


def test_display_answer_fallback_without_ipython(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    colab.display_answer(make_answer())
    assert "Brevity bias" in capsys.readouterr().out


def test_format_ingest_report():
    r = IngestReport(
        doc_id="d",
        title="Doc",
        num_pages=3,
        num_chunks=10,
        num_table_chunks=2,
        ocr_pages=[2],
        empty_pages=[3],
        seconds=1.5,
        cached=True,
        warnings=["page 3 empty"],
    )
    text = colab.format_ingest_report(r)
    assert "chunks 10 (tables 2)" in text and "(cached)" in text
    assert "OCR pages: [2]" in text and "warning: page 3 empty" in text


# --------------------------------------------------------------------------- GPU / Ollama fail-soft


def test_gpu_report_without_nvidia_smi(monkeypatch):
    real_which = colab.shutil.which
    monkeypatch.setattr(
        colab.shutil, "which", lambda name: None if name == "nvidia-smi" else real_which(name)
    )
    report = colab.gpu_report()
    assert report["gpus"] == []
    assert "T4" in report["error"]
    assert "torch_cuda_available" in report
    colab.print_gpu_report()  # must not raise


def test_vram_snapshot_never_raises(monkeypatch):
    monkeypatch.setattr(colab, "_nvidia_smi_gpus", lambda: None)
    assert isinstance(colab.vram_snapshot(), dict)


def test_nvidia_smi_parsing(monkeypatch):
    def fake_run(cmd, timeout=20.0):
        if len(cmd) > 1 and cmd[1].startswith("--query-gpu"):
            return "Tesla T4, 15360, 5821, 535.104.05, 7.5\n"
        return "| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2 |"

    monkeypatch.setattr(colab, "_run", fake_run)
    report = colab.gpu_report()
    gpu = report["gpus"][0]
    assert gpu["name"] == "Tesla T4" and gpu["memory_total_mb"] == 15360
    assert gpu["compute_capability"] == "7.5"
    assert report["cuda_version"] == "12.2" and report["error"] is None
    assert colab.vram_snapshot()["gpu_used_mb"] == 5821


def test_start_ollama_returns_none_when_already_running(monkeypatch):
    monkeypatch.setattr(colab, "_server_up", lambda host, timeout=2.0: True)
    assert colab.start_ollama("http://127.0.0.1:11434") is None


def test_start_ollama_without_binary_raises(monkeypatch):
    monkeypatch.setattr(colab, "_server_up", lambda host, timeout=2.0: False)
    monkeypatch.setattr(colab.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="install_ollama"):
        colab.start_ollama()


def test_start_ollama_failure_includes_log_tail(monkeypatch, tmp_path):
    fake = tmp_path / "ollama"
    fake.write_text("#!/bin/sh\necho 'Error: listen tcp: address already in use'\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setattr(colab, "_server_up", lambda host, timeout=2.0: False)
    monkeypatch.setattr(colab.shutil, "which", lambda name: str(fake) if name == "ollama" else None)
    with pytest.raises(RuntimeError, match="address already in use"):
        colab.start_ollama(log_path=str(tmp_path / "ollama.log"), timeout=10)


def test_install_ollama_skips_when_present(monkeypatch):
    monkeypatch.setattr(
        colab.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None
    )
    monkeypatch.setattr(
        colab.subprocess, "run", lambda *a, **k: pytest.fail("should not run anything")
    )
    assert colab.install_ollama(verbose=False) == "/usr/bin/ollama"


def test_runner_scan_can_be_scoped_to_one_server(tmp_path):
    def proc(pid, name, ppid, anon_kb, cmdline=b""):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "status").write_text(f"Name:\t{name}\nPPid:\t{ppid}\nRssAnon:\t{anon_kb} kB\n")
        (d / "cmdline").write_bytes(cmdline)

    proc(11, "ollama", 1, 10, b"ollama\0serve\0")  # a server, not a runner
    proc(21, "ollama", 11, 5000, b"ollama\0runner\0--model\0x\0")  # runner of server 11
    proc(22, "llama-server", 12, 7000)  # runner of another server (other GPU)
    all_runners = dict(colab._runner_anon_rss(proc_root=str(tmp_path)))
    assert all_runners == {21: 5000 * 1024, 22: 7000 * 1024}
    assert dict(colab._runner_anon_rss(server_pid=11, proc_root=str(tmp_path))) == {21: 5000 * 1024}


def test_timed_records(capsys):
    with colab.timed("unit_step"):
        pass
    assert "unit_step" in colab.TIMINGS
    colab.print_resource_report()
    assert "unit_step" in capsys.readouterr().out


# --------------------------------------------------------------------------- app helpers


def test_app_helpers():
    from local_agentic_rag import app

    assert app.history_pairs([["q", "a"]]) == [("q", "a")]
    assert app.history_pairs(None) == []
    assert app.selected_doc(app.ALL_DOCS) is None and app.selected_doc("d") == "d"
    text = app.chat_reply_text(make_answer(grounded=False))
    assert "Pages: 3, 5, 7" in text and "could not be verified" in text
    side = app.side_panel_html(make_answer())
    assert "verified" in side and "Agent trace" in side and "&lt;early&gt;" in side
    assert "could not be processed" in app.format_ingest_status(
        [], ["x.pdf: could not be processed"]
    )


@pytest.mark.skipif(importlib.util.find_spec("gradio") is None, reason="gradio not installed")
def test_build_chat_ui_with_fake_pipeline():
    from local_agentic_rag import app

    class FakePipe:
        indexes: dict = {}

    demo = app.build_chat_ui(FakePipe())
    assert demo is not None


# --------------------------------------------------------------------------- notebook


@pytest.fixture(scope="module")
def built_notebook_path(tmp_path_factory):
    """Build into a temp dir so running the tests never rewrites the committed notebook."""
    pytest.importorskip("nbformat")
    out = tmp_path_factory.mktemp("nb") / NOTEBOOK.name
    subprocess.run([sys.executable, str(BUILDER), str(out)], check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def notebook(built_notebook_path):
    import nbformat

    return nbformat.read(built_notebook_path, as_version=4)


def test_committed_notebook_is_up_to_date(built_notebook_path):
    # Fails when build_notebook.py changed but notebooks/Local_Agentic_RAG.ipynb was not rebuilt.
    assert built_notebook_path.read_text() == NOTEBOOK.read_text(), (
        "run: python notebooks/build_notebook.py"
    )


def test_notebook_is_valid(notebook):
    import nbformat

    nbformat.validate(notebook)
    meta = notebook.metadata
    assert meta["accelerator"] == "GPU"
    assert meta["colab"]["gpuType"] == "T4"
    assert meta["kernelspec"]["name"] == "python3"
    json.loads(NOTEBOOK.read_text())  # the committed file is plain JSON too


def _strip_magics(source: str) -> str:
    """Replace ``!shell`` / ``%magic`` lines with ``pass`` (keeping indentation)."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            line = line[: len(line) - len(stripped)] + "pass"
        lines.append(line)
    return "\n".join(lines)


def test_every_code_cell_compiles(notebook):
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert len(cells) >= 10
    for i, cell in enumerate(cells):
        compile(_strip_magics(cell.source), f"<cell {i}>", "exec")


def test_notebook_covers_required_sections(notebook):
    text = "\n".join(c.source for c in notebook.cells)
    for needle in [
        "nvidia-smi",
        "REPO_URL",
        "REPO_ZIP",
        "[colab]",
        "setup_llm(",
        "pipe.load_models()",
        "pipe.ingest(",
        "history=",
        "display_answer(",
        "launch_chat_ui(",
        "fetch_pdfs.py",
        "retrieval_ablation(",
        "answer_ablation(",
        "summary_table_markdown",
        "per_category_markdown",
        "LIMIT = None",
        "files.download",
        "print_resource_report",
        "tesseract-ocr",
        "num_ctx=4096",
    ]:
        assert needle in text, needle
    assert "pip install -q torch" not in text and "pip install numpy" not in text
