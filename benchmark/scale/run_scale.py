"""
Scale benchmark: ingestion, storage, memory, retrieval and (optionally) answers for PDFs of
100 to 10,000 pages.  Every number in the output is measured in this run; tiers that were
not run are reported as "not run", never estimated.

    # retrieval-only (no LLM), CPU or GPU
    python benchmark/scale/run_scale.py --tiers 100 250 1000 --corpus-dir local/scale

    # everything, on a T4 with Ollama serving qwen2.5:7b-instruct
    python benchmark/scale/run_scale.py --tiers all --llm ollama:qwen2.5:7b-instruct \
        --recovery --out benchmark/results/scale

Per tier it records:
    ingest    wall seconds per stage (parse, chunk+embed, BM25), pages/s, chunks, warnings
    memory    peak RSS of this process + parse workers (sampled every 0.2 s), peak CUDA memory
    storage   bytes on disk of the parsed pages and the index, PDF size
    load      cold index load seconds (what a restart costs)
    retrieval page recall@1/5/10, MRR, latency p50/p95 per question type and document depth
    answers   (with --llm) accuracy, refusal of unanswerable questions, citation page
              precision/recall, latency p50/p95, LLM calls
    recovery  (with --recovery) a worker crash on one page plus an interrupted job resumed
              from its checkpoint, checked to produce the same index as a clean run
Results: ``<out>/scale-<pages>.json`` per tier and ``<out>/scale_summary.md``.
A tier whose JSON exists is skipped unless ``--force``, so a long run can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from local_agentic_rag.config import RAGConfig  # noqa: E402
from local_agentic_rag.eval.metrics import all_facts_present  # noqa: E402
from local_agentic_rag.ingest.streaming import (  # noqa: E402
    IngestCancelled,
    ingest_pdf,
    load_index,
    store_dirs,
)
from local_agentic_rag.pipeline import RAGPipeline  # noqa: E402

TIERS = [100, 250, 1000, 2500, 5000, 10000]


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


class PeakRSS:
    """Samples RSS of this process and all its children (parse workers)."""

    def __init__(self, interval: float = 0.2) -> None:
        import psutil

        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self.peak_children = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            main = self.proc.memory_info().rss
            kids = sum(c.memory_info().rss for c in self.proc.children(recursive=True))
        except Exception:
            return
        self.peak = max(self.peak, main + kids)
        self.peak_children = max(self.peak_children, kids)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> PeakRSS:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


def cuda_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def cuda_peak_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return None


def gpu_used_mb() -> list[int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return [int(x) for x in out.split()]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values else None


def index_digest(index) -> str:
    h = hashlib.sha256()
    for c in index.chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text.encode())
    h.update(np.asarray(index.embeddings, dtype=np.float16).tobytes())
    return h.hexdigest()[:16]


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "tesseract": shutil.which("tesseract") is not None,
    }
    try:
        import psutil

        env["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import torch

        env["cuda"] = torch.cuda.is_available()
        if env["cuda"]:
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["cuda"] = False
    return env


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def ensure_corpus(corpus_dir: Path, pages: int, seed: int) -> tuple[Path, list[dict]]:
    pdf = corpus_dir / f"scale-{pages}.pdf"
    qfile = corpus_dir / f"scale-{pages}.questions.jsonl"
    if not (pdf.is_file() and qfile.is_file()):
        corpus_dir.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(REPO / "benchmark/scale/make_corpus.py"),
                "--pages",
                str(pages),
                "--out",
                str(corpus_dir),
                "--seed",
                str(seed),
            ],
            check=True,
        )
        print(f"  generated {pdf.name} in {time.perf_counter() - t:.0f}s", flush=True)
    questions = [json.loads(x) for x in qfile.read_text().splitlines() if x.strip()]
    return pdf, questions


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_ingest(pipe: RAGPipeline, pdf: Path, store: Path, workers: int | None) -> dict:
    shutil.rmtree(store, ignore_errors=True)
    cuda_reset()
    stages: dict[str, float] = {}
    marks = {"t0": time.perf_counter()}

    def progress(p) -> None:
        stages.setdefault(f"first_{p.stage}", round(time.perf_counter() - marks["t0"], 2))

    with PeakRSS() as rss:
        index, report = ingest_pdf(
            pdf, store, pipe.config, pipe.embedder, progress=progress, workers=workers
        )
    meta = index.meta
    parse_dir, index_dir = store_dirs(
        store, meta["sha256"], pipe.config, use_ocr=shutil.which("tesseract") is not None
    )
    t = time.perf_counter()
    reloaded = load_index(index_dir)
    load_s = time.perf_counter() - t
    assert len(reloaded) == len(index)
    return {
        "index": index,
        "result": {
            "wall_seconds": round(report.seconds, 2),
            "parse_seconds": round(meta["parse_seconds"], 2),
            "index_seconds": round(meta["index_seconds"], 2),
            "embed_seconds": round(meta["embed_seconds"], 2),
            "bm25_seconds": round(meta["bm25_seconds"], 2),
            "pages_per_second": round(report.num_pages / max(report.seconds, 1e-9), 2),
            "num_pages": report.num_pages,
            "num_chunks": report.num_chunks,
            "num_table_chunks": report.num_table_chunks,
            "chunks_per_second_embedded": round(
                report.num_chunks / max(meta["embed_seconds"], 1e-9), 1
            ),
            "ocr_pages": len(report.ocr_pages),
            "empty_pages": len(report.empty_pages),
            "warnings": report.warnings[:10],
            "peak_rss_mb": round(rss.peak / 2**20, 1),
            "peak_worker_rss_mb": round(rss.peak_children / 2**20, 1),
            "peak_cuda_mb": cuda_peak_mb(),
            "storage": {
                "pdf_bytes": pdf.stat().st_size,
                "parsed_bytes": dir_bytes(parse_dir),
                "index_bytes": dir_bytes(index_dir),
            },
            "cold_load_seconds": round(load_s, 2),
            "index_digest": index_digest(index),
            "stage_marks": stages,
        },
    }


def _gold_rank(pages: list[int], gold: set[int]) -> int | None:
    for rank, p in enumerate(pages, start=1):
        if p in gold:
            return rank
    return None


def run_retrieval(pipe: RAGPipeline, index, questions: list[dict]) -> dict:
    retriever = pipe.retriever()
    rows = []
    for q in questions:
        if not q.get("answerable", True):
            continue
        t = time.perf_counter()
        hits = retriever.retrieve([q["question"]], [index], top_k=10)
        latency = time.perf_counter() - t
        pages = list(dict.fromkeys(h.chunk.page for h in hits))
        rank = _gold_rank(pages, set(q["gold_pages"]))
        rows.append(
            {
                "q_id": q["q_id"],
                "type": q["type"],
                "depth": q.get("depth"),
                "rank": rank,
                "latency": latency,
                "pages": pages[:10],
            }
        )

    def summary(sub: list[dict]) -> dict:
        ranks = [r["rank"] for r in sub]
        return {
            "n": len(sub),
            "recall@1": round(np.mean([r is not None and r <= 1 for r in ranks]), 3),
            "recall@5": round(np.mean([r is not None and r <= 5 for r in ranks]), 3),
            "recall@10": round(np.mean([r is not None for r in ranks]), 3),
            "mrr": round(np.mean([1 / r if r else 0.0 for r in ranks]), 3),
            "latency_p50": pct([r["latency"] for r in sub], 50),
            "latency_p95": pct([r["latency"] for r in sub], 95),
        }

    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        by_type[t] = summary([r for r in rows if r["type"] == t])
    by_depth = {}
    for lo in (0.0, 0.25, 0.5, 0.75):
        sub = [r for r in rows if r["depth"] is not None and lo <= r["depth"] < lo + 0.25 + 1e-9]
        if sub:
            by_depth[f"{lo:.2f}-{lo + 0.25:.2f}"] = summary(sub)
    return {
        "overall": summary(rows),
        "by_type": by_type,
        "by_depth": by_depth,
        "per_question": rows,
    }


def run_answers(pipe: RAGPipeline, index, questions: list[dict]) -> dict:
    pipe.indexes[index.doc_id] = index
    rows = []
    for q in questions:
        ans = pipe.ask(q["question"], doc_id=index.doc_id)
        answerable = q.get("answerable", True)
        if answerable:
            correct = (not ans.refused) and all_facts_present(ans.answer, q["answer_keys"])
        else:
            correct = ans.refused
        cited = set(ans.cited_pages)
        gold = set(q.get("gold_pages") or [])
        rows.append(
            {
                "q_id": q["q_id"],
                "type": q["type"],
                "answerable": answerable,
                "correct": bool(correct),
                "refused": ans.refused,
                "cit_precision": (len(cited & gold) / len(cited)) if (cited and gold) else None,
                "cit_recall": (1.0 if cited & gold else 0.0) if gold and answerable else None,
                "latency": ans.latency_sec,
                "llm_calls": ans.llm_calls,
                "error": ans.error,
                "answer": ans.answer[:400],
                "cited_pages": sorted(cited),
            }
        )

    def mean(key, sub):
        vals = [r[key] for r in sub if r[key] is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    by_type = {
        t: {"n": len(s), "accuracy": mean("correct", s)}
        for t in sorted({r["type"] for r in rows})
        for s in [[r for r in rows if r["type"] == t]]
    }
    return {
        "accuracy": mean("correct", rows),
        "answerable_accuracy": mean("correct", answerable),
        "unanswerable_refused": mean("correct", unanswerable),
        "false_refusals": sum(r["refused"] for r in answerable),
        "citation_precision": mean("cit_precision", rows),
        "citation_recall": mean("cit_recall", rows),
        "latency_p50": pct([r["latency"] for r in rows], 50),
        "latency_p95": pct([r["latency"] for r in rows], 95),
        "mean_llm_calls": mean("llm_calls", rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "n": len(rows),
        "by_type": by_type,
        "per_question": rows,
    }


def run_recovery(
    pipe: RAGPipeline,
    pdf: Path,
    store: Path,
    clean_digest: str,
    num_pages: int,
    workers: int | None,
) -> dict:
    """Crash one page's worker, interrupt the job half-way, resume, compare with clean."""
    shutil.rmtree(store, ignore_errors=True)
    crash_page = max(2, num_pages // 3)
    os.environ["LOCAL_RAG_FAULT_PAGES"] = f"crash:{crash_page}"
    stop_at = num_pages // 2
    state = {"stopped_at": None}

    def progress(p) -> None:
        if p.stage == "reading" and p.pages_done >= stop_at and state["stopped_at"] is None:
            state["stopped_at"] = p.pages_done

    t = time.perf_counter()
    try:
        ingest_pdf(
            pdf,
            store,
            pipe.config,
            pipe.embedder,
            progress=progress,
            should_stop=lambda: state["stopped_at"] is not None,
            workers=workers,
        )
        interrupted = False
    except IngestCancelled:
        interrupted = True
    first_s = time.perf_counter() - t
    os.environ.pop("LOCAL_RAG_FAULT_PAGES")
    t = time.perf_counter()
    index, report = ingest_pdf(pdf, store, pipe.config, pipe.embedder, workers=workers)
    resume_s = time.perf_counter() - t
    crashed_warning = any(f"page {crash_page}: parse error" in w for w in report.warnings)
    digest = index_digest(index)
    return {
        "crash_page": crash_page,
        "worker_crash_isolated": crashed_warning,
        "interrupted_after_pages": state["stopped_at"],
        "interrupted": interrupted,
        "resumed_from_page": report.resumed_from_page,
        "seconds_before_interrupt": round(first_s, 2),
        "seconds_to_finish_after_resume": round(resume_s, 2),
        "pages_lost_to_crash": len([p for p in report.empty_pages if p == crash_page]),
        # The crashed page is empty in the recovered run, so the digest differs from the
        # clean run by exactly that page; the rest must be identical.
        "identical_except_crash_page": _same_except_page(index, clean_digest, crash_page),
        "recovered_digest": digest,
    }


_CLEAN_INDEX: dict[str, Any] = {}


def _same_except_page(index, clean_digest: str, page: int) -> bool:
    clean = _CLEAN_INDEX.get(clean_digest)
    if clean is None:
        return False
    a = [(c.page, c.text) for c in index.chunks if c.page != page]
    b = [(c.page, c.text) for c in clean.chunks if c.page != page]
    return a == b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def summary_markdown(results: list[dict], requested: list[int]) -> str:
    done = {r["pages"]: r for r in results}
    lines = [
        "# Scale benchmark",
        "",
        "Every value below was measured in the run that produced it; "
        "`not run` means the tier was not attempted on that machine.",
        "",
        "| Pages | Ingest (s) | Parse / Index / BM25 (s) | Pages/s | Chunks | Peak RAM (MB) "
        "| Peak CUDA (MB) | Index on disk (MB) | Cold load (s) | Recall@5 | MRR | Retrieval p50 (ms) "
        "| Answer acc. | Refusal (unans.) | Cit. precision | Answer p50 (s) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pages in requested:
        r = done.get(pages)
        if r is None:
            lines.append(f"| {pages:,} | not run |" + " |" * 15)
            continue
        ing, ret, ans = r["ingest"], r["retrieval"]["overall"], r.get("answers") or {}

        def f(x, d=3):
            return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

        lines.append(
            f"| {pages:,} | {ing['wall_seconds']:.0f} | {ing['parse_seconds']:.0f} / "
            f"{ing['index_seconds']:.0f} / {ing['bm25_seconds']:.1f} | {ing['pages_per_second']:.1f} "
            f"| {ing['num_chunks']:,} | {ing['peak_rss_mb']:.0f} | {f(ing['peak_cuda_mb'], 0)} "
            f"| {ing['storage']['index_bytes'] / 2**20:.0f} | {ing['cold_load_seconds']:.1f} "
            f"| {ret['recall@5']:.3f} | {ret['mrr']:.3f} | {ret['latency_p50'] * 1000:.0f} "
            f"| {f(ans.get('answerable_accuracy'))} | {f(ans.get('unanswerable_refused'))} "
            f"| {f(ans.get('citation_precision'))} | {f(ans.get('latency_p50'), 1)} |"
        )
    env = results[0]["environment"] if results else {}
    lines += ["", f"Environment: `{json.dumps(env)}`"]
    rec = [r for r in results if r.get("recovery")]
    if rec:
        lines += [
            "",
            "## Failure recovery",
            "",
            "| Pages | Worker crash isolated | Interrupted at page | Resumed from | "
            "Same index (except crashed page) |",
            "|---:|---|---:|---:|---|",
        ]
        for r in rec:
            x = r["recovery"]
            lines.append(
                f"| {r['pages']:,} | {x['worker_crash_isolated']} | "
                f"{x['interrupted_after_pages']} | {x['resumed_from_page']} | "
                f"{x['identical_except_crash_page']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tiers", nargs="+", default=["100", "250", "1000"])
    ap.add_argument("--corpus-dir", default=str(REPO / "local/scale/corpus"))
    ap.add_argument("--store-dir", default=str(REPO / "local/scale/store"))
    ap.add_argument("--out", default=str(REPO / "local/scale/results"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--llm", default=None, help="e.g. ollama:qwen2.5:7b-instruct (answers)")
    ap.add_argument("--answer-limit", type=int, default=None, help="questions per tier (answers)")
    ap.add_argument("--recovery", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--keep-store", action="store_true", help="keep indexes (disk) after a tier")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", help="RAGConfig overrides key=value (smoke tests)")
    ap.add_argument("--gpu", default=None, help="pin to one GPU (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--ollama-host", default="127.0.0.1:11434")
    ap.add_argument(
        "--setup-ollama", action="store_true", help="install/start Ollama (Colab/Kaggle)"
    )
    args = ap.parse_args(argv)

    tiers = TIERS if args.tiers == ["all"] else [int(t) for t in args.tiers]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.gpu is not None:  # before torch or Ollama start: pins both to one GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    llm = None
    if args.llm and args.setup_ollama:
        from local_agentic_rag.colab import setup_llm

        llm = setup_llm(args.llm.partition(":")[2], num_ctx=8192, host=args.ollama_host)
    elif args.llm:
        from local_agentic_rag.llm.ollama import OllamaLLM

        llm = OllamaLLM(args.llm.partition(":")[2], host=f"http://{args.ollama_host}")
        llm.ensure_ready()
    overrides = {}
    for item in args.set or []:  # e.g. --set embedder_model=BAAI/bge-small-en-v1.5
        key, _, value = item.partition("=")
        overrides[key] = (
            None
            if value == "None"
            else type(getattr(RAGConfig(), key))(value)
            if getattr(RAGConfig(), key) is not None
            else value
        )
    pipe = RAGPipeline(RAGConfig().with_overrides(**overrides), llm=llm, cache_dir=None)
    t = time.perf_counter()
    pipe.load_models()
    model_load_s = time.perf_counter() - t
    env = environment()
    results = []
    for pages in tiers:
        path = out / f"scale-{pages}.json"
        if path.is_file() and not args.force:
            results.append(json.loads(path.read_text()))
            print(f"[{pages}] already done -> {path}", flush=True)
            continue
        print(f"[{pages}] corpus", flush=True)
        pdf, questions = ensure_corpus(Path(args.corpus_dir), pages, args.seed)
        store = Path(args.store_dir) / f"tier-{pages}"
        print(f"[{pages}] ingest {pdf.name} ({pdf.stat().st_size / 2**20:.0f} MB)", flush=True)
        ing = run_ingest(pipe, pdf, store, args.workers)
        index = ing["index"]
        print(
            f"[{pages}] ingest {ing['result']['wall_seconds']}s, "
            f"{ing['result']['num_chunks']} chunks, peak RSS {ing['result']['peak_rss_mb']} MB",
            flush=True,
        )
        retrieval = run_retrieval(pipe, index, questions)
        print(f"[{pages}] retrieval {retrieval['overall']}", flush=True)
        record: dict[str, Any] = {
            "pages": pages,
            "pdf": pdf.name,
            "num_questions": len(questions),
            "environment": env,
            "model_load_seconds": round(model_load_s, 1),
            "gpu_used_mb_after_ingest": gpu_used_mb(),
            "ingest": ing["result"],
            "retrieval": retrieval,
            "measured": True,
        }
        if llm is not None:
            qs = questions[: args.answer_limit] if args.answer_limit else questions
            record["answers"] = run_answers(pipe, index, qs)
            a = record["answers"]
            print(
                f"[{pages}] answers acc={a['accuracy']} refusal={a['unanswerable_refused']} "
                f"p50={a['latency_p50']}s",
                flush=True,
            )
            record["gpu_used_mb_after_answers"] = gpu_used_mb()
        if args.recovery:
            _CLEAN_INDEX[ing["result"]["index_digest"]] = index
            record["recovery"] = run_recovery(
                pipe,
                pdf,
                Path(args.store_dir) / f"recovery-{pages}",
                ing["result"]["index_digest"],
                pages,
                args.workers,
            )
            print(f"[{pages}] recovery {record['recovery']}", flush=True)
            shutil.rmtree(Path(args.store_dir) / f"recovery-{pages}", ignore_errors=True)
            _CLEAN_INDEX.clear()
        path.write_text(json.dumps(record, indent=1, default=str))
        results.append(record)
        if not args.keep_store:
            shutil.rmtree(store, ignore_errors=True)
        pipe.indexes.clear()
        del index, ing
        (out / "scale_summary.md").write_text(summary_markdown(results, tiers))
    (out / "scale_summary.md").write_text(summary_markdown(results, tiers))
    print((out / "scale_summary.md").read_text())


if __name__ == "__main__":
    main()
