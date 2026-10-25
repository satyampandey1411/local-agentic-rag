"""
Benchmark orchestration: retrieval ablations and end-to-end answer ablations.

    python -m local_agentic_rag.bench retrieval
    python -m local_agentic_rag.bench answer --llm ollama:qwen2.5:7b-instruct --configs baseline agentic

Each configuration changes one design decision relative to its neighbour, so
the result tables show what each component is worth:

retrieval (no LLM):
    minilm_dense        MiniLM dense (the simplest baseline)
    bge_en_dense        BGE-base (English) dense
    bm25                BM25 only
    hybrid_en           BGE-base + BM25, RRF fusion
    hybrid_en_rerank    + bge-reranker-base (previous default, English-only)
    m3_dense            BGE-M3 (multilingual) dense
    m3_hybrid_rerank    BGE-M3 + BM25 + bge-reranker-v2-m3 (current default)

answer (LLM):
    baseline            MiniLM dense, single pass
    hybrid_rerank       best retriever, single pass
    agentic             full loop (plan, grade/re-retrieve, verify)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from local_agentic_rag.config import PRESETS, RAGConfig
from local_agentic_rag.eval.dataset import load_benchmark
from local_agentic_rag.eval.metrics import aggregate
from local_agentic_rag.eval.report import per_category_markdown, summary_table_markdown
from local_agentic_rag.eval.runner import run_benchmark, run_retrieval_benchmark
from local_agentic_rag.ingest.pdf import PDFParseError
from local_agentic_rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_EN = {"embedder_model": "BAAI/bge-base-en-v1.5", "reranker_model": "BAAI/bge-reranker-base"}

RETRIEVAL_CONFIGS: dict[str, dict[str, Any]] = {
    "minilm_dense": {
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retrieval_mode": "dense",
        "reranker_model": None,
    },
    "bge_en_dense": {**_EN, "retrieval_mode": "dense", "reranker_model": None},
    "bm25": {**_EN, "retrieval_mode": "bm25", "reranker_model": None},
    "hybrid_en": {**_EN, "retrieval_mode": "hybrid", "reranker_model": None},
    "hybrid_en_rerank": {**_EN},  # previous default (English-only models)
    "m3_dense": {"retrieval_mode": "dense", "reranker_model": None},
    "m3_hybrid_rerank": {},  # current default (multilingual BGE-M3 + bge-reranker-v2-m3)
}

RETRIEVAL_METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "hit@5", "mrr", "latency_p50"]


def ingest_benchmark(
    pipe: RAGPipeline, benchmark_dir: str | Path, pdf_dir: str | Path | None = None
) -> dict[str, Any]:
    docs, _ = load_benchmark(benchmark_dir)
    pdf_dir = Path(pdf_dir or Path(benchmark_dir) / "pdfs")
    reports = {}
    for doc_id in docs:
        path = pdf_dir / f"{doc_id}.pdf"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — run benchmark/fetch_pdfs.py first")
        try:
            reports[doc_id] = pipe.ingest(path, doc_id=doc_id).to_dict()
        except PDFParseError as exc:
            # e.g. a scanned PDF without tesseract: its questions are then scored as errors
            # (never silently dropped), and the rest of the benchmark still runs.
            logger.warning("Skipping %s: %s", doc_id, exc)
            reports[doc_id] = {"error": str(exc)}
    return reports


# --------------------------------------------------------------------------- retrieval


def retrieval_ablation(
    benchmark_dir: str | Path = "benchmark",
    pdf_dir: str | Path | None = None,
    configs: list[str] | None = None,
    k: int = 10,
    base: RAGPipeline | None = None,
) -> dict[str, dict[str, Any]]:
    benchmark_dir = Path(benchmark_dir)
    pdf_dir = Path(pdf_dir or benchmark_dir / "pdfs")
    docs, questions = load_benchmark(benchmark_dir)
    base = base or RAGPipeline(RAGConfig())
    summaries: dict[str, dict[str, Any]] = {}

    for name in configs or list(RETRIEVAL_CONFIGS):
        pipe = base.variant(**RETRIEVAL_CONFIGS[name])
        ingest_benchmark(pipe, benchmark_dir, pdf_dir)
        records = run_retrieval_benchmark(
            lambda q, d, pipe=pipe: [sc.chunk.page for sc in pipe.retrieve(q, d, top_k=k)],
            questions,
        )
        summaries[name] = aggregate(records)
        logger.info("%s: recall@5=%.3f", name, summaries[name]["overall"].get("recall@5") or 0)
    _release_unused_models(base)
    return summaries


def _release_unused_models(base: RAGPipeline) -> None:
    """Free the ablation-only models (Colab has ~12.7 GB host RAM shared with Ollama)."""
    import gc

    for kind, keep in (
        ("embedders", base.config.embedder_model),
        ("rerankers", base.config.reranker_model),
    ):
        for name, model in list(base._shared[kind].items()):
            if name != keep:
                model.unload()
                del base._shared[kind][name]
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- answers


def answer_ablation(
    llm,
    benchmark_dir: str | Path = "benchmark",
    pdf_dir: str | Path | None = None,
    configs: list[str] | None = None,
    out_dir: str | Path = "benchmark/results",
    limit: int | None = None,
    base: RAGPipeline | None = None,
    resume: bool = True,
    retry_errors: bool = False,
    tag: str = "",
) -> dict[str, dict[str, Any]]:
    benchmark_dir = Path(benchmark_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, questions = load_benchmark(benchmark_dir)
    if limit:
        questions = questions[:limit]
    base = base or RAGPipeline(RAGConfig(), llm=llm)
    base.llm = llm
    summaries: dict[str, dict[str, Any]] = {}
    for name in configs or ["baseline", "hybrid_rerank", "agentic"]:
        if name not in PRESETS:
            raise ValueError(f"Unknown config {name!r}; choose from {sorted(PRESETS)}")
        pipe = base.variant(**PRESETS[name])  # keep the base pipeline's other settings
        ingest_benchmark(pipe, benchmark_dir, pdf_dir)

        def answer_fn(question: str, doc_id: str, pipe=pipe) -> dict[str, Any]:
            return _with_vram(lambda: pipe.ask(question, doc_id=doc_id).to_dict())

        label = f"{name}{tag}"
        out_path = out_dir / f"answers_{label}.jsonl"
        records = run_benchmark(
            answer_fn, questions, out_path=out_path, resume=resume, retry_errors=retry_errors
        )
        summary = aggregate(records)
        languages = sorted(
            {r.get("question_language") for r in records if r.get("question_language")}
        )
        if languages:
            summary["per_language"] = {
                code: aggregate([r for r in records if r.get("question_language") == code])[
                    "overall"
                ]
                for code in languages
            }
        summary["config"] = pipe.config.to_dict()
        summary["llm"] = llm.describe() if hasattr(llm, "describe") else str(llm)
        summaries[label] = summary
        (out_dir / f"summary_{label}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summaries


def _with_vram(fn):
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    result = fn()
    if torch is not None and torch.cuda.is_available():
        result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    return result


# --------------------------------------------------------------------------- CLI


def make_llm(spec: str):
    """``ollama:<model>`` or ``llamacpp:<path.gguf>``."""
    backend, _, target = spec.partition(":")
    if backend == "ollama":
        from local_agentic_rag.llm.ollama import OllamaLLM

        llm = OllamaLLM(target or "qwen2.5:7b-instruct")
        llm.ensure_ready()
        return llm
    if backend == "llamacpp":
        from local_agentic_rag.llm.llamacpp import LlamaCppLLM

        return LlamaCppLLM(target)
    raise ValueError(f"Unknown LLM spec {spec!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["retrieval", "answer"])
    parser.add_argument("--benchmark-dir", default="benchmark")
    parser.add_argument("--pdf-dir", default=None)
    parser.add_argument("--out-dir", default="benchmark/results")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--llm", default="ollama:qwen2.5:7b-instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix for result files, e.g. _run2")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="re-run questions whose saved record has an error",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "retrieval":
        summaries = retrieval_ablation(args.benchmark_dir, args.pdf_dir, args.configs)
        table = summary_table_markdown(summaries, RETRIEVAL_METRICS)
    else:
        summaries = answer_ablation(
            make_llm(args.llm),
            args.benchmark_dir,
            args.pdf_dir,
            args.configs,
            out_dir,
            args.limit,
            retry_errors=args.retry_errors,
            tag=args.tag,
        )
        table = summary_table_markdown(summaries)
        for name, s in summaries.items():
            table += f"\n\n### {name} by category\n\n" + per_category_markdown(s)
    (out_dir / f"{args.mode}{args.tag}_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str)
    )
    (out_dir / f"{args.mode}{args.tag}_report.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
