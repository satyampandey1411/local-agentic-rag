"""
Evaluation harness for the local agentic RAG candidates.

Pure Python + NumPy; importing this package never imports a pipeline or a model.

Typical use::

    from local_agentic_rag.eval import (
        aggregate, load_benchmark, run_benchmark, summary_table_markdown,
    )

    docs, questions = load_benchmark("benchmark")
    records = run_benchmark(my_answer_fn, questions, out_path="runs/c0.jsonl")
    summary = aggregate(records)
    print(summary_table_markdown({"baseline": summary}))
"""

from __future__ import annotations

from local_agentic_rag.eval.dataset import (
    CATEGORIES,
    BenchmarkValidationError,
    Document,
    Question,
    filter_questions,
    load_benchmark,
    questions_by_doc,
)
from local_agentic_rag.eval.metrics import (
    aggregate,
    all_facts_present,
    bootstrap_ci,
    citation_precision,
    citation_recall,
    citation_valid,
    fact_recall,
    hit_at_k,
    mrr,
    normalize_text,
    page_recall_at_k,
    question_metrics,
    token_f1,
)
from local_agentic_rag.eval.report import (
    per_category_markdown,
    per_doc_markdown,
    save_summary_json,
    summary_table_markdown,
)
from local_agentic_rag.eval.runner import load_records, run_benchmark, run_retrieval_benchmark

__all__ = [
    "CATEGORIES",
    "BenchmarkValidationError",
    "Document",
    "Question",
    "aggregate",
    "all_facts_present",
    "bootstrap_ci",
    "citation_precision",
    "citation_recall",
    "citation_valid",
    "fact_recall",
    "filter_questions",
    "hit_at_k",
    "load_benchmark",
    "load_records",
    "mrr",
    "normalize_text",
    "page_recall_at_k",
    "per_category_markdown",
    "per_doc_markdown",
    "question_metrics",
    "questions_by_doc",
    "run_benchmark",
    "run_retrieval_benchmark",
    "save_summary_json",
    "summary_table_markdown",
    "token_f1",
]
