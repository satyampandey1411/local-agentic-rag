"""The scale/real-document benchmark scripts: quote location and reporting (no models)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCALE = Path(__file__).resolve().parents[1] / "benchmark" / "scale"
sys.path.insert(0, str(SCALE))

run_realdocs = pytest.importorskip("run_realdocs")
run_scale = pytest.importorskip("run_scale")


def test_locate_handles_hyphenation_case_and_dashes():
    pages = [
        run_realdocs.norm("Intro text."),
        run_realdocs.norm(
            "The Secretary shall pre-\nscribe REGULATIONS under this section — including rules"
        ),
    ]
    assert run_realdocs.locate(
        "the Secretary shall prescribe regulations under this section - including", pages
    ) == [2]
    assert run_realdocs.locate("text that is nowhere in the document at all, really", pages) == []


def test_locate_falls_back_to_head_and_tail_of_long_quotes():
    page = run_realdocs.norm(
        "one two three four five six [column break] seven eight nine ten eleven twelve"
    )
    quote = "one two three four five six seven eight nine ten eleven twelve"
    assert run_realdocs.locate(quote, [page]) == [1]


def test_summary_marks_tiers_not_run():
    record = {
        "pages": 100,
        "environment": {"cuda": True},
        "ingest": {
            "wall_seconds": 10.0,
            "parse_seconds": 6.0,
            "index_seconds": 3.0,
            "bm25_seconds": 0.1,
            "pages_per_second": 10.0,
            "num_chunks": 500,
            "peak_rss_mb": 900.0,
            "peak_cuda_mb": 1200.0,
            "storage": {"index_bytes": 2 * 2**20},
            "cold_load_seconds": 0.2,
        },
        "retrieval": {"overall": {"recall@5": 0.9, "mrr": 0.8, "latency_p50": 0.05}},
        "answers": {
            "answerable_accuracy": 0.7,
            "unanswerable_refused": 1.0,
            "citation_precision": 0.8,
            "latency_p50": 6.0,
        },
    }
    md = run_scale.summary_markdown([record], [100, 10000])
    assert "| 100 | 10 |" in md
    assert "| 10,000 | not run |" in md
