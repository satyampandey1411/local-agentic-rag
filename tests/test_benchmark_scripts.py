"""The scale/real-document benchmark scripts: quote location and reporting (no models)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCALE = Path(__file__).resolve().parents[1] / "benchmark" / "scale"
sys.path.insert(0, str(SCALE))

run_realdocs = pytest.importorskip("run_realdocs")
run_scale = pytest.importorskip("run_scale")
load_test = pytest.importorskip("load_test")


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


# ---------------------------------------------------------------------------- load_test


def _sse(*events: tuple[str, dict]) -> str:
    import json

    return (
        ":"
        + " " * 20
        + "\n\n"
        + "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events)
    )


def test_iter_sse_skips_comments_and_parses_events():
    body = _sse(("queued", {"position": 0}), ("token", {"text": "hi", "reset": False}))
    got = list(load_test.iter_sse(body.split("\n")))
    assert got == [("queued", {"position": 0}), ("token", {"text": "hi", "reset": False})]


def _client(body: str):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ask/stream":
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"answer": "A", "citations": []})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ask_once_records_first_token_and_answer():
    body = _sse(
        ("queued", {"position": 0}),
        ("started", {"queue_wait_s": 0.2}),
        ("token", {"text": "", "reset": True}),
        ("token", {"text": "Rev", "reset": False}),
        ("answer", {"answer": "Revenue", "citations": [], "queue_wait_s": 0.2}),
    )
    res = load_test.ask_once(_client(body), "http://x", "q", None, timeout=5)
    assert res["ok"] and res["queue_wait"] == 0.2
    assert res["first_token_s"] is not None and res["first_token_s"] <= res["latency"]


def test_ask_once_first_token_null_without_tokens_and_on_error():
    body = _sse(("started", {}), ("answer", {"answer": "x", "citations": []}))
    assert load_test.ask_once(_client(body), "http://x", "q", None, 5)["first_token_s"] is None
    err = _sse(("error", {"status": 504, "code": "timeout", "detail": "slow"}))
    res = load_test.ask_once(_client(err), "http://x", "q", None, 5)
    assert not res["ok"] and res["status"] == 504
    plain = load_test.ask_once(_client(""), "http://x", "q", None, 5, stream=False)
    assert plain["ok"] and plain["first_token_s"] is None


def test_summary_reports_first_token_percentiles_and_keeps_old_fields():
    results = [
        {"ok": True, "status": 200, "latency": 10.0, "first_token_s": t, "queue_wait": 0.1}
        for t in (1.0, 2.0, 3.0, 4.0)
    ] + [{"ok": True, "status": 200, "latency": 9.0, "first_token_s": None}]
    out = load_test.summarize(2, results, wall=30.0)
    assert out["first_token_p50"] == 2.5 and out["first_token_count"] == 4
    assert out["first_token_p95"] == pytest.approx(3.85, abs=0.01)
    for key in ("latency_p50", "latency_p95", "queue_wait_p50", "answers_per_minute", "ok"):
        assert key in out
    empty = load_test.summarize(1, [{"ok": False, "status": 500, "latency": 1.0}], wall=1.0)
    assert empty["first_token_p50"] is None
