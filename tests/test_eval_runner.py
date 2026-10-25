"""
Tests for local_agentic_rag.eval.dataset, .runner and .report

Uses a tiny benchmark written to tmp_path and fake answer/retrieve callables,
so no models or PDFs are needed.

Covers:
- load_benchmark happy path + validation errors (with file/line context)
- filter_questions / questions_by_doc
- run_benchmark: record shape, page dedupe, exception -> error record,
  non-dict return, incremental JSONL, resume after a crash, truncated last
  line, retry_errors, resume=False truncation, QueryResult-style fields
- run_retrieval_benchmark
- report formatting (side-by-side table, per-category, JSON save/load)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from local_agentic_rag.eval import (
    BenchmarkValidationError,
    Question,
    aggregate,
    filter_questions,
    load_benchmark,
    load_records,
    per_category_markdown,
    questions_by_doc,
    run_benchmark,
    run_retrieval_benchmark,
    save_summary_json,
    summary_table_markdown,
)
from local_agentic_rag.eval.report import format_metric, load_summary_json, per_doc_markdown

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOCS = [
    {
        "doc_id": "DOC-A",
        "title": "Annual report",
        "url": "https://example.org/a.pdf",
        "sha256": "0" * 64,
        "num_pages": 10,
        "doc_type": "financial",
        "source_license_note": "public domain",
    },
    {
        "doc_id": "DOC-B",
        "title": "Paper",
        "url": "https://example.org/b.pdf",
        "sha256": "1" * 64,
        "num_pages": 5,
        "doc_type": "paper",
        "source_license_note": "CC-BY",
    },
]

QUESTIONS = [
    {
        "q_id": "A-1",
        "doc_id": "DOC-A",
        "category": "numerical",
        "question": "By how much did revenue grow?",
        "answerable": True,
        "reference_answer": "Revenue grew 4.2%.",
        "required_facts": [["4.2%", "4.2 percent"]],
        "evidence_pages": [3],
        "notes": "",
    },
    {
        "q_id": "A-2",
        "doc_id": "DOC-A",
        "category": "table",
        "question": "What was headcount?",
        "answerable": True,
        "reference_answer": "1,234 employees",
        "required_facts": [["1234"]],
        "evidence_pages": [7, 8],
        "notes": "",
    },
    {
        "q_id": "A-3",
        "doc_id": "DOC-A",
        "category": "unanswerable",
        "question": "What is the CEO's shoe size?",
        "answerable": False,
        "reference_answer": "",
        "required_facts": [],
        "evidence_pages": [],
        "notes": "",
    },
    {
        "q_id": "B-1",
        "doc_id": "DOC-B",
        "category": "definition",
        "question": "What is RAG?",
        "answerable": True,
        "reference_answer": "Retrieval-augmented generation",
        "required_facts": [["retrieval-augmented generation", "retrieval augmented generation"]],
        "evidence_pages": [1],
        "notes": "",
        "difficulty": "easy",
    },
]

ANSWERS: dict[str, dict[str, Any]] = {
    "By how much did revenue grow?": {
        "answer": "Revenue grew 4.2 percent [Source: Page 3]",
        "refused": False,
        "cited_pages": [3],
        "retrieved_pages": [3, 3, 5, 6],
        "latency_sec": 0.5,
        "error": None,
        "iterations": 2,
        "peak_vram_bytes": 1000,
    },
    "What was headcount?": {
        "answer": "About 1,000 employees",
        "refused": False,
        "cited_pages": [7],
        "retrieved_pages": [2, 7],
        "latency_sec": 0.7,
        "error": None,
    },
    "What is the CEO's shoe size?": {
        "answer": "ANSWER_NOT_FOUND",
        "refused": True,
        "cited_pages": [],
        "retrieved_pages": [1, 2],
        "latency_sec": 0.3,
        "error": None,
    },
    "What is RAG?": {
        "answer": "Retrieval–augmented generation",
        "refused": False,
        "cited_pages": [1],
        "retrieved_pages": [1],
        "latency_sec": 0.4,
        "error": None,
    },
}


def write_benchmark(root: Path, docs: Any = None, questions: Any = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "documents.json").write_text(json.dumps(DOCS if docs is None else docs))
    lines = QUESTIONS if questions is None else questions
    (root / "questions.jsonl").write_text(
        "\n".join(q if isinstance(q, str) else json.dumps(q) for q in lines) + "\n"
    )
    return root


@pytest.fixture()
def bench_dir(tmp_path: Path) -> Path:
    return write_benchmark(tmp_path / "benchmark")


@pytest.fixture()
def questions(bench_dir: Path) -> list[Question]:
    return load_benchmark(bench_dir)[1]


class FakePipeline:
    """Records calls; optionally raises for chosen questions."""

    def __init__(self, fail_on: set[str] | None = None, interrupt_after: int | None = None):
        self.calls: list[str] = []
        self.fail_on = fail_on or set()
        self.interrupt_after = interrupt_after

    def __call__(self, question: str, doc_id: str) -> dict[str, Any]:
        if self.interrupt_after is not None and len(self.calls) >= self.interrupt_after:
            raise KeyboardInterrupt  # simulates a Colab disconnect / kernel kill
        self.calls.append(question)
        if question in self.fail_on:
            raise RuntimeError("CUDA out of memory")
        return dict(ANSWERS[question])


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class TestLoadBenchmark:
    def test_happy_path(self, bench_dir: Path) -> None:
        docs, qs = load_benchmark(bench_dir)
        assert set(docs) == {"DOC-A", "DOC-B"}
        assert docs["DOC-A"].num_pages == 10
        assert [q.q_id for q in qs] == ["A-1", "A-2", "A-3", "B-1"]
        assert qs[0].required_facts == [["4.2%", "4.2 percent"]]
        assert qs[2].answerable is False
        assert qs[3].extra == {"difficulty": "easy"}
        assert qs[3].to_dict()["difficulty"] == "easy"

    def test_missing_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="documents"):
            load_benchmark(tmp_path)

    def test_collects_all_errors_with_line_numbers(self, tmp_path: Path) -> None:
        bad = [
            dict(QUESTIONS[0]),
            "{not json",
            {**QUESTIONS[1], "doc_id": "DOC-Z"},
            {**QUESTIONS[1], "q_id": "A-9", "category": "trivia"},
            {**QUESTIONS[1], "q_id": "A-10", "evidence_pages": [0]},
            {**QUESTIONS[1], "q_id": "A-11", "evidence_pages": [11]},
            {**QUESTIONS[1], "q_id": "A-12", "required_facts": ["1234"]},
            {**QUESTIONS[1], "q_id": "A-13", "required_facts": []},
            {**QUESTIONS[1], "q_id": "A-14", "answerable": "yes"},
            {**QUESTIONS[2], "q_id": "A-15", "answerable": True, "required_facts": [["x"]]},
            dict(QUESTIONS[0]),  # duplicate q_id
        ]
        root = write_benchmark(tmp_path / "b", questions=bad)
        with pytest.raises(BenchmarkValidationError) as exc_info:
            load_benchmark(root)
        errors = exc_info.value.errors
        text = "\n".join(errors)
        assert "questions.jsonl:2: invalid JSON" in text
        assert "unknown doc_id 'DOC-Z'" in text
        assert "'category' must be one of" in text
        assert "1-indexed" in text
        assert "exceed num_pages=10" in text
        assert "'required_facts' must be a list of non-empty lists" in text
        assert "needs at least one required_fact" in text
        assert "'answerable' must be true/false" in text
        assert "category 'unanswerable' but answerable=true" in text
        assert "duplicate q_id 'A-1'" in text
        assert len(errors) == 10

    def test_document_errors(self, tmp_path: Path) -> None:
        root = write_benchmark(tmp_path / "b", docs=[DOCS[0], DOCS[0], {"title": "x"}])
        with pytest.raises(BenchmarkValidationError) as exc_info:
            load_benchmark(root)
        text = str(exc_info.value)
        assert "duplicate doc_id" in text
        assert "'doc_id' must be a non-empty string" in text

    def test_documents_not_a_list(self, tmp_path: Path) -> None:
        root = write_benchmark(tmp_path / "b", docs={"doc_id": "x"})
        with pytest.raises(BenchmarkValidationError, match="expected a JSON list"):
            load_benchmark(root)

    def test_empty_questions(self, tmp_path: Path) -> None:
        root = write_benchmark(tmp_path / "b", questions=[])
        with pytest.raises(BenchmarkValidationError, match="no questions"):
            load_benchmark(root)


class TestFilters:
    def test_by_doc(self, questions: list[Question]) -> None:
        assert [q.q_id for q in filter_questions(questions, doc_ids="DOC-B")] == ["B-1"]

    def test_by_category(self, questions: list[Question]) -> None:
        got = filter_questions(questions, categories=["table", "unanswerable"])
        assert [q.q_id for q in got] == ["A-2", "A-3"]

    def test_unknown_category_raises(self, questions: list[Question]) -> None:
        with pytest.raises(ValueError, match="Unknown categories"):
            filter_questions(questions, categories="trivia")

    def test_answerable_and_limit(self, questions: list[Question]) -> None:
        got = filter_questions(questions, answerable=True, limit=2)
        assert [q.q_id for q in got] == ["A-1", "A-2"]
        assert filter_questions(questions, limit=0) == []

    def test_by_q_ids(self, questions: list[Question]) -> None:
        assert [q.q_id for q in filter_questions(questions, q_ids={"B-1", "A-1"})] == [
            "A-1",
            "B-1",
        ]

    def test_questions_by_doc(self, questions: list[Question]) -> None:
        groups = questions_by_doc(questions)
        assert list(groups) == ["DOC-A", "DOC-B"]
        assert len(groups["DOC-A"]) == 3


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_records_and_metrics(self, questions: list[Question]) -> None:
        records = run_benchmark(FakePipeline(), questions, progress=False)
        assert [r["q_id"] for r in records] == ["A-1", "A-2", "A-3", "B-1"]
        a1 = records[0]
        assert a1["retrieved_pages"] == [3, 5, 6]  # deduped, order preserved
        assert a1["category"] == "numerical"
        assert a1["iterations"] == 2
        assert a1["metrics"]["answer_correct"] is True
        assert a1["metrics"]["recall@1"] == 1.0
        assert records[1]["metrics"]["answer_correct"] is False  # 1000 != 1234
        assert records[1]["metrics"]["recall@3"] == 0.5
        assert records[2]["metrics"]["refusal_correct"] is True
        assert records[3]["metrics"]["answer_correct"] is True  # en dash normalised
        assert records[3]["difficulty"] == "easy"

        s = aggregate(records)["overall"]
        assert s["answer_accuracy"] == pytest.approx(2 / 3)
        assert s["refusal_accuracy"] == 1.0
        assert s["error_rate"] == 0.0

    def test_accepts_plain_dict_questions(self) -> None:
        records = run_benchmark(FakePipeline(), [dict(QUESTIONS[0])], progress=False)
        assert records[0]["metrics"]["answer_correct"] is True

    def test_exception_becomes_error_record(self, questions: list[Question]) -> None:
        fake = FakePipeline(fail_on={"What was headcount?"})
        records = run_benchmark(fake, questions, progress=False)
        assert len(records) == 4
        err = records[1]
        assert err["error"] == "RuntimeError: CUDA out of memory"
        assert "Traceback" in err["result_extra"]["traceback"]
        assert err["answer"] == ""
        assert err["metrics"]["answer_correct"] is False
        assert err["latency_sec"] >= 0.0
        assert aggregate(records)["overall"]["error_rate"] == pytest.approx(0.25)

    def test_non_dict_return_is_error(self, questions: list[Question]) -> None:
        records = run_benchmark(lambda q, d: "oops", questions[:1], progress=False)
        assert records[0]["error"].startswith("TypeError")

    def test_query_result_style_fields(self) -> None:
        def fn(question: str, doc_id: str) -> dict[str, Any]:
            return {
                "extracted_answer": "4.2%",
                "raw_response": "4.2% [Source: Page 3]",
                "refused": False,
                "cited_pages": [3],
                "retrieved_pages": [3],
                "agent_iterations": None,
                "latency_sec": 0.1,
                "peak_vram_bytes": 0,
                "error": None,
                "q_id": "ignored",
                "retrieved_units": [{"page": 3}],
            }

        rec = run_benchmark(fn, [dict(QUESTIONS[0])], progress=False)[0]
        assert rec["answer"] == "4.2%"
        assert rec["q_id"] == "A-1"  # question field wins over result field
        assert rec["result_extra"]["retrieved_units"] == [{"page": 3}]
        assert rec["metrics"]["answer_correct"] is True

    def test_missing_latency_uses_wall_time(self) -> None:
        def fn(question: str, doc_id: str) -> dict[str, Any]:
            return {"answer": "x", "refused": False}

        rec = run_benchmark(fn, [dict(QUESTIONS[0])], progress=False)[0]
        assert rec["latency_sec"] == rec["wall_time_sec"] >= 0.0
        assert rec["cited_pages"] == [] and rec["retrieved_pages"] == []

    def test_incremental_jsonl(self, questions: list[Question], tmp_path: Path) -> None:
        out = tmp_path / "runs" / "c0.jsonl"
        records = run_benchmark(FakePipeline(), questions, out_path=out, progress=False)
        lines = out.read_text().splitlines()
        assert len(lines) == 4
        assert [json.loads(line)["q_id"] for line in lines] == ["A-1", "A-2", "A-3", "B-1"]
        assert load_records(out) == records

    def test_resume_after_crash(self, questions: list[Question], tmp_path: Path) -> None:
        out = tmp_path / "run.jsonl"
        crashing = FakePipeline(interrupt_after=2)
        with pytest.raises(KeyboardInterrupt):
            run_benchmark(crashing, questions, out_path=out, progress=False)
        assert len(out.read_text().splitlines()) == 2  # work done before the crash is saved

        resumed = FakePipeline()
        records = run_benchmark(resumed, questions, out_path=out, progress=False)
        assert resumed.calls == ["What is the CEO's shoe size?", "What is RAG?"]
        assert [r["q_id"] for r in records] == ["A-1", "A-2", "A-3", "B-1"]
        assert len(out.read_text().splitlines()) == 4

        again = FakePipeline()
        run_benchmark(again, questions, out_path=out, progress=False)
        assert again.calls == []

    def test_resume_tolerates_truncated_line(
        self, questions: list[Question], tmp_path: Path
    ) -> None:
        out = tmp_path / "run.jsonl"
        run_benchmark(FakePipeline(), questions[:1], out_path=out, progress=False)
        with out.open("a") as fh:
            fh.write('{"q_id": "A-2", "answ')  # died mid-write, no newline
        fake = FakePipeline()
        records = run_benchmark(fake, questions, out_path=out, progress=False)
        assert len(fake.calls) == 3  # A-2 is re-run
        assert [r["q_id"] for r in records] == ["A-1", "A-2", "A-3", "B-1"]
        assert [r["q_id"] for r in load_records(out)] == ["A-1", "A-2", "A-3", "B-1"]

    def test_retry_errors(self, questions: list[Question], tmp_path: Path) -> None:
        out = tmp_path / "run.jsonl"
        run_benchmark(
            FakePipeline(fail_on={"What is RAG?"}), questions, out_path=out, progress=False
        )
        skip = FakePipeline()
        assert run_benchmark(skip, questions, out_path=out, progress=False)[3]["error"]
        assert skip.calls == []

        retry = FakePipeline()
        records = run_benchmark(retry, questions, out_path=out, progress=False, retry_errors=True)
        assert retry.calls == ["What is RAG?"]
        assert records[3]["error"] is None
        assert load_records(out)[-1]["error"] is None  # newer record supersedes

    def test_resume_false_truncates(self, questions: list[Question], tmp_path: Path) -> None:
        out = tmp_path / "run.jsonl"
        run_benchmark(FakePipeline(), questions, out_path=out, progress=False)
        fake = FakePipeline()
        run_benchmark(fake, questions[:2], out_path=out, resume=False, progress=False)
        assert len(fake.calls) == 2
        assert len(out.read_text().splitlines()) == 2

    def test_duplicate_questions_run_once(self, questions: list[Question]) -> None:
        fake = FakePipeline()
        records = run_benchmark(fake, [questions[0], questions[0]], progress=False)
        assert len(records) == 1 and len(fake.calls) == 1

    def test_progress_with_tqdm(self, questions: list[Question]) -> None:
        assert len(run_benchmark(FakePipeline(), questions, progress=True)) == 4

    def test_progress_fallback_without_tqdm(
        self,
        questions: list[Question],
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # makes the import fail
        fake = FakePipeline(fail_on={"What is RAG?"})
        run_benchmark(fake, questions, progress=True)
        err = capsys.readouterr().err.splitlines()
        assert len(err) == 4
        assert err[0].startswith("[1/4] A-1 ok")
        assert err[1].startswith("[2/4] A-2 miss")
        assert err[3].startswith("[4/4] B-1 ERROR")

    def test_non_serialisable_trace(self) -> None:
        class Obj:
            def __repr__(self) -> str:
                return "<Obj>"

        def fn(question: str, doc_id: str) -> dict[str, Any]:
            return {"answer": "4.2%", "refused": False, "trace": [Obj(), {1, 2}]}

        rec = run_benchmark(fn, [dict(QUESTIONS[0])], progress=False)[0]
        assert rec["trace"] == ["<Obj>", [1, 2]]


# ---------------------------------------------------------------------------
# run_retrieval_benchmark
# ---------------------------------------------------------------------------


class TestRetrievalBenchmark:
    def test_metrics(self, questions: list[Question]) -> None:
        def retrieve(question: str, doc_id: str) -> list[int]:
            return {
                "By how much did revenue grow?": [5, 5, 3],
                "What was headcount?": [7, 1, 2, 8],
                "What is RAG?": [2],
            }[question]

        records = run_retrieval_benchmark(retrieve, questions)
        assert [r["q_id"] for r in records] == ["A-1", "A-2", "B-1"]  # unanswerable skipped
        assert records[0]["retrieved_pages"] == [5, 3]
        assert records[0]["metrics"]["recall@1"] == 0.0
        assert records[0]["metrics"]["recall@3"] == 1.0
        assert records[0]["metrics"]["rr"] == 0.5
        assert records[1]["metrics"]["recall@3"] == 0.5
        assert records[1]["metrics"]["recall@10"] == 1.0

        s = aggregate(records)["overall"]
        assert s["hit@5"] == pytest.approx(2 / 3)
        assert s["mrr"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
        assert s["answer_accuracy"] is None

    def test_errors_and_dict_return(self, questions: list[Question]) -> None:
        def retrieve(question: str, doc_id: str) -> Any:
            if doc_id == "DOC-B":
                raise ValueError("index missing")
            return {"retrieved_pages": [3, 7]}

        records = run_retrieval_benchmark(retrieve, questions, ks=(1, 2), answerable_only=False)
        assert len(records) == 4
        assert records[1]["metrics"]["recall@2"] == 0.5
        assert records[2]["metrics"]["recall@1"] is None  # unanswerable: no evidence
        assert records[3]["error"] == "ValueError: index missing"
        assert records[3]["metrics"]["recall@1"] == 0.0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestReport:
    @pytest.fixture()
    def summary(self, questions: list[Question]) -> dict[str, Any]:
        return aggregate(run_benchmark(FakePipeline(), questions, progress=False))

    def test_format_metric(self) -> None:
        assert format_metric("answer_accuracy", 0.625, (0.5, 0.75)) == "62.5% [50.0-75.0]"
        assert format_metric("fact_recall", 1 / 3) == "33.3%"
        assert format_metric("recall@20", 0.5) == "50.0%"
        assert format_metric("mrr", 0.5) == "0.500"
        assert format_metric("latency_p50", 1.234) == "1.23s"
        assert format_metric("max_peak_vram_bytes", 2 * 1024**3) == "2.00 GiB"
        assert format_metric("n", 4) == "4"
        assert format_metric("fact_recall", None) == "n/a"
        assert format_metric("fact_recall", float("nan")) == "n/a"

    def test_summary_table_side_by_side(self, summary: dict[str, Any]) -> None:
        worse = aggregate(
            run_benchmark(
                FakePipeline(fail_on=set(ANSWERS)), [dict(q) for q in QUESTIONS], progress=False
            )
        )
        md = summary_table_markdown({"cand0": summary, "cand1": worse})
        lines = md.splitlines()
        assert lines[0] == "| Metric | cand0 | cand1 |"
        assert lines[1] == "|---|---:|---:|"
        acc = next(line for line in lines if line.startswith("| Answer accuracy"))
        assert "66.7% [" in acc and "| 0.0% [0.0-0.0] |" in acc
        assert any(line.startswith("| Error rate | 0.0% | 100.0% |") for line in lines)
        assert any(line.startswith("| MRR |") for line in lines)

    def test_summary_table_custom_metrics_and_overall_dict(self, summary: dict[str, Any]) -> None:
        md = summary_table_markdown({"x": summary["overall"]}, metrics=["n", "recall@1"])
        assert md.splitlines()[2:] == ["| Questions | 4 |", "| Page recall@1 | 66.7% |"]

    def test_per_category(self, summary: dict[str, Any]) -> None:
        md = per_category_markdown(summary, metrics=["n", "answer_accuracy", "refusal_accuracy"])
        lines = md.splitlines()
        assert lines[0].startswith("| Category | Questions | Answer accuracy |")
        assert any(line.startswith("| unanswerable | 1 | n/a | 100.0%") for line in lines)
        assert lines[-1].startswith("| **overall** | 4 |")

    def test_per_doc(self, summary: dict[str, Any]) -> None:
        md = per_doc_markdown(summary)
        assert "| DOC-A |" in md and "| DOC-B |" in md

    def test_per_category_requires_aggregate(self, summary: dict[str, Any]) -> None:
        with pytest.raises(KeyError):
            per_category_markdown(summary["overall"])

    def test_save_and_load_json(self, summary: dict[str, Any], tmp_path: Path) -> None:
        path = save_summary_json(summary, tmp_path / "out" / "s.json", metadata={"cfg": "c0"})
        loaded = load_summary_json(path)
        assert loaded["metadata"] == {"cfg": "c0"}
        assert loaded["summary"]["overall"]["n"] == 4
        assert loaded["summary"]["overall"]["answer_accuracy_ci"] == list(
            summary["overall"]["answer_accuracy_ci"]
        )
        plain = load_summary_json(save_summary_json(summary, tmp_path / "p.json"))
        assert plain["overall"]["n"] == 4
