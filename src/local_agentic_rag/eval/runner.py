"""
Benchmark runner: call a pipeline on every question and score the results.

The runner is pipeline-agnostic.  It takes a plain callable::

    answer_fn(question: str, doc_id: str) -> dict
        # {"answer", "refused", "cited_pages", "retrieved_pages",
        #  "latency_sec", "error", optional "iterations", "trace", "peak_vram_bytes"}

and produces one *record* per question = question fields + normalised result
fields + ``metrics`` (from :func:`~local_agentic_rag.eval.metrics.question_metrics`).

Robustness guarantees
---------------------
* An exception (or a non-dict return) inside ``answer_fn`` becomes a record
  with ``error`` set — one bad question never kills a long run.
* With ``out_path`` every record is appended to a JSONL file and flushed
  immediately, so a Colab disconnect loses at most the in-flight question.
* With ``resume=True`` q_ids already present in ``out_path`` are skipped; a
  truncated final line (crash mid-write) is ignored and re-run.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from local_agentic_rag.eval.dataset import Question
from local_agentic_rag.eval.metrics import (
    DEFAULT_KS,
    as_pages,
    hit_at_k,
    mrr,
    page_recall_at_k,
    question_metrics,
    result_answer,
)

logger = logging.getLogger(__name__)

AnswerFn = Callable[[str, str], Mapping[str, Any]]
RetrieveFn = Callable[[str, str], Any]

#: Result keys the runner normalises into top-level record fields.
RESULT_FIELDS: tuple[str, ...] = (
    "answer",
    "refused",
    "cited_pages",
    "retrieved_pages",
    "latency_sec",
    "error",
    "iterations",
    "peak_vram_bytes",
    "trace",
)
# Aliases from ``types.QueryResult`` that are folded into RESULT_FIELDS.
_RESULT_ALIASES = {"extracted_answer", "raw_response", "agent_iterations"}

_MAX_TRACEBACK_CHARS = 4000


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays, sets, paths and arbitrary objects JSON-safe."""
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _to_jsonable(record: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON so in-memory records equal what is on disk."""
    return json.loads(json.dumps(record, default=_json_default, ensure_ascii=False))


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL results file.

    Malformed lines (e.g. a line truncated by a crash) are skipped with a
    warning.  If a q_id appears more than once the *last* record wins.
    """
    path = Path(path)
    if not path.is_file():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s:%d: skipping malformed JSONL line", path, lineno)
                continue
            if isinstance(rec, dict) and "q_id" in rec:
                by_id.pop(str(rec["q_id"]), None)  # re-insert so order follows last write
                by_id[str(rec["q_id"])] = rec
    return list(by_id.values())


class _JsonlWriter:
    """Append-only JSONL writer that flushes (and fsyncs) after every record."""

    def __init__(self, path: Path, *, truncate: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if truncate:
            path.write_text("", encoding="utf-8")
        needs_newline = path.is_file() and path.stat().st_size > 0 and not _ends_with_newline(path)
        self._fh = path.open("a", encoding="utf-8")
        if needs_newline:  # previous run died mid-line; don't glue onto the garbage
            self._fh.write("\n")

    def write(self, record: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
        self._fh.flush()
        # Some FUSE mounts (e.g. Google Drive in Colab) may not support fsync.
        with contextlib.suppress(OSError):
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


class _Progress:
    """tqdm progress bar when available, otherwise one short line per question."""

    def __init__(self, total: int, *, enabled: bool, desc: str) -> None:
        self.total = total
        self.enabled = enabled and total > 0
        self.done = 0
        self.correct = 0
        self.scored = 0
        self._bar: Any = None
        if self.enabled:
            try:
                from tqdm.auto import tqdm

                self._bar = tqdm(total=total, desc=desc, unit="q")
            except ImportError:
                self._bar = None

    def update(self, record: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self.done += 1
        m = record.get("metrics", {})
        ok = m.get("answer_correct")
        if ok is None:
            ok = m.get("refusal_correct")
        if ok is not None:
            self.scored += 1
            self.correct += int(bool(ok))
        acc = f"{100 * self.correct / self.scored:.0f}%" if self.scored else "-"
        if self._bar is not None:
            self._bar.set_postfix_str(f"acc={acc}", refresh=False)
            self._bar.update(1)
            return
        status = "ERROR" if record.get("error") else ("ok" if ok else "miss")
        latency = record.get("latency_sec") or 0.0
        print(
            f"[{self.done}/{self.total}] {record.get('q_id')} {status} {latency:.1f}s acc={acc}",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def _question_dict(question: Question | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(question, Question):
        return question.to_dict()
    if isinstance(question, Mapping):
        return dict(question)
    raise TypeError(f"question must be a Question or mapping, got {type(question).__name__}")


def normalize_result(result: Mapping[str, Any], *, wall_time_sec: float) -> dict[str, Any]:
    """Coerce a duck-typed pipeline result into the harness's canonical fields.

    Pages are de-duplicated preserving order; a missing ``latency_sec`` falls
    back to the measured wall time; ``iterations`` falls back to
    ``agent_iterations`` and ``answer`` to ``extracted_answer``/``raw_response``.
    """
    error = result.get("error")
    latency = result.get("latency_sec")
    iterations = result.get("iterations", result.get("agent_iterations"))
    return {
        "answer": result_answer(result),
        "refused": bool(result.get("refused", False)),
        "cited_pages": as_pages(result.get("cited_pages")),
        "retrieved_pages": as_pages(result.get("retrieved_pages")),
        "latency_sec": float(latency) if latency is not None else wall_time_sec,
        "error": None if error in (None, "") else str(error),
        "iterations": iterations,
        "peak_vram_bytes": result.get("peak_vram_bytes"),
        "trace": result.get("trace"),
    }


def build_record(
    question: Question | Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    wall_time_sec: float = 0.0,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Merge question fields, normalised result fields and per-question metrics."""
    qd = _question_dict(question)
    norm = normalize_result(result, wall_time_sec=wall_time_sec)
    known = set(RESULT_FIELDS) | _RESULT_ALIASES | set(qd)
    extra = {k: v for k, v in result.items() if k not in known}
    record: dict[str, Any] = {**qd, **norm, "wall_time_sec": wall_time_sec}
    if extra:
        record["result_extra"] = extra
    record["metrics"] = question_metrics(qd, norm, ks)
    return _to_jsonable(record)


def _error_result(exc: BaseException) -> dict[str, Any]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "answer": "",
        "refused": False,
        "cited_pages": [],
        "retrieved_pages": [],
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": tb[-_MAX_TRACEBACK_CHARS:],
    }


def _call_answer_fn(answer_fn: AnswerFn, q: Mapping[str, Any]) -> tuple[Mapping[str, Any], float]:
    t0 = time.perf_counter()
    try:
        result = answer_fn(q["question"], q["doc_id"])
        if not isinstance(result, Mapping):
            raise TypeError(f"answer_fn must return a dict, got {type(result).__name__}")
    except Exception as exc:  # a failing question must not crash the run
        logger.warning("q_id=%s failed: %s: %s", q.get("q_id"), type(exc).__name__, exc)
        result = _error_result(exc)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_benchmark(
    answer_fn: AnswerFn,
    questions: Iterable[Question | Mapping[str, Any]],
    *,
    out_path: str | Path | None = None,
    resume: bool = True,
    progress: bool = True,
    retry_errors: bool = False,
    ks: Sequence[int] = DEFAULT_KS,
) -> list[dict[str, Any]]:
    """Run ``answer_fn`` on every question and return scored records.

    Args:
        answer_fn: ``(question_text, doc_id) -> result dict``.
        questions: :class:`Question` objects or equivalent dicts.
        out_path: Optional JSONL file; each record is appended and flushed as
            soon as it is computed.
        resume: If True and ``out_path`` exists, reuse records whose q_id is
            already there instead of re-running them.  If False the file is
            truncated first.
        progress: Show a tqdm bar (or plain stderr lines if tqdm is missing).
        retry_errors: When resuming, re-run questions whose saved record has
            ``error`` set (the new record supersedes the old one on reload).
        ks: Cut-offs for recall@k / hit@k.

    Returns:
        One record per question, in the order of ``questions`` (duplicate
        q_ids are run once).
    """
    qdicts = [_question_dict(q) for q in questions]
    path = Path(out_path) if out_path is not None else None

    existing: dict[str, dict[str, Any]] = {}
    if path is not None and resume:
        for rec in load_records(path):
            if retry_errors and rec.get("error"):
                continue
            existing[str(rec["q_id"])] = rec
        if existing:
            logger.info("Resuming: %d record(s) already in %s", len(existing), path)

    seen: set[str] = set()
    todo: list[dict[str, Any]] = []
    for q in qdicts:
        qid = str(q.get("q_id"))
        if qid in seen:
            continue
        seen.add(qid)
        if qid not in existing:
            todo.append(q)

    writer = _JsonlWriter(path, truncate=not resume) if path is not None else None
    bar = _Progress(len(todo), enabled=progress, desc="benchmark")
    new: dict[str, dict[str, Any]] = {}
    try:
        for q in todo:
            result, wall = _call_answer_fn(answer_fn, q)
            record = build_record(q, result, wall_time_sec=wall, ks=ks)
            new[str(q.get("q_id"))] = record
            if writer is not None:
                writer.write(record)
            bar.update(record)
    finally:
        bar.close()
        if writer is not None:
            writer.close()

    ordered: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for q in qdicts:
        qid = str(q.get("q_id"))
        if qid in emitted:
            continue
        emitted.add(qid)
        rec = new.get(qid) or existing.get(qid)
        if rec is not None:
            ordered.append(rec)
    return ordered


def run_retrieval_benchmark(
    retrieve_fn: RetrieveFn,
    questions: Iterable[Question | Mapping[str, Any]],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    answerable_only: bool = True,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Retrieval-only evaluation (no generation).

    Args:
        retrieve_fn: ``(question_text, doc_id) -> ranked pages`` (a list of
            ints, or a dict with ``retrieved_pages``).  Duplicates are removed
            preserving rank order.
        questions: Questions to evaluate.
        ks: Cut-offs for recall@k / hit@k.
        answerable_only: Skip unanswerable questions (they have no ground-truth
            evidence to retrieve).
        progress: Print progress.

    Returns:
        Records with question fields, ``retrieved_pages``, ``latency_sec``,
        ``error`` and ``metrics`` (``recall@k``, ``hit@k``, ``rr``, ``error``).
        They can be passed straight to :func:`~local_agentic_rag.eval.metrics.aggregate`.
    """
    qdicts = [_question_dict(q) for q in questions]
    if answerable_only:
        qdicts = [q for q in qdicts if q.get("answerable", True)]
    bar = _Progress(len(qdicts), enabled=progress, desc="retrieval")
    records: list[dict[str, Any]] = []
    try:
        for q in qdicts:
            t0 = time.perf_counter()
            error: str | None = None
            try:
                raw = retrieve_fn(q["question"], q["doc_id"])
                if isinstance(raw, Mapping):
                    raw = raw.get("retrieved_pages", [])
                pages = as_pages(raw)
            except Exception as exc:  # retrieval failure is recorded, not raised
                logger.warning("q_id=%s retrieval failed: %s", q.get("q_id"), exc)
                pages, error = [], f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - t0
            evidence = q.get("evidence_pages") or []
            metrics: dict[str, Any] = {"error": error is not None}
            for k in ks:
                metrics[f"recall@{k}"] = page_recall_at_k(pages, evidence, k)
                metrics[f"hit@{k}"] = hit_at_k(pages, evidence, k)
            metrics["rr"] = mrr(pages, evidence)
            record = {
                **q,
                "retrieved_pages": pages,
                "latency_sec": latency,
                "error": error,
                "metrics": metrics,
            }
            records.append(_to_jsonable(record))
            bar.update(record)
    finally:
        bar.close()
    return records
