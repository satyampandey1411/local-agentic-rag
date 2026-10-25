"""
Benchmark dataset loading and validation.

On-disk layout (``benchmark/``)::

    documents.json    list of {"doc_id", "title", "url", "sha256", "num_pages",
                               "doc_type", "source_license_note"}
    questions.jsonl   one question object per line (see :class:`Question`)

:func:`load_benchmark` parses both files, validates every record and raises a
single :class:`BenchmarkValidationError` that lists *all* problems found (with
file and line numbers), so a broken dataset is fixed in one pass rather than
one error at a time.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DOCUMENTS_FILE = "documents.json"
QUESTIONS_FILE = "questions.jsonl"

CATEGORIES: tuple[str, ...] = (
    "factual",
    "table",
    "numerical",
    "multi_hop",
    "comparison",
    "definition",
    "unanswerable",
)


class BenchmarkValidationError(ValueError):
    """Raised when the benchmark files are malformed; ``errors`` lists every problem."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        shown = "\n  - ".join(self.errors[:50])
        more = f"\n  ... and {len(self.errors) - 50} more" if len(self.errors) > 50 else ""
        super().__init__(f"{len(self.errors)} benchmark validation error(s):\n  - {shown}{more}")


@dataclass(frozen=True)
class Document:
    """One benchmark PDF."""

    doc_id: str
    title: str = ""
    url: str = ""
    sha256: str = ""
    num_pages: int = 0
    doc_type: str = ""
    source_license_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Question:
    """One benchmark question.

    ``required_facts`` is a list of facts; each fact is a list of aliases and
    is satisfied if ANY alias appears in the (normalised) answer.
    ``evidence_pages`` are 1-indexed.  ``extra`` keeps unknown JSON keys.
    """

    q_id: str
    doc_id: str
    category: str
    question: str
    answerable: bool
    reference_answer: str = ""
    required_facts: list[list[str]] = field(default_factory=list)
    evidence_pages: list[int] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict of the question (``extra`` keys merged at top level)."""
        d = asdict(self)
        extra = d.pop("extra")
        return {**extra, **d}


_QUESTION_FIELDS = {
    "q_id",
    "doc_id",
    "category",
    "question",
    "answerable",
    "reference_answer",
    "required_facts",
    "evidence_pages",
    "notes",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_document(raw: Any, where: str, errors: list[str]) -> Document | None:
    if not isinstance(raw, Mapping):
        errors.append(f"{where}: expected an object, got {type(raw).__name__}")
        return None
    doc_id = raw.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        errors.append(f"{where}: 'doc_id' must be a non-empty string")
        return None
    num_pages = raw.get("num_pages", 0)
    if num_pages is None:
        num_pages = 0
    if not _is_int(num_pages) or num_pages < 0:
        errors.append(f"{where} ({doc_id}): 'num_pages' must be a non-negative int")
        num_pages = 0
    str_fields = ("title", "url", "sha256", "doc_type", "source_license_note")
    values: dict[str, str] = {}
    for name in str_fields:
        value = raw.get(name, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            errors.append(f"{where} ({doc_id}): '{name}' must be a string")
            value = str(value)
        values[name] = value
    return Document(doc_id=doc_id, num_pages=num_pages, **values)


def _parse_question(
    raw: Any, where: str, docs: Mapping[str, Document], errors: list[str]
) -> Question | None:
    if not isinstance(raw, Mapping):
        errors.append(f"{where}: expected a JSON object, got {type(raw).__name__}")
        return None
    q_id = raw.get("q_id")
    label = f"{where} ({q_id})" if isinstance(q_id, str) else where
    n_before = len(errors)

    for name in ("q_id", "doc_id", "question"):
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}: '{name}' must be a non-empty string")
    doc_id = raw.get("doc_id")
    if isinstance(doc_id, str) and doc_id and doc_id not in docs:
        errors.append(f"{label}: unknown doc_id {doc_id!r} (not in {DOCUMENTS_FILE})")

    category = raw.get("category")
    if category not in CATEGORIES:
        errors.append(f"{label}: 'category' must be one of {list(CATEGORIES)}, got {category!r}")

    answerable = raw.get("answerable")
    if not isinstance(answerable, bool):
        errors.append(f"{label}: 'answerable' must be true/false, got {answerable!r}")

    for name in ("reference_answer", "notes"):
        value = raw.get(name, "")
        if value is not None and not isinstance(value, str):
            errors.append(f"{label}: '{name}' must be a string")

    facts = raw.get("required_facts", []) or []
    facts_ok = isinstance(facts, list) and all(
        isinstance(f, list) and f and all(isinstance(a, str) and a.strip() for a in f)
        for f in facts
    )
    if not facts_ok:
        errors.append(
            f"{label}: 'required_facts' must be a list of non-empty lists of non-empty "
            f'strings, e.g. [["4.2%", "4.2 percent"], ["2023"]]'
        )
        facts = []

    pages = raw.get("evidence_pages", []) or []
    if not isinstance(pages, list) or not all(_is_int(p) and p >= 1 for p in pages):
        errors.append(f"{label}: 'evidence_pages' must be a list of 1-indexed ints")
        pages = []
    elif isinstance(doc_id, str) and doc_id in docs:
        n_pages = docs[doc_id].num_pages
        out_of_range = [p for p in pages if n_pages and p > n_pages]
        if out_of_range:
            errors.append(
                f"{label}: evidence_pages {out_of_range} exceed num_pages={n_pages} of {doc_id}"
            )

    if isinstance(answerable, bool):
        if answerable and facts_ok and not facts:
            errors.append(f"{label}: answerable question needs at least one required_fact")
        if answerable and category == "unanswerable":
            errors.append(f"{label}: category 'unanswerable' but answerable=true")
        if not answerable and category in CATEGORIES and category != "unanswerable":
            logger.warning("%s: answerable=false but category=%r", label, category)

    if len(errors) > n_before:
        return None
    return Question(
        q_id=str(q_id),
        doc_id=str(doc_id),
        category=str(category),
        question=str(raw["question"]),
        answerable=bool(answerable),
        reference_answer=raw.get("reference_answer") or "",
        required_facts=[list(f) for f in facts],
        evidence_pages=list(pages),
        notes=raw.get("notes") or "",
        extra={k: v for k, v in raw.items() if k not in _QUESTION_FIELDS},
    )


def load_documents(path: str | Path) -> dict[str, Document]:
    """Load and validate ``documents.json``; returns ``{doc_id: Document}``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark documents file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkValidationError([f"{path}: invalid JSON: {exc}"]) from exc
    if not isinstance(raw, list):
        raise BenchmarkValidationError([f"{path}: expected a JSON list of documents"])

    errors: list[str] = []
    docs: dict[str, Document] = {}
    for i, item in enumerate(raw):
        doc = _parse_document(item, f"{path.name}[{i}]", errors)
        if doc is None:
            continue
        if doc.doc_id in docs:
            errors.append(f"{path.name}[{i}]: duplicate doc_id {doc.doc_id!r}")
            continue
        docs[doc.doc_id] = doc
    if errors:
        raise BenchmarkValidationError(errors)
    return docs


def load_questions(path: str | Path, docs: Mapping[str, Document]) -> list[Question]:
    """Load and validate ``questions.jsonl`` against the loaded documents."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark questions file not found: {path}")

    errors: list[str] = []
    questions: list[Question] = []
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON: {exc.msg} (col {exc.colno})")
                continue
            q = _parse_question(raw, where, docs, errors)
            if q is None:
                continue
            if q.q_id in seen:
                errors.append(f"{where}: duplicate q_id {q.q_id!r} (first on line {seen[q.q_id]})")
                continue
            seen[q.q_id] = lineno
            questions.append(q)
    if errors:
        raise BenchmarkValidationError(errors)
    if not questions:
        raise BenchmarkValidationError([f"{path}: contains no questions"])
    return questions


def load_benchmark(benchmark_dir: str | Path) -> tuple[dict[str, Document], list[Question]]:
    """Load ``documents.json`` + ``questions.jsonl`` from ``benchmark_dir``.

    Raises:
        FileNotFoundError: if either file is missing.
        BenchmarkValidationError: listing every schema/consistency problem.
    """
    root = Path(benchmark_dir)
    docs = load_documents(root / DOCUMENTS_FILE)
    questions = load_questions(root / QUESTIONS_FILE, docs)
    return docs, questions


def filter_questions(
    questions: Iterable[Question],
    *,
    doc_ids: str | Iterable[str] | None = None,
    categories: str | Iterable[str] | None = None,
    answerable: bool | None = None,
    q_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[Question]:
    """Filter questions, preserving order.

    Args:
        doc_ids: Keep only these doc_id(s).
        categories: Keep only these category(ies).
        answerable: Keep only answerable (True) / unanswerable (False) questions.
        q_ids: Keep only these question ids.
        limit: Keep at most this many (applied after the other filters).
    """
    docs = {doc_ids} if isinstance(doc_ids, str) else set(doc_ids) if doc_ids else None
    cats = {categories} if isinstance(categories, str) else set(categories) if categories else None
    if cats:
        unknown = cats - set(CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown categories {sorted(unknown)}; valid: {list(CATEGORIES)}")
    ids = set(q_ids) if q_ids is not None else None

    out: list[Question] = []
    for q in questions:
        if limit is not None and len(out) >= limit:
            break
        if docs is not None and q.doc_id not in docs:
            continue
        if cats is not None and q.category not in cats:
            continue
        if answerable is not None and q.answerable != answerable:
            continue
        if ids is not None and q.q_id not in ids:
            continue
        out.append(q)
    return out


def questions_by_doc(questions: Iterable[Question]) -> dict[str, list[Question]]:
    """Group questions by doc_id (insertion order) — useful to ingest each PDF once."""
    groups: dict[str, list[Question]] = {}
    for q in questions:
        groups.setdefault(q.doc_id, []).append(q)
    return groups
