"""
Pure scoring functions for the benchmark harness.

Everything here is deterministic, dependency-light (stdlib + NumPy) and free
of pipeline imports, so it can be unit-tested and re-run over saved JSONL
records without loading any model.

Layers
------
1. Text normalisation  — :func:`normalize_text`.
2. Answer metrics      — :func:`fact_recall`, :func:`all_facts_present`, :func:`token_f1`.
3. Retrieval metrics   — :func:`page_recall_at_k`, :func:`hit_at_k`, :func:`mrr`.
4. Citation metrics    — :func:`citation_precision`, :func:`citation_recall`,
   :func:`citation_valid`.
5. Per-question scoring — :func:`question_metrics` turns (question, result) into a
   flat dict where ``None`` means "not applicable to this question".
6. Aggregation         — :func:`aggregate` computes overall / per-category /
   per-doc summaries with seeded bootstrap confidence intervals.

Conventions
-----------
* Pages are 1-indexed ints.  Ranked page lists are de-duplicated preserving
  first occurrence before any rank-based metric is computed.
* A metric that is undefined for a question (e.g. recall with no evidence
  pages, precision with no citations) is ``None`` and is skipped by
  :func:`aggregate` rather than counted as zero.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_DASH_CHARS = "‐‑‒–—―−⁃﹘﹣－"
_SINGLE_QUOTES = "‘’‚‛′`´"
_DOUBLE_QUOTES = "“”„‟″«»"

_TRANSLATION = {
    # Indic digits (Devanagari ०-९, Bengali, Gujarati, ...) compare equal to ASCII digits.
    **{0x0966 + i: str(i) for i in range(10)},
    **{0x09E6 + i: str(i) for i in range(10)},
    **{0x0AE6 + i: str(i) for i in range(10)},
    **{ord(c): "-" for c in _DASH_CHARS},
    **{ord(c): "'" for c in _SINGLE_QUOTES},
    **{ord(c): '"' for c in _DOUBLE_QUOTES},
}

_PERCENT_WORD_RE = re.compile(r"\s*\bper\s?cent\b")
_SPACE_BEFORE_PERCENT_RE = re.compile(r"(?<=\d)\s+%")
# Thousands separator: a comma between a digit and exactly three digits.
# "1,234,567" -> "1234567" but "pages 1,2" is left alone.
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalize_text(text: str | None) -> str:
    """Normalise text for lenient string comparison.

    Steps: Unicode NFKC, lowercase, unify dash/quote variants to ASCII,
    ``"percent"``/``"per cent"`` -> ``"%"`` (and ``"5 %"`` -> ``"5%"``),
    drop thousands-separator commas (``"1,234"`` -> ``"1234"``), collapse
    whitespace and strip.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = s.translate(_TRANSLATION)
    s = _PERCENT_WORD_RE.sub("%", s)
    s = _SPACE_BEFORE_PERCENT_RE.sub("%", s)
    s = _THOUSANDS_COMMA_RE.sub("", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

# A "purely numeric" alias after normalisation: optional sign/currency prefix,
# digits with an optional decimal part, optional trailing percent sign.
_NUMERIC_ALIAS_RE = re.compile(r"[-+$€£¥]*\d+(?:\.\d+)?%?")


def is_numeric_alias(alias: str) -> bool:
    """True if ``alias`` (after normalisation) is a bare number like ``"4.2"`` or ``"$1.5%"``."""
    return bool(_NUMERIC_ALIAS_RE.fullmatch(normalize_text(alias)))


def alias_matches(answer: str, alias: str, *, normalized: bool = False) -> bool:
    """Return True if ``alias`` occurs in ``answer`` after normalisation.

    Non-numeric aliases use plain substring matching.  Purely numeric aliases
    use number-boundary matching so ``"4.2"`` does not match ``"14.25"`` and
    ``"4"`` does not match ``"4.5"`` or ``"1.4"``.

    Args:
        answer: Answer text (raw, or already normalised if ``normalized``).
        alias: One alias of a required fact.
        normalized: Set True when ``answer`` is already normalised (perf only).
    """
    ans = answer if normalized else normalize_text(answer)
    al = normalize_text(alias)
    if not al or not ans:
        return False
    if _NUMERIC_ALIAS_RE.fullmatch(al):
        pattern = r"(?<!\d)(?<!\d\.)" + re.escape(al) + r"(?!\d)(?!\.\d)"
        return re.search(pattern, ans) is not None
    return al in ans


def _iter_facts(required_facts: Iterable[Any]) -> list[list[str]]:
    """Coerce required_facts to list[list[str]] (a bare string is a single-alias fact)."""
    facts: list[list[str]] = []
    for fact in required_facts or []:
        if isinstance(fact, str):
            facts.append([fact])
        else:
            facts.append([str(a) for a in fact])
    return facts


def fact_hits(answer: str | None, required_facts: Iterable[Any]) -> list[bool]:
    """Per-fact satisfaction flags: fact i is satisfied if ANY of its aliases matches."""
    ans = normalize_text(answer)
    return [
        any(alias_matches(ans, alias, normalized=True) for alias in aliases)
        for aliases in _iter_facts(required_facts)
    ]


def fact_recall(answer: str | None, required_facts: Iterable[Any]) -> float:
    """Fraction of required facts present in ``answer``.

    Vacuously 1.0 when there are no required facts.
    """
    hits = fact_hits(answer, required_facts)
    if not hits:
        return 1.0
    return sum(hits) / len(hits)


def all_facts_present(answer: str | None, required_facts: Iterable[Any]) -> bool:
    """True if every required fact is present (vacuously True for no facts)."""
    return all(fact_hits(answer, required_facts))


_ARTICLES = {"a", "an", "the"}
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?%?|\w+")


def _f1_tokens(text: str | None) -> list[str]:
    """SQuAD-style tokens: normalised, punctuation dropped, articles removed.

    Decimal numbers and percentages are kept as single tokens ("4.2", "15%").
    """
    return [t for t in _TOKEN_RE.findall(normalize_text(text)) if t not in _ARTICLES]


def token_f1(pred: str | None, ref: str | None) -> float:
    """SQuAD-style bag-of-tokens F1 between prediction and reference.

    Both empty -> 1.0; exactly one empty -> 0.0.
    """
    p, r = _f1_tokens(pred), _f1_tokens(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def as_pages(pages: Iterable[Any] | None) -> list[int]:
    """Coerce to a de-duplicated list of int pages, preserving first occurrence.

    Values that cannot be parsed as ints (e.g. ``None``, ``"abc"``) are dropped.
    """
    out: list[int] = []
    seen: set[int] = set()
    for p in pages or []:
        try:
            page = int(p)
        except (TypeError, ValueError):
            continue
        if page not in seen:
            seen.add(page)
            out.append(page)
    return out


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def page_recall_at_k(
    ranked_pages: Sequence[Any] | None, evidence_pages: Iterable[Any] | None, k: int
) -> float | None:
    """Fraction of evidence pages found in the top-``k`` unique ranked pages.

    Returns None when there are no evidence pages (metric undefined).
    """
    _check_k(k)
    evidence = set(as_pages(evidence_pages))
    if not evidence:
        return None
    top = set(as_pages(ranked_pages)[:k])
    return len(top & evidence) / len(evidence)


def hit_at_k(
    ranked_pages: Sequence[Any] | None, evidence_pages: Iterable[Any] | None, k: int
) -> float | None:
    """1.0 if any evidence page is in the top-``k`` unique pages, else 0.0; None if no evidence."""
    _check_k(k)
    evidence = set(as_pages(evidence_pages))
    if not evidence:
        return None
    return 1.0 if set(as_pages(ranked_pages)[:k]) & evidence else 0.0


def mrr(ranked_pages: Sequence[Any] | None, evidence_pages: Iterable[Any] | None) -> float | None:
    """Reciprocal rank of the first evidence page in the unique ranking.

    0.0 if no evidence page is retrieved; None if there are no evidence pages.
    Averaging this over questions gives Mean Reciprocal Rank.
    """
    evidence = set(as_pages(evidence_pages))
    if not evidence:
        return None
    for rank, page in enumerate(as_pages(ranked_pages), start=1):
        if page in evidence:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------


def citation_precision(
    cited_pages: Iterable[Any] | None, evidence_pages: Iterable[Any] | None
) -> float | None:
    """Fraction of unique cited pages that are evidence pages.

    None if there are no citations or no evidence pages (undefined).
    """
    cited = as_pages(cited_pages)
    evidence = set(as_pages(evidence_pages))
    if not cited or not evidence:
        return None
    return sum(1 for p in cited if p in evidence) / len(cited)


def citation_recall(
    cited_pages: Iterable[Any] | None, evidence_pages: Iterable[Any] | None
) -> float | None:
    """Fraction of evidence pages that were cited; None if there are no evidence pages."""
    evidence = set(as_pages(evidence_pages))
    if not evidence:
        return None
    return len(set(as_pages(cited_pages)) & evidence) / len(evidence)


def citation_valid(
    cited_pages: Iterable[Any] | None, retrieved_pages: Iterable[Any] | None
) -> bool:
    """True if every cited page was actually retrieved (cited ⊆ retrieved).

    Vacuously True when nothing is cited.  A False here means the model cited
    a page it never saw — a fabricated citation.
    """
    return set(as_pages(cited_pages)) <= set(as_pages(retrieved_pages))


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------


def result_answer(result: Mapping[str, Any]) -> str:
    """Extract the answer text from a pipeline result.

    Accepts ``answer`` (harness contract) and falls back to
    ``extracted_answer`` / ``raw_response`` (``types.QueryResult``).
    """
    for key in ("answer", "extracted_answer", "raw_response"):
        value = result.get(key)
        if value:
            return str(value)
    return ""


def question_metrics(
    question: Mapping[str, Any], result: Mapping[str, Any], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, Any]:
    """Score one pipeline result against its benchmark question.

    Args:
        question: Question fields (``answerable``, ``required_facts``,
            ``reference_answer``, ``evidence_pages``).
        result: Pipeline result fields (``answer``, ``refused``,
            ``cited_pages``, ``retrieved_pages``, ``error``).
        ks: Cut-offs for recall@k / hit@k.

    Returns:
        Flat dict of metrics.  ``None`` means "not applicable":
        answer/retrieval metrics are only defined for answerable questions,
        refusal/hallucination only for unanswerable ones, citation quality only
        for answered (not refused, no error) answerable questions that cite.
    """
    answerable = bool(question.get("answerable", True))
    answer = result_answer(result)
    refused = bool(result.get("refused", False))
    error = result.get("error")
    errored = error is not None and error != ""
    answered = not refused and not errored
    evidence = as_pages(question.get("evidence_pages"))
    retrieved = as_pages(result.get("retrieved_pages"))
    cited = as_pages(result.get("cited_pages"))
    facts = question.get("required_facts") or []

    m: dict[str, Any] = {"error": errored, "refused": refused}

    if answerable:
        all_present = all_facts_present(answer, facts)
        m["fact_recall"] = fact_recall(answer, facts)
        m["all_facts_present"] = all_present
        m["answer_correct"] = all_present and answered
        m["token_f1"] = token_f1(answer, question.get("reference_answer", ""))
        m["false_refusal"] = refused
        m["has_citation"] = bool(cited) if answered else None
        cites = answered and bool(cited)
        m["citation_precision"] = citation_precision(cited, evidence) if cites else None
        m["citation_recall"] = citation_recall(cited, evidence) if cites else None
        m["citation_valid"] = citation_valid(cited, retrieved) if cites else None
        for k in ks:
            m[f"recall@{k}"] = page_recall_at_k(retrieved, evidence, k)
            m[f"hit@{k}"] = hit_at_k(retrieved, evidence, k)
        m["rr"] = mrr(retrieved, evidence)
        m["refusal_correct"] = None
        m["hallucinated"] = None
    else:
        for key in (
            "fact_recall",
            "all_facts_present",
            "answer_correct",
            "token_f1",
            "false_refusal",
            "has_citation",
            "citation_precision",
            "citation_recall",
            "citation_valid",
            "rr",
        ):
            m[key] = None
        for k in ks:
            m[f"recall@{k}"] = None
            m[f"hit@{k}"] = None
        # An errored run is not a correct refusal, but it is not a hallucination either.
        m["refusal_correct"] = refused
        m["hallucinated"] = answered and bool(answer.strip())
    return m


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Seeded and therefore reproducible.  Returns None for an empty sample.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _metric_values(metrics: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(m[key]) for m in metrics if m.get(key) is not None]


def record_metrics(record: Mapping[str, Any], ks: Sequence[int] = DEFAULT_KS) -> dict[str, Any]:
    """Return a record's stored ``metrics`` dict, computing it if absent."""
    stored = record.get("metrics")
    if isinstance(stored, Mapping):
        return dict(stored)
    return question_metrics(record, record, ks)


def summarize_group(
    records: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarise one group of records (all questions, one category, one doc...)."""
    metrics = [record_metrics(r) for r in records]
    n_ans = sum(1 for r in records if bool(r.get("answerable", True)))

    out: dict[str, Any] = {
        "n": len(records),
        "n_answerable": n_ans,
        "n_unanswerable": len(records) - n_ans,
    }

    acc = _metric_values(metrics, "answer_correct")
    out["answer_accuracy"] = _mean(acc)
    out["answer_accuracy_ci"] = bootstrap_ci(
        acc, n_resamples=n_resamples, confidence=confidence, seed=seed
    )
    out["fact_recall"] = _mean(_metric_values(metrics, "fact_recall"))
    out["token_f1"] = _mean(_metric_values(metrics, "token_f1"))
    out["false_refusal_rate"] = _mean(_metric_values(metrics, "false_refusal"))

    out["citation_precision"] = _mean(_metric_values(metrics, "citation_precision"))
    out["citation_recall"] = _mean(_metric_values(metrics, "citation_recall"))
    out["citation_coverage"] = _mean(_metric_values(metrics, "has_citation"))
    out["citation_validity"] = _mean(_metric_values(metrics, "citation_valid"))

    rank_keys = sorted(
        {k for m in metrics for k in m if k.startswith(("recall@", "hit@"))},
        key=lambda k: (k.split("@")[0] != "recall", int(k.split("@")[1])),
    )
    for key in rank_keys:
        out[key] = _mean(_metric_values(metrics, key))
    out["mrr"] = _mean(_metric_values(metrics, "rr"))

    ref = _metric_values(metrics, "refusal_correct")
    out["refusal_accuracy"] = _mean(ref)
    out["refusal_accuracy_ci"] = bootstrap_ci(
        ref, n_resamples=n_resamples, confidence=confidence, seed=seed
    )
    out["hallucination_rate"] = _mean(_metric_values(metrics, "hallucinated"))

    out["error_rate"] = _mean(_metric_values(metrics, "error")) if records else None

    latencies = [float(r["latency_sec"]) for r in records if r.get("latency_sec") is not None]
    if latencies:
        p50, p95 = np.percentile(latencies, [50, 95])
        out["latency_p50"] = float(p50)
        out["latency_p95"] = float(p95)
        out["latency_mean"] = float(np.mean(latencies))
    else:
        out["latency_p50"] = out["latency_p95"] = out["latency_mean"] = None

    iterations = [
        r.get("iterations", r.get("agent_iterations"))
        for r in records
        if r.get("iterations", r.get("agent_iterations")) is not None
    ]
    out["mean_iterations"] = _mean(iterations)
    vram = [int(r["peak_vram_bytes"]) for r in records if r.get("peak_vram_bytes") is not None]
    out["max_peak_vram_bytes"] = max(vram) if vram else None
    return out


def _group_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r.get(key, "unknown")), []).append(r)
    return dict(sorted(groups.items()))


def aggregate(
    records: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Aggregate per-question records into a benchmark summary.

    Args:
        records: Records from :func:`~local_agentic_rag.eval.runner.run_benchmark`
            (or any dicts with question + result fields; ``metrics`` is
            computed on the fly when missing).
        n_resamples: Bootstrap resamples for the confidence intervals.
        seed: RNG seed for the bootstrap (reproducible CIs).
        confidence: CI level (default 95%).

    Returns:
        ``{"overall": {...}, "per_category": {cat: {...}}, "per_doc": {doc: {...}}}``
        where each inner dict comes from :func:`summarize_group`.  Fractions are
        in [0, 1]; ``None`` means no question in the group made the metric defined.
    """
    kw: dict[str, Any] = {"n_resamples": n_resamples, "seed": seed, "confidence": confidence}
    return {
        "overall": summarize_group(records, **kw),
        "per_category": {
            cat: summarize_group(group, **kw)
            for cat, group in _group_by(records, "category").items()
        },
        "per_doc": {
            doc: summarize_group(group, **kw) for doc, group in _group_by(records, "doc_id").items()
        },
    }
