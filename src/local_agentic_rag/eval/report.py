"""
Human-readable reports from :func:`~local_agentic_rag.eval.metrics.aggregate` summaries.

* :func:`summary_table_markdown` — several configurations side by side
  (rows = metrics, columns = config names).
* :func:`per_category_markdown` / :func:`per_doc_markdown` — one configuration
  broken down by question category / document.
* :func:`save_summary_json` / :func:`load_summary_json` — persist summaries.

Fractions are rendered as percentages with one decimal (``62.5%``); accuracy
metrics with a bootstrap CI render as ``62.5% [50.0-75.0]``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: (key, label, kind).  kind: pct | pct_ci | float3 | sec | bytes | float2 | int
METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("n", "Questions", "int"),
    ("answer_accuracy", "Answer accuracy", "pct_ci"),
    ("fact_recall", "Fact recall", "pct"),
    ("token_f1", "Token F1", "pct"),
    ("false_refusal_rate", "False refusal rate", "pct"),
    ("refusal_accuracy", "Refusal accuracy (unanswerable)", "pct_ci"),
    ("hallucination_rate", "Hallucination rate (unanswerable)", "pct"),
    ("citation_precision", "Citation precision", "pct"),
    ("citation_recall", "Citation recall", "pct"),
    ("citation_coverage", "Citation coverage", "pct"),
    ("citation_validity", "Citation validity (cited in retrieved)", "pct"),
    ("recall@1", "Page recall@1", "pct"),
    ("recall@3", "Page recall@3", "pct"),
    ("recall@5", "Page recall@5", "pct"),
    ("recall@10", "Page recall@10", "pct"),
    ("hit@5", "Hit@5", "pct"),
    ("mrr", "MRR", "float3"),
    ("error_rate", "Error rate", "pct"),
    ("latency_p50", "Latency p50", "sec"),
    ("latency_p95", "Latency p95", "sec"),
    ("latency_mean", "Latency mean", "sec"),
    ("mean_iterations", "Mean iterations", "float2"),
    ("max_peak_vram_bytes", "Peak VRAM", "bytes"),
)

DEFAULT_CATEGORY_METRICS: tuple[str, ...] = (
    "n",
    "answer_accuracy",
    "fact_recall",
    "refusal_accuracy",
    "citation_precision",
    "recall@5",
    "latency_p50",
)

_SPEC_BY_KEY = {key: (label, kind) for key, label, kind in METRIC_SPECS}
_MISSING = "n/a"


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def format_metric(key: str, value: Any, ci: Sequence[float] | None = None) -> str:
    """Format one metric value for display according to its kind."""
    if _is_missing(value):
        return _MISSING
    kind = _SPEC_BY_KEY.get(key, ("", _guess_kind(key)))[1]
    if kind in ("pct", "pct_ci"):
        text = _pct(float(value))
        if kind == "pct_ci" and ci is not None and len(ci) == 2:
            text += f" [{100 * ci[0]:.1f}-{100 * ci[1]:.1f}]"
        return text
    if kind == "float3":
        return f"{float(value):.3f}"
    if kind == "float2":
        return f"{float(value):.2f}"
    if kind == "sec":
        return f"{float(value):.2f}s"
    if kind == "bytes":
        return f"{int(value) / 1024**3:.2f} GiB"
    if kind == "int":
        return str(int(value))
    return str(value)


def _guess_kind(key: str) -> str:
    if key.startswith(("recall@", "hit@")) or key.endswith(("_rate", "_accuracy")):
        return "pct"
    return "float3"


def _label(key: str) -> str:
    return _SPEC_BY_KEY.get(key, (key, ""))[0]


def _group(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a full :func:`aggregate` output or a single group summary."""
    overall = summary.get("overall")
    return overall if isinstance(overall, Mapping) else summary


def _cell(group: Mapping[str, Any], key: str) -> str:
    return format_metric(key, group.get(key), group.get(f"{key}_ci"))


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    out = [line(header), "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def summary_table_markdown(
    summaries: Mapping[str, Mapping[str, Any]],
    metrics: Sequence[str] | None = None,
) -> str:
    """Markdown table comparing configurations: rows = metrics, columns = config names.

    Args:
        summaries: ``{config_name: aggregate(records)}`` (or the ``overall`` dicts).
        metrics: Metric keys to show (default: :data:`METRIC_SPECS` rows that
            are present in at least one summary).
    """
    groups = {name: _group(s) for name, s in summaries.items()}
    if metrics is None:
        metrics = [k for k, _, _ in METRIC_SPECS if any(k in g for g in groups.values())]
    rows = [[_label(k)] + [_cell(g, k) for g in groups.values()] for k in metrics]
    return _table(["Metric", *groups.keys()], rows)


def per_group_markdown(
    summary: Mapping[str, Any],
    group: str = "per_category",
    metrics: Sequence[str] | None = None,
) -> str:
    """Markdown breakdown of one configuration: rows = groups, columns = metrics.

    Args:
        summary: Output of :func:`aggregate`.
        group: ``"per_category"`` or ``"per_doc"``.
        metrics: Metric keys to show (default: :data:`DEFAULT_CATEGORY_METRICS`).
    """
    groups = summary.get(group)
    if not isinstance(groups, Mapping):
        raise KeyError(f"summary has no {group!r} section; pass the output of aggregate()")
    metrics = list(metrics or DEFAULT_CATEGORY_METRICS)
    first = "Category" if group == "per_category" else "Document"
    rows = [[str(name)] + [_cell(g, k) for k in metrics] for name, g in groups.items()]
    overall = summary.get("overall")
    if isinstance(overall, Mapping):
        rows.append(["**overall**"] + [_cell(overall, k) for k in metrics])
    return _table([first, *(_label(k) for k in metrics)], rows)


def per_category_markdown(summary: Mapping[str, Any], metrics: Sequence[str] | None = None) -> str:
    """Per-category breakdown table for one configuration."""
    return per_group_markdown(summary, "per_category", metrics)


def per_doc_markdown(summary: Mapping[str, Any], metrics: Sequence[str] | None = None) -> str:
    """Per-document breakdown table for one configuration."""
    return per_group_markdown(summary, "per_doc", metrics)


def save_summary_json(
    summary: Mapping[str, Any], path: str | Path, *, metadata: Mapping[str, Any] | None = None
) -> Path:
    """Write a summary (optionally wrapped with run ``metadata``) as pretty JSON.

    Returns the written path.  Parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Any = {"metadata": dict(metadata), "summary": summary} if metadata else summary
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def load_summary_json(path: str | Path) -> dict[str, Any]:
    """Load a summary written by :func:`save_summary_json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)
