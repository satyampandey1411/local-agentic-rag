#!/usr/bin/env python3
"""Offline measurement of the F11 presupposition gate (no LLM, no models).

For every question of a development set, extract the presupposed entities from the question
and test whether any of them is absent from the FULL text of its document.  The full
document is a proxy for the retrieved passages: an entity missing from the whole document
is certainly missing from the passages, but the real gate also fires on entities that are in
the document and merely not retrieved, so these counts are a lower bound on real fires.

    python benchmark/f11_offline.py [--scale-dir DIR ...] [--out docs/f11_offline.md]

Development data only: benchmark/questions.jsonl, benchmark/multilingual/ and the synthetic
corpora from benchmark/scale/make_corpus.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pymupdf  # noqa: E402

from local_agentic_rag.agent.presupposition import extract_entities, missing_entities  # noqa: E402


def doc_text(pdf: Path) -> tuple[str, int, int]:
    """(text of all pages, pages, pages with no text layer)."""
    with pymupdf.open(pdf) as doc:
        pages = [p.get_text() for p in doc]
    return "\n".join(pages), len(pages), sum(1 for t in pages if len(t.strip()) < 20)


def load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def measure(name: str, questions: list[dict], texts: dict[str, str]) -> list[dict]:
    rows = []
    for q in questions:
        text = texts.get(q["doc_id"])
        if text is None:
            continue
        ents = extract_entities(q["question"])
        miss = missing_entities(ents, [text])
        rows.append(
            {
                "set": name,
                "q_id": q["q_id"],
                "answerable": bool(q["answerable"]),
                "type": q.get("type") or q.get("category", ""),
                "question": q["question"],
                "entities": [(e.kind, e.text) for e in ents],
                "missing": [(e.kind, e.text) for e in miss],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-dir", action="append", default=[], type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "f11_offline.md")
    ap.add_argument("--json", type=Path, default=None, help="also dump per-question rows")
    args = ap.parse_args()

    rows: list[dict] = []
    notes: list[str] = []

    for label, qdir, pdfdir in (
        ("english", ROOT / "benchmark", ROOT / "benchmark" / "pdfs"),
        (
            "multilingual",
            ROOT / "benchmark" / "multilingual",
            ROOT / "benchmark" / "multilingual" / "pdfs",
        ),
    ):
        texts = {}
        for pdf in sorted(pdfdir.glob("*.pdf")):
            texts[pdf.stem], n, blank = doc_text(pdf)
            if blank:
                notes.append(
                    f"{label}/{pdf.name}: {blank} of {n} pages have no text layer (OCR-only)"
                )
        rows += measure(label, load(qdir / "questions.jsonl"), texts)

    for sdir in args.scale_dir:
        for qfile in sorted(sdir.glob("scale-*.questions.jsonl")):
            stem = qfile.name.split(".")[0]
            pdf = sdir / f"{stem}.pdf"
            text, n, blank = doc_text(pdf)
            notes.append(
                f"{sdir.name}/{pdf.name}: {blank} of {n} pages have no text layer (scanned)"
            )
            rows += measure(f"synthetic {sdir.name}/{stem}", load(qfile), {stem: text})

    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    def group(r: dict) -> str:
        return "synthetic" if r["set"].startswith("synthetic") else r["set"]

    lines = [
        "| set | answerable q | with an entity | fires | unanswerable q | with an entity | caught |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in ("english", "multilingual", "synthetic"):
        a = [r for r in rows if group(r) == g and r["answerable"]]
        u = [r for r in rows if group(r) == g and not r["answerable"]]
        lines.append(
            f"| {g} | {len(a)} | {sum(bool(r['entities']) for r in a)} | **{sum(bool(r['missing']) for r in a)}** "
            f"| {len(u)} | {sum(bool(r['entities']) for r in u)} | **{sum(bool(r['missing']) for r in u)}** |"
        )
    kinds = Counter(k for r in rows if r["answerable"] for k, _ in r["entities"])
    kinds_u = Counter(k for r in rows if not r["answerable"] for k, _ in r["missing"])
    print("\n".join(lines))
    print("entities extracted from answerable questions:", dict(kinds))
    print("missing entities on unanswerable questions:", dict(kinds_u))
    for n in notes:
        print("note:", n)
    for r in rows:
        if r["answerable"] and r["missing"]:
            print("FALSE FIRE", r["set"], r["q_id"], r["missing"], "|", r["question"])
    for r in rows:
        if not r["answerable"]:
            print(
                "UNANS",
                "CAUGHT" if r["missing"] else "miss  ",
                r["set"],
                r["q_id"],
                r["entities"],
                "|",
                r["question"],
            )
    out = {"table": lines, "kinds": dict(kinds), "kinds_u": dict(kinds_u), "notes": notes}
    (ROOT / "local").mkdir(exist_ok=True)
    (ROOT / "local" / "f11_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
