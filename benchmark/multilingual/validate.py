#!/usr/bin/env python3
"""Validate the multilingual benchmark against the real PDF text.

    .venv/bin/python benchmark/multilingual/validate.py

Checks
  1. documents.json / questions.jsonl schema (incl. question_language).
  2. every PDF exists, has the recorded page count and SHA-256.
  3. every answerable question: for each required fact, at least one alias occurs
     on the union of its evidence pages (NFC + lowercase + whitespace collapsed,
     commas inside numbers removed). Devanagari text layers sometimes split a word
     at an i-matra ("स्थि त"), so a second pass also compares with all whitespace removed.
  4. scanned (derived) docs: the PDF really has no text layer, and the facts are
     checked on the corresponding pages of the source document instead.

Exit 0 only when there are no errors. Warnings (e.g. a fact that matched only in
the whitespace-free pass) do not fail the run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
PDF_DIRS = [HERE / "pdfs", HERE.parent / "pdfs"]   # multilingual first, then main benchmark
LANGS = {"en", "hi", "hinglish", "mr"}
CATEGORIES = {"factual", "table", "numerical", "multi_hop", "comparison", "definition", "unanswerable"}
DOC_KEYS = {"doc_id", "title", "url", "sha256", "num_pages", "doc_type", "source_license_note"}
Q_KEYS = {"q_id", "doc_id", "category", "question", "answerable", "reference_answer",
          "required_facts", "evidence_pages", "notes", "question_language"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def nospace(s: str) -> str:
    return re.sub(r"\s+", "", s)


def find_pdf(doc_id: str) -> Path | None:
    for d in PDF_DIRS:
        p = d / f"{doc_id}.pdf"
        if p.exists():
            return p
    return None


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    docs = json.loads((HERE / "documents.json").read_text(encoding="utf-8"))
    by_id = {}
    for d in docs:
        missing = DOC_KEYS - d.keys()
        if missing:
            errors.append(f"doc {d.get('doc_id')}: missing keys {sorted(missing)}")
        by_id[d["doc_id"]] = d

    pages: dict[str, list[str]] = {}
    for doc_id, d in by_id.items():
        p = find_pdf(doc_id)
        if p is None:
            errors.append(f"{doc_id}: PDF not found in {[str(x) for x in PDF_DIRS]}")
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != d["sha256"]:
            errors.append(f"{doc_id}: sha256 {h[:12]}... != manifest {d['sha256'][:12]}... "
                          "(regenerated with a different LibreOffice/font version?)")
        doc = pymupdf.open(p)
        if doc.page_count != d["num_pages"]:
            errors.append(f"{doc_id}: {doc.page_count} pages != manifest {d['num_pages']}")
        pages[doc_id] = [norm(pg.get_text()) for pg in doc]
        if "derived_from" in d and any(pages[doc_id]):
            errors.append(f"{doc_id}: expected an image-only PDF but found a text layer")

    qs = [json.loads(line) for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    checked = 0
    for q in qs:
        qid = q.get("q_id", "?")
        if set(q) != Q_KEYS:
            errors.append(f"{qid}: keys differ: missing {sorted(Q_KEYS - set(q))}, extra {sorted(set(q) - Q_KEYS)}")
            continue
        if qid in seen:
            errors.append(f"{qid}: duplicate q_id")
        seen.add(qid)
        if q["doc_id"] not in by_id:
            errors.append(f"{qid}: unknown doc_id {q['doc_id']}")
            continue
        if q["question_language"] not in LANGS:
            errors.append(f"{qid}: bad question_language {q['question_language']}")
        if q["category"] not in CATEGORIES:
            errors.append(f"{qid}: bad category {q['category']}")
        if q["answerable"] != (q["category"] != "unanswerable"):
            errors.append(f"{qid}: answerable flag inconsistent with category")
        if not q["answerable"]:
            if q["required_facts"] or q["evidence_pages"]:
                errors.append(f"{qid}: unanswerable must have empty facts/pages")
            continue
        if not q["required_facts"] or not q["evidence_pages"]:
            errors.append(f"{qid}: answerable but no facts/pages")
            continue
        d = by_id[q["doc_id"]]
        text_doc = d.get("derived_from", q["doc_id"])
        if text_doc not in pages:
            continue
        n = by_id[q["doc_id"]]["num_pages"]
        bad = [p for p in q["evidence_pages"] if not (1 <= p <= n)]
        if bad:
            errors.append(f"{qid}: evidence pages {bad} out of range 1..{n}")
            continue
        src_pages = [d.get("derived_pages", list(range(1, n + 1)))[p - 1] for p in q["evidence_pages"]]
        text = " ".join(pages[text_doc][p - 1] for p in src_pages)
        for fact in q["required_facts"]:
            aliases = [norm(a) for a in fact]
            if any(a in text for a in aliases):
                continue
            if any(nospace(a) in nospace(text) for a in aliases):
                warnings.append(f"{qid}: fact {fact[:2]} matched only after removing whitespace")
                continue
            errors.append(f"{qid}: no alias of {fact} on {text_doc} pages {src_pages}")
        checked += 1

    from collections import Counter
    print(f"documents: {len(docs)}  questions: {len(qs)}  answerable fact-checked: {checked}")
    print("by doc:", dict(Counter(q["doc_id"] for q in qs)))
    print("by question_language:", dict(Counter(q.get("question_language") for q in qs)))
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"FAILED: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
