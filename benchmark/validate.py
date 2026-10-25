#!/usr/bin/env python3
"""Validate the benchmark ground truth against the actual PDFs.

Checks
------
Hard failures (exit code 1):
  * documents.json / questions.jsonl schema, allowed categories, unique q_ids,
    q_id prefix matches doc_id, doc_id exists.
  * Each PDF exists, its SHA-256 matches documents.json and its page count
    equals num_pages.
  * evidence_pages are 1-indexed PHYSICAL pages (page index + 1) inside the PDF.
  * Answerable questions: non-empty required_facts and evidence_pages; for every
    category except "numerical", each required fact must have at least one alias
    present in the normalized text of the union of the evidence pages.
  * Unanswerable questions: answerable=false, empty facts/pages, canonical answer.

Warnings (reviewed manually, do not fail the run):
  * Unanswerable question whose distinctive keywords all occur on one page
    (heuristic signal that it may actually be answerable).
  * A quoted snippet in "notes" that cannot be found on the evidence pages.

Normalization: Unicode NFKC (expands ligatures such as "fi"), curly quotes and
dashes mapped to ASCII, lowercase, whitespace collapsed. An alias also matches
if it appears after removing thousands separators from numbers (1,234 -> 1234).

Usage:
    python benchmark/validate.py [--pdf-dir benchmark/pdfs]
Requires PyMuPDF (``pip install pymupdf``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf  # PyMuPDF

HERE = Path(__file__).resolve().parent
CATEGORIES = {"factual", "table", "numerical", "multi_hop", "comparison",
              "definition", "unanswerable"}
UNANSWERABLE_ANSWER = "Not stated in the document."
DOC_FIELDS = {"doc_id": str, "title": str, "url": str, "sha256": str,
              "num_pages": int, "doc_type": str, "source_license_note": str}
Q_FIELDS = {"q_id": str, "doc_id": str, "category": str, "question": str,
            "answerable": bool, "reference_answer": str, "required_facts": list,
            "evidence_pages": list, "notes": str}

_TRANS = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "−": "-", " ": " ", "­": "",
})
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")

STOPWORDS = set("""
a an the of in on at to for from by with and or but if then than that this these those
is are was were be been being do does did done have has had what which who whom whose
when where why how many much did does it its as into about over under between after before
during per each any all some such not no nor can could would should may might will shall
according paper document report profile release study authors author describe described
use used using show shown give given value values number amount rate figure table their
they them there here also more most less least other than only same both either
achieve achieved reach reached report reported took take part people follow impose
require required list listed mention mentioned state stated provide provided
""".split())  # noqa: SIM905


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_TRANS).lower()
    return re.sub(r"\s+", " ", s).strip()


def strip_num_commas(s: str) -> str:
    return _NUM_COMMA.sub("", s)


def alias_in(alias: str, text: str, text_nocomma: str) -> bool:
    a = norm(alias)
    return bool(a) and (a in text or strip_num_commas(a) in text_nocomma)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def type_errors(obj: dict, spec: dict, where: str) -> list[str]:
    errs = []
    for k, t in spec.items():
        if k not in obj:
            errs.append(f"{where}: missing field '{k}'")
        elif t is int and isinstance(obj[k], bool) or not isinstance(obj[k], t):
            errs.append(f"{where}: field '{k}' should be {t.__name__}")
    extra = set(obj) - set(spec)
    if extra:
        errs.append(f"{where}: unexpected fields {sorted(extra)}")
    return errs


def keywords(question: str) -> list[str]:
    q = re.sub(r"'s\b", "", norm(question))  # drop possessives: "paper's" -> "paper"
    words = re.findall(r"[a-z0-9][a-z0-9\-\.']*[a-z0-9]", q)
    return sorted({w for w in words if len(w) >= 4 and w not in STOPWORDS})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate benchmark ground truth.")
    ap.add_argument("--documents", type=Path, default=HERE / "documents.json")
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    ap.add_argument("--pdf-dir", type=Path, default=HERE / "pdfs")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every question")
    args = ap.parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------ documents
    docs = json.loads(args.documents.read_text(encoding="utf-8"))
    pages: dict[str, list[str]] = {}       # doc_id -> normalized text per page
    for i, d in enumerate(docs):
        where = f"documents[{i}]"
        errs = type_errors(d, DOC_FIELDS, where)
        errors += errs
        if errs:
            continue
        doc_id = d["doc_id"]
        if doc_id in pages:
            errors.append(f"{where}: duplicate doc_id {doc_id}")
            continue
        pdf = args.pdf_dir / f"{doc_id}.pdf"
        if not pdf.exists():
            errors.append(f"{doc_id}: PDF not found at {pdf} (run fetch_pdfs.py)")
            continue
        got = sha256_of(pdf)
        if got != d["sha256"].lower():
            errors.append(f"{doc_id}: sha256 mismatch ({got[:12]}... != {d['sha256'][:12]}...)")
        with pymupdf.open(pdf) as doc:
            if doc.page_count != d["num_pages"]:
                errors.append(f"{doc_id}: num_pages {d['num_pages']} != actual {doc.page_count}")
            pages[doc_id] = [norm(p.get_text()) for p in doc]

    # ------------------------------------------------------------ questions
    questions = []
    for ln, line in enumerate(args.questions.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            questions.append((ln, json.loads(line)))
        except json.JSONDecodeError as e:
            errors.append(f"questions.jsonl line {ln}: invalid JSON ({e})")

    seen = Counter(q.get("q_id") for _, q in questions)
    for qid, n in seen.items():
        if n > 1:
            errors.append(f"duplicate q_id {qid} ({n}x)")

    by_doc_cat: Counter = Counter()
    checked_facts = 0
    for ln, q in questions:
        qid = q.get("q_id", f"line{ln}")
        errs = type_errors(q, Q_FIELDS, qid)
        errors += errs
        if errs:
            continue
        doc_id, cat = q["doc_id"], q["category"]
        by_doc_cat[(doc_id, cat)] += 1
        if cat not in CATEGORIES:
            errors.append(f"{qid}: unknown category '{cat}'")
        if not re.fullmatch(re.escape(doc_id) + r"-q\d{2}", qid):
            errors.append(f"{qid}: q_id must be '<doc_id>-qNN'")
        if not q["question"].strip() or not q["reference_answer"].strip():
            errors.append(f"{qid}: empty question or reference_answer")
        facts, evp = q["required_facts"], q["evidence_pages"]
        if not all(isinstance(f, list) and f and all(isinstance(a, str) and a.strip() for a in f)
                   for f in facts):
            errors.append(f"{qid}: required_facts must be a list of non-empty lists of strings")
            continue
        if not all(isinstance(p, int) and not isinstance(p, bool) for p in evp):
            errors.append(f"{qid}: evidence_pages must be integers")
            continue
        if doc_id not in pages:
            if doc_id not in {d.get("doc_id") for d in docs}:
                errors.append(f"{qid}: unknown doc_id {doc_id}")
            continue
        npages = len(pages[doc_id])
        bad = [p for p in evp if not 1 <= p <= npages]
        if bad:
            errors.append(f"{qid}: evidence pages {bad} outside 1..{npages}")
            continue

        is_unans = cat == "unanswerable"
        if is_unans != (not q["answerable"]):
            errors.append(f"{qid}: answerable must be false iff category is 'unanswerable'")
        if not q["answerable"]:
            if facts or evp:
                errors.append(f"{qid}: unanswerable question must have empty facts and pages")
            if q["reference_answer"] != UNANSWERABLE_ANSWER:
                errors.append(f"{qid}: unanswerable reference_answer must be '{UNANSWERABLE_ANSWER}'")
            kws = keywords(q["question"])
            hits = [i + 1 for i, t in enumerate(pages[doc_id]) if all(k in t for k in kws)]
            if kws and hits:
                warnings.append(f"{qid}: all keywords {kws} co-occur on page(s) {hits} - "
                                f"review that it is truly unanswerable")
            if args.verbose:
                print(f"ok   {qid} [unanswerable] keywords={kws}")
            continue

        if not facts or not evp:
            errors.append(f"{qid}: answerable question needs required_facts and evidence_pages")
            continue
        text = " ".join(pages[doc_id][p - 1] for p in sorted(set(evp)))
        text_nc = strip_num_commas(text)
        if cat != "numerical":
            for fact in facts:
                checked_facts += 1
                if not any(alias_in(a, text, text_nc) for a in fact):
                    errors.append(f"{qid}: none of the aliases {fact} found on pages {evp}")
        for snip in re.findall(r'"([^"]{8,})"', q["notes"]):
            # "..." marks an elision: every fragment must appear on the evidence pages
            frags = [norm(f) for f in snip.split("...") if len(f.strip()) >= 4]
            if any(f not in text and strip_num_commas(f) not in text_nc for f in frags):
                warnings.append(f"{qid}: notes snippet not found verbatim on pages {evp}: "
                                f'"{snip[:60]}"')
        if args.verbose:
            print(f"ok   {qid} [{cat}] pages={evp}")

    # ------------------------------------------------------------ report
    print("\nCategory counts")
    cats = sorted(CATEGORIES)
    doc_ids = [d["doc_id"] for d in docs if isinstance(d, dict) and "doc_id" in d]
    print(f"{'category':<14}" + "".join(f"{d[:18]:>20}" for d in doc_ids) + f"{'total':>8}")
    for c in cats:
        row = [by_doc_cat[(d, c)] for d in doc_ids]
        print(f"{c:<14}" + "".join(f"{n:>20}" for n in row) + f"{sum(row):>8}")
    tot = [sum(by_doc_cat[(d, c)] for c in cats) for d in doc_ids]
    print(f"{'TOTAL':<14}" + "".join(f"{n:>20}" for n in tot) + f"{sum(tot):>8}")

    print(f"\nquestions: {len(questions)}   documents: {len(docs)}   "
          f"facts checked against evidence pages: {checked_facts}")
    for w in warnings:
        print(f"WARNING {w}")
    for e in errors:
        print(f"ERROR   {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    print("VALIDATION PASSED" if not errors else "VALIDATION FAILED")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
