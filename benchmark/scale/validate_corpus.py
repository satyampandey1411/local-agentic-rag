#!/usr/bin/env python3
"""Validate a scale-benchmark corpus directory written by ``make_corpus.py``.

    python benchmark/scale/validate_corpus.py DIR [--doc scale-1000]

For every ``scale-N.pdf`` in DIR (or only ``--doc``) this checks, using nothing but
PyMuPDF text extraction (``page.get_text()`` with whitespace collapsed):

* the PDF has exactly N pages, matching the file name and the meta file;
* the meta file's scanned pages really are image-only (no text layer, >= 1 image);
* every answerable question's answer keys occur on its gold page(s):
    - lookup / table_cell / section_scoped / page_ref / distractor: every key on every
      gold page;
    - multi_hop: every key on at least one gold page, and the bridging entity on all;
    - numeric: each operand's text occurs on its stated page, the gold pages are
      exactly the operand pages, and ``numeric_answer`` equals the operation applied
      to the operands (the computed answer itself is, by design, not in the text);
    - ocr: every gold page is image-only, and the OCR-only text is absent from the
      text layer of the whole document (so the fact is reachable only through OCR);
* page_ref table questions: exactly one table caption on the gold page;
* every unanswerable question's ``absent_keys`` occur nowhere in the document, and
  no page contains both members of an ``absent_cooccur`` pair (entity + year as a
  whole word, ignoring the "Page N" footer);
* ids are unique, required fields are present, depths are within [0, 1].

Prints a summary per document and exits non-zero if anything fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

REQUIRED = ("q_id", "doc_id", "question", "type", "answerable", "reference_answer",
            "gold_pages", "answer_keys", "numeric_answer", "depth")
TYPES = {"lookup", "table_cell", "numeric", "multi_hop", "section_scoped", "page_ref", "ocr",
         "unanswerable", "distractor"}
_FOOTER_RE = re.compile(r"\bPage \d+\b")
_CAPTION_RE = re.compile(r"\bTable \d+\.\d+:")


def norm(text: str) -> str:
    return " ".join(text.split())


def validate_doc(pdf_path: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    stem = pdf_path.name[: -len(".pdf")]
    expected_pages = int(stem.split("-")[-1])
    qpath = pdf_path.with_name(f"{stem}.questions.jsonl")
    mpath = pdf_path.with_name(f"{stem}.meta.json")
    meta = json.loads(mpath.read_text()) if mpath.exists() else {}
    if not meta:
        errors.append(f"missing meta file {mpath.name}")
    questions = [json.loads(line) for line in qpath.read_text().splitlines() if line.strip()]

    doc = pymupdf.open(pdf_path)
    n = doc.page_count
    if n != expected_pages:
        errors.append(f"page count {n} != {expected_pages}")
    if meta and meta.get("pages") != n:
        errors.append(f"meta pages {meta.get('pages')} != {n}")
    texts = [norm(page.get_text()) for page in doc]
    image_only = [not texts[i] and bool(doc[i].get_images(full=False)) for i in range(n)]
    whole = "\n".join(texts)

    for p in meta.get("scanned_pages", []):
        if not (1 <= p <= n and image_only[p - 1]):
            errors.append(f"scanned page {p} is not image-only")

    def on(page: int, key: str) -> bool:
        return 1 <= page <= n and key in texts[page - 1]

    ids = Counter(q.get("q_id") for q in questions)
    errors += [f"duplicate q_id {k}" for k, c in ids.items() if c > 1]
    for q in questions:
        qid = q.get("q_id", "?")
        missing = [f for f in REQUIRED if f not in q]
        if missing:
            errors.append(f"{qid}: missing fields {missing}")
            continue
        if q["type"] not in TYPES:
            errors.append(f"{qid}: unknown type {q['type']}")
        if q["doc_id"] != stem:
            errors.append(f"{qid}: doc_id {q['doc_id']} != {stem}")
        if not 0.0 <= q["depth"] <= 1.0:
            errors.append(f"{qid}: depth {q['depth']} outside [0, 1]")
        gold = q["gold_pages"]
        if any(not 1 <= p <= n for p in gold):
            errors.append(f"{qid}: gold page out of range {gold}")
            continue
        qtype = q["type"]

        if not q["answerable"]:
            if gold or q["answer_keys"]:
                errors.append(f"{qid}: unanswerable question has gold pages / keys")
            if not q.get("absent_keys") and not q.get("absent_cooccur"):
                errors.append(f"{qid}: unanswerable question has nothing to check")
            for key in q.get("absent_keys", []):
                if key in whole:
                    pages = [i + 1 for i, t in enumerate(texts) if key in t][:5]
                    errors.append(f"{qid}: absent key {key!r} occurs on pages {pages}")
            for entity, year in q.get("absent_cooccur", []):
                year_re = re.compile(rf"\b{re.escape(year)}\b")
                for i, t in enumerate(texts):
                    if entity in t and year_re.search(_FOOTER_RE.sub(" ", t)):
                        errors.append(f"{qid}: {entity!r} and {year} co-occur on page {i + 1}")
                        break
            continue

        if not gold:
            errors.append(f"{qid}: answerable question without gold pages")
            continue
        keys = q["answer_keys"]
        if not keys:
            errors.append(f"{qid}: answerable question without answer keys")
        if qtype == "ocr":
            for p in gold:
                if not image_only[p - 1]:
                    errors.append(f"{qid}: OCR gold page {p} has a text layer or no image")
            ocr_text = q.get("ocr_text")
            if ocr_text and ocr_text in whole:
                errors.append(f"{qid}: OCR-only text {ocr_text!r} found in the text layer")
        elif qtype == "numeric":
            ops = q.get("operands") or []
            if len(ops) < 2:
                errors.append(f"{qid}: numeric question needs >= 2 operands")
                continue
            for op in ops:
                if not on(op["page"], op["text"]):
                    errors.append(f"{qid}: operand {op['text']!r} not on page {op['page']}")
            if sorted({op["page"] for op in ops}) != sorted(set(gold)):
                errors.append(f"{qid}: gold pages {gold} != operand pages")
            vals = [op["value"] for op in ops]
            expected = {"sum": sum(vals), "difference": vals[-1] - vals[0]}.get(q.get("operation"))
            if expected is None or abs(round(expected, 6) - q["numeric_answer"]) > 1e-6:
                errors.append(f"{qid}: numeric_answer {q['numeric_answer']} != {expected}")
        elif qtype == "multi_hop":
            if len(set(gold)) < 2:
                errors.append(f"{qid}: multi_hop question needs two distinct gold pages")
            for key in keys:
                if not any(on(p, key) for p in gold):
                    errors.append(f"{qid}: key {key!r} on none of the gold pages {gold}")
            bridge = q.get("bridge")
            if not bridge or not all(on(p, bridge) for p in gold):
                errors.append(f"{qid}: bridge {bridge!r} missing from a gold page {gold}")
        else:
            for p in gold:
                for key in keys:
                    if not on(p, key):
                        errors.append(f"{qid}: key {key!r} not extractable from gold page {p}")
            if qtype == "page_ref" and q.get("table"):
                caps = _CAPTION_RE.findall(texts[gold[0] - 1])
                if len(caps) != 1:
                    errors.append(f"{qid}: page {gold[0]} has {len(caps)} table captions")

    summary = {
        "doc": stem, "pages": n, "questions": len(questions),
        "types": dict(Counter(q.get("type") for q in questions)),
        "image_only_pages": sum(image_only), "scanned_in_meta": len(meta.get("scanned_pages", [])),
        "errors": len(errors),
    }
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate scale-benchmark corpus files.")
    ap.add_argument("dir", type=Path)
    ap.add_argument("--doc", help="validate only this document stem, e.g. scale-1000")
    args = ap.parse_args(argv)
    pdfs = sorted(args.dir.glob("scale-*.pdf"), key=lambda p: int(p.stem.split("-")[-1]))
    if args.doc:
        pdfs = [p for p in pdfs if p.stem == args.doc]
    if not pdfs:
        print(f"no scale-*.pdf files found in {args.dir}", file=sys.stderr)
        return 2
    failed = False
    for pdf in pdfs:
        errors, s = validate_doc(pdf)
        status = "OK" if not errors else "FAIL"
        print(f"{status} {s['doc']}: {s['pages']} pages, {s['questions']} questions "
              f"{s['types']}, image-only pages {s['image_only_pages']} "
              f"(meta scanned {s['scanned_in_meta']})")
        for e in errors[:50]:
            print(f"    - {e}")
        if len(errors) > 50:
            print(f"    ... {len(errors) - 50} more")
        failed |= bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
