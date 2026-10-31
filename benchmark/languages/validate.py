#!/usr/bin/env python3
"""Validate the many-language benchmark against the real PDF text.

    .venv/bin/python benchmark/languages/validate.py            # dev set
    .venv/bin/python benchmark/languages/validate.py --heldout  # sealed held-out set
    .venv/bin/python benchmark/languages/validate.py --set more       # questions_more.jsonl on the dev docs
    .venv/bin/python benchmark/languages/validate.py --set heldout2   # second sealed held-out set
    .venv/bin/python benchmark/languages/validate.py --set special    # mixed-language doc (+ Chromium Arabic)

Checks
  1. documents.json / questions.jsonl schema, languages and combo rules
     (e.g. ``same`` is asked in the document's language, ``answer-in`` asks for another language,
     ``library`` / ``two-books`` search ``"all"``).
  2. every PDF exists with the recorded page count and SHA-256.
  3. round-trip: every text PDF's ``get_text()`` (NFC, whitespace removed) recovers at least
     ROUNDTRIP_MIN of its source characters. A document that does not round-trip may not be
     used as evidence (directly or through a scanned copy).
  4. scanned (derived) docs really have no text layer.
  5. every answerable question: each required fact has an alias on the union of its evidence
     pages, using the grader's own matcher (``local_agentic_rag.eval.metrics.alias_matches``).
     Scanned docs are checked on the source document's pages. The text layer sometimes splits a
     word ("तेन्जि ङ", "2012 年2 月"), so a second pass compares with all whitespace removed and
     reports a warning.
  6. two-books questions: at least two evidence docs in different languages, each holding a fact.
  7. (more/heldout2/special) ``romanized``: a Latin-script question (hinglish, roman-ur, roman-ta,
     roman-bn) on a document in another script; ``mixed``: a question on the Hindi-English document.
     ``--set more`` also refuses q_ids that are already in questions.jsonl.

Exit 0 only when there are no errors. Warnings do not fail the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
from make_pdfs import ROUNDTRIP_MIN, roundtrip_score  # noqa: E402

from local_agentic_rag.eval.metrics import alias_matches, normalize_text  # noqa: E402

DOC_LANGS = {"hi", "mr", "sa", "ne", "bh", "awa", "mai", "ur", "bn", "te", "ta", "kn", "ml", "gu",
             "pa", "or", "en", "fr", "ar", "zh", "ja", "ru", "hi-en"}
ROMANIZED = {"hinglish", "roman-ur", "roman-ta", "roman-bn"}
Q_LANGS = (DOC_LANGS - {"hi-en"}) | ROMANIZED
SAME_SCRIPT = {"hinglish": "hi", "roman-ur": "ur", "roman-ta": "ta", "roman-bn": "bn"}
# writing system of a language (Devanagari and Perso-Arabic are shared by several languages)
SCRIPT = {**dict.fromkeys(("hi", "mr", "sa", "ne", "bh", "awa", "mai", "hi-en"), "deva"), "ur": "arab", "ar": "arab"}
FIXED_COMBOS = {"same", "cross", "unanswerable", "answer-in", "library", "two-books", "scanned", "romanized", "mixed"}
# set -> (directory, questions file)
SETS = {"dev": (".", "questions.jsonl"), "heldout": ("heldout", "questions.jsonl"),
        "more": (".", "questions_more.jsonl"), "heldout2": ("heldout2", "questions.jsonl"),
        "special": ("special", "questions.jsonl")}
CATEGORIES = {"factual", "table", "numerical", "multi_hop", "comparison", "definition", "unanswerable"}
DOC_KEYS = {"doc_id", "title", "url", "sha256", "num_pages", "doc_type", "source_license_note", "language"}
Q_KEYS = {"q_id", "doc_id", "category", "question", "answerable", "reference_answer", "required_facts",
          "evidence_pages", "notes", "question_language", "answer_language", "doc_ids", "evidence", "combo"}


def nospace(s: str) -> str:
    return "".join(s.split())


def check_docs(base: Path, docs: list[dict], errors: list[str]) -> tuple[dict, dict]:
    by_id = {d["doc_id"]: d for d in docs}
    pages: dict[str, list[str]] = {}
    for d in docs:
        doc_id = d["doc_id"]
        missing = DOC_KEYS - d.keys()
        if missing:
            errors.append(f"doc {doc_id}: missing keys {sorted(missing)}")
        if d.get("language") not in DOC_LANGS:
            errors.append(f"doc {doc_id}: bad language {d.get('language')}")
        p = base / "pdfs" / f"{doc_id}.pdf"
        if not p.exists():
            errors.append(f"{doc_id}: PDF not found at {p}")
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != d["sha256"]:
            errors.append(f"{doc_id}: sha256 {h[:12]}... != manifest {d['sha256'][:12]}... (rebuilt with "
                          "another LibreOffice/font version? run make_pdfs.py and re-check evidence)")
        pdf = pymupdf.open(p)
        if pdf.page_count != d["num_pages"]:
            errors.append(f"{doc_id}: {pdf.page_count} pages != manifest {d['num_pages']}")
        raw = [pg.get_text() for pg in pdf]
        if "derived_from" in d:
            if any(t.strip() for t in raw):
                errors.append(f"{doc_id}: expected an image-only PDF but found a text layer")
            src = by_id.get(d["derived_from"])
            if src is None or len(d.get("derived_pages", [])) != d["num_pages"]:
                errors.append(f"{doc_id}: bad derived_from/derived_pages")
            continue
        pages[doc_id] = [normalize_text(t) for t in raw]
        src_text = (base / d["source_text"]).read_text(encoding="utf-8")
        # share of the source characters recovered; extra PDF text (title, footer) does not count
        score = roundtrip_score(src_text, "".join(raw))
        ok = score >= d.get("roundtrip_min", ROUNDTRIP_MIN)
        d["_roundtrip"] = (score, ok)
        if ok != d.get("roundtrip_ok"):
            errors.append(f"{doc_id}: round-trip {score:.3f} disagrees with manifest roundtrip_ok={d.get('roundtrip_ok')}")
    return by_id, pages


def text_doc(by_id: dict, doc_id: str) -> tuple[str, list[int]]:
    d = by_id[doc_id]
    if "derived_from" in d:
        return d["derived_from"], d["derived_pages"]
    return doc_id, list(range(1, d["num_pages"] + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heldout", action="store_true", help="validate heldout/ instead of the dev set (= --set heldout)")
    ap.add_argument("--set", choices=sorted(SETS), default="dev", help="which set to validate (default: dev)")
    args = ap.parse_args()
    which = "heldout" if args.heldout else args.set
    sub, qfile = SETS[which]
    base = HERE / sub if sub != "." else HERE
    errors: list[str] = []
    warnings: list[str] = []
    docs = json.loads((base / "documents.json").read_text(encoding="utf-8"))
    by_id, pages = check_docs(base, docs, errors)
    library = sorted(d for d in by_id if "derived_from" not in by_id[d])

    qs = [json.loads(line) for line in (base / qfile).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    if which == "more":   # the extra questions must not reuse an id (or a question) of the published set
        dev = [json.loads(line) for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        dev_ids, dev_q = {q["q_id"] for q in dev}, {q["question"] for q in dev}
        for q in qs:
            if q.get("q_id") in dev_ids or q.get("question") in dev_q:
                errors.append(f"{q.get('q_id')}: q_id or question already in questions.jsonl")
    checked = 0
    for q in qs:
        qid = q.get("q_id", "?")
        if set(q) != Q_KEYS:
            errors.append(f"{qid}: keys differ: missing {sorted(Q_KEYS - set(q))}, extra {sorted(set(q) - Q_KEYS)}")
            continue
        if qid in seen:
            errors.append(f"{qid}: duplicate q_id")
        seen.add(qid)
        ql, al, combo = q["question_language"], q["answer_language"], q["combo"]
        if ql not in Q_LANGS or al not in Q_LANGS:
            errors.append(f"{qid}: bad question/answer language {ql}/{al}")
        if q["category"] not in CATEGORIES:
            errors.append(f"{qid}: bad category {q['category']}")
        if q["answerable"] != (q["category"] != "unanswerable"):
            errors.append(f"{qid}: answerable flag inconsistent with category")
        if combo not in FIXED_COMBOS and not (combo.startswith("en-on-") and combo[6:] in DOC_LANGS):
            errors.append(f"{qid}: bad combo {combo}")
        # what is searched
        if q["doc_ids"] == "all":
            searched = library
        elif isinstance(q["doc_ids"], list) and q["doc_ids"]:
            searched = q["doc_ids"]
            unknown = [d for d in searched if d not in by_id]
            if unknown:
                errors.append(f"{qid}: unknown doc_ids {unknown}")
                continue
        else:
            errors.append(f"{qid}: doc_ids must be a non-empty list or 'all'")
            continue
        if combo in ("library", "two-books") and q["doc_ids"] != "all":
            errors.append(f"{qid}: {combo} must search 'all'")
        if combo == "scanned" and not all("derived_from" in by_id[d] for d in searched):
            errors.append(f"{qid}: scanned question must search scanned copies only")
        if combo == "answer-in" and al == ql:
            errors.append(f"{qid}: answer-in must ask for a different answer language")
        if combo not in ("answer-in",) and al != ql:
            errors.append(f"{qid}: answer_language {al} != question_language {ql} outside answer-in")
        if combo.startswith("en-on-") and ql != "en":
            errors.append(f"{qid}: {combo} must be asked in English")
        if combo != "two-books":
            if q["doc_id"] not in by_id:
                errors.append(f"{qid}: unknown doc_id {q['doc_id']}")
                continue
            dlang = by_id[q["doc_id"]]["language"]
            if combo in ("same", "library", "scanned") and SAME_SCRIPT.get(ql, ql) != dlang:
                errors.append(f"{qid}: {combo} question in {ql} on a {dlang} doc")
            if combo == "cross" and (ql == "en" or SAME_SCRIPT.get(ql, ql) == dlang):
                errors.append(f"{qid}: cross question must be in another non-English language")
            if combo == "romanized" and (ql not in ROMANIZED or SCRIPT.get(SAME_SCRIPT[ql], SAME_SCRIPT[ql])
                                         == SCRIPT.get(dlang, dlang)):
                errors.append(f"{qid}: romanized question must be romanized and on a doc in another script")
            if combo == "mixed" and (dlang != "hi-en" or ql not in {"en", "hi", "hinglish"}):
                errors.append(f"{qid}: mixed question must be en/hi/hinglish on the hi-en document")
        # unanswerable
        if not q["answerable"]:
            if q["required_facts"] or q["evidence"] or q["evidence_pages"]:
                errors.append(f"{qid}: unanswerable must have empty facts/evidence")
            continue
        if not q["required_facts"] or not q["evidence"]:
            errors.append(f"{qid}: answerable but no facts/evidence")
            continue
        ev_docs = [e["doc_id"] for e in q["evidence"]]
        if any(d not in searched for d in ev_docs):
            errors.append(f"{qid}: evidence docs {ev_docs} not all in the searched set")
            continue
        if combo != "two-books" and (ev_docs != [q["doc_id"]]
                                     or q["evidence_pages"] != q["evidence"][0]["pages"]):
            errors.append(f"{qid}: evidence / evidence_pages / doc_id disagree")
        # collect evidence text per evidence doc
        per_doc: dict[str, str] = {}
        bad = False
        for e in q["evidence"]:
            n = by_id[e["doc_id"]]["num_pages"]
            if not e["pages"] or any(not (1 <= p <= n) for p in e["pages"]):
                errors.append(f"{qid}: evidence pages {e['pages']} out of range 1..{n} for {e['doc_id']}")
                bad = True
                continue
            tdoc, mapping = text_doc(by_id, e["doc_id"])
            score, ok = by_id[tdoc].get("_roundtrip", (0.0, False))
            if not ok:
                errors.append(f"{qid}: evidence doc {tdoc} does not round-trip ({score:.3f}); not usable")
                bad = True
                continue
            per_doc[e["doc_id"]] = " ".join(pages[tdoc][mapping[p - 1] - 1] for p in e["pages"])
        if bad:
            continue
        holders: Counter = Counter()
        for fact in q["required_facts"]:
            hit = [d for d, t in per_doc.items() if any(alias_matches(t, a, normalized=True) for a in fact)]
            if not hit:
                hit = [d for d, t in per_doc.items() if any(nospace(normalize_text(a)) in nospace(t) for a in fact)]
                if hit:
                    warnings.append(f"{qid}: fact {fact[:2]} matched only after removing whitespace")
            if not hit:
                errors.append(f"{qid}: no alias of {fact} on evidence {q['evidence']}")
            holders.update(hit)
        if combo == "two-books":
            langs = {by_id[d]["language"] for d in ev_docs}
            if len(ev_docs) < 2 or len(langs) < 2 or any(holders[d] == 0 for d in ev_docs):
                errors.append(f"{qid}: two-books needs >=2 docs in different languages, each holding a fact")
        checked += 1

    print(f"set: {which}  documents: {len(docs)} "
          f"(library of {len(library)} text docs)  questions: {len(qs)}  answerable fact-checked: {checked}")
    rt = {d: round(v["_roundtrip"][0], 3) for d, v in by_id.items() if "_roundtrip" in v}
    print("round-trip below threshold:", {d: s for d, s in rt.items() if not by_id[d]["_roundtrip"][1]} or "none")
    print("by combo:", dict(sorted(Counter(q.get("combo") for q in qs).items())))
    print("by question_language:", dict(sorted(Counter(q.get("question_language") for q in qs).items())))
    print("by answer_language:", dict(sorted(Counter(q.get("answer_language") for q in qs).items())))
    print("unanswerable:", sum(1 for q in qs if q.get("answerable") is False))
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"FAILED: {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
