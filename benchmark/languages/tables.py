#!/usr/bin/env python3
"""Markdown tables from run.py result files: before vs after, by combo and by language.

    python benchmark/languages/tables.py --before before_dev.jsonl --after after_dev.jsonl \
        [--heldout after_heldout.jsonl] [--failures]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import rescore  # noqa: E402

NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "sa": "Sanskrit", "ne": "Nepali", "bh": "Bhojpuri",
    "awa": "Awadhi", "mai": "Maithili", "ur": "Urdu", "bn": "Bengali", "te": "Telugu", "ta": "Tamil",
    "kn": "Kannada", "ml": "Malayalam", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia", "fr": "French",
    "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "ru": "Russian", "hinglish": "Hinglish",
    "roman-ur": "Roman Urdu",
}  # fmt: skip
COMBOS = [
    ("same", "Question in the document's language"),
    ("en-on", "English question about a non-English document"),
    ("cross", "Question in another non-English language (Telugu on Sanskrit, Urdu on Hindi, ...)"),
    ("answer-in", 'Asks for the answer in another language ("Answer in Urdu")'),
    ("library", "Searches all 22 books at once: must find the right one"),
    ("two-books", "Needs one fact from each of two books in different languages"),
    ("scanned", "Image-only copy: needs OCR"),
    ("romanized", "Indian language typed in Latin letters (Hinglish, Roman Urdu, Tanglish, ...)"),
    ("mixed", "One PDF mixing Hindi and English paragraphs"),
    ("unanswerable", "Not in the document: should be refused"),
]


def load(path: str | None) -> list[dict]:
    """Rows of a run.py results file, rescored with run.py's current checks."""
    if not path:
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return rescore([json.loads(line) for line in lines if line.strip()])


def combo(r: dict) -> str:
    return "en-on" if r["combo"].startswith("en-on-") else r["combo"]


def frac(rows: list[dict], key: str = "correct") -> str:
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    return f"{sum(vals)}/{len(vals)}" if vals else "–"


def overall(rows: list[dict]) -> dict[str, str]:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    books = [r for r in rows if r.get("doc_ok") is not None]
    return {
        "Answerable questions answered correctly": frac(ans),
        "... with spelling-tolerant name matching": frac(ans, "correct_lenient"),
        "Unanswerable questions refused": frac(una),
        "Answerable questions wrongly refused": f"{sum(r['refused'] for r in ans)}/{len(ans)}",
        "Answer written in the expected script": frac(rows, "script_ok"),
        "Library / two-book questions citing the right book(s)": frac(books, "doc_ok"),
        "Median seconds per question": f"{statistics.median(r['latency_sec'] for r in rows):.1f}"
        if rows
        else "–",
    }


def table_overall(before: list[dict], after: list[dict]) -> str:
    b, a = overall(before), overall(after)
    lines = ["| | Before (committed code) | After (multi-language fixes) |", "|---|---:|---:|"]
    lines += [f"| {k} | {b.get(k, '–')} | **{a[k]}** |" for k in a]
    return "\n".join(lines)


def table_combo(before: list[dict], after: list[dict]) -> str:
    lines = [
        "| Kind of question | Before: correct · right script | After: correct · right script |",
        "|---|---:|---:|",
    ]
    for key, label in COMBOS:
        b = [r for r in before if combo(r) == key]
        a = [r for r in after if combo(r) == key]
        if a:
            lines.append(
                f"| {label} | {frac(b)} · {frac(b, 'script_ok')} | **{frac(a)}** · {frac(a, 'script_ok')} |"
            )
    return "\n".join(lines)


def table_language(before: list[dict], after: list[dict], field: str) -> str:
    groups_a: dict[str, list[dict]] = defaultdict(list)
    groups_b: dict[str, list[dict]] = defaultdict(list)
    for r in after:
        groups_a[r.get(field) or "?"].append(r)
    for r in before:
        groups_b[r.get(field) or "?"].append(r)
    head = "Document language" if field == "doc_language" else "Question language"
    lines = [f"| {head} | Before | After | After, right script |", "|---|---:|---:|---:|"]
    for code in sorted(groups_a, key=lambda c: (-len(groups_a[c]), NAMES.get(c, c))):
        a, b = groups_a[code], groups_b.get(code, [])
        lines.append(
            f"| {NAMES.get(code, code)} | {frac(b)} | **{frac(a)}** | {frac(a, 'script_ok')} |"
        )
    return "\n".join(lines)


def failures(rows: list[dict]) -> str:
    out = []
    for r in rows:
        if not r["correct"] or not r["script_ok"]:
            out.append(
                f"- {r['q_id']} [{r['combo']}] correct={r['correct']} script={r['script']} "
                f"(want {r['answer_language']}) refused={r['refused']} cites={r['citations'][:3]}\n"
                f"  Q: {r['question']}\n  A: {(r['answer'] or '')[:300]}"
            )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before")
    ap.add_argument("--after", required=True)
    ap.add_argument("--heldout")
    ap.add_argument("--failures", action="store_true")
    args = ap.parse_args(argv)
    before, after, heldout = load(args.before), load(args.after), load(args.heldout)
    print("## Overall (dev set)\n\n" + table_overall(before, after))
    print("\n## By kind of question\n\n" + table_combo(before, after))
    print("\n## By document language\n\n" + table_language(before, after, "doc_language"))
    print("\n## By question language\n\n" + table_language(before, after, "question_language"))
    if heldout:
        print("\n## Held-out set (after only, run once)\n\n" + table_overall([], heldout))
        print("\n" + table_language([], heldout, "doc_language"))
    if args.failures:
        print("\n## After: misses\n\n" + failures(after))
        if heldout:
            print("\n## Held-out: misses\n\n" + failures(heldout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
