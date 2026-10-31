#!/usr/bin/env python3
"""Run the many-languages benchmark and score it.

    python benchmark/languages/run.py --set dev --out results/languages_dev.jsonl
    python benchmark/languages/run.py --set heldout --out results/languages_heldout.jsonl
    python benchmark/languages/run.py --summarize results/languages_dev.jsonl

``--src`` puts another checkout's ``src/`` first on the import path, so the same script (and
the same scoring) measures an older version of the system for a before/after comparison.

Per question it records the answer and four checks:
  correct    answerable: every required fact appears in the answer (and it was not refused);
             unanswerable: the system refused
  script_ok  the answer (or refusal) is written in the script of the expected answer language
             (a script check: it cannot tell Hindi from Sanskrit, both Devanagari)
  doc_ok     library / two-book questions: the citations include every evidence document
  refused    whether the system said the answer is not in the document
Result lines on stdout start with "LANG|" (one per question) and "LANG SUMMARY|".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Expected script of each answer language (question_language / answer_language codes).
SCRIPT_OF_LANGUAGE = {
    **dict.fromkeys(["en", "fr", "es", "de", "pt", "it", "id", "hinglish", "roman-ur"], "latin"),
    **dict.fromkeys(["hi", "mr", "ne", "sa", "bh", "awa", "mai"], "devanagari"),
    **dict.fromkeys(["ur", "ar", "fa"], "arabic"),
    **dict.fromkeys(["bn", "as"], "bengali"),
    "te": "telugu",
    "ta": "tamil",
    "kn": "kannada",
    "ml": "malayalam",
    "gu": "gujarati",
    "pa": "gurmukhi",
    "or": "oriya",
    "zh": "cjk",
    "ja": "japanese",
    "ru": "cyrillic",
}
_RANGES = [
    ("devanagari", 0x0900, 0x097F), ("bengali", 0x0980, 0x09FF), ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF), ("oriya", 0x0B00, 0x0B7F), ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F), ("kannada", 0x0C80, 0x0CFF), ("malayalam", 0x0D00, 0x0D7F),
    ("arabic", 0x0600, 0x06FF), ("arabic", 0x0750, 0x077F), ("arabic", 0xFB50, 0xFDFF),
    ("arabic", 0xFE70, 0xFEFF), ("cyrillic", 0x0400, 0x04FF), ("cjk", 0x3400, 0x4DBF),
    ("cjk", 0x4E00, 0x9FFF), ("kana", 0x3040, 0x30FF),
]  # fmt: skip
# Every script's decimal digits → ASCII, so scoring does not depend on the system's own
# normaliser (the before/after runs use different versions of it).
_DIGIT_ZEROS = (
    0x0660,
    0x06F0,
    0x0966,
    0x09E6,
    0x0A66,
    0x0AE6,
    0x0B66,
    0x0BE6,
    0x0C66,
    0x0CE6,
    0x0D66,
)
DIGITS = {zero + i: str(i) for zero in _DIGIT_ZEROS for i in range(10)}


def script_of(text: str) -> str:
    """Dominant script of an answer, ignoring citation markers, digits and punctuation."""
    counts: dict[str, int] = defaultdict(int)
    for ch in re.sub(r"\[\d+(?:[,;\s]*\d+)*\]", " ", text or ""):
        cp = ord(ch)
        if ch.isascii() or 0x00C0 <= cp <= 0x024F:
            if ch.isalpha():
                counts["latin"] += 1
            continue
        for name, lo, hi in _RANGES:
            if lo <= cp <= hi:
                counts[name] += 1
                break
    if counts.get("kana"):
        return "japanese"
    return max(counts, key=counts.get) if counts else "unknown"


def facts_ok(answer: str, facts: list[list[str]], all_facts_present) -> bool:
    answer = (answer or "").translate(DIGITS)
    facts = [[a.translate(DIGITS) for a in group] for group in facts]
    return bool(all_facts_present(answer, facts))


def load_set(name: str) -> tuple[Path, list[dict], list[dict]]:
    base = HERE if name == "dev" else HERE / "heldout"
    docs = json.loads((base / "documents.json").read_text(encoding="utf-8"))
    questions = [
        json.loads(line)
        for line in (base / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return base, docs, questions


def evidence_docs(q: dict) -> list[str]:
    if q.get("evidence"):
        return [e["doc_id"] for e in q["evidence"]]
    return list(q["doc_ids"]) if isinstance(q.get("doc_ids"), list) else []


def run(args) -> int:
    if args.src:
        src = str(Path(args.src).resolve())
        sys.path.insert(0, src)
        # Parse workers are separate `python -m` processes: they must import the same tree.
        os.environ["PYTHONPATH"] = os.pathsep.join(
            p for p in (src, os.environ.get("PYTHONPATH")) if p
        )
    import local_agentic_rag
    from local_agentic_rag.config import RAGConfig
    from local_agentic_rag.eval.metrics import all_facts_present
    from local_agentic_rag.llm.ollama import OllamaLLM
    from local_agentic_rag.pipeline import RAGPipeline

    print(
        f"LANG CODE| {local_agentic_rag.__file__} PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
        flush=True,
    )
    base, docs, questions = load_set(args.set)
    if args.limit:
        questions = questions[: args.limit]
    llm = OllamaLLM(args.model, host=args.host)
    llm.ensure_ready()
    pipe = RAGPipeline(RAGConfig(), llm=llm, cache_dir=args.cache)
    pipe.load_models()
    doc_lang = {}
    for d in docs:
        t0 = time.time()
        report = pipe.ingest(str(base / "pdfs" / f"{d['doc_id']}.pdf"), doc_id=d["doc_id"])
        doc_lang[d["doc_id"]] = d.get("language", "")
        print(
            f"LANG INGEST| {d['doc_id']} {time.time() - t0:.1f}s {getattr(report, 'num_chunks', '')}",
            flush=True,
        )
    library = [d["doc_id"] for d in docs if not d["doc_id"].endswith("-scan")]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for q in questions:
            ids = library if q.get("doc_ids") == "all" else q.get("doc_ids") or [q["doc_id"]]
            t0 = time.time()
            try:
                ans = pipe.ask(q["question"], ids)
                answer, refused, error = ans.answer, bool(ans.refused), ans.error
                cites = [(c.doc_id, c.page) for c in ans.citations]
                calls, steps = ans.llm_calls, [s.step for s in ans.trace]
            except Exception as exc:  # one failure must not stop the run
                answer, refused, error, cites, calls, steps = "", False, repr(exc), [], 0, []
            latency = time.time() - t0
            expected_lang = q.get("answer_language") or q.get("question_language")
            if q.get("answerable", True):
                correct = (not refused) and facts_ok(
                    answer, q.get("required_facts", []), all_facts_present
                )
            else:
                correct = refused
            want = set(evidence_docs(q))
            rec = {
                "q_id": q["q_id"],
                "combo": q.get("combo", ""),
                "question_language": q.get("question_language"),
                "answer_language": expected_lang,
                "doc_language": doc_lang.get(next(iter(evidence_docs(q) or ids), ""), ""),
                "answerable": q.get("answerable", True),
                "question": q["question"],
                "answer": answer,
                "refused": refused,
                "correct": bool(correct),
                "script": script_of(answer),
                "script_ok": script_of(answer) == SCRIPT_OF_LANGUAGE.get(expected_lang, "?"),
                "doc_ok": (want <= {d for d, _ in cites}) if q.get("doc_ids") == "all" else None,
                "citations": cites,
                "llm_calls": calls,
                "steps": steps,
                "latency_sec": round(latency, 2),
                "error": error,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                "LANG|",
                json.dumps(
                    {
                        k: rec[k]
                        for k in (
                            "q_id",
                            "combo",
                            "correct",
                            "script_ok",
                            "doc_ok",
                            "refused",
                            "latency_sec",
                        )
                    }
                ),
                flush=True,
            )
    summarize(out)
    return 0


def _rate(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{sum(vals)}/{len(vals)}" if vals else "-"


def summarize(path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "file": str(path),
        "questions": len(rows),
        "answerable_correct": _rate(answerable, "correct"),
        "unanswerable_refused": _rate(unanswerable, "correct"),
        "false_refusals": f"{sum(r['refused'] for r in answerable)}/{len(answerable)}",
        "answer_script_ok": _rate(rows, "script_ok"),
        "library_right_book": _rate([r for r in rows if r["doc_ok"] is not None], "doc_ok"),
        "median_latency_sec": round(statistics.median(r["latency_sec"] for r in rows), 2)
        if rows
        else None,
        "errors": sum(1 for r in rows if r.get("error")),
    }
    for field in ("combo", "question_language", "doc_language"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r.get(field) or "?"].append(r)
        summary[f"by_{field}"] = {
            k: {"correct": _rate(v, "correct"), "script_ok": _rate(v, "script_ok")}
            for k, v in sorted(groups.items())
        }
    print("LANG SUMMARY|", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--src", default=None, help="checkout src/ to import (before/after runs)")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--cache", default=None, help="index cache directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", default=None, help="only summarise an existing results file")
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(Path(args.summarize))
        return 0
    if not args.out:
        ap.error("--out is required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
