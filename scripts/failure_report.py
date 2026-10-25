"""Print every failed benchmark question compactly (for error analysis).

    python scripts/failure_report.py /content/results/answers_agentic.jsonl [more.jsonl ...]

A question "fails" when it is answerable and not answered correctly (all required
facts present, not refused), or unanswerable and not refused.
"""

from __future__ import annotations

import json
import sys


def _short(text, n: int) -> str:
    return " ".join(str(text or "").split())[:n]


def main(paths: list[str]) -> None:
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        failed = []
        for r in records:
            m = r.get("metrics", {})
            ok = bool(m.get("answer_correct")) if r.get("answerable") else bool(r.get("refused"))
            if not ok:
                failed.append(r)
        print(f"=== {path}: {len(failed)} failed / {len(records)}")
        for r in failed:
            steps = [t["step"] for t in r.get("trace", [])]
            calc = [t["detail"].get("output") for t in r.get("trace", []) if t["step"] == "calculate"]
            print(
                f"- {r['q_id']} [{r['category']}|{r.get('question_language', 'en')}] "
                f"refused={r.get('refused')} ev={r.get('evidence_pages')} "
                f"ret={(r.get('retrieved_pages') or [])[:6]} cited={r.get('cited_pages')} "
                f"esc={'escalate' in steps} grounded={r.get('grounded')}"
            )
            print(f"    Q: {_short(r['question'], 140)}")
            print(f"    REF: {_short(r.get('reference_answer'), 110)} | FACTS: {_short(r.get('required_facts'), 90)}")
            print(f"    ANS: {_short(r.get('answer'), 220)}")
            if calc:
                print(f"    CALC: {_short(calc[0], 200)}")


if __name__ == "__main__":
    main(sys.argv[1:])
