"""Write the extra question files from their specs, with evidence pages computed from the real PDF text.

    .venv/bin/python benchmark/languages/build_more.py more      # -> questions_more.jsonl (dev docs)
    .venv/bin/python benchmark/languages/build_more.py heldout2  # -> heldout2/questions.jsonl (SEALED)
    .venv/bin/python benchmark/languages/build_more.py special   # -> special/questions.jsonl

Specs: specs_more.py (MORE, MORE_TWO), heldout2/specs.py (sealed; HELDOUT2, HELDOUT2_TWO,
HELDOUT2_SCANNED) and SPECIAL below. Same spec tuples and helpers as build_questions.py, whose
output (questions.jsonl, heldout/questions.jsonl) this script never touches. A fact that is on no
page is an error. Build the PDFs first (make_pdfs.py --set heldout2 / --set special).
"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_questions import evidence_for, load_pages, rec, single, snippet, write  # noqa: E402

TAGS = {"same": "same", "cross": "cross", "unanswerable": "unans", "answer-in": "answerin", "scanned": "scan",
        "romanized": "roman", "mixed": "mixed"}

# special/: (doc, combo, category, q_lang, a_lang, question, reference_answer, facts)
# sp-hien-indiagate alternates Hindi and English paragraphs (para_langs in special/config.json).
# sp-ar-burjkhalifa (Chromium print) does not round-trip (0.837 < 0.97), so it has no questions.
SPECIAL = [
    # fact in a Hindi paragraph, asked in English
    ("sp-hien-indiagate", "mixed", "multi_hop", "en", "en",
     "What is the diameter of the hexagonal area around India Gate, and how large an area does it cover?",
     "The hexagonal area has a diameter of 625 metres and covers 306,000 square metres.",
     [["625", "६२५"], ["306000", "306,000", "३०६,०००", "३०६०००"]]),
    # fact in an English paragraph, asked in Hindi
    ("sp-hien-indiagate", "mixed", "multi_hop", "hi", "hi", "इंडिया गेट का उद्घाटन किसने और किस वर्ष किया था?",
     "इंडिया गेट का उद्घाटन वायसराय लॉर्ड इरविन ने 12 फ़रवरी 1931 को किया था।",
     [["irwin", "इरविन", "इर्विन", "अर्विन"], ["1931", "१९३१"]]),
    # one fact from an English paragraph and one from a Hindi paragraph, asked in Hinglish
    ("sp-hien-indiagate", "mixed", "multi_hop", "hinglish", "hinglish",
     "India Gate ka foundation stone kis din rakha gaya tha, aur Amar Jawan Jyoti ki lau kab Rashtriya Samar "
     "Smarak ki lau mein mila di gayi?",
     "Foundation stone 10 February 1921 ko rakha gaya tha; Amar Jawan Jyoti ki lau January 2022 mein Rashtriya "
     "Samar Smarak ki lau mein mila di gayi.",
     [["10 february 1921", "10 feb 1921", "10 फ़रवरी 1921", "10 फरवरी 1921", "१० फ़रवरी १९२१"],
      ["2022", "२०२२"]]),   # "जनवरी 2022" reads back from the PDF as "जनववरी 2022", so only the year is matched
]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(base: Path, prefix: str, specs, two, scanned_docs=()) -> list[dict]:
    pages = load_pages(base)
    out = []
    for s in specs:
        out.append(single(base, pages, prefix, s))
    if scanned_docs:
        same = {s[0]: s for s in specs if s[1] == "same"}
        for doc in scanned_docs:
            sid = f"{doc}-scan"
            out.append(single(base, pages, prefix, same[doc], combo="scanned", doc_override=sid,
                              doc_ids=[sid], scan=True))
    # q_id = <prefix>-<doc>-<tag>, numbered when a doc has several questions of one kind
    keys = [(r["doc_id"].removesuffix("-scan"), TAGS.get(r["combo"], "en")) for r in out]
    total, seen = Counter(keys), Counter()
    for r, k in zip(out, keys, strict=True):
        seen[k] += 1
        r["q_id"] = f"{prefix}-{k[0]}-{k[1]}" + (f"-{seen[k]}" if total[k] > 1 else "")
    for i, (docs, cat, ql, q, ans, facts) in enumerate(two, start=1):
        qid = f"{prefix}-twobooks-{i:02d}"
        ev = evidence_for(pages, docs, facts, qid)
        note = " | ".join(f"{d}: {snippet(base, d, facts)}" for d in docs)
        out.append(rec(qid, None, cat, q, ans, facts, None, note, ql, ql, "all", ev, "two-books"))
    ids = [r["q_id"] for r in out]
    assert len(ids) == len(set(ids)), "duplicate q_id"
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "more":
        m = _load(HERE / "specs_more.py", "specs_more")
        write(HERE / "questions_more.jsonl", build(HERE, "lm", m.MORE, m.MORE_TWO))
    elif which == "heldout2":
        h = _load(HERE / "heldout2" / "specs.py", "heldout2_specs")
        write(HERE / "heldout2" / "questions.jsonl",
              build(HERE / "heldout2", "lgh2", h.HELDOUT2, h.HELDOUT2_TWO, h.HELDOUT2_SCANNED))
    elif which == "special":
        write(HERE / "special" / "questions.jsonl", build(HERE / "special", "lsp", SPECIAL, []))
    else:
        sys.exit("usage: build_more.py more|heldout2|special")
