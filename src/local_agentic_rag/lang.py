"""
Lightweight language/script detection (no model, no dependency).

Used to (a) decide when a question and the document are in different languages,
so the agent adds a translated search query, (b) tell the generator which
language to answer in, and (c) localise the refusal message.

Script detection is reliable; telling Hindi from Marathi (both Devanagari) is
not attempted here — the LLM is told to "answer in the language of the
question" and sees the question itself.  Hinglish (Hindi written in Latin
script) is detected with a small function-word list.
"""

from __future__ import annotations

import re
from collections import Counter

_SCRIPTS = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("arabic", 0x0600, 0x06FF),
    ("cyrillic", 0x0400, 0x04FF),
    ("greek", 0x0370, 0x03FF),
    ("cjk", 0x4E00, 0x9FFF),
    ("hiragana_katakana", 0x3040, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
]

# Hindi function words written in Latin script.  Words that are also common English words
# ("the", "me", "par", "to", "is") are deliberately excluded: with them, ordinary English
# questions were classified as Hinglish (observed on Colab: answers came back with "aur").
_HINGLISH_TEXT = """hai hain tha thi thay kya kaun kaunsa kaunse kis kisne kisko kitna kitne kitni kab kahan
    kaise kyun kyon ka ki ke ko se mein nahi nahin bhi yeh woh unka unki uska uski iska iski
    batao bataiye banwaya banaya banane hua hui gaya gayi saal ne kiya liye wala wali aur
    raha rahe rahi kar karte karta lagbhag kaam"""
_HINGLISH_WORDS = frozenset(_HINGLISH_TEXT.split())


def script_of(text: str) -> str:
    """Dominant script of ``text``: 'latin', 'devanagari', ... or 'unknown'."""
    counts: Counter[str] = Counter()
    for ch in text:
        cp = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                counts["latin"] += 1
            continue
        if 0x00C0 <= cp <= 0x024F:
            counts["latin"] += 1
            continue
        for name, lo, hi in _SCRIPTS:
            if lo <= cp <= hi:
                counts[name] += 1
                break
    return counts.most_common(1)[0][0] if counts else "unknown"


def is_hinglish(text: str) -> bool:
    """Latin-script text that reads like Hindi (>= 2 common Hindi function words)."""
    if script_of(text) != "latin":
        return False
    words = re.findall(r"[a-z]+", text.lower())
    hits = {w for w in words if w in _HINGLISH_WORDS}
    return len(hits) >= 2 and len([w for w in words if w in _HINGLISH_WORDS]) >= 0.2 * len(words)


def dominant_script(texts: list[str], sample: int = 200) -> str:
    counts: Counter[str] = Counter()
    for t in texts[:sample]:
        counts[script_of(t)] += 1
    counts.pop("unknown", None)
    return counts.most_common(1)[0][0] if counts else "unknown"


def needs_translation(question: str, doc_script: str) -> bool:
    """True when the question is unlikely to share vocabulary with the document."""
    if doc_script in ("unknown", ""):
        return False
    q_script = script_of(question)
    if q_script == "unknown":
        return False
    if q_script != doc_script:
        return True
    return doc_script == "latin" and is_hinglish(question)


SCRIPT_LANGUAGE_HINT = {
    "latin": "English (or the Latin-script language of the document)",
    "devanagari": "the document's Devanagari-script language (e.g. Hindi or Marathi)",
}

REFUSALS = {
    "devanagari": "दस्तावेज़ में इस प्रश्न का उत्तर नहीं मिला। (The answer was not found in the document.)",
}


def refusal_message(question: str, default: str) -> str:
    return REFUSALS.get(script_of(question), default)


def answer_language(question: str) -> str:
    """Explicit language instruction for the generator (vague wording made a 7B model
    answer an English question in Hindi on Colab)."""
    script = script_of(question)
    if script == "latin":
        return "Hinglish (Hindi written in Latin script)" if is_hinglish(question) else "English"
    if script == "devanagari":
        return "the same Devanagari-script language as the question (Hindi or Marathi, matching the question)"
    if script == "unknown":
        return "English"
    return f"the same language as the question ({script} script)"
