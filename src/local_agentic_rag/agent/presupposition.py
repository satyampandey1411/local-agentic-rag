"""
Presupposition gate (F11, off by default): refuse when a question names something the
retrieved passages do not contain.

Background.  "What powers does the Intelligence Reform Act of 2004 give the DNI?" asked of a
report that predates the Act is answerable only by pretending the Act exists.  Two earlier
attempts to catch this were too blunt: a prompt rule ("refuse unless the exact fact is
present") raised false refusals from 3.8% to 21.2%, and a check for any capitalised phrase
missing from the document also fired on answerable questions.  This module is deliberately
narrow instead:

* it only looks at three kinds of *entity in the question* that a document almost always
  spells the same way it is asked about: a 4-digit year, a multi-word proper-noun phrase
  (two or more capitalised words in a row), and a code with letters and digits ("RIP-4471-B");
* it fires only when one of them is absent from every retrieved passage; and
* the agent applies it only when the verifier has already said the answer is not supported
  (see ``AgenticRAG._presupposition_gate``), so a verified answer is never touched.

Everything here is pure string work: no model, no I/O.  Matching is on a "compact" form
(NFKC, case-folded, digits mapped to ASCII, every non-letter/digit dropped) so that case,
spacing, line breaks and the different dash characters never decide the outcome.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

EntityKind = Literal["year", "phrase", "code"]


@dataclass(frozen=True)
class Entity:
    text: str  # as written in the question (phrase tokens joined by one space)
    kind: EntityKind


# 1800-2100, not glued to other digits/letters and not part of a longer number ("1,2005", "3.2010").
_YEAR_RE = re.compile(r"(?<![\w,])(?<!\d\.)(1[89]\d\d|20\d\d|2100)(?![\w]|[.,]\d)")

# Units / ordinals that mix digits and letters but are not identifiers ("3rd", "10k", "5G", "4.2B").
_UNITISH_RE = re.compile(
    r"^\d+(?:[.,]\d+)?(?:st|nd|rd|th|k|m|b|bn|g|gb|mb|tb|kb|ms|s|x|d|px|pt|pp|kg|km|cm|mm|hz|khz|mhz|ghz)$",
    re.IGNORECASE,
)

_EDGE_PUNCT = " \t\r\n\"'`()[]{}<>.,;:!?*«»“”‘’–—−"
_POSSESSIVE_RE = re.compile(r"(?:'s|’s|'|’)$")


def _strip(token: str) -> str:
    return token.strip(_EDGE_PUNCT)


def _is_capitalised(word: str) -> bool:
    """First letter is an uppercase letter (works for acronyms and CamelCase, False for digits)."""
    for ch in word:
        if ch.isalpha():
            return ch.isupper()
        if ch.isdigit():
            return False
    return False


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!])\s+|\n+", text) if s.strip()]


def _code_candidates(token: str) -> str | None:
    """``token`` if it looks like an identifier: letters and digits, length >= 3, not a unit."""
    core = _strip(token)
    if len(core) < 3 or not core.isascii():
        return None
    if not (any(c.isalpha() for c in core) and any(c.isdigit() for c in core)):
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_/]*[A-Za-z0-9]", core):
        return None
    if _UNITISH_RE.match(core):
        return None
    return core


def extract_entities(question: str) -> list[Entity]:
    """Presupposed entities named in ``question`` (years, proper-noun phrases, codes).

    The first word of every sentence is ignored (it is capitalised by position, and for a
    question it is usually an interrogative or an imperative such as "What", "Which",
    "Explain"), unless it is an all-caps acronym.  Text with no Latin capitals or digits,
    for example Devanagari, yields nothing except years.
    """
    text = unicodedata.normalize("NFKC", question)
    found: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    def add(ent: Entity) -> None:
        key = (ent.kind, compact(ent.text))
        if key not in seen:
            seen.add(key)
            found.append(ent)

    for m in _YEAR_RE.finditer(_digits_to_ascii(text)):
        add(Entity(m.group(1), "year"))

    for sentence in _split_sentences(text):
        for ent in _scan_sentence(sentence):
            add(ent)
    return found


def _scan_sentence(sentence: str) -> list[Entity]:
    """Codes and proper-noun phrases of one sentence (its first word is ignored, see above)."""
    out: list[Entity] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2:
            out.append(Entity(" ".join(run), "phrase"))
        run.clear()

    for i, raw in enumerate(sentence.split()):
        # "U.S." style abbreviations and initials keep the phrase going
        ends_run = raw.endswith((",", ";", ":", ")", "]", "?", "!", ".")) and not re.fullmatch(
            r"(?:[A-Z]\.)+", raw
        )
        possessive = bool(_POSSESSIVE_RE.search(_strip_keep_apostrophe(raw)))
        word = _POSSESSIVE_RE.sub("", _strip(_strip_keep_apostrophe(raw)))
        code = _code_candidates(word)
        if code:
            out.append(Entity(code, "code"))
        if (
            word
            and _is_capitalised(word)
            and not any(c.isdigit() for c in word)
            and not (i == 0 and not _is_acronym(word))
        ):
            run.append(word)
            if possessive or ends_run:
                flush()
        else:
            flush()
    flush()
    return out


def _strip_keep_apostrophe(token: str) -> str:
    return token.strip(' \t\r\n"`()[]{}<>,;:!?*«»“”‘–—−')


def _is_acronym(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def _digits_to_ascii(text: str) -> str:
    """Map any script's decimal digits (Devanagari, Arabic-Indic, ...) to ASCII 0-9."""
    return "".join(
        str(unicodedata.decimal(ch)) if ch.isdigit() and _is_decimal(ch) else ch for ch in text
    )


def _is_decimal(ch: str) -> bool:
    try:
        unicodedata.decimal(ch)
    except (ValueError, TypeError):
        return False
    return True


def compact(text: str) -> str:
    """Case, spacing, dash and digit-script insensitive form used for presence tests."""
    text = _digits_to_ascii(unicodedata.normalize("NFKC", text)).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LNM")


def _latin_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isascii() or "LATIN" in unicodedata.name(ch, "")) / len(
        letters
    )


#: Below this share of Latin letters the passages are in another script (Hindi, Marathi, ...).
_LATIN_MIN_SHARE = 0.6


def missing_entities(entities: Iterable[Entity], passages: Iterable[str]) -> list[Entity]:
    """Entities whose compact form occurs in none of ``passages``.

    Latin-script phrases and codes are not tested against passages that are mostly in
    another script: a Devanagari document writes "UNESCO World Heritage Site" as
    "\u092f\u0942\u0928\u0947\u0938\u094d\u0915\u094b ...", so absence there says nothing.  Years are
    still tested (digits of any script are mapped to ASCII before comparing).
    """
    joined = " ".join(passages)
    haystack = compact(joined)
    same_script = _latin_share(joined) >= _LATIN_MIN_SHARE
    out = []
    for ent in entities:
        if ent.kind != "year" and not same_script:
            continue
        needle = compact(ent.text)
        if needle and needle not in haystack:
            out.append(ent)
    return out


def find_missing_presupposition(question: str, passages: Iterable[str]) -> Entity | None:
    """The first entity of ``question`` that is absent from every passage (None = no gate)."""
    missing = missing_entities(extract_entities(question), list(passages))
    return missing[0] if missing else None
