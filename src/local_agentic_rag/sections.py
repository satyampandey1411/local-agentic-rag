"""Shared patterns for numbered sections ("§ 164. Taxes", "Section 2.1 Scope").

Used by the chunker (to record which numbered section each chunk sits in, at ingest
time) and by ``DocumentIndex.section_chunks`` (the fallback for indexes built before
chunks carried that number).  A separate module avoids an import cycle between the two.
"""

from __future__ import annotations

import re

#: A numbered section heading: "§ 164. Taxes", "Sec. 7203.", "Section 2.1 Scope", "4.3 Results".
#: A bare integer does not count ("2010 (I) In general" is a year, not section 2010).
SECTION_HEAD_RE = re.compile(
    r"^\s*(?:(?:§+\s*|(?:section|sec\.)\s+)(\d+[A-Za-z]?(?:\.\d+)*)|(\d+(?:\.\d+)+))"
    r"(?![\d])(?!\.\d)[.:\s]",
    re.IGNORECASE,
)
#: A section start inside running text, as codes print it: "§ 164. Taxes".  Cross-references
#: are written "section 164" and amendment notes "§ 221(a)(2)", so neither matches.
INLINE_SECTION_RE = re.compile(r"(?:^|\s)§\s*(\d+[A-Za-z]?)\.\s+[A-Z]")
#: A structural heading ABOVE sections: "CHAPTER 79—DEFINITIONS", "Subchapter A—Crimes",
#: "PART I—GENERAL", "ARTICLE 3.", "TITLE 26".  The keyword must be followed by an
#: identifier (number, roman numeral or a short capital run) and then a dash, colon, period
#: or the end of the heading, so prose like "Part of the problem" or "Title page" is not one.
#: Such a heading ends the current numbered section: what follows (a chapter's table of
#: contents, a part preamble) does not belong to the previous section.
STRUCTURAL_HEAD_RE = re.compile(
    r"^\s*(?:chapter|subchapter|part|subpart|title|subtitle|article)\s+"
    r"(?:\d+[A-Za-z]?|[IVXLCDM]+|[A-Z]{1,3})\s*(?:[—–:.\-]|$)",
    re.IGNORECASE,
)


def section_number(text: str) -> str | None:
    """Number of the section that ``text`` (a heading) opens, lower-cased, or None."""
    m = SECTION_HEAD_RE.match(text)
    return (m.group(1) or m.group(2)).lower() if m else None
