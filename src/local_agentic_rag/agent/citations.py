"""
Citation parsing, validation and fallback attribution.

The generator cites passages as ``[n]``.  We map each marker back to the
chunk (and therefore page) it refers to, discard markers that point at
passages that do not exist, and — if a non-refusal answer carries no valid
citation at all — attribute each sentence to the passage it overlaps most,
so the user still sees where the information came from.
"""

from __future__ import annotations

import re

from local_agentic_rag.retrieval.bm25 import tokenize
from local_agentic_rag.schema import ANSWER_NOT_FOUND, Citation, ScoredChunk

_MARKER_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_REFUSAL_RE = re.compile(
    r"(does not (contain|provide|mention|include|specify|state|say|give|list)|"
    r"do not (contain|provide|mention|include|specify|state|say|give|list)|"
    r"not (mentioned|provided|specified|stated|found|available|included|given|listed) in|"
    r"no (information|mention|data|details?) (about|on|regarding|of)|"
    r"cannot be (determined|answered|found)|unable to (find|determine|answer)|"
    # Hindi / Marathi
    r"उल्लेख नहीं|नहीं दिया गया|नहीं दी गई|जानकारी नहीं|नहीं बताया|नहीं मिला|"
    r"उल्लेख नाही|दिलेली नाही|दिलेले नाही|माहिती नाही|नमूद केलेले नाही|सापडले नाही)",
    re.IGNORECASE,
)
_CONTRAST_RE = re.compile(
    r"\b(but|however|although|though|yet|whereas)\b|लेकिन|परंतु|पण|मात्र", re.IGNORECASE
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?।])\s+")


def parse_markers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in _MARKER_RE.finditer(text):
        for token in re.split(r"[,;\s]+", match.group(1)):
            if token.isdigit() and int(token) not in numbers:
                numbers.append(int(token))
    return numbers


def strip_invalid_markers(text: str, n_passages: int) -> str:
    def repl(match: re.Match[str]) -> str:
        valid = [
            t
            for t in re.split(r"[,;\s]+", match.group(1))
            if t.isdigit() and 1 <= int(t) <= n_passages
        ]
        return f"[{', '.join(valid)}]" if valid else ""

    return re.sub(r"\s+([.,;])", r"\1", _MARKER_RE.sub(repl, text)).strip()


def build_citations(text: str, passages: list[ScoredChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for n in parse_markers(text):
        if 1 <= n <= len(passages):
            c = passages[n - 1].chunk
            citations.append(
                Citation(
                    marker=n,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    page=c.page,
                    snippet=c.text[:200],
                )
            )
    return citations


def infer_citations(
    text: str, passages: list[ScoredChunk], min_overlap: float = 0.35
) -> list[Citation]:
    """Attribute answer sentences to passages by content-word overlap (fallback only)."""
    citations: dict[int, Citation] = {}
    passage_tokens = [set(tokenize(p.chunk.text)) for p in passages]
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = set(tokenize(sentence))
        if len(words) < 3:
            continue
        best, best_overlap = -1, 0.0
        for i, toks in enumerate(passage_tokens):
            overlap = len(words & toks) / len(words)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        if best >= 0 and best_overlap >= min_overlap and best + 1 not in citations:
            c = passages[best].chunk
            citations[best + 1] = Citation(
                marker=best + 1,
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                page=c.page,
                snippet=c.text[:200],
            )
    return list(citations.values())


def strip_trailing_sentinel(text: str) -> str:
    """Drop a "not found" sentinel the model appended after a cited answer.

    Observed on Kaggle (Hinglish Taj Mahal question): the model answered both parts with
    citations and then added ANSWER_NOT_FOUND, so the whole correct answer was discarded as
    a refusal.  The text before the sentinel is kept only if it carries a [n] citation; a
    sentinel at the start, or after uncited text, is left in place (still a refusal).
    """
    idx = text.find(ANSWER_NOT_FOUND)
    if idx <= 0:
        return text
    head = text[:idx].rstrip()
    return head if parse_markers(head) else text


def is_refusal(text: str) -> bool:
    """True for the sentinel, or an answer whose message is "the document does not say".

    Two forms are recognised:
    * an uncited, short statement that the document lacks the information;
    * an answer whose *first sentence* is such a statement (citations allowed) without a
      contrast clause ("…does not say X, but it does say Y") — observed on Colab: a 7B model
      cites the passages it searched while declining, which used to be scored as an answer.
    """
    stripped = text.strip()
    if not stripped or ANSWER_NOT_FOUND in stripped:
        return True
    if not parse_markers(stripped) and _REFUSAL_RE.search(stripped) and len(stripped) < 400:
        return True
    first = _SENTENCE_END_RE.split(_MARKER_RE.sub("", stripped), maxsplit=1)[0]
    # Conservative: a first sentence carrying data ("…; it says a dividend of 0.50 was
    # approved") or a contrast clause is a substantive answer, not a refusal.
    if _CONTRAST_RE.search(first) or re.search(r"\d|[०-९]|;", first):
        return False
    return bool(_REFUSAL_RE.search(first))


# ---------------------------------------------------------------------------
# "Which table / figure ...?" answers must name the table (roadmap F6)
# ---------------------------------------------------------------------------
#
# A 7B model often answers "Which table shows X?" by describing the table it found
# without printing its number.  The number is in the cited chunk (the caption line
# that ingest puts in front of the table), so it is appended deterministically rather
# than asked for in the prompt: no prompt text changes, no effect on other questions.

#: Questions of the form "which/what table|figure|exhibit ...", English and Hindi.  A
#: following number ("what table 3 ...") or a verb that makes "figure" mean "number"
#: ("which figure is higher") does not count.
_WHICH_TABLE_RE = re.compile(
    r"\b(?:which|what)\s+(?:(?:other|specific|particular|statistical|data)\s+)?"
    r"(?:table|exhibit|figure(?!\s+(?:is|was|are|were|has|had|did|does|do|for|of)\b))s?\b"
    r"(?!\s*(?:no\.?\s*)?\d)"
    r"|(?:कौन\s*सी|कौनसी|किस)\s+(?:तालिका|सारणी|टेबल)"
    r"|(?:कौन\s*सा|कौनसा|किस)\s+(?:चित्र|ग्राफ़?)",
    re.IGNORECASE,
)
_KEYWORD = r"(?i:Table|Figure|Fig\.|Exhibit|तालिका|सारणी|चित्र)"
#: "12.4", "A.3", "IV", "2" (digits may be Devanagari).
_IDENT = r"(?:[A-Z]\.?[0-9०-९]+(?:\.[0-9०-९]+)*|[0-9०-९]+(?:\.[0-9०-९]+)*|[IVXL]+)"
#: An identifier anywhere in an answer ("Table 12.4", "table 2", "Figure A.3").
_IDENT_IN_ANSWER_RE = re.compile(rf"(?<!\w){_KEYWORD}\s*{_IDENT}(?![\w])")
#: A caption line: starts with the identifier and a separator ("Table 12.4: Title", "Figure 2. Title").
_CAPTION_LINE_RE = re.compile(
    rf"^\s*(?P<ident>{_KEYWORD}\s*{_IDENT})\s*(?:[:.\-–—]\s*|$)(?P<title>.*)$"
)
MAX_CAPTION_CHARS = 120


def asks_which_table(question: str) -> bool:
    """True for "which table shows X?" style questions (conservative)."""
    return bool(_WHICH_TABLE_RE.search(question))


def has_table_identifier(text: str) -> bool:
    return bool(_IDENT_IN_ANSWER_RE.search(text))


def find_caption(chunk_text: str) -> tuple[str, str] | None:
    """(identifier, caption title) from the first caption line of a chunk, if it has one."""
    lines = chunk_text.splitlines()
    for i, line in enumerate(lines):
        m = _CAPTION_LINE_RE.match(line)
        if not m:
            continue
        ident = re.sub(r"\s+", " ", m.group("ident")).strip()
        title = m.group("title").strip()
        # A caption wrapped over two text lines continues on a lower-case line.
        if i + 1 < len(lines) and re.match(r"[a-z]", lines[i + 1].strip()):
            title = f"{title} {lines[i + 1].strip()}".strip()
        title = re.sub(r"\[[^\]]*\]", "", title)  # a footnote "[1]" must not read as a citation
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) > MAX_CAPTION_CHARS:
            title = title[:MAX_CAPTION_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:.-") + "…"
        return ident, title
    return None


def table_reference_suffix(
    question: str, answer: str, citations: list[Citation], passages: list[ScoredChunk]
) -> tuple[str, dict[str, object]] | None:
    """Sentence naming the cited table/figure when a "which table" answer does not.

    Returns ``(suffix, trace detail)`` or ``None``.  The identifier always comes from a
    chunk the answer already cites, and the sentence re-uses that chunk's citation number.
    Tables are preferred over text chunks that merely contain a caption line, but the
    citation order decides within each kind.
    """
    if not asks_which_table(question) or has_table_identifier(answer):
        return None
    candidates = []
    for c in citations:
        if 1 <= c.marker <= len(passages):
            chunk = passages[c.marker - 1].chunk
            found = find_caption(chunk.text)
            if found:
                candidates.append((chunk.kind != "table", len(candidates), c.marker, chunk, found))
    if not candidates:
        return None
    _, _, marker, chunk, (ident, title) = min(candidates, key=lambda t: t[:2])
    body = f"{ident}: {title}" if title else ident
    detail = {"identifier": ident, "marker": marker, "page": chunk.page, "chunk_id": chunk.chunk_id}
    return f" ({body} [{marker}])", detail
