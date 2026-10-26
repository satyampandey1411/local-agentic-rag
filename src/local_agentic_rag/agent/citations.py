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
