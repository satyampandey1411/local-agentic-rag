"""
BM25 keyword retrieval.

Dense embeddings blur exact tokens — model numbers, table labels, acronyms,
years.  BM25 catches those.  Tokenisation keeps decimals and hyphenated codes
intact ("3.5", "SP-800") and applies light plural stripping so "rates"
matches "rate".  Implemented directly (no rank_bm25) — see ``BM25Index``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
# Any-script words: letters/digits plus combining marks (Indic vowel signs are marks,
# not letters, so a plain \w+ would split "भारत" style words at every matra).
_UNICODE_WORD_RE = re.compile(r"(?:[^\W_]|[\u0300-\u036f\u0900-\u0dff\u0e00-\u0e7f])+")

_STOPWORD_TEXT = """a an and are as at be but by for from has have how in is it its of on or that the
    their there these this to was were what when where which who why will with does did do
    than then they them into about can could would should our we you your i me my"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _stem(token: str) -> str:
    """Tiny plural stripper: rates→rate, tables→table, policies→policy, boxes→box."""
    if len(token) <= 3 or token[0].isdigit() or token.endswith("ss"):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    out: list[str] = []
    for word in _UNICODE_WORD_RE.findall(text):
        if word.isascii():
            continue  # ASCII handled below with the code/decimal-aware tokenizer
        if all("\u4e00" <= ch <= "\u9fff" for ch in word):
            out.extend(word)  # CJK has no spaces: index characters
        else:
            out.append(word)
    for tok in _TOKEN_RE.findall(text):
        if tok in STOPWORDS:
            continue
        out.append(_stem(tok))
        if "-" in tok or "." in tok and not tok.replace(".", "").isdigit():
            out.extend(_stem(p) for p in re.split(r"[.\-]", tok) if p and p not in STOPWORDS)
    return out


class BM25Index:
    """Okapi BM25 with Lucene's always-positive IDF.

    ``rank_bm25.BM25Okapi`` uses ``log((N - n + 0.5) / (n + 0.5))``, which is
    zero or negative for any term in at least half the chunks — so a short PDF
    (1-2 chunks) or a frequent-but-important term scores nothing.  Lucene's
    ``log(1 + (N - n + 0.5) / (n + 0.5))`` is always positive.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(t)) for t in texts]
        self._lengths = np.array([sum(d.values()) for d in self._docs], dtype=np.float64)
        self._avgdl = float(self._lengths.mean()) if len(self._docs) else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(self._docs):
            for t, tf in d.items():
                self._postings.setdefault(t, []).append((i, tf))

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        """Top-n (index, score) pairs with score > 0, best first."""
        if not self._docs:
            return []
        scores = np.zeros(len(self._docs), dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self._lengths / max(self._avgdl, 1e-9))
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in self._postings[term]:
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm[i])
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
