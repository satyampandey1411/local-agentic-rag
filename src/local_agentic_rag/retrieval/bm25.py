"""
BM25 keyword retrieval.

Dense embeddings blur exact tokens — model numbers, table labels, acronyms,
years.  BM25 catches those.  Tokenisation keeps decimals and hyphenated codes
intact ("3.5", "SP-800") and applies light plural stripping so "rates"
matches "rate".  Implemented directly (no rank_bm25) — see ``BM25Index``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
# Any-script words: letters/digits plus combining marks (Indic vowel signs are marks,
# not letters, so a plain \w+ would split "भारत" style words at every matra), Arabic
# harakat and the zero-width (non-)joiners used inside Urdu, Persian and Malayalam words.
# The danda (।, ॥) is a full stop, not part of the word before it.
_UNICODE_WORD_RE = re.compile(
    r"(?:[^\W_]|[\u0300-\u036f\u0900-\u0963\u0966-\u0dff\u0e00-\u0e7f"
    r"\u0e80-\u0eff\u1000-\u109f\u1780-\u17ff\u064b-\u065f\u0670\u200c\u200d])+"
)
# Scripts written without spaces between words: indexed as character bigrams (plus single
# characters for Chinese/Japanese ideographs, which are often words on their own).
_IDEOGRAPH_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_SPACELESS_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u0e00-\u0e7f\u0e80-\u0eff\u1000-\u109f\u1780-\u17ff]+"
)

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


def _spaceless_tokens(run: str) -> list[str]:
    out = [ch for ch in run if _IDEOGRAPH_RE.match(ch)]
    out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out or [run]


def tokenize(text: str) -> list[str]:
    # NFKC also folds Arabic presentation forms (how some PDFs store Urdu) into letters.
    text = unicodedata.normalize("NFKC", text).lower()
    out: list[str] = []
    for word in _UNICODE_WORD_RE.findall(text):
        if word.isascii():
            continue  # ASCII handled below with the code/decimal-aware tokenizer
        word = word.replace("\u200c", "").replace("\u200d", "")
        pos = 0
        for m in _SPACELESS_RE.finditer(word):
            if m.start() > pos:
                out.append(word[pos : m.start()])
            out.extend(_spaceless_tokens(m.group()))
            pos = m.end()
        if pos < len(word):
            out.append(word[pos:])
    for tok in _TOKEN_RE.findall(text):
        if tok in STOPWORDS:
            continue
        out.append(_stem(tok))
        if "-" in tok or "." in tok and not tok.replace(".", "").isdigit():
            out.extend(_stem(p) for p in re.split(r"[.\-]", tok) if p and p not in STOPWORDS)
    return out


class BM25Index:
    """Okapi BM25 with Lucene's always-positive IDF, stored as a sparse weight matrix.

    ``rank_bm25.BM25Okapi`` uses ``log((N - n + 0.5) / (n + 0.5))``, which is
    zero or negative for any term in at least half the chunks — so a short PDF
    (1-2 chunks) or a frequent-but-important term scores nothing.  Lucene's
    ``log(1 + (N - n + 0.5) / (n + 0.5))`` is always positive.

    Each (chunk, term) weight ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``
    is precomputed into a CSC matrix, so a query is a sum of a few sparse columns.
    Measured on 39,852 chunks (16-core laptop CPU): 0.6 ms per query, against 27 ms for the
    previous Python loop over postings, with identical rankings.
    The matrix is saved next to the vectors so a reloaded index skips tokenisation.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        tfs: list[int] = []
        lengths = np.zeros(len(texts), dtype=np.float64)
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                rows.append(i)
                cols.append(vocab.setdefault(term, len(vocab)))
                tfs.append(tf)
        self._set(vocab, len(texts), np.array(rows), np.array(cols), np.array(tfs), lengths)

    def _set(self, vocab, n, rows, cols, tfs, lengths) -> None:
        from scipy.sparse import csc_matrix

        self.vocab = vocab
        self.n_docs = n
        avgdl = float(lengths.mean()) if n else 0.0
        if len(tfs):
            df = np.bincount(cols, minlength=len(vocab)).astype(np.float64)
            idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
            tf = tfs.astype(np.float64)
            norm = self.k1 * (1 - self.b + self.b * lengths[rows] / max(avgdl, 1e-9))
            weights = idf[cols] * tf * (self.k1 + 1) / (tf + norm)
        else:
            weights = np.zeros(0)
        self._weights = csc_matrix((weights, (rows, cols)), shape=(n, max(len(vocab), 1)))

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        """Top-n (index, score) pairs with score > 0, best first."""
        if not self.n_docs:
            return []
        cols = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not cols:
            return []
        scores = np.asarray(self._weights[:, cols].sum(axis=1)).ravel()
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        w = self._weights.tocsc()
        np.savez(
            path / "bm25.npz",
            data=w.data,
            indices=w.indices,
            indptr=w.indptr,
            shape=np.array(w.shape),
            params=np.array([self.k1, self.b]),
        )
        terms = sorted(self.vocab, key=self.vocab.__getitem__)
        (path / "bm25_vocab.json").write_text(json.dumps(terms, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> BM25Index:
        from scipy.sparse import csc_matrix

        path = Path(directory)
        z = np.load(path / "bm25.npz")
        terms = json.loads((path / "bm25_vocab.json").read_text("utf-8"))
        self = cls.__new__(cls)
        self.k1, self.b = (float(x) for x in z["params"])
        self.vocab = {t: i for i, t in enumerate(terms)}
        self.n_docs = int(z["shape"][0])
        self._weights = csc_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"])
        )
        return self

    @staticmethod
    def exists(directory: str | Path) -> bool:
        path = Path(directory)
        return (path / "bm25.npz").is_file() and (path / "bm25_vocab.json").is_file()
