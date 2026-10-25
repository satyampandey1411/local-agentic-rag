"""
Dense embedder (sentence-transformers) with model-specific query/passage prefixes.

BGE v1.5, E5 and Nomic models are trained with asymmetric instructions; using
the wrong (or no) prefix costs several points of recall.  ``PREFIXES`` maps a
model-name substring to its (query_prefix, passage_prefix).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

PREFIXES: dict[str, tuple[str, str]] = {
    "bge-m3": ("", ""),  # multilingual BGE-M3 is trained without an instruction (checked first)
    "bge-": ("Represent this sentence for searching relevant passages: ", ""),
    "e5-": ("query: ", "passage: "),
    "nomic-embed": ("search_query: ", "search_document: "),
}


def resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Embedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = None
        lowered = model_name.lower()
        self.query_prefix, self.passage_prefix = next(
            (p for key, p in PREFIXES.items() if key in lowered), ("", "")
        )

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedder %s on %s", self.model_name, self.device)
        kwargs = {"trust_remote_code": True} if "nomic" in self.model_name.lower() else {}
        model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
        if self.device.startswith("cuda"):
            model = model.half()
        model.eval()
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def max_tokens(self) -> int:
        return int(getattr(self.model, "max_seq_length", 512) or 512)

    def count_tokens(self, text: str) -> int:
        tokenizer = self.model.tokenizer
        return len(
            tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + q for q in queries])

    def unload(self) -> None:
        self._model = None
