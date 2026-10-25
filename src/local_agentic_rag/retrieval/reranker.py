"""
Cross-encoder reranker.

A bi-encoder embeds question and passage separately; a cross-encoder reads
them *together* and is far more precise at deciding "does this passage
answer this question".  It is too slow to score a whole document, so it only
re-scores the ~24 fused hybrid candidates.  Its calibrated probability is
also what the agent uses as a cheap first relevance signal.
"""

from __future__ import annotations

import logging

import numpy as np

from local_agentic_rag.retrieval.embedder import resolve_device

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s on %s", self.model_name, self.device)
        self._model = CrossEncoder(self.model_name, max_length=512, device=self.device)
        if self.device.startswith("cuda"):
            self._model.model.half()

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance probability in [0, 1] for each passage."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        self.load()
        raw = self._model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw, dtype=np.float32).reshape(-1)
        # Older/newer sentence-transformers differ on whether a sigmoid is applied.
        if scores.min() < 0.0 or scores.max() > 1.0:
            scores = 1.0 / (1.0 + np.exp(-scores))
        return scores

    def unload(self) -> None:
        self._model = None
