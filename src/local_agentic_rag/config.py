"""
Runtime configuration for the agentic RAG pipeline.

One dataclass holds every tunable so that a benchmark run can be reproduced
from a single JSON blob (``RAGConfig.to_dict()`` is written into every result
file).  Ablation configurations are built with ``dataclasses.replace``.

Defaults are chosen for a free Colab T4 (15 GB VRAM):
    embedder  BAAI/bge-m3               ~1.1 GB fp16 (multilingual, cross-lingual)
    reranker  BAAI/bge-reranker-v2-m3   ~1.1 GB fp16 (multilingual)
    LLM       qwen2.5:7b-instruct Q4_K_M ~5.5 GB incl. 8k KV cache (served by Ollama)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

RetrievalMode = Literal["dense", "bm25", "hybrid"]

#: Bump whenever parsing/chunking code changes output, so cached indexes are rebuilt.
INGEST_VERSION = "2026.10.26-11"  # ... + overlap-seeded chunks take the section number of their new text

#: Environment flag: run PyMuPDF's table-layout ONNX model on the GPU in the parse workers
#: (needs ``onnxruntime-gpu`` and CUDA; falls back to the CPU silently).  Off by default
#: because GPU arithmetic can differ from the CPU's in the last bits, which could move a
#: table box on a borderline page.  ``INGEST_VERSION`` is not bumped: nothing changes unless
#: the flag is switched on, and chunk identity must then be checked on that machine.
ONNX_GPU_ENV = "LOCAL_RAG_ONNX_GPU"


def onnx_gpu_requested(environ: Any = None) -> bool:
    """True when ``LOCAL_RAG_ONNX_GPU`` is set to a truthy value (1, true, yes, on)."""
    import os

    env = os.environ if environ is None else environ
    return str(env.get(ONNX_GPU_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RAGConfig:
    # --- Ingestion -----------------------------------------------------
    chunk_tokens: int = 320  # target chunk size (embedder tokens)
    chunk_overlap_tokens: int = 48  # carried-over context between chunks
    min_chunk_tokens: int = 24  # tiny trailing fragments are merged
    strip_headers_footers: bool = True
    extract_tables: bool = True
    ocr: Literal["auto", "off"] = "auto"  # OCR image-only pages if tesseract exists
    ocr_language: str = "eng+hin+mar"  # tesseract languages; falls back to "eng" if missing

    # --- Retrieval -----------------------------------------------------
    # Multilingual by default: BGE-M3 (100+ languages, cross-lingual) + its reranker.
    embedder_model: str = "BAAI/bge-m3"
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"
    retrieval_mode: RetrievalMode = "hybrid"
    candidates_per_retriever: int = 30  # dense top-N and BM25 top-N before fusion
    rrf_k: int = 60
    rerank_candidates: int = 24  # fused candidates sent to the cross-encoder
    top_k: int = 6  # passages given to the generator
    device: str | None = None  # None = auto (cuda if available)

    # --- Agent -----------------------------------------------------------
    agentic: bool = True  # False = single-pass retrieve → generate
    agent_mode: Literal["adaptive", "full"] = "adaptive"
    # adaptive: answer single-pass, verify, and run the plan/re-retrieve/regenerate loop only
    #           when the first answer is refused or fails verification (measured best).
    # full:     always plan → retrieve⇄grade → generate⇄verify.
    use_planner: bool = True  # query decomposition / search-query writing
    use_grader: bool = True  # evidence relevance + sufficiency check
    use_verifier: bool = True  # post-hoc grounding verification
    translate_queries: bool = True  # cross-lingual: add a query in the document's language
    use_calculator: bool = True  # deterministic arithmetic/counting tool (agentic modes)
    max_retrieval_rounds: int = 2  # initial retrieval + (n-1) corrective rewrites
    max_answer_attempts: int = 2  # generation + one regeneration on failed verification
    grader_override_score: float = 0.2  # keep grader-rejected passages above this reranker score
    history_turns: int = 3  # chat turns used to condense follow-ups
    # F11 (experimental): after the answer is final, if the verifier did NOT support it and the
    # question names a year / proper-noun phrase / code found in none of the retrieved
    # passages, replace the answer with the standard refusal (agent/presupposition.py).
    presupposition_check: bool = False
    # Numerical questions (agent/calculator.py). Measured on the English dev set (Kaggle runs
    # 5-6, 27 Oct): alignment + date candidates + measure-aware retrieval took accuracy from
    # 0.769 to 0.846 (numerical 2/6 -> 5/6) with no p95 latency cost and multilingual
    # unchanged, so those three are on by default. The measure check is off: on its own it
    # added +6.7 s p95 and its one gain also came from measure-aware retrieval.
    # - calc_cell_alignment: an operand copied from a table cell other than the one its label
    #   names ("gpt-4o | No Doc" = 52.0, which is gpt-4o-mini's cell) is replaced by the
    #   named cell's value, when that cell is unambiguous.
    # - calc_measure_check: an operand from a table whose measure contradicts the question
    #   (year-over-year vs annualized, percent vs dollars, contribution vs rate) is rejected;
    #   a question that needs arithmetic and ends with no grounded calculation then escalates
    #   (wider search) even if the verifier accepted the first answer.
    # - calc_date_candidates: for time-span questions the calculator is shown the month-year
    #   dates found in the passages, the title-page date marked as the publication date.
    # - measure_aware_retrieval: after reranking, table passages none of whose numbers can be
    #   the measure the question asks for are moved behind the others (stable order).
    measure_aware_retrieval: bool = True
    calc_cell_alignment: bool = True
    calc_measure_check: bool = False
    calc_date_candidates: bool = True

    # --- Generation ------------------------------------------------------
    temperature: float = 0.0
    max_answer_tokens: int = 400
    seed: int = 42

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self, keys: tuple[str, ...] | None = None) -> str:
        """Short stable hash of (a subset of) the config — used for cache keys."""
        data = self.to_dict()
        if keys is not None:
            data = {k: data[k] for k in keys}
        blob = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def ingestion_fingerprint(self) -> str:
        return (
            INGEST_VERSION
            + "-"
            + self.fingerprint(
                (
                    "chunk_tokens",
                    "chunk_overlap_tokens",
                    "min_chunk_tokens",
                    "strip_headers_footers",
                    "extract_tables",
                    "ocr",
                    "ocr_language",
                    "embedder_model",
                )
            )
        )

    def with_overrides(self, **overrides: Any) -> RAGConfig:
        valid = {f.name for f in fields(self)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return replace(self, **overrides)


#: Named configurations used by the ablation benchmark.  Each one isolates
#: a single design decision relative to its neighbour.
PRESETS: dict[str, dict[str, Any]] = {
    # The starting design: MiniLM dense-only, no reranker, no agent.
    "baseline": {
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reranker_model": None,
        "retrieval_mode": "dense",
        "agentic": False,
        "top_k": 5,
        "measure_aware_retrieval": False,
        "calc_cell_alignment": False,
        "calc_date_candidates": False,
    },
    # Better retrieval (hybrid BGE + BM25 + cross-encoder), still single pass.
    "hybrid_rerank": {"agentic": False},
    # Previous English-only retriever, kept to measure the multilingual switch.
    "agentic_en": {
        "embedder_model": "BAAI/bge-base-en-v1.5",
        "reranker_model": "BAAI/bge-reranker-base",
    },
    # Adaptive agent (default config): single pass first, escalate on refusal / failed check.
    "agentic": {},
    # Always-on agent loop, kept for the ablation.
    "agentic_full": {"agent_mode": "full"},
    # Adaptive agent + the F11 presupposition gate (flag off in every other preset).
    "agentic_presup": {"presupposition_check": True},
    # The default agent before the numerical fixes (results up to 27 Oct were measured with
    # this), and each fix added to it on its own (the run 5 ablation).
    "agentic_pre_numeric": {
        "measure_aware_retrieval": False,
        "calc_cell_alignment": False,
        "calc_date_candidates": False,
    },
    "agentic_align": {
        "calc_cell_alignment": True,
        "measure_aware_retrieval": False,
        "calc_date_candidates": False,
    },
    "agentic_measure": {
        "calc_measure_check": True,
        "measure_aware_retrieval": False,
        "calc_cell_alignment": False,
        "calc_date_candidates": False,
    },
    "agentic_dates": {
        "calc_date_candidates": True,
        "measure_aware_retrieval": False,
        "calc_cell_alignment": False,
    },
    "agentic_measure_retrieval": {
        "measure_aware_retrieval": True,
        "calc_cell_alignment": False,
        "calc_date_candidates": False,
    },
    "agentic_numeric": {
        "calc_cell_alignment": True,
        "calc_measure_check": True,
        "calc_date_candidates": True,
        "measure_aware_retrieval": True,
    },
    # The current default, named for the run 6 comparison.
    "agentic_numeric_lite": {
        "calc_cell_alignment": True,
        "calc_date_candidates": True,
        "measure_aware_retrieval": True,
    },
}


def preset(name: str, **overrides: Any) -> RAGConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return RAGConfig().with_overrides(**{**PRESETS[name], **overrides})
