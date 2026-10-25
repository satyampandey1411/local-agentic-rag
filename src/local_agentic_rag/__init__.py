"""local_agentic_rag — local agentic RAG over PDFs with cited answers.

from local_agentic_rag import RAGPipeline, RAGConfig, OllamaLLM
"""

from local_agentic_rag.config import RAGConfig, preset
from local_agentic_rag.llm.ollama import OllamaLLM
from local_agentic_rag.pipeline import RAGPipeline
from local_agentic_rag.schema import Answer, IngestReport

__all__ = ["Answer", "IngestReport", "OllamaLLM", "RAGConfig", "RAGPipeline", "preset"]
__version__ = "0.1.0"
