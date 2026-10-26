"""Web product: FastAPI server + offline single-page UI over a document library.

python -m local_agentic_rag.server --library ./library          # real library
python -m local_agentic_rag.server --demo                       # UI demo, no GPU/LLM
"""

from local_agentic_rag.server.app import create_app

__all__ = ["create_app"]
