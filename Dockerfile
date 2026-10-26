# Local Agentic RAG web app (document library + chat) in one image.
#
#   docker build -t local-agentic-rag .
#   docker run --gpus all -p 127.0.0.1:8000:8000 -v rag-data:/data \
#       --add-host host.docker.internal:host-gateway \
#       -e LOCAL_RAG_OLLAMA_HOST=host.docker.internal:11434 local-agentic-rag
#
# Ollama (the language model server) is NOT in this image: run it on the host or use
# docker-compose.yml, which starts it as a second container.
#
# GPU: the torch wheel installed from pip bundles its CUDA libraries, so the container only
# needs the host's NVIDIA driver exposed through the NVIDIA Container Toolkit (`--gpus all`).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    LOCAL_RAG_CACHE=/data/cache

# Tesseract with English, Hindi and Marathi data for scanned pages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[web]"

# Non-root user. /data is created and owned before the VOLUME line so that a fresh named
# volume inherits the right owner.
RUN useradd --create-home --uid 1000 rag \
    && mkdir -p /data/library /data/hf /data/cache \
    && chown -R rag:rag /data
USER rag
VOLUME /data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

# The Ollama address and model come from LOCAL_RAG_OLLAMA_HOST and LOCAL_RAG_MODEL
# (see src/local_agentic_rag/server/__main__.py); defaults are 127.0.0.1:11434 and
# qwen2.5:7b-instruct.
CMD ["python", "-m", "local_agentic_rag.server", "--library", "/data/library", "--host", "0.0.0.0", "--port", "8000"]
