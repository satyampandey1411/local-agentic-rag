"""Build notebooks/Quick_Start.ipynb: one cell that installs everything and opens the app.

python notebooks/build_quickstart.py
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = "https://github.com/satyampandey1411/local-agentic-rag.git"
OCR_PACKS = "osd hin mar san nep urd ara ben asm tel tam kan mal guj pan ori"

INTRO = """# Local Agentic RAG: quick start

Ask questions about your PDFs, in many languages, with page citations you can click.
Everything runs on this Colab GPU: no API key, and your PDFs never leave this runtime.

1. **Runtime > Change runtime type > T4 GPU** (if Colab did not pick it already).
2. Press **Run** on the cell below. The first start takes a few minutes: it installs the app,
   downloads the answer model (Qwen2.5-7B, about 4.7 GB) and the search models.
3. The app opens under the cell. Upload PDFs on the left, pick the answer language under the
   question box if you want one, and ask.

The full notebook with benchmarks is `notebooks/Local_Agentic_RAG.ipynb`."""

CELL = f"""#@title Start Local Agentic RAG {{ display-mode: "form" }}
import os, sys, time
t0 = time.time()
REPO = "/content/local-agentic-rag"
def step(msg):
    print(f"[{{time.time() - t0:5.0f}} s] {{msg}}", flush=True)

step("Checking the GPU")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU: Runtime > Change runtime type > T4 GPU"
if not os.path.exists(f"{{REPO}}/pyproject.toml"):
    step("Downloading the code from GitHub")
    !git clone -q --depth 1 {REPO_URL} {{REPO}}
step("Installing OCR language packs and the app")
!apt-get -qq install -y tesseract-ocr {" ".join("tesseract-ocr-" + p for p in OCR_PACKS.split())} > /dev/null 2>&1
!pip install -q -e "{{REPO}}[colab,web]" > /dev/null 2>&1
sys.path.insert(0, f"{{REPO}}/src")

from local_agentic_rag.colab import launch_web_app, setup_llm
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.pipeline import RAGPipeline

step("Starting the answer model (Qwen2.5-7B)")
llm = setup_llm("qwen2.5:7b-instruct", num_ctx=8192)
step("Loading the search models (BGE-M3 and the reranker)")
pipe = RAGPipeline(RAGConfig(), llm=llm)
pipe.load_models()
step("Ready. The app opens below.")
library, web_server = launch_web_app(pipe, library_dir="/content/library", port=8000, height=900, tab_link=True)
"""


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(INTRO), new_code_cell(CELL)]
    nb.cells[1].metadata["cellView"] = "form"
    nb.metadata.update(
        {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "name": "Quick_Start.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )
    return nb


if __name__ == "__main__":
    out = Path(__file__).with_name("Quick_Start.ipynb")
    nb = build()
    nbformat.validate(nb)
    nbformat.write(nb, out)
    print("wrote", out)
