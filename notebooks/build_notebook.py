#!/usr/bin/env python3
"""Generate ``notebooks/Local_Agentic_RAG.ipynb`` (the Google Colab deliverable).

    python notebooks/build_notebook.py

The notebook is generated rather than hand-edited so that its code cells stay
short, reviewable in diffs, and checked by ``tests/test_colab_helpers.py``
(JSON schema validity + every code cell compiles).  All real logic lives in
``local_agentic_rag`` (``colab.py``, ``app.py``, ``bench.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent / "Local_Agentic_RAG.ipynb"


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


CELLS = [
    # ------------------------------------------------------------------ 0. title
    md(r"""
# Local Agentic RAG: ask questions about your PDFs, with page citations

This notebook runs a **fully local** retrieval-augmented generation (RAG) system on a free
Colab **T4 GPU**. No API keys, and no document text leaves the runtime.
You upload a PDF and ask questions. Each answer cites the pages it came from, and the
agent checks its own answer against the source text before showing it.
**Multilingual:** documents and questions can be in different languages (English, Hindi,
Hinglish, Marathi, and the other languages BGE-M3 covers); answers come back in the
question's language, and scanned pages are OCR'd (English, Hindi, Marathi packs installed).

**How a question is answered**

```
 PDF ──► PyMuPDF parse ───────────────► token-aware section chunks ──► index
         (reading order, tables → markdown,   (never cross a page, so
          header/footer strip, OCR for         citations stay exact)
          image-only pages)
                                                    │
 question ─► RETRIEVE ─► GENERATE ─► VERIFY ─► cited answer  (escalates to PLAN ─► RE-RETRIEVE ─► GRADE ─► REGENERATE only when refused or unverified)
             │        │          │          │          │
             │        │          │          │          └ every claim supported? if not, regenerate once
             │        │          │          └ answer with [n] markers, or "not found"
             │        │          └ LLM: which passages are relevant? enough? if not, search again
             │        └ hybrid: BGE-M3 dense + BM25 → RRF fusion → bge-reranker-v2-m3 (multilingual)
             └ LLM: rewrite follow-ups into standalone questions + 1-3 search queries
```

* **Hybrid retrieval** finds both paraphrases (dense embeddings) and exact terms such as
  table labels, IDs and numbers (BM25). **RRF** (reciprocal rank fusion) merges the two
  ranked lists, and a **cross-encoder reranker** reads each question/passage pair together
  to put the best passages first.
* The **agent loop** has a fixed number of rounds, so latency stays predictable. If a step
  fails (for example, invalid JSON from the model), the agent falls back to the simpler
  path instead of failing the question.
* The **LLM** is `qwen2.5:7b-instruct` (4-bit) served by **Ollama**. Ollama ships CUDA
  binaries, so nothing needs to compile.

**Expected runtime on a T4** (rough estimates, they vary with Colab load)

| step | time |
|---|---|
| install the library + Ollama | 2-4 min |
| download the model (4.7 GB) + load it into VRAM | 1-3 min |
| ingest a 20-page PDF | 10-30 s |
| one agentic answer (4-5 LLM calls) | 5-20 s |
| full benchmark (61 questions x 3 configs) | 30-60 min |

Run the cells top to bottom (**Runtime > Run all** works).
"""),
    # ------------------------------------------------------------------ 1. GPU
    md(r"""
## 1. Check the GPU

The 7B model needs about 5.5 GB of VRAM, and the embedder and reranker need about 1 GB more.
A CPU runtime works, but each answer takes minutes. If no GPU is listed below, switch the
runtime: **Runtime > Change runtime type > T4 GPU**, then run the notebook again.
"""),
    code(r"""
!nvidia-smi || echo "No GPU visible: Runtime > Change runtime type > T4 GPU"
"""),
    # ------------------------------------------------------------------ 2. install
    md(r"""
## 2. Get the code and install it

Set `REPO_URL` to clone from GitHub. Otherwise, upload `local-agentic-rag.zip` to
`/content` (Files pane on the left, or run the cell and choose the file when it asks).
We do **not** reinstall `torch` or `numpy`: Colab's builds already match its CUDA driver,
and replacing them is the most common way to break a Colab runtime.
"""),
    code(r"""
REPO_URL = ""                                   # e.g. "https://github.com/<you>/local-agentic-rag.git" (optional)
REPO_ZIP = "/content/local-agentic-rag.zip"     # used when REPO_URL is empty
REPO_DIR = "/content/local-agentic-rag"
FAST_TABLE_DETECTION = True                     # GPU table detection for PDFs (needs a GPU; falls back to the CPU)
"""),
    code(r"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


def unpack(zip_path, dest=REPO_DIR):
    tmp = Path("/content/_repo_unzip")
    shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(zip_path).extractall(tmp)
    root = next(p.parent for p in sorted(tmp.rglob("pyproject.toml"), key=lambda p: len(p.parts)))
    shutil.move(str(root), dest)


if Path(REPO_DIR, "pyproject.toml").exists():
    print(f"Using existing checkout at {REPO_DIR}")
elif REPO_URL:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
elif Path(REPO_ZIP).exists():
    unpack(REPO_ZIP)
else:
    print("No code found. Upload local-agentic-rag.zip (or set REPO_URL and re-run).")
    from google.colab import files
    uploaded = files.upload()
    unpack(next(name for name in uploaded if name.endswith(".zip")))
print("Code at", REPO_DIR)
"""),
    code(r"""
# Tesseract OCR with Hindi + Marathi language packs (for scanned / image-only PDFs).
!apt-get -qq install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar > /dev/null 2>&1 && tesseract --list-langs | tail -n +2 | tr '\n' ' '
!pip install -q -e "{REPO_DIR}[colab,web]"
import importlib
import sys

sys.path.insert(0, f"{REPO_DIR}/src")   # editable installs are not visible to an already-running kernel
importlib.invalidate_caches()
for mod in ["torch", "transformers", "sentence_transformers", "fitz", "gradio", "local_agentic_rag"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:<22} {getattr(m, '__version__', None) or getattr(m, 'VersionBind', 'ok')}")
    except ImportError as exc:
        print(f"{mod:<22} MISSING ({exc})")
"""),
    md(r"""
### Faster table detection on the GPU (optional)

PDF table detection runs a small neural network (PyMuPDF's ONNX layout model). On the CPU it
is the slowest part of reading table-heavy PDFs. With `FAST_TABLE_DETECTION = True` the next cell
swaps in the GPU build of ONNX Runtime and lets the parse workers use the T4: a 1,000-page,
table-heavy PDF parsed **1.9x faster** (225 s to 119 s on a Kaggle T4) with identical output.
It costs about 0.7 GB of GPU memory per parse worker. The cell checks that the GPU build really
works before turning it on, and if anything fails it keeps the CPU build and says so. Set the
flag to `False` to skip it. Run it before ingesting any PDF.
"""),
    code(r"""
def enable_gpu_table_detection():
    probe = ("import onnxruntime as o; o.preload_dlls(); "
             "raise SystemExit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)")

    def run(*cmd):
        try:
            return subprocess.run([sys.executable, *cmd], capture_output=True, timeout=900).returncode == 0
        except Exception:
            return False

    def pip(*args):
        return run("-m", "pip", "-q", *args)

    try:
        if not shutil.which("nvidia-smi") or subprocess.run(["nvidia-smi"], capture_output=True).returncode:
            return "no CUDA GPU visible"
        pip("uninstall", "-y", "onnxruntime")
        if pip("install", "onnxruntime-gpu[cuda,cudnn]") and run("-c", probe):
            os.environ["LOCAL_RAG_ONNX_GPU"] = "1"
            return None
        pip("uninstall", "-y", "onnxruntime-gpu")
        pip("install", "onnxruntime")
        return "the GPU build of ONNX Runtime did not work"
    except Exception as exc:
        pip("install", "onnxruntime")
        return f"{type(exc).__name__}: {exc}"


problem = enable_gpu_table_detection() if FAST_TABLE_DETECTION else "FAST_TABLE_DETECTION is False"
print("Table detection runs on the GPU." if problem is None
      else f"Table detection will run on the CPU ({problem}).")
"""),
    # ------------------------------------------------------------------ 3. LLM
    md(r"""
## 3. Start the local LLM (Ollama)

`setup_llm` installs Ollama, starts `ollama serve` in the background, downloads the model,
and sends a warm-up request so the weights are already in VRAM before the first question.
It is safe to re-run: steps that are already done are skipped.

| model | VRAM (8k context) | notes |
|---|---|---|
| `qwen2.5:7b-instruct` (default) | ~5.5 GB | best quality/speed on a T4, reliable JSON output |
| `qwen2.5:3b-instruct` | ~2.8 GB | about 2x faster, weaker on multi-step questions |
| `llama3.1:8b` | ~6.2 GB | good alternative, slightly slower |
| `gemma2:9b` | ~7.5 GB | strong reader; set `num_ctx=4096` if you run out of memory |

The server log is at `/content/ollama.log`.
"""),
    code(r"""
from local_agentic_rag.colab import print_gpu_report, setup_llm

print_gpu_report()
llm = setup_llm("qwen2.5:7b-instruct", num_ctx=8192)
"""),
    # ------------------------------------------------------------------ 4. pipeline
    md(r"""
## 4. Build the pipeline

This loads the multilingual **BGE-M3** embedder (100+ languages, cross-lingual) and the
**bge-reranker-v2-m3** cross-encoder onto the GPU in fp16.
Every setting lives in one `RAGConfig`, so a run can be reproduced from its JSON. For
example, `RAGConfig(top_k=8, max_retrieval_rounds=3)` changes how many passages and
retrieval rounds the agent uses. The VRAM snapshot below has two parts: torch memory
(embedder and reranker) and total GPU memory. Ollama runs as its own process, so only
`nvidia-smi` sees its memory.
"""),
    code(r"""
from local_agentic_rag.colab import timed, vram_snapshot
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.pipeline import RAGPipeline

pipe = RAGPipeline(RAGConfig(), llm=llm)
with timed("load_retrieval_models"):
    pipe.load_models()
vram_snapshot()
"""),
    # ------------------------------------------------------------------ 5. ingest
    md(r"""
## 5. Add a PDF

Set `PDF_SOURCE = "upload"` to use your own file. The default `"sample"` downloads the
three benchmark PDFs (each checked against its SHA-256 hash) and uses the arXiv paper
*Collapse of Dense Retrievers* (ACL 2025, 17 pages, two-column layout with many tables).
Indexes are cached by file hash, so ingesting the same file again is instant.

Long PDFs (thousands of pages) are read in page batches by parallel worker processes and
indexed batch by batch, with a checkpoint after each one. The status line shows the page
reached and an estimate of the time left. If the runtime stops half-way, running the cell
again continues from the last checkpoint instead of page 1.
"""),
    code(r"""
PDF_SOURCE = "sample"     # "sample" or "upload"
PDF_DIR = "/content/pdfs"
"""),
    code(r"""
from local_agentic_rag.colab import ingest_progress, print_ingest_report

if PDF_SOURCE == "upload":
    from google.colab import files
    os.makedirs("/content/uploads", exist_ok=True)
    uploaded = files.upload()
    pdf_paths = []
    for name, data in uploaded.items():
        Path("/content/uploads", name).write_bytes(data)
        pdf_paths.append(f"/content/uploads/{name}")
else:
    !python {REPO_DIR}/benchmark/fetch_pdfs.py --dest {PDF_DIR}
    pdf_paths = [f"{PDF_DIR}/arxiv-2503.05037.pdf"]

for path in pdf_paths:
    with timed("ingest"):
        report = pipe.ingest(path, progress=ingest_progress())
    print_ingest_report(report)
DOC_ID = report.doc_id
"""),
    # ------------------------------------------------------------------ 6. ask
    md(r"""
## 6. Ask questions

Each answer card shows:
- the answer, with `[n]` markers that link to the cited passages
- a **verification badge**. Green means the verifier found support for every claim. Amber
  means some claims could not be verified (they are listed). Grey means verification did not run.
- the cited pages with snippets, and an expandable **agent trace** showing every step,
  how long it took, and what it decided

The second question is a follow-up. It only makes sense together with the first one, so we
pass `history` and the planner rewrites it into a standalone question. The third question is
not answered in the paper, so the right response is a refusal, not a guess.
(These questions are for the sample paper. Write your own if you uploaded a different PDF.)
"""),
    code(r"""
from local_agentic_rag.colab import display_answer

q1 = "According to the bias-interplay analysis, which biases were the most problematic for dense retrievers?"
a1 = pipe.ask(q1, DOC_ID)
display_answer(a1)
"""),
    code(r"""
q2 = "And which one had a comparatively lower impact?"          # follow-up: needs the chat history
a2 = pipe.ask(q2, DOC_ID, history=[(q1, a1.answer)])
display_answer(a2)
print("Planner rewrote it as:", a2.standalone_question)
"""),
    code(r"""
q3 = "What GPU hardware and how many GPU hours were used to run the experiments?"   # not in the paper
display_answer(pipe.ask(q3, DOC_ID))
"""),
    # ------------------------------------------------------------------ 7. UI
    md(r"""
## 7. Chat UI (Gradio)

This opens a chat app with PDF upload, a document selector, multi-turn chat, and a side panel
showing the citations and agent trace for the latest answer. `share=True` prints a public
`*.gradio.live` link that stays valid while this runtime is alive. **Anyone with the link can use
it**, so it is off by default (`SHARE_PUBLIC_LINK = False` keeps the app inside the notebook). The cell returns right away
and the app keeps running in the background. Run `demo.close()` to stop it. Avoid chatting while
the benchmark below is running, because both use the same GPU.
"""),
    code(r"""
from local_agentic_rag.app import launch_chat_ui

SHARE_PUBLIC_LINK = False   # True prints a public *.gradio.live URL that anyone can open
demo = launch_chat_ui(pipe, share=SHARE_PUBLIC_LINK)
"""),
    md(r"""
### 7b. Document library web app

A fuller app for real use: a library of PDFs that are processed in the background (upload
a 5,000-page PDF and keep asking questions about the others), live progress, answers with
citation chips that open the **source page with the cited passage highlighted**, and document
management (delete, retry). It reuses the models already loaded above. In Colab it opens
through Colab's private port proxy, so it is not public. Uploaded files and indexes live in
`/content/library` and are gone when the runtime is deleted.
"""),
    code(r"""
from local_agentic_rag.colab import launch_web_app

library, web_server = launch_web_app(pipe, library_dir="/content/library", port=8000)
# web_server.should_exit = True   # stops the app
"""),
    # ------------------------------------------------------------------ 8. benchmark
    md(r"""
## 8. Benchmark: what does each component add?

The benchmark has 61 hand-checked questions over three PDFs: an academic paper, a statistical
report that is mostly tables, and a standards document. 9 of the questions are unanswerable.
Each configuration changes **one** design decision compared with the one before it:

* **Retrieval ablation** (no LLM, a few minutes): MiniLM dense →
  BGE-en dense → BM25 → hybrid → hybrid + reranker (English models) → BGE-M3 dense →
  BGE-M3 hybrid + multilingual reranker (default). Metric: is an evidence page in the top-k results?
* **Answer ablation** (LLM, the slow part): `hybrid_rerank` (multilingual retriever, single pass)
  → `agentic` (adaptive agent with the calculator; the default). Metrics: answer accuracy,
  citation precision, refusal accuracy and latency.
* **Multilingual benchmark**: Hindi, Marathi and scanned (OCR) documents, plus English-document
  questions asked in Hindi/Hinglish, comparing the default against the previous English-only retriever.

`LIMIT = 10` gives a quick smoke run. `None` runs all 61 questions. Answers are appended to
`/content/results/answers_<config>.jsonl` as they finish, so **if the runtime disconnects, re-run
the cell and it resumes** from where it stopped (as long as `/content` survived, or you
copied the results to Drive).
"""),
    code(r"""
BENCH_DIR = f"{REPO_DIR}/benchmark"
RESULTS_DIR = "/content/results"
LIMIT = None                     # e.g. 10 for a quick smoke run
!python {BENCH_DIR}/fetch_pdfs.py --dest {PDF_DIR}
"""),
    code(r"""
import json

from IPython.display import Markdown, display

from local_agentic_rag.bench import RETRIEVAL_METRICS, answer_ablation, retrieval_ablation
from local_agentic_rag.eval.report import per_category_markdown, summary_table_markdown

os.makedirs(RESULTS_DIR, exist_ok=True)
with timed("retrieval_ablation"):
    retrieval = retrieval_ablation(BENCH_DIR, pdf_dir=PDF_DIR, base=pipe)
Path(RESULTS_DIR, "retrieval_summary.json").write_text(json.dumps(retrieval, indent=2, default=str))
display(Markdown(summary_table_markdown(retrieval, RETRIEVAL_METRICS)))
"""),
    code(r"""
# "agentic" = adaptive agent with the multilingual retriever (default); "hybrid_rerank" = single pass.
# Add "baseline" (MiniLM, single pass) or "agentic_full" (always-on loop, ~4x slower) for the full ablation.
CONFIGS = ["hybrid_rerank", "agentic"]
with timed("answer_ablation"):
    answers = answer_ablation(llm, BENCH_DIR, pdf_dir=PDF_DIR, configs=CONFIGS,
                              out_dir=RESULTS_DIR, limit=LIMIT, base=pipe)
Path(RESULTS_DIR, "answer_summary.json").write_text(json.dumps(answers, indent=2, default=str))
display(Markdown(summary_table_markdown(answers)))
"""),
    code(r"""
for name, summary in answers.items():
    display(Markdown(f"### {name}: by question category\n\n" + per_category_markdown(summary)))
"""),
    code(r"""
# Error analysis: every question the default agent got wrong (answer, evidence vs retrieved pages, calculator).
!python {REPO_DIR}/scripts/failure_report.py {RESULTS_DIR}/answers_agentic.jsonl | head -120
"""),
    code(r"""
DOWNLOAD_RESULTS = False   # True downloads /content/results.zip to your computer
archive = shutil.make_archive("/content/results", "zip", RESULTS_DIR)
print("Saved", archive)
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(archive)
"""),
    # ------------------------------------------------------------------ 9. multilingual
    md(r"""
## 9. Other languages: Hindi, Hinglish, Marathi, and scanned pages

The embedder (BGE-M3) and reranker (bge-reranker-v2-m3) are multilingual, so a question can be
in a different language from the document. When the scripts differ (or the question is Hinglish),
the agent adds one translated search query in the document's language, and the answer is written
in **the question's language**. Image-only (scanned) pages go through Tesseract OCR with Hindi and
Marathi packs. The multilingual test documents are Wikipedia articles (CC BY-SA), regenerated from
saved text by `benchmark/multilingual/make_pdfs.py`.
"""),
    code(r"""
import json

ML_DIR = f"{REPO_DIR}/benchmark/multilingual"
ML_PDFS = f"{ML_DIR}/pdfs"
os.makedirs(ML_PDFS, exist_ok=True)
!cp -n {PDF_DIR}/arxiv-2503.05037.pdf {ML_PDFS}/ 2>/dev/null; ls {ML_PDFS}
# The Hindi/Marathi PDFs ship with the code; make_pdfs.py (needs LibreOffice) only rebuilds missing ones.
ml_docs = json.loads(Path(ML_DIR, "documents.json").read_text()) if Path(ML_DIR, "documents.json").exists() else []
for d in ml_docs:
    with timed("ingest_" + d["doc_id"]):
        print_ingest_report(pipe.ingest(f"{ML_PDFS}/{d['doc_id']}.pdf", doc_id=d["doc_id"]))
"""),
    code(r"""
# English paper, questions in Hindi and Hinglish: answers should come back in the question's language.
for q in ["इस शोध में लेखकों ने किस डेटासेट का पुनः उपयोग किया?",
          "Authors ne kaunsa dataset repurpose kiya aur har setting mein kitni queries thi?"]:
    display_answer(pipe.ask(q, DOC_ID))
"""),
    code(r"""
# Hindi / Marathi documents asked in English, Hindi and Marathi (skipped if the multilingual set is absent).
mlq = [json.loads(line) for line in Path(ML_DIR, "questions.jsonl").read_text().splitlines()] if ml_docs else []
for q in [q for q in mlq if q["doc_id"] != "arxiv-2503.05037"][:4]:
    print(f"[{q['question_language']}] expected: {q['reference_answer']}")
    display_answer(pipe.ask(q["question"], q["doc_id"]))
"""),
    code(r"""
# Multilingual benchmark: multilingual default vs the previous English-only retriever.
if ml_docs:
    with timed("multilingual_ablation"):
        ml = answer_ablation(llm, ML_DIR, pdf_dir=ML_PDFS, configs=["agentic", "agentic_en"],
                             out_dir=f"{RESULTS_DIR}/multilingual", base=pipe)
    display(Markdown(summary_table_markdown(ml)))
    for name, summary in ml.items():
        rows = [f"| {code} | {v.get('n')} | {v.get('answer_accuracy') or 0:.1%} | {v.get('refusal_accuracy') or 0:.1%} |"
                for code, v in summary.get("per_language", {}).items()]
        display(Markdown(f"**{name} by question language**\n\n| lang | n | answer acc. | refusal acc. |\n|---|---|---|---|\n" + "\n".join(rows)))
"""),
    # ------------------------------------------------------------------ 10. resources
    md(r"""
## 10. Resource report

Peak VRAM and the time each stage took in this session. `gpu_used_mb` (from `nvidia-smi`)
includes the Ollama server. The `torch_*` numbers cover only the embedder and reranker.
"""),
    code(r"""
from local_agentic_rag.colab import print_resource_report

print_resource_report()
"""),
    # ------------------------------------------------------------------ 10. limits
    md(r"""
## 11. Limitations and troubleshooting

**Limitations**
- Answers depend on the 7B model. Questions that need arithmetic across several tables, or
  reading a wide table's correct column, are the weakest categories (see the per-category table).
- Charts and figures are not interpreted. Only text, tables and OCR'd text are indexed.
- The benchmark is small (61 questions), so differences of a few points are noise.
- The Gradio share link is public, and the runtime (with your PDFs) is deleted when Colab disconnects.

**Troubleshooting**

| symptom | fix |
|---|---|
| `Ollama server failed to start` | read the error's log tail, or `!tail -50 /content/ollama.log`; re-run the `setup_llm` cell |
| `LLM error: ... unreachable` while answering | the server died (often an OOM). Re-run `setup_llm`, which restarts it |
| CUDA out of memory / very slow answers | use `qwen2.5:3b-instruct`, or `setup_llm(..., num_ctx=4096)`, then rebuild the pipeline |
| a scanned PDF gives `empty pages` / OCR warnings | `!apt-get install -y tesseract-ocr` and ingest again with `pipe.ingest(path, use_cache=False)` |
| `ModuleNotFoundError: local_agentic_rag` | re-run the install cell (it adds `src/` to `sys.path`) or restart the runtime |
| benchmark interrupted | re-run the cell; finished answers are reused from `/content/results` |
| Colab: "cannot connect to a GPU due to usage limits" | the notebook also runs unchanged on a Kaggle notebook with a T4 (Settings > Accelerator, internet on): copy the code to `/content/local-agentic-rag` and Run all |
"""),
]


def build() -> nbformat.NotebookNode:
    nb = new_notebook(cells=CELLS)
    for i, cell in enumerate(nb.cells):  # stable ids: rebuilding an unchanged notebook is a no-op
        cell.id = f"cell-{i:02d}"
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": [], "name": OUT.name},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    nb = build()
    nbformat.validate(nb)
    nbformat.write(nb, out)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out} ({len(nb.cells)} cells, {n_code} code)")


if __name__ == "__main__":
    main()
