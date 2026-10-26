# Local Agentic RAG for PDFs

Ask questions about any PDF and get answers **with page citations**, using only
open-weight models running locally. The whole system fits on a **free Google
Colab T4**.

```
PDF ─► PyMuPDF (reading order, tables, header/footer strip, OCR eng+hin+mar)
    ─► heading-aware 320-token chunks
    ─► hybrid retrieval: BGE-M3 (multilingual) + Unicode BM25 ─► RRF ─► bge-reranker-v2-m3
    ─► agent: GENERATE ─► VERIFY ─► (refused/unsupported? PLAN ─► RE-RETRIEVE ─► GRADE ─► REGENERATE ─► VERIFY)
    ─► answer with [n] citations → pages, or an explicit "not in the document"
```

* **Grounded:** the model only sees numbered passages and must cite them. Invalid
  citations are dropped, and a verifier step checks every claim against the passages.
* **Agentic where it pays off (measured):** every answer is verified against its
  passages. Only when the first answer is refused or unsupported does the agent
  escalate: it decomposes the question, re-retrieves, grades the evidence and
  regenerates with feedback. Follow-up questions are rewritten using chat history.
  An always-on loop was measured to be *worse* than this; see docs/RESULTS.md.
* **Multilingual:** documents and questions can be in different languages (English,
  Hindi, Hinglish, Marathi, and the other languages BGE-M3 covers). Retrieval is
  cross-lingual (BGE-M3 + bge-reranker-v2-m3, Unicode-aware BM25), a translated query is
  added when the question and document scripts differ, answers come back in the
  question's language, and scanned pages are OCR'd (Tesseract eng+hin+mar). Measured on a
  separate 20-question Hindi/Marathi/Hinglish/OCR set; limits in docs/STATUS.md.
* **Robust ingestion:** two-column papers, ruled tables and whitespace-aligned
  statistical tables (rebuilt cell-by-cell), running headers, scanned pages (OCR).
* **Measured:** a 61-question hand-verified benchmark over 3 public PDFs, with
  retrieval and end-to-end ablations against the original baseline; a scale benchmark from
  100 to 10,000 pages; and an independent question set on four long real PDFs.
* **Built for long documents:** streaming, checkpointed ingestion with crash-isolated parse
  workers, an on-disk index, and a web app with a background processing queue.

![Chat UI running on a free Colab T4](docs/img/chat_ui_colab_t4.png)

![Hindi and Hinglish questions about an English paper, answered in the question's language](docs/img/multilingual_answers_colab_t4.png)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and rationale,
[docs/RESULTS.md](docs/RESULTS.md) for benchmark numbers, and
[docs/STATUS.md](docs/STATUS.md) for what is done and what is outstanding.

## Quick start (Colab)

Open `notebooks/Local_Agentic_RAG.ipynb` in Colab, select **Runtime → Change
runtime type → T4 GPU**, and run the cells in order. The notebook:

1. installs the package and starts an Ollama server with `qwen2.5:7b-instruct`,
2. lets you upload a PDF and ask questions (with a Gradio chat UI),
3. optionally reproduces the benchmark.

## Web app (document library)

A browser app for real use: upload PDFs of any length, watch them process in the background,
ask questions across any subset of documents, and click a citation to see the source page
with the cited passage highlighted.

```bash
pip install -e ".[web]"
ollama serve & ollama pull qwen2.5:7b-instruct          # the local LLM
python -m local_agentic_rag.server --library ~/rag-library --port 8000
# open http://127.0.0.1:8000/
```

* **Large PDFs:** uploads stream to disk (default limit 2 GB). Processing runs in a
  background queue with live progress, page counts and an ETA. A 10,000-page PDF is read by
  parallel, crash-isolated worker processes and indexed in checkpointed batches. If the
  server stops, the job resumes from its last checkpoint on restart.
* **Several users:** answers run with a configurable concurrency limit
  (`--max-concurrent-asks`, default 2 for one GPU); others queue and see their position.
  Questions can be asked while another document is still processing.
* **Sources you can check:** every answer has citation chips (`[1] p.42`) that open the
  rendered page with the passage highlighted, plus a "how this answer was found" trace.
* **Private:** everything runs locally; it listens on 127.0.0.1 unless you pass `--host`.
* In Colab, the notebook's section 7b starts the same app through Colab's private port proxy.
* `python -m local_agentic_rag.server --demo` runs the interface with a fake backend
  (no GPU or LLM needed), for trying the UI.

## Quick start (Python)

```python
from local_agentic_rag.config import RAGConfig
from local_agentic_rag.pipeline import RAGPipeline
from local_agentic_rag.llm.ollama import OllamaLLM   # or llm.llamacpp.LlamaCppLLM("model.gguf")

llm = OllamaLLM("qwen2.5:7b-instruct"); llm.ensure_ready()
pipe = RAGPipeline(RAGConfig(), llm=llm)
print(pipe.ingest("report.pdf"))
ans = pipe.ask("What was revenue growth in 2024?")
print(ans.answer, ans.cited_pages, ans.grounded)
follow = pipe.ask("And in 2023?", history=[("What was revenue growth in 2024?", ans.answer)])
```

## Results at a glance (final code, project notebook run top to bottom on a T4, 2026-10-26)

| | English set (61 q) | Multilingual set (20 q) |
|---|---|---|
| Page recall@5, original baseline → default retriever | 73.1% → 94.2% (recall@1 43% → 68%; English-only reranker 96.2%) | English-only → multilingual models: 0.944 → 1.000 |
| Answer accuracy, default adaptive agent | 76.9% [63.5–88.5]; single pass 71.2%; original baseline design 50.0% | 88.9% (16/18) in the final run, 94.4% in the previous run of the same code: Hindi 7–8/8 (incl. scanned/OCR), Hinglish 3/3, Marathi 3/3, English→Indic 3/4 |
| Unanswerable questions refused | 9 / 9 | 2 / 2 |
| GPU memory in use (all models loaded) | 8.9 of 15 GB | |

The final runs used a Kaggle notebook with the same GPU (Tesla T4, pinned to one card) because the
free Colab GPU quota was used up; the notebook itself is unchanged and targets Colab.

**Long documents and several users** (Kaggle T4, details in [docs/SCALE.md](docs/SCALE.md)):

| | 100 pages | 1,000 pages | 10,000 pages |
|---|---:|---:|---:|
| Ingestion (2 parse workers, table-heavy synthetic PDF) | 55 s | 5.7 min | 71 min |
| Page recall@5 / answer accuracy (40 independent questions per size) | 0.97 / 0.75 | 0.94 / 0.72 | 0.89 / 0.72 |
| Answer latency p50 | 6.5 s | 5.9 s | 6.4 s |
| Index on disk / cold load | 2 MB / 0.0 s | 19 MB / 0.1 s | 188 MB / 0.7 s |

A crashed parse worker costs one page; an interrupted job resumes from its checkpoint (tested at
every size). On real PDFs (9/11 Report, Statistical Abstract 2012, US Code Title 26) with 46
independently written questions: 26/37 answerable correct, 6/9 unanswerable refused. The web app
served 1–8 concurrent users with no failed request (about 10.5 answers per minute on one T4).

A 4-page progress report with the architecture, benchmarks and plan is in [report/](report/Local_Agentic_RAG_Progress_Report.pdf).
Details, confidence intervals, per-category and per-language breakdowns: [docs/RESULTS.md](docs/RESULTS.md).

## Benchmark

```bash
python benchmark/fetch_pdfs.py                       # downloads the 3 public PDFs (sha256-verified)
python -m local_agentic_rag.bench retrieval          # retrieval ablation, CPU is fine
python -m local_agentic_rag.bench answer --llm ollama:qwen2.5:7b-instruct
# multilingual set (Hindi / Marathi / Hinglish / OCR)
python -m local_agentic_rag.bench answer --benchmark-dir benchmark/multilingual --configs agentic agentic_en
```

### Scale benchmark (100 to 10,000 pages)

```bash
python benchmark/scale/make_corpus.py --pages all --out local/scale/corpus   # synthetic, with planted facts
python benchmark/scale/run_scale.py --tiers 100 250 1000 2500 5000 10000 \
    --llm ollama:qwen2.5:7b-instruct --recovery --out local/scale/results
python benchmark/scale/run_realdocs.py --llm ollama:qwen2.5:7b-instruct      # real public-domain PDFs
python benchmark/scale/load_test.py --url http://127.0.0.1:8000 --users 1 2 4 8 \
    --questions local/scale/corpus/scale-1000.questions.jsonl
```

`run_scale.py` measures ingestion time per stage, peak RAM and GPU memory, index size,
cold-load time, retrieval recall/latency by question type and document depth, answer
accuracy and citation precision, and failure recovery (a crashed parse worker plus a job
interrupted half-way and resumed). Every number it writes is measured in that run; tiers not
run are reported as "not run". See `benchmark/scale/README.md` for what the synthetic corpus
contains and its limits.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # offline unit tests (models are faked)
ruff check src tests
```

## License

Not yet licensed: this is a private repository, and a license will be chosen before any
public release. Note that PyMuPDF (the PDF parser) is AGPL-3.0, which affects that choice;
`pypdfium2` is the usual replacement if a permissive license is needed.

The English benchmark PDFs are downloaded from their public sources and are not stored in
the repository. The multilingual test PDFs are generated from Wikipedia text
(CC BY-SA 4.0, attribution in `benchmark/multilingual/README.md`).
