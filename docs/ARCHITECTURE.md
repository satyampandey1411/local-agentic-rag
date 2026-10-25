# Architecture

A local, open-source chatbot that answers questions about any uploaded PDF with
page-level citations, using only open-weight models that fit on a free Google
Colab T4 GPU (15 GB VRAM).

## 1. End-to-end flow

```
                      INGEST (once per PDF, cached by sha256 + config)
 ┌──────────┐   ┌──────────────────────────────┐   ┌───────────────────────┐   ┌─────────────────────┐
 │ PDF file │──►│ PyMuPDF parse                │──►│ Structure-aware chunks │──►│ DocumentIndex        │
 └──────────┘   │ • column-aware reading order │   │ • ≤320 embedder tokens │   │ • BGE-base vectors   │
                │ • ruled tables → Markdown    │   │ • split at headings,   │   │ • BM25 inverted idx  │
                │ • whitespace tables → rows   │   │   paragraphs, sentences│   │ • chunks + metadata  │
                │   "label | 2025 Q2 = 3.8"    │   │ • never cross pages    │   │ (saved to disk)      │
                │ • strip running headers      │   │ • "title > section"    │   └─────────────────────┘
                │ • OCR image-only pages       │   │   header for embedding │
                └──────────────────────────────┘   └───────────────────────┘

                      ASK (per question) — default "adaptive" agent mode
 question + chat history
        │
        ├─ history present? ──► PLAN: rewrite follow-up as a standalone question        (1 LLM call, JSON)
        ▼
 ┌─────────────┐  dense top-30 + BM25 top-30 → Reciprocal Rank Fusion → top-24
 │  RETRIEVE   │  → cross-encoder rerank against the question → top-6               (0 LLM calls)
 └─────┬───────┘
       ▼
 quantitative question? ──► CALCULATE: LLM extracts operands + expression (JSON),    (1 LLM call)
                            Python evaluates it safely; result handed to GENERATE
 ┌─────────────┐  answer from numbered passages, cite [n], or ANSWER_NOT_FOUND       (1 LLM call)
 │  GENERATE   │
 └─────┬───────┘
       ▼
 ┌─────────────┐  every claim supported by the passages?                            (1 LLM call, JSON)
 │   VERIFY    │
 └─────┬───────┘
       │ supported ───────────────────────────────────────────────► answer (2 calls total)
       │ refused or unsupported → ESCALATE:
       ▼
 PLAN sub-queries (one per compared item / missing fact)                             (1 LLM call, JSON)
 RETRIEVE again: ranked by the question; each sub-query's best passage gets a slot
 GRADE: drop passages that the grader AND the reranker both reject                   (1 LLM call, JSON)
 GENERATE with feedback about the unsupported claims, VERIFY again                   (2 LLM calls)
 if escalation only yields a refusal, keep the first (flagged) answer
       ▼
 answer + citations (marker → chunk → page) + grounded flag + trace
```

Cost: 2 LLM calls for most questions, at most 6. `agent_mode="full"` keeps the
original always-on loop (plan → retrieve⇄grade → generate⇄verify) for comparison.

**Why adaptive, which we measured.** The always-on loop (v1) was *worse* than a single pass on
the same questions (23 vs 27 of 38). Ranking by the max reranker score over the planner's
keyword queries let sub-query matches crowd out the passage that answers the actual
question. Letting the grader reorder the context also pushed evidence down. Fixes:
(1) rank by the user's question and let sub-queries only *add* their best passage;
(2) the grader filters but never reorders; (3) spend the loop only on questions whose first
answer is refused or fails verification. See `docs/RESULTS.md`.

## 2. Why each component exists (and what it replaced)

| Stage | Choice | Problem it fixes (evidence) |
|---|---|---|
| Parsing | PyMuPDF | pdfminer (used by the starting code) is slow and loses layout. Reference `unstructured hi_res` downloads layout models and takes minutes per PDF. PyMuPDF parses a 64-page PDF in ~4 s. |
| Reading order | column classification + bands | Naive y-then-x sorting interleaves the two columns of an academic paper line by line. |
| Tables | `find_tables` → Markdown; whitespace tables rebuilt from word coordinates | On BEA-style statistical tables `find_tables` stacks whole columns into one cell (`2.5<br>2.9<br>…`). Row serialisation with header labels lets a 7B model read the right cell. |
| Chunking | ≤320 *embedder* tokens, heading-aware, page-bound | The starting code's 2048-char chunks were silently truncated to 256 tokens by MiniLM. That is roughly half of every chunk invisible to retrieval. |
| Embedder | BAAI/bge-base-en-v1.5 (+ query instruction) | MiniLM is weak on long, technical passages. |
| Keyword search | BM25, decimal/code-aware tokenizer | Dense vectors blur exact numbers, acronyms and table labels. |
| Fusion | Reciprocal Rank Fusion | The reference's min-max + 0.7/0.3 weighting always rescales the best hit to 1.0, even when nothing matches. RRF uses ranks only and needs no tuning. |
| Reranker | BAAI/bge-reranker-base cross-encoder | Bi-encoder similarity is coarse. The cross-encoder reads question and passage together. |
| Agent | bounded Python state machine | Reference grades passages one LLM call at a time (6 calls/round). We grade in one batched call and add planning, verification and chat memory. No LangGraph dependency for 5 nodes. |
| Citations | numbered passages → `[n]` → chunk → page; invalid markers dropped; overlap fallback | Reference never gives page numbers to the LLM, so it cannot cite. |
| Refusal | `ANSWER_NOT_FOUND` sentinel + uncited "document does not say" detection | Unanswerable questions must not get confident fabricated answers. |
| Calculator tool | LLM picks a **structured operation** (difference, change, sum, product, ratio, percent_change, count, months_between) and copies the operands with citations → Python computes it. If the answer omits the result, a cited "(Calculated: … = x)" sentence is appended | Every configuration scored 0% on numerical questions: evidence retrieved, arithmetic and counting wrong (222.8 vs 443.6; 8 vs 9 items). Free-form expressions often failed to parse, and the generator sometimes ignored the result. |
| LLM serving | Ollama (prebuilt CUDA, JSON-schema output) | llama-cpp-python CUDA wheels for current Colab are unreliable, and source builds take 15+ min. Ollama installs in about a minute. |

## 2b. Multilingual design

| Concern | What the code does | Where |
|---|---|---|
| Cross-language retrieval | **BGE-M3** embedder (100+ languages, one shared vector space) + **bge-reranker-v2-m3** cross-encoder | `config.py` defaults |
| Keyword search in any script | Unicode tokenizer that keeps Indic vowel signs/combining marks inside words (a plain `\w+` splits "निर्माण" at every matra); CJK indexed per character; ASCII path keeps decimals/codes | `retrieval/bm25.py` |
| Question ≠ document language, Hinglish | Script detection + Hinglish function-word check; if they differ, **one LLM call translates the question into the document's language** (given a text sample). Passages are ranked by the best of original and translation | `lang.py`, `agent._lead_queries`, `HybridRetriever(n_primary=2)` |
| Answer language | Generator rule: answer in the question's language and script (Hinglish → Hinglish), keep names/numbers/[n] | `agent/prompts.py` |
| Refusals | Localised refusal message (Devanagari questions get a Hindi + English message) | `lang.refusal_message` |
| Scanned PDFs | **Tesseract CLI** on a 300-dpi render with `eng+hin+mar` (falls back to `eng` if a pack is missing, then to PyMuPDF's OCR layer). Measured on a scanned Hindi page: 99% word recall vs 70% for PyMuPDF's OCR text layer, which split Devanagari words at vowel signs. OCR text becomes plain paragraphs, so OCR noise can't masquerade as headings | `ingest/pdf.py` |
| Grading | Indic digits (१६३२) normalised to ASCII | `eval/metrics.py` |

Limits (not claimed): Hindi vs Marathi are not distinguished by script detection (the LLM sees the
question and is told to mirror its language); generation quality in Marathi and lower-resource
languages is bounded by Qwen2.5-7B; OCR quality depends on scan quality and the installed Tesseract packs.

## 3. Memory budget on Colab (T4 15 GB VRAM, 12.7 GB host RAM)

| Component | Approx. VRAM |
|---|---|
| Qwen2.5-7B-Instruct Q4_K_M + 8k KV cache (Ollama) | ~5.5–6 GB |
| BGE-M3 embedder (fp16) | ~1.1 GB |
| bge-reranker-v2-m3 (fp16) | ~1.1 GB |
| **Total** | **~7 GB**, leaving room for longer contexts or a 14B model at Q4 |

Measured peak: 6.9 GB VRAM. Host RAM, not VRAM, is the binding limit. Ollama's
model runner was measured leaking about 100 MB of anonymous memory per request,
and under memory pressure it also disables mmap and copies the 4.7 GB model into RAM.
The kernel OOM-killer then took down the notebook. Mitigations, all in the code:
`use_mmap: true` on every request, `OLLAMA_NUM_PARALLEL=1`, retry of 5xx
"runner stopped" errors, and a watchdog thread (`colab.start_runner_watchdog`) that
restarts the runner when its anonymous RSS passes 6.5 GB, but only while no
request is in flight.

## 4. Code map

```
src/local_agentic_rag/
  config.py          RAGConfig (every tunable) + ablation presets
  schema.py          Chunk, ScoredChunk, Answer, Citation, TraceStep, IngestReport
  ingest/pdf.py      PyMuPDF parsing, reading order, headers/footers, OCR, table detection
  ingest/tables.py   whitespace-table reconstruction from word coordinates
  ingest/chunker.py  token-aware, heading-aware, page-bound chunking
  retrieval/         embedder, bm25, index (cache), hybrid (RRF + rerank), reranker
  llm/               LLM interface; Ollama, llama.cpp and scripted (test) backends
  agent/             prompts, citations, calculator tool, the AgenticRAG loop (adaptive / full)
  pipeline.py        RAGPipeline facade: ingest() / ask() / retrieve() / variant()
  eval/              dataset loader, metrics, resumable runner, markdown reports
  bench.py           retrieval + answer ablations (CLI: python -m local_agentic_rag.bench)
  colab.py, app.py   Colab helpers (Ollama setup, HTML answer rendering), Gradio chat UI
```

## 5. Failure handling

* Unreadable / encrypted / empty / image-only PDFs raise `PDFParseError` with a
  message telling the user what to do. One bad page never fails a document.
* Any structured agent step that returns invalid JSON falls back to the simpler
  behaviour (e.g. no planner → search with the raw question).
* LLM server failures surface as `Answer.error` with a friendly message. The
  chat UI and benchmark runner never crash on one bad question.
* The benchmark runner writes JSONL incrementally and resumes after a Colab
  disconnect.
