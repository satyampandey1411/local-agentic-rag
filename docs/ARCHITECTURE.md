# Architecture

A local chatbot that answers questions about any uploaded PDF with
page-level citations, using only open-weight models that fit on a free Google
Colab T4 GPU (15 GB VRAM).

## 1. End-to-end flow

```
                      INGEST (once per PDF, streaming, resumable; see section 1b)
 ┌──────────┐   ┌──────────────────────────────┐   ┌───────────────────────┐   ┌─────────────────────┐
 │ PDF file │──►│ PyMuPDF parse (worker procs) │──►│ Structure-aware chunks │──►│ Index on disk        │
 └──────────┘   │ • column-aware reading order │   │ • ≤320 embedder tokens │   │ • BGE-M3 fp16 vectors│
                │ • ruled tables → Markdown    │   │ • split at headings,   │   │ • BM25 sparse matrix │
                │ • whitespace tables → rows   │   │   paragraphs, sentences│   │ • chunks + metadata  │
                │   "label | 2025 Q2 = 3.8"    │   │ • never cross pages    │   │ • checkpoint state   │
                │ • strip running headers      │   │ • "title > section"    │   └─────────────────────┘
                │ • OCR image-only / garbled   │   │   header for embedding │
                └──────────────────────────────┘   └───────────────────────┘

                      ASK (per question) — default "adaptive" agent mode
 question + chat history
        │
        ├─ history present? ──► PLAN: rewrite follow-up as a standalone question        (1 LLM call, JSON)
        ▼
 ┌─────────────┐  dense top-30 + BM25 top-30 → Reciprocal Rank Fusion → top-24
 │  RETRIEVE   │  → cross-encoder rerank against the question → top-6               (0 LLM calls)
 └─────┬───────┘  ("page 512" in the question: that page's passages join and keep 3 slots)
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

Cost, measured in the 27 Oct regression run (run 3, English set, 61 questions): the median question
used 2 LLM calls and the most was 9 (a numerical question). `agent_mode="full"` keeps the
original always-on loop (plan → retrieve⇄grade → generate⇄verify) for comparison.

**Why adaptive, which we measured.** The always-on loop (v1) was *worse* than a single pass on
the same questions (23 vs 27 of 38). Ranking by the max reranker score over the planner's
keyword queries let sub-query matches crowd out the passage that answers the actual
question. Letting the grader reorder the context also pushed evidence down. Fixes:
(1) rank by the user's question and let sub-queries only *add* their best passage;
(2) the grader filters but never reorders; (3) spend the loop only on questions whose first
answer is refused or fails verification. See `docs/RESULTS.md`.

## 1b. Ingesting very long PDFs (100 to 10,000+ pages)

The first design parsed the whole PDF in memory, embedded every chunk in one call and saved
the index at the end. That works for 60 pages and fails at 10,000: memory grows with the
document, one malformed page could crash the process, and a Colab disconnect after an hour
lost everything. Ingestion is now a two-pass stream with checkpoints
(`ingest/streaming.py`, `ingest/workers.py`):

```
 PDF ─► pass 1: READ (parallel)                         pass 2: INDEX (in page order)
        worker processes, 16-page batches        ┌────► strip running headers/footers
        tables, OCR, text blocks per page        │      (needs the document-wide counts)
        ──► parsed/<sha>/raw.jsonl  ──── stats ──┘      chunk (section carried across pages)
            checkpoint after every batch                embed 48-page batches on the GPU
                                                        ──► chunks.jsonl + vectors.f16
                                                            checkpoint after every batch
                                                  final: BM25 sparse matrix, meta.json
```

* **Why two passes.** Removing running headers needs to know which margin lines repeat on
  40% of pages (or on 3+ consecutive pages, for per-chapter headers), and heading detection
  needs the body font size. Both are document-wide counts, collected while pass 1 streams.
  The output is identical to the in-memory parser; this was checked chunk by chunk on every
  benchmark PDF.
* **Bounded memory.** Pass 1 keeps at most two batches per worker in flight; pass 2 holds one
  48-page batch. Vectors go to disk as float16 as they are produced.
* **Resumable.** Each pass appends to its files, fsyncs, then atomically replaces a
  `state.json` checkpoint. A crash, a cancelled job or a closed notebook resumes from the last
  checkpoint; files are truncated to the checkpointed byte offsets, so a half-written batch
  is discarded. Tested by killing jobs mid-way and comparing with a clean run.
* **Crash-isolated parsing.** Pages are parsed by `python -m local_agentic_rag.ingest.workers`
  subprocesses speaking JSON lines, one reply per page. A page that crashes MuPDF or hangs
  past its time budget kills only its worker: that page becomes an empty page with a warning,
  the worker restarts and the batch continues. (`multiprocessing` was rejected: `spawn`
  re-runs the caller's script in every worker, `fork` copies CUDA and server threads.)
* **Shared parse.** Parsed pages are keyed by the PDF hash and the parse settings, so an
  ablation that only changes the embedder or chunk size re-indexes without re-parsing.
* **Search at scale.** The vectors stay a flat matrix: about 40,000 chunks × 1,024 dims is
  160 MB in float32 and one exact dot product per query. FAISS would only add approximation:
  only its flat indexes are exact, and IVF needs more training vectors than one document
  has. BM25 is a precomputed CSC sparse matrix (a query sums a few columns), loaded from
  disk instead of re-tokenising.
* **Pages named in the question.** "What does page 4,512 say about…": page numbers are
  stripped from the text as running footers, so neither retriever could find that page. The
  question's page references (`page 12`, `pp. 4-6`, `पृष्ठ 7`) pull that page's passages into
  the candidate pool. The best three by reranker score keep their slots.

## 1c. The document library and web app

```
 browser ──HTTP──► FastAPI (server/app.py) ──► Library (library.py) ──► streaming ingest ─► store/
   upload (streamed to disk, %PDF check, size limit)      catalog.sqlite (status, progress, errors)
   SSE progress, page images with the cited passage highlighted, ask (+ live agent steps)
```

* One background ingestion thread. The GPU embeds one document at a time, and the parse
  workers already use every core. Documents left queued or processing when the server
  stopped are re-queued at start-up and resume from their checkpoint.
* Questions are answered while a long document is ingesting. The embedder and reranker are
  shared behind one lock, held for a single encode/score call, so a question waits at most
  one 48-page batch, never the whole document.
* At most `--max-concurrent-asks` answers run at once (default 2); further users wait in a
  first-come queue and see their position. A timed-out answer returns 504 to the client,
  while its GPU slot stays held until the work really ends.
* Answer text streams while it is written. `LLM.chat_stream()` (real `"stream": true` in the
  Ollama backend; other backends fall back to one chunk) feeds `AgenticRAG.run(on_token=…)`,
  which streams only the free-text answer generations (never the JSON plan/grade/verify
  calls) and sends `{"reset": true}` at the start of each one, so a regenerated answer
  replaces the first. The server batches pieces (at most one per ~50 ms, or 200 characters)
  into SSE `token` events between `started` and `answer`. Citations, verification and the
  final text still come only from the `answer` event, which replaces the streamed text.
  `benchmark/scale/load_test.py` reports `first_token_p50/p95`.
* A document that is still being indexed becomes askable at its first pass-2 checkpoint.
  `load_partial_index()` reads only the byte ranges of `chunks.jsonl` and `vectors.f16` named
  by the atomically replaced `state.json` (data is fsynced before the checkpoint names it),
  builds BM25 in memory, and never writes. The library caches it apart from the LRU of final
  indexes and rebuilds it at most every 5 s; answers carry `partial` (pages indexed / total).
* "Which table/figure…" questions: after the agent finishes, if the answer names no table
  identifier, the caption line ("Table 12.4: …") of the first cited table chunk is appended
  with that chunk's citation number. Deterministic, no prompt change, never on refusals.
* Optional access token (`--token`): every `/api/*` route except `/api/health` needs a Bearer
  header or an HttpOnly session cookie set by `POST /api/session` (EventSource and `<img>`
  cannot send headers). One upload per client IP at a time; optional daily byte quota.
* A citation chip opens the page rendered by PyMuPDF with the cited passage highlighted (the
  passage is cut into sentence pieces, or table body cells, and searched on the page).
* Uploads, indexes and the catalog live under one directory (`--library`). Deleting a
  document removes its file, parsed pages and index.

## 2. Why each component exists (and what it replaced)

| Stage | Choice | Problem it fixes (evidence) |
|---|---|---|
| Parsing | PyMuPDF | pdfminer (used by the starting code) is slow and loses layout. Reference `unstructured hi_res` downloads layout models and takes minutes per PDF. PyMuPDF parses a 64-page PDF in ~4 s. |
| Reading order | column classification + bands | Naive y-then-x sorting interleaves the two columns of an academic paper line by line. |
| Tables | `find_tables` → Markdown with its caption as the first line; whitespace tables rebuilt from word coordinates | On BEA-style statistical tables `find_tables` stacks whole columns into one cell (`2.5<br>2.9<br>…`). Row serialisation with header labels lets a 7B model read the right cell. Captions let it tell look-alike tables apart (arXiv Table 2 vs Table A.3). |
| Chunking | ≤320 *embedder* tokens, heading-aware, page-bound | The starting code's 2048-char chunks were silently truncated to 256 tokens by MiniLM. That is roughly half of every chunk invisible to retrieval. |
| Embedder | BAAI/bge-m3 (default; earlier BAAI/bge-base-en-v1.5 with a query instruction, still the `hybrid_rerank` preset) | MiniLM is weak on long, technical passages. BGE-M3 keeps English quality and adds 100+ languages in one vector space (see 2b). |
| Keyword search | BM25, decimal/code-aware, Unicode-aware tokenizer | Dense vectors blur exact numbers, acronyms and table labels. |
| Fusion | Reciprocal Rank Fusion | The reference's min-max + 0.7/0.3 weighting always rescales the best hit to 1.0, even when nothing matches. RRF uses ranks only and needs no tuning. |
| Reranker | BAAI/bge-reranker-v2-m3 cross-encoder (default; earlier BAAI/bge-reranker-base) | Bi-encoder similarity is coarse. The cross-encoder reads question and passage together. The multilingual v2-m3 keeps English quality and ranks Hindi and Marathi passages better. |
| Agent | bounded Python state machine | Reference grades passages one LLM call at a time (6 calls/round). We grade in one batched call and add planning, verification and chat memory. No LangGraph dependency for 5 nodes. |
| Citations | numbered passages → `[n]` → chunk → page; invalid markers dropped; overlap fallback | Reference never gives page numbers to the LLM, so it cannot cite. |
| Refusal | `ANSWER_NOT_FOUND` sentinel + uncited "document does not say" detection | Unanswerable questions must not get confident fabricated answers. |
| Calculator tool | LLM picks a **structured operation** (difference, change, sum, product, ratio, percent_change, count, months_between) and copies the operands with citations → each operand is checked against the cited passage (row and column) → Python computes it. Ungrounded extractions are retried once with the reasons, then dropped; stated changes skip the calculator; ID ranges are counted deterministically. If the answer omits a checked result, a cited "(Calculated: … = x)" sentence is appended | Every configuration scored 0% on numerical questions: evidence retrieved, arithmetic and counting wrong (222.8 vs 443.6; 8 vs 9 items). Free-form expressions often failed to parse, and the generator sometimes ignored the result. |
| Numerical fixes | on by default (`config.py`): `calc_cell_alignment` (an operand copied from the wrong table cell is replaced by the cell its label names), `measure_aware_retrieval` (tables of the wrong measure move down the ranking), `calc_date_candidates` (time-span questions see the dates in the passages); `calc_measure_check` stays off | The six numerical questions failed on the wrong measure or the wrong cell, never on arithmetic. With them on, 5/6 numerical questions on the English development set (2/6 with them off; RESULTS.md section 6). |
| LLM serving | Ollama (prebuilt CUDA, JSON-schema output) | llama-cpp-python CUDA wheels for current Colab are unreliable, and source builds take 15+ min. Ollama installs in about a minute. |

## 2b. Multilingual design

| Concern | What the code does | Where |
|---|---|---|
| Cross-language retrieval | **BGE-M3** embedder (100+ languages, one shared vector space) + **bge-reranker-v2-m3** cross-encoder | `config.py` defaults |
| Keyword search in any script | Unicode tokenizer that keeps vowel signs, Arabic harakat and zero-width joiners inside words (a plain `\w+` splits "निर्माण" at every matra) and ends a word at the danda (।); Chinese, Japanese, Thai, Lao, Khmer and Burmese, written without spaces, are indexed as character bigrams (plus single ideographs); NFKC folds Arabic presentation forms; the ASCII path keeps decimals/codes | `retrieval/bm25.py` |
| Passages in any script | Sentences also end at ।, ॥, ۔, ؟ and 。！？; a run of text with no spaces that is longer than a passage is cut by characters | `ingest/chunker.py` |
| Question ≠ document language, Hinglish | Script detection + Hinglish (and romanised Tamil, Bengali, Telugu) word lists; if they differ, **one LLM call translates the question into the document's language** (given a text sample). With several documents in different scripts, a cheap fused search with the untranslated question first ranks the documents, and the question is translated only into the scripts of the top 3 (at most 3 calls). Passages are ranked by the best of the original and the translations | `lang.py`, `agent._lead_queries`, `agent._likely_indexes`, `HybridRetriever(n_primary=...)` |
| Answer in another language than the passages | When the user's language is not English and differs from the language of the top passages (one script among them), the answer is **written in the passages' language first**, checked there by the verifier, and **translated in a second call** that keeps the [n] markers; if the translation fails, the checked answer is kept | `agent._pivot_language`, `agent._translate_answer`, `RAGConfig.pivot_translation` |
| Two books in one answer | When several documents are searched, a document whose best passage clears a reranker probability of 0.3 gets one of the passage slots, replacing the weakest passage of a document that has several | `HybridRetriever._make_room_for_other_documents` |
| Answer language | The web app's **"Answer in" menu** (or `answer_language` in `POST /api/ask`) overrides everything. Otherwise the question's language is named for the generator: from the script where one script means one language (Telugu, Tamil, Gujarati, ...), from small function-word lists inside shared scripts (Hindi / Marathi / Nepali / Sanskrit / Bhojpuri / Awadhi / Maithili in Devanagari; Urdu / Arabic / Persian; English / French / Spanish / German / ... / Hinglish in Latin). If the words do not settle it, the model is told to answer in the question's own language and, for Devanagari, not to fall back to Hindi. An explicit request ("Answer in Urdu", "उर्दू में बताइए", "హిందీలో చెప్పండి") overrides it | `lang.choose_answer_language`, `agent/prompts.py` |
| Refusals | "Not found" in the answer language (about 30 languages, each with the English sentence after it) | `lang.refusal_message` |
| Scanned PDFs | **Tesseract CLI** on a 300-dpi render. With `ocr_language="auto"` (default) Tesseract's script detection (OSD) runs first and picks the installed packs for that script (Urdu/Arabic, Bengali/Assamese, Telugu, Tamil, ...); Latin, Devanagari and undetected pages use `eng+hin+mar`, the setting the scanned-Hindi results were measured with. Falls back to `eng`, then to PyMuPDF's OCR layer. Pages drawn with legacy Hindi fonts (Kruti Dev, DevLys, Chanakya, Shusha), whose text layer is ASCII look-alikes, are OCR'd too. A Tesseract run that overruns `LOCAL_RAG_OCR_TIMEOUT` (100 s) is retried once with the main pack only at 200 dpi; the fallback OCR uses the same detected packs | `ingest/pdf.py` |
| Page references | "page 5" in English, Hindi, Marathi, Urdu, Arabic, Bengali, Punjabi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Spanish, German, Russian and Japanese, with any script's digits | `retrieval/hybrid.py` |
| Grading and arithmetic | Every script's decimal digits (१६३२, ১৬৩২, ۱۶۳۲, ౧౬౩౨, ...) normalised to ASCII | `lang.DIGITS_TO_ASCII` |

Limits (not claimed): telling languages apart inside one script uses short word lists, so a very
short question may get the generic "answer in the question's language" instruction; generation quality
in lower-resource languages is bounded by Qwen2.5-7B; OCR quality depends on scan quality and on which
Tesseract packs are installed; Urdu set in a Nastaliq font and Arabic mixed with digits can come out
of a PDF's text layer garbled or out of order (seen while building the test PDFs).

## 3. Memory budget on Colab (T4 15 GB VRAM, 12.7 GB host RAM)

| Component | Approx. VRAM |
|---|---|
| Qwen2.5-7B-Instruct Q4_K_M + 8k KV cache (Ollama) | ~5.5–6 GB |
| BGE-M3 embedder (fp16) | ~1.1 GB |
| bge-reranker-v2-m3 (fp16) | ~1.1 GB |
| **Total** | **~7 GB**, leaving room for longer contexts or a 14B model at Q4 |

Measured peak on Colab (25 Oct): 6.9 GB VRAM; with all three models loaded in the final notebook run, 9,069 of 15,360 MiB in use. Long documents do not change this: the embedder and reranker used 2.4 GB of CUDA memory at every size from 100 to 10,000 pages, and a 10,000-page index is 188 MB on disk (SCALE.md). Host RAM, not VRAM, is the binding limit. Ollama's
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
  ingest/workers.py  crash-isolated parse worker processes (JSON lines, per-page timeouts)
  ingest/streaming.py two-pass, checkpointed, resumable ingestion → on-disk index
  ingest/tables.py   whitespace-table reconstruction from word coordinates
  ingest/chunker.py  token-aware, heading-aware, page-bound chunking (PageChunker)
  retrieval/         embedder, bm25, index (cache), hybrid (RRF + rerank), reranker
  llm/               LLM interface; Ollama, llama.cpp and scripted (test) backends
  agent/             prompts, citations, calculator tool, the AgenticRAG loop (adaptive / full)
  pipeline.py        RAGPipeline facade: ingest() / ask() / retrieve() / variant()
  eval/              dataset loader, metrics, resumable runner, markdown reports
  bench.py           retrieval + answer ablations (CLI: python -m local_agentic_rag.bench)
  colab.py, app.py   Colab helpers (Ollama setup, HTML answer rendering), Gradio chat UI
  library.py         document library: catalog, background ingestion queue, ask, page images
  library_types.py   the contract between the library and the web server
  server/            FastAPI app + offline single-page frontend (python -m local_agentic_rag.server)
```

## 5. Failure handling

* Unreadable / encrypted / empty / image-only PDFs raise `PDFParseError` with a
  message telling the user what to do. One bad page never fails a document: a page that
  raises, crashes its parse worker or exceeds its time budget becomes an empty page with a
  warning in the ingest report.
* Interrupted ingestion (crash, cancel, restart, disconnect) resumes from its last
  checkpoint; see section 1b.
* Any structured agent step that returns invalid JSON falls back to the simpler
  behaviour (e.g. no planner → search with the raw question).
* LLM server failures surface as `Answer.error` with a friendly message. The
  chat UI and benchmark runner never crash on one bad question.
* The benchmark runner writes JSONL incrementally and resumes after a Colab
  disconnect.
