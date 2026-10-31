# Status and known limitations

Snapshot of 30 Oct 2026: what is implemented, how each part was verified, and what is not done or
not tested. Benchmark numbers are in [RESULTS.md](RESULTS.md) and [SCALE.md](SCALE.md); this page
names the run behind each claim.

## Implemented and how it was verified

### Reading PDFs

| Area | Verified by |
|---|---|
| PDF ingestion with PyMuPDF: column-aware reading order, running headers and footers removed, headings, error handling for unreadable, encrypted, empty and non-PDF files | unit tests with generated PDFs (two-column order, header stripping, blank/non-PDF/missing files); the 3 benchmark PDFs ingest in 4–37 s |
| Ruled tables → Markdown with captions; whitespace statistical tables → labelled rows (`label \| 2025 Q2 = 3.8`) | unit tests incl. spanning headers; BEA tables answered correctly by the 7B model on a T4 |
| Token-aware, heading-aware, page-bound chunking | unit tests (size limits, overlap, sections, no page crossing) |
| OCR of scanned pages with the Tesseract CLI at 300 dpi; the page's script is detected first and the matching pack used (Urdu/Arabic, Bengali/Assamese, Telugu, Tamil, Kannada, Malayalam, Gujarati, Punjabi, Odia, Chinese, Japanese, Russian; Latin and Devanagari keep eng+hin+mar); garbled text layers and legacy Hindi fonts sent to OCR | scanned Hindi document: word recall 0.99–1.00, both scanned-page questions correct (Colab T4, 25 Oct); 14 scanned copies in 14 scripts: 6/14 after the update, 2/14 before (run 9, slowed by OCR timeouts on Kaggle's shared CPU) |
| Streaming, resumable ingestion (two passes, checkpoint after every batch, bounded memory) | 100- to 10,000-page documents on a T4; output identical to the in-memory parser on every benchmark PDF; resume after interruption at every size (scale runs v1 and 4) |
| Crash-isolated parse workers (per-page time limits, restart, empty-page fallback) | a killed worker cost exactly one page at every size; hang and crash unit tests |
| Table detection on the GPU (opt-in, `LOCAL_RAG_ONNX_GPU=1`) | 1,000 table-heavy pages parsed 1.89× faster (224.7 s → 119.0 s) with byte-identical output (run 2); the GPU and CPU paths parsed all 8 test PDFs identically (run 3) |
| Persistent index (fp16 vectors, BM25 sparse matrix on disk) | 188 MB and 0.7 s cold load at 10,000 pages; BM25 0.6 ms vs 27 ms per query at 40k chunks, identical rankings |

### Answering

| Area | Verified by |
|---|---|
| Hybrid retrieval (BGE-M3 + Unicode BM25, reciprocal rank fusion) and cross-encoder rerank (bge-reranker-v2-m3) | retrieval ablation: page recall@5 73.1% (starting code) → 94.2%; reproduced on CPU and T4 |
| Page and section references in questions ("page 4,512", "section 7203") | page questions rank 3–4 instead of 87–895 before reranking. Section numbers on chunks: US Code Title 26 recall@5 0.75 → 0.92, answerable 6/12 → 7/12 (run 2, not held out); held-out US Code Title 21: 8/9 questions naming a section correct (run 3) |
| Adaptive agent (verify, escalate only on refusal or an unsupported claim), full-loop mode for comparison, chat-history condensing, bounded loops, fallbacks on invalid JSON | 33 agent unit tests with a scripted LLM; agent ablation on a T4 ([RESULTS.md section 3](RESULTS.md#3-the-agent-what-the-ablation-taught-us)) |
| Calculator for quantitative questions (fixed operations, operands checked against the cited row and column, safe evaluation) and three numerical fixes (cell alignment, measure-aware retrieval, date candidates) | unit tests incl. 30 grounding tests; numerical questions 2/6 → 5/6 on the English development set (runs 5 and 6) |
| Citations: `[n]` → passage → page, invalid markers dropped, overlap fallback | 34 citation tests; live answers cite the evidence pages |
| "Which table" answers name the table (the first cited table's caption is appended when the answer omits it) | 40 tests; Statistical Abstract "which table" questions 5/6 correct (run 3) |
| Many languages: keyword search for every script, cross-language retrieval, one translated query per document script, the answer written in the question's language (named from the script and word lists) or in the one the user asks for, "not found" in about 30 languages, every script's digits, book names on citations in a multi-book search | `tests/test_languages.py`; 22-language set 81/111 correct, 112/122 in the right script (run 9); sealed held-out set 18/20; Hindi/Marathi set 17/18 (runs 8 and 10) |
| Answer grading of number words ("Seventeen" = 17) | 93 unit tests incl. negatives |
| LLM backends: Ollama (default), llama.cpp (CPU), scripted (tests) | Ollama on T4 runs; llama.cpp loads and runs locally |

### Web app and deployment

| Area | Verified by |
|---|---|
| Document library and web app: upload, background queue, live progress, delete and retry, highlighted source pages | 0 failed requests with 1–8 concurrent users on a T4 (scale run v1, run 1); visual checks in Chrome; races found in review fixed with regression tests |
| The web app on a free Colab T4 (through Colab's port proxy) | 27 Oct: the notebook ran to the web-app cell without errors with GPU table detection on; a 64-page PDF processed in 79 s with live progress; a streamed, then checked answer with page citations in 11 s; the citation opened page 10 with the passage highlighted; a 250-page PDF processed in about 2 min and its page-17 table question answered correctly |
| Streamed answers (text appears while written, then the checked answer replaces it) | first words after 0.47 s median with one user, 0.70 s with two (run 1, T4) |
| Asking while a document is still being processed (answers say "first n of N pages") | askable 206 s after uploading a 1,000-page PDF with 96 pages indexed; the answer was correct and cited (run 1); final index byte-identical with or without partial queries (test) |
| Several users: two answers at a time, then a first-come queue with visible position | load test 1–8 users, about 10.5 answers per minute on one T4 |
| Optional access token (`--token`: Bearer header or HttpOnly session cookie), one upload per client address, optional daily quota | 32 tests; the login form checked in Chrome against the demo server (401 → form → unlocked) |
| Clear user-facing messages (unchecked drafts labelled, "Not checked" answers, partial-index refusals, upload limits shown before sending) | an audit of 25 message cases; 12 fixed with 12 tests |
| Phone layout | checked at 390 and 360 px in an emulated phone browser (Chromium); no horizontal scroll; touch targets enlarged to 40–44 px |
| Docker image and Compose stack | `docker build` passed (6.5 GB, Tesseract eng/hin/mar); `docker compose up` with `docker-compose.cpu.yml` and `scripts/smoke_install.sh` passed with the real 7B model on a CPU host ([DEPLOY.md](DEPLOY.md#tested)) |
| Project notebook `notebooks/Local_Agentic_RAG.ipynb` | executed top to bottom on a Kaggle T4 in 32 min (26 Oct) and 34.5 min with the GPU table-detection cell (27 Oct), no errors; earlier in a fresh free Colab T4 runtime (25 Oct) |
| Gradio chat UI and HTML answer card | launched inline in Colab on a T4; a live chat turn returned a checked, cited answer (`docs/img/chat_ui_colab_t4.png`); follow-up questions rewritten from chat history |
| Evaluation harness: metrics, bootstrap confidence intervals, resumable runner, reports | 118 tests; the benchmark validator reports 0 errors |

### Tests, CI and reviews

* **Unit tests:** 879 passed, `ruff check` clean (29 Oct). The suite runs offline with fake models
  (`pytest -q`).
* **CI:** GitHub Actions runs lint and the offline tests on Python 3.10 and 3.12.
* **Independent code reviews:** the first found 7 edge-case bugs (empty candidate pools,
  sub-query slot eviction, a stale verdict or hint, non-finite calculator results, the cache key, an
  ablation config), all fixed with regression tests. The second (27 Oct) found 10 issues in the
  streaming, partial-index, auth and upload code; 9 were fixed with regression tests (chunk section
  number after an overflow, token and step event order, presupposition refusal flags, no second
  generation after a timeout, a partial-index race and duplicate builds, a quota bypass without
  Content-Length, quota charged for duplicates and rejections, API docs public in token mode). Later
  fixed: an over-quota or over-size upload was reported by some browsers and proxies as a closed
  connection; the page now asks `GET /api/upload-check` first and again after a network error.
* **Docs checked against the evidence** (27 Oct): 29 discrepancies (stale limitations, run dates, a page
  count) fixed.

### Evaluation status

* English and multilingual development sets: current results in [RESULTS.md](RESULTS.md).
* Held-out set 2 is complete: US Code Title 21 and the CAG Hindi report were each used once (run 3,
  27 Oct), the GAO Cost Guide once (run 7, 29 Oct).
* The 46 real-document questions were used twice (26 and 27 Oct); they are not a held-out set.
* No-document control (run 7) and answer-model comparison (run 8) are done.

## Not done or not tested

* **GPU inside Docker.** GPU reservation in `docker-compose.yml` and `--gpus all` were not run (no
  NVIDIA GPU with the Container Toolkit on the test machine; needs a CUDA 13-capable driver, R580+).
* **Security behind a proxy.** No rate limit on token guesses. Behind a reverse proxy all users share one
  address, so the per-address upload limit and quota apply to everyone together. Cookie behaviour over
  real HTTPS is unit-tested only.
* **Phones.** Not tried on a real iPhone or Safari, or in dark mode; citation chips keep a 22 px visual
  height (with a larger invisible tap area).
* **Library title from PDF metadata.** The 9/11 Commission Report's metadata title is "201-635.job" (a
  print-job name). The title guesser (`ingest/pdf.py`, `_guess_title`) does not treat it as junk, so the
  library shows it. Not fixed yet.
* **Two answers at once inside Ollama** (`OLLAMA_NUM_PARALLEL=2`): the only load test was confounded by a
  second model copy on the same GPU, so the default stays at one slot.
* **Not built:** visual (image-based) retrieval, deliberately deferred: the text pipeline already reaches
  about 95% page recall, and a vision model would roughly double GPU memory and latency. The benchmarks
  have no questions that span several documents.

## Known limitations

1. **The development benchmark is small** (61 English, 20 Hindi/Marathi and 122 many-language questions, single annotator),
   and it was used for tuning. Grading uses alias matching, which can under-credit paraphrased numbers.
   Confidence intervals are reported for that reason.
2. **Multi-hop questions are the weakest category** (5/9). Still failing on the English set: BEA q09 (the
   dollar-level row ranks too low), BEA q06/q11/q13/q15 (a renamed row, gross output vs value added, a table
   whose caption sits in a separate chunk), arXiv q14 and NIST q13.
3. **The numerical fixes were tuned on the English development set.** They showed no measurable gain on
   synthetic documents they were not tuned on (run 7: 100 pages 27/36 → 26/36, 250 pages 26/36 → 26/36).
4. **Questions that presuppose something absent** from the document (for example an act passed after it
   was written) are sometimes answered from the nearest related passage: 6/9 unanswerable real-document
   questions were refused. An optional presupposition check brought no gain and raised false refusals
   (1.9% → 3.8%), so it is off.
5. **Long legal codes:** after the section-number fix, 5 of 12 US Code Title 26 questions are still wrong
   (numeric, multi-hop, a definition).
6. **Agent loop vs single pass:** accuracy gains are within the 95% interval on this benchmark. The default
   adaptive mode buys fewer false refusals and better citations for about 2.5× the latency;
   `preset("hybrid_rerank")` is the fast option.
7. **Ingestion of table-heavy PDFs is slow on 2 CPUs:** about 2.3–2.9 pages/s (10,000 pages ≈ 71 min;
   37 min with the opt-in GPU table detection, at up to 9.4 GB peak RAM; the 1,224-page Budget Appendix
   ≈ 34 min). PyMuPDF's ONNX table model is about 80% of it. A faster line-only detector invents tables
   in academic papers, so it is not used. Table detection on two-column papers sometimes merges a table
   with neighbouring equation text (arXiv p.6); the values survive, but the structure is noisy.
8. **Scanned PDFs** need Tesseract (`apt-get install tesseract-ocr`; the notebook and Docker image install
   it, with the Indian-language packs and script detection). OCR quality was measured on one Hindi scan;
   the 14-script scanned set (run 9) was slowed by OCR timeouts on a shared CPU, and the fix that makes the
   fallback OCR reuse the detected script is not measured on a GPU yet. Quality depends on the scan.
9. **Languages:** 22 document languages and 24 question languages measured (run 9), a few questions each.
   Weakest: a question in a different non-English language from the book (10/21) and questions that need
   two books in different languages (1/6); answers in low-resource languages are bounded by Qwen2.5-7B
   (one Telugu→Hindi answer translated "feet" as "fingers"). Inside a shared script the language is
   guessed from short word lists, so a very short question gets the generic "answer in the question's
   language" instruction. Arabic mixed with digits, and Urdu set in a Nastaliq font, can come out of a
   PDF's text layer out of order or garbled.
10. **Test documents:** the synthetic scale corpus is template prose; real long documents were four US
    federal publications plus the held-out set.
11. **PyMuPDF is licensed AGPL-3.0**, which applies to anyone who redistributes this project.

## Development history

### Validation round on Kaggle (26 Oct)

Colab's free GPU quota was used up, so this round ran on a Kaggle T4. Each change was measured on the GPU
before being kept ([RESULTS.md section 10](RESULTS.md#round-by-round-tuning-on-kaggle-26-oct) has the
round-by-round table and the noise check).

| Change | Evidence it was needed | Measured (Kaggle T4) |
|---|---|---|
| Calculator operands grounded in the cited passage (row and column), stated changes skip the calculator, ID ranges counted deterministically, one retry for empty calculations | wrong cells (2.61/−3.52 for the GDP swing), a correct "+$6.8 billion" overridden, 8 of 9 actions counted | English 73.1% → 75.0% (BEA q13, NIST q10 fixed; NIST q11 regressed) |
| Keep a cited answer when the model appends the "not found" marker | a correct, cited Hindi answer followed by the marker was scored as a refusal | multilingual 83.3% → 94.4% (Hinglish two-part and scanned-Hindi questions) |
| Table captions attached to Markdown tables; table chunks no longer inherit the previous table's title | arXiv Table 2 (NQ) indistinguishable from the look-alike Table A.3; BEA Table 12 rows labelled "Table 10" | English 75.0% → 76.9%, table questions 9/12 → 10/12, page recall@5 94.2% → 96.2%; BEA q09 regressed |
| Runner watchdog scoped to its own Ollama server | two servers (one per GPU) could restart each other's runner | parallel runs on 2 GPUs completed with no errors |
| Notebook built into a temp dir by the tests, with stable cell ids | running the tests rewrote the committed notebook | rebuilding is byte-identical; a test catches a stale notebook |

Result (notebook, top to bottom): English agent 76.9% [63.5–88.5] (single pass 71.2%), 9/9 unanswerable
refused, citation precision 78.4%; multilingual 88.9–94.4% across two runs; GPU memory 8.9 of 15 GB.
The numerical fixes of 27 Oct later raised English to 84.6%.

### Improvement pass (25 Oct evening, Colab T4)

Driven by the failure analysis of the previous full run (`scripts/failure_report.py`, also a notebook
cell). Before/after numbers are in [RESULTS.md section 10](RESULTS.md#improvement-pass-on-a-fresh-colab-t4-25-oct-evening-notebook-run-all)
and [section 8](RESULTS.md#improvement-pass-on-a-fresh-colab-t4-25-oct-same-20-questions); each fix has
regression tests.

| Change | Evidence it was needed | Measured after (fresh T4 run) |
|---|---|---|
| OCR via the Tesseract CLI at 300 dpi | scanned-page question failed; OCR text lost ~30% of Hindi words | word recall 0.70 → 0.99/1.00; scanned questions 1/2 → 2/2 |
| Refusal detection (cited refusals, Hindi/Marathi phrases) | an unanswerable question was "answered" with a cited "the document does not say" | unanswerable refused 8/9 → 9/9, false refusals unchanged |
| Calculator with fixed operations; result always stated | returned nothing for 16 × 16; model ignored its result | numerical 1/6 → 2/6 (one question); still picks wrong cells or the wrong operation |
| Table headers / stacked rows | wrong values read from two BEA tables | values correct on the inspected tables; **retrieval recall@5 −1 question on BEA** (94.2% → 92.3%) |
| Parse each PDF once per benchmark run | every ablation variant re-parsed | benchmark wall time 31 → 27 min |
| Grader alias fix (multilingual q04) | correct "over a thousand" answer scored wrong | disclosed in `benchmark/multilingual/README.md` |
| **Reverted:** stricter "refuse unless the exact fact is present" prompt rule | – | raised single-pass false refusals 3.8% → 21.2%; removed, a test keeps it out |

Net: English answer accuracy unchanged (73.1%), multilingual 88.9% (equal to the re-graded before score),
refusals and OCR improved, citation precision −1 to −5 points.

### Multilingual support (added 25 Oct)

Two bugs were found by running in Colab and fixed with regression tests: an English question got a Hindi
answer because the answer-language instruction was too vague, and English questions were classified as
Hinglish because "the" was in the Hinglish word list. Before the change, the keyword index dropped every
non-ASCII character.

## Starting code

The "starting code" numbers in RESULTS come from the original prototype this project
began from. That prototype is not part of this repository. The current design is
described in [ARCHITECTURE.md](ARCHITECTURE.md).
