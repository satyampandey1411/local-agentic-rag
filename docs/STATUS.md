# Project status — 2026-10-27 (final sprint)

Honest snapshot of what is implemented, what is verified and how, and what is
still outstanding.

## Final sprint progress log (26–31 Oct)

Legend: ✅ verified (tests or a measured run named in the row) · 🔧 in progress · ⏳ planned · ⛔ blocked.
Nothing in this table is a benchmark result unless a run is named.

| Item | State | Evidence / next action |
|---|---|---|
| F1 token streaming (backend + web UI) | ✅ measured (run 1) | merged 26 Oct: Ollama NDJSON streaming, agent `on_token`, SSE `token` events batched every 50 ms, UI renders text then the verified answer; a broken stream falls back to the plain call. Tests pass; demo server sends queued → started → step → token ×18 → answer. Run 1 (T4): first words after 0.47 s median with one user, 0.70 s with two |
| F2 faster table-heavy ingestion | ✅ measured | prose-page triage could not help (0 of 1,000 pages have lines without a table). Layout model on the GPU (opt-in `LOCAL_RAG_ONNX_GPU=1` + `onnxruntime-gpu[cuda,cudnn]`): run 2 (T4) parse of 1,000 table-heavy pages 224.7 s → 119.0 s (1.89×), byte-identical pages, ~0.7 GB GPU per worker. Notebook gets an opt-in install cell |
| F5 Docker / one-command install, DEPLOY.md | ✅ built and smoke-tested on CPU / ⛔ GPU-in-Docker untested | 27 Oct: `docker build` passed (6.5 GB, torch 2.14.1 CUDA 13.0 build, Tesseract eng/hin/mar); Compose stack up on a laptop without an NVIDIA GPU via the new `docker-compose.cpu.yml` (Ollama healthy, 7B model pulled, app healthy); `scripts/smoke_install.sh` passed with the real model: "417 megawatts", 1 citation, page 1 (first run 8 min incl. model downloads, warm 40 s). Dockerfile now installs dependencies in their own layer, so a code change rebuilds in seconds-to-minutes instead of re-downloading ~6 GB. Not run: GPU reservation / `--gpus all` (needs a GPU host with the NVIDIA Container Toolkit and a CUDA 13-capable driver, R580+) |
| Grader: number words ("Seventeen" = 17) | ✅ | 93 unit tests incl. negatives; old grading tests unchanged |
| F9 CI workflow | ✅ | first run on GitHub after the 27 Oct push (commit e7abee2): lint and offline tests passed on Python 3.10 and 3.12 |
| heldout2 evaluation set | ✅ used once (Title 21, CAG); GAO not run | 70 questions written independently of the system (authors read only the official sources): US Code Title 21 (30, from the HTML edition), GAO Cost Guide (22), CAG Hindi GST audit report (18, written in Hindi/Hinglish/English from the English edition; scored on answers only; its 4 unanswerable questions were checked against the executive summary only). PDFs are downloaded only on Kaggle (approved 26 Oct); the PDF copies the web reader cached on the laptop were deleted. Used once, in the final run (27 Oct): Title 21 19/25 answerable, 4/5 refused, recall@5 0.917; CAG 9/14 answerable, 2/4 unanswerable refused. GAO Cost Guide: **not run**, gao.gov answered HTTP 403 to the download from Kaggle, so its 22 questions remain unused |
| F3 ask while ingesting | ✅ measured (run 1) | merged 26 Oct: a processing document is askable after its first indexed batch; answers say "first n of N pages"; final index byte-identical with or without partial queries (test). Run 1 (T4): askable 206 s after upload of a 1,000-page PDF (96 pages indexed); answer correct and cited |
| F7 token auth, upload limits | ✅ | merged 26 Oct: `--token` (Bearer or HttpOnly session cookie), one upload per client IP, optional daily quota; 32 tests; login form checked in Chrome against the demo server (401 → form → unlocked). No rate limit on token guesses; behind a proxy all users share one IP |
| F4 US Code section questions | ✅ measured (not held out) | cause: bold section heading directly followed by a bold sub-heading, chunks kept only the latest; fix: chunks carry the last section number (text unchanged). Run 2 (T4): sections identified 355 → 1,877; US Code recall@5 0.75 → 0.92, answerable 6/12 → 7/12, §164/§6501/§7203 now correct. Held-out check on Title 21 (final run, 27 Oct): 19/25 answerable, 4/5 refused, recall@5 0.917; the 9 questions naming a section 8/9 correct |
| GPU run 1 (regression + tiers 100/1,000 + streaming load test + partial ask + F4 diagnosis + layout model on GPU) | ✅ done | 26 Oct (~22:45–23:35 IST), 50 min on T4 ×2: English 76.9% (same 12 failures), multilingual 94.4%, tiers 100/1,000 unchanged with recovery passing, partial ask correct at 96/1,000 pages, first words 0.5 s (1 user), 0 failed requests up to 8 users; layout-on-GPU inconclusive (CUDA 13 libs missing). SCALE.md "Sprint run 1" |
| GPU run 2 (F11, F4, layout model on GPU, two Ollama slots) | ✅ | two-slot load test confounded (a second model copy on the same GPU); default kept. 26–27 Oct, T4 ×2, notebook `notebook4b1ff5d5fe`; results in SCALE.md "Sprint run 2" |
| GPU run 3 — FINAL (real documents, final use of the 46 questions; held-out set 2, single use; regression; six tiers) | ✅ done (tiers failed to start) | 27 Oct (~00:35–02:30 IST), notebook `notebookcd9bf75c36`, T4 ×2, snapshot run3 (base 2623aa1 + diff 433e8647; same behaviour as the pushed commits except 9 later review fixes that change no chunk text). Identity check passed: parsing with table detection on the GPU gave byte-identical pages to the CPU on all 8 PDFs (3 benchmark, 4 multilingual incl. the scanned one, scale-100), so the run used the GPU; one document at a time on GPU 0. Real documents: 30/37 answerable correct (26/37 before the sprint fixes), 6/9 unanswerable refused (unchanged); 9/11 Report 12/12, Statistical Abstract 11/13, US Code Title 26 7/12; Statistical Abstract parse 521 s (earlier 17.5 min on CPU with two documents at once). Held-out set 2: Title 21 19/25, 4/5 refused, recall@5 0.917; CAG Hindi 9/14 answers only; GAO not run. Regression: English 76.9% (same 12 failures), multilingual 94.4%. **The six synthetic tiers did not run**: the scale job failed at start because two jobs downloaded the model into the same Ollama directory at the same moment (`ollama pull ... failed: remove /root/.ollama/...`); re-run is GPU run 4. SCALE.md "Final run" |
| GPU run 4 (six tiers, scale only) | ✅ done | 27 Oct (~02:50–05:55 IST), notebook `notebookf8de2130c7`, T4 ×2 (tiers on GPU 1), snapshot run4 (base e7abee2 + diff fce5cf45: the model-download retry), table detection on the GPU, scale job alone; exit 0 after 10,916 s. All six tiers passed: chunks, recall@5 (0.972 → 0.889), MRR, answer accuracy (0.69–0.75) and refusals identical to run v1; citation precision identical except 100 pages (0.917 vs 0.931). Ingest 10,000 pages 2,214 s = 36.9 min (run v2, CPU table detection, also alone: 4,237 s); 4.3–4.6 pages/s. Recovery passed at every tier (resumed 10,000-page job 2,209 s in total). Peak RAM 5.6–9.4 GB (run v1: 3.3–7.1 GB). Numbers read from the run log (results archive not downloaded). SCALE.md "Six synthetic tiers (run 4, final code)" |
| Project notebook top to bottom on a T4 (incl. the GPU table-detection cell) | ✅ | 27 Oct, Kaggle T4 (one card), code snapshot run3: completed without errors in 34.5 min; the new cell switched table detection to the GPU; English benchmark 12 of 61 failed (76.9%, same as before); GPU memory 9,069 of 15,360 MB |
| Independent code review of the sprint diff | ✅ | 10 findings (27 Oct). 9 fixed with regression tests after the final-run snapshot was taken: chunk section number after an overflow, token/step event order, presupposition refusal flags, no second generation after a timeout, partial-index race and duplicate builds, quota bypass without Content-Length, quota charged for duplicates/rejections, API docs public in token mode. Chunk text identical on the benchmark PDFs. Later fixed: an over-quota or over-limit upload was reported by some browsers/proxies as a closed connection (413 sent before the body was read); the page now asks `GET /api/upload-check` first and re-asks after a network error |
| Docs audit against evidence | ✅ | 27 Oct: 29 discrepancies found (stale limitations, run dates, a page count, authorship wording) and fixed in README, RESULTS, SCALE, STATUS, DEPLOY and the report; report rebuilt (5 pages) |
| Demo readiness | 🔧 | demo questions verified (USS Cole 17; page 4,512 = Table 346.13; Veloria 2031 unanswerable; live upload scale-250). **Colab rehearsal of the web app (27 Oct, free Colab T4):** notebook ran to section 7b without errors, GPU table detection active, the app opened through Colab's port proxy; NIST paper (64 pages) processed in 79 s with live progress; streamed answer then verified answer with page citations (11 s); citation opened page 10 with the passage highlighted; 250-page synthetic PDF processed in about 2 min and the page-17 table question answered correctly (Table 1.9). The 250-page PDF finishes too fast to show asking while processing: use a 1,000-page PDF for that step. Screens saved locally; a screen recording of the full script is still to be made by the presenter |
| Model download race (cause of run 3's scale failure) | ✅ | `OllamaLLM.ensure_ready` now waits and re-checks after a failed pull, then retries once (two servers sharing one model directory); 2 tests. Run kits start the second job only after the first model is ready |
| Phone layout at 390 px (and 360 px) | ✅ checked, fixed | 27 Oct: Playwright Chromium emulating a 390×844 phone (device scale 3, touch) against the demo server: 22 screens incl. upload, progress, failed + retry, streamed and final answer, unverified, error, slow + stop, citation → highlighted source page, token form. No horizontal scroll at 390 or 360 px. Fixed (CSS only, phone widths only): buttons, tabs, checkbox, send, pager and summaries were 20–38 px tall → 40–44 px; input text 15 → 16 px so iOS does not zoom on focus. Desktop 1280 px unchanged. Not checked: a real iPhone/Safari, dark mode; citation chips keep a 22 px visual height (larger invisible hit area) |
| User-message audit | ✅ fixed 12 of 25 cases | 27 Oct, `local/audit/MESSAGES_AUDIT.md`. Fixed: streamed text now labelled as an unchecked draft until the checked answer replaces it; the live step says "Checked the answer", not "Verified", when the verdict can be negative; answers with no verifier result get "Not checked" instead of looking verified; refusals during partial indexing say "Not found in the first n pages"; citations matched afterwards by wording are labelled so; the viewer says "highlighted" only when a highlight was found; a PDF that fails at upload no longer toasts "uploaded"; duplicate uploads say so (`already_in_library`); quota messages give real sizes; a deleted document gets its own error title. Over-quota / over-size uploads: the page asks `GET /api/upload-check` before sending and again after a network error, so the reason is shown instead of "connection closed" (reproduced only through a simulated resetting proxy; direct to the server the 413 already arrived). 12 tests; suite 857 passed |
| Unit suite | ✅ | 845 passed, ruff clean (27 Oct); 829 at the final-run snapshot |
| F6 "which table" answers | ✅ measured (final run) | merged 26 Oct: deterministic, no prompt change; appends the first cited table's caption when the answer omits the number; 40 tests. Final run (27 Oct, T4): "which table" questions on the Statistical Abstract 5/6 correct (wrong: statab-q10); English table questions unchanged 10/12 in runs 1–2 |
| F12 source viewer polish | ✅ code | merged 26 Oct: citation scrolls the source pane to the highlight (seen in Chrome), focus styles, live region announces only the final answer, phone close button; 390 px layout checked in an emulated phone browser on 27 Oct (row above) |
| F11 presupposition refusals (flag, off by default) | ✅ measured, not adopted | run 2 (T4): English 76.9% with the same failures but false refusals 1.9% → 3.8%; multilingual unchanged. No benefit, stays off |
| F8 report and docs | 🔧 | report draft rebuilt (5 pages): final framing, users, long-document architecture, 26 Oct scale and real-document tables, limitations; GPU-pending items marked `\pending{}`. `docs/DEMO.md` written. Final-run numbers filled in 27 Oct; six-tier numbers from run 4 filled in 27 Oct |

## Scale round (2026-10-26, evening)

Goal: a product that handles PDFs of 100 to 10,000 pages, for several users, with failure
recovery, measured on the target GPU. Details and every number: [SCALE.md](SCALE.md).

| Area | State | Evidence |
|---|---|---|
| Streaming, resumable ingestion (two passes, checkpoints, bounded memory) | ✅ measured | 100–10,000-page tiers ingested on a T4; output identical to the in-memory parser on every benchmark PDF; resume after interruption at every tier |
| Crash-isolated parse workers (per-page timeouts, restart, empty-page fallback) | ✅ measured | a killed worker cost exactly one page at every tier; hang and crash unit tests |
| Persistent index (fp16 vectors, BM25 sparse matrix on disk) | ✅ measured | 188 MB and 0.7 s cold load at 10,000 pages; BM25 0.6 ms vs 27 ms per query at 40k chunks, identical rankings |
| Document library + web app (upload, background queue, progress, citations with highlighted source pages, delete/retry, several users) | ✅ measured | 0 failures with 1–8 concurrent users on a T4; visual QA in Chrome with fake models; review-found races fixed with regression tests |
| Page and section references in questions ("page 4,512", "section 7203") | ✅ page / ⚠️ section | page questions rank 3–4 instead of 87–895 before reranking; section routing on the US Code improved by the section fix (sprint run 2: recall@5 0.75 → 0.92, answerable 6/12 → 7/12; not held out, 5 of 12 still wrong) |
| Garbled text layers → OCR; per-chapter running headers removed | ✅ unit-tested | no change on the benchmark PDFs except 8 BEA chunks losing a repeated date |
| Independent evaluation: synthetic corpus (240 questions over 6 tiers) and real PDFs (46 questions) | ✅ | written separately by authors who did not see the existing benchmark or its failures |
| Notebook (streaming ingest with progress line, web-app cell) | ✅ | executed top to bottom on a Kaggle T4 (one card) in 32.7 min, no errors |

Test suite: **571 passed at that time** (historical; the current count is 845, see the sprint table above), `ruff check` clean.

## Implemented and verified

| Area | State | Evidence |
|---|---|---|
| PDF ingestion (PyMuPDF): reading order, headers/footers, headings, OCR hook, error handling | ✅ | unit tests with generated PDFs (two-column order, header stripping, blank/non-PDF/missing files); 3 real PDFs ingest in 4–37 s |
| Ruled tables → Markdown (word-based cell text) | ✅ | unit tests; NIST action tables |
| Whitespace statistical tables → labelled rows (`label \| 2025 Q2 = 3.8`) | ✅ | unit tests incl. spanning headers; BEA tables answered correctly by the 7B model on T4 |
| Token-aware, heading-aware, page-bound chunking | ✅ | unit tests (size limits, overlap, sections, no page crossing) |
| Hybrid retrieval (BGE + own BM25, RRF) + cross-encoder rerank | ✅ | retrieval ablation: recall@5 73% (starting code) → 95%; reproduced on CPU and T4 |
| Agent: adaptive (verify, escalate only on refusal/unsupported) and full-loop modes, chat-history condensing, bounded loops, graceful fallbacks | ✅ | 33 agent unit tests with a scripted LLM; live T4 runs exercise every branch; answer ablation on the T4 |
| Citations: `[n]` → chunk → page, invalid markers dropped, overlap fallback | ✅ | 34 citation tests; live answers cite the evidence pages |
| LLM backends: Ollama (Colab), llama.cpp (CPU), scripted (tests) | ✅ | Ollama verified on T4; llama.cpp loads and runs locally |
| Index cache keyed by PDF hash + ingestion config + code version | ✅ | pipeline tests (cache hit) |
| Evaluation harness: metrics, bootstrap CIs, resumable runner, reports | ✅ | 118 tests |
| Benchmark: 61 questions / 3 PDFs, validator | ✅ | `validate.py`: 0 errors |
| Colab helpers (Ollama install/start, GPU/VRAM report, HTML answer card) | ✅ | Ollama setup verified on a real T4 (3 min incl. model pull) |
| Notebook `notebooks/Local_Agentic_RAG.ipynb` | ✅ | 2026-10-26: executed top to bottom on a Kaggle T4 (same GPU, one card) in 32 min with no errors, every benchmark included (RESULTS §0). Earlier: executed top to bottom ("Run all", then "Restart session and run all" after two bug fixes) in a fresh free Colab T4 runtime on 2026-10-25: install, OCR packs, Ollama, ingestion, Q&A, chat UI, retrieval + answer benchmarks, multilingual section |
| Gradio chat UI | ✅ | launched inline in Colab on the T4 (gradio 6.30); a live chat turn returned a verified, cited answer (screenshot: docs/img/chat_ui_colab_t4.png) |
| HTML answer card + follow-up questions | ✅ | rendered in the Colab kernel; follow-up rewritten from chat history; the verifier flagged a contradictory claim with an amber badge |

| Calculator tool for quantitative questions (operand extraction → grounding check → AST-safe evaluation) | ✅ / ⚠️ | unit tests incl. 30 grounding tests; operands must sit on the cited row and column, ungrounded extractions are retried once then dropped; numerical questions are still 2/6 because the 7B model often picks a grounded but wrong cell (RESULTS §0) |
| Independent code review | ✅ | 7 confirmed edge-case bugs (empty pools, sub-query slot eviction, stale verdict/hint, non-finite calculator results, cache key, ablation config) fixed with regression tests |

Test suite: **472 passed at that time** (historical; current count 845), offline, ~15 s (`pytest -q`). `ruff check .` clean.

## Validation round on Kaggle (2026-10-26)

Colab's free GPU quota was used up, so this round ran on a Kaggle T4. Each change was measured on the GPU
before being kept (RESULTS §0 has the round-by-round table and the noise check).

| Change | Evidence it was needed | Measured (Kaggle T4) |
|---|---|---|
| Calculator operands grounded in the cited passage (row and column), stated changes skip the calculator, ID ranges counted deterministically, one retry for empty calculations | wrong cells (2.61/−3.52 for the GDP swing), a correct "+$6.8 billion" overridden, 8 of 9 actions counted | English 73.1% → 75.0% (BEA q13, NIST q10 fixed; NIST q11 regressed) |
| Keep a cited answer when the model appends the "not found" marker | Kaggle trace: a correct, cited Hindi answer followed by the marker was scored as a refusal | multilingual 83.3% → 94.4% (Hinglish two-part and scanned-Hindi questions) |
| Table captions attached to Markdown tables; table chunks no longer inherit the previous table's title | arXiv Table 2 (NQ) indistinguishable from the look-alike Table A.3; BEA Table 12 rows labelled "Table 10" | English 75.0% → 76.9%, table questions 9/12 → 10/12, page recall@5 94.2% → 96.2%; BEA q09 regressed |
| Runner watchdog scoped to its own Ollama server | two servers (one per GPU) could restart each other's runner | parallel runs on 2 GPUs completed with no errors |
| Notebook built into a temp dir by the tests, with stable cell ids | running the tests rewrote the committed notebook | rebuilding is byte-identical; a test catches a stale notebook |

Final (notebook, top to bottom): English agent **76.9%** [63.5–88.5] (single pass 71.2%), 9/9 unanswerable refused,
citation precision 78.4%; multilingual **88.9–94.4%** across two runs on the final code; GPU memory 8.9 of 15 GB.

## Final improvement pass (2026-10-25 evening)

Driven by the failure analysis of the previous full run (`scripts/failure_report.py`, now also a notebook cell).
Before/after numbers are in RESULTS §3 and §5; each fix has regression tests.

| Change | Evidence it was needed | Measured after (fresh T4 run) |
|---|---|---|
| OCR via the Tesseract CLI at 300 dpi | scanned-page question failed; OCR text lost ~30% of Hindi words | word recall 0.70 → 0.99/1.00; scanned questions 1/2 → 2/2 |
| Refusal detection (cited refusals, Hindi/Marathi phrases) | an unanswerable question was "answered" with a cited "the document does not say" | unanswerable refused 8/9 → 9/9, false refusals unchanged |
| Calculator with fixed operations; result always stated | returned nothing for 16 × 16; model ignored its result | numerical 1/6 → 2/6 (one question); still picks wrong cells or the wrong operation |
| Table headers / stacked rows | wrong values read from two BEA tables | values correct on the inspected tables; **retrieval recall@5 −1 question on BEA** (94.2% → 92.3%) |
| Parse each PDF once per benchmark run | every ablation variant re-parsed | benchmark wall time 31 → 27 min |
| Grader alias fix (multilingual q04) | correct "over a thousand" answer scored wrong | disclosed in `benchmark/multilingual/README.md` |
| **Reverted:** stricter "refuse unless the exact fact is present" prompt rule | – | raised single-pass false refusals 3.8% → 21.2%; removed, test keeps it out |

Net: English answer accuracy unchanged (73.1%), multilingual 88.9% (equal to the re-graded before score),
refusals and OCR improved, citation precision −1 to −5 points.

## Multilingual support (added 2026-10-25 evening)

| Capability | State | Evidence |
|---|---|---|
| Hindi/Marathi/any-script keyword search | ✅ | BM25 tokenizer previously dropped all non-ASCII text; fixed + unit tests (Devanagari matras kept inside words) |
| Cross-language retrieval (question ≠ document language, Hinglish) | ✅ measured | BGE-M3 + bge-reranker-v2-m3; translated lead query; retrieval eval on the multilingual set (RESULTS §5) |
| Answer in the question's language | ✅ | Colab: Hindi question on English paper → Hindi answer; Hinglish → Hinglish answer (docs/img/multilingual_answers_colab_t4.png); two bugs found this way and fixed (vague language instruction; "the" in the Hinglish word list) |
| Localised refusals | ✅ | Hindi + English refusal for Devanagari questions; unit test |
| Scanned PDFs (OCR eng+hin+mar) | ✅ measured | Tesseract CLI at 300 dpi; image-only Hindi doc → "OCR pages: [1, 2]" in 41 s; word recall 0.99–1.00; both scanned-page questions correct |
| Devanagari digits in grading | ✅ | normaliser maps ०-९ → 0-9; unit test |

Not claimed: universal language support. Coverage is what BGE-M3 and Qwen2.5-7B handle well
(major languages incl. Hindi and Marathi; measured here on 20 questions only), and Hindi vs Marathi are
not distinguished by script detection. The multilingual set is 20 questions, single annotator.

## Measured on a free Colab T4

* Peak GPU memory 6.9 GB / 15.4 GB with Qwen2.5-7B Q4_K_M + BGE + reranker.
* Agentic answer 8–16 s typical, ~23 s with a corrective loop.
* Smoke test 12/13 correct, 3/3 unanswerable refused.
* Full answer-level ablation: see `docs/RESULTS.md` §3.

## Outstanding / known limitations

0. **Scale round findings** (see SCALE.md):
   - Ingestion of table-heavy PDFs is slow on 2 CPUs: about 2.3–2.9 pages/s (10,000 pages ≈ 71 min;
     37 min with the opt-in GPU table detection, run 4, at the cost of up to 9.4 GB peak RAM;
     the Budget Appendix 1,224 pages ≈ 34 min). PyMuPDF's ONNX table model is ~80% of it; the
     faster line-only detector invents tables in academic papers, so it is not used.
   - Long legal codes: questions naming a section ("section 7203") failed on the US Code because chunks
     kept only the latest of two consecutive bold headings. Fixed in the sprint (each chunk records its
     section number; run 2: sections identified 355 → 1,877, recall@5 0.75 → 0.92, answerable 6/12 → 7/12,
     not held out). 5 of 12 questions are still wrong (numeric, multi-hop, a definition); the held-out
     check on a new legal code (US Code Title 21, final run): 19/25 answerable, 4/5 refused.
   - Questions that presuppose something absent from the document (an act passed after it was
     written) are sometimes answered from the nearest related passage: 6/9 unanswerable real-document
     questions refused.
   - "Which table…" answers sometimes omit the table number although the right table is retrieved.
     F6 (deterministic table naming) was added in the sprint; final run: 5/6 "which table" questions on the Statistical Abstract correct (statab-q10 wrong).
   - ~~Questions cannot be asked about a document while it is still being ingested~~: fixed on
     26 Oct (F3, answers from the indexed pages so far, labelled as partial); unit-tested. Measured
     in run 1 on a T4: askable 206 s after upload with 96/1,000 pages indexed, answer correct and cited.
   - The phone-width layout was checked at 390 and 360 px in an emulated phone browser (Chromium, 27 Oct) and touch targets were enlarged; it has not been tried on a real iPhone/Safari.
   - The synthetic corpus is template prose; real documents were limited to four US federal PDFs.

1. **The benchmark is small** (61 questions, single annotator). Grading uses
   alias matching, which can under-credit paraphrased numbers. CIs are reported
   for that reason. An LLM-judge pass would complement it.
2. **Table detection on two-column papers** sometimes merges a table with
   neighbouring equation text (arXiv p.6). The values survive, but the
   structure is noisy.
3. **Ingestion speed on Colab's 2 vCPUs:** table-heavy PDFs take ~1.3 s/page
   (BEA: 37 s for 28 pages). The result is cached, but very large reports
   (300+ pages) would take minutes.
4. **Numerical and multi-hop questions remain the weakest categories.** Retrieval
   finds the evidence. The 7B model then picks numbers from the wrong row or column of a
   dense table, or does arithmetic wrong. The calculator removes the arithmetic error,
   but operand extraction is still model-limited. Next: cell-level table retrieval,
   and a larger model (Qwen2.5-14B Q4 fits in the measured VRAM headroom) for the
   extraction step only.
4b. **Agent loop vs single pass:** accuracy gains are within the 95% interval on this
   benchmark; see RESULTS. The default (adaptive) mode buys fewer false refusals and better
   citations for ~2.5× latency. `preset("hybrid_rerank")` is the fast option.
5. **Scanned PDFs** need `apt-get install tesseract-ocr` (the notebook installs it). OCR is measured on one
   2-page Hindi scan only (word recall 0.99–1.00, 2/2 questions); quality depends on the scan.
6. **Not yet built:** visual/ColPali retrieval (deliberately deferred: the text
   pipeline already reaches 95% page recall, and a VLM would roughly double
   VRAM and latency), cross-document questions in the benchmark.
7. **PyMuPDF is AGPL-3.0.** The project is private and not distributed, so this has no effect
   now; it would matter only for a future distribution.

## Starting code

The "starting code" numbers in RESULTS come from the original prototype this project
began from. That prototype is not part of this repository. The current design is
described in `ARCHITECTURE.md`.
