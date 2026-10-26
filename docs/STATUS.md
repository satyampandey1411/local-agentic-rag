# Project status — 2026-10-26 (after the Kaggle validation round)

Honest snapshot of what is implemented, what is verified and how, and what is
still outstanding.

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

Test suite: **472 passed**, offline, ~15 s (`pytest -q`). `ruff check .` clean.

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
   VRAM and latency), cross-document questions in the benchmark, streaming
   answers in the UI.
7. **PyMuPDF is AGPL-3.0.** This must be considered when a license is chosen; a
   permissive license would need `pypdfium2` instead.

## Starting code

The "starting code" numbers in RESULTS come from the original prototype this project
began from. That prototype is not part of this repository. The current design is
described in `ARCHITECTURE.md`.
