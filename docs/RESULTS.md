# Results

All numbers come from code in this repository and can be reproduced. Section 4
lists the exact commands.

## 00000. Numerical-question fixes (2026-10-27, Kaggle T4 x2, runs 5 and 6)

The six numerical questions of the English set failed mostly on the wrong *measure* (a
year-over-year change for an annualized rate, a percent change for a dollar amount, a contribution
for a growth rate) or the wrong cell of the right table; the arithmetic itself was never wrong.
Four fixes were added behind flags (`config.py`) and measured on the dev sets only (English 61 q,
multilingual 20 q); the held-out sets were not used.

| English set | Default before | Three fixes (now the default) | All four fixes |
|---|---|---|---|
| Answer accuracy (52 answerable) | 76.9% (40/52) | **84.6%** (44/52) | 84.6% (44/52) |
| Numerical questions | 2/6 | 5/6 | 5/6 |
| Table questions | 10/12 | 11/12 | 11/12 |
| False refusals | 1.9% | 0% | 0% |
| Unanswerable refused | 9/9 | 9/9 | 9/9 |
| Answer latency p50 / p95 | 6.7 / 22.1 s | 6.5 / 21.8 s | 6.8 / 29.1 s |

* **The same four questions were fixed in both runs** (arXiv q10, BEA q08, BEA q10, NIST q11) and
  no question that the old default answers was lost in any configuration.
* **Which fix does what** (run 5, one flag at a time): cell alignment fixes arXiv q10 (the value had
  been copied from the `gpt-4o-mini` row for `gpt-4o`: 21.2 → 34); measure-aware retrieval fixes
  BEA q08 and q10; date candidates fix NIST q11 (a false refusal). The measure check also fixes
  BEA q10 but raised p95 latency by 6.7 s, so it stays off.
* **Multilingual unchanged**: run 5 94.4% with and without the fixes; run 6 88.9% for both the old
  default and the new one (the extra miss, the scanned Hindi question, failed in both, as it has
  in earlier runs).
* Rows: run 6 (`notebook4ee260bf2d`: new default on GPU 0, old default on GPU 1, same session);
  "All four fixes" from run 5 (`notebook64384bdfea`). Numbers were read from the run logs.
* **Not re-measured with the fixes:** the real-document questions, held-out set 2 and the six scale
  tiers below were single-use or long runs made with the previous default.

## 0000. Final run (2026-10-27, Kaggle T4 x2)

Full tables and method: [SCALE.md](SCALE.md#final-run-kaggle-t4-27-oct). Code snapshot `run3` (table detection
on the GPU, parsing identical to the CPU path on all 8 test PDFs), one document at a time.

| Measurement | Result |
|---|---|
| The 46 real-document questions (final, single use) | **30/37 answerable correct** (26/37 before the sprint fixes), 6/9 unanswerable refused (unchanged). 9/11 Report 12/12 (was 11/12), Statistical Abstract 11/13 (was 9/13), US Code Title 26 7/12 (same as run 2) |
| "Which table" questions, Statistical Abstract | 5/6 correct (wrong: statab-q10) |
| Statistical Abstract parse | 521 s (8.7 min), ingest 942 s; earlier 17.5 min parse / 23.8 min ingest with CPU table detection while two documents ran at once on the same 4 vCPUs, so not a like-for-like comparison |
| Held-out set 2, US Code Title 21 (single use) | 19/25 answerable, 4/5 unanswerable refused, page recall@5 0.917, MRR 0.769, citation precision 0.741; questions naming a section 8/9 |
| Held-out set 2, CAG Hindi GST report (answers only) | 9/14 answerable (Hindi 6/7, Hinglish 2/3, English 1/4); 2/4 unanswerable refused, checked against the executive summary only |
| Held-out set 2, GAO Cost Guide | not run: gao.gov answered HTTP 403 to the download from Kaggle; its 22 questions remain unused |
| Regression | English 76.9% (same 12 failures), multilingual 94.4% |
| Six synthetic tiers, 100 to 10,000 pages (run 4, 27 Oct, notebook `notebookf8de2130c7`, table detection on the GPU) | Run 3's scale job failed at start (two jobs pulled the model into the same Ollama directory at the same time), so the tiers were re-run alone. All six passed: recall@5 0.972 → 0.889 and answer accuracy 0.69–0.75 from 100 to 10,000 pages, identical to run v1; refusals 3/4–4/4; answer p50 6.0–6.8 s. 10,000 pages ingested in 36.9 min (4.5 pages/s; 4,237 s with CPU table detection in run v2). Crash isolation and resume passed at every tier. Peak RAM 5.6–9.4 GB |

## 000. Sprint run 1 (2026-10-26, late evening, Kaggle T4)

Regression check of the sprint changes (token streaming, ask while processing, "which table" naming,
access token) with the default agent: **English 76.9%** (the same 12 failures as before; table
questions 10/12, numerical 2/6), unanswerable 9/9, citation precision 78.4%; **multilingual 94.4%**
(17/18; Hindi 9/9, Hinglish 3/3, Marathi 4/4). No regression. Streaming: first words after 0.5 s
(median, one user). Details: [SCALE.md](SCALE.md#sprint-run-1-streaming-ask-while-processing-regression-kaggle-t4-26-oct-late-evening).

**Sprint run 2 (26–27 Oct, Kaggle T4).** Section numbers on chunks (US Code Title 26): sections identified
355 → 1,877, page recall@5 0.75 → 0.92, answerable 6/12 → 7/12 (not held out: this document was used to find
the cause; 5 of 12 are still wrong). The layout model on the GPU parsed a 1,000-page table-heavy PDF 1.89x
faster (224.7 s → 119.0 s) with identical output. The presupposition check stayed off: no gain, and false
refusals rose from 1.9% to 3.8%. The two-slot Ollama load test is confounded (a second copy of the model
shared the GPU), so one slot is kept. Details: SCALE.md, "Sprint run 2".

## 00. Long documents, real documents and several users (2026-10-26, evening)

Full tables and method: [SCALE.md](SCALE.md). All measured on a Kaggle Tesla T4, one GPU per process.

| Measurement | Result |
|---|---|
| Synthetic PDFs, 100 → 10,000 pages, 40 independent questions each | answer accuracy 0.750 → 0.722 (answerable), page recall@5 0.972 → 0.889, answer p50 6.5 → 6.4 s |
| Ingestion, 10,000 pages (58% table pages), 2 parse workers | 71 min (2.3 pages/s), 53,862 chunks, index 188 MB, cold load 0.7 s, retrieval GPU memory 2.4 GB |
| Failure recovery at every tier (killed worker + job interrupted half-way) | resumed from the checkpoint; index identical to a clean run except the crashed page; +1.4% total time at 10,000 pages |
| Real PDFs (9/11 Report, Statistical Abstract, US Code Title 26; 46 independent questions) | 26/37 answerable correct (70%), 6/9 unanswerable refused; US Code weakest (6/12; 7/12 after the section fix in sprint run 2, not held out) |
| Web server, 1–8 concurrent users | 0 failed requests; about 10.5 answers/min on one T4; p50 latency 13 s (1 user) → 45 s (8 users, queueing) |
| This benchmark (section 0) on the new architecture | English 76.9% with the same 12 failures, multilingual 94.4%: no regression |
| Project notebook, top to bottom on one T4, with the new progress line and web-app cell | completed in 32.7 min with no errors; same 12/61 English failures |

## 0. Final validation (2026-10-26): the project notebook, top to bottom, on a T4

Colab's free GPU quota was used up by the earlier runs, so the final runs used a **Kaggle
notebook with the same GPU (Tesla T4, 15,360 MiB)**, pinned to one T4 with
`CUDA_VISIBLE_DEVICES=0`. Kaggle offers two T4s; only the final notebook run was
restricted to one. The final run is `notebooks/Local_Agentic_RAG.ipynb`, unchanged, plus
one setup cell that copies the code to `/content`. It completed in 32 minutes:

* install, Tesseract with eng+hin+mar, Ollama install, and the Qwen2.5-7B pull;
* Q&A demo, including a follow-up resolved from chat history and a correct refusal;
* chat UI launched (`Running on local URL`);
* retrieval ablation, English answer benchmark, multilingual benchmark, resource report.

There were no errors. The memory watchdog restarted the Ollama runner 5 times and no question failed.

| English set (61 q) | single pass (`hybrid_rerank`) | **adaptive agent (default)** |
|---|---:|---:|
| Answer accuracy [95% CI] | 71.2% [57.7–82.7] | **76.9% [63.5–88.5]** |
| Fact recall | 74.7% | 80.1% |
| False refusal rate | 3.8% | 1.9% |
| Unanswerable correctly refused | 9/9 | 9/9 |
| Citation precision | 73.7% | 78.4% |
| Page recall@5 / MRR | 94.2% / 0.883 | 96.2% / 0.915 |
| Latency p50 / p95 | 3.2 / 6.1 s | 6.6 / 23.0 s |

Agent by category: factual 14/14 · definition 5/5 · table 10/12 · comparison 4/6 ·
multi-hop 5/9 · numerical 2/6 · unanswerable 9/9.

| Multilingual set (20 q) | agentic (default) | agentic with English-only models |
|---|---:|---:|
| Answer accuracy (18 answerable) | 88.9% (16/18) | 88.9% |
| Unanswerable refused | 2/2 | 2/2 |
| Page recall@5 / MRR | 100% / 0.972 | 94.4% / 0.880 |
| Citation precision | 85.3% | 94.1% |
| By language (default) | Hindi 7/8 · Hinglish 3/3 · Marathi 3/3 · English→Indic 3/4 | Hindi 8/8 · Hinglish 2/3 · Marathi 3/3 · English→Indic 3/4 |

Retrieval ablation (page recall@5): MiniLM 76.0% · BGE-en 80.8% · BM25 81.7% · hybrid-en
81.7% · hybrid-en + reranker 96.2% · BGE-M3 dense 84.6% · **BGE-M3 hybrid + reranker
(default) 94.2%** (recall@1 68.3%, MRR 0.883; it was 92.3% / 66.3% / 0.872 before the
table fixes).

Resources: GPU memory in use 9,069 of 15,360 MiB (LLM + embedder + reranker). Ollama install 96 s,
model pull and warm-up 134 s, retrieval models 61 s, retrieval ablation 113 s, English answer benchmark
(2 configurations) 859 s, multilingual (2 configurations) 444 s, scanned 2-page Hindi OCR ingest 16 s.

### How we got there: measured round by round (Kaggle 2×T4, one run per GPU)

Each round ran the default agent on both sets; changes were kept only when the numbers held.

| Round | Change | English | Multilingual |
|---|---|---:|---:|
| Baseline | `main` as of 2026-10-25 (reproduces the last Colab run exactly: same 14 failures) | 73.1% | 83.3% (15/18) |
| it1 | calculator grounding: operands must sit on the cited row and column; stated changes skip it; ID ranges counted | 75.0% | 83.3% |
| it2 (×2) | + keep a cited answer that ends with the "not found" marker | 75.0% / 75.0% | 94.4% / 94.4% |
| it3 | + table captions and correct table section labels | 76.9% | 94.4% |
| Final notebook | + one retry when an arithmetic question gets no calculation | **76.9%** | **88.9%** |

* **Noise.** The two it2 runs used identical code and failed exactly the same questions,
  so single-question changes in the English agent column are real. The multilingual set
  is noisier: one Hindi question flipped between it3 and the final run (the planner
  returned invalid JSON once and fell back). On the final code, multilingual is
  **88.9–94.4%** across two runs. Single pass varied by two questions (67.3% vs 71.2%).
* **What each fix bought.**
  * Captions: the model now tells arXiv Table 2 (NQ) from the look-alike Table A.3, which
    fixed q06, q11 and q13.
  * Trailing-marker fix: the Hinglish two-part question and the scanned-Hindi question
    were correct answers that had been discarded.
  * Calculator: fixed BEA q13 (no longer overrides "increased $6.8 billion") and NIST q10
    (9 actions across a page break).
* **What got worse.**
  * BEA q09 regressed: its page lost rank once its *wrong* section label was corrected.
    This was predicted by the CPU retrieval check.
  * BEA q13 regressed again in it3: the level ($3,929.7 billion) sits on page 17, which
    dropped out of the top passages.
  * NIST q11 ("months between") passes on `main` but not after the calculator change: the
    first calculator call returns nothing and the generator refuses. In one traced run the
    agent's second round computed the right 18 months. A retry for empty calculations was
    then added, but the final run still refused this question.
* **Caveat.** These fixes were designed after reading the failures of this same benchmark,
  so gains on exactly those questions overstate how well they generalise. The fixes
  themselves are generic (operand grounding, caption attachment, sentinel handling), but
  the benchmark is not a held-out test of them.

Remaining English failures (12):
* numerical: wrong table cell chosen (arXiv q10, BEA q09, q10) and NIST q11;
* table: BEA q06 (GDP-level rows never reach the candidate pool) and BEA q08 (right page
  retrieved, wrong value);
* multi-hop: arXiv q14 (needs pages 8 and 12), BEA q11, BEA q13, NIST q13 (one of two
  listed characteristics missing);
* comparison: BEA q15 (revision values) and NIST q15 (definition on page 14 not used).

## 1. Benchmark

61 hand-written questions over 3 public PDFs (details in `benchmark/README.md`):

| Doc | Type | Pages | Why it is hard |
|---|---|---|---|
| `arxiv-2503.05037` | ACL 2025 paper, two-column | 17 | column reading order, results tables, appendix |
| `bea-gdp2q25-3rd` | BEA GDP release | 28 | 18 whitespace-aligned statistical tables |
| `nist-ai-600-1` | NIST Generative AI Profile | 64 | long, many similar sections, ID-coded action tables |

Per doc: about 5 factual, 4 table, 2 numerical, 3 multi-hop, 2 comparison,
1–2 definition and 3 unanswerable questions. Each answerable question lists
its evidence pages and the "required facts" (alias lists) a correct answer must
contain. `benchmark/validate.py` checks that every fact appears on its
evidence pages.

## 2. Retrieval ablation (no LLM)

Page-level metrics over the 52 answerable questions, top-10 chunks, CPU run.
Each column changes one thing relative to its neighbour. "Starting code" is the original prototype this
project began from (pdfminer text, 2048-character chunks, MiniLM search). It is not part of this repository;
its numbers are kept as the reference point.

| Metric | starting code | minilm_dense | bge_dense | bm25 | hybrid | **hybrid_rerank** |
|---|---:|---:|---:|---:|---:|---:|
| Page recall@1 | 43.3% | 40.4% | 46.2% | 53.8% | 52.9% | **57.7%** |
| Page recall@3 | 67.3% | 56.7% | 75.0% | 75.0% | 76.0% | **91.3%** |
| Page recall@5 | 73.1% | 70.2% | 80.8% | 85.6% | 80.8% | **95.2%** |
| Page recall@10 | 79.8% | 75.0% | 84.6% | 89.4% | 93.3% | **97.1%** |
| Hit@5 | 80.8% | 76.9% | 88.5% | 94.2% | 88.5% | **100.0%** |
| MRR | 0.633 | 0.599 | 0.708 | 0.743 | 0.753 | **0.835** |

Recall@5 by document:

| | starting code | minilm | bge | bm25 | hybrid | hybrid_rerank |
|---|---:|---:|---:|---:|---:|---:|
| arxiv | 0.71 | 0.77 | 0.88 | 0.97 | 0.94 | **1.00** |
| bea (tables) | 0.71 | 0.53 | 0.74 | 0.74 | 0.65 | **0.88** |
| nist | 0.78 | 0.81 | 0.81 | 0.86 | 0.83 | **0.97** |

What this shows:
* **The cross-encoder reranker is the single biggest win.** It adds about 14
  points of recall@5 over the fused candidates and lifts MRR from 0.75 to 0.84.
  RRF fusion on its own does not beat BM25 at k=5, but it gives the best
  *candidate pool* (recall@10 93%), which is what the reranker needs.
* BM25 is unusually strong here, because questions written by an annotator
  reading the PDF reuse document terms. Real users paraphrase more, which
  favours dense retrieval, so keeping both is the robust choice.
* `minilm_dense` is *worse* than the starting code on BEA. Small token-sized
  table chunks give MiniLM very little to work with. BGE with title and
  section headers recovers this.
* Reranking costs about 5.7 s per query on CPU (24 pairs × up to 3 queries),
  and well under a second on the T4.

### Final retrieval ablation with the multilingual default (fresh Colab T4 run, final code)

| Metric | starting code | minilm_dense | bge_en_dense | bm25 | hybrid_en | hybrid_en_rerank | m3_dense | **m3_hybrid_rerank (default)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Page recall@1 | 43.3% | 40.4% | 46.2% | 53.8% | 52.9% | 59.6% | 51.0% | **66.3%** |
| Page recall@5 | 73.1% | 70.2% | 80.8% | 85.6% | 82.7% | **95.2%** | 85.6% | 94.2% |
| Page recall@10 | 79.8% | 75.0% | 84.6% | 89.4% | 92.3% | **97.1%** | 89.4% | 94.2% |
| MRR | 0.633 | 0.599 | 0.707 | 0.743 | 0.751 | 0.845 | 0.751 | **0.883** |

Switching to the multilingual models does not cost English quality. BGE-M3 + bge-reranker-v2-m3
gives the best top-1 recall (+6.7 points) and MRR. Recall@5 is one question lower than the
English-only reranker.

## 3. End-to-end on a real Colab T4 (Qwen2.5-7B-Instruct Q4_K_M via Ollama)

Environment, verified in Colab on 2026-10-25: Tesla T4 15,360 MiB, driver 580,
CUDA 13.0, Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2,
sentence-transformers 5.7.0, gradio 6.30.0 (preinstalled). `pip install -e .[colab]`
adds pymupdf 2.0 and changes no preinstalled package.

| Step | Time on Colab |
|---|---|
| Ollama install | 41 s |
| Model pull (4.7 GB) + warm-up | 141 s |
| Embedder + reranker download/load | ~60 s (first time) |
| Ingest arXiv paper (17 pp) / NIST (64 pp) / BEA (28 pp, table-heavy) | ~8 s / 24 s / 37 s (2 vCPUs, cached afterwards) |
| Agentic answer, typical (4 LLM calls) | 8–16 s |
| Agentic answer with re-retrieval + regeneration (7 calls) | ~23 s |

**Peak GPU memory: 6.9 GB / 15.4 GB.** Ollama holds 5.2 GB; the embedder and
reranker peak at 1.0–1.6 GB in PyTorch.

Smoke test: 13 benchmark questions across all
categories, 12 correct. All 3 unanswerable questions were refused. Table lookups
on the serialised BEA tables were correct (e.g. personal saving rate Q2 2025 =
5.3%). The one miss was an arXiv multi-hop question: right entity, wrong number
(0.8% instead of 0.4%).

### Improvement pass: before and after (fresh Colab T4 runtime, notebook "Run all", 2026-10-25 ~22:00–23:00 IST)

The last pass analysed every failed question of the previous run (below) and changed only what that
evidence pointed to: Tesseract-CLI OCR, refusal detection, a calculator with fixed operations whose result is
always stated, table header/row fixes, and parsing each PDF once per benchmark run. Same 61 questions, same LLM.

| Metric | single pass, before | **single pass, after** | agent, before | **agent, after (default)** |
|---|---:|---:|---:|---:|
| Answer accuracy [95% CI] | 69.2% | **69.2%** [55.8–80.8] | 73.1% | **73.1%** [61.5–84.6] |
| Fact recall | 72.8% | 73.7% | 75.6% | 77.2% |
| False refusal rate | 3.8% | 3.8% | 1.9% | 1.9% |
| Unanswerable correctly refused | 8/9 | **9/9** | 8/9 | **9/9** |
| Citation precision | 74.3% | 69.3% | 74.8% | 73.5% |
| Page recall@5 (answer run) | 94.2% | 92.3% | – | 93.3% |
| Latency p50 / p95 | 3.7 / 5.9 s | 2.8 / 4.7 s | 8.7 / 32.4 s | 6.4 / 33.0 s |
| Errors | 0 | 0 | 0 | 0 |

Agent accuracy by category, before → after: factual 1.00 → 1.00 · definition 1.00 → 1.00 · table 0.75 → 0.75 ·
comparison 0.67 → 0.67 · multi-hop 0.56 → **0.44** · numerical 0.17 → **0.33** · unanswerable 0.89 → **1.00**.

Honest reading:
* **Refusals improved** (9/9 unanswerable refused in both modes, with no rise in false refusals).
* **Answer accuracy did not move** (73.1% → 73.1%). Numerical gained one question (2/6) and multi-hop lost one (4/9);
  both are single questions, inside run-to-run noise.
* **Citation precision fell** 1–5 points, and **retrieval recall@5 fell by one question** (94.2% → 92.3% in the
  retrieval ablation, all on the BEA tables document). Most likely cause: the table-row changes alter BEA table
  chunks (BEA recall@5 0.85 with the previous table code in a local CPU run vs 0.79 now on the T4). A per-question comparison of the old and new table code (same CPU machine) pins it on BEA q06
  (nominal GDP level): its evidence page 15 drops out of the top 5 (recall 0.5 → 0.0), although the new code now reads
  that table's values correctly. Next step: check how the rebuilt row labels rank for level-type queries. We report it rather than revert, because the
  same changes fixed wrong values on two BEA tables.
* **A regression we caused and reverted:** a stricter prompt rule ("if the specific fact is not in the passages, reply
  exactly <not found>: do not describe related content") raised single-pass false refusals from 3.8% to 21.2%
  (accuracy 61.5%) in the first run of this pass. We removed it, kept a test that it stays out, and re-ran everything.
* The remaining 14 agent failures (failure report cell in the notebook) are mostly wrong table cells chosen as
  numbers (e.g. 2.61 and -3.52 instead of -0.6 and 3.8 for the GDP swing), plus one retrieval miss (BEA q06).

Retrieval ablation after the pass (fresh T4): starting code 73.1% · MiniLM 73.1% · BGE-en 79.8% · BM25 82.7% ·
hybrid-en 81.7% · hybrid-en + reranker **96.2%** · BGE-M3 dense 83.7% · BGE-M3 hybrid + reranker (default) 92.3%
(recall@1 66.3%, MRR 0.872).

### Previous final answer benchmark (before the improvement pass; fresh Colab T4 runtime, multilingual models)

Measured 2026-10-25 ~20:30 IST in a fresh Colab T4 runtime ("Run all"), final code, 61 questions.

| Metric | hybrid_rerank (multilingual retriever, single pass) | **agentic (adaptive + calculator + translation; default)** |
|---|---:|---:|
| Answer accuracy [95% CI] | 69.2% [55.8–80.8] | **73.1%** [59.6–84.6] |
| Fact recall | 72.8% | **75.6%** |
| False refusal rate | 3.8% | **1.9%** |
| Unanswerable correctly refused | 88.9% (8/9) | 88.9% (8/9) |
| Citation precision | 74.3% | 74.8% |
| Errors | 0 | 0 |
| Latency p50 / p95 | **3.7 / 5.9 s** | 8.7 / 32.4 s |

Accuracy by category (single pass / agent): factual 1.00 / 1.00 · definition 1.00 / 1.00 ·
table 0.75 / 0.75 · comparison 0.67 / 0.67 · multi-hop 0.44 / **0.56** · numerical 0.00 / **0.17** ·
unanswerable (refusal) 0.89 / 0.89.

Compared with the earlier English-only retriever (table below), the multilingual default holds English
accuracy (73.1% vs 75.0% for v4, within noise). It cuts false refusals further (1.9%), but it answered
one of the 9 unanswerable questions instead of refusing (earlier runs refused all 9). That is a real
regression and we report it.

### Earlier answer-level ablation (61 questions, English-only retriever, before the multilingual switch)

Measured on 2026-10-25. 52 answerable + 9 unanswerable questions. Accuracy = every required
fact present and not refused. Bootstrap 95% CI in brackets.

| Metric | baseline | hybrid_rerank | agentic v2 (full loop) | agentic v3 (adaptive) | **agentic v4 (adaptive + calculator, default)** |
|---|---:|---:|---:|---:|---:|
| Answer accuracy | 50.0% [37–64] | 71.2% [58–83] | 73.1% [60–85] | 71.2% [58–83] | **75.0%** |
| Fact recall | 53.2% | 74.0% | 75.6% | 74.4% | **77.6%** |
| False refusal rate | 26.9% | 9.6% | 7.7% | **3.8%** | **3.8%** |
| Unanswerable correctly refused | 100% | 100% | 100% | 100% | 100% |
| Citation precision | 70.6% | 78.7% | **82.3%** | 81.7% | 78.5% |
| Latency p50 / p95 | 2.7 / 7.8 s | **3.1 / 5.1 s** | 12.8 / 36.1 s | 7.8 / 28.0 s | 8.4 / 34.0 s |

* `baseline` = MiniLM dense, no reranker, single pass (the starting code's retrieval with the new prompt).
* `hybrid_rerank` = the new retriever, single pass (retrieve → generate).
* `agentic v2` = always-on loop (plan → retrieve⇄grade → generate⇄verify) with the v2 retrieval fix.
* `agentic v3` = adaptive: retrieve → generate → verify, and escalate only when the first
  answer is refused or unsupported. It escalated on 44% of questions.

Accuracy by category (baseline / hybrid / v2 / v3):

| Category (n) | baseline | hybrid | v2 full | v3 adaptive | v4 + calculator |
|---|---:|---:|---:|---:|---:|
| factual (14) | 0.71 | **1.00** | **1.00** | **1.00** | **1.00** |
| table (12) | 0.42 | **0.83** | 0.67 | **0.83** | 0.75 |
| comparison (6) | 0.67 | 0.83 | **1.00** | 0.83 | 0.83 |
| definition (5) | 0.60 | 0.80 | **1.00** | 0.80 | **1.00** |
| multi_hop (9) | 0.33 | 0.44 | **0.56** | 0.44 | **0.56** |
| numerical (6) | 0.17 | 0.00 | 0.00 | 0.00 | **0.17** |
| unanswerable (9, refusal) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

v4 ran on code from just before the independent review's edge-case fixes (see STATUS). Differences of
one question per category (definition, multi-hop, table between v3 and v4) are within run-to-run
noise: greedy decoding on GPU is not bit-exact across runner restarts.

**Conclusions**
1. **Retrieval quality is the dominant factor: +21 points of accuracy** (50 → 71%) and
   false refusals 27% → 10%, for +0.5 s.
2. **The agent loop's accuracy gains are small and within the confidence interval.**
   The full loop wins exactly where decomposition should help (comparison, definition,
   multi-hop) but loses on table lookups. It costs 4× the latency. Adaptive mode ties
   single-pass accuracy, but has the fewest false refusals (3.8%) and better citation
   precision, at 2.5× the latency.
3. **Numerical questions fail for every configuration without tools.** The evidence is retrieved,
   but the 7B model does arithmetic and counting wrong (222.8 instead of 443.6; 4.0 instead
   of 4.4; counts 8 of 9 items). The calculator tool (v4) computes correctly, but only lifts
   numerical accuracy to 1/6. The remaining bottleneck is *operand extraction*: the model picks
   the wrong table cells (e.g. 0.52 and 0.25 instead of 0.71 and 0.41), and the generator
   sometimes ignores the computed hint.
4. All configurations refuse 100% of unanswerable questions.

### What the agent ablation taught us

The first always-on agent (**v1**: plan → retrieve⇄grade → generate⇄verify) was
**worse than a single pass**: 23 vs 27 correct on the same 38 questions before the run
was stopped. Per-question diagnosis on Colab:

| Question | Single pass | Agent v1 | Why |
|---|---|---|---|
| BEA q02 (next release date) | correct | refused | planner keyword queries + max-over-queries reranking pushed the evidence page (p5) out of the top 6 |
| BEA q08 (mining industry growth) | correct, evidence at rank 2 | "not specified" | the same dilution moved the evidence to rank 4; the grader's reordering pushed it further |
| arXiv q09 (Wikidata ID) | correct | wrong row (P179) | the right page was retrieved; the model read the wrong row of a dense table |
| arXiv q04, BEA q06 | correct | error | the Ollama runner was killed by the OOM / watchdog (infrastructure, since fixed with retries) |

Fixes:
1. **v2:** rerank by the user's question and let each sub-query only *add* its best
   passage; the grader filters but never reorders.
2. **v3 (adaptive, now the default):** answer single-pass, verify, and escalate to the
   loop only when the answer is refused or unsupported.

## 5. Multilingual (Hindi, Hinglish, Marathi, scanned Hindi)

Set: `benchmark/multilingual/` has 20 questions (hi 9, en 4, hinglish 3, mr 4; 2 unanswerable)
over a Hindi Wikipedia article (8 pp), a Marathi article (3 pp), an image-only scan of 2 Hindi pages
(OCR), and the English arXiv paper asked in Hindi/Hinglish. Facts were validated on evidence pages.

### Cross-lingual retrieval (CPU run, question only, no translation step, no OCR available locally)

Page recall@5 by question language. The 2 scanned-document questions are misses here, because
Tesseract is not installed on the dev machine; they are scored in the Colab run below.

| Question language | English-only retriever (bge-base-en + bge-reranker-base) | **Multilingual default (BGE-M3 + bge-reranker-v2-m3)** |
|---|---:|---:|
| Hindi (incl. 2 scanned) | 0.62 | **0.75** |
| Hinglish | 0.67 | **1.00** |
| Marathi | 1.00 | 1.00 |
| English (about Hindi/Marathi docs) | 1.00 | 1.00 |
| **Overall** | 0.78 | **0.89** (= every non-scanned question) |

Multilingual default: recall@1 80.6%, MRR 0.889.

### After the improvement pass (fresh Colab T4, same 20 questions)

| | agentic, before | **agentic, after (default)** | agentic_en, after |
|---|---:|---:|---:|
| Answer accuracy (18 answerable) | 83.3% (15/18; 88.9% re-graded with the corrected q04 alias) | **88.9%** (16/18) [72–100] | 88.9% |
| Unanswerable refused | 2/2 | 2/2 | 2/2 |
| False refusals | 11.1% | 5.6% | 5.6% |
| Page recall@5 / MRR | 0.944 / – | **1.000 / 0.972** | 0.944 / 0.880 |
| Citation precision | 85.4% | 91.2% | 92.2% |
| Scanned Hindi (OCR) questions | 1/2 | **2/2** | 2/2 |
| Latency p50 | 13.5 s | 11.0 s | 8.2 s |

After, by question language (default): Hindi 8/8 (incl. both scanned-page questions), Hinglish 2/3, Marathi 3/3,
English questions about the Hindi/Marathi documents 3/4. The two misses: a Hinglish two-part question that was
refused although its evidence pages were retrieved, and an English question needing 16 × 16 = 256 that the
calculator computed as a sum (16 + 16). On equal grading the overall score is unchanged (88.9% → 88.9%); the OCR
fix gained the scanned question, and Hinglish lost one. English-only models with the same agent again tie on
answers but retrieve worse (recall@5 0.944 vs 1.000).

### Previous end-to-end run on the T4 (before the improvement pass)

Default config (adaptive agent, BGE-M3 + bge-reranker-v2-m3, translated lead query, Qwen2.5-7B),
fresh Colab T4 run, 20 questions (18 answerable, 2 unanswerable):

| Slice | n | Answer accuracy | Other |
|---|---:|---:|---|
| **Overall** | 20 | **83.3%** (15/18) | unanswerable refused 2/2 · false refusals 11.1% (2/18) · citation precision 85.4% · page recall@5 94.4% · p50 13.5 s |
| Hindi questions | 9 | 87.5% (7/8) | refusal 1/1 · recall@5 0.88 |
| Hinglish questions | 3 | **100%** | recall@5 1.00 |
| Marathi questions | 4 | **100%** | refusal 1/1 |
| English questions about the Hindi/Marathi docs | 4 | 50% (2/4) | recall@5 1.00, so the evidence was found and the failures are in answering |
| By document: English paper asked in Hindi/Hinglish | 3 | 100% | |
| By document: Hindi article (text) | – | 86% | |
| By document: Marathi article | – | 83% | |
| By document: **scanned Hindi (OCR)** | 2 | 50% (1/2) | OCR worked (2 pages, 12 chunks), and one evidence page ranked low |

**Ablation: same agent, English-only retriever (`agentic_en`), same run:**

| | English-only models + agent | Multilingual default |
|---|---:|---:|
| Answer accuracy (18 answerable) | 83.3% | 83.3% |
| Page recall@5 | 0.889 | **0.944** |
| Hindi questions: accuracy / recall@5 | 0.88 / 0.75 | 0.88 / **0.88** |
| Hinglish accuracy | 0.67 | **1.00** |
| Marathi accuracy | 1.00 | 1.00 |
| English → Indic doc accuracy | **0.75** | 0.50 |
| Citation precision | **87.5%** | 85.4% |
| Latency p50 | **9.9 s** | 13.5 s |

Honest reading: at the answer level the two are **tied on this 20-question set**. The
query-translation step does much of the cross-language work even with English-only models. The
multilingual models clearly improve *retrieval* (recall@5 +5.5 points, Hindi +13) and Hinglish, and
lose one English→Indic question. A larger multilingual set is needed to separate them.

What this supports: cross-language retrieval works (recall@5 0.94 including OCR pages), and answers
come back in the question's language (Hindi → Hindi, Hinglish → Hinglish, Marathi → Marathi; see
`docs/img/multilingual_answers_colab_t4.png`). Weak spots: English questions about Indic documents
(2/4), and OCR'd pages. The set is tiny (20 questions), so each question is 5 points.

## 4. Reproduce

```bash
python benchmark/fetch_pdfs.py
python -m local_agentic_rag.bench retrieval                    # CPU OK (~15 min), GPU ~3 min
python -m local_agentic_rag.bench answer --llm ollama:qwen2.5:7b-instruct
```

Results land in `benchmark/results/` (`*_report.md`, `*_summary.json`, per-question JSONL).
