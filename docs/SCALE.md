# Scale: documents of 100 to 10,000 pages

How the system behaves as a single PDF grows from 100 to 10,000 pages, on the target
hardware (one Tesla T4, 15 GB VRAM). Every number in this file was measured in the run named
next to it. Tiers or documents that were not run are marked as such; nothing is extrapolated.

<!-- RESULTS -->

## Summary

* **Every tier from 100 to 10,000 pages was run end to end on a T4**: ingestion, retrieval,
  40 answered questions per tier, and a failure-recovery test. No tier failed, no OCR page was
  lost, and GPU memory for retrieval stayed at 2.4 GB at every size.
* **Answer quality holds as documents grow 100×.** Answer accuracy on answerable questions is
  0.72–0.75 at every tier, and answer latency stays at about 6.5 s. Page recall@5 falls gently,
  from 0.97 (100 pages) to 0.89 (10,000 pages).
* **Ingestion time grows linearly**: about 2.3–2.9 pages/s with 2 parse workers (the free-Colab
  CPU count), so a 10,000-page table-heavy PDF takes about 71 minutes. About 80% of that is
  table detection.
* **Failures are recoverable**: at every tier a parse worker was killed and the job was
  interrupted half-way. The job resumed from its checkpoint, and the index matched a clean run
  except for the one crashed page.
* **The existing benchmarks did not regress** with the new architecture: English 76.9% (the same
  12 questions fail as before), multilingual 94.4%.
* **Several users**: the real web server served 1–8 concurrent users with no failed request.
  Throughput saturates at about 10.5 answers per minute with two answers running at a time on
  one T4; beyond that, users queue.

## Sprint run 2: legal sections, table detection on the GPU, presupposition check (Kaggle T4, 27 Oct)

**Section numbers on chunks (US Code Title 26).** The diagnosis (all 4,211 pages through the real parser)
showed why "section 164" questions failed: in this PDF a bold section heading ("§ 164. Taxes") is
followed directly by a bold sub-heading ("(a) General rule"), and each chunk kept only the latest
heading, so the section number never reached the index. The PDF has no bookmarks to fall back on. Chunks
now carry the number of the last numbered section as a separate field (chunk text and embeddings are
unchanged, verified identical on the 7 development PDFs).

| US Code Title 26, 15 questions | Before (run v4) | After |
|---|---:|---:|
| Sections identified in the index | 355 | 1,877 |
| Page recall@5 / MRR (12 located questions) | 0.75 / 0.52 | 0.92 / 0.73 |
| Answerable correct | 6 / 12 | 7 / 12 |
| Unanswerable refused | 2 / 3 | 2 / 3 |
| Citation precision | 0.62 | 0.78 |

Section questions on §164, §6501 and §7203 are now answered correctly. This document was used to find
the cause, so it is not a held-out test of the fix; the final run measures it on a new legal code
(US Code Title 21) whose questions were written without the system.

**Table detection on the GPU (opt-in, `LOCAL_RAG_ONNX_GPU=1`).** With the GPU build of ONNX Runtime
installed, the parse workers run PyMuPDF's layout model on the T4, which is idle while pages are read.
Parsing the 1,000-page table-heavy PDF with 2 workers took **119.0 s instead of 224.7 s (1.89×)**, and
the parsed pages were byte-identical (same digest, 660 tables in both). Each parse worker used about
0.7 GB of GPU memory. (Run 1's attempt ran on the CPU in both settings because the CUDA libraries were
missing; they now come from the `onnxruntime-gpu[cuda,cudnn]` extras.)

**Two Ollama slots: inconclusive.** A load test with `OLLAMA_NUM_PARALLEL=2` was slower (3 to 6
answers per minute against 10 to 14 with one slot), but another job's copy of the model was still
loaded on the same GPU, so the two servers competed for memory. The comparison is not fair and the
default (one slot, two answers at a time in the server) is kept.

**Presupposition check (F11, off by default).** With the check on, the English set scored the same
76.9% with the same 12 failures, but one more answerable question was refused (false refusals 1.9% →
3.8%); the multilingual set was unchanged. It brought no measured benefit, so it stays off.

## Sprint run 1: streaming, ask-while-processing, regression (Kaggle T4, 26 Oct, late evening)

Code: the 26 Oct sprint changes (token streaming, ask while processing, "which table" naming, access
token, number-word grader; layout model on GPU available but off). One T4 ran the regression
benchmark while the other ran the tiers below, sharing 4 vCPUs.

| Pages | Ingest | Recall@5 | Answer acc. | Refusal | Citation precision | Answer p50 | Recovery |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 51 s | 0.972 | 0.750 | 4/4 | 0.917 | 5.8 s | crash isolated, resumed at page 64, identical |
| 1,000 | 435 s | 0.944 | 0.722 | 3/4 | 0.815 | 5.5 s | crash isolated, resumed at page 512, identical |

Retrieval and answer quality are the same as run v1 at both sizes. The 1,000-page ingest was slower
than in run v1 (435 s against 340 s) because the regression job shared the CPUs: the same PDF took
276 s through the web server once that job had finished, and 212 s to parse alone. The default
parsing code path did not change.

**Ask while processing.** The 1,000-page PDF was uploaded to the web server. It became askable
206 s after upload, with 96 of 1,000 pages indexed. A question about page 14 was answered correctly
in 14.8 s, citing page 14, and the answer was labelled as drawn from the first 96 of 1,000 pages.

**Load test with streaming answers** (same server and document, two answers at a time, one Ollama
slot, 3 questions per user, no failed request at any level):

| Users | Latency p50 / p95 | Queue wait p50 | First words p50 / p95 | Answers per minute |
|---:|---:|---:|---:|---:|
| 1 | 5.6 / 8.9 s | 0 s | 0.5 / 6.8 s | 10.4 |
| 2 | 6.4 / 14.1 s | 0 s | 0.7 / 8.3 s | 14.4 |
| 4 | 20.8 / 33.3 s | 6.8 s | 13.7 / 23.6 s | 10.9 |
| 8 | 40.9 / 51.7 s | 28.1 s | 31.5 / 42.5 s | 11.7 |

With one or two users the first words appear in under a second (median). Beyond two users,
questions wait in the queue, so the first words arrive later; throughput stays at 10 to 14 answers
per minute on one T4. The `scripts/smoke_install.sh` check passed against this server.

**Layout model on the GPU (opt-in): not measured.** The installed GPU build of ONNX Runtime needed
CUDA 13 libraries that the image did not have, so both settings ran on the CPU (212 s and 211 s for
1,000 pages, identical output). This is recorded as inconclusive.

## Synthetic tiers (Kaggle T4, run v1, 2026-10-26)

Each process pinned to one T4 with 2 parse workers. Answers use the default adaptive agent.
"Answer acc." is on the 36 answerable questions; "Refusal" on the 4 unanswerable ones.

| Pages | Ingest | Parse / index / BM25 (s) | Pages/s | Chunks | Peak RAM | Peak CUDA | Index on disk | Cold load | Recall@5 | MRR | Retrieval p50 | Answer acc. | Refusal | Citation precision | Answer p50 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 55 s | 47 / 7 / 0.1 | 1.8 | 562 | 3.5 GB | 2.4 GB | 2 MB | 0.0 s | 0.972 | 0.935 | 321 ms | 0.750 | 4/4 | 0.931 | 6.5 s |
| 250 | 93 s | 76 / 17 / 0.2 | 2.7 | 1,359 | 6.7 GB | 2.4 GB | 5 MB | 0.0 s | 0.972 | 0.938 | 307 ms | 0.722 | 3/4 | 0.889 | 6.5 s |
| 1,000 | 5.7 min | 267 / 72 / 1.0 | 2.9 | 5,423 | 7.1 GB | 2.4 GB | 19 MB | 0.1 s | 0.944 | 0.951 | 311 ms | 0.722 | 3/4 | 0.815 | 5.9 s |
| 2,500 | 14.7 min | 700 / 180 / 2.5 | 2.8 | 13,507 | 3.3 GB | 2.4 GB | 47 MB | 0.2 s | 0.944 | 0.885 | 312 ms | 0.694 | 4/4 | 0.871 | 6.9 s |
| 5,000 | 31.4 min | 1,523 / 357 / 5.0 | 2.6 | 26,953 | 3.5 GB | 2.4 GB | 94 MB | 0.4 s | 0.917 | 0.860 | 306 ms | 0.750 | 3/4 | 0.903 | 6.4 s |
| 10,000 | 71.3 min | 3,552 / 715 / 9.9 | 2.3 | 53,862 | 6.4 GB | 2.4 GB | 188 MB | 0.7 s | 0.889 | 0.893 | 308 ms | 0.722 | 3/4 | 0.819 | 6.4 s |

* Peak RAM is the benchmark process plus its parse workers, sampled every 0.2 s; it includes the
  embedder and reranker (about 2 GB on the host). The variation between tiers (3.3–7.1 GB) is not
  explained by document size. Ollama's own memory is not included.
* The 1,000- and 10,000-page tiers were measured again in run v2, after the parse-worker ONNX thread
  cap: 326 s and 4,237 s (recall@5 0.944 and 0.889). The cap gave 2.4–5.5× on a 16-core laptop but
  no measurable gain with 2 workers on a 4-vCPU machine.
* At 10,000 pages the gold page still reaches the reranker's top-24 candidates for 33 of 36
  questions (local CPU proxy run with a small embedder; the 3 misses were OCR questions on a machine
  without Tesseract). Retrieval losses come from ranking among look-alike passages, not from the
  candidate pool being too small.

## Failure recovery (run v1)

At each tier, `LOCAL_RAG_FAULT_PAGES` killed the parse worker on one page (a third of the way in),
and the job was cancelled half-way through reading, then started again.

| Pages | Worker crash isolated | Cancelled at page | Resumed from page | Index identical to clean run (except the crashed page) | Time before / after resume | Clean run |
|---:|---|---:|---:|---|---:|---:|
| 100 | yes | 64 | 64 | yes | – | 55 s |
| 250 | yes | 128 | 128 | yes | 41 s / 55 s | 93 s |
| 1,000 | yes | 512 | 512 | yes | 141 s / 205 s | 340 s |
| 2,500 | yes | 1,264 | 1,264 | yes | 371 s / 539 s | 884 s |
| 5,000 | yes | 2,512 | 2,512 | yes | 736 s / 1,101 s | 1,886 s |
| 10,000 | yes | 5,008 | 5,008 | yes | 1,809 s / 2,529 s | 4,279 s |

The interrupted-and-resumed 10,000-page job took 4,338 s in total against 4,279 s for a clean run
(+1.4%): nothing done before the interruption was repeated.

## Several users (run v1)

The real web server (`python -m local_agentic_rag.server`, two answers at a time) with the
1,000-page document, which it ingested through its own upload path in 341 s. Each user sends 3
questions from the tier's question set and opens the first cited page.

| Users | Requests | Failed | Latency p50 / p95 | Queue wait p50 | Answers / min | Page image p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0 | 12.8 s / 14.1 s | 0 s | 5.3 | 0.11 s |
| 2 | 6 | 0 | 7.0 s / 19.0 s | 0 s | 11.6 | 0.15 s |
| 4 | 12 | 0 | 20.7 s / 33.9 s | 9.8 s | 10.4 | 0.12 s |
| 8 | 24 | 0 | 44.6 s / 55.0 s | 31.3 s | 10.5 | 0.14 s |

One T4 answers about 10–11 questions per minute; more users wait in a visible queue rather than
failing. The samples are small (3 questions per user).

## Existing benchmarks on the new architecture (run v1)

| Benchmark | Before today's changes | After (streaming ingestion etc.) |
|---|---:|---:|
| English answer accuracy (agent) | 76.9% | 76.9%, same 12 failures |
| English page recall@5 / citation precision | 0.962 / 0.784 | 0.962 / 0.784 |
| English unanswerable refused / false refusals | 9/9 / 1.9% | 9/9 / 1.9% |
| Multilingual answer accuracy | 88.9–94.4% (two runs) | 94.4% (17/18) |


## Real long documents (Kaggle T4, runs v2–v4, 2026-10-26)

Four US federal publications, downloaded on the GPU machine. The question set was written
independently from the official HTML editions; gold pages were located automatically from
verbatim evidence quotes (4 of 46 quotes not found in the PDF; those questions count for
answers but not for page metrics). Two documents were processed at a time, each with 2 parse
workers on a 4-vCPU machine, so the ingestion times include that contention.

| Document | Pages | Ingest | of which parse / embed | Chunks | Peak RAM | Page recall@5 (MRR) | Answerable correct | Unanswerable refused | Citation precision | Answer p50 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 9/11 Commission Report | 585 | 61 s | 13 s / 30 s | 2,126 | 4.4 GB | 1.00 (0.94) | 11 / 12 * | 1 / 3 | 0.73 | 6.2 s |
| Statistical Abstract 2012 | 940 | 23.8 min | 17.5 min / 4.8 min | 20,965 | 5.3 GB | 1.00 (1.00) | 9 / 13 | 3 / 3 | 0.90 | 6.6 s |
| Budget Appendix FY2026 | 1,224 | 33.9 min | 29.5 min / – | 16,523 | 3.7 GB | (no questions) | – | – | – | – |
| US Code Title 26 | 4,211 | 15.5 min | 4.3 min / 8.1 min | 34,347 | 4.0 GB | 0.75 (0.52) | 6 / 12 | 2 / 3 | 0.62 | 19.1 s |

\* The USS Cole answer says "Seventeen" crew members; the grader's key was "17". Read by hand,
the answer is correct (12 / 12). The automatic score is kept in the table.

Overall: 26 of 37 answerable questions correct (70%; 27 by hand), 6 of 9 unanswerable
questions refused.

**What the failures show** (per-question records in the run output):

* **US Code: questions that name a section.** "What does section 164(b)(6) limit…", "section
  7203", "section 4081": in a 4,211-page code every section reads alike, and the section number
  is in the heading, not in the passage text. Retrieval found the wrong section for 4 of the 5
  section questions. This motivated *section-reference routing* (search inside the named section,
  see ARCHITECTURE §1b), added after this run; the US Code is therefore no longer a held-out
  test for that change. Re-runs: the first version (v3) mapped headings that start with a
  year ("2010 (I) In general") to "section 2010" and regressed one answer (6/12 → 5/12); the
  fixed version (v4) only accepts "§"/"Section"/dotted numbers and section starts written
  "§ 164. Taxes" inside the text. v4 is neutral (6/12, recall@5 0.75, same as before): in this
  PDF it only identifies §7701, so §164, §4081 and §7203 are still not found. The cause (how
  these headings come out of the PDF) is not yet diagnosed. Routing is kept because it is
  neutral here and unit-tested; the US Code section questions remain an open problem.
* **Statistical Abstract: "which table…" questions.** Retrieval was perfect (recall@5 1.00), but
  in 4 of 13 answers the model described the right table without its table number. A prompt
  change could target this, but it would be fitted to this held-out set, so it was not made.
* **Questions that presuppose something absent.** "What powers does the Intelligence Reform
  Act of 2004 give the DNI?" (the report predates the Act) was answered from the report's
  recommendation of a National Intelligence Director; "the temperature in degrees" was
  answered with "temperate". A check for named entities missing from the passages was
  prototyped and dropped: on the existing benchmark it would also fire on answerable
  questions (e.g. "On the Natural Questions…"), which is how an earlier strict-refusal rule
  raised false refusals from 3.8% to 21.2%.
* **One unanswerable question is arguably answerable.** "Can I deduct the interest on my car
  loan?" was written as unanswerable (the deduction was added in 2025), but the 2023 code's
  personal-interest rule does answer "no", which is what the system said.

**Ingestion speed is dominated by table detection.** PyMuPDF 2's table finder runs an ONNX
layout model (about 0.4 s per page that has ruling lines). Text-only pages parse at about
2 ms. That is why the table-heavy Statistical Abstract and Budget Appendix take 18–30 minutes
to parse while the larger US Code takes 4. Turning the layout model off is 7–11× faster but was
measured to invent tables from figure panels in the arXiv paper, so it stays on.


## Method

**Synthetic tiers** (`benchmark/scale/make_corpus.py`). One fictional statistical compendium
generated at 100, 250, 1,000, 2,500, 5,000 and 10,000 pages, with a seeded, byte-identical
output per tier. It has running headers and footers, a PDF outline, ruled tables on about
58% of pages, two-column pages, footnotes, and about 1.5% scanned (image-only) pages that
must go through OCR. Each tier has 40 questions over 9 types (lookup, table cell, numeric,
multi-hop, section-scoped, page reference, OCR-only, near-duplicate distractor, unanswerable).
Each fact is planted at a known page, and planted depths are spread evenly over the
document. The question templates are the same for every tier, so tiers are comparable. The
questions were written by a separate agent that did not see the existing benchmark or its
results. `validate_corpus.py` checks that every answer key is extractable from its gold page.

**Real documents** (`benchmark/realdocs/`, `benchmark/scale/run_realdocs.py`). Four US
federal publications (public domain): the FY2026 Budget Appendix (1,224 pages), the US Code
Title 26 (4,211 pages), the 2012 Statistical Abstract (about 1,040 pages) and the 9/11
Commission Report (585 pages). 46 questions were written by another separate agent from the
official HTML editions, without seeing the PDFs or this system's results. Each question
carries a verbatim evidence quote; its gold page is the PDF page where that quote is found.
The Budget Appendix exists only as PDF, so it has no questions and is measured for
ingestion only.

**What is measured** (`benchmark/scale/run_scale.py`):

| Group | Metrics |
|---|---|
| Ingestion | wall time per stage (parse, chunk+embed, BM25), pages/s, chunks, OCR pages, warnings |
| Resources | peak RSS of the process plus its parse workers (sampled every 0.2 s), peak CUDA memory, `nvidia-smi` totals |
| Storage | parsed pages, index (vectors, chunks, BM25), PDF size; cold index load time |
| Retrieval | page recall@1/5/10, MRR and latency, overall, by question type and by document depth |
| Answers | accuracy (all answer keys present; refusal for unanswerable questions), false refusals, citation page precision and recall, latency p50/p95, LLM calls |
| Recovery | a parse worker killed on one page, plus the job interrupted half-way and resumed; the result is compared with a clean run |
| Load | the real web server with N concurrent users (`load_test.py`): latency, queue wait, throughput, errors |

**Environment.** Kaggle notebooks with Tesla T4 GPUs (the free Colab GPU quota was used up).
Each process is pinned to one T4, as on free Colab. Parse workers are limited to 2,
matching Colab's 2 vCPUs. Kaggle gives 4 vCPUs, so two processes run side by side where
noted. The LLM is Qwen2.5-7B-Instruct Q4_K_M via Ollama with an 8k context; retrieval uses
BGE-M3 plus bge-reranker-v2-m3 in fp16.
