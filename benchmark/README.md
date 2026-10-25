# PDF-QA benchmark (v1)

A small question/answer benchmark that was checked by hand, for measuring
**retrieval recall, answer correctness, citation accuracy and refusal behaviour**
of RAG pipelines that run on local LLMs and answer questions about a single
uploaded PDF.

- 3 public PDFs of different kinds (a two-column paper, a statistical report that
  is mostly tables, a structured standards document)
- 61 questions in 7 categories, including 9 that cannot be answered from the document
- Every answerable question records its **evidence pages** and the **required facts** a
  correct answer must contain. A script checks every fact against the PDF text.

The PDFs are **not** stored in this repository. `fetch_pdfs.py` downloads them from
the original publishers and checks them by SHA-256 hash.

## Documents

| doc_id | Document | Pages | Why it's included |
|---|---|---|---|
| `arxiv-2503.05037` | Fayyaz et al., *Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence* (ACL 2025), [arXiv v2](https://arxiv.org/pdf/2503.05037v2) | 17 | A two-column academic layout with many small numeric tables (Tables 2–5, A.1–A.9) and figures. It is from 2025, so local LLMs are unlikely to have memorized its numbers. Its topic (retriever bias) also fits a RAG project. |
| `bea-gdp2q25-3rd` | U.S. BEA news release *GDP, 2nd Quarter 2025 (Third Estimate), GDP by Industry, Corporate Profits (Revised), and Annual Update* (Sept 25 2025), [PDF](https://www.bea.gov/sites/default/files/2025-09/gdp2q25-3rd.pdf) | 28 | Six pages of prose followed by 18 wide, dense time-series tables plus two appendix tables. Most answers are in table cells, so the pipeline must find the right row and column (for example, the last column is 2025 Q2). US government work, public domain. |
| `nist-ai-600-1` | NIST AI 600-1, *AI RMF: Generative AI Profile* (July 2024), [PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | 64 | A structured standards document: a numbered list of 12 risks, definitions, footnotes, about 30 pages of tables of action IDs (GV-/MP-/MS-/MG-x.y-nnn), and an appendix. The text uses typographic ligatures (ﬁ, ﬀ), which test how well text extraction holds up. Public domain. |

`documents.json` holds the URL, SHA-256, page count, document type and license note
for each document. arXiv versioned URLs (`.../2503.05037v2`) are stable. The arXiv
PDF is copyrighted by its authors and is only downloaded, never redistributed.

## Files

| File | Purpose |
|---|---|
| `documents.json` | The document manifest |
| `questions.jsonl` | One question per line (schema below) |
| `fetch_pdfs.py` | Downloads the PDFs and checks their SHA-256 (standard library only, works in Colab) |
| `validate.py` | Checks the ground truth against the real PDF text (needs PyMuPDF) |
| `pdfs/`, `results/` | Ignored by git: downloaded PDFs and run outputs |

## Question schema

```json
{"q_id": "bea-gdp2q25-3rd-q05",
 "doc_id": "bea-gdp2q25-3rd",
 "category": "table",
 "question": "What was the personal saving rate (saving as a share of disposable personal income) in Q2 2025?",
 "answerable": true,
 "reference_answer": "5.3 percent (Table 8).",
 "required_facts": [["5.3"]],
 "evidence_pages": [16],
 "notes": "p16 Table 8 line 44: \"Personal saving as a percentage of disposable personal income 3.3 5.6 5.4 5.8 5.1 4.7 5.2 5.3\" (last column = 2025 Q2)"}
```

- `category` is one of `factual`, `table`, `numerical`, `multi_hop`, `comparison`, `definition`, `unanswerable`.
  - `table`: the answer appears only in a table.
  - `numerical`: needs simple arithmetic on values from the document. The reference answer is the computed value, which usually does not appear in the PDF.
  - `multi_hop`: needs two or more separate passages, usually on different pages.
  - `unanswerable`: on-topic and plausible, but the answer is not in the document. The full text was searched for the key terms to confirm this.
- `required_facts` is a list of facts, and each fact is a list of acceptable **aliases**. An answer
  passes when, **for every fact, at least one alias** appears in it after normalization:
  lowercase, whitespace collapsed, and ideally also NFKC normalization and removal of the
  commas in numbers (`1,234` → `1234`). It is an empty list for unanswerable questions.
- `evidence_pages` are **1-indexed physical pages** (PyMuPDF page index + 1), **not** the
  printed page labels. In NIST AI 600-1, for example, printed page "1" is physical page 5.
- `notes` explains why the answer is correct and quotes a short verbatim snippet from each evidence page.
  Ellipses (`...`) mark text left out of a quote.
- Unanswerable questions have `answerable: false`, `reference_answer: "Not stated in the document."`,
  and empty `required_facts` and `evidence_pages`.

## Category counts

| category | arxiv-2503.05037 | bea-gdp2q25-3rd | nist-ai-600-1 | total |
|---|---:|---:|---:|---:|
| factual | 5 | 4 | 5 | 14 |
| table | 4 | 4 | 4 | 12 |
| numerical | 2 | 2 | 2 | 6 |
| multi_hop | 3 | 3 | 3 | 9 |
| comparison | 2 | 2 | 2 | 6 |
| definition | 1 | 2 | 2 | 5 |
| unanswerable | 3 | 3 | 3 | 9 |
| **total** | **20** | **20** | **21** | **61** |

## Usage

```bash
# 1. download + verify the PDFs (-> benchmark/pdfs/<doc_id>.pdf)
python benchmark/fetch_pdfs.py            # add --dest /content/pdfs in Colab

# 2. check the ground truth against the PDFs
pip install pymupdf
python benchmark/validate.py    # -v prints every question
```

The validator **fails** (exit code 1) on any of these: a schema problem, a duplicate `q_id`, a
hash or page-count mismatch, an evidence page outside the document's page range, or an
answerable question (other than `numerical`) where some required fact has no alias in the
text of its evidence pages. It **warns** on two things. The first is an unanswerable
question whose distinctive keywords all appear on one page, a heuristic hint that it might
be answerable after all. The second is a quoted snippet in `notes` that it cannot find on the
evidence pages.

Current result: `0 error(s), 1 warning(s) — VALIDATION PASSED`. The single warning
(`nist-ai-600-1-q21`, "how many people took part in the Generative AI Public Working Group")
was reviewed by hand. Pages 3 and 6 name the working group, but neither gives a participant count.

### Suggested metrics

- **Retrieval recall@k**: did the retrieved chunks include at least one evidence page (for
  `multi_hop`, all of them)?
- **Answer correctness**: did the answer include every required fact? Use an LLM judge against
  `reference_answer` as a second signal.
- **Citation accuracy**: what fraction of the pages the system cites are in `evidence_pages`?
- **Refusal behaviour**: on `unanswerable` questions the system should abstain; on answerable
  ones it should not. Report both rates.

## Known limitations

- **Small**: 61 questions over 3 documents. Confidence intervals are wide, so treat differences
  of a few points as noise.
- **One annotator**: the questions, answers and evidence were written and checked by a single
  annotator, an LLM agent. The validator confirms each fact appears on the cited pages. It does
  **not** confirm that the question is unambiguous or that no other page also contains the
  answer: evidence lists can be incomplete, and other passages may also support the answer.
- **The alias grader can give too little or too much credit.** It gives too little when a correct answer
  is phrased differently ("-$42.1 billion" vs "a 42.1 billion loss" both pass, but "42 billion"
  does not), or when values are rounded or written in other units ("$30.5 trillion" will not
  match `30,485.7`). It gives too much when an alias happens to appear as a substring, which is
  more likely for short numbers such as `5.3` or `0.54`, or when an answer lists many candidates.
  Pair it with an LLM judge.
- **Table alignment**: answers in the BEA document depend on reading the right column of a wide
  table. Column order was checked by hand against the column headers and cross-checked with the
  text. Plain text extraction separates chart labels from their values, so chart-only facts were
  avoided or given a table as the evidence.
- **Memorization**: the headline numbers in the BEA release (Q2 2025 GDP +3.8%) and the
  12-risk list in NIST AI 600-1 may be in some models' training data. A model could then answer
  correctly without retrieving anything, so check citations and not only correctness.
- **Ground truth depends on the exact version**: page numbers and facts refer to the exact files
  identified by SHA-256 in `documents.json`. If a publisher replaces a file, `fetch_pdfs.py` will
  report a hash mismatch and the benchmark needs to be checked again.
