# Multilingual PDF-QA benchmark (v0)

A small, hand-checked add-on to the main benchmark (`benchmark/`) for Indian-language use.
It uses the same schema, plus one extra field, `question_language`.

## What it tests

| Skill | How |
|---|---|
| Retrieval and answering in Hindi and Marathi | Hindi questions on a Hindi PDF, Marathi questions on a Marathi PDF |
| Retrieval across languages | English questions on Hindi and Marathi PDFs, Hindi questions on the Marathi PDF, Hindi and Hinglish questions on an English paper |
| Hinglish (Hindi written in Latin script) | e.g. "Taj Mahal kisne banwaya tha aur yeh UNESCO World Heritage Site kis saal bana?" |
| Answering in the user's language | `reference_answer` is written in the question's language. Grading uses aliases in both scripts, so an answer passes whichever script it uses |
| Numbers written in different forms | The documents use Devanagari digits (१६३२) and numbers spelled out as words (बीस हज़ार, दोनशे छप्पन्न). Each fact lists aliases for every form |
| OCR | `hi-scanned` is image-only (no text layer). A pipeline without OCR should find nothing in it |
| Refusal | 2 questions are on topic but cannot be answered from the document |

## Documents

| doc_id | Language | Pages | Source |
|---|---|---:|---|
| `hi-tajmahal` | Hindi | 8 | Hindi Wikipedia "ताजमहल", page id 5804, revision 6620327 |
| `mr-shaniwarwada` | Marathi | 3 | Marathi Wikipedia "शनिवार वाडा", page id 4954, revision 2705147 |
| `hi-scanned` | Hindi (image) | 2 | Pages 1–2 of `hi-tajmahal` rendered as 200-dpi grayscale images |
| `arxiv-2503.05037` | English | 17 | The same PDF as in the main benchmark (`benchmark/pdfs/`, get it with `benchmark/fetch_pdfs.py`) |

## Questions (20)

| doc | hi | en | hinglish | mr | unanswerable (included in the counts) |
|---|---:|---:|---:|---:|---:|
| hi-tajmahal | 4 | 2 | 2 | – | 1 (hi) |
| mr-shaniwarwada | 1 | 2 | – | 4 | 1 (mr) |
| arxiv-2503.05037 | 2 | – | 1 | – | – |
| hi-scanned | 2 | – | – | – | – |

`q_id`s start with `ml-` so they never clash with the main set. For `hi-scanned`,
`evidence_pages` are pages within the scanned PDF. `documents.json` maps them to the
source pages through `derived_from` and `derived_pages`.

## Files

| File | Purpose |
|---|---|
| `sources/*.txt` | Plain-text Wikipedia extracts (`action=query&prop=extracts&explaintext=1`), stored here because Wikipedia articles change |
| `make_pdfs.py` | Builds `pdfs/*.pdf` from `sources/`. It runs LibreOffice headless for the text PDFs, then uses PyMuPDF to add footers, fix the metadata and build the image-only copy. Running it twice gives byte-identical files on the same machine |
| `documents.json` | The manifest (main schema plus `language`, `source_text`, `derived_from`, `derived_pages`, `pdf_location`) |
| `questions.jsonl` | The questions (main schema plus `question_language`: `en`, `hi`, `hinglish` or `mr`) |
| `validate.py` | Checks the schema, the files, page counts and SHA-256, that every fact is on its evidence pages, and that `hi-scanned` has no text layer |
| `pdfs/` | Generated PDFs. Ignored by git |

```bash
.venv/bin/python benchmark/multilingual/make_pdfs.py   # needs soffice + Noto Devanagari fonts
.venv/bin/python benchmark/fetch_pdfs.py               # for arxiv-2503.05037
.venv/bin/python benchmark/multilingual/validate.py
```

### Why the PDFs are generated and not downloaded
- Wikipedia's `/api/rest_v1/page/pdf/<title>` endpoint still works, but it renders the PDF fresh on every request. The timestamps change, so the SHA-256 changes each time.
- Its text layer (Chrome/Skia) also duplicates Devanagari vowel signs when extracted ("सम्रााट", "स्थाान").
- PyMuPDF's own `Story`/`insert_htmlbox` draws Devanagari correctly on the page, but `get_text()` returns garbage ("उǴर Śदेश").
- LibreOffice's export reads back correctly. The one quirk: an i-matra word is sometimes split by a newline ("स्थि\nत"). Facts avoid such words, and `validate.py` warns if a fact matches only after whitespace is removed.

## Licences
- Wikipedia text: © Wikipedia contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes made: plain text only (no infobox, images, tables or references), rendered to PDF. The attribution is printed in every page footer. If you share the PDFs or `sources/`, keep the attribution and the licence.
- arXiv paper: see the main benchmark. It is not redistributed.

## Limitations
- Very small (20 questions, 4 documents). Treat results as a smoke test, not as statistically meaningful.
- One annotator, who is not a native Marathi speaker. The Hinglish spellings are one plausible choice among many.
- Grading by alias is lenient for short numbers (e.g. `5` or `9` can appear by accident) and strict for paraphrases. A correct answer that uses a spelling not in the list fails. An LLM judge or a human check is recommended for a final score.
- Single-digit and transliterated aliases (e.g. `saath`, `bees hazar`) were chosen by hand and are not exhaustive.
- The documents are clean, single-column, generated text. They do not cover real-world Indic PDF problems such as legacy non-Unicode fonts (Kruti Dev), broken ToUnicode maps or noisy scans. `hi-scanned` is a clean render, so it is the easy case for OCR.
- If you rebuild with a different LibreOffice or font version, the page breaks may move. `validate.py` then reports a SHA-256 mismatch and shows whether the evidence pages still hold.

## Changelog

* Grading fix: `ml-hi-tajmahal-q04` gained the aliases "a thousand", "1,000", "हज़ार"/"हजार". A
  correct answer ("over a thousand elephants") was being scored wrong. This was found by reading the
  failures of the final run, so results before and after this date differ by this one grader fix.
