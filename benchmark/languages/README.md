# Many-language PDF-QA benchmark (v0)

A many-language companion to `benchmark/multilingual/` (Hindi/Marathi). It is built the same way
and uses the same schema. It covers 22 Wikipedia languages, 24 question languages
(including Hinglish and Roman Urdu) and 8 kinds of question.

* **Dev set** (`./`): 22 short documents (1–2 PDF pages each), 14 image-only "scanned" copies, 122 questions.
* **Held-out set** (`heldout/`): 6 other documents (sa, bh, ur, te, ta, fr) and 20 questions. It is
  **sealed**: run it once, at the very end. `heldout/SEALED.txt` holds the SHA-256 of its
  `questions.jsonl`. Do not read the held-out questions while you tune the system.

## What each `combo` tests

| combo | What it tests | doc_ids searched |
|---|---|---|
| `same` | A question in the document's own language | that doc |
| `en-on-<lang>` | An English question on a document in `<lang>` | that doc |
| `cross` | A question in a different non-English language (Urdu on Hindi, Telugu on Sanskrit, Hinglish on Kannada, Roman Urdu on Marathi, Arabic on Urdu, ...) | that doc |
| `unanswerable` | Plausible but not in the text. The system should say so | that doc |
| `answer-in` | The question asks for an answer in another language ("Answer in Urdu."). `answer_language` is the requested one | that doc |
| `library` | A variant of the `same` question that searches all 22 text docs. The system must find the right one | `"all"` |
| `two-books` | Needs one fact from each of two documents in different languages | `"all"` |
| `scanned` | The `same` question on the image-only copy (needs OCR) | the `-scan` doc |

## Coverage (dev set, 122 questions)

Questions by **document** language. A two-books question counts once for each of its two
documents, so the `total` column leaves two-books out.

| Doc language | same | en-on | cross | unanswerable | answer-in | library | scanned | two-books | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hi Hindi | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 7 |
| mr Marathi | 1 | 1 | 1 | – | 1 | 1 | – | 1 | 5 |
| sa Sanskrit | 1 | 1 | 1 | 1 | – | 1 | 1 | – | 6 |
| ne Nepali | 1 | 1 | 1 | – | – | 1 | – | – | 4 |
| bh Bhojpuri | 1 | 1 | 1 | 1 | – | 1 | – | – | 5 |
| awa Awadhi | 1 | 1 | 1 | – | 1 | 1 | – | – | 5 |
| mai Maithili | 1 | 1 | 1 | 1 | – | 1 | – | – | 5 |
| ur Urdu | 1 | 1 | 1 | – | – | 1 | 1 | 2 | 5 |
| bn Bengali | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 7 |
| te Telugu | 1 | 1 | 1 | – | 1 | 1 | 1 | 2 | 6 |
| ta Tamil | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 7 |
| kn Kannada | 1 | 1 | 1 | – | 1 | 1 | 1 | – | 6 |
| ml Malayalam | 1 | 1 | 1 | 1 | – | 1 | 1 | – | 6 |
| gu Gujarati | 1 | 1 | 1 | – | – | 1 | 1 | 1 | 5 |
| pa Punjabi | 1 | 1 | 1 | 1 | – | 1 | 1 | – | 6 |
| or Odia | 1 | 1 | 1 | – | – | 1 | 1 | – | 5 |
| en English | 1 | – | 1 | 1 | 1 | 1 | – | 1 | 5 |
| fr French | 1 | 1 | 1 | – | – | 1 | – | – | 4 |
| ar Arabic | – | – | – | – | – | – | – | – | 0 (see Findings) |
| zh Chinese | 1 | 1 | 1 | 1 | – | 1 | 1 | – | 6 |
| ja Japanese | 1 | 1 | 1 | – | – | 1 | 1 | 1 | 5 |
| ru Russian | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 | 6 |
| **questions** | 21 | 20 | 21 | 11 | 8 | 21 | 14 | 6 | 122 |

Questions by **question** language: en 31, hi 12, bn 7, te 7, ur 5, ta 5, zh 5, ru 5, mr 4,
kn 4, gu 4, fr 4, ja 4, sa 3, ne 3, ml 3, pa 3, or 3, bh 2, awa 2, mai 2, hinglish 2, ar 1,
roman-ur 1. The answer-in requests are ur, mr, ta, bn, hi, en, te and fr (one each).

## Files

| File | Purpose |
|---|---|
| `config.json`, `heldout/config.json` | Build input. One entry per document: language, article title, page id, revision id, revision timestamp, whether to make a scanned copy |
| `sources/*.txt` | Plain-text Wikipedia extracts (`action=query&prop=extracts&explaintext=1`). The first ~2,000–2,900 characters (~1,100–1,700 for zh/ja) were kept, cut at a paragraph boundary, with headings |
| `make_pdfs.py` | Builds `pdfs/` and writes `documents.json` (`--heldout` for the held-out set, `--set heldout2` / `--set special` for the extra sets). Running it again gives byte-identical files on the same machine |
| `documents.json` | The manifest: main schema plus `language`, `source_text`, `page_id`, `rev_id`, `roundtrip_score`, `roundtrip_ok`, and `derived_from`/`derived_pages` for scanned copies |
| `questions.jsonl` | The questions (schema below) |
| `.gitignore` | Keeps the generated `pdfs/`, `heldout/pdfs/`, `heldout2/pdfs/` and `special/pdfs/` out of git, as in the 2-language set |
| `validate.py` | Checks the schema and combo rules, the SHA-256 and page counts, the round-trip, that scanned copies have no text layer, and that every fact is on its evidence pages (using the grader's own `alias_matches`) |

```bash
.venv/bin/python benchmark/languages/make_pdfs.py             # needs soffice + Noto fonts
.venv/bin/python benchmark/languages/validate.py
.venv/bin/python benchmark/languages/make_pdfs.py --heldout   # only for the final sealed run
.venv/bin/python benchmark/languages/validate.py --heldout
```

## Extra sets (added after the first run)

Built the same way (same schema, evidence pages computed from the real PDF text, checked by
`validate.py`). `run.py` still knows only `--set dev|heldout`.

| Path | What it is |
|---|---|
| `questions_more.jsonl` | 64 more questions on the 22 dev documents (q_id `lm-`): 42 `cross` (2 per document, none on `ar`), 10 `two-books`, 6 `romanized`, 6 `answer-in` (answers requested in gu, kn, mr, ur, hi, te). Specs in `specs_more.py` |
| `heldout2/` | A **second sealed** held-out set: 10 new documents (hi Taj Mahal, mr Shaniwar Wada, ne Pashupatinath, awa Lucknow, mai Janakpur, bn Victoria Memorial, kn Jog Falls, ml Idukki Dam, or Chilika, ja Himeji Castle), image-only copies of hi, bn, kn and ja, 48 questions. `heldout2/SEALED.txt` holds the SHA-256. Do not read `heldout2/questions.jsonl` or `heldout2/specs.py` while you tune the system |
| `special/` | `sp-hien-indiagate`: one PDF of alternating Hindi and English paragraphs (India Gate, from the hi and en articles; `para_langs` in `config.json`), with 3 `mixed` questions. `sp-ar-burjkhalifa`: the dev Arabic text printed with headless Chromium instead of LibreOffice. No questions (see below) |
| `build_more.py` | Writes the three question files above from their specs. It never touches `questions.jsonl` or `heldout/` |

New combos: `romanized` (a Latin-script question on a document in another script: `hinglish`,
`roman-ur`, and the new codes `roman-ta` and `roman-bn`) and `mixed` (en, hi or hinglish on the
`hi-en` document). `--set more` also refuses q_ids or questions that are already in `questions.jsonl`.

```bash
.venv/bin/python benchmark/languages/validate.py --set more
.venv/bin/python benchmark/languages/make_pdfs.py --set special && .venv/bin/python benchmark/languages/validate.py --set special
.venv/bin/python benchmark/languages/make_pdfs.py --set heldout2   # only for the final sealed run
.venv/bin/python benchmark/languages/validate.py --set heldout2
.venv/bin/python benchmark/languages/build_more.py more|heldout2|special   # regenerate a question file
```

What these sets found:
- **Chromium does not fix Arabic.** Printed with headless Chromium (`--no-sandbox` is needed on
  this machine), the Arabic text reads back at 0.837 (0.841 as `validate.py` measures it), worse
  than LibreOffice (0.942). Each line comes back in visual order: numbers and brackets jump to
  the other end of the line ("(2010 ... ( ُبرج خليفة"). The threshold for this document is 0.97,
  so it has no questions.
- **A doubled Devanagari letter.** In the mixed PDF, "जनवरी 2022" reads back as "जनववरी 2022"
  (the same kind of fault as the Telugu one above). The Hinglish question therefore matches only
  the year.
- In `heldout2/`, 5 facts match only after whitespace is removed (a hyphenated Devanagari word,
  one Kannada, one Malayalam and two Odia words split by the text layer). `validate.py` shows
  them as warnings.
- Legacy-font (Kruti Dev) PDFs are not tested: they would need a font download.

### Question schema
Same as `benchmark/multilingual/questions.jsonl`, plus:
- `answer_language`: the language the answer should be in. It equals `question_language`
  except for `answer-in`.
- `doc_ids`: the documents to search, either a list or `"all"`. `"all"` means every text
  (non-scanned) document of the same set.
- `evidence`: a list of `{"doc_id", "pages"}`. For single-document questions, `doc_id` and
  `evidence_pages` are filled in too. For `two-books`, `doc_id` and `evidence_pages` are `null`.
- `combo`: the tag from the table above.

Language codes are the Wikipedia codes, plus `hinglish` (Hindi in Latin letters),
`roman-ur` (Urdu in Latin letters), `roman-ta` and `roman-bn` (Tamil and Bengali in Latin letters;
extra sets only) and the document language `hi-en` (the mixed document in `special/`). `bh` is Bhojpuri (bh.wikipedia.org). For scanned copies,
`evidence_pages` are pages of the scanned PDF. They have the same numbers as the source PDF,
because every page was scanned.

`required_facts` are groups of aliases, matched the way `local_agentic_rag.eval.metrics`
matches them (NFKC, lowercase, Devanagari/Bengali/Gujarati digits folded to ASCII, thousands
commas dropped, numbers matched as whole numbers). The grader does **not** fold Gurmukhi, Odia,
Telugu, Kannada, Tamil or Arabic-Indic digits. Where they matter, native-digit aliases (`੧੫੮੮`,
`୨୪`, `౧౫౯౧`, `೧೯೧೨`, `۱۶۷۳`) are listed explicitly.

## How the PDFs are built (and what the round-trip found)
Text → HTML → LibreOffice headless PDF → PyMuPDF adds a Latin footer (with the attribution and
the page/revision id) and frozen metadata. This is the same method as `benchmark/multilingual/`.
The font is chosen per script in CSS, with `lang` set, and `dir="rtl"` for ur/ar. Each text
PDF is checked by reading it back: `get_text()`, NFC, whitespace removed, then the share of
source characters recovered in order. The threshold is 0.95.

### Findings
- **Arabic does not round-trip (0.942).** Lines that mix Arabic with digits come back in the
  wrong order ("متراً" comes back as "ار مت ً"). Only 17 of 33 sentences come back verbatim.
  Naskh, Sans, alignment and `dir` variants all gave 0.94. `ar-burjkhalifa` stays in the
  library as a distractor but has **no questions**. Its scanned copy is built but not used. Arabic
  still appears as a question language (Arabic question on the Urdu doc).
- **Urdu in Noto Nastaliq does not round-trip (0.44).** The text comes back with doubled
  letters and broken order. Set in Noto Naskh Arabic it reads back at 0.99, so Urdu uses Naskh.
- **Telugu doubles a vowel sign in one word.** "గోలకొండ" comes back as "గోోలకొండ", so an exact
  search for Golconda in the Telugu PDF fails. The question that used it was reworded.
- **Japanese gets spaces between digits and kanji.** "2012年2月29日" comes back as
  "2012 年2 月29 日", and "日建設計" is split by a line break. An Odia word is split before a
  nukta ("ପାଗୋ ଡ଼ା"). These facts match only once whitespace is removed, and `validate.py`
  shows them as warnings. A system that does exact or BM25 matching on these strings will miss them.
- **Odia rendering defect (visual only).** LibreOffice with Noto Sans Oriya draws reph + i-matra
  wrongly: "ନିର୍ମିତ" shows as "ନିମ ି ତ". The text layer is correct (0.9975), but `or-konark-scan`
  inherits the broken glyphs, so OCR on that word is harder. The scanned question (24 wheels, "୨୪ଟି")
  does not depend on it. Spot checks of te, kn, bn, gu, ta, ml, pa, hi, sa, ur and ja pages showed no
  shaping problems.
- Devanagari i-matra words are sometimes split by a newline ("तेन्जि\nङ"), as in the 2-language set.
- All other scripts read back at ≥ 0.997, and the held-out set at ≥ 0.955.

## Licence and attribution
- Wikipedia text: © Wikipedia contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Changes: plain text only, the first part of each article (cut at a paragraph boundary), no
  infobox, images, tables or references, rendered to PDF. Each page footer gives the
  attribution, with the page id and revision id. `documents.json` links every exact revision
  (`url`). If you share the PDFs or `sources/`, keep the attribution and the licence.

## Limitations
- Small: 1 document per language and 1–7 questions per document. Treat the results as
  per-language smoke tests, not as statistically meaningful scores.
- One annotator, who is not a native speaker of most of these languages. Questions in Sanskrit,
  Awadhi, Maithili, Bhojpuri, Odia, Malayalam, Arabic, Japanese and Russian are kept short and
  simple. The facts were checked against the text, but the wording may not be idiomatic.
- The low-resource Wikipedia articles (awa, bh, mai, sa) contain some non-standard spelling and
  mixed registers. The text is kept exactly as published.
- Matching by alias is lenient for short numbers and strict for paraphrases. Transliterated
  aliases (e.g. Telugu "ఇర్విన్" for Irwin, Bengali "লেসলি" for Leslie) are one plausible spelling
  each, not a full list. An LLM judge or a human check is recommended for a final score.
  Whether the answer is in the right language (`answer_language`) needs its own check, for
  example a language-ID model. Matching by alias cannot tell.
- The documents are clean, generated text. Scanned copies are clean 200-dpi renders, which is
  the easy case for OCR.
- Rebuilding with another LibreOffice or font version can move page breaks. `validate.py`
  then reports a SHA-256 mismatch and shows whether the evidence pages still hold.
