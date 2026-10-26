# Real-document QA evaluation set (`realdocs`)

An independent question set for testing a local "chat with your PDF" system on
long, real, public-domain US government PDFs. The questions were written to
reflect what a real reader of each document would ask. They were **not** tuned
to any system: the author did not look at this repository's code, other
benchmarks, or earlier results while writing them.

## Files

| File | Contents |
|---|---|
| `documents.json` | The four source documents: `doc_id`, title, official PDF URL, approximate page count, licence note |
| `questions.jsonl` | One question per line (schema below) |

## Coverage

| `doc_id` | Questions | Unanswerable | Notes |
|---|---|---|---|
| `uscode-2023-title26` | 15 | 3 | Full coverage, from the 2023-edition HTML |
| `statab-2012` | 16 | 3 | Table-locator and section-scope questions only (see Limitations) |
| `gpo-911report` | 15 | 3 | Narrative, timeline and recommendation questions |
| `budget-2026-appendix` | 0 | 0 | **Not covered.** See Limitations |

That is 46 questions, 9 of them (about 20%) unanswerable. Types:
`lookup` 15, `section_ref` 11, `definition` 4, `numeric` 4, `multi_hop` 3,
`unanswerable` 9. There are no `table_cell` questions.

## Question schema

- `q_id`, `doc_id`, `question`, `type`, `answerable`
- `reference_answer`: a short gold answer.
- `answer_keys`: short strings that must **all** appear in a correct answer.
  They are copied as the source writes them (e.g. `"$250,000"`, `"18.3 cents"`,
  `"9:58:59"`). For `numeric` questions the key is the computed result.
- `numeric_answer`: a number, or `null`.
- `evidence`: one or two **verbatim** quotes (mostly 8 to 25 words) from the
  official text that holds the answer. For Statistical Abstract table
  questions, the quote is the table title.
- `position_hint`: `beginning`, `middle` or `end`. This is the author's
  estimate of where the answer sits in the PDF.
- `source_url`: the HTML page the quote was read from.
- `why_unanswerable` (only when `answerable` is false): what the document
  lacks, and how that was checked. Unanswerable rows have empty
  `answer_keys` and `evidence`.

## How the set was built

1. The PDFs were **not** downloaded on the authoring machine. Text was read
   with a web-fetch tool from official HTML editions of the same content:
   - Title 26: govinfo's `USCODE-2023-title26` HTML, reached through
     `https://www.govinfo.gov/link/uscode/26/<section>?year=2023&link-type=html`.
     The current text on uscode.house.gov was deliberately **not** used,
     because 2025 legislation (Pub. L. 119-21) amended many sections after the
     2023 edition.
   - 9/11 Report: the chapter HTML pages on `www.9-11commission.gov/report/`.
   - Statistical Abstract 2012: the Census Bureau's section pages for the
     131st edition. These pages give each section's introduction and its table
     numbers and titles.
2. Each evidence quote was taken one sentence at a time. It was then checked
   with a second, differently worded "does the page contain this exact string"
   request.
3. Each unanswerable question was checked against the relevant chapter, section
   or table list. Most rely on a date the document cannot cover, such as events
   or laws after its publication, or data years after the edition was compiled.
4. Questions were spread across each document's front, middle and back, so that
   accuracy can be measured against position.

## Gold pages are located automatically

The questions do not record page numbers. On the machine that has the PDFs, a
gold page is found by searching the extracted PDF text for each `evidence`
quote. When a question has two quotes, both pages are gold.

Normalise before matching:
- Treat line breaks and repeated whitespace as one space.
- Treat curly and straight quotes and apostrophes as the same.
- Treat en dashes, em dashes, `--` and `-` as the same.
- Allow line-break hyphenation.

Two Title 26 quotes cross a subparagraph marker that sits on its own line in
the source (`uscode26-q02`, `uscode26-q09`). Statistical Abstract titles may be
printed with a `Table N.` prefix. Use fuzzy matching. Report any quote that
cannot be found instead of dropping it silently: a missing quote means the
quote is wrong, or the PDF layout differs from the HTML.

## Licence

All four documents are works of the United States Government. They are not
subject to copyright in the United States under 17 U.S.C. 105. The questions,
answers and short quotes in this folder may be redistributed. A few
Statistical Abstract tables reproduce data from non-federal sources. Those
sources may carry their own terms, but this set quotes only table titles and
section introductions.

## Limitations

- **Budget Appendix: no questions.** govinfo and whitehouse.gov publish the
  FY2026 Appendix only as PDF, whole or per agency. govinfo's help page says
  budget documents are available as PDF and spreadsheets, and no HTML or text
  edition was found. Since the authoring rule banned downloading PDFs, its
  text could not be read verbatim. Web-search snippets were tried and were not
  reliable enough for verbatim evidence. The document stays in
  `documents.json` so it can be ingested as a large distractor. To add its 15
  questions, write them from text extracted from the PDF itself, for example
  on the GPU machine, or from the per-agency PDFs.
- **Statistical Abstract: no cell values.** The Census site offers individual
  tables only as XLS, and sections as PDF. Neither could be opened under the
  no-file rule, and the HTML pages have no data values. Its questions
  therefore test *finding the right table* (number and data year from the
  title) and *section scope* (from the section introductions). There are no
  `table_cell` or `numeric` questions for this document. The table titles were
  taken from the HTML listings, which have a few typos (e.g. "Clamydia",
  "DailyTobacco"), so the chosen titles avoid those. The printed PDF titles may
  still differ slightly in punctuation.
- **Position hints are estimates.** They come from section, chapter and table
  order, not from PDF page numbers. Statistical Abstract positions are
  estimated from table numbers (about 1,400 tables across about 1,040 pages).
  Title 26 positions follow section order: subtitle A is the first part of the
  PDF, and the procedure and administration provisions (§§6001-7874) are near
  the end.
- **Verbatim checks used an LLM-based page reader**, not a character-level
  diff, so a quote can still differ slightly from the PDF. The automatic
  gold-page step above is the final check.
- **HTML and PDF can differ in typography**: em dashes, "a.m." versus "A.M.",
  and footnote markers. Quotes were chosen to avoid these where possible.
- **Unanswerable questions mostly rest on dates.** Eight of the nine depend on
  a year or law the document cannot cover. The ninth (`911-q05`, the
  temperature on the morning of 9/11) asks for a detail the report never gives.
  Several are designed to tempt a system into answering with a nearby fact:
  `uscode26-q04` (the 2024 IRA limit; the code states only the $5,000 base
  amount), `statab-q14` and `statab-q15` (earlier election and park-visit
  years), and `statab-q16` (population projections are not a 2020 Census
  count). Graders should not accept those near-miss answers as correct.
