# Scale benchmark corpus

Synthetic long PDFs (100 to 10,000 pages) with ground-truth questions. Use them to
measure how ingestion time, memory, retrieval accuracy and citation accuracy change
as a document gets longer. Everything here is generated, so no PDFs are committed.

## Generate

```bash
# one tier
python benchmark/scale/make_corpus.py --pages 1000 --out local/scale --seed 7
# every tier: 100, 250, 1000, 2500, 5000, 10000
python benchmark/scale/make_corpus.py --pages all --out local/scale
# optional: add a few Hindi summary pages (needs a Devanagari TTF in the directory)
python benchmark/scale/make_corpus.py --pages 1000 --out local/scale --font-dir /usr/share/fonts/truetype/noto

python benchmark/scale/validate_corpus.py local/scale      # exits 1 on any failure
```

Each tier writes three files:

* `scale-N.pdf`, with exactly N pages.
* `scale-N.questions.jsonl`, with 40 questions.
* `scale-N.meta.json`, with generation time, peak RSS, file size, sha256,
  page-kind counts, the scanned (and Hindi) page numbers, and entity counts.

The same `--pages`/`--seed` gives a byte-identical PDF. `--font-dir` is opt-in, so
output does not depend on which fonts a machine has. If the directory has no
Devanagari font, the Hindi pages are skipped with a warning.

Measured on a shared 16-core machine under heavy load from other jobs (load
average about 30). "generation" is the in-process time recorded in the meta file.
Process wall time adds about 1-3 s for `import pymupdf`, which also loads the
`pymupdf_layout` ONNX models when that package is installed; about 140 MB of the
RSS is that import alone.

| pages | generation | process wall | peak RSS | PDF size | scanned pages |
|---|---|---|---|---|---|
| 100 | 0.7 s | 3.9 s | 184 MB | 0.58 MB | 2 |
| 1,000 | 2.7 s | 4.6 s | 190 MB | 5.0 MB | 15 |
| 10,000 | 13-32 s (load-dependent) | 35 s | 444 MB | 50 MB | 150 |

The document is built in 500-page parts joined with `insert_pdf`. A single
`Document` would take about 2 minutes at 10,000 pages, because
`Document.new_page` costs O(pages so far).

## What the document looks like

It is one coherent, fictional publication: the *Statistical and Administrative
Compendium 2010-2023* of the National Statistics and Development Agency (NSDA) of
the "Republic of Veloria". The document contains:

* A title page, then a printed table of contents for the first chapters. The full
  structure is in the PDF outline (bookmarks).
* One chapter per region. A chapter opens with a two-column introduction, then a
  regional overview with a ruled summary table. Then comes one numbered section
  per district:
  * an overview (population, area, seat, coordinator);
  * facilities (commissioning year, capacity, manager);
  * school statistics, written without naming the district;
  * a multi-year indicator trend in prose;
  * 0-2 ruled ward tables with a caption ("Table 12.4: ...");
  * development projects with codes like `RIP-4471-B`, sometimes at a facility in
    another district;
  * some free commentary.
* A running header on every page (publication title + chapter) and a "Page N"
  footer. Physical page number = printed page number.
* Footnotes on about 15-18% of pages. The two appendices are a two-column glossary
  and methodological notes.
* About 1.5% scanned pages, at least 2 per document. These are archival facility
  inspection certificates in a typewriter font. They are rasterised at 150 dpi to
  grey JPEG with a slight skew and noise, and inserted as image-only pages, so
  they exercise the OCR path.
* All names (regions, districts, wards, towns, facilities, people, project codes)
  come from seeded syllable and name vocabularies. Districts, regions, facilities,
  project codes and certificate numbers are unique within a document.

### What is synthetic, and what that means

* The prose comes from templates: several variants per paragraph type, filled
  with sampled numbers and names. At 10,000 pages the sentence structures repeat
  heavily, even though every fact is distinct. Real long documents vary more in
  vocabulary, discourse and layout.
* Numbers are internally plausible but not consistent across sections. Ward
  table values are not reconciled with district totals, for example.
* The layout is simpler than in real documents. It uses only Base-14 fonts
  (Times, Helvetica, Courier). There are no figures, charts, merged table cells,
  rotated tables, multi-page tables, or paragraphs split across pages. Paragraphs
  and tables are kept whole, so page bottoms are often ragged.
* A 10,000-page tier has about 770 "regions" and 10,000 "districts" (roughly one
  district profile per page). That is more administrative units than any real
  country, but it keeps every fact unique.
* About 58% of pages carry a ruled table. That is realistic for a
  statistical compendium, and it makes the system's `find_tables` pass a large
  part of ingestion cost.
* Scanned pages are clean digital renders with mild noise. They have no
  bleed-through, stains, handwriting or heavy skew.
* Optional Hindi pages: MuPDF's HTML shaping renders Devanagari correctly, but
  the extracted text layer is garbled (wrong ToUnicode mapping). Real PDFs fail
  the same way, but no questions are planted on these pages.

## Questions

Each tier has 40 questions, built from fact slots that are the same in every tier.
Every slot always uses the same question type, template and wording variant; only
the planted fact changes. Slot *i* targets document depth `(k + 0.5) / 40`, where
*k* is the slot's position in a fixed permutation, so the planted facts spread
evenly from 0% to 100% of the document. The `depth` field holds the actual
position: `(page - 1) / (N - 1)`, averaged over the gold pages.

| type | n | what is tested |
|---|---|---|
| `lookup` | 6 | a unique prose fact: district population or coordinator, facility commissioning year or manager, project budget, regional capital |
| `table_cell` | 6 | the value at a named row and column of a named table (ward tables, regional summary tables) |
| `numeric` | 5 | a computed answer: a change in percentage points within a series, a population sum over two districts on different pages, a capacity difference between two facilities on different pages. `operands` lists each stated value and its page |
| `multi_hop` | 4 | project code → facility (page A) → that facility's manager or commissioning year (page B ≠ A). `bridge` names the joining entity |
| `section_scoped` | 4 | school counts or pupil-teacher ratio. The sentence never names the district; only the section heading does |
| `page_ref` | 4 | "Which table is printed on page N?" / "Which district's profile begins on page N?" (gold also includes the TOC page when it lists that section) |
| `ocr` | 3 | certificate number, inspection date or compliant-item count, found only on a scanned page |
| `unanswerable` | 4 | a near-miss entity that does not exist: an unused district name one syllable away from a real one, a real district in a year outside 2010-2023, a real project code with a different suffix letter, or a real locality with a different facility type |
| `distractor` | 4 | one year of an indicator that the same district reports for 5-7 years, and that other districts report too |

JSONL fields:

* Required: `q_id`, `doc_id`, `question`, `type`, `answerable`,
  `reference_answer`, `gold_pages` (1-indexed physical pages), `answer_keys`
  (every one must appear in a correct answer), `numeric_answer`, `depth`.
* Extras: `kind`, `target_depth`, `table`/`row`/`column`, `section`, `bridge`,
  `operation` + `operands` (numeric), `ocr_text` (ocr),
  `absent_keys` / `absent_cooccur` / `near_miss` / `near_miss_pages` (unanswerable).

## Validation

`validate_corpus.py` checks the corpus using only PyMuPDF text extraction, with
whitespace normalised:

* exact page counts;
* scanned pages have no text layer;
* every answer key can be extracted from its gold page(s). For `numeric`
  questions it checks the operands and recomputes `numeric_answer`. For `ocr`
  questions it checks that the gold page is image-only and that the OCR-only
  text is absent from every text layer;
* table `page_ref` gold pages hold exactly one table caption;
* unanswerable `absent_keys` occur nowhere, and no page contains both an
  `absent_cooccur` entity and year (the "Page N" footer is ignored).

Tests: `pytest tests/test_scale_corpus.py`.
