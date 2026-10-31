# Held-out set 2 (each document used once)

70 questions on three documents that were never used while building or tuning the system. The
questions were written independently of the system: their authors read only the official sources
listed below, never this repository's code, benchmarks or results. The PDFs are not stored here;
`documents.json` gives each source URL and its terms. Gold pages are located automatically from
verbatim evidence quotes by `benchmark/scale/run_realdocs.py --realdocs benchmark/heldout2`;
questions whose quote is not found in the PDF are still scored on their answers and are counted in
the report.

`answer_keys`: every item must appear in a correct answer; an item that is itself a list is a set of
acceptable alternatives (for example a numeral or the spelled-out number, Western or Devanagari digits).

## US Code, 2024 edition, Title 21 (Food and Drugs): 30 questions

- Source read: the govinfo HTML edition. The single HTML file stops after about section 347, so
  later chapters were read from govinfo's per-section HTML pages of the same edition. No PDF was read.
- Mix: 9 questions naming a section, 8 lookups, 3 definitions, 3 numeric, 2 multi-hop, 5 unanswerable
  (2 about programmes the title does not contain, 1 about a law passed after the edition, 2 near-miss details).
- Evidence quotes are short (8 to 35 words); a few join two adjacent line fragments, so exact punctuation
  may differ from the PDF.
- Two unanswerable items (Medicare drug-price negotiation, retail dispensary licences) were judged absent
  from knowledge of the title, not by a full-text search.

## GAO Cost Estimating and Assessment Guide, GAO-20-195G (March 2020, 476 pages): 22 questions

- Source read: the text of the PDF at gao.gov (no HTML edition was reachable), covering the whole guide.
- Mix: 7 lookups, 3 definitions, 4 about specific tables, 3 numeric, 1 multi-hop, 4 unanswerable
  (2 about items that do not exist, such as a 13th step; 2 near-miss details).
- Quotes are short; a few table-cell quotes are under 8 words, and curly versus straight apostrophes
  were not verified. Page hints are approximate.

## CAG of India, Report No. 11 of 2019 (GST compliance audit), Hindi edition: 18 questions

- The test document is the Hindi PDF (190 pages). Questions were written from the English edition of the
  same report (cag.gov.in, report details 52307; Hindi details page 52342), using its Executive Summary,
  and are anchored on numbers, years, form names and acronyms so that answers can be checked in either language.
- Questions: 9 in Hindi, 4 in Hinglish, 5 in English. 14 answerable, 4 unanswerable.
- Evidence quotes are in English (`evidence_lang: "en"`), so gold pages cannot be located in the Hindi PDF:
  this document is scored on answers only.
- The 4 unanswerable questions were checked against the Executive Summary only (field
  `unanswerable_confidence`), not against every chapter, and are reported separately.
- The Hindi PDF's text layer is partly garbled (Devanagari with missing vowel signs); digits survive.
