# Live demo script (5 minutes)

What the demo shows, in order, and what to fall back on if something fails. Every number you
say on stage must come from [RESULTS.md](RESULTS.md) or [SCALE.md](SCALE.md) with its run date.

## Before the demo (allow 90 minutes)

1. Start a GPU session (Kaggle T4 or Colab T4) from `notebooks/Local_Agentic_RAG.ipynb`
   section 7b, or on a GPU machine:

   ```bash
   docker compose up --build          # or the pip + Ollama path in docs/DEPLOY.md
   ```

2. Pre-process the demo documents through the web app, so nothing waits on stage:
   * the 9/11 Commission Report (585 pages, about 1 minute),
   * the Statistical Abstract of the United States 2012 (940 pages, about 24 minutes),
   * the synthetic 10,000-page compendium `scale-10000.pdf` from
     `benchmark/scale/make_corpus.py` (about 71 minutes on 2 vCPUs; start it first).
3. Keep a 250-page PDF ready for the live upload (`scale-250.pdf` or any real report).
4. Ask one warm-up question so the model is loaded. Check that `/api/health` says the model is ready.
5. Record a screen capture of the whole script once. It is the fallback if the network or
   GPU session fails during the live slot.

## Script

| Time | Action | What to point out |
|---|---|---|
| 0:00 | Show the library: three processed documents and their page counts. | Everything runs on one free T4; nothing leaves the machine. |
| 0:30 | 9/11 Report: "How many crew members were killed in the USS Cole attack?" | The answer streams in, then the verified answer replaces it. Click `[1]`: the source page opens with the passage highlighted. |
| 1:15 | Same document, in Hindi: "यूएसएस कोल हमले में कितने नाविक मारे गए?" | The answer comes back in Hindi, with a citation into the English report. |
| 1:45 | 10,000-page document: "Which table is printed on page 4,512?" (or another page with a table) | Page routing. The table's printed number is named. Point at the latency. |
| 2:30 | Unanswerable: "What was the population of Veloria in 2031?" | An explicit "not in the document" instead of a guess. |
| 3:00 | Upload the 250-page PDF live. | Progress bar, pages done, ETA. After the first batch is indexed, ask about its first pages: the answer says "answered from the first n of N pages". |
| 4:00 | Open SCALE.md: the six-tier table and the recovery table. | Measured on a T4 at every size, with a crashed worker and an interrupted job resumed. |
| 4:40 | Close: install is one command; limitations are listed in STATUS.md. | |

## If something fails

* **Model not answering:** switch to the recorded capture.
* **GPU session lost:** `python -m local_agentic_rag.server --demo` shows the same interface with
  a fake backend (no GPU). Say clearly that it is the demo backend.
* **Slow answer (queue):** the UI shows the queue position; answers p50 is about 6 s, longer with
  several users (see SCALE.md load test).
