"""
Real long documents: download, locate gold pages from verbatim evidence quotes, then measure
ingestion, retrieval and answers exactly like ``run_scale.py``.

    python benchmark/scale/run_realdocs.py --realdocs benchmark/realdocs \
        --llm ollama:qwen2.5:7b-instruct --setup-ollama --out local/realdocs

The question set (``benchmark/realdocs/questions.jsonl``) was written from the official
HTML/text editions without looking at the PDFs or at this system's results.  Each answerable
question carries 1-2 verbatim evidence quotes; its gold pages are the PDF pages whose
extracted text contains a quote (whitespace, case, dashes and line-break hyphens
normalised).  Questions whose quote cannot be found are kept for answer scoring but left out
of page-based metrics, and counted in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_scale as rs  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # line-break hyphenation
    text = re.sub(r"[‐‑‒–—―−]", "-", text)
    text = re.sub(r"[‘’´`]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    return re.sub(r"\s+", " ", text).strip()


def download(url: str, dest: Path) -> dict[str, Any]:
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        t = time.perf_counter()
        with (
            urllib.request.urlopen(req, timeout=600) as r,
            open(dest.with_suffix(".part"), "wb") as f,
        ):
            while chunk := r.read(1 << 20):
                f.write(chunk)
        dest.with_suffix(".part").rename(dest)
        print(f"  downloaded {dest.name} in {time.perf_counter() - t:.0f}s", flush=True)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {"bytes": dest.stat().st_size, "sha256": sha}


def page_texts(pdf: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        return [norm(page.get_text()) for page in doc]


def locate(quote: str, pages: list[str]) -> list[int]:
    q = norm(quote)
    hits = [i + 1 for i, t in enumerate(pages) if q in t]
    if hits or len(q.split()) < 8:
        return hits
    # Layout can split a quote across columns or lines differently: accept a page holding
    # every 6-word window of the quote's first and last 6 words.
    words = q.split()
    head, tail = " ".join(words[:6]), " ".join(words[-6:])
    return [i + 1 for i, t in enumerate(pages) if head in t and tail in t]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--realdocs", default=str(rs.REPO / "benchmark/realdocs"))
    ap.add_argument("--pdf-dir", default=str(rs.REPO / "local/realdocs/pdfs"))
    ap.add_argument("--store-dir", default=str(rs.REPO / "local/realdocs/store"))
    ap.add_argument("--out", default=str(rs.REPO / "local/realdocs/results"))
    ap.add_argument("--docs", nargs="*", default=None)
    ap.add_argument("--llm", default=None)
    ap.add_argument("--setup-ollama", action="store_true")
    ap.add_argument("--ollama-host", default="127.0.0.1:11434")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import os

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    llm = None
    if args.llm and args.setup_ollama:
        from local_agentic_rag.colab import setup_llm

        llm = setup_llm(args.llm.partition(":")[2], num_ctx=8192, host=args.ollama_host)
    elif args.llm:
        from local_agentic_rag.llm.ollama import OllamaLLM

        llm = OllamaLLM(args.llm.partition(":")[2], host=f"http://{args.ollama_host}")
        llm.ensure_ready()
    from local_agentic_rag.config import RAGConfig
    from local_agentic_rag.pipeline import RAGPipeline

    pipe = RAGPipeline(RAGConfig(), llm=llm, cache_dir=None)
    pipe.load_models()
    env = rs.environment()

    rd = Path(args.realdocs)
    docs = json.loads((rd / "documents.json").read_text())
    questions = [
        json.loads(x) for x in (rd / "questions.jsonl").read_text().splitlines() if x.strip()
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for d in docs:
        doc_id = d["doc_id"]
        if args.docs and doc_id not in args.docs:
            continue
        path = out / f"{doc_id}.json"
        if path.is_file() and not args.force:
            results.append(json.loads(path.read_text()))
            continue
        print(f"[{doc_id}] download", flush=True)
        pdf = Path(args.pdf_dir) / f"{doc_id}.pdf"
        try:
            info = download(d["url"], pdf)
        except Exception as exc:  # recorded, never faked
            record = {"doc_id": doc_id, "error": f"download failed: {exc}"}
            path.write_text(json.dumps(record, indent=1))
            results.append(record)
            continue
        texts = page_texts(pdf)
        qs = [dict(q) for q in questions if q["doc_id"] == doc_id]
        unlocated = []
        for q in qs:
            if not q.get("answerable", True):
                q["gold_pages"] = []
                continue
            pages = sorted({p for quote in q.get("evidence", []) for p in locate(quote, texts)})
            q["gold_pages"] = pages
            q["depth"] = (pages[0] / len(texts)) if pages else None
            if not pages:
                unlocated.append(q["q_id"])
        print(
            f"[{doc_id}] {len(texts)} pages; {len(qs)} questions, {len(unlocated)} not located",
            flush=True,
        )
        ing = rs.run_ingest(pipe, pdf, Path(args.store_dir) / doc_id, args.workers)
        index = ing["index"]
        print(
            f"[{doc_id}] ingest {ing['result']['wall_seconds']}s, {ing['result']['num_chunks']} chunks",
            flush=True,
        )
        located = [q for q in qs if q.get("gold_pages")]
        record: dict[str, Any] = {
            "doc_id": doc_id,
            "title": d.get("title"),
            "url": d["url"],
            "pages": len(texts),
            "download": info,
            "num_questions": len(qs),
            "unlocated_questions": unlocated,
            "environment": env,
            "ingest": ing["result"],
            "retrieval": rs.run_retrieval(pipe, index, located) if located else None,
            "measured": True,
        }
        if llm is not None:
            record["answers"] = rs.run_answers(pipe, index, qs)
            a = record["answers"]
            print(
                f"[{doc_id}] answers acc={a['accuracy']} refusal={a['unanswerable_refused']}",
                flush=True,
            )
        path.write_text(json.dumps(record, indent=1, default=str))
        results.append(record)
        pipe.indexes.clear()
    lines = [
        "# Real long documents",
        "",
        "| Document | Pages | Ingest (s) | Chunks | Peak RAM (MB) | Questions (located) "
        "| Recall@5 | MRR | Answer acc. | Refusal (unans.) | Cit. precision | Answer p50 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['doc_id']} | {r['error']} |" + " |" * 10)
            continue
        ret = (r.get("retrieval") or {}).get("overall") or {}
        a = r.get("answers") or {}

        def f(x):
            return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

        lines.append(
            f"| {r['doc_id']} | {r['pages']:,} | {r['ingest']['wall_seconds']:.0f} | "
            f"{r['ingest']['num_chunks']:,} | {r['ingest']['peak_rss_mb']:.0f} | "
            f"{r['num_questions']} ({r['num_questions'] - len(r['unlocated_questions'])}) | "
            f"{f(ret.get('recall@5'))} | {f(ret.get('mrr'))} | {f(a.get('answerable_accuracy'))} | "
            f"{f(a.get('unanswerable_refused'))} | {f(a.get('citation_precision'))} | "
            f"{f(a.get('latency_p50'))} |"
        )
    (out / "realdocs_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
