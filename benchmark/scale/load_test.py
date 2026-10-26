"""
Load test for the web app: N concurrent users asking questions (and opening cited pages)
against a running server, e.g.

    python -m local_agentic_rag.server --library ~/rag-lib --port 8000 &
    python benchmark/scale/load_test.py --url http://127.0.0.1:8000 --users 1 2 4 8 \
        --questions local/scale/corpus/scale-1000.questions.jsonl --per-user 5

For each concurrency level it reports successful / failed / timed-out requests, latency
percentiles (end to end, including time queued behind other users), the server's reported
queue wait, and throughput in answers per minute.  The server limits concurrent answers
(``--max-concurrent-asks``), so with more users than slots latency grows while throughput
should stay flat: that is the expected behaviour of one GPU, and what this measures.

By default questions go through ``/api/ask/stream`` (what the web app uses), which also
gives ``first_token_s``: seconds from sending the request to the first ``token`` event
(null when the model streamed none).  ``--no-stream`` uses the plain ``/api/ask`` instead.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path

import httpx
import numpy as np


def iter_sse(lines):
    """Yield ``(event, data)`` from the text lines of a ``text/event-stream`` body.

    Comment lines (heartbeats, padding) are skipped; ``data`` is parsed JSON.
    """
    name, data = "message", []
    for line in lines:
        if line == "":
            if data:
                yield name, json.loads("\n".join(data))
            name, data = "message", []
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
    if data:
        yield name, json.loads("\n".join(data))


def _ask_stream(client: httpx.Client, url: str, payload: dict, timeout: float, t: float):
    """POST /api/ask/stream; return (status, body, first_token_s) with body None on failure."""
    first_token = None
    deadline = t + timeout
    with client.stream("POST", f"{url}/api/ask/stream", json=payload, timeout=timeout) as r:
        if r.status_code != 200:
            r.read()
            return r.status_code, None, None
        for name, data in iter_sse(r.iter_lines()):
            if name == "token" and first_token is None:
                first_token = time.perf_counter() - t
            elif name == "answer":
                return 200, data, first_token
            elif name == "error":
                return data.get("status", "error"), None, first_token
            if time.perf_counter() > deadline:
                raise httpx.ReadTimeout("overall deadline exceeded")
    return "no_answer", None, first_token


def ask_once(
    client: httpx.Client, url: str, question: str, doc_ids, timeout: float, stream: bool = True
) -> dict:
    t = time.perf_counter()
    first_token = None
    try:
        payload = {"question": question, "doc_ids": doc_ids, "history": []}
        if stream:
            status, body, first_token = _ask_stream(client, url, payload, timeout, t)
        else:
            r = client.post(f"{url}/api/ask", json=payload, timeout=timeout)
            status, body = r.status_code, (r.json() if r.status_code == 200 else None)
        latency = time.perf_counter() - t
        if body is None:
            return {"ok": False, "status": status, "latency": latency}
        page_s = None
        if body.get("citations"):
            c = body["citations"][0]
            t2 = time.perf_counter()
            img = client.get(
                f"{url}/api/documents/{c['doc_id']}/pages/{c['page']}.png",
                params={"chunk_id": c["chunk_id"]},
                timeout=timeout,
            )
            page_s = time.perf_counter() - t2 if img.status_code == 200 else None
        return {
            "ok": True,
            "status": 200,
            "latency": latency,
            "queue_wait": body.get("queue_wait_s"),
            "refused": body.get("refused"),
            "page_render": page_s,
            "first_token_s": first_token,
        }
    except httpx.TimeoutException:
        return {"ok": False, "status": "timeout", "latency": time.perf_counter() - t}
    except httpx.HTTPError as exc:
        return {"ok": False, "status": type(exc).__name__, "latency": time.perf_counter() - t}


def run_level(
    url: str,
    users: int,
    questions: list[str],
    per_user: int,
    doc_ids,
    timeout: float,
    stream: bool = True,
):
    results: list[dict] = []
    lock = threading.Lock()

    def user(seed: int) -> None:
        rng = random.Random(seed)
        with httpx.Client() as client:
            for _ in range(per_user):
                res = ask_once(client, url, rng.choice(questions), doc_ids, timeout, stream)
                with lock:
                    results.append(res)

    t = time.perf_counter()
    threads = [threading.Thread(target=user, args=(i,)) for i in range(users)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t
    return summarize(users, results, wall)


def summarize(users: int, results: list[dict], wall: float) -> dict:
    ok = [r for r in results if r["ok"]]
    lat = [r["latency"] for r in ok]

    def p(q, xs):
        return round(float(np.percentile(xs, q)), 2) if xs else None

    first_tokens = [r["first_token_s"] for r in ok if r.get("first_token_s") is not None]
    renders = [r["page_render"] for r in ok if r.get("page_render") is not None]
    waits = [r["queue_wait"] for r in ok if r.get("queue_wait") is not None]
    return {
        "users": users,
        "requests": len(results),
        "ok": len(ok),
        "failed": {
            str(s): sum(1 for r in results if not r["ok"] and r["status"] == s)
            for s in {r["status"] for r in results if not r["ok"]}
        },
        "latency_p50": p(50, lat),
        "latency_p95": p(95, lat),
        "latency_max": p(100, lat),
        "queue_wait_p50": p(50, waits),
        "first_token_p50": p(50, first_tokens),
        "first_token_p95": p(95, first_tokens),
        "first_token_count": len(first_tokens),
        "page_render_p50": p(50, renders),
        "answers_per_minute": round(len(ok) / wall * 60, 2),
        "wall_seconds": round(wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--users", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--per-user", type=int, default=3)
    ap.add_argument("--questions", required=True, help="JSONL with a 'question' field")
    ap.add_argument("--doc-ids", nargs="*", default=None)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument(
        "--no-stream", action="store_true", help="use /api/ask (no first_token_s) instead of SSE"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    questions = [
        json.loads(x)["question"]
        for x in Path(args.questions).read_text().splitlines()
        if x.strip()
    ]
    health = httpx.get(f"{args.url}/api/health", timeout=30).json()
    levels = [
        run_level(
            args.url, u, questions, args.per_user, args.doc_ids, args.timeout, not args.no_stream
        )
        for u in args.users
    ]
    report = {"server": health, "levels": levels}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
