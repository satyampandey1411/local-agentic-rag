#!/usr/bin/env python3
"""Download the benchmark PDFs listed in documents.json and verify their SHA-256.

Standard library only, so it runs unchanged on a laptop or in Google Colab:

    python benchmark/fetch_pdfs.py                     # -> benchmark/pdfs/
    python benchmark/fetch_pdfs.py --dest /content/pdfs
    python benchmark/fetch_pdfs.py --force             # re-download everything

Each file is saved as ``<doc_id>.pdf``. A file that already exists with the
expected hash is skipped. Downloads go to a temporary ``.part`` file and are only
moved into place after the hash matches, so a failed or tampered download never
leaves a file that looks valid.

Exit code is 0 only if every document ends up present and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "documents.json"
DEFAULT_DEST = HERE / "pdfs"

# Some government sites reject the default "Python-urllib" agent.
USER_AGENT = (
    "Mozilla/5.0 (compatible; local-agentic-rag-benchmark/1.0; "
    "+https://github.com/) Python-urllib"
)
CHUNK = 1 << 16


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, out: Path, timeout: float, retries: int) -> None:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp, out.open("wb") as f:
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    attempt {attempt} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"download failed after {retries} attempts: {last_err}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"path to documents.json (default: {DEFAULT_MANIFEST})")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"directory to store PDFs (default: {DEFAULT_DEST})")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if a verified file exists")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-request timeout (s)")
    ap.add_argument("--retries", type=int, default=3, help="attempts per file")
    args = ap.parse_args(argv)

    docs = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.dest.mkdir(parents=True, exist_ok=True)

    rows = []  # (doc_id, status, detail)
    for doc in docs:
        doc_id, url, expected = doc["doc_id"], doc["url"], doc["sha256"].lower()
        target = args.dest / f"{doc_id}.pdf"
        print(f"[{doc_id}] {url}")

        if target.exists() and not args.force:
            got = sha256_of(target)
            if got == expected:
                rows.append((doc_id, "skipped", "already present, sha256 ok"))
                print("    already present, sha256 ok")
                continue
            print(f"    existing file has wrong sha256 ({got[:12]}...); re-downloading")

        part = target.with_name(target.name + ".part")
        try:
            download(url, part, args.timeout, args.retries)
            got = sha256_of(part)
            if got != expected:
                part.unlink(missing_ok=True)
                msg = f"sha256 mismatch: expected {expected[:12]}..., got {got[:12]}..."
                rows.append((doc_id, "FAILED", msg))
                print(f"    {msg}")
                continue
            os.replace(part, target)
            size_mb = target.stat().st_size / 1e6
            rows.append((doc_id, "downloaded", f"{size_mb:.2f} MB, sha256 ok"))
            print(f"    downloaded {size_mb:.2f} MB, sha256 ok")
        except Exception as e:  # noqa: BLE001 - report and continue with other docs
            part.unlink(missing_ok=True)
            rows.append((doc_id, "FAILED", str(e)))
            print(f"    FAILED: {e}")

    print("\nSummary")
    print("-" * 72)
    for doc_id, status, detail in rows:
        print(f"{doc_id:<22} {status:<11} {detail}")
    failed = [r for r in rows if r[1] == "FAILED"]
    print("-" * 72)
    print(f"{len(rows) - len(failed)}/{len(rows)} documents ready in {args.dest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
