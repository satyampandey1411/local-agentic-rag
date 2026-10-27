# Install and deployment

Three ways to run the web app (upload PDFs, ask questions, click a citation to see the page):

| Path | Best for | Command |
|---|---|---|
| Docker Compose | a GPU machine you control | `docker compose up --build` |
| pip + Ollama | a machine without Docker, or a Kaggle/cloud notebook | see "pip + Ollama" below |
| Colab notebook | trying it for free on a T4 | `notebooks/Local_Agentic_RAG.ipynb` |

All three run the same code. Everything stays on your machine: PDFs, indexes and questions are
never sent to a third-party service (the models are downloaded once from Hugging Face and the
Ollama registry).

## Requirements

* **GPU:** an NVIDIA GPU is expected. Measured on a Tesla T4: **8.9 GB of 15 GB** of GPU
  memory in use with every model loaded (the 4-bit `qwen2.5:7b-instruct` language model in
  Ollama, the BGE-M3 embedder and the bge-reranker-v2-m3 reranker in the app). Cards with less
  memory than that are untested. CPU-only operation is not measured and would
  be very slow for the 7B model.
* **Host RAM:** the free Colab T4 has about 12.7 GB, and the pipeline runs there.
* **Disk:** room for the models (the 7B model is a multi-GB download, plus the embedder and
  reranker) and for your library. A 10,000-page index was 188 MB (docs/SCALE.md); the PDFs are
  stored too.
* **Docker path:** Docker with Compose v2, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so containers can use the GPU.
* **pip path:** Python 3.10+ (the Docker image uses 3.12), `tesseract-ocr` with the `hin` and
  `mar` language packs for scanned pages (without it, scanned pages are skipped with a
  warning), and [Ollama](https://ollama.com).

## Docker Compose

```bash
git clone <this repository> && cd local-agentic-rag
docker compose up --build
# open http://127.0.0.1:8000/
```

What starts:

| Service | What it does |
|---|---|
| `ollama` | the language-model server, GPU reserved, models kept in the `ollama-models` volume |
| `ollama-pull` | one-shot: downloads `qwen2.5:7b-instruct`, then exits |
| `app` | the web app. Starts after the model is downloaded. Library, Hugging Face model cache and indexes live in the `rag-data` volume (`/data`) |

The first start downloads the language model and, on the first PDF, the embedding and reranking
models, so expect several minutes. Later starts reuse the volumes.

Check it works end to end (uploads a one-page PDF, waits for it, asks a question, requires a
citation, then deletes the test document):

```bash
scripts/smoke_install.sh http://127.0.0.1:8000
```

Useful commands: `docker compose logs -f app`, `docker compose down` (keeps your data),
`docker compose down -v` (deletes the library and the downloaded models).

**GPU with plain `docker run`.** The torch wheel installed by pip carries its own CUDA
libraries, so the image only needs the host driver, exposed with `--gpus all`:

```bash
docker build -t local-agentic-rag .
docker run --gpus all -p 127.0.0.1:8000:8000 -v rag-data:/data \
    --add-host host.docker.internal:host-gateway \
    -e LOCAL_RAG_OLLAMA_HOST=host.docker.internal:11434 \
    local-agentic-rag        # Ollama running on the host
```

Compose does the same through a `deploy.resources.reservations.devices` entry for the `nvidia`
driver on both `ollama` and `app`. If the app container cannot see a GPU it falls back to the
CPU for the embedder and reranker and is much slower; `GET /api/health` shows `"device"`.

**Other model.** Set `LOCAL_RAG_MODEL` on `app` and change the model in the `ollama-pull`
entrypoint. Only `qwen2.5:7b-instruct` is benchmarked.

## pip + Ollama

```bash
# 1. system packages (Debian/Ubuntu); skip OCR packages if you will not read scanned PDFs
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar

# 2. the language-model server
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                       # skip if the installer already started a service
ollama pull qwen2.5:7b-instruct

# 3. the app
python3 -m venv .venv && . .venv/bin/activate
pip install ".[web]"                 # from a checkout of this repository
python -m local_agentic_rag.server --library ~/rag-library --port 8000
# open http://127.0.0.1:8000/

# 4. check it (from another shell, in the checkout)
scripts/smoke_install.sh http://127.0.0.1:8000
```

`pip install` pulls `torch` for your platform; on a GPU machine the default Linux wheel
already includes CUDA support. Use `pip install -e ".[web]"` if you want to edit the code.

## Colab

Open `notebooks/Local_Agentic_RAG.ipynb` in Colab, choose **Runtime, Change runtime type, T4
GPU**, and run the cells in order. It installs the package, starts Ollama with
`qwen2.5:7b-instruct`, and offers a Gradio chat. Section 7b starts the same web app through
Colab's private port proxy. Nothing in this document changes that path.

## Options

Command line (`python -m local_agentic_rag.server --help`):

| Option | Default | Meaning |
|---|---|---|
| `--library DIR` | `./library` | PDFs, indexes and job state. In Docker: `/data/library` |
| `--host` | `127.0.0.1` | interface to listen on. The Docker image uses `0.0.0.0` inside the container |
| `--port` | `8000` | |
| `--model` | `$LOCAL_RAG_MODEL`, else `qwen2.5:7b-instruct` | Ollama model |
| `--ollama-host` | `$LOCAL_RAG_OLLAMA_HOST`, else `127.0.0.1:11434` | Ollama `host[:port]` or URL |
| `--token TOKEN` | `$LOCAL_RAG_TOKEN`, else off | require this access token on every `/api` route except `/api/health` (see "Access token") |
| `--max-upload-mb` | `2048` | largest accepted PDF |
| `--max-concurrent-asks` | `2` | questions answered at once (they share one GPU); others queue |
| `--ask-timeout` | `300` | seconds allowed per answer |
| `--upload-dir` | `<library>/.incoming` | where uploads are streamed before ingestion |
| `--demo` | off | fake backend without GPU or LLM, for trying the UI |
| `--log-level` | `info` | |

An explicit flag always overrides the environment variable.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_RAG_OLLAMA_HOST` | `127.0.0.1:11434` | where the app finds Ollama (set to `ollama:11434` in Compose) |
| `LOCAL_RAG_MODEL` | `qwen2.5:7b-instruct` | Ollama model name |
| `LOCAL_RAG_TOKEN` | unset (no authentication) | access token; the `--token` flag wins |
| `LOCAL_RAG_UPLOAD_QUOTA_BYTES` | unset (no quota) | most bytes of PDF one client address may upload per UTC day |
| `LOCAL_RAG_PARSE_WORKERS` | CPU count, at most 8 | parallel PDF parse processes |
| `LOCAL_RAG_PAGE_TIMEOUT` | `180` | seconds before one page parse is given up |
| `LOCAL_RAG_ONNX_THREADS` | `0` (no cap) | ONNX Runtime threads per parse worker (PyMuPDF's table finder); cap it when using many workers |
| `LOCAL_RAG_CACHE` | `~/.cache/local_agentic_rag/indexes` (`/data/cache` in Docker) | index cache for the Python API |
| `HF_HOME` | Hugging Face default (`/data/hf` in Docker) | where the embedder and reranker are cached |
| `TESSDATA_PREFIX` | autodetected | Tesseract language data directory |

### Faster table detection on a GPU

PDF table detection runs PyMuPDF's ONNX layout model, the slowest step on table-heavy pages.
It can run on the GPU, which is otherwise idle during parsing. This is **off by default**.

```bash
pip uninstall -y onnxruntime                    # the CPU and GPU builds cannot coexist
pip install "onnxruntime-gpu[cuda,cudnn]"
export LOCAL_RAG_ONNX_GPU=1                     # then start the server as usual
```

Measured on a Kaggle T4 (26–27 Oct 2026): a 1,000-page, table-heavy PDF parsed 1.89x faster
(224.7 s to 119.0 s) with byte-identical output. Each parse worker holds about 0.7 GB of GPU
memory, so budget that per worker next to the language model (`LOCAL_RAG_PARSE_WORKERS`). If the
CUDA provider cannot be loaded, the workers silently fall back to the CPU. The Colab notebook
does all of this in a cell controlled by `FAST_TABLE_DETECTION`.

## Access token and upload limits

### Access token

```bash
python -m local_agentic_rag.server --token "$(openssl rand -hex 24)"     # or: LOCAL_RAG_TOKEN=...
```

Without a token nothing changes: no authentication, as before. With one, every `/api/*` route
except `GET /api/health` answers `401 {"detail": "Authentication required. ..."}` unless the
request carries either

* `Authorization: Bearer <token>` (curl and other scripts; `scripts/smoke_install.sh` does not send a token yet, so run it
  against an instance started without `--token`), or
* the session cookie `local_rag_session`, set by `POST /api/session` with body
  `{"token": "<token>"}` (204 on success, 401 for a wrong token). `DELETE /api/session` clears it.

The cookie exists because the browser's `EventSource` (live progress) and `<img src>` (page
images) cannot send an `Authorization` header. It is `HttpOnly` (page scripts cannot read it),
`SameSite=Strict` (other sites cannot make the browser send it), `Secure` when the request came
over HTTPS (or `X-Forwarded-Proto: https`), and a session cookie (gone when the browser closes).
Its value is a keyed hash of the token, not the token. The web page's static files stay public so
the login prompt can load; on a 401 the page shows a small token form, posts it to
`/api/session` and retries. The page stores nothing in `localStorage` or `sessionStorage`.

Tokens are compared with `hmac.compare_digest`. There is **no rate limit on token guesses**, so use
a long random token (24 random bytes as above), not a word. The token is visible to anyone who can
read the process list or environment of the host, and travels in clear text unless TLS is in front.

With `--host` set to a non-loopback address and no token, the server logs a warning at startup.
(The Docker image binds `0.0.0.0` inside the container, so it logs that warning too; Compose
publishes the port on `127.0.0.1` only.)

### Upload limits

* **One upload at a time per client address.** A second `POST /api/documents` while one is still
  being received and registered answers `429` (`Retry-After: 10`). The web page uploads one file
  at a time to match.
* **Optional daily quota** `LOCAL_RAG_UPLOAD_QUOTA_BYTES` (bytes per client address per UTC day).
  An upload that would exceed it answers `413` before its body is read (using `Content-Length`,
  which includes a few hundred bytes of multipart framing, so the check is slightly
  conservative). Bytes are counted when an upload is accepted; rejected or aborted uploads are not
  counted.
* **Accounting is in memory**: counters reset when the server restarts and are not shared between
  processes (the server runs one).
* **The "client address" is the TCP peer address.** Behind a reverse proxy every user shares the
  proxy's address, so they share one upload slot and one quota. `X-Forwarded-For` is deliberately
  not trusted (any client can forge it); enforce per-user limits in the proxy instead.
* **Non-PDF files are rejected by content, not by name or content type**: the first 1 KB of the
  file must contain `%PDF-` (`415` otherwise, nothing left on disk). This check already existed;
  `tests/test_server_auth.py` now covers it with ZIP, HTML and PNG bodies named `.pdf`.

## Threat model

This is a **single-tenant, single-user** tool for documents you are allowed to read.

* **Authentication is one shared token, off by default.** With `--token` every `/api` route
  except `/api/health` needs it (see "Access token"); without it, anyone who can reach the port can
  upload, read, ask about and delete every document. Either way there is **no per-user
  separation**: everyone holding the token sees and can delete the whole library, and there are no
  accounts, roles or audit log.
* **Local by default.** The server listens on `127.0.0.1`; `docker-compose.yml` publishes the
  app as `127.0.0.1:8000:8000`, and does not publish Ollama's port at all. Note that a bare
  `docker run -p 8000:8000` publishes on **all** interfaces; always write `127.0.0.1:` in front.
* **Before exposing it to a network** put a reverse proxy in front that provides TLS (the token
  and cookie are otherwise sent in clear text), optionally with its own authentication (for
  example Caddy or nginx with basic auth, or an identity-aware proxy), and restrict the network
  path to trusted users. `--token` is a minimal gate, not a replacement for that. Two things the proxy must allow: the streaming
  endpoint `/api/ask/stream` is Server-Sent Events, so disable response buffering for it, and
  the upload limit (`--max-upload-mb`, default 2 GB) must also be raised in the proxy.
  This recipe is not tested.
* **Untrusted PDFs** are parsed by PyMuPDF in separate worker processes, so a crashing PDF costs
  one page, not the server. The container runs as a non-root user, but PDF parsers have had
  memory-safety bugs: do not feed the app files from sources you would not open yourself on a
  shared machine.
* **Data at rest** (PDFs, text indexes, rendered pages) is stored unencrypted in the library
  directory or the `rag-data` volume. Use disk encryption if that matters.
* **Prompt injection.** Text inside a PDF is shown to the language model. The answer is verified
  against the retrieved passages and shows its citations, but a hostile PDF can still try to
  steer the wording of an answer. The model has no tools that act outside the app.
* **Resource exhaustion.** Upload size, question length and concurrent answers are bounded
  (see the options). Uploads are limited to one in flight and, optionally, a daily byte quota per
  client address (in memory, reset on restart). Ask volume is not rate-limited per user, and token
  guesses are not rate-limited.

## Tested

What was and was not checked for the files in this change (run on a development machine
without a usable GPU or model in this session).

| Check | Result |
|---|---|
| `scripts/smoke_install.sh` against `python -m local_agentic_rag.server --demo` (fake backend): upload, poll, ask, citation | Passed |
| Same script, failure cases: server unreachable, non-PDF upload, unanswerable question (refused, no citation) | Each exits non-zero with a message |
| `docker compose config -q` (syntax of `docker-compose.yml`) | Passed |
| `LOCAL_RAG_OLLAMA_HOST` / `LOCAL_RAG_MODEL` defaults and flag override (`tests/test_server_cli.py`) | Passed |
| `docker build` of the `Dockerfile` | **Not run** |
| `docker compose up`, including the model pull, GPU reservation and `/api/health` OK | **Not run** |
| GPU visibility inside the container (`--gpus all`) | **Not run** |
| The pip + Ollama commands above on a clean machine | **Not run** (they are the project's existing commands plus the apt and Ollama installer steps) |
| `scripts/smoke_install.sh` against a real Ollama model (answer text, `417` in the answer) | Passed on a Kaggle T4, 26 Oct (1 citation) |
| Colab notebook | Not re-run for this change; the notebook is unchanged. Earlier benchmark runs used the same code on a T4 (see README) |
| `--token` / `LOCAL_RAG_TOKEN`: 401 without credentials, Bearer, cookie from `POST /api/session`, wrong token, health and static files public, SSE and page images with the cookie only, logout (`tests/test_server_auth.py`, FastAPI test client) | Passed |
| One in-flight upload per address (429), daily quota (413), non-PDF bodies rejected by magic bytes (`tests/test_server_auth.py`) | Passed |
| The browser login form and retry in `app.js` (JavaScript syntax checked with `node --check`) | Checked in Chrome against the demo server (26 Oct): 401, then the token form, then unlocked |
| Cookie behaviour over real HTTPS (`Secure` flag, proxy `X-Forwarded-Proto`) | **Not run** (header logic only, unit-tested) |
| Quota and 429 behind a reverse proxy (all users share the proxy address) | **Not run**, known limitation |
| TLS / authentication proxy in front of the app | **Not run** |

Until the Docker and pip rows are run on a GPU host, treat those paths as written but unverified.
