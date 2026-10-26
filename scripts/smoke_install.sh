#!/usr/bin/env bash
# Install smoke test: upload a tiny PDF, wait until it is processed, ask one question and
# check that the answer carries a citation. Exits non-zero on any failure.
#
#   scripts/smoke_install.sh [BASE_URL] [PDF]
#
#   BASE_URL   default http://127.0.0.1:8000
#   PDF        optional PDF to use instead of the built-in one-page fact sheet. With your
#              own PDF set SMOKE_QUESTION too, and drop SMOKE_EXPECT.
#
# Environment: SMOKE_TIMEOUT (seconds to wait for processing, default 600),
#   SMOKE_QUESTION, SMOKE_EXPECT (text the answer should contain; a miss is only a warning),
#   SMOKE_KEEP=1 (do not delete the uploaded document afterwards), PYTHON (default python3,
#   used only to read JSON).
#
# The endpoints used are the ones in src/local_agentic_rag/server/app.py:
#   GET /api/health, POST /api/documents (multipart field "file"),
#   GET /api/documents/{id}, POST /api/ask, DELETE /api/documents/{id}.
set -u

BASE="${1:-http://127.0.0.1:8000}"
BASE="${BASE%/}"
PDF_ARG="${2:-}"
PY="${PYTHON:-python3}"
TIMEOUT="${SMOKE_TIMEOUT:-600}"
QUESTION="${SMOKE_QUESTION:-What is the rated output of the Aldermoor reactor?}"
EXPECT="${SMOKE_EXPECT-417}"

WORK="$(mktemp -d)"
DOC_ID=""
cleanup() {
    if [ -n "$DOC_ID" ] && [ "${SMOKE_KEEP:-0}" != "1" ]; then
        curl -s -o /dev/null -X DELETE "$BASE/api/documents/$DOC_ID" || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
step() { echo "== $*"; }

# json_get FILE EXPR: print a field of a JSON file (EXPR is a Python expression over d).
json_get() { "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }

command -v curl >/dev/null || fail "curl is required"
command -v "$PY" >/dev/null || fail "$PY is required (only used to read JSON)"

# ---------------------------------------------------------------- the PDF
if [ -n "$PDF_ARG" ]; then
    [ -f "$PDF_ARG" ] || fail "no such file: $PDF_ARG"
    PDF="$PDF_ARG"
else
    PDF="$WORK/aldermoor.pdf"
    cat > "$PDF" <<'PDF_END'
%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 223 >>
stream
BT /F1 12 Tf 14 TL 72 720 Td (The Aldermoor Reactor Fact Sheet) Tj T* (The Aldermoor reactor has a rated output of 417 megawatts.) Tj T* (It was commissioned in 1987 and is operated by the Northern Grid Authority.) Tj T* ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000241 00000 n
0000000515 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
585
%%EOF
PDF_END
fi

# ---------------------------------------------------------------- health
step "health: $BASE/api/health"
code=$(curl -s -o "$WORK/health.json" -w '%{http_code}' --max-time 15 "$BASE/api/health") \
    || fail "cannot reach $BASE (is the server up?)"
[ "$code" = "200" ] || fail "/api/health returned HTTP $code"
llm=$(json_get "$WORK/health.json" "d.get('llm')")
echo "llm=$llm model=$(json_get "$WORK/health.json" "d.get('model')") device=$(json_get "$WORK/health.json" "d.get('device')")"
[ "$llm" = "ok" ] || fail "the language model server is not reachable (health.llm=$llm); is Ollama running?"

# ---------------------------------------------------------------- upload
step "upload: $(basename "$PDF")"
code=$(curl -s -o "$WORK/upload.json" -w '%{http_code}' --max-time 120 \
    -F "file=@$PDF;type=application/pdf" "$BASE/api/documents") || fail "upload request failed"
[ "$code" = "201" ] || fail "upload returned HTTP $code: $(cat "$WORK/upload.json")"
DOC_ID=$(json_get "$WORK/upload.json" "d['doc_id']") || fail "upload response has no doc_id"
echo "doc_id=$DOC_ID"

# ---------------------------------------------------------------- wait until ready
step "waiting for processing (up to ${TIMEOUT}s)"
start=$(date +%s)
status=""
while :; do
    code=$(curl -s -o "$WORK/doc.json" -w '%{http_code}' --max-time 15 "$BASE/api/documents/$DOC_ID") \
        || fail "status request failed"
    [ "$code" = "200" ] || fail "document status returned HTTP $code"
    status=$(json_get "$WORK/doc.json" "d['status']")
    case "$status" in
        ready) break ;;
        failed) fail "ingestion failed: $(json_get "$WORK/doc.json" "d.get('error')")" ;;
    esac
    [ $(( $(date +%s) - start )) -lt "$TIMEOUT" ] || fail "timed out after ${TIMEOUT}s (status=$status)"
    sleep 2
done
echo "ready: $(json_get "$WORK/doc.json" "str(d['num_pages'])+' page(s), '+str(d['num_chunks'])+' chunk(s)'")"

# ---------------------------------------------------------------- ask
step "ask: $QUESTION"
"$PY" -c 'import json,sys; print(json.dumps({"question": sys.argv[1], "doc_ids": [sys.argv[2]]}))' \
    "$QUESTION" "$DOC_ID" > "$WORK/ask_body.json"
code=$(curl -s -o "$WORK/ask.json" -w '%{http_code}' --max-time 400 \
    -H 'Content-Type: application/json' --data-binary "@$WORK/ask_body.json" \
    "$BASE/api/ask") || fail "ask request failed"
[ "$code" = "200" ] || fail "ask returned HTTP $code: $(cat "$WORK/ask.json")"

answer=$(json_get "$WORK/ask.json" "d['answer']")
refused=$(json_get "$WORK/ask.json" "d.get('refused')")
ncite=$(json_get "$WORK/ask.json" "len(d.get('citations') or [])")
pages=$(json_get "$WORK/ask.json" "d.get('cited_pages')")
echo "answer: $answer"
echo "refused=$refused citations=$ncite cited_pages=$pages"

[ "$refused" = "False" ] || fail "the question was refused"
[ "$ncite" -ge 1 ] 2>/dev/null || fail "the answer has no citation"
if [ -n "$EXPECT" ] && ! printf '%s' "$answer" | grep -qiF -- "$EXPECT"; then
    echo "WARN: the answer does not contain '$EXPECT' (the citation check still passed)" >&2
fi

echo "PASS: upload, processing, answer with $ncite citation(s) on pages $pages"
