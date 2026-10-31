#!/usr/bin/env bash
# Compile main.tex with pdfLaTeX (same engine as Overleaf) in the texlive Docker image, 2 passes.
# The repository root is mounted so the workflow figure in docs/img resolves.
set -euo pipefail
cd "$(dirname "$0")"
root="$(cd .. && pwd)"
for i in 1 2; do
  docker run --rm -u "$(id -u):$(id -g)" -v "$root":/work -w /work/report texlive/texlive:latest \
    pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { tail -40 build.log; exit 1; }
done
cp main.pdf Local_Agentic_RAG_Report.pdf
grep -E 'Overfull|Underfull|Warning' main.log | grep -v 'Font shape' || true
pdfinfo Local_Agentic_RAG_Report.pdf | grep Pages
