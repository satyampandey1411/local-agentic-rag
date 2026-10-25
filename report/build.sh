#!/usr/bin/env bash
# Compile main.tex with pdfLaTeX (same engine as Overleaf) in the texlive Docker image.
set -euo pipefail
cd "$(dirname "$0")"
for i in 1 2; do
  docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/work -w /work texlive/texlive:latest \
    pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { tail -40 build.log; exit 1; }
done
grep -E 'Overfull|Underfull|Warning' main.log | grep -v 'Font shape' || true
pdfinfo main.pdf | grep Pages
