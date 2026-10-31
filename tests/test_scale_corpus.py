"""
Fast tests for the scale-benchmark corpus generator (benchmark/scale/).

A 40-page document is generated with the same code path as the 100-10,000 page
tiers (with a small part size, so the multi-part build-and-join path is exercised)
and checked with the corpus validator plus a few direct PyMuPDF checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

SCALE_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "scale"
PAGES = 40
PART_PAGES = 16  # forces three parts, joined with insert_pdf
DEVANAGARI_DIR = Path("/usr/share/fonts/truetype/noto")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"scale_{name}", SCALE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_corpus = _load("make_corpus")
validate_corpus = _load("validate_corpus")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("scale")
    meta = make_corpus.generate(PAGES, out, seed=7, part_pages=PART_PAGES)
    return out, meta


def _paths(out: Path, pages: int = PAGES) -> tuple[Path, Path, Path]:
    stem = f"scale-{pages}"
    return out / f"{stem}.pdf", out / f"{stem}.questions.jsonl", out / f"{stem}.meta.json"


def _questions(out: Path) -> list[dict]:
    return [json.loads(line) for line in _paths(out)[1].read_text().splitlines()]


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_writes_all_files_with_exact_page_count(corpus):
    out, meta = corpus
    pdf, qfile, mfile = _paths(out)
    assert pdf.exists() and qfile.exists() and mfile.exists()
    with pymupdf.open(pdf) as doc:
        assert doc.page_count == PAGES
        toc = doc.get_toc()
    assert meta["pages"] == PAGES
    assert json.loads(mfile.read_text())["sha256"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert sum(v for k, v in meta["page_kinds"].items() if not k.startswith("with_")) == PAGES
    # outline has chapters (level 1) and sections (level 2)
    assert any(lvl == 1 and title.startswith("Chapter 1:") for lvl, title, _ in toc)
    assert any(lvl == 2 for lvl, _, _ in toc)


def test_validator_passes(corpus):
    out, _ = corpus
    errors, summary = validate_corpus.validate_doc(_paths(out)[0])
    assert errors == []
    assert summary["questions"] == make_corpus.N_QUESTIONS


def test_question_set_shape(corpus):
    out, _ = corpus
    qs = _questions(out)
    assert len(qs) == make_corpus.N_QUESTIONS
    assert len({q["q_id"] for q in qs}) == len(qs)
    assert Counter(q["type"] for q in qs) == Counter(t for t, _ in make_corpus.SLOTS)
    for q in qs:
        for field in validate_corpus.REQUIRED:
            assert field in q, (q["q_id"], field)
        assert q["doc_id"] == f"scale-{PAGES}"
        assert q["answerable"] == (q["type"] != "unanswerable")
        if q["answerable"]:
            assert q["gold_pages"] and q["answer_keys"]
        else:
            assert q["gold_pages"] == [] and q["answer_keys"] == []
    depths = sorted(q["depth"] for q in qs)
    assert depths[0] < 0.15 and depths[-1] > 0.85  # planted facts span the document
    multi = [q for q in qs if q["type"] == "multi_hop"]
    assert all(len(set(q["gold_pages"])) == 2 for q in multi)


def test_same_seed_is_byte_identical(corpus, tmp_path):
    out, meta = corpus
    again = make_corpus.generate(PAGES, tmp_path, seed=7, part_pages=PART_PAGES)
    assert again["sha256"] == meta["sha256"]
    assert _paths(tmp_path)[0].read_bytes() == _paths(out)[0].read_bytes()
    assert _paths(tmp_path)[1].read_text() == _paths(out)[1].read_text()


def test_different_seed_gives_different_document(corpus, tmp_path):
    _, meta = corpus
    other = make_corpus.generate(PAGES, tmp_path, seed=8, part_pages=PART_PAGES)
    assert other["sha256"] != meta["sha256"]


def test_part_size_does_not_change_content(corpus, tmp_path):
    out, _ = corpus
    make_corpus.generate(PAGES, tmp_path, seed=7)  # default part size: one part
    with pymupdf.open(_paths(out)[0]) as a, pymupdf.open(_paths(tmp_path)[0]) as b:
        assert [p.get_text() for p in a] == [p.get_text() for p in b]
    assert _paths(out)[1].read_text() == _paths(tmp_path)[1].read_text()


def test_table_and_page_ref_questions_point_to_the_right_page(corpus):
    out, _ = corpus
    qs = _questions(out)
    with pymupdf.open(_paths(out)[0]) as doc:
        texts = [_norm(p.get_text()) for p in doc]
        checked_grid = False
        for q in qs:
            if q.get("kind") == "table_on_page":
                page = q["gold_pages"][0]
                assert f"Page {page}" in texts[page - 1]
                assert f"Table {q['table']}:" in texts[page - 1]
                if not checked_grid:  # the ruled grid is detected as a real table
                    tabs = doc[page - 1].find_tables().tables
                    assert tabs and max(t.row_count for t in tabs) >= 5
                    checked_grid = True
            elif q["type"] == "table_cell":
                page = q["gold_pages"][0]
                text = texts[page - 1]
                assert f"Table {q['table']}:" in text
                assert q["row"] in text and q["answer_keys"][0] in text
            elif q.get("kind") == "section_on_page":
                body_page = max(q["gold_pages"])  # the other one, if any, is the TOC page
                assert f"{q['section']} {q['answer_keys'][0]} District" in texts[body_page - 1]
        assert checked_grid


def test_scanned_pages_are_image_only(corpus):
    out, meta = corpus
    scanned = meta["scanned_pages"]
    assert len(scanned) >= 2
    with pymupdf.open(_paths(out)[0]) as doc:
        for p in scanned:
            page = doc[p - 1]
            assert page.get_text().strip() == ""
            assert len(page.get_images(full=False)) == 1
        text_pages = [i for i in range(PAGES) if i + 1 not in scanned]
        assert all(doc[i].get_text().strip() for i in text_pages)
    ocr_qs = [q for q in _questions(out) if q["type"] == "ocr"]
    assert ocr_qs and all(set(q["gold_pages"]) <= set(scanned) for q in ocr_qs)


def test_validator_catches_broken_ground_truth(corpus, tmp_path):
    out, _ = corpus
    for path in _paths(out):
        shutil.copy(path, tmp_path / path.name)
    pdf, qfile, _ = _paths(tmp_path)
    qs = _questions(tmp_path)
    lookup = next(q for q in qs if q["type"] == "lookup")
    lookup["answer_keys"] = ["no such value 98,765"]
    real_district = next(q for q in qs if q.get("kind") == "section_on_page")["answer_keys"][0]
    fake = next(q for q in qs if q.get("kind") == "fake_district")
    fake["absent_keys"] = [f"{real_district} District"]
    qfile.write_text("".join(json.dumps(q) + "\n" for q in qs))
    errors, _ = validate_corpus.validate_doc(pdf)
    assert any(lookup["q_id"] in e for e in errors)
    assert any(fake["q_id"] in e for e in errors)
    assert validate_corpus.main([str(tmp_path)]) == 1


def test_missing_font_dir_skips_hindi_pages(tmp_path):
    meta = make_corpus.generate(24, tmp_path, seed=3, font_dir=tmp_path / "no-fonts")
    assert meta["hindi_pages"] == [] and meta["pages"] == 24


@pytest.mark.skipif(
    not list(DEVANAGARI_DIR.glob("*Devanagari*Regular*.ttf")), reason="no Devanagari font installed"
)
def test_hindi_pages_with_font(tmp_path):
    meta = make_corpus.generate(PAGES, tmp_path, seed=7, font_dir=DEVANAGARI_DIR)
    assert meta["hindi_pages"]
    with pymupdf.open(_paths(tmp_path)[0]) as doc:
        assert doc.page_count == PAGES
        text = doc[meta["hindi_pages"][0] - 1].get_text()
    assert any("ऀ" <= ch <= "ॿ" for ch in text)
    errors, _ = validate_corpus.validate_doc(_paths(tmp_path)[0])
    assert errors == []


def test_cli_writes_requested_tier(tmp_path, capsys):
    assert make_corpus.main(["--pages", "24", "--out", str(tmp_path), "--seed", "5"]) == 0
    assert "scale-24" in capsys.readouterr().out
    assert validate_corpus.main([str(tmp_path)]) == 0
    assert make_corpus.parse_pages("all") == list(make_corpus.TIERS)
