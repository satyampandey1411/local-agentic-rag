"""Regression tests for BEA-style statistical-table serialisation bugs.

Synthetic word tuples reproduce each mislabeling found on the BEA GDP release
(benchmark/pdfs/bea-gdp2q25-3rd.pdf); real-PDF assertions run when the file exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_agentic_rag.ingest.tables import serialize_table_words, split_stacked_cells

H = 8.0
CW = 4.0
PDFS = Path(__file__).resolve().parents[1] / "benchmark" / "pdfs"
BEA = PDFS / "bea-gdp2q25-3rd.pdf"
ARXIV = PDFS / "arxiv-2503.05037.pdf"


def _right(x1: float, y: float, text: str) -> tuple:
    return (x1 - CW * len(text), y, x1, y + H, text, 0, 0, 0)


def _centred(cx: float, y: float, text: str) -> tuple:
    half = CW * len(text) / 2
    return (cx - half, y, cx + half, y + H, text, 0, 0, 0)


def _label(y: float, text: str, x: float = 20.0) -> list[tuple]:
    out = []
    for token in text.split():
        out.append((x, y, x + CW * len(token), y + H, token, 0, 0, 0))
        x += CW * len(token) + 2
    return out


def _pairs(line: str) -> dict[str, str]:
    body = line.split(" | ")[-1]
    return dict(p.split(" = ") for p in body.split("; "))


# Columns: 3 annual levels, 5 quarterly levels (2024 Q2-Q4, 2025 Q1-Q2),
# 2 annual changes, 4 quarterly changes — the layout of BEA Table 10.
RIGHTS = [200 + 28 * i for i in range(14)]
CENTRES = [r - 12 for r in RIGHTS]  # 6-char values are 24 wide


def _span_cx(a: int, b: int) -> float:
    return (CENTRES[a] + CENTRES[b]) / 2


def bea_table10() -> list[tuple]:
    w = [
        _centred(_span_cx(0, 7), 100, "Level"),
        _centred(_span_cx(8, 13), 108, "Change"),
        _centred(_span_cx(3, 7), 116, "SAAR"),
    ]
    # Year row; the annual change years sit lower (vertically centred in a tall cell).
    w += [_centred(CENTRES[i], 128, y) for i, y in enumerate(["2022", "2023", "2024"])]
    w += [_centred(_span_cx(3, 5), 128, "2024"), _centred(_span_cx(6, 7), 128, "2025")]
    w += [_centred(CENTRES[8], 134, "2023"), _centred(CENTRES[9], 134, "2024")]
    w += [_centred(_span_cx(10, 11), 128, "2024"), _centred(_span_cx(12, 13), 128, "2025")]
    quarters = {3: "Q2", 4: "Q3", 5: "Q4", 6: "Q1", 7: "Q2", 10: "Q3", 11: "Q4", 12: "Q1", 13: "Q2"}
    w += [_centred(CENTRES[i], 140, q) for i, q in quarters.items()]
    # Row label wrapped over three lines, values on the last one.
    w += _label(154, "Corporate profits with inventory")
    w += _label(164, "valuation and capital", x=22)
    w += _label(174, "consumption adjustments", x=22)
    w += [_right(RIGHTS[i], 174, f"{100 + i}.0") for i in range(14)]
    w += _label(186, "Federal Reserve banks")
    fed = [
        "59.5",
        "-114.7",
        "-78.1",
        "-82.3",
        "-91.6",
        "-45.8",
        "-26.8",
        "-42.1",
        "-174.1",
        "36.6",
        "-9.3",
        "45.8",
        "18.9",
        "-15.3",
    ]
    w += [_right(RIGHTS[i], 186, v) for i, v in enumerate(fed)]
    w += _label(198, "Less: Payments to the rest of the")
    w += _label(208, "world", x=22)
    w += [_right(RIGHTS[i], 208, f"{200 + i}.0") for i in range(14)]
    return w


def _row(lines: list[str], label: str) -> str:
    return next(ln for ln in lines if ln.split(" | ")[-2] == label)


def test_year_header_does_not_absorb_columns_owned_by_lower_row():
    """Bug: 2025 spanned the annual-change columns, 2024 Q4 became '2025 Q4'."""
    fed = _pairs(_row(serialize_table_words(bea_table10()), "Federal Reserve banks"))
    assert fed["Level SAAR 2024 Q2"] == "-82.3"
    assert fed["Level SAAR 2024 Q4"] == "-45.8"
    assert fed["Level SAAR 2025 Q2"] == "-42.1"
    assert fed["Change 2023"] == "-174.1"
    assert fed["Change 2025 Q2"] == "-15.3"


def test_single_spanning_labels_get_their_full_group():
    """Bug: lone spanning labels (Level / SAAR) were spread over arbitrary columns."""
    fed = _pairs(_row(serialize_table_words(bea_table10()), "Federal Reserve banks"))
    assert fed["Level 2022"] == "59.5"  # SAAR does not reach the annual columns
    assert not any(k.startswith("Level") and "Change" in k for k in fed)


def test_wrapped_row_labels_are_joined_not_turned_into_sections():
    lines = serialize_table_words(bea_table10())
    labels = [ln.split(" | ")[:-1] for ln in lines]
    assert labels == [
        ["Corporate profits with inventory valuation and capital consumption adjustments"],
        ["Federal Reserve banks"],
        ["Less: Payments to the rest of the world"],
    ]


def test_ruled_header_spans_follow_vertical_rules():
    """With rules, a label sits in the cell between the rules around it."""
    words = bea_table10()
    edge = [RIGHTS[0] - 26] + [r + 2 for r in RIGHTS]  # rule right of each column
    vrules = [(edge[0], 90, 150), (edge[8], 90, 150), (edge[14], 90, 150)]  # Level | Change
    vrules += [(edge[3], 112, 150)]  # SAAR cell starts at the first quarter
    vrules += [(edge[i], 124, 150) for i in (1, 2, 6, 9, 10, 12)]  # year cells
    vrules += [(edge[i], 136, 150) for i in (4, 5, 7, 11, 13)]  # quarter cells
    fed = _pairs(_row(serialize_table_words(words, vrules=vrules), "Federal Reserve banks"))
    assert fed["Level SAAR 2025 Q2"] == "-42.1"
    assert fed["Change 2024 Q3"] == "-9.3"


def test_tall_annual_cell_keeps_spanning_label_off_its_column():
    """BEA Table 3: '2024' annual column, then 2024 Q2..2025 Q2 under 'SAAR'."""
    rights = [200 + 30 * i for i in range(6)]
    centres = [r - 14 for r in rights]
    w = [_centred((centres[0] + centres[5]) / 2, 90, "Billions")]
    w += [_centred((centres[1] + centres[5]) / 2, 100, "SAAR")]
    w += [_centred(centres[0], 110, "2024")]  # tall cell, centred in rows 100..120
    w += [_centred((centres[1] + centres[3]) / 2, 115, "2024")]
    w += [_centred((centres[4] + centres[5]) / 2, 115, "2025")]
    w += [
        _centred(centres[i], 125, q)
        for i, q in zip(range(1, 6), ["Q2", "Q3", "Q4", "Q1", "Q2"], strict=True)
    ]
    vals = ["29,298.0", "29,147.0", "29,511.7", "29,825.2", "30,042.1", "30,485.7"]
    for name, y in (("Gross product", 140), ("Net product", 152)):
        w += _label(y, name)
        w += [_right(rights[i], y, v) for i, v in enumerate(vals)]
    pairs = _pairs(serialize_table_words(w)[0])
    assert pairs["Billions 2024"] == "29,298.0"
    assert pairs["Billions SAAR 2025 Q1"] == "30,042.1"
    assert pairs["Billions SAAR 2025 Q2"] == "30,485.7"


def test_footnote_marker_is_not_a_value():
    rights = [200 + 30 * i for i in range(3)]
    w = [_centred(r - 8, 100, y) for r, y in zip(rights, ["2022", "2023", "2024"], strict=True)]
    names = ["Goods industries", "Private services-producing industries", "Total"]
    for k, (name, y) in enumerate(zip(names, (112, 124, 136), strict=True)):
        w += _label(y, name)
        if k < 2:  # superscript note glued to the label
            x = 20 + CW * len(name.replace(" ", "")) + 2 * (len(name.split()) - 1) + 1
            w.append((x, y - 2, x + 3, y + 3, str(k + 1), 0, 0, 0))
        w += [_right(r, y, v) for r, v in zip(rights, ["-2.8", "3.6", "3.6"], strict=True)]
    lines = serialize_table_words(w)
    assert lines[0].startswith("Goods industries | ")
    assert _pairs(lines[1]) == {"2022": "-2.8", "2023": "3.6", "2024": "3.6"}


def test_inline_year_stays_inside_header_phrase():
    rights = [200 + 30 * i for i in range(2)]
    w = _label(90, "Billions of chained (2017) dollars", x=150)
    w += [_centred(r - 8, 100, y) for r, y in zip(rights, ["2023", "2024"], strict=True)]
    for k, y in enumerate((112, 124, 136)):
        w += _label(y, f"Row {k}")
        w += [_right(r, y, v) for r, v in zip(rights, ["1.0", "2.0"], strict=True)]
    assert "Billions of chained (2017) dollars 2024 = 2.0" in serialize_table_words(w)[0]


def test_split_stacked_cells_separates_merged_booktabs_rows():
    def wd(x, y, t):
        return (x, y, x + CW * len(t), y + H, t, 0, 0, 0)

    cells = [
        [wd(10, 100, "Dragon"), wd(40, 100, "RoBERTa"), wd(10, 110, "Dragon+")],
        [wd(100, 100, "0.55"), wd(100, 110, "0.54")],
        [wd(140, 100, "0.75"), wd(140, 110, "0.74")],
    ]
    rows = split_stacked_cells(cells)
    texts = [[" ".join(w[4] for w in c) for c in r] for r in rows]
    assert texts == [["Dragon RoBERTa", "0.55", "0.75"], ["Dragon+", "0.54", "0.74"]]
    # A wrapped text cell next to single-line cells is not split.
    wrapped = [[wd(10, 100, "facebook/a"), wd(10, 110, "facebook/b")], [wd(100, 100, "Lin")]]
    assert split_stacked_cells(wrapped) == [wrapped]


# ---------------------------------------------------------------------------
# Real PDFs
# ---------------------------------------------------------------------------


def _page_tables(path: Path, page: int) -> list[str]:
    pymupdf = pytest.importorskip("pymupdf")
    from local_agentic_rag.ingest.pdf import _tables_on_page

    with pymupdf.open(path) as doc:
        return [md for md, _ in _tables_on_page(doc[page - 1])]


def _line(path: Path, page: int, label: str) -> str:
    for md in _page_tables(path, page):
        for ln in md.splitlines():
            if label in ln:
                return ln
    raise AssertionError(f"{label!r} not found on page {page}")


@pytest.mark.skipif(not BEA.exists(), reason="BEA benchmark PDF absent")
def test_bea_federal_reserve_profits_q2_2025():
    pairs = _pairs(_line(BEA, 18, "Federal Reserve banks |"))
    assert pairs["Level Seasonally adjusted at annual rates 2025 Q2"] == "-42.1"
    assert pairs["Level Seasonally adjusted at annual rates 2024 Q2"] == "-82.3"


@pytest.mark.skipif(not BEA.exists(), reason="BEA benchmark PDF absent")
def test_bea_mining_value_added_q2_2025():
    line = _line(BEA, 20, "Mining |")
    assert line.startswith("Mining |")  # no stale "Agriculture, forestry, fishing," prefix
    assert _pairs(line)["Seasonally adjusted at annual rates 2025 Q2"] == "29.0"


@pytest.mark.skipif(not BEA.exists(), reason="BEA benchmark PDF absent")
def test_bea_nominal_gdp_levels():
    pairs = _pairs(_line(BEA, 10, "Gross domestic product (GDP) |"))
    prefix = "Billions of dollars Seasonally adjusted at annual rates"
    assert pairs[f"{prefix} 2025 Q2"] == "30,485.7"
    assert pairs[f"{prefix} 2025 Q1"] == "30,042.1"


@pytest.mark.skipif(not BEA.exists(), reason="BEA benchmark PDF absent")
def test_bea_gdp_estimates_table():
    assert "| Real GDP | 3.0 | 3.3 | 3.8 |" in _page_tables(BEA, 3)[0]


@pytest.mark.skipif(not ARXIV.exists(), reason="arXiv benchmark PDF absent")
def test_arxiv_nq_table_rows_are_separated():
    md = _page_tables(ARXIV, 3)[0]
    assert "| Contriever MSMARCO | avg | 0.50 | 0.71 |" in md
    assert "| Contriever | avg | 0.25 | 0.41 |" in md
    assert "| Dragon+ | cls | 0.54 | 0.74 |" in md
