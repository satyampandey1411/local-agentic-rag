"""Unit tests for local_agentic_rag.ingest.tables (whitespace-aligned table serialisation)."""

from __future__ import annotations

import pytest

from local_agentic_rag.ingest.tables import (
    is_garbled_markdown,
    numeric_word_ratio,
    serialize_table_words,
)

H = 8.0  # word height
CW = 4.0  # character width


def _word(x0: float, y: float, text: str) -> tuple:
    return (x0, y, x0 + CW * len(text), y + H, text, 0, 0, 0)


def _right(x1: float, y: float, text: str) -> tuple:
    return (x1 - CW * len(text), y, x1, y + H, text, 0, 0, 0)


def _centred(cx: float, y: float, text: str) -> tuple:
    half = CW * len(text) / 2
    return (cx - half, y, cx + half, y + H, text, 0, 0, 0)


def _label(y: float, text: str, x: float = 20.0) -> list[tuple]:
    out = []
    for token in text.split():
        out.append(_word(x, y, token))
        x += CW * len(token) + 2
    return out


QUARTERS = ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2"]
COL_NAMES = ["2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1", "2024 Q2"]
ROWS = {
    "Gross domestic product": ["1.5", "2.0", "2.5", "3.0", "3.5", "4.0"],
    "Durable goods": ["0.5", "0.6", "0.7", "0.8", "0.9", "1.1"],
    "Nondurable goods": ["-0.2", "0.1", "0.3", "0.4", "0.6", "0.7"],
}


def build_table(gap: float, section_position: str = "middle") -> list[tuple]:
    """Six right-aligned numeric columns under "2023" (Q1..Q4) and "2024" (Q1..Q2).

    section_position: "middle" puts the "Goods sector" label row between the first
    and second data rows; "top" puts a "Percent change" label row directly under
    the column headers.
    """
    rights = [200 + gap * i for i in range(6)]
    centres = [r - 8 for r in rights]  # 4-char numbers are 16 wide
    words = [
        _centred(sum(centres[:4]) / 4, 100, "2023"),
        _centred(sum(centres[4:]) / 2, 100, "2024"),
    ]
    words += [_centred(c, 112, q) for c, q in zip(centres, QUARTERS, strict=True)]
    y = 124.0
    if section_position == "top":
        words += _label(y, "Percent change")
        y += 12
    for i, (label, values) in enumerate(ROWS.items()):
        if section_position == "middle" and i == 1:
            words += _label(y, "Goods sector")
            y += 12
        words += _label(y, label)
        words += [_right(rights[j], y, v) for j, v in enumerate(values)]
        y += 12
    return words


def _expected_pairs(values: list[str]) -> str:
    return "; ".join(f"{name} = {v}" for name, v in zip(COL_NAMES, values, strict=True))


# ---------------------------------------------------------------------------
# serialize_table_words
# ---------------------------------------------------------------------------


def test_spanning_header_and_section_rows_compact_columns():
    lines = serialize_table_words(build_table(gap=20), title="Table 1. Real GDP")
    assert lines == [
        "Table 1. Real GDP",
        f"Gross domestic product | {_expected_pairs(ROWS['Gross domestic product'])}",
        f"Goods sector | Durable goods | {_expected_pairs(ROWS['Durable goods'])}",
        f"Goods sector | Nondurable goods | {_expected_pairs(ROWS['Nondurable goods'])}",
    ]


def test_specific_cell_labels():
    lines = serialize_table_words(build_table(gap=20))
    durable = next(ln for ln in lines if "Durable goods" in ln and "Non" not in ln)
    assert "2023 Q1 = 0.5" in durable
    assert "2023 Q4 = 0.8" in durable
    assert "2024 Q2 = 1.1" in durable
    nondurable = next(ln for ln in lines if "Nondurable" in ln)
    assert "2023 Q1 = -0.2" in nondurable


def test_no_title_line_when_title_empty():
    lines = serialize_table_words(build_table(gap=20))
    assert lines[0].startswith("Gross domestic product | ")


def test_spanning_header_wide_columns():
    lines = serialize_table_words(build_table(gap=40))
    assert lines[0] == f"Gross domestic product | {_expected_pairs(ROWS['Gross domestic product'])}"


def test_section_row_directly_above_first_data_row():
    lines = serialize_table_words(build_table(gap=20, section_position="top"))
    assert lines[0].startswith("Percent change | Gross domestic product | ")


def test_too_few_rows_returns_empty():
    words = _label(100, "Revenue") + [_right(200, 100, "1.0"), _right(240, 100, "2.0")]
    assert serialize_table_words(words) == []


def test_text_only_region_returns_empty():
    words = []
    for i in range(5):
        words += _label(100 + 12 * i, "just some prose words in a box")
    assert serialize_table_words(words) == []


def test_empty_words_returns_empty():
    assert serialize_table_words([]) == []


def test_numeric_word_ratio():
    words = [_word(0, 0, "1.5"), _word(20, 0, "abc"), _word(40, 0, "12%"), _word(60, 0, "x")]
    assert numeric_word_ratio(words) == pytest.approx(0.5)
    assert numeric_word_ratio([]) == 0.0


# ---------------------------------------------------------------------------
# is_garbled_markdown
# ---------------------------------------------------------------------------


def test_clean_markdown_is_not_garbled():
    md = "|Region|Revenue|\n|---|---|\n|North|1250|\n|South|980|"
    assert is_garbled_markdown(md) is False


def test_stacked_cells_are_garbled():
    md = "|Item|2023|2024|\n|---|---|---|\n|GDP<br>CPI<br>PCE|2.5<br>2.9<br>2.8|3.1<br>3.0<br>2.7|"
    assert is_garbled_markdown(md) is True


def test_escaped_br_counts_as_stacked():
    md = "|a|b|\n|---|---|\n|1&lt;br&gt;2|3&lt;br&gt;4|"
    assert is_garbled_markdown(md) is True


def test_empty_markdown_is_garbled():
    assert is_garbled_markdown("") is True
    assert is_garbled_markdown("| | |") is True


def test_few_stacked_cells_below_threshold():
    rows = "\n".join(f"|r{i}|{i}|{i + 1}|" for i in range(10))
    md = "|h|a|b|\n|---|---|---|\n" + rows + "\n|x<br>y|1|2|"
    assert is_garbled_markdown(md) is False
