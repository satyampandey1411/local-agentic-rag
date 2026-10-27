"""
Calculator tool for quantitative questions.

Measured failure (docs/RESULTS.md): on numerical questions every configuration
scored 0% even though the evidence was retrieved. The 7B model subtracts and
counts wrong (222.8 instead of 443.6; "8 actions" when 9 are listed). So
arithmetic is moved out of the model:

    1. a cheap regex decides whether the question is quantitative;
    2. one JSON-constrained LLM call extracts the operands (each with the passage
       it came from) and an arithmetic expression, or the items to count;
    3. ``safe_eval`` computes the result deterministically (numbers and + - * /
       only — no names, calls or attributes, so model output cannot execute code);
    4. the result is handed to the generator as a verified hint.

Measured failures of that design (2026-10-25 run) and the guards added for them:

* operands from the wrong row/column/passage (2.61/-3.52 from the "Nonfarm" row
  for a GDP swing; 93.6 from the "Evidence Doc" column for "no document"):
  ``ground`` checks every operand against the passage it cites — the number must
  appear there, and on a table line its label must name the row it sits on and
  not match another column better.  A rejected extraction is retried once with
  the reasons; an ungrounded calculation is dropped, never shown;
* the calculator overriding a value the document states ("rose by how much" →
  computed -785.2 while the release says "increased $6.8 billion"): only
  questions that need arithmetic by their wording (``needs_arithmetic``) are
  *forced* into the answer; for other quantitative questions the calculator is
  skipped when a passage states the change (``stated_change``) and otherwise
  only offers a hint;
* "16 jets in each of 16 petals" summed: a per-group label turns a two-value
  sum into a product;
* counting IDs spread over pages (MS-2.7-001..009 on pp. 37-38, answered "8"):
  ``id_prefix``/``count_ids`` count identifiers deterministically.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from local_agentic_rag import lang

_QUANT_RE = re.compile(
    r"\b(how much (higher|lower|more|less|larger|smaller|bigger|greater|did)"
    r"|how many (percentage points|points|times|more|fewer|months|years|days|weeks)"
    r"|differen(ce|t)\b.*\b(between|from)|what is the difference"
    r"|(increase|decrease|grow|drop|fall|rise|change)[a-z]* (from|between|by how much)"
    r"|by how (much|many)|ratio|percent(age)? (change|increase|decrease|of)"
    r"|what (share|fraction|proportion)|sum of|in total|combined|average of"
    r"|how many [a-z ]{0,40}(are|were) (listed|there|included|named|given))",
    re.IGNORECASE,
)

CALC_SYSTEM = (
    "You extract the numbers needed to answer a quantitative question from document passages."
)

CALC_USER = """Passages:
{passages}

Question: {question}

Decide which calculation answers the question and copy the needed numbers EXACTLY as written in the passages.
- If a passage already states the change the question asks for (e.g. "profits increased $6.8 billion", "contributed -0.54 percentage point"), no calculation is needed: set "needed" to false. A time span between two dates always needs "months_between", even when both dates are stated.
- Take the numbers from the passage and table that match every qualifier of the question (entity, dataset, model, period, measure). Check the row AND the column/period of every table value.
- A decrease is negative ("decreased 0.6 percent" -> -0.6).
{feedback}
Respond in JSON with:
- "needed": true if the answer requires arithmetic, counting, or a time span; otherwise false
- "operation": one of
    "difference"      absolute gap between two values ("how much higher/lower", "difference", "swing")
    "change"          second value minus first value ("increase/decrease from A to B")
    "sum"             all values added
    "product"         all values multiplied: a number of groups times the amount in EACH group ("16 petals with 16 jets each" -> 16 x 16, not 16 + 16)
    "ratio"           first value divided by second
    "percent_change"  (second - first) / first x 100
    "count"           number of distinct listed items (fill "items")
    "months_between"  months between two dates (fill "dates")
- "values": the numbers, in the order the operation needs, each with "label" (table row + column/period, e.g. "Gross domestic product | 2025 Q2"; start it with "per" for an amount per group, e.g. "per petal: jets"), "value" (a plain number: no units, commas or % signs) and "passage" (its number)
- "items": the listed items to count, each exactly once, by ID if they have one (only for "count")
- "dates": two dates as {{"label", "year", "month", "passage"}} (only for "months_between")
- "unit": the unit of the result, e.g. "billion dollars", "percentage points", "months"
"""

CALC_FEEDBACK = """
A previous answer to this task was rejected:
{problems}
Copy each number from the passage you cite, on the row and column its label names. If the passages do not hold the right numbers, set "needed" to false.
"""

CALC_MISSING = """
A previous answer to this task gave no calculation, but the question asks for one by its wording
(a difference, change, total, ratio, count or time span). Set "needed" to true and fill "operation"
with the numbers or dates it needs, copied from the passages.
"""

OPERATIONS = (
    "difference",
    "change",
    "sum",
    "product",
    "ratio",
    "percent_change",
    "count",
    "months_between",
)

CALC_SCHEMA = {
    "type": "object",
    "properties": {
        "needed": {"type": "boolean"},
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "values": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                    "passage": {"type": "integer"},
                },
                "required": ["label", "value"],
            },
        },
        "items": {"type": "array", "items": {"type": "string"}},
        "dates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "year": {"type": "integer"},
                    "month": {"type": "integer"},
                    "passage": {"type": "integer"},
                },
                "required": ["year", "month"],
            },
        },
        "expression": {"type": "string"},
        "unit": {"type": "string"},
    },
    "required": ["needed"],
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


class CalculatorError(ValueError):
    pass


def is_quantitative(question: str) -> bool:
    return bool(_QUANT_RE.search(question))


# Wording that can only be answered by combining two or more numbers (or counting): the
# result is then *forced* into the answer.  Other quantitative wordings ("rose by how much
# in Q2", "how many percentage points did X contribute") usually ask for a value the
# document states, so the calculator only offers a hint there (BEA q11/q13).
_ARITH_RE = re.compile(
    r"\b(than|swing|swung|differen(ce|t)|between|ratio|times|in total|total number|combined"
    r"|altogether|sum of|average|percent(age)? (change|increase|decrease))\b"
    r"|\bfrom\b.+\bto\b"
    r"|\bhow many [a-z ]{0,40}(are|were) (listed|there|included|named|given)\b",
    re.IGNORECASE,
)


def needs_arithmetic(question: str) -> bool:
    return bool(_ARITH_RE.search(question))


# --------------------------------------------------------------------------- grounding

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# A number in passage text.  Digits glued to letters or joined by a hyphen to a word are
# names, not values: "gpt-4o", "Recall@10", "Q2", "MS-2.7-001".
_TEXT_NUM_RE = re.compile(r"(?<![\w.@])(?<!\w-)(\d[\d,]*(?:\.\d+)?|\.\d+)(?![\w@]|\.\d|-\w)")
_LABEL_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_TEXT = """a an and as at be by for from in is its of on or per the their to was were with
    value values table row column"""
_STOP = frozenset(_STOP_TEXT.split())
_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
_PER_GROUP_RE = re.compile(r"\b(each|per|every)\b|प्रत्येक|प्रति", re.IGNORECASE)


def _to_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int | float):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", "").replace("%", "").replace("−", "-").strip())
        except ValueError:
            return None
    return None


def _same(a: float, b: float) -> bool:
    """Equal up to sign ("decreased 0.6" is the operand -0.6) and float noise."""
    return abs(abs(a) - abs(b)) <= 1e-9 * max(1.0, abs(b))


def _numbers_in(text: str) -> list[float]:
    return [float(m.group(1).replace(",", "")) for m in _TEXT_NUM_RE.finditer(text)]


def _tokens(text: str) -> set[str]:
    """Content words, plus joined neighbours ("MS MARCO" ~ "MSMARCO") and the initials of
    a multi-word name ("Gross domestic product" ~ "GDP")."""
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]
    out = set(words) | {a + b for a, b in zip(words, words[1:], strict=False)}
    alpha = [w for w in words if w.isalpha()]
    if len(alpha) >= 2:
        out.add("".join(w[0] for w in alpha))
    return out


def _match(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def _overlap(a: set[str], b: set[str]) -> int:
    return sum(any(_match(x, y) for y in b) for x in a)


def _table_rows(text: str) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """(line, row label, [(column header, cell text), ...]) for every table row in a chunk.

    Two serialisations (``ingest/tables.py``): statistical rows
    ``section | row label | header = value; header = value`` and Markdown grids, whose
    header is the first line plus short label-only lines right under the ``|---|`` rule.
    """
    rows: list[tuple[str, str, list[tuple[str, str]]]] = []
    headers: list[str] | None = None
    in_body, body_rows = False, 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if any(cells) and all(_SEP_CELL_RE.match(c) for c in cells if c):
                in_body = True
                continue
            if headers is None:
                headers = cells
                continue
            padded = headers + [""] * max(0, len(cells) - len(headers))
            numeric = any(_TEXT_NUM_RE.search(c) for c in cells)
            if (
                in_body
                and not body_rows
                and not numeric
                and all(len(c.split()) <= 3 for c in cells)
            ):
                # second header line under the rule ("| MODEL | Doc | Doc |")
                headers = [f"{h} {c}".strip() for h, c in zip(padded, cells, strict=False)]
                continue
            label = next((c for c in cells if c and not _TEXT_NUM_RE.search(c)), "")
            rows.append((line, label, list(zip(padded, cells, strict=False))))
            body_rows += 1
            continue
        headers, in_body, body_rows = None, False, 0
        if " = " not in line:
            continue
        labels, cells_kv = [], []
        for segment in line.split(" | "):
            if " = " not in segment:
                labels.append(segment.strip())
                continue
            for part in segment.split("; "):
                head, sep, val = part.rpartition(" = ")
                if sep:
                    cells_kv.append((head.strip(), val.strip()))
        if cells_kv:
            rows.append((line, labels[-1] if labels else "", cells_kv))
    return rows


def _check_in_passage(value: float, label: str, text: str) -> tuple[bool, str]:
    """(found, problem): problem is "" when one occurrence of ``value`` fits ``label``."""
    text = text.translate(_DEVANAGARI_DIGITS)
    if not any(_same(x, value) for x in _numbers_in(text)):
        return False, ""
    rows = _table_rows(text)
    table_lines = {line for line, _, _ in rows}
    for line in text.splitlines():
        if line not in table_lines and any(_same(x, value) for x in _numbers_in(line)):
            return True, ""  # prose: the sentence itself carries the context
    want = _tokens(label)
    problem = ""
    for _, row_label, cells in rows:
        for i, (header, cell) in enumerate(cells):
            if not any(_same(x, value) for x in _numbers_in(cell)):
                continue
            head = _tokens(header)
            residual = {t for t in want if not any(_match(t, h) for h in head)}
            if not residual or not _overlap(residual, _tokens(row_label)):
                problem = problem or (
                    f"the label does not name the table row of {_fmt(value)}, which is "
                    f'"{row_label}" (column "{header}")'
                )
                continue
            others = [h for j, (h, _) in enumerate(cells) if j != i and h != header]
            rival = max(others, key=lambda h: _overlap(want, _tokens(h)), default="")
            if rival and _overlap(want, _tokens(rival)) > _overlap(want, head):
                problem = problem or (
                    f'{_fmt(value)} is in column "{header}" of row "{row_label}", but the '
                    f'label matches column "{rival}"'
                )
                continue
            return True, ""
    return True, problem or f"{_fmt(value)} is not in a table cell that matches its label"


_NAME_WORD_RE = re.compile(r"[a-z0-9]+\+*")  # keeps "dragon+" apart from "dragon"


def _words(text: str) -> list[str]:
    return [w for w in _NAME_WORD_RE.findall(text.lower()) if w not in _STOP]


def _name_tokens(text: str) -> set[str]:
    """Like ``_tokens`` (joined neighbours, initials) but with exact name words."""
    words = _words(text)
    out = set(words) | {a + b for a, b in zip(words, words[1:], strict=False)}
    alpha = [w for w in words if w.isalpha()]
    if len(alpha) >= 2:
        out.add("".join(w[0] for w in alpha))
    return out


def _row_key(row_label: str, want: set[str]) -> tuple[int, int] | None:
    """(matched words, unmatched words) of a row label against an operand label, or ``None``.

    Exact words only: "gpt-4o" fits the label "gpt-4o | No Doc" as (2, 0); "gpt-4o-mini" as
    (2, 1) — its word "mini" is not in the label.
    """
    words = set(_words(row_label))
    if not words:
        return None
    matched = sum(w in want for w in words)
    return matched, len(words) - matched


def align_cell(value: float, label: str, text: str) -> tuple[float, str, str] | None:
    """``(cell value, row, column)`` of the one table cell that ``label`` names, when ``value``
    sits in a table cell of ``text`` but not in that one; otherwise ``None``.

    Kaggle trace (arXiv q10): the label "gpt-4o | No Doc" was right, but the value 52.0 was
    copied from the "gpt-4o-mini" row of the same grid.  The name check in ``ground`` accepts
    it ("gpt-4o" is a prefix of "gpt-4o-mini"); this looks up the cell the label names
    instead.  Deliberately conservative — no alignment when:

    * the value also occurs in prose (a sentence carries its own context) or not in a cell;
    * any cell of the passage holds several numbers (merged columns: positions are unsafe);
    * no row is named in full by the label, the best such row or column is tied, or the
      value sits on another row that the label also names in full (then it stays);
    * the named cell does not hold exactly one number.
    """
    text = text.translate(_DEVANAGARI_DIGITS)
    rows = _table_rows(text)
    if not rows:
        return None
    table_lines = {line for line, _, _ in rows}
    for line in text.splitlines():
        if line not in table_lines and any(_same(x, value) for x in _numbers_in(line)):
            return None
    if any(len(_numbers_in(cell)) > 1 for _, _, cells in rows for _, cell in cells):
        return None
    hits = [
        (r, header)
        for r, (_, _, cells) in enumerate(rows)
        for header, cell in cells
        if any(_same(x, value) for x in _numbers_in(cell))
    ]
    if not hits:
        return None
    want = _name_tokens(label)
    keys = [_row_key(row_label, want) for _, row_label, _ in rows]
    # Candidate rows are named in full by the label (every word of the row label is in it).
    full = [(k[0], r) for r, k in enumerate(keys) if k is not None and k[0] >= 1 and k[1] == 0]
    if not full:
        return None
    most = max(m for m, _ in full)
    best_rows = [r for m, r in full if m == most]
    if len(best_rows) != 1:
        return None
    row = best_rows[0]
    if any(r != row and r in {fr for _, fr in full} for r, _ in hits):
        return None  # the value's own row is also named by the label: leave it
    _, row_label, cells = rows[row]
    row_words = set(_words(row_label))
    residual = {t for t in _tokens(label) if not any(_match(t, w) for w in row_words)}
    numeric = [(h, c) for h, c in cells if _numbers_in(c)]
    scores = [(_overlap(residual, _tokens(h)), h, c) for h, c in numeric]
    if not scores:
        return None
    top = max(s for s, _, _ in scores)
    best_cols = [(h, c) for s, h, c in scores if s == top]
    if top < 1 or len(best_cols) != 1:
        return None
    header, cell = best_cols[0]
    if (row, header) in hits:
        return None  # the value already sits in the cell its label names
    found = _numbers_in(cell)
    if len(found) != 1:
        return None
    signed = -found[0] if re.search(r"[-−–(]\s*\d", cell) else found[0]
    return signed, row_label, header


# --------------------------------------------------------------------------- measures
#
# Kaggle traces (BEA release): the operand's row and period were right but the *measure*
# was not — a year-over-year change for an "annualized" rate (q10: 2.0/2.1 from Table 6
# "Percent Change from Quarter One Year Ago" instead of -0.6/3.8), a percent change for a
# question in billions of dollars (q09: 6.0/2.9 from Table 1).  Statistical releases state
# the measure in the table title, a unit line or the column header; the question states it
# in words.  Only clear contradictions count; anything unclassified passes.

_Q_DOLLARS_RE = re.compile(
    r"\b(in|of) (thousands|millions|billions|trillions) of dollars\b|\bin dollars\b"
    r"|\blevel\b|\$\s?\d",
    re.IGNORECASE,
)
_Q_RATE_RE = re.compile(
    r"\b(growth rate|annual(ized)? rate|annualized|percent change|percentage change"
    r"|at what (\w+ )?rate)\b",
    re.IGNORECASE,
)
_Q_ANNUALIZED_RE = re.compile(r"\b(annuali[sz]ed|(at an? )?annual rates?)\b", re.IGNORECASE)
_M_YOY_RE = re.compile(
    r"\b(from (the same )?(quarter|month|period) (one )?a? ?year (ago|earlier)"
    r"|year[- ]over[- ]year|from a year (ago|earlier)|from quarter one year ago)\b",
    re.IGNORECASE,
)
_M_CONTRIB_RE = re.compile(r"\b(contributions? to|percentage points?)\b", re.IGNORECASE)
_M_PERCENT_RE = re.compile(r"\b(percent change|percent|percentage change)\b", re.IGNORECASE)
_M_DOLLARS_RE = re.compile(
    r"\b(thousands|millions|billions|trillions) of (chained \(\d{4}\) )?dollars\b|\blevels?\b",
    re.IGNORECASE,
)


_Q_CONTRIB_RE = re.compile(r"\bcontribut", re.IGNORECASE)
_Q_NOMINAL_RE = re.compile(r"\b(current[- ]dollars?|nominal)\b", re.IGNORECASE)
_M_REAL_RE = re.compile(r"\bchained\b", re.IGNORECASE)


def question_measure(question: str) -> dict[str, bool]:
    """What the question's operands must be: ``dollars`` (a level or change in dollars),
    ``rate`` (a percent change) and/or ``annualized``.  A question that names two measures
    — a dollar amount and a rate, or a contribution and anything else (BEA q11: "how many
    percentage points did retail trade contribute ..., and at what annual rate ...") — is
    left unclassified for both (only ``annualized`` can still be set)."""
    dollars = bool(_Q_DOLLARS_RE.search(question))
    rate = bool(_Q_RATE_RE.search(question))
    if (dollars and rate) or _Q_CONTRIB_RE.search(question):
        dollars = rate = False
    return {
        "dollars": dollars,
        "rate": rate,
        "annualized": bool(_Q_ANNUALIZED_RE.search(question)),
        # "current-dollar GDP": not the chained (inflation-adjusted) dollar figures beside it
        "nominal": bool(_Q_NOMINAL_RE.search(question)),
    }


def _measure_of(text: str) -> str:
    """ "contribution" | "percent" | "dollars" | "" for a header, unit line or title."""
    if _M_CONTRIB_RE.search(text):
        return "contribution"
    if _M_PERCENT_RE.search(text):
        return "percent"
    if _M_DOLLARS_RE.search(text):
        return "dollars"
    return ""


def cell_measure(text: str, value: float | None = None) -> list[tuple[str, bool, bool, str]]:
    """``(measure, year_over_year, chained, context)`` of every table cell of ``text`` holding ``value``
    (every numeric cell when ``value`` is None).

    The most specific statement wins: the column header, then the row's section labels
    ("Percentage points at annual rates | Mining"), then the passage's non-table lines
    (the table title, e.g. "Table 6. ... Percent Change from Quarter One Year Ago").
    """
    text = text.translate(_DEVANAGARI_DIGITS)
    rows = _table_rows(text)
    table_lines = {line for line, _, _ in rows}
    title = " ".join(line for line in text.splitlines() if line not in table_lines)
    out = []
    for line, row_label, cells in rows:
        sections = (
            ""
            if line.lstrip().startswith("|")  # Markdown grid: no section labels on the line
            else " ".join(s for s in line.split(" | ")[:-1] if " = " not in s and s != row_label)
        )
        for header, cell in cells:
            numbers = _numbers_in(cell)
            if not numbers or (value is not None and not any(_same(x, value) for x in numbers)):
                continue
            levels = (header, sections, title)
            measure = next((m for m in map(_measure_of, levels) if m), "")
            yoy = any(_M_YOY_RE.search(t) for t in levels)
            chained = bool(_M_REAL_RE.search(f"{header} {sections}"))
            out.append((measure, yoy, chained, f"{title[:90]} | {header}".strip(" |")))
    return out


def _conflict(want: dict[str, bool], measure: str, yoy: bool, chained: bool, context: str) -> str:
    if want["dollars"] and measure in ("percent", "contribution"):
        return f'it is a {measure} figure ("{context}"), not an amount in dollars'
    if want["rate"] and measure in ("contribution", "dollars"):
        what = (
            "a contribution in percentage points"
            if measure == "contribution"
            else "a dollar amount"
        )
        return f'it is {what} ("{context}"), not a rate of change'
    if want["annualized"] and yoy:
        return f'it is a change from a year earlier ("{context}"), not an annualized rate'
    if want["nominal"] and chained:
        return f'it is in chained (inflation-adjusted) dollars ("{context}"), not current dollars'
    return ""


def measure_conflict(question: str, value: float, text: str) -> str:
    """Why ``value`` (a table cell of ``text``) cannot be the measure the question asks for,
    or "" when one of its cells fits (or nothing is known)."""
    want = question_measure(question)
    if not any(want.values()):
        return ""
    rows = _table_rows(text.translate(_DEVANAGARI_DIGITS))
    table_lines = {line for line, _, _ in rows}
    for line in text.translate(_DEVANAGARI_DIGITS).splitlines():
        if line not in table_lines and any(_same(x, value) for x in _numbers_in(line)):
            return ""  # stated in a sentence: the sentence carries its own measure
    reasons = [_conflict(want, *found) for found in cell_measure(text, value)]
    return reasons[0] if reasons and all(reasons) else ""


def table_conflicts(question: str, text: str) -> bool:
    """Is ``text`` a table passage none of whose numbers can be the measure the question
    asks for (every numeric cell conflicts, see ``measure_conflict``)?"""
    want = question_measure(question)
    if not any(want.values()):
        return False
    found = cell_measure(text)
    return bool(found) and all(_conflict(want, *f) for f in found)


def _label_names_value(value: float, label: str) -> bool:
    """Is ``value`` really part of a name in its own label ("gpt-4o" → 4, "Recall@10" → 10)?"""
    for m in _LABEL_NUM_RE.finditer(label):
        before = label[m.start() - 1] if m.start() else " "
        after = label[m.end()] if m.end() < len(label) else " "
        glued = before.isalnum() or before in "@-" or after.isalnum() or after == "@"
        if glued and _same(float(m.group()), value):
            return True
    return False


def _readable(text: str) -> bool:
    """Can numbers in this passage be checked?  Non-Latin text often writes them as words."""
    return lang.script_of(text) in ("latin", "unknown")


def ground(
    data: dict[str, Any],
    passages: list[str],
    align: bool = False,
    question: str = "",
) -> tuple[dict[str, Any], list[str], bool]:
    """Check an extraction against the passages: ``(data, problems, verified)``.

    Every operand must appear in the passage it cites (sign, commas and % ignored) and,
    on a table line, its label must name that row and not match another column better.
    A value found only in another passage gets that passage number instead (citation
    slips are common and harmless).  ``verified`` is False when a passage could not be
    checked (numbers written as words in a non-Latin script) — such results are hinted,
    never forced.  Free-form expressions may only use the listed operands.

    ``align`` (``RAGConfig.calc_cell_alignment``): an operand copied from a table cell
    other than the one its label names is replaced by that cell's value (``align_cell``);
    the old value is kept as ``"aligned_from"``.  ``question`` (with
    ``RAGConfig.calc_measure_check``): an operand from a table whose measure contradicts
    the question's is rejected (``measure_conflict``).
    """
    data = dict(data)
    problems: list[str] = []
    verified = True
    operation = str(data.get("operation", "") or "").strip().lower()
    n = len(passages)
    checkable = all(_readable(p) for p in passages)

    if operation == "count" or str(data.get("expression", "")).strip().lower() == "count":
        bags = [_tokens(p) for p in passages]
        for item in data.get("items", []) or []:
            want = _tokens(str(item))
            if want and not any(_overlap(want, bag) >= 0.6 * len(want) for bag in bags):
                if checkable:
                    problems.append(f'item "{str(item)[:60]}" is not in the passages')
                else:
                    verified = False
        return data, problems, verified

    if operation == "months_between":
        for d in data.get("dates", []) or []:
            year = str(d.get("year", "")) if isinstance(d, dict) else ""
            if year and not any(re.search(rf"(?<!\d){re.escape(year)}(?!\d)", p) for p in passages):
                if checkable:
                    problems.append(f"the year {year} is not in the passages")
                else:
                    verified = False
        return data, problems, verified

    values = []
    for raw in data.get("values", []) or []:
        if not isinstance(raw, dict):
            continue
        v = dict(raw)
        values.append(v)
        x = _to_float(v.get("value"))
        label = str(v.get("label", "") or "")
        if x is None:
            problems.append(f'value {v.get("value")!r} ("{label}") is not a number')
            continue
        if _label_names_value(x, label):
            problems.append(f'{_fmt(x)} is part of a name in its label "{label}", not a value')
            continue
        cited = v.get("passage") if isinstance(v.get("passage"), int) else None
        cited = cited if cited is not None and 1 <= cited <= n else None
        candidates = ([cited] if cited else []) + [i for i in range(1, n + 1) if i != cited]
        if align:
            for i in candidates:
                aligned = align_cell(x, label, passages[i - 1])
                if aligned is not None:
                    v["aligned_from"], v["value"], v["passage"] = x, aligned[0], i
                    x, cited = aligned[0], i
                    candidates = [i] + [j for j in candidates if j != i]
                    break
                if any(_same(y, x) for y in _numbers_in(passages[i - 1])):
                    break  # only the first passage that holds the value is realigned
        first_problem, found = "", False
        for i in candidates:
            hit, problem = _check_in_passage(x, label, passages[i - 1])
            if hit and not problem and question:
                problem = measure_conflict(question, x, passages[i - 1])
            if hit and not problem:
                v["passage"] = i
                break
            found = found or hit
            first_problem = first_problem or problem
        else:
            if found:
                problems.append(f'{_fmt(x)} ("{label}"): {first_problem}')
            elif not (_readable(passages[cited - 1]) if cited else checkable):
                verified = False  # e.g. a Marathi passage that writes 16 as "सोळा"
            else:
                where = f"passage {cited}" if cited else "the passages"
                problems.append(f'{_fmt(x)} ("{label}") does not appear in {where}')
    data["values"] = values

    expression = str(data.get("expression", "") or "")
    if operation not in OPERATIONS and expression:
        operands = [x for x in (_to_float(v.get("value")) for v in values) if x is not None]
        for m in _LABEL_NUM_RE.finditer(expression.replace(",", "")):
            if not any(_same(float(m.group()), x) for x in operands):
                problems.append(f"the expression uses {m.group()}, which is not one of the values")
    return data, problems, verified


def evaluate(
    data: dict[str, Any] | None,
    passages: list[str],
    align: bool = False,
    question: str = "",
) -> dict[str, Any] | None:
    """``run`` on grounded operands only (``align``/``question``: see ``ground``).

    Returns the calculation (with ``"verified"``), ``{"rejected": [reasons], ...}`` when an
    operand is not grounded, or ``None`` when no calculation is needed or possible.
    """
    if not data or not data.get("needed"):
        return None
    data, problems, verified = ground(data, passages, align=align, question=question)
    if problems:
        return {"rejected": problems[:6], "values": data.get("values", [])}
    calc = run(data)
    if calc is not None:
        calc["verified"] = verified
        aligned = [v for v in calc.get("values", []) if "aligned_from" in v]
        if aligned:
            calc["aligned"] = [
                f'{_fmt(float(v["aligned_from"]))} -> {_fmt(float(v["value"]))} ("{v.get("label", "")}")'
                for v in aligned
            ]
    return calc


# --------------------------------------------------------------------------- stated values

_CHANGE_VERB_RE = re.compile(
    r"\b(increased|decreased|rose|fell|declined|grew|dropped|gained|climbed|jumped|contracted"
    r"|expanded|contributed|subtracted|added|(?:was )?revised (?:up|down))\b"
    r"(?P<gap>[^.;\d]{0,40}?)(?P<num>[-−]?\$?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_NOT_SUBJECT_TEXT = """how much many what which did does do was were when level reach reached
    quarter year percent percentage point points rate annual first second third fourth q1 q2 q3 q4"""
_NOT_SUBJECT = frozenset(_NOT_SUBJECT_TEXT.split())
_CHANGE_STEMS_TEXT = (
    "increas decreas rose rise fell fall declin grew grow drop gain climb contribut chang revis add"
)
_CHANGE_STEMS = tuple(_CHANGE_STEMS_TEXT.split())


def stated_change(question: str, passages: list[str]) -> str:
    """A passage sentence that states the asked-for change ("profits ... increased $6.8 billion").

    The sentence must pair a change verb with a number — not "increased to X" or "from X to
    Y", which give levels — and share at least two subject words with the question.
    Returns the sentence, or "".
    """
    subject = {
        t
        for t in _WORD_RE.findall(question.lower())
        if t not in _STOP
        and t not in _NOT_SUBJECT
        and not t.isdigit()
        and not t.startswith(_CHANGE_STEMS)
    }
    if len(subject) < 2:
        return ""
    for text in passages:
        for sentence in re.split(r"(?<=[.;])\s+", text):
            stated = any(
                not re.search(r"\b(to|from)\b", m.group("gap"), re.IGNORECASE)
                for m in _CHANGE_VERB_RE.finditer(sentence)
            )
            if stated and _overlap(subject, set(_WORD_RE.findall(sentence.lower()))) >= 2:
                return sentence.strip()[:200]
    return ""


# --------------------------------------------------------------------------- dates

_SPAN_Q_RE = re.compile(
    r"\bhow (many|much) (months|years|weeks|days|time)\b|\bhow long\b", re.IGNORECASE
)
_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_DATE_RE = re.compile(
    r"\b(" + "|".join(m.capitalize() for m in _MONTHS) + r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?"
    r"|Oct|Nov|Dec)\.?\s+(?:\d{1,2},?\s+)?((?:19|20)\d\d)\b"
)
DATES_NOTE = """
Dates found in the passages (month year, passage, where):
{dates}
A date on the title page (page 1-2) of a document is normally its publication date.
"""


def asks_time_span(question: str) -> bool:
    return bool(_SPAN_Q_RE.search(question))


def date_candidates(passages: list[tuple[int, str]], limit: int = 8) -> str:
    """Prompt note listing the month-year dates in the passages (``(page, text)`` pairs).

    Kaggle trace (NIST q11, "how many months passed between the release of the AI RMF 1.0
    and the publication of this profile?"): both dates were retrieved — "released in
    January 2023" on p5 and "July 2024" alone on the title page — but the calculator call
    twice returned no calculation and the generator refused ("the passages do not provide
    specific dates").  Listing the dates, with the title-page one named as the publication
    date, gives the extraction step the operands it missed.  "" when there are none.
    """
    seen: set[tuple[int, int]] = set()
    lines = []
    for number, (page, text) in enumerate(passages, start=1):
        for m in _DATE_RE.finditer(text):
            name = m.group(1).lower()
            month = next(i for i, full in enumerate(_MONTHS, start=1) if full.startswith(name[:3]))
            year = int(m.group(2))
            if (year, month) in seen:
                continue
            seen.add((year, month))
            start = max(0, m.start() - 60)
            context = " ".join(text[start : m.end() + 20].split())
            where = "title page" if page <= 2 else f"page {page}"
            lines.append(
                f"- {_MONTHS[month - 1].capitalize()} {year} (month {month}): passage "
                f'{number}, {where}: "...{context}..."'
            )
            if len(lines) >= limit:
                return DATES_NOTE.format(dates="\n".join(lines))
    return DATES_NOTE.format(dates="\n".join(lines)) if lines else ""


# --------------------------------------------------------------------------- identifier counts

_ID_RE = re.compile(r"\b([A-Z]{1,6}-(\d+(?:\.\d+)*))-(\d{1,4})\b")
_COUNT_Q_RE = re.compile(
    r"\bhow many\b(?! (percentage|points?|times|months|years|days|weeks|more|fewer)\b)",
    re.IGNORECASE,
)


def id_prefix(question: str, passages: list[str]) -> str:
    """For "how many ... under MEASURE 2.7?": the ID family ("MS-2.7") the passages list.

    The family's number must be named in the question and exactly one family may match;
    otherwise "" (the LLM counts as before).
    """
    if not _COUNT_Q_RE.search(question):
        return ""
    numbers = set(re.findall(r"(?<![\d.])\d+(?:\.\d+)*(?![\d.]*\d)", question))
    families = {
        m.group(1) for text in passages for m in _ID_RE.finditer(text) if m.group(2) in numbers
    }
    return families.pop() if len(families) == 1 else ""


def ids_with_prefix(text: str, prefix: str) -> set[str]:
    return {m.group(0) for m in _ID_RE.finditer(text) if m.group(1) == prefix}


def count_ids(prefix: str, passages: list[str]) -> dict[str, Any] | None:
    """Count the distinct ``<prefix>-NNN`` identifiers in the passages (no LLM)."""
    found: dict[str, int] = {}
    for i, text in enumerate(passages, start=1):
        for ident in ids_with_prefix(text, prefix):
            found.setdefault(ident, i)
    if not found:
        return None
    items = sorted(found, key=lambda s: int(s.rsplit("-", 1)[1]))
    return {
        "expression": "count",
        "result": str(len(items)),
        "unit": "",
        "items": items,
        "values": [{"passage": p} for p in sorted(set(found.values()))],
        "verified": True,
    }


def safe_eval(expression: str) -> float:
    """Evaluate ``+ - * /`` over numeric literals; anything else raises ``CalculatorError``."""
    cleaned = expression.replace(",", "").replace("−", "-").replace("×", "*").replace("÷", "/")
    cleaned = re.sub(r"(?<=\d)\s*%", "", cleaned)
    if len(cleaned) > 200:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"not an arithmetic expression: {expression!r}") from exc

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise CalculatorError("division by zero")
            return _OPS[type(node.op)](left, right)
        raise CalculatorError(f"unsupported element {type(node).__name__} in {expression!r}")

    return walk(tree)


def format_number(value: float, operands: list[float] | None = None) -> str:
    """Round to the precision of the operands (at least 1 decimal unless the result is whole)."""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}" if abs(value) >= 10000 else str(int(round(value)))
    decimals = 1
    for x in operands or []:
        text = repr(float(x))
        if "." in text and not text.endswith(".0"):
            decimals = max(decimals, len(text.split(".")[1]))
    return f"{round(value, min(decimals, 4)):.{min(decimals, 4)}f}"


def _numbers(values: list[dict[str, Any]]) -> list[float]:
    out = []
    for v in values:
        x = v.get("value")
        if isinstance(x, bool):
            continue
        if isinstance(x, int | float):
            out.append(float(x))
        elif isinstance(x, str):
            try:
                out.append(float(x.replace(",", "").replace("%", "").strip()))
            except ValueError:
                continue
    return out


def _apply(operation: str, nums: list[float]) -> tuple[float, str] | None:
    """(result, human-readable expression) for a structured operation, or None if not applicable."""
    if operation in ("difference", "change", "ratio", "percent_change") and len(nums) != 2:
        return None
    a = nums[0] if nums else 0.0
    b = nums[1] if len(nums) > 1 else 0.0
    if operation == "difference":
        return abs(a - b), f"|{_fmt(a)} - {_fmt(b)}|"
    if operation == "change":
        return b - a, f"{_fmt(b)} - {_fmt(a)}"
    if operation == "sum" and nums:
        return sum(nums), " + ".join(_fmt(x) for x in nums)
    if operation == "product" and nums:
        return math.prod(nums), " x ".join(_fmt(x) for x in nums)
    if operation == "ratio" and b != 0:
        return a / b, f"{_fmt(a)} / {_fmt(b)}"
    if operation == "percent_change" and a != 0:
        return (b - a) / a * 100, f"({_fmt(b)} - {_fmt(a)}) / {_fmt(a)} x 100"
    return None


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else repr(x)


def run(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compute the result from the extraction JSON; ``None`` when not needed or unusable."""
    if not data or not data.get("needed"):
        return None
    operation = str(data.get("operation", "") or "").strip().lower()
    expression = str(data.get("expression", "") or "").strip()
    unit = str(data.get("unit", "") or "").strip()
    values = [v for v in data.get("values", []) if isinstance(v, dict)]

    if operation == "count" or expression.lower() == "count":
        items = [str(i).strip() for i in data.get("items", []) if str(i).strip()]
        if not items:
            return None
        unique = list(dict.fromkeys(items))
        return {
            "expression": "count",
            "result": str(len(unique)),
            "unit": unit,
            "items": unique,
            "values": values,
        }

    if operation == "months_between":
        dates = [d for d in data.get("dates", []) if isinstance(d, dict)]
        try:
            stamps = [int(d["year"]) * 12 + int(d["month"]) for d in dates[:2]]
        except (KeyError, TypeError, ValueError):
            return None
        if len(stamps) != 2 or not all(1 <= int(d["month"]) <= 12 for d in dates[:2]):
            return None
        months = abs(stamps[1] - stamps[0])
        expr = " to ".join(f"{int(d['year'])}-{int(d['month']):02d}" for d in dates[:2])
        cites = [{"passage": d["passage"]} for d in dates[:2] if isinstance(d.get("passage"), int)]
        return {
            "expression": expr,
            "result": str(months),
            "unit": unit or "months",
            "values": cites,
        }

    nums = _numbers(values)
    if operation == "sum" and len(nums) == 2 == len(values):
        per_group = [bool(_PER_GROUP_RE.search(str(v.get("label", "")))) for v in values]
        if sum(per_group) == 1:  # "16 petals" + "per petal: 16 jets" is 16 x 16 (Marathi q05)
            operation = "product"
    if operation in OPERATIONS:
        applied = _apply(operation, nums)
        if applied is None:
            return None
        result, expression = applied
    elif expression:  # free-form arithmetic expression
        try:
            result = safe_eval(expression)
        except (CalculatorError, OverflowError):
            return None
    else:
        return None
    if not math.isfinite(result):
        return None
    if operation in ("ratio", "percent_change"):
        # Operand precision does not bound a quotient's precision (3/4 must not become 0.8).
        formatted = f"{round(result, 2):.2f}".rstrip("0").rstrip(".")
    else:
        formatted = format_number(result, nums)
    return {"expression": expression, "result": formatted, "unit": unit, "values": values}


def result_in_answer(answer: str, calc: dict[str, Any]) -> bool:
    """Does the answer already state the computed result (ignoring thousands separators)?"""
    target = calc["result"].replace(",", "")
    text = answer.replace(",", "")
    return bool(re.search(rf"(?<![\d.]){re.escape(target)}(?![\d])", text))


def answer_suffix(calc: dict[str, Any]) -> str:
    """Sentence appended when the generator ignored the computed result."""
    sources = sorted(
        {int(v["passage"]) for v in calc.get("values", []) if isinstance(v.get("passage"), int)}
    )
    cite = "".join(f"[{n}]" for n in sources)
    unit = f" {calc['unit']}" if calc.get("unit") else ""
    if calc["expression"] == "count":
        return f" (Count: {calc['result']}{unit}.{(' ' + cite) if cite else ''})"
    return f" (Calculated: {calc['expression']} = {calc['result']}{unit}.{(' ' + cite) if cite else ''})"


def hint(calc: dict[str, Any], forced: bool = True) -> str:
    """Instruction appended to the generator prompt.

    ``forced=False`` (the question may be a lookup, or an operand could not be checked):
    the generator is told to prefer a value the passages state.
    """
    sources = sorted(
        {int(v["passage"]) for v in calc.get("values", []) if isinstance(v.get("passage"), int)}
    )
    cite = "".join(f"[{n}]" for n in sources)
    if calc["expression"] == "count":
        what = f"counted {calc['result']} distinct items: " + "; ".join(calc["items"][:20])
    else:
        what = f"computed {calc['expression']} = {calc['result']}"
    unit = f" {calc['unit']}" if calc.get("unit") else ""
    if not forced:
        return (
            f"- A calculator {what}{unit} using values from the passages {cite}. Use this "
            "result only if the question asks for exactly this calculation; if a passage "
            "states the requested value directly, give that value instead."
        )
    return (
        f"- A calculator {what}{unit} using values from the passages {cite}. "
        "Use this result in your answer (do not redo the arithmetic) and cite those passages."
    )
