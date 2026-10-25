"""
Statistical-table serialisation from word positions.

Government and financial reports (BEA, BLS, Census, annual reports) lay out
wide numeric tables using whitespace alone — no row rules.  For these,
``page.find_tables`` merges whole columns into single cells
(``2.5<br>2.9<br>2.8``) and plain text extraction emits a vertical soup of
numbers, so neither retrieval nor a 7B model can tell which value belongs
to which row/column.

This module rebuilds the grid geometrically and writes each data row as
self-describing text:

    Percent change at annual rate | Gross domestic product | 2024 = 2.8; 2025 Q1 = -0.6; 2025 Q2 = 3.8

    1. words → rows        (cluster by vertical centre)
    2. rows  → cells       (merge words separated by small gaps)
    3. numeric cells of the data rows → column anchors (cluster right edges,
       numbers are right-aligned)
    4. header rows (above the first labelled data row) → column labels.  Each
       header label is given a contiguous *span* of columns:
         - from the vertical rules around it, when the header is ruled;
         - otherwise geometrically: a label with nothing beneath it is a leaf
           (its cell is vertically centred, so it also occupies the header rows
           mirrored above it), and every other label gets the widest span that
           is centred on it, holds its text, nests with the spans of the rows
           below and does not overlap its neighbours in the same row.
    5. data rows → "section | label | header = value; ..." lines, joining row
       labels that wrap over several lines and dropping footnote markers.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

_NUMBER_RE = re.compile(
    r"^[(\-–−+$€£]?\d[\d,]*(\.\d+)?%?\)?[*†‡]?$|^[(\-–−]?\.\d+\)?$|^\(?[A-Z]?\)$"
)
_NUM_TOKEN_RE = re.compile(r"(?<![\w.])[-−–(]?\d[\d,]*(?:\.\d+)?%?\)?(?![\w.])")
_YEAR_RE = re.compile(r"^(19|20)\d\d$")
_NA_RE = re.compile(r"^(\(?[A-Z]\)|n\.?a\.?|—|–|-|\.\.\.|…|\*+|x)$", re.IGNORECASE)


@dataclass
class _Cell:
    text: str
    x0: float
    x1: float
    y0: float = 0.0
    y1: float = 0.0

    @property
    def numeric(self) -> bool:
        t = self.text.replace(" ", "")
        return bool(_NUMBER_RE.match(t)) or bool(_NA_RE.match(t))


def is_garbled_markdown(md: str) -> bool:
    """find_tables output where many cells contain stacked values (merged columns)."""
    cells = [c for line in md.splitlines() for c in line.split("|") if c.strip()]
    if not cells:
        return True
    stacked = sum(("<br>" in c or "&lt;br&gt;" in c) for c in cells)
    return stacked / len(cells) > 0.15


def is_garbled_rows(rows: list[list[str]]) -> bool:
    """Table rows where many cells hold several numbers (columns merged into one cell)."""
    cells = [c for row in rows for c in row if c and c.strip()]
    if not cells:
        return True
    stacked = sum(len(_NUM_TOKEN_RE.findall(c)) >= 3 for c in cells)
    return stacked / len(cells) > 0.15


def numeric_word_ratio(words: list[tuple]) -> float:
    if not words:
        return 0.0
    return sum(bool(_NUMBER_RE.match(w[4])) for w in words) / len(words)


def vertical_rules(page, bbox: tuple) -> list[tuple[float, float, float]]:
    """Vertical ruling segments ``(x, y0, y1)`` drawn inside ``bbox`` on a PyMuPDF page.

    BEA-style tables rule their header cells even when the body is whitespace-aligned;
    the rules give the exact extent of each spanning header.
    """
    x0, y0, x1, y1 = bbox
    out: list[tuple[float, float, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return out
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":
                p, q = item[1], item[2]
                if abs(p.x - q.x) > 1.0 or abs(p.y - q.y) < 3.0:
                    continue
                x, top, bottom = (p.x + q.x) / 2, min(p.y, q.y), max(p.y, q.y)
            elif item[0] == "re":
                r = item[1]
                if r.width > 1.5 or r.height < 3.0:
                    continue
                x, top, bottom = (r.x0 + r.x1) / 2, r.y0, r.y1
            else:
                continue
            if x0 - 1 <= x <= x1 + 1 and bottom >= y0 and top <= y1:
                out.append((x, top, bottom))
    return out


def split_stacked_cells(cells: list[list[tuple]]) -> list[list[list[tuple]]]:
    """Split one detected table row whose cells stack several visual rows.

    Booktabs-style tables (rules only above/below the body) make
    ``page.find_tables`` merge consecutive rows into one, giving cells like
    ``"Dragon RoBERTa Dragon+ COCO-DR"`` next to ``"0.55 0.54 0.50"`` — the value
    for each model is then only recoverable by position.  When several cells hold
    the same number of text lines on the same baselines and at least one of them
    is a stack of single numbers, each line is a row of its own.

    ``cells`` holds the PyMuPDF word tuples of each cell; returns the rows.
    """
    heights = [w[3] - w[1] for ws in cells for w in ws if w[3] > w[1]]
    if not heights:
        return [cells]
    tol = 0.45 * statistics.median(heights)

    def lines(ws: list[tuple]) -> list[list[tuple]]:
        out: list[list[tuple]] = []
        for w in sorted(ws, key=lambda w: (w[1] + w[3]) / 2):
            cy = (w[1] + w[3]) / 2
            if out and abs(cy - statistics.mean((v[1] + v[3]) / 2 for v in out[-1])) <= tol:
                out[-1].append(w)
            else:
                out.append([w])
        return out

    per_cell = [lines(ws) for ws in cells]
    k = max((len(ls) for ls in per_cell), default=0)
    if k < 2:
        return [cells]
    stacked = [ls for ls in per_cell if len(ls) == k]
    filled = [ls for ls in per_cell if ls]
    if len(stacked) < 2 or len(stacked) < 0.5 * len(filled):
        return [cells]
    ys = [statistics.mean((w[1] + w[3]) / 2 for w in line) for line in stacked[0]]
    for ls in stacked[1:]:
        own = [statistics.mean((w[1] + w[3]) / 2 for w in line) for line in ls]
        if any(abs(a - b) > tol for a, b in zip(ys, own, strict=True)):
            return [cells]
    if not any(
        all(len(line) == 1 and _NUMBER_RE.match(line[0][4]) for line in ls) for ls in stacked
    ):
        return [cells]
    rows: list[list[list[tuple]]] = [[[] for _ in cells] for _ in ys]
    for c, ws in enumerate(cells):
        for w in ws:
            cy = (w[1] + w[3]) / 2
            rows[min(range(len(ys)), key=lambda i: abs(ys[i] - cy))][c].append(w)
    return rows


# Words that leave a row label unfinished ("Payments to the rest of the" / "world").
_CONNECTORS = frozenset(
    [
        "a",
        "an",
        "and",
        "at",
        "by",
        "except",
        "excluding",
        "for",
        "from",
        "in",
        "including",
        "less",
        "nor",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    ]
)
_FOOTNOTE_RE = re.compile(r"^(\d{1,2}|[*†‡§]+)$")


@dataclass
class _Column:
    right: float  # anchor: numbers are right-aligned
    centre: float
    left: float  # left edge of the widest value


@dataclass
class _Label:
    row: int  # index into the header rows, top to bottom
    text: str
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def serialize_table_words(
    words: list[tuple],
    title: str = "",
    vrules: list[tuple[float, float, float]] | None = None,
) -> list[str]:
    """Serialise a table region's words (PyMuPDF ``get_text('words')`` tuples) into row lines.

    ``vrules`` are optional vertical ruling segments ``(x, y0, y1)`` in the same
    coordinates; when the header is ruled they delimit the spanning header cells
    exactly.  Returns an empty list if the region does not look like a numeric grid.
    """
    rows = _rows_of_cells(words)
    if len(rows) < 3:
        return []
    if sum(c.numeric for row in rows for c in row) < 6:
        return []

    def label_cells(row: list[_Cell]) -> list[_Cell]:
        return [c for c in row if not c.numeric]

    def label_of(row: list[_Cell]) -> str:
        return " ".join(c.text for c in label_cells(row)).strip()

    def is_data_row(row: list[_Cell]) -> bool:
        nums = [c for c in row if c.numeric and not _is_line_number(c, row)]
        # Rows whose only "numbers" are years are column headers (e.g. "2022 2023 2024").
        return (
            len(label_of(row)) > 2
            and len(nums) >= 2
            and not all(_YEAR_RE.match(c.text) for c in nums)
        )

    first_data = next((i for i, row in enumerate(rows) if is_data_row(row)), None)
    if first_data is None:
        return []
    # Column anchors come from the body only: header years and the BEA "Line"
    # number columns must not become columns of their own.
    body_numbers = [
        c for row in rows[first_data:] for c in row if c.numeric and not _is_line_number(c, row)
    ]
    if len(body_numbers) < 6:
        return []
    columns = _columns(body_numbers)
    if len(columns) < 2:
        return []
    anchors = [col.right for col in columns]
    centres = [col.centre for col in columns]
    gap = _median_gap(centres)
    align_tol = max(6.0, 0.3 * gap)

    def col_of(cell: _Cell) -> int:
        return min(range(len(anchors)), key=lambda i: abs(anchors[i] - cell.x1))

    # Row labels may wrap over several lines; label-only lines are held in
    # ``pending`` until it is clear whether they start a wrapped label or a section.
    section, section_x0 = "", 0.0
    pending: list[str] = []
    pending_x0 = 0.0

    def take_pending(label: str, x0: float) -> str:
        """Resolve held label lines against the next labelled row; returns its full label."""
        nonlocal section, section_x0, pending
        if not pending:
            return label
        held = " ".join(pending)
        pending = []
        if not label or _continues(held, label):
            return f"{held} {label}".strip()
        section, section_x0 = held.rstrip(":"), pending_x0
        return label

    def hold(text: str, x0: float) -> None:
        nonlocal pending, pending_x0, section, section_x0
        if pending and _continues(" ".join(pending), text):
            pending.append(text)
            return
        if pending:  # held lines followed by an unrelated label line: they were a section
            section, section_x0 = " ".join(pending).rstrip(":"), pending_x0
        pending, pending_x0 = [text], x0

    header_rows: list[list[_Cell]] = []
    for row in rows[:first_data]:
        cells = [c for c in row if not _is_line_number(c, row) and c.text.lower() != "line"]
        if not cells:
            continue
        if len(cells) == 1 and not cells[0].numeric and cells[0].x1 < min(centres) - gap / 2:
            # A label alone in the stub column (e.g. "Percent change:") is a section or the
            # first line of a wrapped row label, not a column header.
            hold(cells[0].text, cells[0].x0)
            continue
        header_rows.append(cells)
    col_names = _header_names(header_rows, columns, vrules or [])

    entries: list[list[str]] = []  # [section, label, pairs]
    last_was_data = False
    for row in rows[first_data:]:
        label = label_of(row)
        label_x0 = min((c.x0 for c in label_cells(row)), default=0.0)
        values = [
            c
            for c in row
            if c.numeric
            and not _is_line_number(c, row)
            and not _is_footnote_marker(c, row, anchors, align_tol)
        ]
        if not values:
            if not label:
                continue
            if not pending and last_was_data and label[:1].islower():
                # Values printed on the first line of a wrapped label: the rest follows below.
                entries[-1][1] = f"{entries[-1][1]} {label}".strip()
            else:
                hold(label, label_x0)
                last_was_data = False
            continue
        if pending:
            label = take_pending(label, label_x0)
        elif section and label and label_x0 < section_x0 - 1.0:
            section = ""  # outdented past the section heading: the section has ended
        pairs = [f"{col_names[col_of(c)]} = {c.text}" for c in values]
        entries.append([section, label, "; ".join(pairs)])
        last_was_data = True

    lines = []
    for sec, label, pairs in entries:
        prefix = " | ".join(p for p in (sec, label) if p)
        lines.append(f"{prefix} | {pairs}" if prefix else pairs)
    if title and lines:
        lines.insert(0, title)
    return lines


# ---------------------------------------------------------------------------


def _continues(held: str, nxt: str) -> bool:
    """Whether label line ``nxt`` continues the unfinished label ``held``."""
    held, nxt = held.strip(), nxt.strip()
    if not held or not nxt or held.endswith(":"):
        return False
    if nxt[0].islower():
        return True
    return held.endswith((",", "-", "/", "&")) or held.split()[-1].lower() in _CONNECTORS


def _is_footnote_marker(cell: _Cell, row: list[_Cell], anchors: list[float], tol: float) -> bool:
    """A superscript note number glued to a row label ("industries¹"), not a value."""
    if not _FOOTNOTE_RE.match(cell.text):
        return False
    i = next(k for k, c in enumerate(row) if c is cell)
    if i == 0 or row[i - 1].numeric:
        return False
    return min(abs(a - cell.x1) for a in anchors) > tol


def _rows_of_cells(words: list[tuple]) -> list[list[_Cell]]:
    if not words:
        return []
    heights = [w[3] - w[1] for w in words if w[3] > w[1]]
    h = statistics.median(heights) if heights else 8.0
    ordered = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    current: list[tuple] = []
    current_y = None
    for w in ordered:
        cy = (w[1] + w[3]) / 2
        if current_y is None or abs(cy - current_y) <= h * 0.45:
            current.append(w)
            current_y = (
                cy if current_y is None else (current_y * (len(current) - 1) + cy) / len(current)
            )
        else:
            rows.append(current)
            current, current_y = [w], cy
    if current:
        rows.append(current)

    gap_limit = h * 0.55
    result: list[list[_Cell]] = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        cells: list[_Cell] = []
        for w in row:
            text = w[4]
            is_num = bool(_NUMBER_RE.match(text)) or bool(_NA_RE.match(text))
            if cells and w[0] - cells[-1].x1 <= gap_limit and not is_num and not cells[-1].numeric:
                cells[-1].text += " " + text
                cells[-1].x1 = w[2]
                cells[-1].y0 = min(cells[-1].y0, w[1])
                cells[-1].y1 = max(cells[-1].y1, w[3])
            else:
                cells.append(_Cell(text=text, x0=w[0], x1=w[2], y0=w[1], y1=w[3]))
        result.append(_merge_inline_numbers(cells, gap_limit))
    return result


def _merge_inline_numbers(cells: list[_Cell], gap_limit: float) -> list[_Cell]:
    """Re-join a number set tightly inside a phrase ("Billions of chained (2017) dollars")."""
    out: list[_Cell] = []
    i = 0
    while i < len(cells):
        if (
            out
            and i + 1 < len(cells)
            and cells[i].numeric
            and not out[-1].numeric
            and not cells[i + 1].numeric
            and cells[i].x0 - out[-1].x1 <= gap_limit
            and cells[i + 1].x0 - cells[i].x1 <= gap_limit
        ):
            prev, num, nxt = out[-1], cells[i], cells[i + 1]
            prev.text = f"{prev.text} {num.text} {nxt.text}"
            prev.x1 = nxt.x1
            prev.y0, prev.y1 = min(prev.y0, num.y0, nxt.y0), max(prev.y1, num.y1, nxt.y1)
            i += 2
            continue
        out.append(cells[i])
        i += 1
    return out


def _columns(numeric_cells: list[_Cell], tol: float = 6.0) -> list[_Column]:
    """Cluster numeric cells by right edge (numbers are right-aligned) → one column per cluster."""
    cells = sorted(numeric_cells, key=lambda c: c.x1)
    clusters: list[list[_Cell]] = []
    for c in cells:
        if clusters and c.x1 - clusters[-1][-1].x1 <= tol:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    # A real column has values in several rows.
    min_size = max(2, int(0.15 * max(len(c) for c in clusters)))
    return [
        _Column(
            right=statistics.median(c.x1 for c in cl),
            centre=statistics.median((c.x0 + c.x1) / 2 for c in cl),
            left=min(c.x0 for c in cl),
        )
        for cl in clusters
        if len(cl) >= min_size
    ]


def _header_names(
    header_rows: list[list[_Cell]],
    columns: list[_Column],
    vrules: list[tuple[float, float, float]],
) -> list[str]:
    """Column index → full header text ("Level Seasonally adjusted at annual rates 2025 Q2")."""
    n = len(columns)
    labels = [
        _Label(r, c.text, c.x0, c.x1, c.y0, c.y1)
        for r, cells in enumerate(header_rows)
        for c in cells
    ]
    names: list[list[tuple[int, str]]] = [[] for _ in range(n)]
    if labels:
        for label, (a, b) in _header_spans(labels, len(header_rows), columns, vrules):
            for i in range(a, b + 1):
                if all(t != label.text for _, t in names[i]):
                    names[i].append((label.row, label.text))
    return [
        " ".join(t for _, t in sorted(parts, key=lambda p: p[0])) if parts else f"col{i + 1}"
        for i, parts in enumerate(names)
    ]


def _header_spans(
    labels: list[_Label],
    n_rows: int,
    columns: list[_Column],
    vrules: list[tuple[float, float, float]],
) -> list[tuple[_Label, tuple[int, int]]]:
    """Assign each header label a contiguous column span ``(first, last)``."""
    n = len(columns)
    centres = [c.centre for c in columns]
    pitch = _median_gap(centres)
    tol = max(2.0, 0.25 * pitch)

    # Column regions: boundaries halfway through the whitespace between columns.
    pads = [columns[i + 1].left - columns[i].right for i in range(n - 1)]
    pad = max(1.0, statistics.median(pads) / 2) if pads else 2.0
    left_b = [columns[0].left - pad] + [
        (columns[i - 1].right + columns[i].left) / 2 for i in range(1, n)
    ]
    right_b = left_b[1:] + [columns[-1].right + pad]

    def under(label: _Label, i: int) -> bool:
        overlap = min(label.x1, right_b[i]) - max(label.x0, left_b[i])
        return overlap >= 0.5 * min(label.x1 - label.x0, right_b[i] - left_b[i])

    by_row: list[list[_Label]] = [[] for _ in range(n_rows)]
    for lab in labels:
        by_row[lab.row].append(lab)
    cols_under = {id(lab): [i for i in range(n) if under(lab, i)] for lab in labels}
    # Lowest header row (by row index) holding a label over each column.
    lowest = [
        max((lab.row for lab in labels if i in cols_under[id(lab)]), default=-1) for i in range(n)
    ]

    # Leaves: labels with nothing beneath them in any of their columns.  Their cell
    # reaches down to the bottom of the header and, being vertically centred, as far
    # above the label as below it — so other header rows in that band do not reach
    # those columns (e.g. "2024" for an annual column next to quarterly ones).
    header_bottom = max(lab.y1 for lab in by_row[n_rows - 1]) if by_row[n_rows - 1] else 0.0
    row_y = [statistics.median(lab.cy for lab in rl) if rl else 0.0 for rl in by_row]
    blocked = [[False] * n for _ in range(n_rows)]
    pinned: dict[int, tuple[int, int]] = {}
    for lab in labels:
        under_cols = cols_under[id(lab)]
        if not under_cols or any(lowest[i] > lab.row for i in under_cols):
            continue
        pinned[id(lab)] = (min(under_cols), max(under_cols))
        top = 2 * lab.cy - header_bottom
        for r in range(n_rows):
            if r != lab.row and row_y[r] >= top:
                for i in under_cols:
                    blocked[r][i] = True

    ruled = _ruled_spans(labels, columns, left_b, right_b, vrules)

    spans_below: list[tuple[int, int]] = []
    out: list[tuple[_Label, tuple[int, int]]] = []
    for r in range(n_rows - 1, -1, -1):  # bottom-up: spans must nest in the rows below
        row_labels = sorted(by_row[r], key=lambda lab: lab.cx)
        options: list[list[tuple[int, int, float]]] = []
        for lab in row_labels:
            if id(lab) in ruled:
                a, b = ruled[id(lab)]
                options.append([(a, b, float(b - a + 1))])
            elif id(lab) in pinned:
                a, b = pinned[id(lab)]
                options.append([(a, b, float(b - a + 1))])
            else:
                options.append(
                    _span_candidates(lab, blocked[r], left_b, right_b, spans_below, tol)
                    or _nearest_column(lab, blocked[r], centres, pitch)
                )
        chosen = _choose_spans(options)
        for lab, span in zip(row_labels, chosen, strict=True):
            if span is not None:
                out.append((lab, span))
                spans_below.append(span)
    return out


def _ruled_spans(
    labels: list[_Label],
    columns: list[_Column],
    left_b: list[float],
    right_b: list[float],
    vrules: list[tuple[float, float, float]],
) -> dict[int, tuple[int, int]]:
    """Spans read off vertical rules framing each header label (ruled headers only)."""
    if not vrules:
        return {}
    lo_edge, hi_edge = left_b[0], right_b[-1]

    def rules_at(y: float) -> list[float]:
        return [x for x, y0, y1 in vrules if y0 - 1.0 <= y <= y1 + 1.0]

    # Only trust rules that actually split the numeric area (not just a stub separator).
    if not any(lo_edge + 2 < x < hi_edge - 2 for lab in labels for x in rules_at(lab.cy)):
        return {}
    spans: dict[int, tuple[int, int]] = {}
    for lab in labels:
        xs = rules_at(lab.cy)
        left = max((x for x in xs if x <= lab.x0 + 1.0), default=None)
        right = min((x for x in xs if x >= lab.x1 - 1.0), default=None)
        if left is None and right is None:
            continue
        left = -float("inf") if left is None else left
        right = float("inf") if right is None else right
        inside = [i for i, c in enumerate(columns) if left <= c.centre <= right]
        if inside and inside == list(range(inside[0], inside[-1] + 1)):
            spans[id(lab)] = (inside[0], inside[-1])
    return spans


def _span_candidates(
    lab: _Label,
    blocked: list[bool],
    left_b: list[float],
    right_b: list[float],
    spans_below: list[tuple[int, int]],
    tol: float,
) -> list[tuple[int, int, float]]:
    """Spans centred on the label, wide enough for its text and nesting with lower rows."""
    n = len(blocked)
    width = lab.x1 - lab.x0
    out = []
    for a in range(n):
        if blocked[a]:
            continue
        for b in range(a, n):
            if blocked[b]:
                break
            lo, hi = left_b[a], right_b[b]
            err = abs((lo + hi) / 2 - lab.cx)
            if err > tol or hi - lo < 0.9 * width:
                continue
            # A spanning label must cover each lower span entirely or not at all.
            if any(not (d < a or c > b or (a <= c and d <= b)) for c, d in spans_below):
                continue
            # Prefer the widest centred span; centring error breaks ties.
            out.append((a, b, (b - a + 1) - 0.5 * (err / tol) ** 2))
    return out


def _nearest_column(
    lab: _Label, blocked: list[bool], centres: list[float], pitch: float
) -> list[tuple[int, int, float]]:
    free = [i for i in range(len(centres)) if not blocked[i]]
    if not free:
        return []
    i = min(free, key=lambda k: abs(centres[k] - lab.cx))
    return [(i, i, 0.5)] if abs(centres[i] - lab.cx) <= 0.6 * pitch else []


def _choose_spans(options: list[list[tuple[int, int, float]]]) -> list[tuple[int, int] | None]:
    """Pick one span per label (labels left to right, spans may not overlap), maximising score."""
    drop_penalty = 1000.0
    # state: last column used → (score, choices)
    states: dict[int, tuple[float, list[tuple[int, int] | None]]] = {-1: (0.0, [])}
    for opts in options:
        nxt: dict[int, tuple[float, list[tuple[int, int] | None]]] = {}
        for last, (score, picks) in states.items():
            moves = [(last, score - drop_penalty, None)]
            moves += [(b, score + value, (a, b)) for a, b, value in opts if a > last]
            for key, new_score, pick in moves:
                if key not in nxt or new_score > nxt[key][0]:
                    nxt[key] = (new_score, [*picks, pick])
        states = nxt
    return max(states.values(), key=lambda s: s[0])[1]


def _median_gap(anchors: list[float]) -> float:
    gaps = [b - a for a, b in zip(anchors, anchors[1:], strict=False)]
    return statistics.median(gaps) if gaps else 20.0


def _is_line_number(cell: _Cell, row: list[_Cell]) -> bool:
    if not re.fullmatch(r"\d{1,3}", cell.text):
        return False
    return cell is row[0] or cell is row[-1]
