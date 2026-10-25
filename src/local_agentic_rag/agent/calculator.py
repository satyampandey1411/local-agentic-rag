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
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

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

Decide which calculation answers the question and copy the needed numbers EXACTLY as written in the passages (check the row AND the column/period label of every table value). Respond in JSON with:
- "needed": true if the answer requires arithmetic, counting, or a time span; otherwise false
- "operation": one of
    "difference"      absolute gap between two values ("how much higher/lower", "difference", "swing")
    "change"          second value minus first value ("increase/decrease from A to B")
    "sum", "product"  all values added / multiplied (e.g. 16 petals x 16 jets)
    "ratio"           first value divided by second
    "percent_change"  (second - first) / first x 100
    "count"           number of distinct listed items (fill "items")
    "months_between"  months between two dates (fill "dates")
- "values": the numbers, in the order the operation needs, each with "label" (row + column/period), "value" (a plain number: no units, commas or % signs) and "passage" (its number)
- "items": the listed items to count, each exactly once (only for "count")
- "dates": two dates as {{"label", "year", "month", "passage"}} (only for "months_between")
- "unit": the unit of the result, e.g. "billion dollars", "percentage points", "months"
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


def hint(calc: dict[str, Any]) -> str:
    """Instruction appended to the generator prompt."""
    sources = sorted(
        {int(v["passage"]) for v in calc.get("values", []) if isinstance(v.get("passage"), int)}
    )
    cite = "".join(f"[{n}]" for n in sources)
    if calc["expression"] == "count":
        what = f"counted {calc['result']} distinct items: " + "; ".join(calc["items"][:20])
    else:
        what = f"computed {calc['expression']} = {calc['result']}"
    unit = f" {calc['unit']}" if calc.get("unit") else ""
    return (
        f"- A calculator {what}{unit} using values from the passages {cite}. "
        "Use this result in your answer (do not redo the arithmetic) and cite those passages."
    )
