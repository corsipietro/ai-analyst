"""Number checker: a MEASURING tool, not a corrector (ported unchanged in logic from analyst/number_check.py).

It reads the analyst's final answer, extracts every number written in it and checks
whether each one can be traced back to the sources the analyst had (tool results,
the user's question, the reference given in the instructions):
- direct:  the number is a source value, up to the rounding implied by its decimals
           (569.9 matches 569.884; 570 matches 569.884);
- derived: the number is the sum or the difference of two source values
           (2257.31 = 2827.189 - 569.884), with a slightly larger rounding tolerance;
- untraceable: anything else (ratios, products, means computed by hand, invented numbers).
The answer is never sent back to the model: the checker only measures. It also counts the [tool] marks an
answer carries when no step read the data (tool_marks_without_data): claims marked as coming from a tool that
returned nothing.

Known limits: small integers (years, counts) are almost always traceable, because many
differences of two source values are small integers; products and ratios (e.g. a value
derived from the equations) count as untraceable even when correct. The count of untraceable
numbers is a signal to read, not a verdict. Units are not handled yet.
"""
from __future__ import annotations

import re
from bisect import bisect_left

# A number as people write it: optional sign, optional thousands commas, optional decimals or exponent.
# The lookbehind skips digits glued to letters or dots, e.g. "v4", "gemma4", "S1".
NUMBER = re.compile(
    r"(?<![\w.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?(?![\w])")
# "1. ", "2) " at the start of a line are list markers, not numbers.
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def parse_number(text: str) -> tuple[float, int]:
    """Return (value, decimals) of one number as written, e.g. '1,376.85' -> (1376.85, 2).

    A comma followed by exactly three digits is a thousands separator (English);
    any other comma is a decimal comma (Italian, e.g. '569,9').
    """
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    text = text.replace(",", ".")
    mantissa = re.split(r"[eE]", text)[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return float(text), decimals


def extract_numbers(text: str) -> list[tuple[str, float, int]]:
    """Return the distinct numbers written in a text as (as written, value, decimals).

    List markers at the start of a line are skipped; each written form is counted once.
    """
    text = LIST_MARKER.sub(" ", text)
    seen = {}
    for match in NUMBER.finditer(text):
        written = match.group()
        if written not in seen:
            value, decimals = parse_number(written)
            seen[written] = (written, value, decimals)
    return list(seen.values())


def source_values(texts: list[str]) -> list[float]:
    """Return the sorted distinct absolute values of all numbers found in the source texts."""
    return sorted({abs(value) for text in texts for _, value, _ in extract_numbers(text)})


def pair_values(values: list[float]) -> list[float]:
    """Return the sorted sums and absolute differences of every pair of source values."""
    pairs = set()
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            pairs.add(a + b)
            pairs.add(abs(a - b))
    return sorted(pairs)


def close_to_any(x: float, sorted_values: list[float], tolerance: float) -> bool:
    """True if some value in the sorted list is within tolerance of x (binary search)."""
    i = bisect_left(sorted_values, x - tolerance)
    return i < len(sorted_values) and sorted_values[i] <= x + tolerance


def tool_marks_without_data(answer: str, retrieved: bool) -> int:
    """How many [tool] marks the answer has if no reading step gave data (retrieved False), else 0.

    A measure, not a correction: the answer is left as it is.
    """
    return 0 if retrieved else answer.count("[tool]")


def check_numbers(answer: str, sources: list[str]) -> dict:
    """Measure how many numbers of the answer can be traced back to the sources.

    answer: the final answer of the analyst (without its hidden reasoning).
    sources: the texts the analyst had (tool results, question, context, database reference).
    Returns the counts and the list of untraceable numbers, ready for the JSONL record.
    """
    values = source_values(sources)
    pairs = pair_values(values)
    direct, derived, untraceable = [], [], []
    for written, value, decimals in extract_numbers(answer):
        step = 10.0 ** -decimals
        x = abs(value)
        if close_to_any(x, values, 0.5 * step + 1e-9):      # rounding of the written number
            direct.append(written)
        elif close_to_any(x, pairs, 1.5 * step + 1e-9):     # both operands rounded, then the result
            derived.append(written)
        else:
            untraceable.append(written)
    return {"numbers_cited": len(direct) + len(derived) + len(untraceable),
            "numbers_direct": len(direct),
            "numbers_derived": len(derived),
            "numbers_untraceable": len(untraceable),
            "untraceable_list": untraceable}
