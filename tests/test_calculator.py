"""Tests of the safe calculator (written before the code, 30/10/2026)."""
import pytest

from sim_analyst.calculator import calculate


@pytest.mark.parametrize("expression, expected", [
    ("2 + 3 * 4", 14),
    ("(2 + 3) * 4", 20),
    ("5 * 80", 400),
    ("2 ** 10", 1024),
    ("-5 + 2", -3),
    ("7 - 10 / 4", 4.5),
])
def test_arithmetic(expression, expected):
    """The usual order of operations, parentheses, powers and negative numbers."""
    assert calculate(expression) == expected


def test_scientific_notation():
    """1/(f * alpha * tau) with f = 100, alpha = 1e-6, tau = 25: the kind of computation the analyst needs."""
    assert calculate("1/(100*1e-6*25)") == pytest.approx(400)


@pytest.mark.parametrize("expression", [
    "__import__('os')",     # a function call: never executed
    "abs(-3)",              # any function call, even a harmless one
    "x + 1",                # a name
    "'a' * 3",              # text, not a number
    "True + 1",             # True is not a number here
])
def test_anything_else_is_refused(expression):
    """Only numbers and the five operations are accepted; everything else is an error, nothing is run."""
    with pytest.raises(ValueError):
        calculate(expression)


@pytest.mark.parametrize("expression, message", [
    ("1/0", "division by zero"),
    ("9**9**9", "too large"),
    ("2 +", "not a valid"),
    ("", "give an arithmetic expression"),
    ("1+" * 150 + "1", "too long"),
    ("(-8) ** 0.5", "not a finite real number"),
    ("1e308 * 10", "not a finite real number"),
    ("((10**100)**100)**100", "too large"),
])
def test_errors_say_what_is_wrong(expression, message):
    """Every error is a ValueError whose message tells the model what to fix."""
    with pytest.raises(ValueError, match=message):
        calculate(expression)
