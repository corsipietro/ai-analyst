"""Safe arithmetic for the analyst: compute an expression written as text.

The model must not compute in its head, so it sends the expression as text (e.g. "1/(100*1e-6*25)").
The text is turned into a tree with ast.parse, which describes the expression without running it;
then evaluate() walks the tree and accepts only numbers, + - * / ** and the minus sign. Anything else
(names, function calls, text) is refused, so nothing written by the model is ever executed.
"""
from __future__ import annotations

import ast
import math
import operator

MAX_LENGTH = 200          # characters of an expression
MAX_EXPONENT = 100        # |exponent| of a power: 9**9**9 would take forever and fill the memory
MAX_DIGITS = 308          # powers whose result would exceed about 1e308 are refused before computing

OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def check_power(base, exponent) -> None:
    """Refuse a power before computing it if it would take too long or give a number too large.

    Python computes huge whole numbers exactly, so ((10**100)**100)**100 is allowed by the exponent
    limit alone but would fill the memory: the size of the result, |exponent| * log10|base|, is
    checked first against the largest ordinary number (about 1e308).
    """
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponent {exponent} is too large (at most {MAX_EXPONENT} in absolute value).")
    if abs(base) > 1 and exponent * math.log10(abs(base)) > MAX_DIGITS:
        raise ValueError("the result is too large to compute.")


def evaluate(node):
    """Compute one piece of the tree: a number, or an operation on pieces computed the same way."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATIONS:
        left = evaluate(node.left)
        right = evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            check_power(left, right)
        return OPERATIONS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate(node.operand)
    raise ValueError(f"not allowed in a calculation: {type(node).__name__}. "
                     "Use only numbers, + - * / ** and parentheses.")


def calculate(expression: str) -> float:
    """Compute an arithmetic expression; raise ValueError with a clear message if it cannot be computed."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("give an arithmetic expression as text, e.g. '1/(100*1e-6*25)'.")
    if len(expression) > MAX_LENGTH:
        raise ValueError(f"expression too long ({len(expression)} characters, at most {MAX_LENGTH}): split it.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ValueError(f"not a valid arithmetic expression: {expression!r}.") from None
    try:
        result = evaluate(tree.body)
    except ZeroDivisionError:
        raise ValueError("division by zero.") from None
    except OverflowError:
        raise ValueError("the result is too large to compute.") from None
    if isinstance(result, complex) or not math.isfinite(result):
        raise ValueError(f"the result is not a finite real number ({result}).")
    return result
