"""Formula parser for Space Rangers quest conditions and calculations.

Supports operations:
- Arithmetic: +, -, *, /, div, mod
- Comparison: =, <>, <, >, <=, >=
- Logical: and, or
- Range: to, in
- Parameters: [p1], [p2], etc.
- Random ranges: [1..100], [1;5;10]
"""

from __future__ import annotations

import random
import re

MAX_NUMBER = 2_000_000_000

Arg = str | int | float


def _parse_range(arg: str) -> list[tuple[int, int]]:
    """Parse a range expression like [1..100] or [1;5;10..20]."""
    inner = arg[1:-1]  # Remove brackets
    ranges = []
    for part in inner.split(";"):
        if ".." in part:
            low_s, high_s = part.split("..")
            low, high = int(low_s), int(high_s)
            if low > high:
                low, high = high, low
            ranges.append((low, high))
        else:
            val = int(part)
            ranges.append((val, val))
    return ranges


def _arg_to_number(arg: Arg) -> float:
    """Convert an argument to a number, evaluating ranges randomly."""
    if isinstance(arg, (int, float)):
        return arg

    if not arg:
        raise ValueError("No string data for arg_to_number")

    if arg.startswith("[") and arg.endswith("]"):
        # It's a range - pick random value
        if arg.startswith("[p"):
            # This shouldn't happen - parameters should be resolved
            raise ValueError(f"Unresolved parameter: {arg}")

        ranges = _parse_range(arg)
        total = sum(high - low + 1 for low, high in ranges)
        rnd = random.randint(0, total - 1)

        for low, high in ranges:
            length = high - low + 1
            if rnd < length:
                return low + rnd
            rnd -= length

        raise ValueError(f"Error finding random value for {arg}")

    # Try parsing as float
    try:
        return float(arg.replace(",", "."))
    except ValueError:
        raise ValueError(f"Unknown arg: {arg}")


def _floor_ceil(val: float) -> int:
    """Floor for positive, ceil for negative."""
    return int(val) if val >= 0 else -int(-val)


# Operator definitions: (symbol, priority, function)
# Higher priority = evaluated later (processed first in parsing)
OPERATORS = {
    "/": (1, lambda a, b: a / b if b != 0 else (MAX_NUMBER if a > 0 else -MAX_NUMBER)),
    "div": (
        1,
        lambda a, b: _floor_ceil(a / b) if b != 0 else (MAX_NUMBER if a > 0 else -MAX_NUMBER),
    ),
    "mod": (1, lambda a, b: a % b if b != 0 else (MAX_NUMBER if a > 0 else -MAX_NUMBER)),
    "*": (2, lambda a, b: a * b),
    "-": (3, lambda a, b: a - b),
    "+": (4, lambda a, b: a + b),
    "to": (5, None),  # Special handling for range creation
    "in": (6, None),  # Special handling for range membership
    ">=": (7, lambda a, b: 1 if a >= b else 0),
    "<=": (7, lambda a, b: 1 if a <= b else 0),
    ">": (7, lambda a, b: 1 if a > b else 0),
    "<": (7, lambda a, b: 1 if a < b else 0),
    "=": (7, lambda a, b: 1 if a == b else 0),
    "<>": (7, lambda a, b: 1 if a != b else 0),
    "and": (8, lambda a, b: 1 if a and b else 0),
    "or": (9, lambda a, b: 1 if a or b else 0),
}

# Short symbols for multi-char operators (used in parsing)
SYMBOL_MAP = {
    "div": "f",
    "mod": "g",
    "to": "$",
    "in": "#",
    ">=": "c",
    "<=": "b",
    "<>": "e",
    "and": "&",
    "or": "|",
}

REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}


def _find_closing_bracket(s: str, start: int, bracket_type: str) -> int:
    """Find the closing bracket matching the one at start."""
    open_char = "(" if bracket_type == "round" else "["
    close_char = ")" if bracket_type == "round" else "]"

    if s[start] != open_char:
        raise ValueError(f"Not a bracket at position {start}")

    count = 1
    for i in range(start + 1, len(s)):
        if s[i] == open_char:
            count += 1
        elif s[i] == close_char:
            count -= 1
            if count == 0:
                return i

    raise ValueError(f"Unclosed bracket at position {start}")


def _preprocess(s: str) -> str:
    """Preprocess formula: replace long symbols with short ones, remove whitespace."""
    # Replace long symbols with short ones
    for long_sym, short_sym in SYMBOL_MAP.items():
        s = s.replace(long_sym, short_sym)

    # Remove whitespace and newlines
    s = re.sub(r"\s+", "", s)

    return s


def _parse_recursive(s: str, params: list[int]) -> Arg:
    """Recursively parse a formula expression."""
    if not s:
        raise ValueError("Empty string!")

    # Remove covering parentheses
    while s.startswith("("):
        close = _find_closing_bracket(s, 0, "round")
        if close == len(s) - 1:
            s = s[1:-1]
        else:
            break

    # Find operators at the top level (not inside brackets)
    operators_found: list[tuple[int, str, int]] = []  # (position, operator, priority)

    i = 0
    while i < len(s):
        c = s[i]

        if c == "(":
            i = _find_closing_bracket(s, i, "round") + 1
            continue
        elif c == "[":
            i = _find_closing_bracket(s, i, "square") + 1
            continue

        # Check for operators (check short symbols)
        op = REVERSE_SYMBOL_MAP.get(c, c)
        if op in OPERATORS:
            prio = OPERATORS[op][0]
            operators_found.append((i, op, prio))

        i += 1

    # Handle parameter reference [pN]
    if s.startswith("[p") and s.endswith("]") and s.count("[") == 1:
        param_num = int(s[2:-1])
        if param_num < 1 or param_num > len(params):
            raise ValueError(f"Undefined param {s}")
        return params[param_num - 1]

    # Handle range literal [...] - only if it's a single bracket group
    if s.startswith("[") and s.endswith("]") and not s.startswith("[p") and s.count("[") == 1:
        return s  # Return as string for later random evaluation

    # No operators - must be a number
    if not operators_found:
        try:
            return float(s.replace(",", "."))
        except ValueError:
            raise ValueError(f"Unknown elementary value: {s}")

    # Find operator to split on (highest priority, rightmost)
    operators_found.sort(key=lambda x: (x[2], x[0]))
    pos, op, _ = operators_found[-1]

    left = s[:pos]
    # Get the length of the short symbol
    short_sym = SYMBOL_MAP.get(op, op)
    right = s[pos + len(short_sym) :]

    if left:
        left_val = _parse_recursive(left, params)
        right_val = _parse_recursive(right, params)

        # Handle special operators
        if op == "to":
            # Create a combined range
            left_ranges = (
                _parse_range(left_val)
                if isinstance(left_val, str)
                else [(int(left_val), int(left_val))]
            )
            right_ranges = (
                _parse_range(right_val)
                if isinstance(right_val, str)
                else [(int(right_val), int(right_val))]
            )

            all_vals = [v for r in left_ranges + right_ranges for v in r]
            return f"[{min(all_vals)}..{max(all_vals)}]"

        if op == "in":
            # Check if value is in range
            if isinstance(left_val, str) and isinstance(right_val, str):
                val = _arg_to_number(left_val)
                ranges = _parse_range(right_val)
            elif isinstance(left_val, (int, float)) and isinstance(right_val, str):
                val = left_val
                ranges = _parse_range(right_val)
            elif isinstance(left_val, str) and isinstance(right_val, (int, float)):
                val = right_val
                ranges = _parse_range(left_val)
            else:
                return 1 if left_val == right_val else 0

            for low, high in ranges:
                if low <= val <= high:
                    return 1
            return 0

        # Normal binary operator
        func = OPERATORS[op][1]
        if func is None:
            raise ValueError(f"No function for operator {op}")

        a = _arg_to_number(left_val)
        b = _arg_to_number(right_val)
        return func(a, b)
    else:
        # Unary minus
        if op == "-":
            right_val = _parse_recursive(right, params)
            return -_arg_to_number(right_val)
        else:
            raise ValueError(f"Usage of {op} as unary in {s}")


def parse(formula: str, params: list[int] | None = None) -> int:
    """Parse and evaluate a formula expression.

    Args:
        formula: The formula string to evaluate
        params: List of parameter values (accessed as [p1], [p2], etc.)

    Returns:
        The integer result of the evaluation
    """
    if params is None:
        params = []

    preprocessed = _preprocess(formula)
    result = _parse_recursive(preprocessed, params)
    return round(_arg_to_number(result))
