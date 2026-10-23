"""Exact arithmetic for answer checking: the models write a formula, the code computes it.

Never uses eval: expressions are parsed with ast and only numbers, + - * / ** ( ), pi, sqrt and sin/cos/tan (degrees) are allowed."""
import ast
import math
import operator
import re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_FUNCS = {"sqrt": math.sqrt, "sin": lambda d: math.sin(math.radians(d)), "cos": lambda d: math.cos(math.radians(d)),
          "tan": lambda d: math.tan(math.radians(d)), "asin": lambda x: math.degrees(math.asin(x)),
          "acos": lambda x: math.degrees(math.acos(x)), "atan": lambda x: math.degrees(math.atan(x))}
_NAMES = {"pi": math.pi}


def safe_eval(expr: str) -> float | None:
    """Value of an arithmetic expression, or None if it is not one we allow."""
    if not expr or len(expr) > 300:
        return None
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace("π", "pi").replace("√", "sqrt")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and len(n.args) == 1:
            return _FUNCS[n.func.id](ev(n.args[0]))
        if isinstance(n, ast.Name) and n.id in _NAMES:
            return _NAMES[n.id]
        raise ValueError("not allowed")

    try:
        v = ev(ast.parse(expr, mode="eval"))
        return v if math.isfinite(v) else None
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return None


UNITS = r"\b(?:sq\.?\s*units?|units?|cm|mm|km|m|metres?|meters?|degrees?|seconds?|hours?|km/h|m/s|rs|kg|g|l|ml)\b"


def option_value(text: str) -> float | None:
    """Numeric value of an option such as '114.6°', '10√2 cm', '40π sq. units', '3/26', '₹ 1,860'; None if not numeric."""
    t = text.strip().replace("·", ".").replace("₹", "").replace(",", "")
    t = re.sub(r"[°²³]", "", t)
    t = re.sub(UNITS, "", t, flags=re.I)
    t = re.sub(r"√\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?", r"sqrt(\1)", t)
    t = re.sub(r"(\d|\))\s*(sqrt|π|pi)", r"\1*\2", t).replace("π", "pi").strip()
    if not t or not re.fullmatch(r"[\d.+\-*/() ]*(?:(?:sqrt|pi)[\d.+\-*/() ]*)*", t):
        return None
    return safe_eval(t)


def mcq_options(question: str) -> dict:
    """{'A': '30°', ...} from lines like '(A) 30°'."""
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\s*\(([A-D])\)\s*(.+)$", question, flags=re.M)}


def matching_options(question: str, value: float, rel: float = 0.01) -> list[str] | None:
    """Option letters whose numeric value equals `value` (within 1%); None if the options are not all numeric."""
    opts = mcq_options(question)
    vals = {k: option_value(v) for k, v in opts.items()}
    if len(opts) < 2 or any(v is None for v in vals.values()):
        return None
    return [k for k, v in vals.items() if abs(v - value) <= rel * max(abs(value), 1e-9) + 1e-9]
