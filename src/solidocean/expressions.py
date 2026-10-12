"""Safe evaluation of variable expressions with units in brackets.

Expressions are strings such as ``"0.3*a"``, ``"500[nm]"`` or
``"165.7[GPa]"``. Everything is converted to SI. Variables may reference
each other in any order; cycles are reported as errors.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Mapping

import numpy as np

UNITS: dict[str, float] = {
    # length
    "m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9,
    "pm": 1e-12, "A": 1e-10, "angstrom": 1e-10,
    # mass
    "kg": 1.0, "g": 1e-3,
    # time / frequency
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12,
    "Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12,
    # pressure / stiffness
    "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "TPa": 1e12,
    "N": 1.0, "mN": 1e-3, "uN": 1e-6, "nN": 1e-9, "pN": 1e-12,
    # temperature, energy, power
    "K": 1.0, "mK": 1e-3,
    "J": 1.0, "mJ": 1e-3, "kJ": 1e3,
    "W": 1.0, "nW": 1e-9, "uW": 1e-6, "mW": 1e-3, "kW": 1e3, "MW": 1e6, "GW": 1e9,
    # angles
    "rad": 1.0, "deg": math.pi / 180.0,
    # dimensionless
    "1": 1.0,
}

FUNCTIONS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
    "abs": abs, "min": min, "max": max, "floor": math.floor, "ceil": math.ceil,
    "round": round,
}

CONSTANTS = {"pi": math.pi, "e": math.e}

# Functions that return arrays; only allowed in sweep value lists.
ARRAY_FUNCTIONS = {
    # range(start, step, stop), stop inclusive
    "range": lambda start, step, stop: np.arange(start, stop + 0.5 * step, step),
    "linspace": lambda start, stop, n: np.linspace(start, stop, int(n)),
    "logspace": lambda start, stop, n: np.logspace(start, stop, int(n)),   # 10^start .. 10^stop
}

_UNIT_RE = re.compile(r"(?<=[\w.)])\s*\[([^\[\],]+)\]")  # unit brackets follow a value
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ExpressionError(ValueError):
    pass


def _unit_value(unit_expr: str) -> float:
    """Evaluate a unit expression like ``kg/m^3`` or ``1/um``."""
    src = unit_expr.strip().replace("^", "**")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"bad unit [{unit_expr}]") from exc

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in UNITS:
                raise ExpressionError(f"unknown unit '{node.id}' in [{unit_expr}]")
            return UNITS[node.id]
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return a**b
        raise ExpressionError(f"bad unit [{unit_expr}]")

    return ev(tree)


_OPERAND_RE = re.compile(r"(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*)\s*$")
_FUNC_NAME_RE = re.compile(r"[A-Za-z_]\w*\s*$")


def attach_units(expr: str, repl) -> str:
    """Replace every value[unit] by (value*repl(unit)): a unit binds to the operand right before it (a number,
    a name or a parenthesised group / function call), so 1/2[nm] = 1/(2 nm) and 2[um]^2 = (2 um)^2."""
    out, pos = "", 0
    for m in _UNIT_RE.finditer(expr):
        out = (out + expr[pos:m.start()]).rstrip()
        pos = m.end()
        if out.endswith(")"):
            depth, start = 0, 0
            for i in range(len(out) - 1, -1, -1):
                depth += {")": 1, "(": -1}.get(out[i], 0)
                if depth == 0:
                    start = i
                    break
            fn = _FUNC_NAME_RE.search(out[:start])
            if fn:
                start = fn.start()
        else:
            op = _OPERAND_RE.search(out)
            start = op.start() if op else len(out)
        out = f"{out[:start]}({out[start:]}*{repl(m.group(1))})"
    return out + expr[pos:]


def _substitute_units(expr: str) -> str:
    return attach_units(expr, lambda unit: f"({_unit_value(unit)!r})")


def _parse(expr: str) -> ast.Expression:
    src = _substitute_units(str(expr).strip()).replace("^", "**")
    if not src:
        raise ExpressionError("empty expression")
    try:
        return ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error in '{expr}'") from exc


def names_in(expr: str) -> set[str]:
    """Names referenced by an expression (excluding functions/constants)."""
    tree = _parse(expr)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in CONSTANTS:
            out.add(node.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            out.discard(node.func.id)
    return out


_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b, ast.Mod: lambda a, b: a % b,
}
_CMPOPS = {
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}


def _eval_tree(node, names: Mapping[str, float], allow_arrays: bool):
    ev = lambda n: _eval_tree(n, names, allow_arrays)  # noqa: E731
    if isinstance(node, ast.Expression):
        return ev(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"unsupported constant {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ExpressionError(f"unknown name '{node.id}'")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node.op, ast.UAdd):
            return ev(node.operand)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS:
        return float(_CMPOPS[type(node.ops[0])](ev(node.left), ev(node.comparators[0])))
    if isinstance(node, ast.IfExp):
        return ev(node.body) if ev(node.test) else ev(node.orelse)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        fname = node.func.id
        args = [ev(a) for a in node.args]
        if fname in FUNCTIONS:
            return float(FUNCTIONS[fname](*args))
        if fname == "if" and len(args) == 3:
            return args[1] if args[0] else args[2]
        if allow_arrays and fname in ARRAY_FUNCTIONS:
            return ARRAY_FUNCTIONS[fname](*args)
        raise ExpressionError(f"unknown function '{fname}'")
    if allow_arrays and isinstance(node, (ast.List, ast.Tuple)):
        return np.array([float(ev(e)) for e in node.elts])
    raise ExpressionError(f"unsupported syntax: {ast.dump(node)[:60]}")


def _fix_if(expr: str) -> str:
    # 'if' is a Python keyword; allow if(cond, a, b)
    return re.sub(r"\bif\s*\(", "if_(", expr)


class Evaluator:
    """Resolves a variable table and evaluates expressions against it."""

    def __init__(self, variables: Mapping[str, str], overrides: Mapping[str, float] | None = None):
        self.expressions = {k: str(v) for k, v in variables.items()}
        for name in self.expressions:
            if not _NAME_RE.match(name):
                raise ExpressionError(f"invalid variable name '{name}'")
        self.overrides = dict(overrides or {})
        self.values: dict[str, float] = {}
        self._resolve()

    def _resolve(self) -> None:
        state: dict[str, int] = {}  # 1 = in progress, 2 = done

        def visit(name: str, chain: list[str]):
            if name in self.overrides:
                self.values[name] = float(self.overrides[name])
                return
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                raise ExpressionError("circular variable reference: " + " -> ".join(chain + [name]))
            state[name] = 1
            expr = self.expressions[name]
            for dep in names_in(_fix_if(expr)):
                if dep == "if_":
                    continue
                if dep not in self.expressions and dep not in self.overrides:
                    raise ExpressionError(f"variable '{name}': unknown name '{dep}'")
                visit(dep, chain + [name])
            try:
                self.values[name] = self._eval(expr, self.values)
            except ExpressionError as exc:
                raise ExpressionError(f"variable '{name}': {exc}") from exc
            state[name] = 2

        for name in list(self.expressions) + list(self.overrides):
            visit(name, [])

    @staticmethod
    def _eval(expr, names, allow_arrays=False):
        if isinstance(expr, (int, float)) and not isinstance(expr, bool):
            return float(expr)
        tree = _parse(_fix_if(str(expr)))
        local = dict(names)
        result = _eval_tree(_rename_if(tree), local, allow_arrays)
        return result

    def __call__(self, expr) -> float:
        val = self._eval(expr, self.values)
        if isinstance(val, np.ndarray):
            raise ExpressionError(f"'{expr}' is not a scalar")
        return float(val)

    def vector(self, exprs) -> np.ndarray:
        return np.array([self(e) for e in exprs], dtype=float)

    def array(self, expr) -> np.ndarray:
        """Evaluate a sweep list: ``range(0, 0.1, 1)``, ``linspace(...)``, ``[1, 2, 3]``."""
        val = self._eval(expr, self.values, allow_arrays=True)
        return np.atleast_1d(np.asarray(val, dtype=float))

    def with_overrides(self, overrides: Mapping[str, float]) -> "Evaluator":
        return Evaluator(self.expressions, {**self.overrides, **overrides})


def _rename_if(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "if_":
            node.func.id = "if"
    return tree


def resolve_each(variables: Mapping[str, str]) -> tuple[dict[str, float], dict[str, str]]:
    """Evaluate every variable that can be evaluated: (values, errors). A variable with a broken expression
    gets its own error; one that only depends on a broken variable gets "depends on '...'"; all others get
    their values (an Evaluator stops at the first error instead)."""
    exprs = {k: str(v) for k, v in variables.items()}
    values: dict[str, float] = {}
    errors: dict[str, str] = {}
    state: dict[str, int] = {}  # 1 = in progress, 2 = done

    def visit(name: str, chain: list[str]) -> bool:
        if state.get(name) == 2:
            return name in values
        if state.get(name) == 1:
            errors[name] = "circular reference: " + " -> ".join(chain + [name])
            return False
        state[name] = 1
        ok = True
        try:
            if not _NAME_RE.match(name):
                raise ExpressionError(f"invalid variable name '{name}'")
            deps = [d for d in names_in(_fix_if(exprs[name])) if d != "if_"]
        except ExpressionError as exc:
            errors[name], ok, deps = str(exc), False, []
        for dep in deps:
            if dep not in exprs:
                errors[name], ok = f"unknown name '{dep}'", False
                break
            if not visit(dep, chain + [name]):
                errors.setdefault(name, f"depends on '{dep}', which has an error")
                ok = False
                break
        if ok:
            try:
                val = Evaluator._eval(exprs[name], values)
                if isinstance(val, np.ndarray):
                    raise ExpressionError("not a scalar")
                values[name] = float(val)
            except Exception as exc:
                errors[name], ok = str(exc), False
        state[name] = 2
        return ok

    for name in exprs:
        visit(name, [])
    return values, errors
