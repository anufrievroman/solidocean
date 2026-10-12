"""Physical dimensions of expressions and unit-aware display formatting.

Values are always SI internally. The project's preferred units (``length_unit``, default µm;
``frequency_unit``, default GHz; ``time_unit``, default s) only change how lengths (and inverse
lengths), frequencies and times are *displayed* in the GUI, plots and result tables.

Dimensions are tuples (length, mass, time, temperature) of exponents; None = unknown.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np

from .expressions import CONSTANTS, UNITS, _fix_if, attach_units, Evaluator  # noqa: F401  (units.UNITS)

Dim = tuple[float, float, float, float]
DIMENSIONLESS: Dim = (0, 0, 0, 0)
LENGTH: Dim = (1, 0, 0, 0)
INV_LENGTH: Dim = (-1, 0, 0, 0)
PRESSURE: Dim = (-1, 1, -2, 0)
FORCE: Dim = (1, 1, -2, 0)
DENSITY: Dim = (-3, 1, 0, 0)
FREQUENCY: Dim = (0, 0, -1, 0)
TIME: Dim = (0, 0, 1, 0)
TEMPERATURE: Dim = (0, 0, 0, 1)
THERMAL_CONDUCTIVITY: Dim = (1, 1, -3, -1)     # W/(m K)
HEAT_CAPACITY: Dim = (2, 0, -2, -1)            # J/(kg K)
HEAT_FLUX: Dim = (0, 1, -3, 0)                 # W/m^2
POWER_DENSITY: Dim = (-1, 1, -3, 0)            # W/m^3
HEAT_TRANSFER_COEFF: Dim = (0, 1, -3, -1)      # W/(m^2 K)

LENGTH_UNITS = {"nm": 1e-9, "um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0}
FREQUENCY_UNITS = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
TIME_UNITS = {"ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
DEFAULT_LENGTH_UNIT = "um"


@dataclass(frozen=True)
class DisplayUnits:
    """The preferred units of a project (Project.display_units) for displaying values."""
    length: str = DEFAULT_LENGTH_UNIT
    frequency: str = "GHz"
    time: str = "s"

    @property
    def length_scale(self) -> float:
        return LENGTH_UNITS.get(self.length, 1e-6)

    @property
    def frequency_scale(self) -> float:
        return FREQUENCY_UNITS.get(self.frequency, 1e9)

    @property
    def time_scale(self) -> float:
        return TIME_UNITS.get(self.time, 1.0)


def display_units(du: "DisplayUnits | str | None") -> DisplayUnits:
    """DisplayUnits from DisplayUnits, a length unit alone, or None (the defaults)."""
    if isinstance(du, DisplayUnits):
        return du
    return DisplayUnits(length=du) if du else DisplayUnits()

_BASE_DIMS: dict[str, Dim] = {
    **{u: LENGTH for u in ("m", "cm", "mm", "um", "µm", "nm", "pm", "A", "angstrom")},
    "kg": (0, 1, 0, 0), "g": (0, 1, 0, 0),
    **{u: TIME for u in ("s", "ms", "us", "ns", "ps")},
    **{u: FREQUENCY for u in ("Hz", "kHz", "MHz", "GHz", "THz")},
    **{u: PRESSURE for u in ("Pa", "kPa", "MPa", "GPa", "TPa")},
    **{u: TEMPERATURE for u in ("K", "mK")},
    **{u: (2, 1, -2, 0) for u in ("J", "mJ", "kJ")},
    **{u: (2, 1, -3, 0) for u in ("W", "uW", "mW", "kW")},
**{u: FORCE for u in ("N", "mN", "uN", "nN", "pN")}, "rad": DIMENSIONLESS, "deg": DIMENSIONLESS, "1": DIMENSIONLESS,
}


def _pad(dim):
    """Older 3-component dimensions (length, mass, time) gain a zero temperature exponent."""
    if dim is None:
        return None
    dim = tuple(dim)
    return dim + (0,) * (4 - len(dim))
# functions whose result has the dimension of their (first) argument
_PASS_THROUGH = {"abs", "min", "max", "floor", "ceil", "round"}


def unit_label(unit: str) -> str:
    return {"um": "µm", "us": "µs"}.get(unit, unit)


def _unit_dim(unit_expr: str) -> Dim | None:
    tree = ast.parse(unit_expr.strip().replace("^", "**"), mode="eval")

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant):
            return DIMENSIONLESS
        if isinstance(n, ast.Name):
            return _BASE_DIMS.get(n.id)
        if isinstance(n, ast.BinOp):
            a = walk(n.left)
            if isinstance(n.op, ast.Pow) and isinstance(n.right, ast.Constant) and a is not None:
                return tuple(x * n.right.value for x in a)
            b = walk(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Mult):
                return tuple(x + y for x, y in zip(a, b))
            if isinstance(n.op, ast.Div):
                return tuple(x - y for x, y in zip(a, b))
        return None

    return walk(tree)


def _parse_with_unit_names(expr: str) -> tuple[ast.Expression, dict[str, Dim | None]]:
    """Parse an expression, replacing each [unit] by a placeholder name carrying its dimension."""
    units: dict[str, Dim | None] = {}

    def sub(unit):
        name = f"__unit{len(units)}"
        units[name] = _unit_dim(unit)
        return name

    src = attach_units(_fix_if(str(expr).strip()), sub).replace("^", "**")
    return ast.parse(src, mode="eval"), units


def dimension(expr, ev: Evaluator) -> Dim | None:
    """Dimension of an expression in the context of the evaluator's variables."""
    if isinstance(expr, (int, float)):
        return DIMENSIONLESS
    cache: dict[str, Dim | None] = ev.__dict__.setdefault("_dims", {})
    try:
        tree, units = _parse_with_unit_names(expr)
    except SyntaxError:
        return None

    def param(name: str) -> Dim | None:
        if name in cache:
            return cache[name]
        cache[name] = None  # guards against cycles
        if name in ev.overrides and name not in ev.expressions:
            return None
        cache[name] = dimension(ev.expressions[name], ev) if name in ev.expressions else None
        return cache[name]

    def same(a, b):
        return a if a is not None and b is not None and all(abs(x - y) < 1e-12 for x, y in zip(a, b)) else None

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant):
            return DIMENSIONLESS
        if isinstance(n, ast.Name):
            if n.id in units:
                return units[n.id]
            if n.id in CONSTANTS:
                return DIMENSIONLESS
            return param(n.id)
        if isinstance(n, ast.UnaryOp):
            return walk(n.operand)
        if isinstance(n, ast.BinOp):
            a = walk(n.left)
            if isinstance(n.op, ast.Pow):
                if isinstance(n.right, ast.Constant) and a is not None:
                    return tuple(x * n.right.value for x in a)
                return a if a == DIMENSIONLESS else None
            b = walk(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, (ast.Add, ast.Sub, ast.Mod)):
                return same(a, b)
            if isinstance(n.op, ast.Mult):
                return tuple(x + y for x, y in zip(a, b))
            if isinstance(n.op, ast.Div):
                return tuple(x - y for x, y in zip(a, b))
        if isinstance(n, ast.Compare):
            return DIMENSIONLESS
        if isinstance(n, ast.IfExp):
            return same(walk(n.body), walk(n.orelse))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            f = n.func.id
            args = [walk(a) for a in n.args]
            if f == "sqrt" and args and args[0] is not None:
                return tuple(x / 2 for x in args[0])
            if f in _PASS_THROUGH and args:
                out = args[0]
                for a in args[1:]:
                    out = same(out, a)
                return out
            if f in ("if", "if_") and len(args) == 3:
                return same(args[1], args[2])
            return DIMENSIONLESS
        return None

    return walk(tree)


_PREFIXES = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, "")]
_SMALL = [(1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")]


def _engineering(v: float, base: str, small: bool = False) -> str:
    if small and v != 0 and abs(v) < 1:
        for scale, prefix in _SMALL:
            if abs(v) >= scale:
                return f"{v / scale:.6g} {prefix}{base}"
    for scale, prefix in _PREFIXES:
        if abs(v) >= scale:
            return f"{v / scale:.6g} {prefix}{base}"
    return f"{v:.6g} {base}"


def format_value(v: float, dim: Dim | None, du: DisplayUnits | str | None = None) -> str:
    """Human-readable value: lengths, frequencies and times in the preferred units (`du`: the project's
    display_units, or a length unit), other common quantities with SI units."""
    dim = _pad(dim)
    du = display_units(du)
    scale = du.length_scale
    lu = unit_label(du.length)
    if dim == LENGTH:
        return f"{v / scale:.6g} {lu}"
    if dim == INV_LENGTH:
        return f"{v * scale:.6g} rad/{lu}"
    if dim == PRESSURE:
        return _engineering(v, "Pa")
    if dim == FORCE:
        return _engineering(v, "N", small=True)
    if dim == FREQUENCY:
        return f"{v / du.frequency_scale:.6g} {du.frequency}"
    if dim == DENSITY:
        return f"{v:.6g} kg/m³"
    if dim == TIME:
        return f"{v / du.time_scale:.6g} {unit_label(du.time)}"
    if dim == TEMPERATURE:
        return f"{v:.6g} K"
    named = {THERMAL_CONDUCTIVITY: "W/(m·K)", HEAT_CAPACITY: "J/(kg·K)", HEAT_FLUX: "W/m²",
             POWER_DENSITY: "W/m³", HEAT_TRANSFER_COEFF: "W/(m²·K)"}
    if dim in named:
        return f"{v:.6g} {named[dim]}"
    return f"{v:.6g}"


def axis_scale(dim: Dim | None, du: DisplayUnits | str | None = None) -> tuple[float, str]:
    """(factor to multiply SI values by, unit label) for a plot axis, in the preferred units (`du`: the
    project's display_units, or a length unit)."""
    dim = _pad(dim)
    du = display_units(du)
    lu = unit_label(du.length)
    if dim == LENGTH:
        return 1 / du.length_scale, lu
    if dim == INV_LENGTH:
        return du.length_scale, f"rad/{lu}"
    if dim == FREQUENCY:
        return 1 / du.frequency_scale, du.frequency
    if dim == TIME:
        return 1 / du.time_scale, unit_label(du.time)
    return 1.0, ""


def length_expr(v: float, length_unit: str = DEFAULT_LENGTH_UNIT) -> str:
    """A length as an input expression in the preferred unit, e.g. 0.25[um]."""
    if v == 0:
        return "0"
    return f"{v / LENGTH_UNITS.get(length_unit, 1e-6):.6g}[{length_unit}]"
