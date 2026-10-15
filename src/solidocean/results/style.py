"""Plot style and axes shared by every results plot."""

from __future__ import annotations

import numpy as np

PLOT_FONT = ["Arial", "Helvetica", "DejaVu Sans"]  # first available is used

# our own curve colours, in the order curves take them: the plot green, the accent (the first material
# colour) and then the app icon's palette, open sea -> beach
PALETTE_NAME = "SolidOcean"
# light and dark alternate so neighbouring bands stay apart; the palest beach tone is left out, being too
# faint for a line on white
SERIES = ["#64b999", "#9adcd2", "#13608f", "#5cc8c0", "#0b2a5b", "#1b9aaa"]


def use_plot_style() -> None:
    """Arial (with fallbacks) for all matplotlib text, and our own qualitative palette."""
    import matplotlib
    from matplotlib.colors import ListedColormap
    if PALETTE_NAME not in matplotlib.colormaps:
        matplotlib.colormaps.register(ListedColormap(SERIES, name=PALETTE_NAME))
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = PLOT_FONT + [f for f in matplotlib.rcParams["font.sans-serif"]
                                                         if f not in PLOT_FONT]


LINESTYLES = {"solid": "-", "dashed": "--", "dotted": ":", "none": ""}


def outer_axis(results, du=None) -> tuple[np.ndarray, str]:
    """Display values and axis label of the variable (outer) sweep variable, in the preferred units (`du`: the
    project's display_units, or a length unit)."""
    from ..units import axis_scale
    if not results.outer_name:
        return np.arange(len(results.outer_values)), "step"
    dim = results.info.get("outer_dim")
    factor, unit = axis_scale(tuple(dim) if dim else None, du)
    label = f"{results.outer_name} ({unit})" if unit else results.outer_name
    return results.outer_values * factor, label


def plot_ranges(project, pdef, y_scale: float = 1.0) -> tuple[tuple, tuple]:
    """Resolve a plot's range fields: (y range in plotted units, x range in displayed x units).

    Plain numbers are in the displayed unit; expressions with units (e.g. 10[GHz]) are SI.
    """
    ps, ev = pdef.style, project.evaluator()

    def value(text, si_to_display):
        text = str(text).strip()
        if not text:
            return None
        try:
            v = ev(text)
        except Exception:
            return None
        return v / si_to_display if "[" in text else v

    return (value(ps.y_min, y_scale), value(ps.y_max, y_scale)), (value(ps.x_min, 1.0), value(ps.x_max, 1.0))


def set_scales(ax, style, x: bool = True) -> None:
    """Logarithmic axes of an XY plot (style.y_log, style.x_log; `x` = False: the x axis cannot be)."""
    if style.y_log:
        ax.set_yscale("log")
    if style.x_log and x:
        ax.set_xscale("log")


def time_axis(project) -> tuple[float, str]:
    """(scale to divide seconds by, unit label): the project's preferred time unit."""
    from ..units import unit_label
    du = project.display_units
    return du.time_scale, unit_label(du.time)


def outer_text(results, project, i: int) -> str:
    from ..units import format_value
    dim = results.info.get("outer_dim")
    return format_value(results.outer_values[i], tuple(dim) if dim else None, project.display_units)


def slug(name: str) -> str:
    """A plot name as a file name: 'Tip deflection' -> 'tip_deflection'."""
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "plot"
