"""The interface every physics implements, and what the rest of the program may assume about it.

A physics is a subpackage of `solidocean.physics` (solid mechanics, heat transfer, ...) that owns:

- its settings: a dataclass stored on the Project under `section` (also the TOML table name), with
  `to_dict()` and a `from_dict()` classmethod;
- validation of those settings, and of the materials it needs;
- its problems: a solver whose `run(quantities)` returns results (results/field.FieldResults for one field per
  variable value and time step, or a physics' own results type such as the modal results of solid
  mechanics);
- the field and scalar names its quantities may use, and how the fields of a solution are evaluated;
- its results plots (kinds, titles, defaults) and templates.

Shared code (project, CLI, GUI) talks to a physics only through its PhysicsSpec, found with
`physics.get(kind)` or `physics.of(project)`. The GUI side lives in gui/physics/ (tree nodes, panels).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]


class Cancelled(Exception):
    """A run stopped by the user (the cancel event was set)."""


class PhysicsSpec:
    kind = ""            # project.physics_kind, [project] physics in the model file
    title = ""           # e.g. "Heat Transfer"
    description = ""     # one line for the New dialog
    section = ""         # Project attribute and TOML table holding the settings
    settings_class: type = object
    plot_kinds: tuple[str, ...] = ()           # kinds of results plots, in menu order
    plot_titles: dict[str, str] = {}           # kind -> title
    field_names: tuple[str, ...] = ()          # field variables of quantities (besides x, y, z)
    scalar_names: tuple[str, ...] = ()         # scalar variables of quantities (besides the model variables, pi)

    # ---- settings ------------------------------------------------------------------
    def settings(self, project):
        return getattr(project, self.section)

    def settings_from_dict(self, d: dict):
        return self.settings_class.from_dict(d)

    def problem_type(self, project) -> str:
        """The physics' current problem type, e.g. 'stationary'."""
        return ""

    def field_results(self, project) -> bool:
        """True if the current problem gives field results (results/field.FieldResults: one field per variable
        value and time step), False for a physics' own results (e.g. modes)."""
        return True

    def validate(self, project, ev) -> list[str]:
        """Problems with the physics settings, its materials and its problem (empty if none)."""
        return []

    # ---- solving -----------------------------------------------------------------
    def solver(self, project, log: LogFn = print, progress: ProgressFn | None = None, cancel=None):
        """An object with run(quantities) -> results for the project's problem."""
        raise NotImplementedError

    # ---- quantities and plots ----------------------------------------------------
    def quantity_help(self, project) -> dict[str, str]:
        """Reference of the quantities available in the current problem (GUI)."""
        return {}

    def quantities(self, project) -> list[tuple[str, str]]:
        """Named physical quantities of the current problem, (label, expression), e.g. ("Heat flux x (W/m²)",
        "qx"): offered in a menu under every expression editor."""
        return []

    def field_quantity(self, project) -> str:
        """What a 3D field view shows when its quantity is left empty."""
        return ""

    def field_plot(self, name: str = ""):
        """A 3D field view of this physics' solution (name and colour scale)."""
        raise NotImplementedError

    def default_plots(self, project) -> list:
        """Plots a new model (or one without [[plots]]) starts with."""
        return [self.field_plot()]

    def new_plot(self, project, kind: str, box):
        """A plot of `kind` added by the user; box = (lo, hi) bounding box of the model (SI), for placing lines
        and cross-sections."""
        raise NotImplementedError

    def probe_plot(self, project, kind: str, box, quantity: str, colormap: str):
        """A new line (through the middle of the model along x) or cross-section (the xy plane through its
        centre) of a quantity."""
        from ..plots import PlotStyle, ResultPlot
        from ..units import length_expr
        lo, hi = (np.asarray(v, dtype=float) for v in box)
        c = (lo + hi) / 2
        lu = project.length_unit

        def pt(p):
            return [length_expr(float(x), lu) for x in p]
        if kind == "line":
            return ResultPlot(self.plot_titles[kind], quantity, kind="line", start=pt((lo[0], c[1], c[2])),
                              end=pt((hi[0], c[1], c[2])), style=PlotStyle(markers=False))
        return ResultPlot(self.plot_titles[kind], quantity, kind="section", origin=pt(c),
                          style=PlotStyle(colormap=colormap))

    def plot_unavailable(self, project, plot) -> str:
        """Why a plot cannot be shown with the current problem ("" if it can)."""
        return ""


    # ---- templates ---------------------------------------------------------------
    def templates(self) -> dict:
        """name -> templates.Template, in display order."""
        return {}
