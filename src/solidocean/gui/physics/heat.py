"""GUI adapter of heat transfer."""

from __future__ import annotations

import numpy as np

from ... import units
from ..panels import heat as hp
from ..panels.problem import sweep_summary
from .base import PhysicsUI
from .common import condition_highlights

PLOT_HELP = {
    "time": "Any quantity of the temperature field against time (transient problems), one curve per variable "
            "value. Click a point to see the temperature field at that time.",
    "quantity": "Any quantity of the temperature field against the swept variable (Problem → Variable sweep), at "
                "the final or a chosen time. Click a point to see the field.",
    "line": "A cross-section: the quantity along a straight line from a start to an end point, at one time step. "
            "Gaps mark holes or points outside the structure.",
}


class HeatUI(PhysicsUI):
    kind = "heat"
    physics_panel = hp.HeatPanel
    problem_panel = hp.HeatProblemPanel
    field_panel = hp.TemperaturePanel
    node_panels = {"heat_initial": hp.HeatInitialPanel, "heat_symmetry": hp.HeatSymmetryPanel}
    indexed_panels = {"heat_cond": hp.HeatConditionPanel, "heat_src": hp.HeatSourcePanel,
                      "heat_iface": hp.HeatInterfacePanel}
    highlight_nodes = ("heat_initial", "heat_symmetry", "heat_cond", "heat_src")

    # ---- model tree ------------------------------------------------------------------
    def physics_nodes(self, project):
        h = project.heat
        sym = ", ".join(h.symmetry)
        return ([(f"Initial values ({h.initial})", ("heat_initial", None), False),
                 (f"Symmetry ({sym})" if sym else "Symmetry", ("heat_symmetry", None), not sym)] +
                [(hp.heat_condition_label(c), ("heat_cond", i), False) for i, c in enumerate(h.conditions)] +
                [(f"{q.name or 'Heat source'} (in {q.material})", ("heat_src", i), False)
                 for i, q in enumerate(h.sources)] +
                [(hp.interface_label(c), ("heat_iface", i), False) for i, c in enumerate(h.interfaces)])

    def problem_label(self, project):
        typ = project.heat.problem.type
        return f"Problem: {hp.HEAT_PROBLEM_TITLES.get(typ, typ)}"

    def problem_nodes(self, project):
        summary = sweep_summary(project.problem)
        return [(f"Variable sweep ({summary['sweep']})" if summary["sweep"] else "Variable sweep", ("sweep", None),
                 not summary["sweep"])]

    def context_menu(self, win, kind, index, menu):
        if kind == "physics":
            for typ in hp.HEAT_TITLES:
                menu.addAction(f"Add {hp.HEAT_TITLES[typ]}", lambda t=typ: hp.add_condition(win, t))
            menu.addAction("Add Heat source", lambda: hp.add_source(win))
            menu.addSeparator()
            for typ, title in hp.INTERFACE_TITLES.items():
                menu.addAction(f"Add {title}", lambda t=typ: hp.add_interface(win, t))
        elif kind in ("heat_cond", "heat_src", "heat_iface"):
            menu.addAction("Delete", lambda: hp.delete_item(win, kind, index))

    def delete_node(self, win, kind, index):
        if kind not in ("heat_cond", "heat_src", "heat_iface"):
            return False
        hp.delete_item(win, kind, index)
        return True

    def highlights(self, project):
        return condition_highlights(project, project.heat.conditions,
                                    {face: "symmetry" for face in project.heat.symmetry})

    # ---- results ---------------------------------------------------------------------
    def run_label(self, project):
        return {"transient": "Solving heat transfer (transient)…",
                "frequency": "Solving heat transfer (frequency domain)…"}.get(project.heat.problem.type,
                                                                              "Solving heat transfer…")

    def show_field(self, win):
        """The selected time step and variable value, the structure coloured by the field plot's quantity."""
        from ... import fields
        from ...physics.heat.fields import at_nodes
        from ...results.field import named_quantity
        from ...results.style import time_axis
        r = win.results
        i = min(win.selection["i"], len(r.T) - 1)
        j = min(win.selection["j"], len(r.times) - 1)
        key = (id(r), i, j)
        if key not in win.field_cache:          # nodal fields of the displayed solution only
            win.field_cache = {key: at_nodes(r.meshes[i], r.T[i][j], r.kels[i])}
        expr = win.field_quantity.strip() or ("Tamp" if np.iscomplexobj(r.T[i]) else "T")
        ev = win.results_project.evaluator()
        if r.outer_name:
            ev = ev.with_overrides({r.outer_name: r.outer_values[i]})
        values = fields.evaluate_nodal(expr, r.meshes[i].points, {**ev.values, "time": r.times[j]},
                                       win.field_cache[key])
        label = named_quantity(win.project, expr)
        title = label.split(" (")[0]
        if len(r.times) > 1:
            scale, unit = time_axis(win.project)
            title += f", t = {r.times[j] / scale:.4g} {unit}"
        if r.outer_name:
            dim = r.info.get("outer_dim")
            title += f", {r.outer_name} = " + units.format_value(r.outer_values[i], tuple(dim) if dim else None,
                                                                  win.project.display_units)
        win.viewer.show_scalar(r.meshes[i], values, title, label, win.field_style)

    def results_summary(self, win, r):
        parts = ["frequency-domain heat transfer" if r.kind == "frequency" else f"{r.kind} heat transfer"]
        if len(r.times) > 1:
            parts.append(f"{len(r.times)} saved time steps up to {units.format_value(r.times[-1], units.TIME, win.project.display_units)}")
        if r.outer_name:
            parts.append(f"variable sweep over {r.outer_name} ({len(r.outer_values)} values)")
        if np.iscomplexobj(r.T[0]):
            parts.append(f"temperature amplitude up to {max(np.abs(T).max() for T in r.T):.5g} K")
        else:
            parts.append(f"T from {min(T.min() for T in r.T):.5g} to {max(T.max() for T in r.T):.5g} K")
        if r.info.get("elapsed_s") is not None:
            parts.append(f"{r.info['elapsed_s']} s")
        return "; ".join(parts) + "."

    def quantity_placeholder(self, project):
        return "max(T)"

    def line_help(self, project):
        return ("Evaluated point by point along the line: T, Tx, Ty, Tz, qx, qy, qz, x, y, z, e.g. "
                "sqrt(qx^2+qy^2+qz^2) for the heat-flux magnitude.")

    def plot_help(self, project, kind):
        return PLOT_HELP.get(kind, "")
