"""GUI adapter of solid mechanics."""

from __future__ import annotations

import numpy as np

from ... import units
from ...physics.solid.model import SOLID_CONDITION_TITLES
from ..panels import solid as sp
from .base import PhysicsUI
from .common import condition_highlights

STATIONARY_PLOT_HELP = {
    "quantity": "Any quantity of the displacement, strain and stress fields against the swept variable (Problem → "
                "Variable sweep). Click a point to see the field.",
    "line": "A cross-section: the quantity along a straight line from a start to an end point. Gaps mark holes or "
            "points outside the structure.",
}


class SolidUI(PhysicsUI):
    kind = "solid"
    physics_panel = sp.SolidPanel
    problem_panel = sp.SolidProblemPanel
    field_panel = sp.DisplacementPanel
    node_panels = {"periodicity": sp.PeriodicityPanel, "symmetry": sp.SymmetryPanel, "kpath": sp.KPathPanel}
    indexed_panels = {"solid_cond": sp.SolidConditionPanel}
    highlight_nodes = ("periodicity", "symmetry", "solid_cond")

    # ---- model tree ------------------------------------------------------------------
    def physics_nodes(self, project):
        summary = sp.physics_summary(project)
        nodes = [(label if summary[key] == "none" else f"{label} ({summary[key]})", (key, None), summary[key] == "none")
                 for key, label in (("periodicity", "Periodicity"), ("symmetry", "Symmetry"))]
        return nodes + [(sp.solid_condition_label(c), ("solid_cond", i), False)
                        for i, c in enumerate(project.solid.conditions)]

    def problem_label(self, project):
        return f"Problem: {sp.problem_summary(project)['problem_type']}"

    def problem_nodes(self, project):
        summary = sp.problem_summary(project)
        keys = (("sweep", "Variable sweep"),) if project.solid.problem.field_problem else \
            (("sweep", "Variable sweep"), ("kpath", "Wavevector sweep"))
        return [(f"{label} ({summary[key]})" if summary[key] else label, (key, None), not summary[key])
                for key, label in keys]

    def context_menu(self, win, kind, index, menu):
        if kind == "physics":
            for typ, title in SOLID_CONDITION_TITLES.items():
                menu.addAction(f"Add {title}", lambda t=typ: sp.add_condition(win, t))
        elif kind == "solid_cond":
            menu.addAction("Delete", lambda: sp.delete_condition(win, index))
        elif kind == "kpath":
            kp = win.project.solid.problem.kpath
            menu.addAction("Disable" if kp is not None and kp.enabled else "Enable", lambda: self.toggle_kpath(win))

    @staticmethod
    def toggle_kpath(win):
        st = win.project.problem
        if st.kpath is None:
            st.kpath = sp.default_kpath(win.project)
        st.kpath.enabled = not st.kpath.enabled
        win.mark_dirty()
        win.rebuild_tree(("kpath", None))

    def delete_node(self, win, kind, index):
        if kind != "solid_cond":
            return False
        sp.delete_condition(win, index)
        return True

    def highlights(self, project):
        return condition_highlights(project, project.solid.conditions, project.solid.face_conditions(project.lattice))

    # ---- results ---------------------------------------------------------------------
    def run_label(self, project):
        st = project.solid.problem
        if st.harmonic:
            return "Solving the harmonic response…"
        return "Solving stationary solid mechanics…" if st.field_problem else "Computing eigenfrequencies…"

    def show_field(self, win):
        """The stationary displacement of the selected variable value, on the deformed structure coloured by
        the field plot's quantity (displacements in the length unit, stresses in MPa)."""
        from ... import fields
        from ...physics.solid import DISPLACEMENT_MAGNITUDE
        from ...physics.solid.fields import STRESSES, at_nodes
        from ...results.field import named_quantity
        r = win.results
        i = min(win.selection["i"], len(r.U) - 1)
        mesh, u = r.meshes[i], r.U[i]
        key = (id(r), i)
        harmonic = np.iscomplexobj(u)
        if key not in win.field_cache:          # nodal fields of the displayed solution only
            win.field_cache = {key: at_nodes(mesh, u, r.cels[i], keep_complex=harmonic)}
        expr = win.field_quantity.strip() or DISPLACEMENT_MAGNITUDE
        ev = win.results_project.evaluator()
        if r.outer_name:
            ev = ev.with_overrides({r.outer_name: r.outer_values[i]})
        values = fields.evaluate_nodal(expr, mesh.points, ev.values, win.field_cache[key])
        factor, lu = units.axis_scale(units.LENGTH, win.project.length_unit)
        text = " ".join(expr.split())
        label = named_quantity(win.project, expr)
        if text in (DISPLACEMENT_MAGNITUDE, "ux", "uy", "uz", "abs(ux)", "abs(uy)", "abs(uz)",
                    "sqrt(ux^2 + uy^2 + uz^2)"):                          # lengths: in the length unit
            values, label = values * factor, f"{label.split(' (')[0]} ({lu})"
        elif text in ("mises",) + STRESSES:                              # stresses: in MPa
            values, label = values / 1e6, f"{label.split(' (')[0]} (MPa)"
        amp = np.linalg.norm(np.abs(u), axis=1)
        title = (f"Harmonic response: largest amplitude {amp.max() * factor:.4g} {lu}" if harmonic else
                 f"Stationary: largest displacement {amp.max() * factor:.4g} {lu}")
        if r.outer_name:
            title += f", {sp.outer_labels(win)[i]}"
        win.viewer.show_static(mesh, u, values, title, label, win.deform_scale, win.field_style, "%.3g")

    def results_summary(self, win, r):
        harmonic = np.iscomplexobj(r.U[0])
        parts = ["harmonic response (frequency domain)" if harmonic else "stationary solid mechanics"]
        if r.outer_name:
            parts.append(f"variable sweep over {r.outer_name} ({len(r.outer_values)} values)")
        factor, lu = units.axis_scale(units.LENGTH, win.project.length_unit)
        top = max(np.linalg.norm(np.abs(u), axis=1).max() for u in r.U) * factor
        parts.append(f"largest {'amplitude' if harmonic else 'displacement'} {top:.4g} {lu}")
        if r.info.get("elapsed_s") is not None:
            parts.append(f"{r.info['elapsed_s']} s")
        return "; ".join(parts) + "."

    def quantity_placeholder(self, project):
        if project.solid.problem.harmonic:
            return "max(abs(uz))"
        return "max(mises)" if project.solid.problem.field_problem else "mean(abs(uz)^2)"

    def line_help(self, project):
        return ("Evaluated point by point along the line: ux, uy, uz, strains exx..exy, stresses sxx..sxy, mises, "
                "x, y, z, e.g. sqrt(ux^2+uy^2+uz^2) for the displacement magnitude.")

    def plot_help(self, project, kind):
        return STATIONARY_PLOT_HELP.get(kind, "") if project.solid.problem.field_problem else ""
