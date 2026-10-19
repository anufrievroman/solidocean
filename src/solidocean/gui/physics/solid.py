"""GUI adapter of solid mechanics."""

from __future__ import annotations

import numpy as np

from ... import units
from ...physics.solid.model import SOLID_CONDITION_TITLES
from ..panels import solid as sp
from ..panels.common import equations, references
from .base import PhysicsUI
from .common import condition_highlights

STATIONARY_PLOT_HELP = {
    "quantity": "Any quantity of the displacement, strain and stress fields against the swept variable (Problem → "
                "Variable sweep). Click a point to see the field.",
    "line": "A cross-section: the quantity along a straight line from a start to an end point. Gaps mark holes or "
            "points outside the structure.",
}


# How the plots over the Brillouin zone arrive at their numbers, written for the angular frequency the
# literature uses. The sums run over every band j and every triangle m of the wavevector mesh whose range
# contains the frequency, L is the length of the level set in that triangle and S the copies of the meshed
# region that fill the zone.
IN_OMEGA = ("Written for the angular frequency ω = 2πf, as the literature has it. The plots draw these "
            "against the frequency f instead, so a density per unit frequency is 2π times the one per "
            "unit ω; the group velocity, being a velocity, is the same either way.")
BANDS_OVER_THE_ZONE = (
    "ω<sub>j</sub>(k)", "The j-th lowest eigenfrequency at each wavevector of the mesh, from the same "
    "eigenproblem as a dispersion. Over each triangle it is the plane through the three corner "
    "frequencies, which is where the facets come from; bands are numbered by frequency, not followed "
    "through their crossings.")
MESH_SUMS = (
    "L<sub>jm</sub>(ω),&nbsp; ∇<sub>k</sub>ω<sub>jm</sub>", "A band is a plane over each triangle, so the "
    "level set ω<sub>j</sub>(k) = ω inside it is a straight segment of length L, and its gradient is "
    "constant. Every integral over the zone becomes a sum over the (band, triangle) pairs the level "
    "crosses.")
ZONE_FORMALISM = {
    "kmap": equations(BANDS_OVER_THE_ZONE,
                      ("ω<sub>j</sub>(k) = const", "The filled contours: the level sets of that band, drawn "
                       "on the meshed region with the zone outlined behind it.")),
    "surfaces": equations(BANDS_OVER_THE_ZONE,
                          ("", "Each band is drawn as its own surface over k<sub>x</sub>, k<sub>y</sub>. "
                           "Where two surfaces touch, the branches cross and the numbering swaps between "
                           "them.")),
    "dos": equations(MESH_SUMS,
                     ("D(ω) = (1/4π²) S Σ<sub>j,m</sub> L<sub>jm</sub>(ω) / |∇<sub>k</sub>ω<sub>jm</sub>|",
                      "States per unit angular frequency and unit area: each segment counts for its length "
                      "divided by how fast the band climbs across it. Divide by the membrane thickness for "
                      "a per-volume density."),
                     IN_OMEGA),
    "velocity": equations(MESH_SUMS,
                          ("v<sub>j</sub>(k) = ∇<sub>k</sub>ω<sub>j</sub>", "The group velocity is the "
                           "gradient of the band itself (k is in rad/m)."),
                          ("⟨v⟩(ω) = Σ<sub>j,m</sub> L<sub>jm</sub> / Σ<sub>j,m</sub> (L<sub>jm</sub> / "
                           "|∇<sub>k</sub>ω<sub>jm</sub>|)", "Averaged over the states at that frequency, "
                           "weighted as the density of states weights them — not over the wavevectors."),
                          IN_OMEGA),
    "flux": equations(MESH_SUMS,
                      ("n(ω, T) = 1 / (e<sup>ħω/k<sub>B</sub>T</sup> − 1)", "The Bose-Einstein occupation "
                       "of a mode of angular frequency ω."),
                      ("p(ω, T) = ħω n(ω, T) S Σ<sub>j,m</sub> L<sub>jm</sub>(ω) / 4π³", "The energy a "
                       "convex boundary radiates into the structure, per unit angular frequency and unit "
                       "length of that boundary. Averaging the flux over the directions of that boundary "
                       "cancels the group velocity against the Jacobian of the change of variable from k "
                       "to ω, which is what leaves the bare contour length — and what lets a mesh of the "
                       "irreducible wedge stand for the whole zone, since a flux along one direction could "
                       "not."),
                      IN_OMEGA),
    "power": equations(BANDS_OVER_THE_ZONE,
                       ("P(T) = (S/4π²) Σ<sub>j</sub> ∫ d²k ħω<sub>j</sub> n(ω<sub>j</sub>, T) "
                        "|∇<sub>k</sub>ω<sub>j</sub>| / π", "The power a convex boundary radiates into the "
                        "structure per unit of its length: every state of the zone carries ħω with the "
                        "Bose-Einstein occupation n, at the speed it leaves the boundary. Only states moving "
                        "away from the boundary count, and averaged over its directions that speed is "
                        "|v|/π, with v<sub>j</sub> = ∇<sub>k</sub>ω<sub>j</sub>: Eq. (1) of [1], per unit "
                        "length of the boundary."),
                       ("", "Every triangle of the mesh is divided into 64 equal ones, sampled at their "
                        "centroids; states above the complete range are left out. The samples do not depend "
                        "on the temperature, so they are gathered once for the whole curve. Changing the "
                        "variable from k to ω turns the same integral into ∫ p(ω, T) dω over the spectral "
                        "heat flux."),
                       ("S", "Copies of the meshed region that fill the zone: 8 for the wedge of a square "
                        "lattice, 12 for a hexagonal one, 1 for the whole zone."),
                       references("N. Zen, T. A. Puurtinen, T. J. Isotalo, S. Chaudhuri and I. J. Maasilta, Nat. Commun. 5, 3435 (2014)")),
    "conductance": equations(("∂n/∂T = (ħω/k<sub>B</sub>T²) e<sup>x</sup>/(e<sup>x</sup> − 1)²,&nbsp; "
                              "x = ħω/k<sub>B</sub>T", "How the occupation of a mode answers a change of "
                              "temperature."),
                             ("G(T) = dP/dT = (S/4π²) Σ<sub>j</sub> ∫ d²k ħω<sub>j</sub> "
                              "(∂n/∂T) |∇<sub>k</sub>ω<sub>j</sub>| / π",
                              "The ballistic thermal conductance per unit length of the boundary: what the "
                              "band structure carries when nothing scatters [1]. The integral is taken over "
                              "the zone as for the power, and stops where the computed bands stop being "
                              "complete, so it is a lower bound once k<sub>B</sub>T/ħ approaches that "
                              "frequency."),
                             IN_OMEGA,
                             references("N. Zen, T. A. Puurtinen, T. J. Isotalo, S. Chaudhuri and I. J. Maasilta, Nat. Commun. 5, 3435 (2014)")),
}


class SolidUI(PhysicsUI):
    kind = "solid"
    physics_panel = sp.SolidPanel
    problem_panel = sp.SolidProblemPanel
    field_panel = sp.DisplacementPanel
    node_panels = {"periodicity": sp.PeriodicityPanel, "symmetry": sp.SymmetryPanel, "kpath": sp.KPathPanel,
                   "kmesh": sp.KMeshPanel, "fourier": sp.FourierPanel}
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
        st = project.solid.problem
        if st.field_problem:
            keys = (("sweep", "Variable sweep"),)
        elif st.coherent:
            keys = (("sweep", "Variable sweep"), ("kmesh", "Wavevector mesh"))
        else:
            keys = (("sweep", "Variable sweep"), ("kpath", "Wavevector sweep"),
                    ("fourier", "Fourier components"))
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
        elif kind == "kmesh":
            km = win.project.solid.problem.kmesh
            menu.addAction("Disable" if km is not None and km.enabled else "Enable", lambda: self.toggle_kmesh(win))
        elif kind == "fourier":
            fo = win.project.solid.problem.fourier
            menu.addAction("Disable" if fo is not None and fo.enabled else "Enable",
                           lambda: self.toggle_fourier(win))

    @staticmethod
    def toggle_kpath(win):
        st = win.project.solid.problem
        if st.kpath is None:
            st.kpath = sp.default_kpath(win.project)
        st.kpath.enabled = not st.kpath.enabled
        win.mark_dirty()
        win.rebuild_tree(("kpath", None))

    @staticmethod
    def toggle_kmesh(win):
        st = win.project.solid.problem
        if st.kmesh is None:
            st.kmesh = sp.default_kmesh(win.project)
        st.kmesh.enabled = not st.kmesh.enabled
        win.mark_dirty()
        win.rebuild_tree(("kmesh", None))

    @staticmethod
    def toggle_fourier(win):
        st = win.project.solid.problem
        if st.fourier is None:
            st.fourier = sp.default_fourier(win.project)
        else:
            st.fourier.enabled = not st.fourier.enabled
        win.mark_dirty()
        win.rebuild_tree(("fourier", None))

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
        if st.coherent:
            return "Computing the bands over the Brillouin zone…"
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

    def plot_formalism(self, project, plot):
        """The plots over the zone explain their sums; a spectrum explains the one it draws."""
        key = {"spectrum": plot.spectrum, "conductance": plot.transport}.get(plot.kind, plot.kind)
        return ZONE_FORMALISM.get(key, "")
