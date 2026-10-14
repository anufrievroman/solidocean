"""Solid mechanics: linear elasticity. Eigenfrequency problems (modes, phonon dispersion of periodic
structures with Bloch k-sweeps), stationary problems (displacement, strain and stress under loads) and
frequency-domain problems (the harmonic response to oscillating loads and displacements)."""

from __future__ import annotations

from ... import materials as mat
from ..base import PhysicsSpec
from . import fields as solid_fields
from . import plots
from .model import PROBLEM_TYPES, SolidPhysics, validate_settings

DISPLACEMENT_MAGNITUDE = "sqrt(abs(ux)^2 + abs(uy)^2 + abs(uz)^2)"   # also the amplitude of a harmonic response

MODAL_HELP = {
    "Scalars": "freq (Hz), omega (rad/s), kx, ky, kz, k (rad/m), mode (band number), any model variable, pi",
    "Fields": "ux, uy, uz (complex displacement, normalised so mean(|u|²) = 1), strains exx..exy, stresses "
              "sxx..sxy (Pa) and mises of the normalised mode, x, y, z (m)",
    "Reduce fields with": "mean(f), integral(f), max(f), min(f)",
    "Functions": "abs, real, imag, conj, angle, sqrt, exp, log, sin, cos, tan, atan2, if(cond, a, b)",
    "Examples": "freq · mean(abs(uz)^2) (out-of-plane fraction) · mean(abs(ux)^2*(z>0)) · max(abs(uz))",
}
HARMONIC_HELP = {
    "Scalars": "any model variable (e.g. the swept frequency f), pi",
    "Fields": "ux, uy, uz (complex displacement amplitude, m: u(t) = Re(u e^{iωt})), strains exx..exy, stresses "
              "sxx..sxy (Pa, complex), mises (of the real part), x, y, z (m)",
    "Reduce fields with": "mean(f), integral(f), max(f), min(f); over one object: mean(f, \"probe\")",
    "Functions": "abs, real, imag, conj, angle, sqrt, exp, log, sin, cos, tan, atan2, if(cond, a, b)",
    "Examples": "mean(abs(uz), \"probe\") (amplitude of the out-of-plane motion in the object 'probe') · "
                "max(abs(ux)) · angle(mean(uz, \"probe\")) (phase, rad)",
}
STATIC_HELP = {
    "Scalars": "any model variable, pi",
    "Fields": "ux, uy, uz (displacement, m), exx, eyy, ezz, eyz, exz, exy (strain), sxx, syy, szz, syz, sxz, sxy "
              "(stress, Pa), mises (von Mises stress, Pa), x, y, z (m)",
    "Reduce fields with": "mean(f), integral(f), max(f), min(f); over one object (one of the Objects, "
                          "or a material): mean(f, \"pad\")",
    "Functions": "abs, sqrt, exp, log, sin, cos, tan, atan2, if(cond, a, b)",
    "Examples": "min(uz) (largest downward deflection) · max(mises) · mean(sxx) / mean(exx) (Young's modulus "
                "along x) · mean(uz, \"pad\") · -min(uz) / 1[nm] (deflection in nm)",
}


class SolidSpec(PhysicsSpec):
    kind = "solid"
    title = "Solid Mechanics"
    description = "Elastic eigenfrequencies, phonon dispersion and mode shapes; stationary displacement and " \
                  "stress under loads; driven (harmonic) response"
    section = "solid"
    settings_class = SolidPhysics
    plot_kinds = plots.PLOT_KINDS
    plot_titles = plots.PLOT_TITLES
    field_names = solid_fields.FIELD_NAMES
    scalar_names = ("freq", "omega", "kx", "ky", "kz", "k", "mode")
    problem_types = PROBLEM_TYPES

    def problem_type(self, project) -> str:
        return project.solid.problem.type

    def field_results(self, project) -> bool:
        return project.solid.problem.field_problem      # eigenfrequency problems give modal results

    def validate(self, project, ev) -> list[str]:
        errors = []
        for m in project.materials:
            try:
                mat.evaluate(m.props, ev)
            except Exception as exc:
                errors.append(f"material '{m.name}': {exc}")
        return errors + validate_settings(project, ev, [m.name for m in project.materials])

    def solver(self, project, log=print, progress=None, cancel=None):
        from .solver import Solver
        return Solver(project, log=log, progress=progress, cancel=cancel)

    def quantity_help(self, project) -> dict[str, str]:
        if project.solid.problem.harmonic:
            return HARMONIC_HELP
        return STATIC_HELP if project.solid.problem.field_problem else MODAL_HELP

    def quantities(self, project) -> list[tuple[str, str]]:
        if project.solid.problem.harmonic:
            return ([("Displacement amplitude (m)", DISPLACEMENT_MAGNITUDE)]
                    + [(f"Displacement amplitude {a} (m)", f"abs(u{a})") for a in "xyz"]
                    + [(f"Displacement {a}, in phase (m)", f"real(u{a})") for a in "xyz"]
                    + [("von Mises stress (Pa)", "mises")])
        if project.solid.problem.field_problem:
            out = [("Displacement magnitude", DISPLACEMENT_MAGNITUDE)]
        else:   # the scalars of every mode first, then its fields
            out = [("Eigenfrequency (Hz)", "freq"), ("Angular frequency (rad/s)", "omega"),
                   ("Wavevector magnitude (rad/m)", "k")]
            out += [(f"Wavevector {a} (rad/m)", f"k{a}") for a in "xyz"]
            out += [("Mode number", "mode")]
        out += [(f"Displacement {a}", f"u{a}") for a in "xyz"]
        out += [("von Mises stress (Pa)", "mises")]
        out += [(f"Stress {c} (Pa)", f"s{c}") for c in ("xx", "yy", "zz", "yz", "xz", "xy")]
        out += [(f"Strain {c}", f"e{c}") for c in ("xx", "yy", "zz", "yz", "xz", "xy")]
        return out

    def field_quantity(self, project) -> str:
        return DISPLACEMENT_MAGNITUDE

    def field_plot(self, name: str = ""):
        return plots.field_plot(name)

    def default_plots(self, project) -> list:
        return plots.default_plots(project.solid.problem.field_problem)

    def new_plot(self, project, kind: str, box):
        from ...plots import PlotStyle, ResultPlot
        title = self.plot_titles[kind]
        if kind == "field":
            return self.field_plot()
        if kind == "colored":
            return plots.colored_plot()
        if kind == "dispersion":
            return ResultPlot()
        if kind == "line":
            return self.probe_plot(project, kind, box, "uz", "viridis")
        if kind == "section":
            return self.probe_plot(project, kind, box, "", "viridis")
        if project.solid.problem.harmonic:
            return ResultPlot(title, "max(abs(uz))", kind="quantity", y_label="Largest amplitude of uz (m)",
                              style=PlotStyle(x_log=False, y_log=True))
        if project.solid.problem.field_problem:
            return ResultPlot(title, "max(mises)", kind="quantity", y_label="Largest von Mises stress (Pa)")
        return ResultPlot(title, "mean(abs(uz)^2)", kind="quantity")

    def plot_unavailable(self, project, plot) -> str:
        if project.solid.problem.field_problem:
            if plot.kind in ("dispersion", "colored"):
                return "Dispersion plots need an eigenfrequency problem (Problem → Problem type)."
            if plot.kind == "quantity" and not project.problem.active_sweep:
                return "Sweep plots need a variable sweep (Problem → Variable sweep)."
        elif plot.kind in ("line", "section"):
            return "Line and cross-section plots need a stationary or frequency-domain problem (Problem → Problem type)."
        return ""

    def templates(self) -> dict:
        from .templates import TEMPLATES
        return TEMPLATES


SPEC = SolidSpec()
