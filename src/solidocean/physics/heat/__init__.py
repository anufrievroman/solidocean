"""Heat transfer: conduction (Fourier's law), stationary, transient or in the frequency domain, with convection,
radiation, fluxes, fixed temperatures, volume sources and pulses; material properties may depend on the
temperature."""

from __future__ import annotations

import numpy as np

from ..base import PhysicsSpec
from . import fields as heat_fields
from . import plots
from ...fields import SURFACE_HELP
from .model import HEAT_PROBLEM_TYPES, HeatPhysics, validate_settings
from .properties import REFERENCE_T, Material

HELP = {
    "Scalars": "time (s), any model variable, pi",
    "Fields": "T (K), Tx, Ty, Tz (temperature gradient, K/m), qx, qy, qz (heat flux -k∇T, W/m²), x, y, z (m); "
              "frequency domain: Tamp, Tphase (amplitude, K, and phase, degrees, of the oscillation), Tre, Tim (its "
              "in-phase and out-of-phase parts)",
    "Reduce fields with": "mean(f), integral(f), max(f), min(f); over one object (one of the Objects, "
                          "or a material): mean(f, \"pad\")" + SURFACE_HELP,
    "Functions": "abs, sqrt, exp, log, sin, cos, tan, atan2, if(cond, a, b)",
    "Examples": "mean(T) · mean(T, \"pad\") (average temperature of the object 'pad') · max(T) · mean(T*(x>0)) · "
                "integral(qx) / a (heat flow per length) · -mean(qx)·L/ΔT (effective conductivity) · "
                "integral(qx*nx + qy*ny + qz*nz, \"xmax\") (heat leaving through the xmax face, W) · "
                "mean(T, \"Film:zmax\") (temperature of the film's top surface) · "
                "sqrt(mean(Tre)^2 + mean(Tim)^2) (amplitude of the mean oscillation) · "
                "atan2(mean(Tim), mean(Tre)) * 180/pi (its phase)",
}


class HeatSpec(PhysicsSpec):
    kind = "heat"
    title = "Heat Transfer"
    description = "Heat conduction (Fourier): stationary, transient and frequency-domain temperature fields"
    section = "heat"
    settings_class = HeatPhysics
    plot_kinds = plots.PLOT_KINDS
    plot_titles = plots.PLOT_TITLES
    field_names = heat_fields.FIELD_NAMES
    scalar_names = ("time",)
    problem_types = HEAT_PROBLEM_TYPES

    def problem_type(self, project) -> str:
        return project.heat.problem.type

    def validate(self, project, ev) -> list[str]:
        errors = []
        used = {f.material for f in project.objects if f.enabled and f.op == "add"}
        for m in project.materials:
            if m.name in used:
                try:
                    props = Material(m.name, m.props, ev)
                    props.conductivity(np.array(REFERENCE_T)), props.capacity(np.array(REFERENCE_T))
                except Exception as exc:
                    msg = str(exc)
                    errors.append(msg if msg.startswith("material") else f"material '{m.name}': {msg}")
        return errors + validate_settings(project, ev, [m.name for m in project.materials])

    def solver(self, project, log=print, progress=None, cancel=None):
        from .solver import HeatSolver
        return HeatSolver(project, log=log, progress=progress, cancel=cancel)

    def quantity_help(self, project) -> dict[str, str]:
        return HELP

    def quantities(self, project) -> list[tuple[str, str]]:
        timed = [("Time (s)", "time")] if project.heat.problem.type == "transient" else []
        if project.heat.problem.type == "frequency":
            return [("Temperature amplitude (K)", "Tamp"), ("Temperature phase (°)", "Tphase"),
                    ("In-phase temperature (K)", "Tre"), ("Out-of-phase temperature (K)", "Tim"),
                    ("Amplitude of the mean temperature (K)", "sqrt(mean(Tre)^2 + mean(Tim)^2)"),
                    ("Phase of the mean temperature (°)", "atan2(mean(Tim), mean(Tre)) * 180/pi")]
        return ([("Temperature (K)", "T")]
                + [(f"Heat flux {a} (W/m²)", f"q{a}") for a in "xyz"]
                + [("Heat flux magnitude (W/m²)", "sqrt(qx^2 + qy^2 + qz^2)")]
                + [(f"Temperature gradient {a} (K/m)", f"T{a}") for a in "xyz"]
                + [("Temperature gradient magnitude (K/m)", "sqrt(Tx^2 + Ty^2 + Tz^2)")] + timed)

    def field_quantity(self, project) -> str:
        return "Tamp" if project.heat.problem.type == "frequency" else "T"

    def field_plot(self, name: str = ""):
        return plots.field_plot(name)

    def default_plots(self, project) -> list:
        return plots.default_plots(project.heat.problem.type == "transient")

    def new_plot(self, project, kind: str, box):
        from ...plots import PlotStyle, ResultPlot
        title = self.plot_titles[kind]
        if kind == "field":
            return self.field_plot()
        if kind == "time":
            return plots.time_plot("Time plot", "max(T)", "Maximum temperature (K)")
        if kind in ("line", "section"):
            return self.probe_plot(project, kind, box, "T", "magma")
        return ResultPlot(title, "max(T)", kind="quantity", y_label="Maximum temperature (K)")

    def plot_unavailable(self, project, plot) -> str:
        if plot.kind == "time" and project.heat.problem.type != "transient":
            return "Time plots need a transient problem (Problem → Problem type)."
        if plot.kind == "quantity" and not project.problem.active_sweep:
            return "Sweep plots need a variable sweep (Problem → Variable sweep)."
        return ""

    def templates(self) -> dict:
        from .templates import TEMPLATES
        return TEMPLATES


SPEC = HeatSpec()
