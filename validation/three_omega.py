"""3ω method: a heater strip on a semi-infinite substrate, frequency-domain heat transfer against Cahill's
formula."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

import common
from common import Report, Timer, figure, quiet

TITLE = "3ω heater line: frequency-domain heat transfer"
ABOUT = r"""
A long heater strip of width $2b$ on a substrate (conductivity $k$, diffusivity $\alpha$) dissipates a power per
length oscillating as $P\,e^{i\omega t}$ (in a 3ω measurement, the $2\omega$ part of the Joule heating). Reference:
Cahill's exact average temperature oscillation of the strip on a semi-infinite substrate (Rev. Sci. Instrum. 61,
802 (1990)),
$$\Delta \hat T(\omega) = \frac{P}{\pi k}\int_0^\infty \frac{\sin^2(\lambda b)}{(\lambda b)^2\sqrt{\lambda^2 +
i\omega/\alpha}}\,d\lambda,$$
whose in-phase part falls linearly with $\ln\omega$ (slope $-P/2\pi k$, from which 3ω measurements obtain $k$)
while the out-of-phase part stays near $-P/4k$. The model is a slice of the substrate along the line (its two
cut faces insulated: an infinite line), half of it by symmetry, with the heat flux on the strip (a thin block of
the substrate's material under the surface, so the mesh follows the strip's edge) and the far faces held at the
ambient temperature, more than five thermal penetration depths $\sqrt{\alpha/\omega}$ away at the lowest
frequency. Frequency-domain problem, swept over the frequency; quadratic elements.
"""

K, RHOC = 1.4, 1.63e6                # SiO2-like substrate
ALPHA = K / RHOC
B = 1e-6                             # strip half-width
P = 1.0                              # W/m


def cahill(f: float, periods: int = 400) -> complex:
    """Mean temperature oscillation of the strip (K) at frequency f (Hz): the integral over u = lambda b period by
    period of sin^2 up to u = periods * pi, plus the tail, where the integrand tends to sin^2(u) / u^3 (1 / 4U^2)."""
    q2 = 1j * 2 * np.pi * f / ALPHA

    def g(u, part):
        v = np.sin(u) ** 2 / (u * u * np.sqrt((u / B) ** 2 + q2)) / B
        return v.real if part == 0 else v.imag
    total = 0j
    for n in range(periods):
        a, b = n * np.pi, (n + 1) * np.pi
        total += quad(g, a, b, args=(0,), epsabs=0)[0] + 1j * quad(g, a, b, args=(1,), epsabs=0)[0]
    return P / (np.pi * K) * (total + 1 / (4 * (periods * np.pi) ** 2))


def project(freqs: str, h: str = "0.25[um]"):
    from solidocean.physics.heat.model import HeatCondition, HeatPhysics, HeatProblem
    from solidocean.project import Feature, Material, MeshSettings, VariableSweep, Project, Problem

    def mat(name):
        return Material(name, {"model": "isotropic", "rho": "2200[kg/m^3]", "E": "70[GPa]", "nu": "0.17",
                               "k": f"{K:g}[W/(m*K)]", "cp": f"{RHOC / 2200:g}[J/(kg*K)]"})
    return Project(
        name="3 omega strip", physics_kind="heat",
        description="Validation: heater strip on a semi-infinite substrate, frequency domain (Cahill 1990)",
        variables={"b": "1[um]", "D": "20[um]", "Lx": "0.25[um]", "P": f"{P:g}[W/m]", "f": "10[kHz]"},
        materials=[mat("Substrate")],
        objects=[Feature("block", "Substrate", material="Substrate",
                          params={"center": ["0", "D/2", "-D/2"], "size": ["Lx", "D", "D"]}),
                  Feature("block", "Strip", material="Substrate",
                          params={"center": ["0", "b/2", "-b/8"], "size": ["Lx", "b", "b/4"]})],
        mesh=MeshSettings(max_size=h, order=2),
        heat=HeatPhysics(initial="300[K]", conditions=[
            HeatCondition("flux", ["zmax"], "P / (2*b)", name="Heater", on="Strip"),
            HeatCondition("temperature", ["ymax", "zmin"], "300[K]", name="Far field")],
            problem=HeatProblem(type="frequency", frequency="f")),
        problem=Problem(sweep=VariableSweep("f", freqs)))


def strip_mean(mesh, T) -> complex:
    """Mean of the (complex) temperature over the heated strip's surface."""
    from solidocean.fem import surface
    tris = surface.boundary_faces(mesh.cells, mesh.order)
    c = mesh.points[tris[:, :3]]
    on = np.all(np.abs(c[..., 2]) < 1e-12, axis=1) & np.all(c[..., 1] <= B * (1 + 1e-9), axis=1)
    _, w = surface.surface_terms(mesh.points, tris[on], mesh.order, load=1.0)
    return complex(w @ T / w.sum())


def run(quick: bool = False) -> Report:
    from solidocean.physics.heat.solver import HeatSolver
    r = Report("three_omega", TITLE, ABOUT)
    with Timer() as tm:
        p = project("logspace(4, 6, 5)" if quick else "logspace(4, 6, 9)")
        p.problem.sweep.values += " * 1[Hz]"
        common.save_model(p, "three_omega")
        res = HeatSolver(p, log=quiet).run()
        fs = res.outer_values
        fem = np.array([strip_mean(res.meshes[i], res.T[i][0] - 0) for i in range(len(fs))])
        ref = np.array([cahill(f) for f in fs])
        err = np.abs(fem - ref) / np.abs(ref)
        dphase = np.degrees(np.angle(fem) - np.angle(ref))
        fig, ax = figure(6.2, 4.4)
        ff = np.logspace(np.log10(fs[0]), np.log10(fs[-1]), 60)
        rr = np.array([cahill(f) for f in ff])
        ax.plot(ff, rr.real, "-", color=common.SHORE[0], lw=1.2, label="In-phase, Cahill")
        ax.plot(ff, rr.imag, "-", color=common.SHORE[2], lw=1.2, label="Out-of-phase, Cahill")
        ax.plot(fs, fem.real, "o", color=common.SHORE[0], label="In-phase, SolidOcean")
        ax.plot(fs, fem.imag, "o", color=common.SHORE[2], label="Out-of-phase, SolidOcean")
        ax.set_xscale("log")
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel(r"Frequency, $f$ (Hz)")
        ax.set_ylabel(r"Temperature oscillation, $\Delta\hat{T}$ (K)")
        ax.legend()
        r.figure(fig, "three_omega.png", "In-phase and out-of-phase temperature oscillation of the heater strip "
                                         "against Cahill's formula.")
        slope_fem = np.polyfit(np.log(2 * np.pi * fs[:3]), fem.real[:3], 1)[0]
        slope_ref = np.polyfit(np.log(2 * np.pi * fs[:3]), ref.real[:3], 1)[0]
        k_fit = -P / (2 * np.pi * slope_fem)
        r.notes.append(f"The 3ω slope method applied to the three lowest frequencies gives k = {k_fit:.3f} W/(m·K) "
                       f"(true {K:g}; Cahill's formula over the same points: {-P / (2 * np.pi * slope_ref):.3f}).")
        r.check("amplitude error, every frequency", err.max(), 1e-2)
        r.check("phase error (degrees), every frequency", np.abs(dphase).max(), 0.5, fmt=".2f")
    r.seconds = tm.seconds
    return r
