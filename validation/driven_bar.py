"""Harmonic response of a bar: resonance curves with a loss factor against the exact 1D solution."""

from __future__ import annotations

import numpy as np

import common
from common import Report, Timer, figure, quiet

TITLE = "Driven bar: resonances with losses"
ABOUT = r"""
A bar of length $L$ in uniaxial strain (its side faces are symmetry planes, so compressional waves travel at
$c_L = \sqrt{C_{11}/\rho}$ without dispersion), clamped at $x = 0$ and driven by a force amplitude $F$ on its end
$x = L$, with a loss factor $\eta$ (complex stiffness $C(1 + i\eta)$). Reference: the exact harmonic response
$$u(L) = \frac{F\,\tan(kL)}{C_{11}(1+i\eta)\,A\,k},\qquad k = \frac{\omega}{c_L\sqrt{1 + i\eta}},$$
whose resonances (the bar's quarter-wave modes, $f_n = (2n-1)\,c_L/4L$) have the quality factor $1/\eta$.
Frequency-domain problem swept across the first three resonances; quadratic elements.
"""

E, NU, RHO, ETA = 100e9, 0.25, 2000.0, 0.02
C11 = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))
CL = np.sqrt(C11 / RHO)
L, W, F = 4e-6, 0.2e-6, 1e-6


def exact(f):
    k = 2 * np.pi * f / (CL * np.sqrt(1 + 1j * ETA))
    return F * np.tan(k * L) / (C11 * (1 + 1j * ETA) * W * W * k)


def project(freqs: str):
    from solidocean.physics.solid.model import SolidCondition, SolidPhysics, SolidProblem
    from solidocean.project import Feature, Material, MeshSettings, VariableSweep, Project, Problem
    return Project(
        name="Driven bar", physics_kind="solid", description="Validation: harmonic response of a bar with losses",
        variables={"L": "4[um]", "w": "0.2[um]", "F": "1[uN]", "f": "1[GHz]"},
        materials=[Material("M", {"model": "isotropic", "rho": f"{RHO:g}[kg/m^3]", "E": f"{E / 1e9:g}[GPa]",
                                   "nu": str(NU)})],
        objects=[Feature("block", "Bar", material="M", params={"center": ["L/2", "0", "0"], "size": ["L", "w", "w"]})],
        mesh=MeshSettings(max_size="w/2"),
        solid=SolidPhysics(symmetry=["ymin", "ymax", "zmin", "zmax"], conditions=[
            SolidCondition("fixed", ["xmin"], name="Clamp"),
            SolidCondition("load", ["xmax"], load_type="total", load=["F", "0", "0"], name="Drive")],
            problem=SolidProblem(type="frequency", frequency="f", loss=str(ETA))),
        problem=Problem(sweep=VariableSweep("f", freqs)))


def run(quick: bool = False) -> Report:
    from solidocean.physics.solid.solver import Solver
    r = Report("driven_bar", TITLE, ABOUT)
    with Timer() as tm:
        f1 = CL / (4 * L)
        n = 40 if quick else 120
        p = project(f"linspace(0.2, 5.6, {n}) * {f1:.6g}[Hz]")
        common.save_model(p, "driven_bar")
        res = Solver(p, log=quiet).run()
        fs = res.outer_values
        tip = np.array([res.U[i][np.argmax(res.meshes[i].points[:, 0]), 0] for i in range(len(fs))])
        ref = exact(fs)
        err = np.abs(tip - ref) / np.abs(ref).max()
        fig, ax = figure(6.4, 4.2)
        ff = np.linspace(fs[0], fs[-1], 2000)
        ax.plot(ff / 1e9, np.abs(exact(ff)) * 1e12, "-", color=common.REF_COLOR, lw=1.3, label="exact")
        ax.plot(fs / 1e9, np.abs(tip) * 1e12, "o", mfc="none", ms=3.5, color=common.FEM_COLOR, label="SolidOcean")
        ax.set_yscale("log")
        ax.set_xlabel("Drive frequency (GHz)")
        ax.set_ylabel("End displacement amplitude (pm)")
        ax.legend()
        r.figure(fig, "driven_bar.png", f"Resonance curve of the bar's end (loss factor η = {ETA:g}): the first "
                                        "three quarter-wave resonances against the exact solution.")
        rows = []
        for m in range(1, 4):
            fm = (2 * m - 1) * f1
            i = int(np.argmin(np.abs(fs - fm)))
            rows.append((m, f"{fm / 1e9:.4f}", f"{abs(ref[i]) * 1e12:.2f}", f"{abs(tip[i]) * 1e12:.2f}",
                         f"{abs(tip[i] / ref[i]) - 1:+.1e}"))
        r.table("Amplitude near the resonances (the sweep point closest to each).",
                ["mode", "f_n (GHz)", "exact (pm)", "SolidOcean (pm)", "difference"], rows)
        r.check("largest error relative to the peak amplitude", err.max(), 1e-4)
    r.seconds = tm.seconds
    return r
