"""Transient heating of a semi-infinite solid by a constant surface flux: the erfc solution."""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

import common
from common import Report, Timer, figure, order, quiet

TITLE = "Semi-infinite solid heated by a surface flux"
ABOUT = r"""
A constant heat flux $q$ switched on at $t = 0$ on the face $x = 0$ of a solid initially at $T_0$; the bar is
long ($L = 8\sqrt{\alpha t_\mathrm{end}}$, far end held at $T_0$, insulated sides) so it behaves as a
semi-infinite solid. Reference (Carslaw & Jaeger):
$$T - T_0 = \frac{2q}{k}\sqrt{\frac{\alpha t}{\pi}}\,e^{-x^2/4\alpha t}
- \frac{q\,x}{k}\,\mathrm{erfc}\frac{x}{2\sqrt{\alpha t}},\qquad \alpha = \frac{k}{\rho c_p}.$$
Errors are the largest nodal error at $t_\mathrm{end}$, relative to the surface temperature rise. The time
convergence uses a fine mesh, the mesh convergence a fine time step. Backward Euler is first order. Crank-Nicolson
is second order once the step resolves the start: the abrupt switch-on excites the stiff (fine-scale) modes,
which Crank-Nicolson barely damps (its amplification factor tends to −1), so coarse steps leave a decaying
oscillation; this is why backward Euler is the default for abrupt loads.
"""

K, RHOC, Q, T0 = 100.0, 2e6, 1e9, 300.0
ALPHA = K / RHOC
LB = 8e-6
T_END = (LB / 8) ** 2 / ALPHA


def exact(x, t):
    s = np.sqrt(ALPHA * t)
    return T0 + 2 * Q / K * s / np.sqrt(np.pi) * np.exp(-x ** 2 / (4 * s * s)) - Q * x / K * erfc(x / (2 * s))


RISE = exact(0.0, T_END) - T0


def project(steps: int, theta: float, h: str = "w/2", order_: int = 2):
    from solidocean.physics.heat.model import HeatCondition, HeatPhysics, HeatProblem
    from solidocean.project import Feature, Material, MeshSettings, Project
    return Project(
        name="Semi-infinite solid", physics_kind="heat",
        description="Validation: semi-infinite solid heated by a constant surface flux (erfc solution)",
        variables={"Lb": f"{LB * 1e6:g}[um]", "w": "0.25[um]", "q": f"{Q:g}[W/m^2]", "T0": f"{T0:g}[K]"},
        materials=[Material("M", {"model": "isotropic", "rho": "1000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                                  "k": f"{K:g}[W/(m*K)]", "cp": f"{RHOC / 1000:g}[J/(kg*K)]"})],
        objects=[Feature("block", "Bar", material="M", params={"center": ["Lb/2", "0", "0"], "size": ["Lb", "w", "w"]})],
        mesh=MeshSettings(max_size=h, order=order_),
        heat=HeatPhysics(initial="T0", conditions=[HeatCondition("flux", ["xmin"], "q", name="Surface flux"),
                                                   HeatCondition("temperature", ["xmax"], "T0", name="Far end")],
                         problem=HeatProblem(type="transient", t_end=f"{T_END:.6g}[s]", steps=steps, theta=theta)))


def solve(p):
    from solidocean.physics.heat.solver import HeatSolver
    res = HeatSolver(p, log=quiet).run()
    return res, res.meshes[0].points[:, 0]


def error(res, x) -> float:
    return float(np.abs(res.T[0][-1] - exact(x, res.times[-1])).max() / RISE)


def run(quick: bool = False) -> Report:
    r = Report("semi_infinite", TITLE, ABOUT)
    with Timer() as tm:
        p = project(400, 0.5)
        common.save_model(p, "semi_infinite")
        res, x = solve(p)
        fig, ax = figure()
        xs = np.linspace(0, 4e-6, 300)
        order_x = np.argsort(x)
        for n, frac in enumerate((1 / 16, 1 / 4, 1)):
            j = int(round(frac * (len(res.times) - 1)))
            t = res.times[j]
            ax.plot(xs * 1e6, exact(xs, t) - T0, "-", color=common.COLORS[n], lw=1.2)
            sel = order_x[x[order_x] <= 4e-6][::2]
            ax.plot(x[sel] * 1e6, res.T[0][j][sel] - T0, "o", mfc="none", ms=4, color=common.COLORS[n],
                    label=f"t = {t * 1e9:.3g} ns")
        ax.set_xlabel("Depth x (µm)")
        ax.set_ylabel("Temperature rise T − T₀ (K)")
        ax.legend(title="lines: exact", fontsize=8)
        r.figure(fig, "semi_infinite_profiles.png", "Temperature profiles at three times (Crank-Nicolson, 400 steps, "
                                                    "quadratic elements) against the erfc solution.")
        profile_err = error(res, x)

        steps = [10, 20, 40, 80] if quick else [10, 20, 40, 80, 160, 320]
        errs = {1.0: [], 0.5: []}
        for theta in errs:
            for n in steps:
                errs[theta].append(error(*solve(project(n, theta, "w/4"))))
        be = order(1 / np.array(steps), errs[1.0])
        dts = T_END / np.array(steps)
        fig, ax = figure()
        ax.plot(dts * 1e9, errs[1.0], "o-", color="C0", label=f"backward Euler: order {be:.2f}")
        ax.plot(dts * 1e9, errs[0.5], "s-", color="C3", label="Crank-Nicolson")
        common.slope_guide(ax, dts * 1e9, np.array(errs[1.0]), 1)
        common.convergence_axes(ax, "Time step (ns)", "Largest error / surface temperature rise")
        ax.legend(fontsize=8)
        r.figure(fig, "semi_infinite_time.png", "Time-step convergence at t_end (mesh h = w/4, quadratic elements).")
        r.table("Time-step convergence: largest nodal error at t_end relative to the surface temperature rise.",
                ["steps", "dt (ns)", "backward Euler", "Crank-Nicolson"],
                [(n, f"{d * 1e9:.3g}", f"{a:.2e}", f"{b:.2e}") for n, d, a, b in zip(steps, dts, errs[1.0], errs[0.5])])

        sizes = ["0.5[um]", "0.25[um]", "0.125[um]"] + ([] if quick else ["0.0625[um]"])
        hs, errh = {1: [], 2: []}, {1: [], 2: []}
        for o in (1, 2):
            for h in sizes:
                res_h, x_h = solve(project(2000, 0.5, h, o))
                hs[o].append(common.mean_edge(res_h.meshes[0]) * 1e6)
                errh[o].append(error(res_h, x_h))
        orders = {o: order(hs[o], errh[o]) for o in (1, 2)}
        fig, ax = figure()
        for o, c in ((1, "C0"), (2, "C3")):
            ax.plot(hs[o], errh[o], "o-", color=c, label=f"P{o}: order {orders[o]:.1f}")
        common.convergence_axes(ax, "Mean element edge h (µm)", "Largest error / surface temperature rise")
        ax.legend(fontsize=8)
        r.figure(fig, "semi_infinite_mesh.png", "Mesh convergence at t_end (Crank-Nicolson, 2000 steps).")

        r.check("profile error (CN, 400 steps)", profile_err, 1e-4)
        r.check("backward Euler order", be, 0.95, "min", ".2f")
        r.check(f"Crank-Nicolson error ({steps[-1]} steps)", errs[0.5][-1], 1e-4)
        r.check("P2 mesh order (largest nodal error)", orders[2], 1.8, "min", ".2f")
    r.seconds = tm.seconds
    return r
