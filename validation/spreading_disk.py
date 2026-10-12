"""Spreading resistance of a disk heater on a half-space (uniform flux)."""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

import common
from common import Report, Timer, figure, order, quiet

TITLE = "Disk heater on a half-space: spreading resistance"
ABOUT = r"""
A uniform heat flux $q = P / \pi a^2$ into a disk of radius $a$ on the surface of a half-space (conductivity
$k$), the rest of the surface insulated. Reference (Carslaw & Jaeger): the surface temperature
$$\Delta T(r) = \frac{2qa}{\pi k}\,E\!\left(\tfrac{r}{a}\right)\ (r \le a),\qquad
\Delta T(r) = \frac{2qa}{\pi k}\,\frac{r}{a}\left[E\!\left(\tfrac{a}{r}\right) - \left(1 - \tfrac{a^2}{r^2}\right)
K\!\left(\tfrac{a}{r}\right)\right]\ (r > a),$$
so the centre is at $P/\pi k a$ and the disk's mean temperature rise is $8P/3\pi^2 k a$ (spreading resistance
$8/3\pi^2 k a$); $E$, $K$ are complete elliptic integrals. The model is a quarter of a hemisphere of radius
$R = 5a$ (the symmetry planes are insulated faces) inside a shell of a very conductive material held at $T_0$:
an isothermal sink sphere, which lowers the half-space solution by $P/2\pi k R$ (the next correction is of
order $(a/R)^5$). The disk is a thin cylinder of the same material, so the mesh follows its edge. The step of
the flux at the disk's edge makes the solution singular there (like $\rho\ln\rho$ at distance $\rho$ from the
edge), which slows the convergence of the disk's mean temperature to about order 1.5; the centre converges
much faster.
"""

K_M, P, R_FRAC = 100.0, 1e-3, 5.0
A_D = 1e-6
Q = P / (np.pi * A_D ** 2)
SINK = P / (2 * np.pi * K_M * R_FRAC * A_D)


def surface_rise(r):
    """Half-space surface temperature rise at radius r (K)."""
    r = np.asarray(r, float)
    c = 2 * Q * A_D / (np.pi * K_M)
    inside = c * ellipe(np.minimum(r / A_D, 1) ** 2)
    s = A_D / np.maximum(r, A_D)
    outside = c * (1 / s) * (ellipe(s * s) - (1 - s * s) * ellipk(np.minimum(s * s, 1 - 1e-16)))
    return np.where(r <= A_D, inside, outside)


MEAN = 8 * P / (3 * np.pi ** 2 * K_M * A_D)


def project(h_frac: float):
    from solidocean.physics.heat.model import HeatCondition, HeatPhysics
    from solidocean.project import Feature, Material, MeshSettings, Project

    def mat(name, k):
        return Material(name, {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                               "k": k, "cp": "500[J/(kg*K)]"})
    return Project(
        name="Disk heater", physics_kind="heat",
        description="Validation: spreading resistance of a uniform-flux disk heater on a half-space",
        variables={"a": "1[um]", "R": f"{R_FRAC:g}*a", "P": f"{P * 1e3:g}[mW]", "T0": "300[K]"},
        materials=[mat("M", f"{K_M:g}[W/(m*K)]"), mat("Sink", f"{K_M * 1e7:g}[W/(m*K)]")],
        geometry=[
            Feature("sphere", "Shell", material="Sink", params={"center": ["0", "0", "0"], "radius": "1.2*R"}),
            Feature("sphere", "Body", material="M", params={"center": ["0", "0", "0"], "radius": "R"}),
            Feature("cylinder", "Heater", material="M",
                    params={"center": ["0", "0", "-a/4"], "axis": ["0", "0", "1"], "radius": "a", "height": "a/4"}),
            Feature("block", "Cut x", op="subtract", params={"center": ["-R", "0", "0"], "size": ["2*R", "4*R", "4*R"]}),
            Feature("block", "Cut y", op="subtract", params={"center": ["0", "-R", "0"], "size": ["4*R", "2*R", "4*R"]}),
            Feature("block", "Cut z", op="subtract", params={"center": ["0", "0", "R"], "size": ["4*R", "4*R", "2*R"]})],
        mesh=MeshSettings(max_size=f"a*{h_frac:g}", order=2),
        heat=HeatPhysics(initial="T0", conditions=[
            HeatCondition("temperature", ["all"], "T0", name="Sink", on="Shell"),
            HeatCondition("flux", ["zmax"], "P / (pi * a^2)", name="Heater", on="Heater")]))


def disk_mean(mesh, T) -> float:
    """Mean temperature over the heated disk (surface integral with the elements' shape functions)."""
    from solidocean.fem import surface
    tris = surface.boundary_faces(mesh.cells, mesh.order)
    c = mesh.points[tris[:, :3]]
    on = np.all(np.abs(c[..., 2]) < 1e-12, axis=1) & np.all(np.hypot(c[..., 0], c[..., 1]) <= A_D * (1 + 1e-9), axis=1)
    _, f = surface.surface_terms(mesh.points, tris[on], mesh.order, load=1.0)
    return float(f @ T / f.sum())


def run(quick: bool = False) -> Report:
    from solidocean.physics.heat.solver import HeatSolver
    r = Report("spreading_disk", TITLE, ABOUT)
    with Timer() as tm:
        fracs = [1 / 2, 1 / 3, 1 / 4] if quick else [1 / 2, 1 / 3, 1 / 4, 1 / 6]
        rows, hs, e_mean, e_ctr, last = [], [], [], [], None
        for hf in fracs:
            p = project(hf)
            if hf == 1 / 3:
                common.save_model(p, "spreading_disk")
            s = HeatSolver(p, log=quiet)
            res = s.run()
            mesh, T = res.meshes[0], res.T[0][0] - 300
            mean = disk_mean(mesh, T)
            pts = mesh.points
            ctr = T[np.argmin(np.linalg.norm(pts, axis=1))]
            em, ec = abs(mean / (MEAN - SINK) - 1), abs(ctr / (surface_rise(0) - SINK) - 1)
            h = common.mean_edge(mesh) / A_D
            hs.append(h)
            e_mean.append(em)
            e_ctr.append(ec)
            rows.append((f"a/{1 / hf:.0f}", f"{h:.3f}", mesh.n_nodes, f"{mean:.5f}", f"{em:.1e}", f"{ctr:.5f}",
                         f"{ec:.1e}"))
            last = (mesh, T)
        r.table(f"Temperature rise of the disk (P = {P * 1e3:g} mW, a = 1 µm, k = {K_M:g} W/(m·K)); exact: mean "
                f"{MEAN - SINK:.5f} K, centre {surface_rise(0) - SINK:.5f} K (half-space minus the sink's "
                f"{SINK:.5f} K). h is the mean element edge / a.",
                ["max size", "h", "nodes", "mean (K)", "error", "centre (K)", "error"], rows)

        mesh, T = last
        pts = mesh.points
        edge = (np.abs(pts[:, 1]) < 1e-12) & (np.abs(pts[:, 2]) < 1e-12) & (pts[:, 0] <= 3 * A_D)
        xs = np.linspace(0, 3 * A_D, 400)
        fig, ax = figure()
        ax.plot(xs / A_D, surface_rise(xs) - SINK, "-", color=common.REF_COLOR, lw=1.5, label="exact")
        idx = np.flatnonzero(edge)
        ax.plot(pts[idx, 0] / A_D, T[idx], "o", mfc="none", ms=4, color=common.FEM_COLOR, label="SolidOcean")
        ax.axvline(1, color="0.8", lw=1, zorder=0)
        ax.set_xlabel("Distance from the centre r / a")
        ax.set_ylabel("Surface temperature rise (K)")
        ax.legend()
        r.figure(fig, "spreading_profile.png", f"Surface temperature along a radius (mesh a/{1 / fracs[-1]:.0f}) "
                                               "against the exact half-space solution (minus the sink's offset).")
        fig, ax = figure()
        ax.plot(hs, e_mean, "o-", color=common.FEM_COLOR, label=f"disk mean: order {order(hs, e_mean):.1f}")
        ax.plot(hs, e_ctr, "s--", color="C0", mfc="none", label=f"centre: order {order(hs, e_ctr):.1f}")
        common.convergence_axes(ax, "Mean element edge h / a", "Relative error")
        ax.legend(fontsize=8)
        r.figure(fig, "spreading_convergence.png", "Mesh convergence of the disk's mean and centre temperature "
                                                   "(quadratic elements).")
        r.check("mean disk temperature (finest mesh)", e_mean[-1], 1e-3)
        r.check("centre temperature (finest mesh)", e_ctr[-1], 1e-3)
        r.check("convergence order of the mean", order(hs, e_mean), 1.5, "min", ".2f")
    r.seconds = tm.seconds
    return r
