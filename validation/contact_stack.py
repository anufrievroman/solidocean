"""Thermal boundary resistance in a layer stack: steady series law and the transient two-layer solution."""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq

import common
from common import Report, Timer, figure, quiet

TITLE = "Thermal boundary resistance in a layer stack"
ABOUT = r"""
Layers along $x$ (insulated sides) joined by interfaces with a thermal boundary resistance: the temperature
jumps by $q R$ across them.

*Steady state:* three layers between $T_1$ and $T_0$, joined by a contact resistance $R_1$ and by a thin layer
(thickness $d$, conductivity $k_f$, not meshed: $R_2 = d / k_f$). Reference: the series law
$q = (T_1 - T_0) / (\sum_i L_i/k_i + R_1 + R_2)$, linear in each layer, which the elements represent exactly.

*Transient:* two layers (different conductivities and diffusivities) joined by a resistance $R$, initially at
$T_0$; at $t = 0$ the face $x = 0$ is raised to $T_1$. Reference: the eigenfunction series
$T = T_\mathrm{ss} + \sum_n c_n X_n(x)\,e^{-\lambda_n t}$ with $X_n = \sin(\beta_1 x)$ in the first layer and
$B_n \sin(\beta_2 (L - x))$ in the second ($\beta_i = \sqrt{\lambda/\alpha_i}$); the eigenvalues follow from flux
continuity and the jump condition $-k_1 X'(L_1) = (X_1 - X_2)/R$:
$$k_2\beta_2\cos\beta_2L_2\,\sin\beta_1L_1 + k_1\beta_1\cos\beta_1L_1\,\sin\beta_2L_2 +
R\,k_1\beta_1k_2\beta_2\cos\beta_1L_1\cos\beta_2L_2 = 0,$$
and the coefficients from the orthogonality of the $X_n$ with weight $\rho c_p$ (300 terms). The comparison
starts at 5 ns: right after the step, the profile next to the hot face is steeper than the elements and the
Crank-Nicolson start-up transient has not yet decayed (see the semi-infinite solid above).
"""

T1, T0 = 400.0, 300.0


def _material(name, k, rhoc=2e6):
    from solidocean.project import Material
    return Material(name, {"model": "isotropic", "rho": "1000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                           "k": f"{k:g}[W/(m*K)]", "cp": f"{rhoc / 1000:g}[J/(kg*K)]"})


def _block(name, x0, x1):
    from solidocean.project import Feature
    return Feature("block", name, material=name, params={"center": [f"{(x0 + x1) / 2:g}[um]", "0", "0"],
                                                          "size": [f"{x1 - x0:g}[um]", "w", "w"]})


# ---- steady three-layer stack -------------------------------------------------------------
LAYERS = [("A", 0.0, 1.0, 100.0), ("B", 1.0, 1.5, 20.0), ("C", 1.5, 2.5, 50.0)]      # name, x0, x1 (um), k
R1, D_F, K_F = 5e-9, 5e-9, 1.0


def steady_project(order_: int):
    from solidocean.physics.heat.model import HeatCondition, HeatInterface, HeatPhysics
    from solidocean.project import MeshSettings, Project
    return Project(
        name="Layer stack", physics_kind="heat",
        description="Validation: steady heat flow through layers with a boundary resistance and a thin layer",
        variables={"w": "0.25[um]"},
        materials=[_material(n, k) for n, _, _, k in LAYERS] + [_material("Film", K_F)],
        objects=[_block(n, x0, x1) for n, x0, x1, _ in LAYERS],
        mesh=MeshSettings(max_size="w/2", order=order_),
        heat=HeatPhysics(initial="300[K]", conditions=[
            HeatCondition("temperature", ["xmin"], f"{T1:g}[K]", name="Hot"),
            HeatCondition("temperature", ["xmax"], f"{T0:g}[K]", name="Cold")],
            interfaces=[HeatInterface("resistance", ["A", "B"], f"{R1:g}[m^2*K/W]", name="Kapitza"),
                        HeatInterface("layer", ["B", "C"], material="Film", thickness=f"{D_F * 1e9:g}[nm]",
                                      name="Film")]))


def steady_exact():
    """Flux and the temperature at both ends of each layer."""
    rs = [(x1 - x0) * 1e-6 / k for _, x0, x1, k in LAYERS]
    q = (T1 - T0) / (sum(rs) + R1 + D_F / K_F)
    ends, T = [], T1
    for i, (_, x0, x1, k) in enumerate(LAYERS):
        ends.append((T, T - q * rs[i]))
        T -= q * (rs[i] + (R1 if i == 0 else D_F / K_F if i == 1 else 0))
    return q, ends


def layer_nodes(mesh, name):
    nodes = np.unique(mesh.cells[mesh.region(name)])
    return nodes, mesh.points[nodes, 0]


# ---- transient two-layer problem -----------------------------------------------------------
L1_, L2_ = 1e-6, 1e-6
K1, RC1, K2, RC2, R_T = 100.0, 2e6, 20.0, 1.5e6, 1e-8
A1, A2 = K1 / RC1, K2 / RC2
T_END = 60e-9


class TwoLayer:
    """Exact transient solution of the two-layer problem (see ABOUT)."""

    def __init__(self, n_terms=300):
        L = L1_ + L2_

        def F(s):           # s = sqrt(lambda)
            b1, b2 = s / np.sqrt(A1), s / np.sqrt(A2)
            c1, s1, c2, s2 = np.cos(b1 * L1_), np.sin(b1 * L1_), np.cos(b2 * L2_), np.sin(b2 * L2_)
            return k2b2(b2) * c2 * s1 + K1 * b1 * c1 * s2 + R_T * K1 * b1 * k2b2(b2) * c1 * c2

        def k2b2(b2):
            return K2 * b2
        scale = np.sqrt(A1) / L1_
        grid = np.linspace(1e-3, 1.0, 200001) * scale * (n_terms + 20) * 2.2
        vals = F(grid)
        idx = np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)
        roots = np.array([brentq(F, grid[i], grid[i + 1], xtol=1e-14 * scale) for i in idx])[:n_terms]
        self.s = roots
        q = (T1 - T0) / (L1_ / K1 + R_T + L2_ / K2)
        self.q, self.L = q, L
        # quadrature on both layers
        g, w = np.polynomial.legendre.leggauss(400)
        self.x1, self.w1 = (g + 1) / 2 * L1_, w / 2 * L1_
        self.x2, self.w2 = L1_ + (g + 1) / 2 * L2_, w / 2 * L2_
        self.modes = []
        for s in roots:
            b1, b2 = s / np.sqrt(A1), s / np.sqrt(A2)
            c1, s1, c2, s2 = np.cos(b1 * L1_), np.sin(b1 * L1_), np.cos(b2 * L2_), np.sin(b2 * L2_)
            M = np.array([[K1 * b1 * c1, K2 * b2 * c2], [-R_T * K1 * b1 * c1 - s1, s2]])
            a, b = null_space(M, rcond=1e-9)[:, 0]
            X1 = lambda x, a=a, b1=b1: a * np.sin(b1 * x)                      # noqa: E731
            X2 = lambda x, b=b, b2=b2: b * np.sin(b2 * (L - x))                # noqa: E731
            num = RC1 * self.w1 @ ((T0 - self.ss(self.x1, 1)) * X1(self.x1)) + \
                RC2 * self.w2 @ ((T0 - self.ss(self.x2, 2)) * X2(self.x2))
            den = RC1 * self.w1 @ X1(self.x1) ** 2 + RC2 * self.w2 @ X2(self.x2) ** 2
            self.modes.append((s * s, num / den, X1, X2))

    def ss(self, x, side):
        return T1 - self.q * x / K1 if side == 1 else T0 + self.q * (self.L - x) / K2

    def __call__(self, x, t, side):
        """Temperature at positions x (in layer `side`, 1 or 2) and time t > 0."""
        out = self.ss(x, side)
        for lam, c, X1, X2 in self.modes:
            out = out + c * (X1(x) if side == 1 else X2(x)) * np.exp(-lam * t)
        return out


def transient_project(steps=1200):
    from solidocean.physics.heat.model import HeatCondition, HeatInterface, HeatPhysics, HeatProblem
    from solidocean.project import MeshSettings, Project
    return Project(
        name="Two layers", physics_kind="heat",
        description="Validation: transient heat flow through two layers with a boundary resistance",
        variables={"w": "0.2[um]"},
        materials=[_material("A", K1, RC1), _material("B", K2, RC2)],
        objects=[_block("A", 0.0, L1_ * 1e6), _block("B", L1_ * 1e6, (L1_ + L2_) * 1e6)],
        mesh=MeshSettings(max_size="w/2", order=2),
        heat=HeatPhysics(initial=f"{T0:g}[K]", conditions=[
            HeatCondition("temperature", ["xmin"], f"{T1:g}[K]", name="Hot"),
            HeatCondition("temperature", ["xmax"], f"{T0:g}[K]", name="Cold")],
            interfaces=[HeatInterface("resistance", ["A", "B"], f"{R_T:g}[m^2*K/W]", name="Kapitza")],
            problem=HeatProblem(type="transient", t_end=f"{T_END:g}[s]", steps=steps, theta=0.5, save_every=10)))


def run(quick: bool = False) -> Report:
    from solidocean.physics.heat.solver import HeatSolver
    r = Report("contact_stack", TITLE, ABOUT)
    with Timer() as tm:
        # steady stack
        q, ends = steady_exact()
        rows, steady_err, prof = [], 0.0, None
        for o in (1, 2):
            p = steady_project(o)
            if o == 2:
                common.save_model(p, "layer_stack")
            res = HeatSolver(p, log=quiet).run(["mean(qx)"])
            mesh, T = res.meshes[0], res.T[0][0]
            err = 0.0
            for d, (n, x0, x1, k) in enumerate(LAYERS):
                nodes, x = layer_nodes(mesh, n)
                Ta, Tb = ends[d]
                ex = Ta + (Tb - Ta) * (x - x0 * 1e-6) / ((x1 - x0) * 1e-6)
                err = max(err, np.abs(T[nodes] - ex).max() / (T1 - T0))
            qf = res.quantities["mean(qx)"][0, 0]
            rows.append((f"P{o}", f"{qf:.6e}", f"{abs(qf / q - 1):.1e}", f"{err:.1e}"))
            steady_err = max(steady_err, err, abs(qf / q - 1))
            if o == 2:
                prof = (mesh, T)
        r.table(f"Steady stack: exact flux {q:.6e} W/m²; largest nodal temperature error relative to T₁ − T₀.",
                ["elements", "flux (W/m²)", "flux error", "temperature error"], rows)
        mesh, T = prof
        fig, ax = figure(6.2, 4.0)
        for d, (n, x0, x1, k) in enumerate(LAYERS):
            Ta, Tb = ends[d]
            ax.plot([x0, x1], [Ta, Tb], "-", color=common.REF_COLOR, lw=1.5, label="exact" if d == 0 else None)
            nodes, x = layer_nodes(mesh, n)
            ax.plot(x * 1e6, T[nodes], "o", mfc="none", ms=4, color=common.COLORS[d], label=f"layer {n} (k = {k:g})")
        for xi, lab in ((1.0, "R₁"), (1.5, "thin layer")):
            ax.axvline(xi, color="0.85", lw=1, zorder=0)
            ax.annotate(lab, (xi, T1 - 4), ha="center", fontsize=8, color="0.4")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("Temperature (K)")
        ax.legend(fontsize=8)
        r.figure(fig, "stack_steady.png", "Steady temperature through the stack: jumps at the boundary resistance "
                                          "and at the thin layer, against the series law.")

        # transient two-layer problem
        ref = TwoLayer()
        p = transient_project(600 if quick else 1200)
        common.save_model(p, "two_layers")
        res = HeatSolver(p, log=quiet).run()
        mesh = res.meshes[0]
        (n_a, xa), (n_b, xb) = layer_nodes(mesh, "A"), layer_nodes(mesh, "B")
        times = res.times
        use = times >= 5e-9
        err = 0.0
        for j in np.flatnonzero(use):
            t = times[j]
            ea = np.abs(res.T[0][j][n_a] - ref(xa, t, 1)).max()
            eb = np.abs(res.T[0][j][n_b] - ref(xb, t, 2)).max()
            err = max(err, ea, eb)
        err /= T1 - T0
        ia, ib = n_a[np.isclose(xa, L1_)], n_b[np.isclose(xb, L1_)]
        fig, (ax, bx) = __import__("matplotlib.pyplot").pyplot.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
        tt = np.linspace(0.2e-9, T_END, 300)
        ax.plot(tt * 1e9, ref(np.array([L1_]), tt[:, None], 1)[:, 0], "-", color="C3", lw=1.2)
        ax.plot(tt * 1e9, ref(np.array([L1_]), tt[:, None], 2)[:, 0], "-", color="C0", lw=1.2)
        ax.plot(times[1::3] * 1e9, res.T[0][1::3][:, ia].mean(axis=1), "o", mfc="none", ms=4, color="C3",
                label="interface, side A")
        ax.plot(times[1::3] * 1e9, res.T[0][1::3][:, ib].mean(axis=1), "s", mfc="none", ms=4, color="C0",
                label="interface, side B")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Temperature (K)")
        ax.legend(title="lines: exact", fontsize=8)
        xs1, xs2 = np.linspace(0, L1_, 100), np.linspace(L1_, L1_ + L2_, 100)
        for n, tf in enumerate((5e-9, 15e-9, 60e-9)):
            j = int(np.argmin(np.abs(times - tf)))
            c = common.COLORS[n]
            bx.plot(xs1 * 1e6, ref(xs1, times[j], 1), "-", color=c, lw=1.2)
            bx.plot(xs2 * 1e6, ref(xs2, times[j], 2), "-", color=c, lw=1.2)
            sel = np.r_[n_a[np.argsort(xa)][::4], n_b[np.argsort(xb)][::4]]
            bx.plot(mesh.points[sel, 0] * 1e6, res.T[0][j][sel], "o", mfc="none", ms=3.5, color=c,
                    label=f"t = {times[j] * 1e9:.3g} ns")
        bx.set_xlabel("x (µm)")
        bx.set_ylabel("Temperature (K)")
        bx.legend(title="lines: exact", fontsize=8)
        r.figure(fig, "stack_transient.png", "Two layers with a boundary resistance after the left face is raised "
                                             "to T₁: temperature on both sides of the interface against time, and "
                                             "profiles at three times (Crank-Nicolson, quadratic elements).")
        r.table("Transient two-layer problem.", ["quantity", "value"], [
            ("eigenvalues used", len(ref.modes)),
            ("time steps (Crank-Nicolson)", p.heat.problem.steps),
            ("largest nodal error for t ≥ 5 ns, relative to T₁ − T₀", f"{err:.1e}")])
        r.check("steady stack: flux and temperatures (P1 and P2)", steady_err, 1e-9)
        r.check("transient: largest nodal error for t ≥ 5 ns", err, 1e-3)
    r.seconds = tm.seconds
    return r
