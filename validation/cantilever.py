"""Bending modes of a cantilever: 3D elasticity against Euler-Bernoulli and Timoshenko beam theory."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

import common
from common import Report, Timer, figure, quiet

TITLE = "Cantilever bending modes"
ABOUT = r"""
A beam of square cross-section $b \times b$ and length $L$, clamped at one end ($\nu = 0.3$). Beam theories are
exact only for slender beams, so the beam is computed for slenderness $L/b$ from 4 to 40 and compared with
**Euler-Bernoulli** theory, $\omega_n = (\beta_n L)^2 \sqrt{EI/(\rho A L^4)}$ with $1 + \cos\beta L \cosh\beta L = 0$
($\beta_n L$ = 1.8751, 4.6941, 7.8548), and **Timoshenko** theory (shear deformation and rotary inertia,
Cowper's shear coefficient $\kappa = 10(1+\nu)/(12+11\nu)$), whose clamped-free frequencies are solved exactly
with the transfer matrix of its ODEs. The 3D solution must tend to both as $b/L \to 0$, and follow Timoshenko's
correction $\propto (b/L)^2$ for stubby beams. The bending modes come in degenerate pairs (bending in $y$ and
$z$); they are told apart from torsional and axial modes by their mean displacement.

A fully clamped face also stops the Poisson contraction at the root, which beam theory ignores: this
Saint-Venant end effect stiffens the 3D beam by a fraction $\propto b/L$. A root that may contract (symmetry
plane $u_x = 0$, with $u_y = u_z = 0$ held only on a central patch) is much closer to the beam theories' clamp.
"""

NU, E, RHO = 0.3, 100e9, 2000.0
SLENDERNESS = [4, 6, 10, 20, 40]


def eb_roots(n=3):
    f = lambda x: 1 + np.cos(x) * np.cosh(x)   # noqa: E731
    return np.array([brentq(f, (i + 0.5) * np.pi - 1.2, (i + 0.5) * np.pi + 1.2) for i in range(n)])


def euler_bernoulli(L, b, n=3):
    """Angular frequencies (rad/s)."""
    return eb_roots(n) ** 2 * np.sqrt(E * b ** 4 / 12 / (RHO * b * b)) / L ** 2


def timoshenko(L, b, n=3):
    """Clamped-free Timoshenko beam: roots of the 2x2 determinant of the transfer matrix exp(A) over the
    dimensionless length, acting on the state [w/L, psi, M L/EI, V L^2/EI] (deflection, rotation, bending
    moment, shear force) with w = psi = 0 at the clamp and M = V = 0 at the free end."""
    G = E / (2 * (1 + NU))
    kappa = 10 * (1 + NU) / (12 + 11 * NU)
    A_, I_ = b * b, b ** 4 / 12
    s2 = E * I_ / (kappa * G * A_ * L * L)          # shear flexibility
    r2 = I_ / (A_ * L * L)                          # rotary inertia

    def det(w):
        lam = RHO * A_ * w * w * L ** 4 / (E * I_)  # (beta L)^4
        M = np.array([[0, 1, 0, s2],
                      [0, 0, 1, 0],
                      [0, -lam * r2, 0, -1],
                      [-lam, 0, 0, 0]])
        T = expm(M)
        return np.linalg.det(T[2:, 2:])

    eb = euler_bernoulli(L, b, n + 1)
    grid = np.linspace(0.2 * eb[0], eb[n - 1] * 1.05, 4000)
    vals = np.array([det(w) for w in grid])
    roots = [brentq(det, grid[i], grid[i + 1], xtol=1e-13 * eb[0])
             for i in np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)]
    return np.array(roots[:n])


def project(slenderness: float, h_frac: float = 1 / 3, free_root: bool = False):
    from solidocean.physics.solid.model import SolidCondition, SolidPhysics, SolidProblem
    from solidocean.project import Feature, Material, MeshSettings, Project
    return Project(
        name="Cantilever", physics_kind="solid", description="Validation: bending modes of a cantilever",
        variables={"b": "1[um]", "L": f"{slenderness:g}*b"},
        materials=[Material("M", {"model": "isotropic", "rho": f"{RHO:g}[kg/m^3]", "E": f"{E / 1e9:g}[GPa]",
                                   "nu": str(NU)})],
        objects=[Feature("block", "Beam", material="M", params={"center": ["L/2", "0", "0"], "size": ["L", "b", "b"]})],
        mesh=MeshSettings(max_size=f"b*{h_frac:g}", order=2),
        solid=SolidPhysics(conditions=[
            SolidCondition("displacement", ["xmin"], displacement=["", "0", "0"], name="Root centre",
                           box={"min": ["0", "-0.35*b", "-0.35*b"], "max": ["0", "0.35*b", "0.35*b"]})
            if free_root else SolidCondition("fixed", ["xmin"], name="Clamp")],
            symmetry=["xmin"] if free_root else [], problem=SolidProblem(num_eigs=14)))


def bending_modes(p, n=3):
    """Angular frequencies of the first n bending pairs (mean of each pair) and the mode classes."""
    from solidocean.physics.solid.solver import Solver
    s = Solver(p, log=quiet)
    f, u = s.solve_point(p.evaluator(), np.zeros(3), modes=True)
    kinds = []
    for b in range(len(f)):
        v = np.real(u[b])
        rms = np.sqrt(np.mean(np.sum(v * v, axis=1)))
        m = np.abs(v.mean(axis=0)) / rms
        kinds.append("torsion" if m.max() < 0.05 else ("axial" if m.argmax() == 0 else "bending"))
    bend = 2 * np.pi * f[np.array(kinds) == "bending"]
    assert len(bend) >= 2 * n, f"only {len(bend)} bending modes found"
    return bend[:2 * n].reshape(n, 2).mean(axis=1), 2 * np.pi * f, kinds, s._model.constraints.n_free


def run(quick: bool = False) -> Report:
    r = Report("cantilever", TITLE, ABOUT)
    with Timer() as tm:
        b = 1e-6
        slender = [4, 10, 40] if quick else SLENDERNESS
        ratios, free, rows = [], [], []
        for s_ in slender:
            p = project(s_)
            if s_ == 10:
                common.save_model(p, "cantilever")
            fem = bending_modes(p)[0]
            fem_free = bending_modes(project(s_, free_root=True))[0]
            L = s_ * b
            eb, tim = euler_bernoulli(L, b), timoshenko(L, b)
            ratios.append((fem / eb, tim / eb))
            free.append(fem_free / tim - 1)
            for n in range(3):
                rows.append((f"{s_:g}", n + 1, f"{fem[n] / eb[n]:.5f}", f"{tim[n] / eb[n]:.5f}",
                             f"{fem[n] / tim[n] - 1:+.2e}", f"{fem_free[n] / tim[n] - 1:+.2e}"))
        r.table("Bending frequencies relative to Euler-Bernoulli, and the difference of the 3D beam from Timoshenko "
                "with a fully clamped root and with a root free to contract (mesh b/3, quadratic elements).",
                ["L/b", "mode", "3D / EB", "Timoshenko / EB", "3D vs Timoshenko", "free root vs Timoshenko"], rows)

        fig, ax = figure(6.2, 4.4)
        sl = np.array(slender, float)
        fine = np.geomspace(sl.min(), sl.max(), 60)
        tim_curve = np.array([timoshenko(x * b, b) / euler_bernoulli(x * b, b) for x in fine])
        for n in range(3):
            ax.plot(fine, tim_curve[:, n], "-", color=common.COLORS[n], lw=1.2,
                    label=f"Timoshenko, mode {n + 1}")
            ax.plot(sl, [q[0][n] for q in ratios], "o", mfc="none", color=common.COLORS[n],
                    label=f"SolidOcean (3D), mode {n + 1}")
        ax.axhline(1, color=common.REF_COLOR, lw=1, ls="--", label="Euler-Bernoulli")
        ax.set_xscale("log")
        common.plain_ticks(ax, sl)
        ax.set_xlabel("Slenderness L / b")
        ax.set_ylabel("Frequency / Euler-Bernoulli frequency")
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        r.figure(fig, "cantilever_slenderness.png", "Bending frequencies of the 3D cantilever divided by Euler-"
                                                    "Bernoulli's, against slenderness, with Timoshenko's.")
        gap = np.array([np.abs(q[0] / q[1] - 1).max() for q in ratios])
        gap_free = np.array([np.abs(g).max() for g in free])
        slope = -common.order(sl, gap)
        fig, ax = figure()
        ax.plot(sl, gap, "o-", color=common.FEM_COLOR, label=f"clamped root (∝ (L/b)^{-slope:.2f})")
        ax.plot(sl, gap_free, "s-", color="C0", label="root free to contract")
        ax.plot(sl, gap[0] * sl[0] / sl, "--", color="0.6", lw=1, label="∝ b / L")
        common.convergence_axes(ax, "Slenderness L / b", "Largest |3D / Timoshenko − 1|, modes 1-3")
        common.plain_ticks(ax, sl)
        ax.legend()
        r.figure(fig, "cantilever_root.png", "Difference between the 3D beam and Timoshenko theory: it vanishes "
                                             "like b/L, and is much smaller when the root may contract.")

        # mesh convergence at L/b = 10 against the finest mesh
        fracs = [1 / 2, 1 / 3, 1 / 4, 1 / 6]
        res = [bending_modes(project(10, hf)) for hf in (fracs[:3] if quick else fracs)]
        ref = res[-1][0]
        conv = [(f"b/{1 / hf:.0f}", q[3], *[f"{abs(q[0][n] / ref[n] - 1):.1e}" for n in range(3)])
                for hf, q in zip(fracs, res[:-1])]
        r.table(f"Mesh convergence at L/b = 10: change of the bending frequencies against the mesh "
                f"b/{1 / fracs[len(res) - 1]:.0f}.", ["h", "DOFs", "mode 1", "mode 2", "mode 3"], conv)

        last = ratios[-1]
        r.check(f"L/b = {slender[-1]}: 3D vs Timoshenko, modes 1-3", np.abs(last[0] / last[1] - 1).max(), 2e-3)
        r.check("decay of the difference from Timoshenko, order in b/L", slope, 0.85, "min", ".2f")
        r.check(f"L/b = {slender[-1]}, root free to contract: 3D vs Timoshenko", gap_free[-1], 5e-4)
    r.seconds = tm.seconds
    return r
