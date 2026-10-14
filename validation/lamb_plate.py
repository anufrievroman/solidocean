"""Guided waves in a free isotropic plate: the exact Rayleigh-Lamb and shear-horizontal dispersion."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import common
from common import Report, Timer, figure, order, quiet

TITLE = "Lamb waves in a free plate"
ABOUT = r"""
A free isotropic plate (thickness $t$, $\nu = 0.3$), periodic in-plane, with the Bloch wave vector $k$ along
$x$: all guided modes of a plate. Reference: the exact Rayleigh-Lamb equations for the symmetric and
antisymmetric Lamb modes,

$$\frac{\tan(q t/2)}{\tan(p t/2)} = -\left[\frac{4k^2pq}{(q^2-k^2)^2}\right]^{\pm 1},\qquad
p^2 = \frac{\omega^2}{c_L^2} - k^2,\quad q^2 = \frac{\omega^2}{c_T^2} - k^2,$$

and the shear-horizontal modes $\omega^2 = c_T^2\,(k^2 + (n\pi/t)^2)$, solved to machine precision. The cell
($t/4 \times t/4$ in-plane) is small enough that no folded band falls below the compared range
($\Omega = \omega t / c_T < 12$, $kt \le 6$): every computed eigenfrequency must be one of the exact ones.
Quadratic elements; the convergence study refines the mesh at $kt = 3$.
"""

NU = 0.3
OMEGA_MAX = 12.0             # compared range of omega t / c_T
KT_MAX = 6.0


def _c(x2, h):
    """cos(x h) for x^2 of any sign."""
    return np.cos(np.sqrt(x2) * h) if x2 >= 0 else np.cosh(np.sqrt(-x2) * h)


def _s_over(x2, h):
    """sin(x h) / x."""
    if x2 > 0:
        x = np.sqrt(x2)
        return np.sin(x * h) / x
    if x2 < 0:
        x = np.sqrt(-x2)
        return np.sinh(x * h) / x
    return h


def _x_s(x2, h):
    """x sin(x h)."""
    if x2 >= 0:
        x = np.sqrt(x2)
        return x * np.sin(x * h)
    x = np.sqrt(-x2)
    return -x * np.sinh(x * h)


def lamb_frequencies(kt: float, omega_max: float = OMEGA_MAX, nu: float = NU) -> np.ndarray:
    """All guided-mode frequencies omega t / c_T below omega_max (t = 1, c_T = 1): Lamb (symmetric and
    antisymmetric) and shear-horizontal, ascending."""
    cl2 = 2 * (1 - nu) / (1 - 2 * nu)          # (c_L / c_T)^2
    k, h = kt, 0.5

    def pq(w):
        return w * w / cl2 - k * k, w * w - k * k

    def sym(w):        # [(q^2-k^2)^2 cos(ph) sin(qh) + 4k^2 pq sin(ph) cos(qh)] / q
        p2, q2 = pq(w)
        return (q2 - k * k) ** 2 * _c(p2, h) * _s_over(q2, h) + 4 * k * k * _x_s(p2, h) * _c(q2, h)

    def anti(w):       # [(q^2-k^2)^2 sin(ph) cos(qh) + 4k^2 pq cos(ph) sin(qh)] / p
        p2, q2 = pq(w)
        return (q2 - k * k) ** 2 * _s_over(p2, h) * _c(q2, h) + 4 * k * k * _c(p2, h) * _x_s(q2, h)

    roots = []
    grid = np.linspace(1e-6, omega_max, 6000)
    for fn in (sym, anti):
        vals = np.array([fn(w) for w in grid])
        for i in np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0):
            roots.append(brentq(fn, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-14))
    n = 0
    while True:
        w = np.sqrt(k * k + (n * np.pi) ** 2)
        if w >= omega_max:
            break
        if w > 1e-9:
            roots.append(w)
        n += 1
    return np.sort(roots)


def project(h_frac: float, order_: int = 2, k_points: int = 0):
    from solidocean.lattice import Lattice
    from solidocean.physics.solid.model import KPath, SolidPhysics, SolidProblem
    from solidocean.project import Feature, Material, MeshSettings, Project
    a_frac = 0.25
    problem = SolidProblem(num_eigs=28, k=["kx", "0", "0"])
    if k_points:
        reduced = KT_MAX * a_frac / (2 * np.pi)            # k a / (2 pi) at kt = KT_MAX
        problem.kpath = KPath({"G": [0, 0, 0], "K": [reduced, 0, 0]}, ["G", "K"], k_points)
    return Project(
        name="Lamb plate", physics_kind="solid",
        description="Validation: guided waves of a free isotropic plate (Rayleigh-Lamb)",
        variables={"t": "1[um]", "a": f"{a_frac}*t", "kx": "3/t"},
        materials=[Material("M", {"model": "isotropic", "rho": "2700[kg/m^3]", "E": "70[GPa]", "nu": str(NU)})],
        objects=[Feature("block", "Plate", material="M", params={"center": ["0", "0", "0"], "size": ["a", "a", "t"]})],
        mesh=MeshSettings(max_size=f"t*{h_frac:g}", order=order_),
        lattice=Lattice(periodic={"x": "a", "y": "a"}),
        solid=SolidPhysics(problem=problem))


def c_t():
    E, rho = 70e9, 2700.0
    return np.sqrt(E / (2 * (1 + NU) * rho))


def compare(f_hz, kt):
    """Relative errors of the computed frequencies below the compared range against the exact ones."""
    omega = 2 * np.pi * np.asarray(f_hz) * 1e-6 / c_t()          # t = 1 um
    exact = lamb_frequencies(kt)
    if kt < 1e-9:                     # k = 0: the three rigid translations of the periodic plate
        omega = np.sort(omega)[3:]
    assert omega.max() > OMEGA_MAX, "too few eigenfrequencies computed"
    got = np.sort(omega)[:len(exact)]
    return exact, got, np.abs(got - exact) / exact


def run(quick: bool = False) -> Report:
    from solidocean.physics.solid.solver import Solver
    r = Report("lamb_plate", TITLE, ABOUT)
    with Timer() as tm:
        # dispersion along the path
        p = project(1 / 16, k_points=12 if quick else 30)
        common.save_model(p, "lamb_plate")
        res = Solver(p, log=quiet).run()
        kt = res.k[0, :, 0] * 1e-6
        worst = 0.0
        fig, ax = figure(6.4, 4.6)
        kk = np.linspace(0, KT_MAX, 241)
        curves = [lamb_frequencies(x) for x in kk]
        n_max = max(len(c) for c in curves)
        for b in range(n_max):
            ys = np.array([c[b] if b < len(c) else np.nan for c in curves])
            ax.plot(kk, ys, ".", ms=1.2, color=common.REF_COLOR, label="exact" if b == 0 else None)
        for j, x in enumerate(kt):
            exact, got, err = compare(res.freqs[0, j], x)
            worst = max(worst, err.max())
            ax.plot(np.full(len(got), x), got, "o", mfc="none", ms=4, color=common.FEM_COLOR,
                    label="SolidOcean" if j == 0 else None)
        ax.set_xlim(0, KT_MAX)
        ax.set_ylim(0, OMEGA_MAX)
        ax.set_xlabel("Wavenumber  k t")
        ax.set_ylabel("Frequency  ω t / c_T")
        ax.legend(loc="lower right")
        r.figure(fig, "lamb_dispersion.png", "Guided modes of a free plate: exact Rayleigh-Lamb and SH branches "
                                             "(dots) and the computed eigenfrequencies (circles), mesh h = t/16.")

        # mesh convergence at kt = 3
        rows, hs, errs = [], {1: [], 2: []}, {1: [], 2: []}
        fracs = [1 / 12, 1 / 16, 1 / 24] if quick else [1 / 12, 1 / 16, 1 / 24, 1 / 32]
        for o in (1, 2):
            for hf in fracs:
                q = project(hf, o)
                s = Solver(q, log=quiet)
                ev = q.evaluator()
                f = s.solve_point(ev, s.k_of(ev))
                exact, got, err = compare(f, 3.0)
                errs[o].append(err.max())
                h = common.mean_edge(s._model.mesh) / 1e-6
                hs[o].append(h)
                rows.append((f"P{o}", f"{h:.4f}", s._model.constraints.n_free, len(exact),
                             f"{np.median(err):.2e}", f"{err.max():.2e}"))
        p1, p2 = order(hs[1], errs[1]), order(hs[2], errs[2])
        r.table("Mesh convergence at kt = 3: all modes with ω t / c_T < 12.",
                ["elements", "h / t", "DOFs", "modes", "median error", "max error"], rows)
        fig, ax = figure()
        for o, c in ((1, "C0"), (2, "C3")):
            ax.plot(hs[o], errs[o], "o-", color=c, label=f"P{o}: order {order(hs[o], errs[o]):.1f}")
        common.slope_guide(ax, hs[1], np.array(errs[1]), 2)
        common.slope_guide(ax, hs[2], np.array(errs[2]), 4)
        common.convergence_axes(ax, "Mean element edge h / t", "Largest relative frequency error")
        ax.legend()
        r.figure(fig, "lamb_convergence.png", "Mesh convergence of the eigenfrequencies at kt = 3 (largest error "
                                              "of all modes below ω t / c_T = 12); expected orders 2h (P1: 2, P2: 4).")
        r.check("largest frequency error along the path (h = t/16)", worst, 1e-3)
        r.check("P2 convergence order", p2, 3.5, "min", ".2f")
        r.check("P1 convergence order", p1, 1.8, "min", ".2f")
    r.seconds = tm.seconds
    return r
