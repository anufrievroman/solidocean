"""Vibrations of a free elastic sphere: Lamb's (1882) torsional and spheroidal modes."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn

import common
from common import Report, Timer, figure, order, quiet

TITLE = "Free vibrations of an elastic sphere"
ABOUT = r"""
A free isotropic sphere (radius $a$, $\nu = 0.25$), meshed with curved quadratic elements. Reference: Lamb's
exact solution. Torsional modes ${}_nT_l$ satisfy $x\,j_l'(x) = j_l(x)$ with $x = \omega a / c_T$; spheroidal
modes ${}_nS_l$ follow from the traction-free surface ($\sigma_{rr} = \sigma_{r\theta} = 0$) for the
displacement $\mathbf{u} = \nabla\varphi + \nabla\times\nabla\times(\mathbf{r}\,\psi)$ with
$\varphi = A\,j_l(\omega r/c_L)\,Y_{lm}$, $\psi = B\,j_l(\omega r / c_T)\,Y_{lm}$ (a 2×2 determinant, solved to
machine precision; the breathing mode is checked against Lamb's closed form
$\xi\cot\xi = 1 - \xi^2 c_L^2 / (4c_T^2)$). Each mode of angular order $l$ is $(2l+1)$-fold degenerate. All
computed eigenfrequencies up to $\omega a / c_T = 4.6$ (the six rigid-body modes removed) are compared with
the exact spectrum, multiplicities included.
"""

NU = 0.25
X_MAX = 4.6


def _jl(l, x, d=0):
    if d == 0:
        return spherical_jn(l, x)
    if d == 1:
        return spherical_jn(l, x, derivative=True)
    return -2 / x * spherical_jn(l, x, derivative=True) - (1 - l * (l + 1) / x ** 2) * spherical_jn(l, x)


def spheroidal_det(l: int, x: float, nu: float = NU) -> float:
    """Traction determinant of spheroidal modes at x = omega a / c_T (a = 1, c_T = 1, mu = 1)."""
    cl = np.sqrt(2 * (1 - nu) / (1 - 2 * nu))
    lam = cl ** 2 - 2
    h, k, L = x / cl, x, l * (l + 1)
    f, f1, f2 = (h ** d * _jl(l, h, d) for d in range(3))
    g, g1, g2 = (k ** d * _jl(l, k, d) for d in range(3))
    rr_a = -lam * h * h * f + 2 * f2
    if l == 0:
        return rr_a
    rr_b = 2 * L * (g1 - g)
    rt_a = 2 * f1 - 2 * f
    rt_b = g2 + (L - 2) * g
    return rr_a * rt_b - rr_b * rt_a


def torsional_eq(l: int, x: float) -> float:
    return x * _jl(l, x, 1) - _jl(l, x)


def _roots(fn, x_max):
    grid = np.linspace(0.05, x_max, 3000)
    vals = np.array([fn(x) for x in grid])
    return [brentq(fn, grid[i], grid[i + 1], xtol=1e-14)
            for i in np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)]


def spectrum(x_max: float = X_MAX):
    """[(x, label, multiplicity)] of all modes below x_max, ascending."""
    out = []
    for l in range(0, 12):
        if l >= 1:
            out += [(x, f"T{l}", 2 * l + 1) for x in _roots(lambda x: torsional_eq(l, x), x_max)]
        out += [(x, f"S{l}", 2 * l + 1) for x in _roots(lambda x: spheroidal_det(l, x), x_max)]
    return sorted(out)


def breathing_closed_form(nu: float = NU) -> float:
    """Lowest breathing mode from xi cot xi = 1 - xi^2 c_L^2 / (4 c_T^2), as x = omega a / c_T."""
    r2 = 2 * (1 - nu) / (1 - 2 * nu)
    xi = brentq(lambda s: s / np.tan(s) - 1 + s * s * r2 / 4, 0.5, 3.1)
    return xi * np.sqrt(r2)


def project(h_frac: float, order_: int = 2):
    from solidocean.physics.solid.model import SolidPhysics, SolidProblem
    from solidocean.project import Feature, Material, MeshSettings, Project
    return Project(
        name="Free sphere", physics_kind="solid", description="Validation: vibrations of a free sphere (Lamb 1882)",
        variables={"a": "1[um]"},
        materials=[Material("M", {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": str(NU)})],
        objects=[Feature("sphere", "Sphere", material="M", params={"center": ["0", "0", "0"], "radius": "a"})],
        mesh=MeshSettings(max_size=f"a*{h_frac:g}", order=order_),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=60)))


def c_t():
    return np.sqrt(100e9 / (2 * (1 + NU) * 2000.0))


def run(quick: bool = False) -> Report:
    from solidocean.physics.solid.solver import Solver
    r = Report("free_sphere", TITLE, ABOUT)
    with Timer() as tm:
        modes = spectrum()
        exact = np.concatenate([[x] * m for x, _, m in modes])
        breathing = [x for x, lab, _ in modes if lab == "S0"][0]
        r.check("breathing mode: determinant vs closed form", abs(breathing / breathing_closed_form() - 1), 1e-12)
        fracs = [1 / 5, 1 / 6, 1 / 8] if quick else [1 / 5, 1 / 6, 1 / 8, 1 / 10]
        errs, finest, hs = [], None, []
        for hf in fracs:
            p = project(hf)
            if hf == fracs[0]:
                common.save_model(p, "free_sphere")
            s = Solver(p, log=quiet)
            f = s.solve_point(p.evaluator(), np.zeros(3), n=6 + len(exact) + 4)
            x = 2 * np.pi * np.sort(f) * 1e-6 / c_t()
            assert np.all(x[:6] < 1e-3 * x[6]), "expected six rigid-body modes"
            got = x[6:6 + len(exact)]
            err = np.abs(got - exact) / exact
            errs.append(err.max())
            finest = got
            hs.append(common.mean_edge(s._model.mesh) / 1e-6)
        p2 = order(hs, errs)

        fig, ax = figure(6.4, 4.4)
        n = np.arange(1, len(exact) + 1)
        ax.step(n, exact, where="mid", color=common.REF_COLOR, lw=1.5, label="Exact (Lamb)")
        ax.plot(n, finest, "o", ms=4, color=common.FEM_COLOR, label="SolidOcean")
        i = 0
        for x, lab, m in modes:
            ax.annotate(lab, (i + 1 + (m - 1) / 2, x), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=8, color="0.3")
            i += m
        ax.set_xlabel(r"Mode number, $n$ (rigid-body modes removed)")
        ax.set_ylabel(r"Frequency, $\omega a / c_\mathrm{T}$")
        ax.legend(loc="lower right")
        r.figure(fig, "sphere_spectrum.png", f"Spectrum of the free sphere: exact multiplets and the computed "
                                             f"eigenfrequencies, mesh a/{1 / fracs[-1]:.0f}.")
        fig, ax = figure()
        ax.plot(hs, errs, "o-", color=common.FEM_COLOR, label=f"P2: order {p2:.1f}")
        common.slope_guide(ax, hs, np.array(errs), 4)
        common.convergence_axes(ax, r"Element size, $h / a$", r"Largest frequency error, $|\Delta\omega| / \omega$")
        ax.legend()
        r.figure(fig, "sphere_convergence.png", "Mesh convergence of the sphere's eigenfrequencies (curved quadratic "
                                                "elements; expected order 4).")
        r.check(f"largest frequency error (h = a/{1 / fracs[-1]:.0f})", errs[-1], 1e-3)
        r.check("convergence order", p2, 3.3, "min", ".2f")
    r.seconds = tm.seconds
    return r
