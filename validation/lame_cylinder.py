"""Thick-walled cylinder under internal pressure: Lamé's plane-strain solution."""

from __future__ import annotations

import numpy as np

import common
from common import Report, Timer, figure, order, quiet

TITLE = "Thick-walled cylinder under internal pressure"
ABOUT = r"""
A tube (inner radius $a$, outer radius $b = 2a$, $\nu = 0.3$) under internal pressure $p$, in plane strain. A
quarter of a slice is modelled: symmetry planes $x = 0$ and $y = 0$, and symmetry on both faces of the slice
($u_z = 0$, which is exactly plane strain). The pressure acts on the inner surface (a box selection). Reference:
Lamé's solution
$$\sigma_r = A - \frac{B}{r^2},\quad \sigma_\theta = A + \frac{B}{r^2},\quad \sigma_z = 2\nu A,\quad
u_r = \frac{1+\nu}{E}\left[(1-2\nu)A\,r + \frac{B}{r}\right],\qquad A = \frac{p\,a^2}{b^2-a^2},\;
B = \frac{p\,a^2 b^2}{b^2-a^2}.$$
Errors are $L^2$ norms over the volume, relative to the norm of the exact field, integrated with the solver's
quadrature. Expected orders: displacement $h^{p+1}$, stress $h^p$ for elements of degree $p$ (curved boundaries
included, the quadratic elements being isoparametric).
"""

NU, E, P = 0.3, 100e9, 100e6
A_IN, B_OUT = 1e-6, 2e-6
A_ = P * A_IN ** 2 / (B_OUT ** 2 - A_IN ** 2)
B_ = P * A_IN ** 2 * B_OUT ** 2 / (B_OUT ** 2 - A_IN ** 2)


def exact(r):
    sr, st = A_ - B_ / r ** 2, A_ + B_ / r ** 2
    ur = (1 + NU) / E * ((1 - 2 * NU) * A_ * r + B_ / r)
    return sr, st, np.full_like(r, 2 * NU * A_), ur


def project(h_frac: float, order_: int = 2):
    from solidocean.physics.solid.model import SolidCondition, SolidPhysics, SolidProblem
    from solidocean.project import Feature, Material, MeshSettings, Project
    return Project(
        name="Lame cylinder", physics_kind="solid",
        description="Validation: thick-walled cylinder under internal pressure (Lamé, plane strain)",
        variables={"a": "1[um]", "b": "2[um]", "H": "0.25[um]", "p": f"{P / 1e6:g}[MPa]"},
        materials=[Material("M", {"model": "isotropic", "rho": "2000[kg/m^3]", "E": f"{E / 1e9:g}[GPa]",
                                   "nu": str(NU)})],
        objects=[
            Feature("cylinder", "Tube", material="M",
                    params={"center": ["0", "0", "0"], "axis": ["0", "0", "1"], "radius": "b", "height": "H"}),
            Feature("cylinder", "Bore", op="subtract",
                    params={"center": ["0", "0", "0"], "axis": ["0", "0", "1"], "radius": "a", "height": "H"}),
            Feature("block", "Cut x", op="subtract", params={"center": ["-b", "0", "H/2"], "size": ["2*b", "4*b", "2*H"]}),
            Feature("block", "Cut y", op="subtract", params={"center": ["0", "-b", "H/2"], "size": ["4*b", "2*b", "2*H"]})],
        mesh=MeshSettings(max_size=f"(b-a)*{h_frac:g}", order=order_),
        solid=SolidPhysics(
            symmetry=["xmin", "ymin", "zmin", "zmax"],
            conditions=[SolidCondition("load", ["all"], load_type="pressure", pressure="p", name="Internal pressure",
                                       box={"min": ["0", "0", "0"], "max": ["a", "a", "H"]})],
            problem=SolidProblem(type="stationary")))


R2 = "(x^2 + y^2)"
UR = f"({(1 + NU) / E * (1 - 2 * NU) * A_!r} * sqrt{R2} + {(1 + NU) / E * B_!r} / sqrt{R2})"
U_ERR = f"integral((ux - {UR} * x / sqrt{R2})^2 + (uy - {UR} * y / sqrt{R2})^2)"
U_NORM = f"integral({UR}^2)"
SXX = f"({A_!r} - {B_!r} * (x^2 - y^2) / {R2}^2)"
SYY = f"({A_!r} + {B_!r} * (x^2 - y^2) / {R2}^2)"
SXY = f"(-2 * {B_!r} * x * y / {R2}^2)"
SZZ = f"{2 * NU * A_!r}"
S_ERR = f"integral((sxx - {SXX})^2 + (syy - {SYY})^2 + (szz - {SZZ})^2 + 2 * (sxy - {SXY})^2)"
S_NORM = f"integral({SXX}^2 + {SYY}^2 + {SZZ}^2 + 2 * {SXY}^2)"


def run(quick: bool = False) -> Report:
    from solidocean import fields
    from solidocean.physics.solid.solver import Solver
    r = Report("lame_cylinder", TITLE, ABOUT)
    with Timer() as tm:
        fracs = [1 / 3, 1 / 4, 1 / 6] if quick else [1 / 3, 1 / 4, 1 / 6, 1 / 8, 1 / 12, 1 / 16]
        errs = {(o, f): [] for o in (1, 2) for f in ("u", "s")}
        hs = {1: [], 2: []}
        rows, profile = [], None
        for o in (1, 2):
            for hf in fracs:
                p = project(hf, o)
                if o == 2 and hf == 1 / 6:
                    common.save_model(p, "lame_cylinder")
                s = Solver(p, log=quiet)
                res = s.run([U_ERR, U_NORM, S_ERR, S_NORM])
                q = {k: v[0, 0] for k, v in res.quantities.items()}
                eu, es = np.sqrt(q[U_ERR] / q[U_NORM]), np.sqrt(q[S_ERR] / q[S_NORM])
                errs[(o, "u")].append(eu)
                errs[(o, "s")].append(es)
                h = common.mean_edge(res.meshes[0]) / (B_OUT - A_IN)
                hs[o].append(h)
                rows.append((f"P{o}", f"{h:.3f}", s._model.constraints.n_free, f"{eu:.2e}", f"{es:.2e}"))
                if o == 2 and hf == fracs[-1]:
                    rr = np.linspace(A_IN, B_OUT, 60)
                    pts = np.column_stack([rr / np.sqrt(2), rr / np.sqrt(2), np.full_like(rr, 0.125e-6)])
                    smp = fields.PointSampler(res.meshes[0], pts)
                    ev = p.evaluator().values
                    c = s_ = 1 / np.sqrt(2)
                    sxx, syy, sxy, szz, ux, uy = (res.point_values(0, 0, e, ev, smp)
                                                  for e in ("sxx", "syy", "sxy", "szz", "ux", "uy"))
                    profile = (rr, sxx * c * c + syy * s_ * s_ + 2 * sxy * c * s_,
                               sxx * s_ * s_ + syy * c * c - 2 * sxy * c * s_, szz, ux * c + uy * s_)
        r.table("Relative L² errors of the displacement and the stress; h is the mean element edge / (b − a).",
                ["elements", "h", "DOFs", "displacement", "stress"], rows)
        ords = {k: order(hs[k[0]], v) for k, v in errs.items()}

        rr, sr, st, sz, ur = profile
        er, et, ez, eu = exact(rr)
        fig, (ax, bx) = __import__("matplotlib.pyplot").pyplot.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
        rf = np.linspace(A_IN, B_OUT, 200)
        ex = exact(rf)
        for vals, ref, lab, c in ((sr, ex[0], "σ_r", "C0"), (st, ex[1], "σ_θ", "C3"), (sz, ex[2], "σ_z", "C2")):
            ax.plot(rf * 1e6, ref / 1e6, "-", color=c, lw=1.2)
            ax.plot(rr[::3] * 1e6, vals[::3] / 1e6, "o", mfc="none", ms=4, color=c, label=lab)
        ax.set_xlabel("Radius r (µm)")
        ax.set_ylabel("Stress (MPa)")
        ax.legend(title="lines: Lamé", fontsize=8)
        bx.plot(rf * 1e6, ex[3] * 1e9, "-", color="C0", lw=1.2, label="Lamé")
        bx.plot(rr[::3] * 1e6, ur[::3] * 1e9, "o", mfc="none", ms=4, color="C0", label="SolidOcean")
        bx.set_xlabel("Radius r (µm)")
        bx.set_ylabel("Radial displacement u_r (nm)")
        bx.legend(fontsize=8)
        r.figure(fig, "lame_profiles.png", "Stresses and radial displacement through the wall (along the 45° line, "
                                           "quadratic elements, finest mesh) against Lamé's solution.")
        fig, ax = figure()
        for (o, f), c, m in (((1, "u"), "C0", "o-"), ((1, "s"), "C0", "s--"), ((2, "u"), "C3", "o-"),
                             ((2, "s"), "C3", "s--")):
            ax.plot(hs[o], errs[(o, f)], m, color=c, mfc="none" if f == "s" else c,
                    label=f"P{o} {'displacement' if f == 'u' else 'stress'}: order {ords[(o, f)]:.1f}")
        common.convergence_axes(ax, "Mean element edge h / (b − a)", "Relative L² error")
        ax.legend(fontsize=8)
        r.figure(fig, "lame_convergence.png", "Mesh convergence of displacement and stress (expected orders: P1 2 "
                                              "and 1, P2 3 and 2).")
        r.check("P2 stress error (finest mesh)", errs[(2, "s")][-1], 2e-3)
        r.check("P2 displacement order", ords[(2, "u")], 2.7, "min", ".2f")
        r.check("P2 stress order", ords[(2, "s")], 1.8, "min", ".2f")
        r.check("P1 displacement order", ords[(1, "u")], 1.8, "min", ".2f")
        r.check("P1 stress order", ords[(1, "s")], 0.85, "min", ".2f")
    r.seconds = tm.seconds
    return r
