"""FDTR: a modulated Gaussian pump on a gold transducer on silicon, against the layered analytic model."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

import common
from common import Report, Timer, figure, quiet

TITLE = "FDTR: transducer film, boundary conductance and substrate"
ABOUT = r"""
Frequency-domain thermoreflectance: a pump laser (Gaussian spot, FWHM $w$) modulated at $f$ heats a 100 nm gold
transducer on silicon, with a thermal boundary conductance $G$ between them; a probe of the same size reads the
film's temperature. Reference: the standard layered model of FDTR and TDTR (Cahill, Rev. Sci. Instrum. 75, 5119
(2004); Schmidt et al., ibid. 80, 094901 (2009)). In Hankel space the surface temperature is the heat flux times
the impedance of the stack: $Z = 1/(k_s q_s)$ for the substrate, plus $1/G$ for the interface, carried through the
film by
$$Z \leftarrow \frac{Z + \tanh(q_f d)/(k_f q_f)}{1 + k_f q_f Z \tanh(q_f d)},\qquad q = \sqrt{\lambda^2 +
i\omega/\alpha},$$
and the probed temperature is $\Delta T = \frac{P}{2\pi}\int_0^\infty \lambda Z(\lambda)\,
e^{-\lambda^2(w_0^2 + w_1^2)/8}\,d\lambda$ ($w_0$, $w_1$ the $1/e^2$ radii of pump and probe). The model is the
template "FDTR: gold transducer on silicon" (a quarter of the spot, the boundary conductance as a contact
resistance $1/G$, far faces at the ambient temperature); the probed temperature is the film's top
surface temperature weighted by the probe's Gaussian. (The template's plots weight the film's volume instead,
which reads about 1 % lower: the film's top is warmer than its average by about $q d / 2k_f$.)
"""


def layered(f: float, p) -> complex:
    """Probe-averaged temperature oscillation of the layered model (K) at frequency f."""
    from solidocean.materials import LIBRARY
    ev = p.evaluator()

    def props(name):
        m = LIBRARY[name]
        return ev(m["k"]), ev(m["rho"]) * ev(m["cp"])
    kf, cf = props("Au")
    ks, cs = props("Si")
    d, G, w, P = ev("d"), ev("G"), ev("w"), ev("P")
    w0 = w / np.sqrt(2 * np.log(2))
    om = 2 * np.pi * f

    def Z(lam):
        qs, qf = np.sqrt(lam * lam + 1j * om * cs / ks), np.sqrt(lam * lam + 1j * om * cf / kf)
        z = 1 / (ks * qs) + 1 / G
        t = np.tanh(qf * d)
        return (z + t / (kf * qf)) / (1 + kf * qf * z * t)

    def g(lam, part):
        v = lam * Z(lam) * np.exp(-lam * lam * 2 * w0 ** 2 / 8)
        return v.real if part == 0 else v.imag
    top = 40 / w0
    return P / (2 * np.pi) * (quad(g, 0, top, args=(0,), limit=500)[0] + 1j * quad(g, 0, top, args=(1,), limit=500)[0])


def probed(mesh, T, w, d) -> complex:
    """Probe reading: the top surface's temperature weighted by the probe's Gaussian (surface integrals)."""
    from solidocean.fem import surface
    tris = surface.boundary_faces(mesh.cells, mesh.order)
    top = np.all(np.abs(mesh.points[tris[:, :3], 2] - d) < 1e-12, axis=1)
    xq = np.stack([surface.surface_at_points(mesh.points, tris[top], mesh.order, mesh.points[:, i])
                   for i in range(2)], axis=-1)
    g = np.exp(-4 * np.log(2) * (xq[..., 0] ** 2 + xq[..., 1] ** 2) / w ** 2)
    _, wt = surface.surface_terms(mesh.points, tris[top], mesh.order, load=g)
    return complex(wt @ T / wt.sum())


def run(quick: bool = False) -> Report:
    from solidocean import templates
    from solidocean.physics.heat.solver import HeatSolver
    r = Report("fdtr", TITLE, ABOUT)
    with Timer() as tm:
        p = templates.create("heat_fdtr")
        p.study.sweep.values = "logspace(6, 7.3, 4) * 1[Hz]" if quick else "logspace(6, 7.3, 7) * 1[Hz]"
        common.save_model(p, "fdtr")
        res = HeatSolver(p, log=quiet).run()
        fs = res.outer_values
        ev = p.evaluator()
        fem = np.array([probed(res.meshes[i], res.T[i][0], ev("w"), ev("d")) for i in range(len(fs))])
        ref = np.array([layered(f, p) for f in fs])
        dphase = np.degrees(np.angle(fem) - np.angle(ref))
        damp = np.abs(fem) / np.abs(ref) - 1
        r.table(f"Probed temperature oscillation (P = 1 mW, w = 2 µm, G = 150 MW/(m²·K); mesh {p.mesh.max_size}, "
                f"{res.meshes[0].n_nodes} nodes).",
                ["f (MHz)", "layered model: phase (°)", "amplitude (K)", "SolidOcean: phase (°)", "amplitude (K)",
                 "phase difference (°)", "amplitude difference"],
                [(f"{f / 1e6:.3g}", f"{np.degrees(np.angle(a)):.2f}", f"{abs(a):.4f}", f"{np.degrees(np.angle(b)):.2f}",
                  f"{abs(b):.4f}", f"{d:+.2f}", f"{e:+.1e}") for f, a, b, d, e in zip(fs, ref, fem, dphase, damp)])
        fig, ax = figure()
        ff = np.logspace(np.log10(fs[0]), np.log10(fs[-1]), 40)
        ax.plot(ff / 1e6, [np.degrees(np.angle(layered(f, p))) for f in ff], "-", color=common.REF_COLOR,
                label="layered model")
        ax.plot(fs / 1e6, np.degrees(np.angle(fem)), "o", mfc="none", color=common.FEM_COLOR, label="SolidOcean")
        ax.set_xscale("log")
        ax.set_xlabel("Modulation frequency (MHz)")
        ax.set_ylabel("Phase of the probed temperature (°)")
        ax.legend()
        r.figure(fig, "fdtr_phase.png", "FDTR phase against frequency: the 3D model and the layered analytic model.")
        r.notes.append("The difference at the lowest frequencies comes from the finite model: the far faces, held at the "
                       "ambient temperature 10 µm away, are within a few thermal penetration depths (3.7 µm in Si at "
                       "1 MHz).")
        r.check("phase difference (degrees), every frequency", np.abs(dphase).max(), 0.2, fmt=".2f")
        r.check("amplitude difference, every frequency", np.abs(damp).max(), 1e-2)
    r.seconds = tm.seconds
    return r
