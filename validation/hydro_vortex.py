"""Heat vortices of hydrodynamic phonon transport in a ribbon: a source and a drain on opposite walls."""

from __future__ import annotations

import numpy as np

import common
from common import Report, Timer, figure, figures, quiet

TITLE = "Heat vortices in hydrodynamic phonon transport"
ABOUT = r"""
Steady hydrodynamic phonon transport, the Guyer-Krumhansl equation for the heat flux $\mathbf q$ with the viscous
length $\ell$ [1, 2],
$$\mathbf q + k\nabla T = \ell^2\left[\nabla^2\mathbf q + 2\nabla(\nabla\cdot\mathbf q)\right],\qquad \nabla\cdot\mathbf q = 0,$$
in a ribbon $0 < y < w$ (no slip, $\mathbf q = 0$ at the walls) between a source and a drain: heat $I$ per unit
thickness enters through a Gaussian contact of width $\sigma$ on the wall $y = 0$ and leaves through the same contact
on the opposite wall, as in Fig. 1c of [2]. The viscosity of the phonon gas then makes the heat flow in two
counter-rotating vortices on either side of the direct flow, running backwards (from the cold to the hot side of the
contacts) when $\chi = w^2/\ell^2$ is below about 10; at larger $\chi$ the transport is diffusive (Fourier's law)
and the vortices vanish. The reference is the exact solution of the infinite strip, as in [2]. With the stream
function $\mathbf q = (-\partial_y\psi, \partial_x\psi)$ the curl of the equation gives
$\nabla^2(\nabla^2 - \ell^{-2})\psi = 0$, so each Fourier mode $e^{i\xi x}$ of $\psi$, even about the middle of the strip
($s = y - w/2$), is
$$\psi_\xi = P\,\frac{\cosh \xi s}{\cosh(\xi w/2)} + R\,\frac{\cosh \xi' s}{\cosh(\xi' w/2)},\qquad \xi'^2 = \xi^2 + \ell^{-2},$$
and the walls ($\psi_\xi = I/i\xi$ at $y = 0$ and $y = w$ for the contact, $\partial_y\psi_\xi = 0$ for no slip) fix
$$P = \frac{I}{i\xi}\frac{\xi'\tanh(\xi' w/2)}{D},\quad R = -\frac{I}{i\xi}\frac{\xi\tanh(\xi w/2)}{D},\quad
D = \xi'\tanh\tfrac{\xi' w}{2} - \xi\tanh\tfrac{\xi w}{2},$$
times the transform $e^{-\xi^2\sigma^2/2}$ of the Gaussian contact. The $x$-component of the equation gives the
temperature, only the part of $\mathbf q$ with $\cosh \xi s$ driving a gradient,
$$T_\xi = -\frac{I}{k}\,\frac{\xi'\tanh(\xi' w/2)}{D}\,\frac{\sinh \xi s}{\xi\cosh(\xi w/2)}.$$
The inverse transforms are one-dimensional quadratures.

The model is the same problem in 3D: a slab $8w$ long, $w$ wide and $w/4$ thick, closed by the symmetry planes
$z = \pm w/8$ (free slip, so it is exactly two-dimensional), with a heat bath at each end, $4w$ from the contacts
(where the solution has decayed), no-slip walls, and the contacts as the prescribed heat flux
$q_0 \exp(-x^2/2\sigma^2)$ with $\sigma = 0.15\,w$ on the faces $y = \mp w/2$. The viscous length is
$\ell = w/\sqrt{\chi}$ with $\chi = 0.5$, the case of the paper's Fig. 1c. Quadratic elements for $\mathbf q$ and
linear ones for $T$ (Taylor-Hood). The contacts are smooth on purpose: a contact with sharp edges makes the heat flux
jump from its full value to zero within one element, and the heat flowing through a section is then conserved only
slowly under mesh refinement (it was 25 % too large for a contact $0.1\,w$ wide on a mesh of $w/8$).

The comparison is made on a regular grid of points between $|x| \le 1.6\,w$ and $|y - w/2| \le 0.4\,w$, which
includes the vortices and reaches to $0.1\,w$ from the contacts. The reference for the temperature is the infinite
strip, where $T \to 0$ far away, against the model's baths at $300$ K; the temperature is also the less accurate of the
two fields, being linear in the elements.

[1] R. A. Guyer and J. A. Krumhansl, Phys. Rev. 148, 766 (1966).
[2] M.-Y. Shang, C. Zhang, Z. Guo and J.-T. Lü, Sci. Rep. 10, 8272 (2020).
"""

K = 100.0                  # W/(m K)
CHI, SIGMA = 0.5, 0.15     # chi = w^2 / l^2; contact width in w
W = 1e-6
I = 1.0                    # W/m: heat per unit thickness


def strip(x, y, chi, sigma, nk=6000):
    """Exact solution of the infinite strip 0 < y < 1 (lengths in w): stream function psi (units of I), heat flux
    (q_x, q_y) (units of I / w) and temperature (units of I / k) at the grid x (nx,) by y (ny,); no slip, Gaussian
    source and drain of standard deviation sigma on opposite walls at x = 0."""
    half = 0.5
    ell = 1 / np.sqrt(chi)
    s = np.asarray(y, float) - half
    k = (np.arange(nk) + 0.5) * (8.0 / sigma) / nk
    dk = (8.0 / sigma) / nk
    kp = np.sqrt(k ** 2 + 1 / ell ** 2)
    A, B = k * np.tanh(k * half), kp * np.tanh(kp * half)
    c = (B / (B - A))[:, None]                                   # P = (I / ik) c, R = (I / ik) (1 - c)

    def ratios(kk):                                              # cosh(ks) / cosh(kw/2), sinh(ks) / cosh(kw/2)
        a = np.exp(kk * (np.abs(s) - half)) / (1 + np.exp(-2 * kk * half))
        b = np.exp(-kk * (np.abs(s) + half)) / (1 + np.exp(-2 * kk * half))
        return a + b, np.sign(s) * (a - b)

    f, fs = ratios(k[:, None])
    g, gs = ratios(kp[:, None])
    F = c * f + (1 - c) * g
    Fs = c * k[:, None] * fs + (1 - c) * kp[:, None] * gs
    Tk = -c * fs / k[:, None]
    wk = (np.exp(-(k * sigma) ** 2 / 2) * dk / np.pi)[:, None]
    kx = np.outer(np.asarray(x, float), k)
    sin, cos = np.sin(kx), np.cos(kx)
    psi = (sin / k) @ (F * wk)
    qx = -(sin / k) @ (Fs * wk)
    qy = cos @ (F * wk)
    T = cos @ (Tk * wk)
    return psi, qx, qy, T


def project(h_frac: float):
    from solidocean.physics.heat.model import HeatCondition, HeatPhysics, HeatProblem, Profile
    from solidocean.plots import ResultPlot
    from solidocean.project import Feature, Material, MeshSettings, Project
    prof = Profile(shape="expr", expr="exp(-x^2/(2*s^2))")
    return Project(
        name="Heat vortices", physics_kind="heat",
        description="Validation: hydrodynamic phonon transport in a ribbon with a source and a drain, against the exact "
                    "solution of the infinite strip",
        variables={"w": "1[um]", "L": "8*w", "t": "w/4", "chi": f"{CHI:g}", "ell": "w/sqrt(chi)", "s": f"{SIGMA:g}*w",
                   "I": "1[W/m]", "q0": "I / (s * sqrt(2*pi))", "T0": "300[K]"},
        materials=[Material("M", {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                                  "k": f"{K:g}[W/(m*K)]", "cp": "500[J/(kg*K)]"})],
        objects=[Feature("block", "Ribbon", material="M", params={"center": ["0", "0", "0"], "size": ["L", "w", "t"]})],
        mesh=MeshSettings(max_size=f"w*{h_frac:g}", order=2),
        heat=HeatPhysics(initial="T0", symmetry=["zmin", "zmax"], conditions=[
            HeatCondition("temperature", ["xmin", "xmax"], "T0", name="Heat baths"),
            HeatCondition("flux", ["ymin"], "q0", name="Source", profile=prof),
            HeatCondition("flux", ["ymax"], "-q0", name="Drain", profile=prof)],
            problem=HeatProblem(type="hydrodynamic", viscous_length="ell")),
        plots=[ResultPlot("Heat flux", "qy", kind="field")])


def run(quick: bool = False) -> Report:
    from solidocean.fields import PointSampler
    from solidocean.physics.heat.solver import HeatSolver
    r = Report("hydro_vortex", TITLE, ABOUT)
    with Timer() as tm:
        h_frac = 1 / 8 if quick else 1 / 12
        p = project(h_frac)
        common.save_model(p, "hydro_vortex")
        res = HeatSolver(p, log=quiet).run([])
        mesh, q, T = res.meshes[0], res.kels[0].q, res.T[0][0]
        # the fields on a regular grid, in the model's coordinates (y centred on the strip, z = 0)
        xs = np.linspace(-1.6, 1.6, 161)
        ys = np.linspace(-0.4, 0.4, 41)
        X, Y = np.meshgrid(xs * W, ys * W, indexing="ij")
        samp = PointSampler(mesh, np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)])
        shape = X.shape
        qx_f, qy_f = (samp.values(q[:, i]).reshape(shape) * W / I for i in (0, 1))
        T_f = (samp.values(T).reshape(shape) - 300.0) * K / I
        psi_a, qx_a, qy_a, T_a = strip(xs, ys + 0.5, CHI, SIGMA)
        peak = max(np.abs(qx_a).max(), np.abs(qy_a).max())
        e_q = max(np.abs(qx_f - qx_a).max(), np.abs(qy_f - qy_a).max()) / peak
        e_T = np.sqrt(np.mean((T_f - T_a) ** 2)) / np.sqrt(np.mean(T_a ** 2))
        # heat through the strip: q_y integrated along the line y = w/2 against the injected heat
        line = np.linspace(-3.8, 3.8, 1521)
        mid = PointSampler(mesh, np.c_[line * W, np.zeros_like(line), np.zeros_like(line)])
        qy_line = mid.values(q[:, 1]) * W / I
        through = float(np.trapezoid(qy_line, line))
        # the vortex: the backflow beyond the direct flow, the largest x-integral of q_y above the injected heat
        psi_f = np.concatenate([[0.0], np.cumsum(0.5 * (qy_line[1:] + qy_line[:-1]) * np.diff(line))])
        dense = np.linspace(-3.8, 3.8, 1521)
        qy_dense = strip(dense, np.array([0.5]), CHI, SIGMA)[2][:, 0]
        psi_ref = np.concatenate([[0.0], np.cumsum(0.5 * (qy_dense[1:] + qy_dense[:-1]) * np.diff(dense))])
        over_f, over_a = psi_f.max() - psi_f[-1], psi_ref.max() - psi_ref[-1]

        # figure 1: q_y along the middle of the strip, and the backflow
        fig, (ax1, ax2) = figures(2)
        for ax in (ax1, ax2):
            ax.plot(dense, qy_dense, "-", color=common.REF_COLOR, lw=1.5, label="Exact")
            ax.plot(line[common.thin(np.arange(len(line)), 60)], qy_line[common.thin(np.arange(len(line)), 60)], "o",
                    ms=4, color=common.FEM_COLOR, label="SolidOcean")
            ax.axhline(0, color="0.8", lw=1, zorder=0)
            ax.set_xlabel(r"Position along the ribbon, $x / w$")
        ax1.set_xlim(-3, 3)
        ax1.set_ylabel(r"Heat flux across the ribbon, $q_y\, w / I$")
        ax1.legend()
        ax2.set_xlim(0.6, 3.2)
        ax2.set_ylim(-0.045, 0.06)
        ax2.set_ylabel(r"$q_y\, w / I$ (zoom: the backflow)")
        r.figure(fig, "hydro_vortex_profile.png",
                 f"Heat flux across the ribbon along its middle line, $y = w/2$ (mesh w/{1 / h_frac:.0f}), against the "
                 "exact solution; right: the backflow of the vortices, a few percent of the direct flow.")

        # figure 2: the heat flux lines, from the exact solution and the model
        fine_x, fine_y = np.linspace(-2.2, 2.2, 221), np.linspace(-0.5, 0.5, 51)
        FX, FY = np.meshgrid(fine_x * W, fine_y * W, indexing="ij")
        samp2 = PointSampler(mesh, np.c_[FX.ravel(), FY.ravel(), np.zeros(FX.size)])
        qy2 = samp2.values(q[:, 1]).reshape(FX.shape) * W / I
        dx = fine_x[1] - fine_x[0]
        psi_fem = np.cumsum(qy2, axis=0) * dx - 0.5 * qy2 * dx
        psi_exact = strip(fine_x, fine_y + 0.5, CHI, SIGMA)[0] + 0.5     # from 0 on the left to 1 on the right
        levels = np.concatenate([[-0.012, -0.006], np.linspace(0.1, 0.9, 5), [1.006, 1.012]])
        fig, ax = figure(8.0, 3.2)
        ax.contour(fine_x, fine_y, psi_exact.T, levels=levels, colors=common.REF_COLOR, linewidths=2.2, linestyles="solid")
        ax.contour(fine_x, fine_y, np.nan_to_num(psi_fem).T, levels=levels, colors=common.FEM_COLOR, linewidths=1.0,
                   linestyles="solid")
        ax.plot([], [], color=common.REF_COLOR, lw=2.2, label="Exact")
        ax.plot([], [], color=common.FEM_COLOR, lw=1.0, label="SolidOcean")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x / w$")
        ax.set_ylabel(r"$(y - w/2) / w$")
        ax.legend(loc="upper right", ncols=2)
        r.figure(fig, "hydro_vortex_lines.png",
                 "Heat flux lines (contours of the stream function, equal heat flow between the five central ones, and "
                 "the two closed loops of each vortex): exact solution (grey) and SolidOcean (green).")

        r.check("heat flux, largest error over the grid (of its peak)", e_q, 0.015, fmt=".1%")
        r.check("temperature, rms error (of its rms)", e_T, 0.06, fmt=".1%")
        r.check("heat through the strip against the injected (error)", abs(through - 1), 2e-3)
        r.check("recirculating heat of the vortex, error", abs(over_f / over_a - 1), 0.05, fmt=".1%")
        r.notes.append(f"Largest heat flux on the compared grid, $q w / I$ = {peak:.2f}; the vortex returns {over_a * 100:.1f} % of the injected heat "
                       f"(model: {over_f * 100:.1f} %).")
    r.seconds = tm.seconds
    return r
