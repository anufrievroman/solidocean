"""Hydrodynamic heat transport (Guyer-Krumhansl): exact solutions of the steady problem."""
import numpy as np
import pytest

from solidocean.physics.heat.model import HeatCondition, HeatPhysics, HeatProblem
from solidocean.physics.heat.solver import HeatSolver
from solidocean.plots import ResultPlot
from solidocean.project import Feature, Material, MeshSettings, Project

K = 100.0
L, W, TH = 2e-6, 1e-6, 0.25e-6


def slab(ell, symmetry, size=(L, W, TH), conditions=None, max_size="w/5"):
    """Block with a hot end (400 K) and a cold end (300 K) in x; `symmetry` faces are slip walls, the others no slip."""
    mat = Material("A", {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                         "k": f"{K}[W/(m*K)]", "cp": "500[J/(kg*K)]"})
    return Project(
        name="hydro", physics_kind="heat", variables={"L": f"{size[0]}[m]", "w": f"{size[1]}[m]", "t": f"{size[2]}[m]"},
        materials=[mat], objects=[Feature("block", "A", material="A",
                                          params={"center": ["0", "0", "0"], "size": ["L", "w", "t"]})],
        mesh=MeshSettings(max_size=max_size, order=2),
        heat=HeatPhysics(initial="300[K]", symmetry=list(symmetry),
                         conditions=conditions or [HeatCondition("temperature", ["xmin"], "400[K]"),
                                                   HeatCondition("temperature", ["xmax"], "300[K]")],
                         problem=HeatProblem(type="hydrodynamic", viscous_length=f"{ell}[m]")),
        plots=[ResultPlot("qx", "mean(qx)")])


def run(p, quantities=("mean(qx)",)):
    return HeatSolver(p, log=lambda s: None).run(list(quantities))


@pytest.mark.parametrize("ell", [0.0, 0.3e-6])
def test_slip_walls_give_fourier(ell):
    """With slip walls the heat flux is uniform, q = k dT / L, whatever the viscous length."""
    res = run(slab(ell, ["ymin", "ymax", "zmin", "zmax"]))
    mesh, T = res.meshes[0], res.T[0][0]
    x = mesh.points[:, 0]
    np.testing.assert_allclose(T, 350 - 100 * x / L, atol=1e-6)
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(K * 100 / L, rel=1e-6)
    q = res.kels[0].q
    assert np.abs(q[:, 1:]).max() < 1e-6 * K * 100 / L


@pytest.mark.parametrize("ell, tol", [(1e-6, 1e-4), (0.2e-6, 4e-3)])
def test_poiseuille_profile(ell, tol):
    """No slip at y = -w/2, w/2 (slip at z): q_x(y) = k dT/L (1 - cosh(y/l) / cosh(w/2l)), and the flux is reduced
    from Fourier's by the factor 1 - 2 l tanh(w / 2l) / w. (The boundary layer l = w/5 is only just resolved.)"""
    res = run(slab(ell, ["zmin", "zmax"]))
    q = res.kels[0].q
    y = res.meshes[0].points[:, 1]
    q0 = K * 100 / L
    np.testing.assert_allclose(q[:, 0], q0 * (1 - np.cosh(y / ell) / np.cosh(W / 2 / ell)), atol=tol * q0)
    ratio = res.quantities["mean(qx)"][0, 0] / q0
    assert ratio == pytest.approx(1 - 2 * ell * np.tanh(W / 2 / ell) / W, rel=tol)


def strip_flux(x, y, chi, sigma, nk=4000):
    """Heat flux (q_x, q_y) in units of I / w of the infinite strip 0 < y < w (lengths in w, no slip) with Gaussian
    source and drain contacts of standard deviation sigma on opposite walls at x = 0 (Shang et al., Sci. Rep. 10,
    8272 (2020), Fig. 1c), by a stream function psi with q = (-dpsi/dy, dpsi/dx) and a Fourier transform in x."""
    W, ell2 = 0.5, 1 / chi
    s = y - W
    k = (np.arange(nk) + 0.5) * (8 / sigma) / nk
    kp = np.sqrt(k ** 2 + 1 / ell2)
    A, B = k * np.tanh(k * W), kp * np.tanh(kp * W)
    c = (B / (B - A))[:, None]

    def ratios(kk):                              # cosh(ks) / cosh(kW) and sinh(ks) / cosh(kW), without overflow
        a = np.exp(kk * (np.abs(s) - W)) / (1 + np.exp(-2 * kk * W))
        b = np.exp(-kk * (np.abs(s) + W)) / (1 + np.exp(-2 * kk * W))
        return a + b, np.sign(s) * (a - b)
    f, fs = ratios(k[:, None])
    g, gs = ratios(kp[:, None])
    F = c * f + (1 - c) * g
    Fs = c * k[:, None] * fs + (1 - c) * kp[:, None] * gs
    wk = (np.exp(-(k * sigma) ** 2 / 2) * (8 / sigma) / nk / np.pi)[:, None]
    return (-(np.sin(np.outer(x, k)) / k) @ (Fs * wk), np.cos(np.outer(x, k)) @ (F * wk))


def test_heat_vortex_strip():
    """Source and drain on opposite walls of a ribbon with chi = w^2 / l^2 = 1/2 (no slip, slip at z): the current
    recirculates in two vortices beside the direct flow. The 3D model, a slab closed by symmetry planes, gives
    the heat flux of the exact 2D solution of the infinite strip (Gaussian contacts, sigma = 0.15 w; the ends of
    the slab are heat baths 4 w away) to within 1 % of its peak."""
    from solidocean.physics.heat.model import Profile
    chi, sigma, I = 0.5, 0.15 * W, 1.0                       # I: heat per unit length of the contact line (W/m)
    f = I / (sigma * np.sqrt(2 * np.pi))
    prof = Profile(shape="expr", expr=f"exp(-x^2/(2*({sigma})^2))")
    conds = [HeatCondition("temperature", ["xmin", "xmax"], "300[K]"),
             HeatCondition("flux", ["ymin"], f"{f}[W/m^2]", profile=prof),
             HeatCondition("flux", ["ymax"], f"{-f}[W/m^2]", profile=prof)]
    res = run(slab(W / np.sqrt(chi), ["zmin", "zmax"], size=(8 * W, W, W / 4), conditions=conds, max_size="w/8"), [])
    P, q = res.meshes[0].points, res.kels[0].q
    inner = (np.abs(P[:, 0]) < 1.6 * W) & (np.abs(P[:, 1]) < 0.3 * W)
    xs, ys = np.unique(P[inner, 0]), np.unique(np.round(P[inner, 1], 12))
    qx, qy = strip_flux(xs / W, ys / W + 0.5, chi, sigma / W)
    qx, qy = qx * I / W, qy * I / W
    ix = np.searchsorted(xs, P[inner, 0])
    iy = np.searchsorted(ys, np.round(P[inner, 1], 12))
    peak = max(np.abs(qx).max(), np.abs(qy).max())
    assert np.abs(q[inner, 0] - qx[ix, iy]).max() < 0.01 * peak
    assert np.abs(q[inner, 1] - qy[ix, iy]).max() < 0.01 * peak
    # the vortices: against the direct flow (q_y = 1.5 I / w at x = 0), the heat runs backwards at x = w, mid-width
    from solidocean.fields import PointSampler
    qy_line = PointSampler(res.meshes[0], np.array([[0.0, 0.0, 0.0], [W, 0.0, 0.0]])).values(q[:, 1])
    assert qy_line[0] > 1.4 * I / W and qy_line[1] < -0.015 * I / W


def test_unsupported_features_are_reported():
    p = slab(0.3e-6, ["zmin", "zmax"], conditions=[HeatCondition("temperature", ["xmin"], "400[K]"),
                                                   HeatCondition("convection", ["xmax"])])
    assert any("only temperature and flux" in e for e in p.validate())
    p = slab(0.3e-6, ["zmin", "zmax"])
    p.mesh.order = 1
    assert any("quadratic" in e for e in p.validate())
    assert not slab(0.3e-6, ["zmin", "zmax"]).validate()


def test_no_heat_bath_needs_balanced_flux():
    conds = [HeatCondition("flux", ["xmin"], "1e6[W/m^2]"), HeatCondition("flux", ["xmax"], "-0.5e6[W/m^2]")]
    with pytest.raises(ValueError, match="no steady state"):
        run(slab(0.3e-6, ["zmin", "zmax"], conditions=conds))
    conds[1] = HeatCondition("flux", ["xmax"], "-1e6[W/m^2]")
    res = run(slab(0.3e-6, ["zmin", "zmax"], conditions=conds))
    q = res.kels[0].q
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(1e6, rel=1e-6)    # the flux passes through unchanged
