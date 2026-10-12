"""Nonlinear heat transfer: radiating boundaries and temperature-dependent properties, against analytic
solutions (1D slabs, lumped bodies)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from solidocean.physics.heat.model import HeatCondition, HeatPhysics, HeatStudy
from solidocean.physics.heat.solver import SIGMA, HeatSolver
from solidocean.project import Feature, Material, MeshSettings, Project

L, W = 1e-6, 0.2e-6
RHOC = 1e6          # rho*cp, J/(m^3 K)


def bar(k: str, conditions, cp: str = "500[J/(kg*K)]", study=None, initial="300[K]"):
    """Bar of length L along x (cross-section W x W); rho = 2000 kg/m^3."""
    m = Material("A", {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3", "k": k, "cp": cp})
    return Project(
        name="bar", physics_kind="heat", variables={"L": "1[um]", "w": "0.2[um]"}, materials=[m],
        geometry=[Feature("block", "A", material="A", params={"center": ["L/2", "0", "0"], "size": ["L", "w", "w"]})],
        mesh=MeshSettings(max_size="L/8"),
        heat=HeatPhysics(initial=initial, conditions=list(conditions), study=study or HeatStudy(tol=1e-9)))


def run(p, quantities=()):
    return HeatSolver(p, log=lambda s: None).run(list(quantities))


def test_radiating_end_of_a_conducting_bar():
    """Fixed 1000 K at x = 0, the end x = L radiates to 300 K: k (T0 - TL) / L = eps sigma (TL^4 - Ta^4)."""
    k, eps, T0, Ta = 1e-4, 0.8, 1000.0, 300.0
    p = bar(f"{k}[W/(m*K)]", [HeatCondition("temperature", ["xmin"], f"{T0}[K]"),
                              HeatCondition("radiation", ["xmax"], emissivity=str(eps), T_ext=f"{Ta}[K]")])
    res = run(p)
    TL = brentq(lambda t: k * (T0 - t) / L - eps * SIGMA * (t ** 4 - Ta ** 4), Ta, T0)
    x, T = res.meshes[0].points[:, 0], res.T[0][0]
    assert 400 < TL < 900                                           # strongly nonlinear
    np.testing.assert_allclose(T, T0 + (TL - T0) * x / L, rtol=1e-7)


def test_conductivity_linear_in_temperature():
    """k = k0 (1 + b (T - 300 K)) between 400 K and 300 K: u = s + b s^2 / 2 (s = T - 300 K) is linear in x
    (Kirchhoff transform) and the flux is k0 u(400 K) / L."""
    k0, b = 100.0, 0.005
    p = bar(f"{k0}[W/(m*K)] * (1 + {b}*(T/1[K] - 300))",
            [HeatCondition("temperature", ["xmin"], "400[K]"), HeatCondition("temperature", ["xmax"], "300[K]")])
    res = run(p, ["mean(qx)"])
    x, T = res.meshes[0].points[:, 0], res.T[0][0]
    u1 = 100 + b / 2 * 100 ** 2
    u = u1 * (1 - x / L)
    s = (-1 + np.sqrt(1 + 2 * b * u)) / b
    np.testing.assert_allclose(T, 300 + s, atol=5e-3)        # 100 K drop, P2 elements of L/8
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(k0 * u1 / L, rel=1e-4)


def test_radiative_cooling_of_a_lumped_bar():
    """A well-conducting bar radiating from every face cools as rho cp V dT/dt = -eps sigma A (T^4 - Ta^4)."""
    eps, Ta, t_end = 0.5, 300.0, 4e-3
    st = HeatStudy(type="transient", t_end=f"{t_end}[s]", steps=100, theta=0.5, tol=1e-8)
    p = bar("100[W/(m*K)]", [HeatCondition("radiation", ["all"], emissivity=str(eps), T_ext=f"{Ta}[K]")],
            study=st, initial="1000[K]")
    res = run(p, ["mean(T)"])
    A_over_V = (2 * W * W + 4 * L * W) / (L * W * W)
    ode = solve_ivp(lambda t, T: -eps * SIGMA * A_over_V / RHOC * (T ** 4 - Ta ** 4), (0, t_end), [1000.0],
                    t_eval=res.times, rtol=1e-10, atol=1e-10)
    assert ode.y[0, -1] < 800                                        # cooled substantially
    np.testing.assert_allclose(res.quantities["mean(T)"][0], ode.y[0], rtol=2e-4)


def test_heat_capacity_depending_on_temperature_conserves_energy():
    """Heating at x = 0 with cp = c0 (1 + g (T - 300 K)): the stored energy rho int cp dT equals the heat put in."""
    c0, g, q0, t_end = 500.0, 0.002, 1e8, 1e-6
    st = HeatStudy(type="transient", t_end=f"{t_end}[s]", steps=100, tol=1e-8)
    p = bar("100[W/(m*K)]", [HeatCondition("flux", ["xmin"], f"{q0}[W/m^2]")],
            cp=f"{c0}[J/(kg*K)] * (1 + {g}*(T/1[K] - 300))", study=st)
    stored = f"integral(2000*{c0}*((T - 300) + {g}/2*(T - 300)^2))"
    res = run(p, [stored, "mean(T)"])
    E = res.quantities[stored][0]
    np.testing.assert_allclose(E[1:], q0 * W * W * res.times[1:], rtol=2e-3)
    assert res.quantities["mean(T)"][0, -1] > 360                    # cp grows noticeably with T


def test_nonlinear_settings_round_trip_and_validation(tmp_path):
    p = bar("100[W/(m*K)] * 300[K] / T", [HeatCondition("radiation", ["xmax"], emissivity="0.3"),
                                          HeatCondition("temperature", ["xmin"], "400[K]")],
            study=HeatStudy(tol=1e-8, max_iter=20))
    q = Project.load(p.save(tmp_path / "m.toml"))
    c = q.heat.conditions[0]
    assert (c.type, c.emissivity, c.T_ext) == ("radiation", "0.3", "293.15[K]")
    assert (q.heat.study.tol, q.heat.study.max_iter) == (1e-8, 20)
    assert not q.validate()
    q.heat.conditions[0].emissivity = "1.5"
    assert any("emissivity" in e for e in q.validate())
    q.heat.conditions[0].emissivity = "1"
    q.materials[0].props["k"] = "100[W/(m*K)] * (400[K] - T) / 1[K]"     # 0 at 400 K: reported when reached
    with pytest.raises(ValueError, match="conductivity not positive"):
        q.heat.conditions[1].value = "500[K]"
        run(q)


def test_reassembly_matches_assembly():
    """The fast re-assembler (cached geometry and sparsity pattern) gives the matrices of the plain assembly."""
    from solidocean.fem import heat
    p = bar("100[W/(m*K)]", [])
    m = HeatSolver(p, log=lambda s: None).model(p.evaluator())
    mesh = m.mesh
    a = heat.Assembler(mesh.points, mesh.cells, mesh.order)
    E, Q = mesh.cells.shape[0], len(a.qw)
    k = np.broadcast_to(np.diag([100.0, 50.0, 20.0]), (E, Q, 3, 3))
    K, C = a.matrices(k, np.full((E, Q), RHOC))
    K0, C0 = heat.assemble(mesh.points, mesh.cells, mesh.cell_domain, [np.diag([100.0, 50.0, 20.0])], [RHOC], mesh.order)
    assert abs(K - K0).max() <= 1e-12 * abs(K0).max() and abs(C - C0).max() <= 1e-12 * abs(C0).max()
