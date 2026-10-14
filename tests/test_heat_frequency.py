"""Frequency-domain heat transfer: complex amplitudes of the temperature oscillation against exact solutions."""
import numpy as np
import pytest

from solidocean.physics.heat.model import HeatCondition, HeatPhysics, HeatSource, HeatProblem, Pulse
from solidocean.physics.heat.solver import SIGMA, HeatSolver
from solidocean.project import Feature, Material, MeshSettings, VariableSweep, Project, Problem

K, RHO, CP = 100.0, 2000.0, 500.0
ALPHA = K / (RHO * CP)


def bar(conditions=(), sources=(), L="1[um]", h="L/8", f="10[MHz]", initial="300[K]"):
    m = Material("A", {"model": "isotropic", "rho": f"{RHO:g}[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                       "k": f"{K:g}[W/(m*K)]", "cp": f"{CP:g}[J/(kg*K)]"})
    return Project(name="bar", physics_kind="heat", variables={"L": L, "w": "0.2[um]", "f": f}, materials=[m],
                   objects=[Feature("block", "A", material="A", params={"center": ["L/2", "0", "0"],
                                                                         "size": ["L", "w", "w"]})],
                   mesh=MeshSettings(max_size=h),
                   heat=HeatPhysics(initial=initial, conditions=list(conditions), sources=list(sources),
                                    problem=HeatProblem(type="frequency")))


def run(p, quantities=()):
    return HeatSolver(p, log=lambda s: None).run(list(quantities))


def test_insulated_block_lags_by_90_degrees():
    """Uniform source amplitude Q, insulated: i omega rho cp T = Q, at every node (exact)."""
    Q = 1e15
    p = bar(sources=[HeatSource("A", f"{Q:g}[W/m^3]")])
    res = run(p, ["sqrt(mean(Tre)^2 + mean(Tim)^2)", "atan2(mean(Tim), mean(Tre)) * 180/pi", "mean(Tamp)"])
    w = 2 * np.pi * 10e6
    np.testing.assert_allclose(res.T[0][0], Q / (1j * w * RHO * CP), rtol=1e-10)
    q = {k: v[0, 0] for k, v in res.quantities.items()}
    assert q["sqrt(mean(Tre)^2 + mean(Tim)^2)"] == pytest.approx(Q / (w * RHO * CP), rel=1e-10)
    assert q["atan2(mean(Tim), mean(Tre)) * 180/pi"] == pytest.approx(-90, abs=1e-8)


def test_surface_flux_on_a_semi_infinite_solid():
    """q e^{i w t} into x = 0 of a long bar: T = q / (k s) e^{-s x}, s = sqrt(i w / alpha) (thermal wave):
    the surface lags the flux by 45 degrees; a sweep of the frequency through the variable sweep."""
    q0 = 1e9
    p = bar([HeatCondition("flux", ["xmin"], f"{q0:g}[W/m^2]"), HeatCondition("temperature", ["xmax"], "300[K]")],
            L="20[um]", h="0.15[um]")
    p.problem = Problem(sweep=VariableSweep("f", "[10, 40] * 1[MHz]"))
    res = run(p)
    x = res.meshes[0].points[:, 0]
    for i, f in enumerate(res.outer_values):
        s = np.sqrt(1j * 2 * np.pi * f / ALPHA)
        assert np.real(s) * 20e-6 > 8                                    # the wave dies out before the far end
        exact = q0 / (K * s) * np.exp(-s * x)
        assert np.abs(res.T[i][0] - exact).max() < 2e-3 * abs(exact[0])
        surf = res.T[i][0][np.argmin(x)]
        assert np.degrees(np.angle(surf)) == pytest.approx(-45, abs=0.1)


def test_sine_amplitude_drives_and_radiation_is_linearised():
    """A source with a sine profile drives with its amplitude (its mean is steady); radiation from all faces of
    an isothermal bar acts as the sink 4 eps sigma T0^3 A: T = Q V / (i w rho cp V + 4 eps sigma T0^3 A)."""
    Q, T0, eps = 1e13, 600.0, 0.8
    src = HeatSource("A", "5e15[W/m^3]", pulse=Pulse("sine", period="1[s]", amplitude=f"{Q:g}[W/m^3]"))
    p = bar([HeatCondition("radiation", ["all"], emissivity=str(eps), T_ext="300[K]")], [src], f="1[kHz]",
            initial=f"{T0:g}[K]")
    p.materials[0].props["k"] = "1e4[W/(m*K)]"                          # isothermal bar
    res = run(p)
    L, w = 1e-6, 0.2e-6
    V, A = L * w * w, 2 * w * w + 4 * L * w
    exact = Q * V / (1j * 2 * np.pi * 1e3 * RHO * CP * V + 4 * eps * SIGMA * T0 ** 3 * A)
    np.testing.assert_allclose(res.T[0][0], exact, rtol=1e-6)


def test_frequency_problem_input_and_checks(tmp_path):
    p = bar([HeatCondition("flux", ["xmin"], "1e6[W/m^2]", pulse=Pulse("rect"))])     # pulses: ignored
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert (q.heat.problem.type, q.heat.problem.frequency) == ("frequency", "f")
    assert q.validate() == []
    q.variables["f"] = "-1[Hz]"
    assert any("frequency must be positive" in e for e in q.validate())
