"""Harmonic (frequency-domain) response of solids: driven longitudinal waves in a bar against the exact 1D
solution (uniaxial strain), loss factors, and absorbing ends."""
import numpy as np
import pytest

from solidocean.physics.solid.model import SolidCondition, SolidPhysics, SolidProblem
from solidocean.physics.solid.solver import Solver
from solidocean.project import Feature, Material, MeshSettings, Project

E, NU, RHO = 100e9, 0.25, 2000.0
C11 = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))
CL = np.sqrt(C11 / RHO)
L, W = 4e-6, 0.2e-6


def bar(conditions, f, loss="0", eta=None, length="L"):
    """Bar along x in uniaxial strain: the side faces are symmetry planes (no lateral motion)."""
    props = {"model": "isotropic", "rho": f"{RHO:g}[kg/m^3]", "E": f"{E / 1e9:g}[GPa]", "nu": str(NU)}
    if eta:
        props["eta"] = eta
    return Project(name="bar", variables={"L": "4[um]", "w": "0.2[um]", "f": f"{f:g}[Hz]"},
                   materials=[Material("M", props)],
                   objects=[Feature("block", "Bar", material="M", params={"center": [f"{length}/2", "0", "0"],
                                                                           "size": [length, "w", "w"]})],
                   mesh=MeshSettings(max_size="w/2"),
                   solid=SolidPhysics(symmetry=["ymin", "ymax", "zmin", "zmax"], conditions=list(conditions),
                                      problem=SolidProblem(type="frequency", frequency="f", loss=loss)))


def run(p):
    res = Solver(p, log=lambda m: None).run()
    return res.meshes[0].points[:, 0], res.U[0]


@pytest.mark.parametrize("loss", [0.0, 0.05])
def test_driven_end_displacement(loss):
    """u(0) = u0, free end: u = u0 cos(k (L - x)) / cos(k L), k = omega / (c_L sqrt(1 + i eta))."""
    f, u0 = 1.3e9, 1e-9
    p = bar([SolidCondition("displacement", ["xmin"], displacement=[f"{u0:g}[m]", "", ""])], f, loss=str(loss))
    assert p.validate() == []
    x, u = run(p)
    k = 2 * np.pi * f / (CL * np.sqrt(1 + 1j * loss))
    exact = u0 * np.cos(k * (L - x)) / np.cos(k * L)
    assert abs(exact).max() > 2 * u0                                     # near a resonance: a real test
    np.testing.assert_allclose(u[:, 0], exact, atol=2e-3 * abs(exact).max())
    assert np.abs(u[:, 1:]).max() < 1e-4 * abs(exact).max()                # uniaxial strain (round-off)


@pytest.mark.parametrize("alpha, beta, loss", [(2e8, 0.0, 0.0), (0.0, 5e-12, 0.0), (1e8, 3e-12, 0.02)])
def test_rayleigh_damping(alpha, beta, loss):
    """C = alpha M + beta K turns rho omega^2 into rho (omega^2 - i omega alpha) and C11 (1 + i eta) into
    C11 (1 + i eta + i omega beta): the driven bar of test_driven_end_displacement with that wavenumber."""
    f, u0 = 1.3e9, 1e-9
    p = bar([SolidCondition("displacement", ["xmin"], displacement=[f"{u0:g}[m]", "", ""])], f, loss=str(loss))
    p.solid.problem.rayleigh_alpha, p.solid.problem.rayleigh_beta = f"{alpha:g}[1/s]", f"{beta:g}[s]"
    assert p.validate() == []
    x, u = run(p)
    w = 2 * np.pi * f
    k = np.sqrt(RHO * (w ** 2 - 1j * w * alpha) / (C11 * (1 + 1j * loss + 1j * w * beta)))
    exact = u0 * np.cos(k * (L - x)) / np.cos(k * L)
    assert abs(exact).max() > 2 * u0                                      # still near the resonance
    assert np.abs(np.angle(exact[np.argmax(x)])) > 0.05                  # and visibly damped
    np.testing.assert_allclose(u[:, 0], exact, atol=2e-3 * abs(exact).max())


def test_end_force_with_fixed_end():
    """Fixed at x = 0, force amplitude F on x = L: u = F sin(k x) / (C11 A k cos(k L))."""
    f, F = 0.7e9, 1e-6
    p = bar([SolidCondition("fixed", ["xmin"]),
             SolidCondition("load", ["xmax"], load_type="total", load=[f"{F:g}[N]", "0", "0"])], f)
    x, u = run(p)
    k = 2 * np.pi * f / CL
    exact = F * np.sin(k * x) / (C11 * W * W * k * np.cos(k * L))
    np.testing.assert_allclose(u[:, 0].real, exact, atol=2e-3 * abs(exact).max())
    assert np.abs(u[:, 0].imag).max() < 1e-12 * abs(exact).max()        # lossless: in phase


def test_graded_absorber_gives_a_travelling_wave():
    """A loss factor growing over the last 6 um of a long bar absorbs the wave: in the lossless part the
    amplitude is nearly uniform (a travelling wave) and the phase falls by k per unit length."""
    f, u0 = 2e9, 1e-9
    eta = "if(x > 4[um], 2 * ((x - 4[um]) / 6[um])^2, 0)"
    p = bar([SolidCondition("displacement", ["xmin"], displacement=[f"{u0:g}[m]", "", ""])], f, eta=eta,
            length="10[um]")
    x, u = run(p)
    free = x < 3.5e-6
    amp = np.abs(u[free, 0])
    assert amp.max() / amp.min() < 1.1                                  # standing-wave ratio near 1
    k = 2 * np.pi * f / CL
    order = np.argsort(x[free])
    phase = np.unwrap(np.angle(u[free, 0][order]))
    slope = np.polyfit(x[free][order], phase, 1)[0]
    assert slope == pytest.approx(-k, rel=0.02)                         # travelling towards +x


def test_frequency_problem_input_and_checks(tmp_path):
    p = bar([SolidCondition("displacement", ["xmin"], displacement=["1[nm]", "", ""])], 1e9, loss="0.01",
            eta="0.1 * x / L")
    p.solid.problem.rayleigh_alpha, p.solid.problem.rayleigh_beta = "1e6[1/s]", "1e-12[s]"
    q = Project.load(p.save(tmp_path / "m.toml"))
    st = q.solid.problem
    assert (st.type, st.frequency, st.loss) == ("frequency", "f", "0.01")
    assert (st.rayleigh_alpha, st.rayleigh_beta) == ("1e6[1/s]", "1e-12[s]")
    assert q.materials[0].props["eta"] == "0.1 * x / L"
    assert q.validate() == []
    q.variables["f"] = "0[Hz]"
    q.solid.problem.loss = "-1"
    q.solid.problem.rayleigh_beta = "-1[s]"
    errors = " ".join(q.validate())
    assert "frequency must be positive" in errors and "loss must not be negative" in errors
    assert "rayleigh_beta must not be negative" in errors
    q.materials[0].props["eta"] = "foo"
    assert any("eta" in e for e in q.validate())
