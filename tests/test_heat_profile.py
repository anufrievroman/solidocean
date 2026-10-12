"""Spatial profiles of heat fluxes and sources: deposited energy of Gaussian beams and expression profiles."""
import numpy as np
import pytest

from solidocean.physics.heat.model import HeatCondition, HeatPhysics, HeatSource, HeatStudy, Profile, Pulse
from solidocean.physics.heat.solver import HeatSolver
from solidocean.project import Feature, Material, MeshSettings, Project

RHOC = 2000.0 * 500.0
W, T_SLAB = 3e-6, 0.5e-6                     # FWHM, slab thickness
GAUSS_AREA = np.pi * W ** 2 / (4 * np.log(2))   # int exp(-4 ln2 r^2 / w^2) dA


def slab(conditions=(), sources=(), steps=1):
    """Insulated slab 14 x 14 um, 0.5 um thick (z from -t to 0), heated for 1 ns."""
    m = Material("A", {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                       "k": "100[W/(m*K)]", "cp": "500[J/(kg*K)]"})
    return Project(name="slab", physics_kind="heat", variables={"w": "3[um]", "t": "0.5[um]", "x0": "1[um]"},
                   materials=[m], geometry=[Feature("block", "Slab", material="A", params={
                       "center": ["0", "0", "-t/2"], "size": ["14[um]", "14[um]", "t"]})],
                   mesh=MeshSettings(max_size="0.5[um]"),
                   heat=HeatPhysics(initial="300[K]", conditions=list(conditions), sources=list(sources),
                                    study=HeatStudy("transient", t_end="1[ns]", steps=steps)))


def stored(p):
    r = HeatSolver(p, log=lambda s: None).run([f"integral({RHOC} * (T - 300))"])
    return r.quantities[f"integral({RHOC} * (T - 300))"][0, -1], r


def test_gaussian_flux_deposits_the_beam_power():
    """Peak flux q0 at (x0, 0): the absorbed power is q0 pi w^2 / (4 ln 2), and the top is hottest there."""
    q0 = 1e9
    p = slab([HeatCondition("flux", ["zmax"], f"{q0:g}[W/m^2]",
                            profile=Profile("gauss", center=["x0", "0"], fwhm="w"))])
    assert p.validate() == []
    E, r = stored(p)
    assert E == pytest.approx(q0 * GAUSS_AREA * 1e-9, rel=1e-3)
    mesh, T = r.meshes[0], r.T[0][-1]
    top = np.abs(mesh.points[:, 2]) < 1e-12
    hottest = mesh.points[top][np.argmax(T[top])]
    assert np.hypot(hottest[0] - 1e-6, hottest[1]) < 0.5e-6


def test_gaussian_source_and_pulse():
    """A Gaussian volume source through the slab's thickness, switched on for half the time."""
    Q0 = 1e15
    src = HeatSource("A", f"{Q0:g}[W/m^3]", profile=Profile("gauss", fwhm="w"),
                     pulse=Pulse("rect", start="0[s]", duration="0.5[ns]"))
    E, _ = stored(slab(sources=[src], steps=4))
    assert E == pytest.approx(Q0 * GAUSS_AREA * T_SLAB * 0.5e-9, rel=1e-3)


def test_expression_profile():
    """Absorption decaying linearly with depth: Q0 (1 + z / t) integrates to Q0 V / 2 (exact)."""
    Q0 = 1e14
    src = HeatSource("A", f"{Q0:g}[W/m^3]", profile=Profile("expr", expr="1 + z / t"))
    E, _ = stored(slab(sources=[src]))
    assert E == pytest.approx(Q0 * 14e-6 ** 2 * T_SLAB / 2 * 1e-9, rel=1e-10)


def test_profile_input_and_checks(tmp_path):
    p = slab([HeatCondition("flux", ["zmax"], "1e9[W/m^2]", profile=Profile("gauss", center=["x0", "-1[um]"],
                                                                            fwhm="w"))],
             [HeatSource("A", "1e12[W/m^3]", profile=Profile("expr", expr="exp(z / t)"))])
    q = Project.load(p.save(tmp_path / "m.toml"))
    c, s = q.heat.conditions[0].profile, q.heat.sources[0].profile
    assert (c.shape, c.center, c.fwhm) == ("gauss", ["x0", "-1[um]"], "w")
    assert (s.shape, s.expr) == ("expr", "exp(z / t)")
    assert q.validate() == []
    c.fwhm, s.expr = "-1[um]", "foo * z"
    errors = " ".join(q.validate())
    assert "fwhm must be positive" in errors and "unknown name 'foo'" in errors
    q.heat.sources[0].profile = Profile("ring")
    assert any("profile shape" in e for e in q.validate())
