"""Heat transfer: analytic checks of the stationary and transient solvers."""
import numpy as np
import pytest

from solidocean.fem import heat, surface
from solidocean.physics.heat.solver import HeatSolver
from solidocean.project import Feature, Material, MeshSettings, Project
from solidocean.physics.heat.model import HeatCondition, HeatPhysics, HeatSource, HeatProblem
from solidocean.plots import ResultPlot

K1, K2, RHO, CP = 100.0, 25.0, 2000.0, 500.0


def mat(name, k):
    return Material(name, {"model": "isotropic", "rho": f"{RHO}[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
                           "k": f"{k}[W/(m*K)]", "cp": f"{CP}[J/(kg*K)]"})


def bar(conditions, two_materials=False, sources=(), problem=None, order=2):
    """Bar of length L = 1 um along x (cross-section 0.2 x 0.2 um), insulated sides."""
    geo = [Feature("block", "A", material="A",
                   params={"center": ["0", "0", "0"], "size": ["L", "w", "w"]})]
    if two_materials:
        geo = [Feature("block", "A", material="A", params={"center": ["-L/4", "0", "0"], "size": ["L/2", "w", "w"]}),
               Feature("block", "B", material="B", params={"center": ["L/4", "0", "0"], "size": ["L/2", "w", "w"]})]
    return Project(
        name="bar", physics_kind="heat", variables={"L": "1[um]", "w": "0.2[um]"},
        materials=[mat("A", K1), mat("B", K2)], objects=geo, mesh=MeshSettings(max_size="L/8", order=order),
        heat=HeatPhysics(initial="300[K]", conditions=list(conditions), sources=list(sources),
                         problem=problem or HeatProblem()),
        plots=[ResultPlot("mean T", "mean(T)")],
    )


def hot_cold():
    return [HeatCondition("temperature", ["xmin"], "400[K]", name="hot"),
            HeatCondition("temperature", ["xmax"], "300[K]", name="cold")]


def test_surface_rule_and_boundary_area():
    pts, w = surface._tri_rule(3)
    assert w.sum() == pytest.approx(0.5)
    assert np.sum(w * pts[:, 0] ** 2 * pts[:, 1]) == pytest.approx(2 / 120)   # 2! 1! / 5!
    p = bar(hot_cold())
    m = HeatSolver(p, log=lambda s: None).model(p.evaluator())
    lo, hi = m.mesh.bounds()
    tris = heat.faces_on(m.mesh.points, m.tris, "xmax", 1e-12)
    M, f = heat.surface_terms(m.mesh.points, tris, m.mesh.order, coeff=1.0, load=1.0)
    assert f.sum() == pytest.approx(0.2e-6 ** 2, rel=1e-9)            # face area


@pytest.mark.parametrize("order", [1, 2])
def test_linear_profile_and_flux(order):
    p = bar(hot_cold(), order=order)
    p.plots = [ResultPlot("q", "mean(qx)"), ResultPlot("T", "mean(T)")]
    res = HeatSolver(p, log=lambda s: None).run(["mean(qx)", "mean(T)"])
    x = res.meshes[0].points[:, 0]
    np.testing.assert_allclose(res.T[0][0], 350 - 100 * x / 1e-6, atol=1e-8)   # T linear in x
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(K1 * 100 / 1e-6, rel=1e-9)
    assert res.quantities["mean(T)"][0, 0] == pytest.approx(350, rel=1e-12)


def test_two_materials_in_series():
    """Two layers: the flux is continuous; effective conductance is the series sum."""
    p = bar(hot_cold(), two_materials=True)
    res = HeatSolver(p, log=lambda s: None).run(["mean(qx)"])
    k_eff = 1 / (0.5 / K1 + 0.5 / K2)
    assert res.quantities["mean(qx)"][0, 0] == pytest.approx(k_eff * 100 / 1e-6, rel=1e-8)
    x, T = res.meshes[0].points[:, 0], res.T[0][0]
    T_mid = 400 - 100 * (0.5 / K1) / (0.5 / K1 + 0.5 / K2)       # interface temperature
    np.testing.assert_allclose(T[np.abs(x) < 1e-12], T_mid, rtol=1e-9)


def test_convection_and_heat_source():
    # fixed T at xmin, convection at xmax: T(L) = T_ext + (T0 - T_ext) / (1 + h L / k)
    h, L = 5e7, 1e-6
    conds = [HeatCondition("temperature", ["xmin"], "400[K]"),
             HeatCondition("convection", ["xmax"], h=f"{h}[W/(m^2*K)]", T_ext="300[K]")]
    res = HeatSolver(bar(conds), log=lambda s: None).run([])
    x, T = res.meshes[0].points[:, 0], res.T[0][0]
    T_L = 300 + 100 / (1 + h * L / K1)
    np.testing.assert_allclose(T[np.isclose(x, L / 2)], T_L, rtol=1e-9)
    # uniform source Q with both ends at 300 K: parabola, peak Q L^2 / (8 k) above the ends
    Q = 1e16
    p = bar([HeatCondition("temperature", ["xmin", "xmax"], "300[K]")],
            sources=[HeatSource("A", f"{Q}[W/m^3]")])
    res = HeatSolver(p, log=lambda s: None).run([])
    assert res.T[0][0].max() == pytest.approx(300 + Q * L ** 2 / (8 * K1), rel=1e-6)


def test_transient_decay_of_sine_mode():
    """T = 300 + 10 sin(pi (x + L/2) / L), ends at 300 K: amplitude decays as exp(-pi^2 alpha t / L^2)."""
    L = 1e-6
    alpha = K1 / (RHO * CP)
    tau = L ** 2 / (np.pi ** 2 * alpha)
    p = bar([HeatCondition("temperature", ["xmin", "xmax"], "300[K]")],
            problem=HeatProblem("transient", t_end=f"{tau}[s]", steps=400, theta=0.5, save_every=40))
    p.heat.initial = "300[K] + 10*sin(pi*(x + L/2)/L)"
    res = HeatSolver(p, log=lambda s: None).run(["max(T) - 300"])
    amp = res.quantities["max(T) - 300"][0]
    np.testing.assert_allclose(amp, 10 * np.exp(-res.times / tau), rtol=2e-3)
    assert res.times[-1] == pytest.approx(tau)


def test_heat_model_file_roundtrip(tmp_path):
    p = bar(hot_cold(), sources=[HeatSource("A", "1e10[W/m^3]", "src")],
            problem=HeatProblem("transient", t_end="2[ns]", steps=10))
    q = Project.load(p.save(tmp_path / "heat.toml"))
    assert q.physics_kind == "heat" and q.heat == p.heat and q.validate() == []


def test_convection_template_matches_lumped_estimate():
    """High-conductivity plate, uniform convection: the mean temperature rise is close to P / (h A)."""
    from solidocean import templates
    p = templates.create("heat_convection")
    assert p.validate() == []
    p.problem.sweep.values = "[10, 100] * 1[W/(m^2*K)]"
    p.mesh.max_size = "L/10"
    res = HeatSolver(p, log=lambda s: None).run(["(mean(T) - T_ext) / (P / (h*A))"])
    ratio = res.quantities["(mean(T) - T_ext) / (P / (h*A))"][:, 0]
    np.testing.assert_allclose(ratio, 1, atol=0.01)


def slab_with_pad(conditions, problem, sources=()):
    """Slab 2 x 2 x 0.2 um (material A) with a cylindrical pad (material B, r = 0.5 um, h = 0.1 um) on top."""
    return Project(
        name="pad", physics_kind="heat", variables={"w": "2[um]", "t": "0.2[um]", "R": "0.5[um]", "h": "0.1[um]"},
        materials=[mat("A", K1), mat("B", K2)],
        objects=[Feature("block", "slab", material="A", params={"center": ["0", "0", "0"], "size": ["w", "w", "t"]}),
                  Feature("cylinder", "pad", material="B",
                          params={"center": ["0", "0", "t/2"], "axis": ["0", "0", "1"], "radius": "R", "height": "h"})],
        mesh=MeshSettings(max_size="w/6"),
        heat=HeatPhysics(initial="300[K]", conditions=list(conditions), sources=list(sources), problem=problem))


def deposited_energy(p, results):
    """Heat stored in the (insulated) structure at the end: int rho cp (T - T0)."""
    from solidocean.physics.heat.solver import compute_quantity as compute_heat_quantity
    vals = compute_heat_quantity(results, p, "integral(T - 300[K])")
    return RHO * CP * vals[0]


@pytest.mark.parametrize("shape, factor", [("rect", 1.0), ("gauss", np.sqrt(np.pi / (4 * np.log(2))))])
def test_pulse_on_object_face_conserves_energy(shape, factor):
    """Flux pulse on the top of the pad only, everything insulated: stored heat = q * pad area * pulse
    energy, for a time step that does not resolve the pulse edges."""
    from solidocean.physics.heat.model import Pulse
    q, dur = 1e9, 3e-9
    cond = HeatCondition("flux", ["zmax"], f"{q}[W/m^2]", name="laser", on="pad",
                         pulse=Pulse(shape, "4[ns]", f"{dur}[s]"))
    p = slab_with_pad([cond], HeatProblem("transient", t_end="20[ns]", steps=7))
    assert p.validate() == []
    r = HeatSolver(p, log=lambda m: None).run()
    E = deposited_energy(p, r)
    area = np.pi * 0.5e-6 ** 2
    assert E[-1] == pytest.approx(q * area * dur * factor, rel=1e-3)   # all of the pulse
    assert E[0] == pytest.approx(0, abs=1e-20)
    assert np.all(np.diff(E) >= -1e-9 * E[-1])


def test_repeated_pulse_and_object_mean():
    from solidocean.physics.heat.solver import compute_quantity as compute_heat_quantity
    from solidocean.physics.heat.model import Pulse
    cond = HeatCondition("flux", ["zmax"], "1e9[W/m^2]", on="pad", pulse=Pulse("rect", "0[s]", "1[ns]", "5[ns]"))
    p = slab_with_pad([cond], HeatProblem("transient", t_end="12[ns]", steps=24))
    r = HeatSolver(p, log=lambda m: None).run(["mean(T, \"pad\")", "mean(T, \"A\")", "mean(T)"])
    E = deposited_energy(p, r)
    assert E[-1] == pytest.approx(1e9 * np.pi * 0.5e-6 ** 2 * 3e-9, rel=1e-3)   # pulses at 0, 5, 10 ns
    pad, slab, whole = (r.values(q)[0, -1] for q in ("mean(T, \"pad\")", "mean(T, \"A\")", "mean(T)"))
    assert pad > whole > slab > 300                                            # heated from the pad
    v_pad, v_slab = (compute_heat_quantity(r, p, f"integral(1, \"{o}\")")[0, 0] for o in ("pad", "slab"))
    assert v_pad == pytest.approx(np.pi * 0.5e-6 ** 2 * 0.1e-6, rel=0.02)
    assert whole == pytest.approx((pad * v_pad + slab * v_slab) / (v_pad + v_slab), rel=1e-9)
    with pytest.raises(Exception, match="unknown object"):
        compute_heat_quantity(r, p, "mean(T, \"nothing\")")


@pytest.mark.parametrize("theta, steps", [(1.0, 13), (0.5, 40)])
def test_oscillating_source_stores_its_energy(theta, steps):
    """Insulated bar with Q(t) = Q0 + Q1 sin(2 pi (t - t0) / P): rho cp dT/dt integrates Q exactly, whatever the
    time step (the source is averaged over each step), so the mean temperature follows
    T0 + [Q0 t + Q1 P / 2 pi (cos(2 pi t0 / P) - cos(2 pi (t - t0) / P))] / (rho cp)."""
    from solidocean.physics.heat.model import Pulse
    Q0, Q1, P, t0 = 1e15, 2e15, 4e-9, 1e-9
    src = HeatSource("A", f"{Q0:g}[W/m^3]", pulse=Pulse("sine", start=f"{t0:g}[s]", period=f"{P:g}[s]",
                                                     amplitude=f"{Q1:g}[W/m^3]"))
    p = bar([], sources=[src], problem=HeatProblem("transient", t_end="10[ns]", steps=steps, theta=theta))
    assert p.validate() == []
    r = HeatSolver(p, log=lambda m: None).run(["mean(T)"])
    t = r.times
    exact = 300 + (Q0 * t + Q1 * P / (2 * np.pi) * (np.cos(2 * np.pi * -t0 / P) - np.cos(2 * np.pi * (t - t0) / P))) / (RHO * CP)
    np.testing.assert_allclose(r.quantities["mean(T)"][0], exact, rtol=1e-12)
    assert np.ptp(np.diff(r.quantities["mean(T)"][0])) > 0          # not a constant heating rate


def test_oscillation_input_and_checks(tmp_path):
    from solidocean.physics.heat.model import Pulse
    flux = HeatCondition("flux", ["xmin"], "1e6[W/m^2]",
                         pulse=Pulse("sine", period="1/(10[MHz])", amplitude="5e5[W/m^2]"))
    p = bar([flux], problem=HeatProblem("transient", t_end="1[us]"))
    q = Project.load(p.save(tmp_path / "m.toml"))
    pl = q.heat.conditions[0].pulse
    assert (pl.shape, pl.period, pl.amplitude, pl.duration) == ("sine", "1/(10[MHz])", "5e5[W/m^2]", "10[ns]")
    assert "duration" not in (tmp_path / "m.toml").read_text(encoding="utf-8")
    assert q.validate() == []
    pl.amplitude, pl.period = "", "-1[s]"
    errors = " ".join(q.validate())
    assert "period and an amplitude" in errors and "period must be positive" in errors
    q.heat.problem.type = "stationary"
    assert any("oscillation needs a transient or frequency-domain problem" in e for e in q.validate())


def test_time_dependent_boundary_temperature():
    """T(xmin) ramps linearly in time; with a large diffusivity the bar follows quasi-statically."""
    p = bar([HeatCondition("temperature", ["xmin"], "300[K] + 1e9[K/s]*time"),
             HeatCondition("temperature", ["xmax"], "300[K]")],
            problem=HeatProblem("transient", t_end="10[ns]", steps=10))
    assert p.validate() == []
    r = HeatSolver(p, log=lambda m: None).run()
    np.testing.assert_allclose(r.T[0].max(axis=1), 300 + 1e9 * r.times, rtol=1e-12)   # the hot end, nodal


def test_pulse_needs_transient_problem():
    from solidocean.physics.heat.model import Pulse
    p = bar(hot_cold() + [HeatCondition("flux", ["ymax"], "1[W/m^2]", pulse=Pulse("rect"))])
    assert any("transient" in e for e in p.validate())


def test_pad_template(tmp_path):
    from pathlib import Path
    from solidocean import templates
    p = Project.load(templates.create("heat_pad").save(tmp_path / "pad.toml"))
    assert p.validate() == []
    flux = p.heat.conditions[1]
    assert (flux.on, flux.faces, flux.pulse.shape) == ("Pad", ["zmax"], "rect")
    q = Project.load(p.save(Path(__import__("tempfile").mkdtemp()) / "x.toml"))   # round trip keeps the pulse
    assert q.heat.conditions[1].pulse == flux.pulse and q.heat.conditions[1].on == "Pad"


def test_conductivity_sweep_reuses_mesh():
    """Sweeping a variable used in a material property: same mesh for every value, and each value gives
    the same result as a separate run."""
    from solidocean.project import VariableSweep
    p = bar(hot_cold(), problem=HeatProblem())
    p.variables["kA"] = "100[W/(m*K)]"
    p.materials[0].props["k"] = "kA"
    p.heat.sources = [HeatSource("A", "1e15[W/m^3]")]
    p.problem.sweep = VariableSweep("kA", "[50, 100] * 1[W/(m*K)]")
    logs = []
    r = HeatSolver(p, log=logs.append).run(["max(T)"])
    assert sum(m.startswith("Mesh:") for m in logs) == 1
    assert r.meshes[0] is r.meshes[1]
    single = HeatSolver(p, log=lambda m: None)
    p.problem.sweep.enabled = False
    p.variables["kA"] = "50[W/(m*K)]"
    assert single.run(["max(T)"]).values("max(T)")[0, 0] == pytest.approx(r.values("max(T)")[0, 0], rel=1e-3)


def test_negative_start_time():
    """Transient from t = -10 ns with a pulse at t = 0: times start at t_start, nothing happens before the
    pulse, and the full pulse energy is deposited."""
    from solidocean.physics.heat.model import Pulse
    cond = HeatCondition("flux", ["zmax"], "1e9[W/m^2]", on="pad", pulse=Pulse("rect", "0[s]", "2[ns]"))
    p = slab_with_pad([cond], HeatProblem("transient", t_start="-10[ns]", t_end="10[ns]", steps=20))
    assert p.validate() == []
    r = HeatSolver(p, log=lambda m: None).run()
    np.testing.assert_allclose(r.times[[0, 10, -1]], [-10e-9, 0.0, 10e-9], atol=1e-18)
    E = deposited_energy(p, r)
    assert np.all(np.abs(E[:11]) < 1e-9 * E[-1])
    assert E[-1] == pytest.approx(1e9 * np.pi * 0.5e-6 ** 2 * 2e-9, rel=1e-3)
    p.heat.problem.t_end = "-20[ns]"
    assert any("later than t_start" in e for e in p.validate())


def test_geometry_builds_without_heat_conditions():
    """Building the geometry/mesh needs no physics: a heat model without any condition is fine for that."""
    p = bar([], problem=HeatProblem())
    p.heat.conditions = [HeatCondition("flux", ["nowhere"], "1[W/m^2]")]   # a broken condition
    assert p.validate_geometry() == []
    assert any("bad face" in e for e in p.validate())


def test_stationary_without_heat_sink_conserves_energy():
    """All faces insulated: the steady state keeps the initial heat content (capacity-weighted mean of the
    initial temperature); with equal heat in and out the profile is linear around that level; a net input
    has no steady state."""
    p = bar([], problem=HeatProblem(), two_materials=True)
    p.heat.initial = "350[K] + 100[K]*x/L"               # 300 K .. 400 K along the bar, mean 350 K
    assert p.validate() == []
    T = HeatSolver(p, log=lambda m: None).run().T[0][0]
    np.testing.assert_allclose(T, 350.0, rtol=1e-6)
    q = 1e6
    p.heat.initial = "300[K]"
    p.heat.conditions = [HeatCondition("flux", ["xmin"], f"{q}[W/m^2]"), HeatCondition("flux", ["xmax"], f"{-q}[W/m^2]")]
    p.materials[1] = mat("B", K1)                          # one conductivity: linear profile, slope -q/k
    r = HeatSolver(p, log=lambda m: None).run(["mean(T)", "mean(Tx)"])
    assert r.values("mean(T)")[0, 0] == pytest.approx(300.0, rel=1e-9)
    assert r.values("mean(Tx)")[0, 0] == pytest.approx(-q / K1, rel=1e-6)
    p.heat.conditions = p.heat.conditions[:1]              # heat in, nothing out
    with pytest.raises(ValueError, match="no steady state"):
        HeatSolver(p, log=lambda m: None).run()


def test_line_plot_profile():
    """Line plot through a hot/cold bar: exact linear profile and flux; points outside give NaN; the plot
    definition survives the file round trip; the CLI writes the profile."""
    from solidocean.results.field import line_profile, save_line_table
    p = bar(hot_cold())
    pl = ResultPlot("profile", "T", kind="line", start=["-L/2", "0", "0"], end=["L/2 + 0.1[um]", "0", "0"],
                    points=23, line_axis="x")
    p.plots.append(pl)
    assert p.validate() == []
    r = HeatSolver(p, log=lambda m: None).run()
    xs, vals, j, label = line_profile(r, p, pl)
    xs = xs[0]                                    # x values of the first (only) variable value
    inside = xs <= 0.5e-6 + 1e-15
    np.testing.assert_allclose(vals[0, inside], 350 - 100 * xs[inside] / 1e-6, atol=1e-8)
    assert np.all(np.isnan(vals[0, ~inside])) and label.startswith("x (")
    pl.quantity = "qx"
    _, q, _, _ = line_profile(r, p, pl)
    np.testing.assert_allclose(q[0, inside], K1 * 100 / 1e-6, rtol=1e-8)
    pl.quantity = "mean(T)"
    with pytest.raises(Exception, match="point by point"):
        line_profile(r, p, pl)
    pl.quantity = "T"
    q = Project.from_dict(__import__("tomllib").loads(__import__("tomli_w").dumps(p.to_dict())))
    assert q.plots[-1] == pl


def test_line_plot_validation():
    p = bar(hot_cold())
    p.plots.append(ResultPlot("bad", "T", kind="line", start=["0", "0", "0"], end=["0", "0", "0"]))
    assert any("same point" in e for e in p.validate())
