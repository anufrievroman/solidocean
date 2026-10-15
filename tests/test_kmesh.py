"""Wavevector meshes of the Brillouin zone (coherent heat conduction): the mesh itself and the spectrum it
computes, checked against the exact Rayleigh-Lamb solution of a free plate."""
import sys
from pathlib import Path

import numpy as np
import pytest

from solidocean import lattice as lat
from solidocean import physics
from solidocean.expressions import Evaluator
from solidocean.lattice import Lattice
from solidocean.physics.solid.model import KMesh, SolidPhysics, SolidProblem
from solidocean.project import Feature, Material, MeshSettings, Project

SQ = (np.array([2 * np.pi, 0.0, 0.0]), np.array([0.0, 2 * np.pi, 0.0]))
HEX_CELL = [["a", "0", "0"], ["a/2", "sqrt(3)*a/2", "0"]]
NU = 0.3            # the plate of validation/lamb_plate.py
RHO, E = 2700.0, 70e9


def hex_reciprocal():
    return [b for b in lat.reciprocal(lat.slots(Lattice(vectors=HEX_CELL), Evaluator({"a": "1"}))) if b is not None]


def area(points, tris) -> float:
    """Total area of a triangulation, in the (reduced) coordinates of its points."""
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    u, v = b - a, c - a
    return float(0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum())


@pytest.mark.parametrize("region, fraction", [("irreducible", 1 / 8), ("full", 1.0)])
def test_square_zone_mesh_covers_its_region(region, fraction):
    """A square lattice: the wedge is an eighth of the zone, which itself has unit area in reduced
    coordinates. The mesh is conforming (no point repeated) and starts at Gamma."""
    points, tris = lat.zone_mesh(*SQ, 8, region, "square")
    assert area(points, tris) == pytest.approx(fraction)
    assert len(points) == len(np.unique(np.round(points, 9), axis=0))
    np.testing.assert_allclose(points[0], [0, 0], atol=1e-15)
    assert tris.max() == len(points) - 1


def test_wedge_point_count_follows_the_divisions():
    """A triangular wedge divided n times per edge: (n + 1)(n + 2) / 2 points and n**2 triangles (the
    113 points per edge of a published calculation give its ~6,400 k points)."""
    for n in (1, 3, 12, 112):
        points, tris = lat.zone_mesh(*SQ, n, "irreducible", "square")
        assert len(points) == (n + 1) * (n + 2) // 2
        assert len(tris) == n * n
    assert len(lat.zone_mesh(*SQ, 112, "irreducible", "square")[0]) == 6441


def test_hexagonal_wedge_sits_on_the_zone():
    """The corners of the hexagonal wedge are Gamma, the zone edge centre M and the zone corner K, and the
    wedge is a twelfth of the zone."""
    b1, b2 = hex_reciprocal()
    points, tris = lat.zone_mesh(b1, b2, 6, "irreducible", "hexagonal")
    assert area(points, tris) == pytest.approx(1 / 12)
    cartesian = points[:, :1] * b1 + points[:, 1:] * b2
    radius = np.linalg.norm(cartesian, axis=1)
    corner = lat.brillouin_zone_2d(b1, b2)
    assert radius.max() == pytest.approx(np.linalg.norm(corner, axis=1).max())          # K
    assert np.linalg.norm(cartesian[points[:, 1] == 0], axis=1).max() == pytest.approx(np.pi / np.sqrt(3) * 2)  # M
    assert area(*lat.zone_mesh(b1, b2, 6, "full", "hexagonal")) == pytest.approx(1.0)


def test_zone_mesh_rejects_a_lattice_without_a_wedge():
    with pytest.raises(ValueError, match="no standard irreducible wedge"):
        lat.zone_mesh(*SQ, 4, "irreducible", "oblique")


def plate(n: int = 2, a_frac: float = 0.5, h: str = "t/5", num_eigs: int = 12) -> Project:
    """Free isotropic plate with an artificial square cell, meshed over the irreducible wedge of its zone."""
    return Project(
        name="Plate over the zone", physics_kind="solid",
        variables={"t": "1[um]", "a": f"{a_frac}*t"},
        materials=[Material("M", {"model": "isotropic", "rho": f"{RHO}[kg/m^3]", "E": f"{E / 1e9}[GPa]",
                                  "nu": str(NU)})],
        objects=[Feature("block", "Plate", material="M", params={"center": ["0", "0", "0"],
                                                                 "size": ["a", "a", "t"]})],
        mesh=MeshSettings(max_size=h, order=2),
        lattice=Lattice(periodic={"x": "a", "y": "a"}),
        solid=SolidPhysics(problem=SolidProblem(type="conductance", num_eigs=num_eigs, kmesh=KMesh(n=n))))


def lamb_reference():
    """validation/lamb_plate.lamb_frequencies: the exact Rayleigh-Lamb and shear-horizontal spectrum."""
    directory = str(Path(__file__).resolve().parent.parent / "validation")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from lamb_plate import lamb_frequencies
    return lamb_frequencies


def test_plate_over_the_zone_matches_lamb_theory():
    """Every eigenfrequency computed on the wedge of a homogeneous plate is a guided mode of that plate.

    The cell is artificial, so the spectrum at k is the exact one at every |k + G| folded into the zone; the
    whole set must be accounted for. Since the plate is isotropic the exact frequencies depend on |k + G|
    alone, so this also checks that the zone is sampled consistently in every direction (Gamma-X rotated
    around the zone).
    """
    lamb_frequencies = lamb_reference()
    p = plate()
    assert p.validate() == []
    results = physics.of(p).solver(p, log=lambda m: None).run()
    assert results.is_kmesh and results.ktris.shape == (4, 3) and len(results.kred) == 6

    c_t = np.sqrt(E / (2 * (1 + NU)) / RHO)
    thickness, b = 1e-6, 2 * np.pi / (0.5e-6)
    window = 10.0                       # compared range of omega t / c_T
    compared, worst = 0, 0.0
    for j, k in enumerate(results.k[0]):
        exact = set()
        for i in range(-2, 3):
            for m in range(-2, 3):
                folded = np.linalg.norm(k + np.array([i * b, m * b, 0.0])) * thickness
                exact.update(lamb_frequencies(folded, window + 1.0, NU).tolist())
        exact = np.array(sorted(exact))
        omega = 2 * np.pi * results.freqs[0, j] * thickness / c_t
        for w in omega[(omega > 1e-6) & (omega < window)]:
            compared += 1
            worst = max(worst, float(np.min(np.abs(exact - w)) / w))
    assert compared > 50                # not a vacuous comparison
    assert worst < 0.01                 # discretisation error of t/5 quadratic elements


def test_kmesh_round_trip(tmp_path):
    """The mesh and the problem type survive the model file; a disabled mesh is kept but not run."""
    p = plate(n=5)
    p.solid.problem.kmesh.region = "full"
    q = Project.load(p.save(tmp_path / "zone.toml"))
    assert q.solid.problem.type == "conductance"
    assert q.solid.problem.kmesh == KMesh(n=5, region="full")
    assert q.solid.problem.active_kmesh is q.solid.problem.kmesh
    q.solid.problem.kmesh.enabled = False
    r = Project.load(q.save(tmp_path / "off.toml"))
    assert r.solid.problem.kmesh.enabled is False and r.solid.problem.active_kmesh is None


def test_spectrum_and_conductance_plots_round_trip(tmp_path):
    """The settings of the plots of a wavevector mesh survive the model file: all bands (0) rather than the
    six a band surface plot defaults to, the spectrum, the flux temperature and the conductance range."""
    from solidocean.physics.solid.plots import band_map, band_surfaces, conductance_plot, spectrum_plot
    p = plate(n=5)
    flux = spectrum_plot(spectrum="flux")
    flux.temperature, flux.bins, flux.show_incomplete = "0.1[K]", 300, True
    power = conductance_plot(transport="power")
    power.t_min, power.t_max, power.t_points = "0.02[K]", "0.15[K]", 40
    shifted = band_map()
    shifted.band = 3
    p.plots = [band_surfaces(bands=8), shifted, spectrum_plot(spectrum="velocity"), flux, power]
    q = Project.load(p.save(tmp_path / "plots.toml"))
    for before, after in zip(p.plots, q.plots):
        assert after == before


def test_a_mesh_belongs_to_a_coherent_problem_and_a_plane_lattice():
    """The mesh runs only for a coherent heat conduction problem, which in turn needs one."""
    p = plate()
    p.solid.problem.type = "eigenfrequency"
    assert p.solid.problem.active_kmesh is None            # an ordinary eigenfrequency run follows its k-path
    p.solid.problem.type, p.solid.problem.kmesh = "conductance", None
    assert any("needs a wavevector mesh" in e for e in p.validate())
    p = plate()
    p.lattice = Lattice(periodic={"x": "a"})
    assert any("periodic in exactly two directions" in e for e in p.validate())
    p = plate()
    p.lattice = Lattice(vectors=[["a", "0", "0"], ["a/3", "a", "0"]])
    assert any("no standard irreducible wedge" in e for e in p.validate())
    p.solid.problem.kmesh.region = "full"
    assert p.validate() == []


def test_the_mesh_has_a_panel_and_a_node_in_the_window():
    """Switching to coherent heat conduction gives the model tree a wavevector mesh node with its panel, and
    the project a band map; the mesh panel reports how many wavevectors the run will solve."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    from solidocean.gui.panels.solid import set_problem_type

    w = MainWindow()
    w.set_project(templates.create("square"))
    set_problem_type(w, "conductance")
    app.processEvents()
    assert w.project.solid.problem.active_kmesh is not None
    assert [pl.kind for pl in w.project.plots].count("kmap") == 1
    w.select_node("kmesh")
    app.processEvents()
    texts = " ".join(lbl.text() for lbl in w.settings.widget().findChildren(QLabel))
    assert "91 wavevectors" in texts          # the default 12 divisions of a triangular wedge
    w.dirty = False
    w.close()


def mesh_results():
    """A small run over the zone, reused by the plot tests."""
    p = plate(n=2, num_eigs=6)
    return p, physics.of(p).solver(p, log=lambda m: None).run()


def test_band_map_and_surfaces_draw_from_the_mesh():
    """Both zone plots draw: the map in 2D with a pickable point per wavevector, the surfaces in a 3D axes
    with nothing to pick, honouring the band range asked for."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solidocean.physics.solid.plots import band_map, band_surfaces
    from solidocean.results.modal import draw

    p, r = mesh_results()
    fig = plt.figure()
    ax = fig.add_subplot()
    table = draw(r, ax, p, band_map(band=1))
    assert table.shape == (len(r.kred), 5)                 # one pickable point per wavevector
    assert ax.get_aspect() == 1.0 and ax.collections       # filled contours, equal aspect

    ax3 = fig.add_subplot(projection="3d")
    assert draw(r, ax3, p, band_surfaces(bands=3)).shape == (0, 5)    # a 3D view has nothing to pick
    assert len(ax3.collections) == 3                       # one surface per band drawn
    ax3.clear()
    draw(r, ax3, p, band_surfaces(band=0, bands=0))         # 0 = every band
    assert len(ax3.collections) == r.freqs.shape[2]
    plt.close(fig)


def test_zone_plots_need_a_mesh_and_get_3d_axes():
    """The plots are offered only with a wavevector mesh, and the canvas puts the surfaces in a 3D axes
    while the map stays in a flat one."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from solidocean.gui.plot_canvas import PlotCanvas
    from solidocean.physics import of
    from solidocean.physics.solid.plots import band_map, band_surfaces

    p = plate()
    spec = of(p)
    assert spec.plot_unavailable(p, band_map()) == ""
    p.solid.problem.type = "eigenfrequency"
    assert "Band maps need" in spec.plot_unavailable(p, band_map())
    assert "Band surfaces need" in spec.plot_unavailable(p, band_surfaces())

    project, results = mesh_results()
    canvas = PlotCanvas()
    canvas.show_results(results, project, band_surfaces(bands=2))
    app.processEvents()
    assert canvas.ax.name == "3d" and len(canvas.table) == 0
    canvas.show_results(results, project, band_map())
    app.processEvents()
    assert canvas.ax.name == "rectilinear" and len(canvas.table) == len(results.kred)
    canvas.deleteLater()


def test_zone_plot_ranges_apply_to_the_right_axes():
    """The quantity range is the vertical axis of the surfaces and the colour scale of a map; kx and ky have
    a range each, in the displayed wavevector unit unless the expression carries one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solidocean.physics.solid.plots import band_map, band_surfaces
    from solidocean.results.modal import draw

    p, r = mesh_results()
    top = r.freqs[0].max() / 2e9                       # GHz, half way up the computed range
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    pl = band_surfaces(bands=3)
    pl.style.y_max, pl.style.x_min, pl.style.x_max = str(top), "0", "4e6[1/m]"
    pl.style.ky_min, pl.style.ky_max = "-1", "3"
    draw(r, ax, p, pl)
    assert ax.get_zlim() == pytest.approx((0.0, top))
    assert ax.get_xlim() == pytest.approx((0.0, 4.0))   # 4e6 rad/m in the default 1/um
    assert ax.get_ylim() == pytest.approx((-1.0, 3.0))

    ax2 = fig.add_subplot()
    pm = band_map(band=1)
    pm.style.y_min, pm.style.y_max = "1", "9"
    draw(r, ax2, p, pm)
    assert ax2.collections[0].get_clim() == pytest.approx((1.0, 9.0))
    plt.close(fig)


def test_zone_plots_offer_only_the_style_that_applies():
    """The panel drops the rows that do nothing for these plots (lines, single colours, legends; points and
    the colour bar belong to the map alone) and gives both a ky range."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QCheckBox, QLabel
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    from solidocean.gui.panels.solid import set_problem_type

    w = MainWindow()
    w.set_project(templates.create("square"))
    set_problem_type(w, "conductance")
    app.processEvents()
    rows = {}
    for kind in ("surfaces", "kmap"):
        index = next(i for i, pl in enumerate(w.project.plots) if pl.kind == kind)
        w.select_node("plot", index)
        app.processEvents()
        panel = w.settings.widget()
        rows[kind] = (" ".join(lbl.text() for lbl in panel.findChildren(QLabel)),
                      {b.text() for b in panel.findChildren(QCheckBox)})
    for kind in ("surfaces", "kmap"):
        text, checks = rows[kind]
        assert "ky range" in text and "kx range" in text
        assert "Lines" not in text and "Colours" not in text
        assert "Legend" not in checks and "Log" not in checks
    assert "Points" not in rows["surfaces"][0] and "Colour bar" not in rows["surfaces"][1]
    assert "Wavevectors" in rows["kmap"][0] and "Colour bar" in rows["kmap"][1]
    w.dirty = False
    w.close()


# ---- spectra of the mesh ---------------------------------------------------------------

def cone_results(n: int = 40, region: str = "full", velocity: float = 5000.0, period: float = 1e-6,
                 branches=(1.0,), offsets=None):
    """A synthetic mesh carrying isotropic linear bands f = c v |k| / 2pi, whose spectra are exact.
    `offsets` lifts a branch off zero, which a band has to be for any range to count as complete."""
    from solidocean.physics.solid.modal import ModalResults
    b1 = np.array([2 * np.pi / period, 0.0, 0.0])
    b2 = np.array([0.0, 2 * np.pi / period, 0.0])
    reduced, tris = lat.zone_mesh(b1, b2, n, region, "square")
    k = reduced[:, :1] * b1 + reduced[:, 1:] * b2
    base = velocity * np.linalg.norm(k, axis=1) / (2 * np.pi)
    lifts = offsets if offsets is not None else [0.0] * len(branches)
    freqs = np.column_stack([base / c + lift for c, lift in zip(branches, lifts)])
    return ModalResults(freqs[None], k[None], np.arange(len(k), dtype=float), "k point",
                        ktris=tris, kred=reduced)


def test_spectra_of_a_linear_band_match_the_analytic_ones():
    """For one isotropic branch ω = v k the spectra are known in closed form: D(f) = 2πf/v², the average
    group velocity is v, and the radiated flux is 2 h f² n(f, T) / v. Compared inside the circle that fits
    in the zone, away from the badly resolved cone tip."""
    from solidocean.physics.solid import spectra
    v, a, temperature = 5000.0, 1e-6, 1.0
    r = cone_results(n=60, velocity=v, period=a)
    top = v / (2 * a)                       # the contour is a full circle below this
    for kind, exact in (("dos", lambda f: 2 * np.pi * f / v ** 2),
                        ("velocity", lambda f: np.full_like(f, v)),
                        ("flux", lambda f: 2 * spectra.H_PLANCK * f ** 2
                         * spectra.bose_einstein(f, temperature) / v)):
        freq, value = spectra.spectrum(r, kind, bins=400, temperature=temperature, incomplete=True)
        band = (freq > 0.3e9) & (freq < top)
        assert np.max(np.abs(value[band] - exact(freq)[band]) / exact(freq)[band]) < 0.02, kind


def test_the_density_of_states_counts_every_state_once():
    """Integrating the density of states over complete bands gives one state per band per unit cell area —
    the sum rule that ties together the contour lengths, the gradients and the symmetry factor of a wedge.
    """
    from solidocean.physics.solid import spectra
    a = 1e-6
    for region in ("full", "irreducible"):
        r = cone_results(n=60, region=region, period=a, branches=(1.0, 1.7))
        freq, dos = spectra.spectrum(r, "dos", bins=6000, incomplete=True)
        assert np.trapezoid(dos, freq) * a ** 2 == pytest.approx(2.0, rel=2e-3)


def test_a_wedge_and_the_whole_zone_give_the_same_spectra():
    """The symmetry factor is what makes a mesh of the wedge stand for the whole zone: two meshes that
    share nothing but the lattice must give the same spectrum.

    Only above the lowest frequencies. A branch is a cone at its apex, which no triangulation resolves —
    flat patches over a corner stay equally wrong however fine the mesh — so the two meshes disagree there
    by an amount that does not converge, and real acoustic branches carry the same error at low frequency.
    """
    from solidocean.physics.solid import spectra
    freq, whole = spectra.spectrum(cone_results(n=48, region="full"), "dos", bins=200, incomplete=True)
    same, wedge = spectra.spectrum(cone_results(n=48, region="irreducible"), "dos", bins=200, incomplete=True)
    np.testing.assert_allclose(freq, same)
    band = (freq > 0.3e9) & (freq < 5000.0 / (2 * 1e-6))
    np.testing.assert_allclose(whole[band], wedge[band], rtol=0.02)


def test_heat_flux_follows_the_occupation():
    """The flux is the contour sum weighted by h f n(f, T): no states are occupied at absolute zero, and
    raising the temperature can only add to it."""
    from solidocean.physics.solid import spectra
    r = cone_results(n=30)
    freq, cold = spectra.spectrum(r, "flux", bins=120, temperature=0.0, incomplete=True)
    _, warm = spectra.spectrum(r, "flux", bins=120, temperature=0.2, incomplete=True)
    _, hot = spectra.spectrum(r, "flux", bins=120, temperature=2.0, incomplete=True)
    assert np.all(cold == 0)
    assert np.all(hot >= warm) and hot.sum() > 10 * warm.sum()
    ratio = hot / np.where(warm > 0, warm, np.inf)
    assert np.all(ratio[warm > 0] > 1)


def test_spectrum_plot_draws_and_is_offered_only_with_a_mesh():
    """The plot draws a curve per spectrum, picks nothing, and needs a wavevector mesh."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solidocean.physics import of
    from solidocean.physics.solid.plots import spectrum_plot
    from solidocean.results.modal import draw

    p, r = mesh_results()
    fig, ax = plt.subplots()
    for kind in ("dos", "velocity", "flux"):
        ax.clear()
        pl = spectrum_plot(spectrum=kind)
        pl.temperature = "0.5[K]"
        assert draw(r, ax, p, pl).shape == (0, 5)
        assert len(ax.lines) == 1 and np.all(np.isfinite(ax.lines[0].get_ydata()))
    assert "Frequency" in ax.get_xlabel()
    plt.close(fig)
    assert of(p).plot_unavailable(p, spectrum_plot()) == ""
    p.solid.problem.type = "eigenfrequency"
    assert "Spectra need" in of(p).plot_unavailable(p, spectrum_plot())


# ---- the conductance -------------------------------------------------------------------

def test_power_and_conductance_match_the_analytic_low_temperature_laws():
    """One isotropic branch in two dimensions radiates P = 4 zeta(3) kB^3 T^3 / (v h^2), so the ballistic
    conductance G = dP/dT goes as T^2. Checked where the computed bands still reach far above the occupied
    frequencies — the integral stops at the highest band, and below that it is a lower bound."""
    from scipy.special import zeta

    from solidocean.physics.solid import spectra
    v = 5000.0
    r = cone_results(n=60, region="full", velocity=v, period=1e-6)
    kb, h = spectra.K_BOLTZMANN, spectra.H_PLANCK
    cold = np.array([0.005, 0.01])
    assert spectra.thermal_frequency(cold[-1]) * 9 < r.freqs.max()      # the bands reach far enough
    for kind, exact in (("power", lambda t: 4 * zeta(3) * kb ** 3 * t ** 3 / (v * h ** 2)),
                        ("conductance", lambda t: 12 * zeta(3) * kb ** 3 * t ** 2 / (v * h ** 2))):
        _, value = spectra.transport(r, kind, cold, subdivisions=16, incomplete=True)
        np.testing.assert_allclose(value, exact(cold), rtol=0.02)


def test_the_conductance_is_the_derivative_of_the_power():
    """G(T) = dP/dT, the two computed from the same contour sums by different occupation factors."""
    from solidocean.physics.solid import spectra
    r = cone_results(n=40)
    t = np.array([0.0099, 0.0101])
    _, power = spectra.transport(r, "power", t, incomplete=True)
    _, conductance = spectra.transport(r, "conductance", np.array([0.01]), incomplete=True)
    assert (power[1] - power[0]) / (t[1] - t[0]) == pytest.approx(conductance[0], rel=1e-3)


def test_the_conductance_over_frequency_equals_the_one_over_the_zone():
    """The k-space integral of the papers and the frequency integral over contour lengths are one integral
    after a change of variable: with the same planar bands they agree once both quadratures are fine, and
    the k-space one matches the analytic law as the frequency one does."""
    from scipy.special import zeta

    from solidocean.physics.solid import spectra
    v = 5000.0
    r = cone_results(n=60, region="full", velocity=v, period=1e-6)
    t = np.array([0.005, 0.01])
    _, over_f = spectra.transport_over_frequency(r, "conductance", t, bins=5000, incomplete=True)
    _, over_k = spectra.transport(r, "conductance", t, subdivisions=16, incomplete=True)
    np.testing.assert_allclose(over_k, over_f, rtol=1e-3)
    kb, h = spectra.K_BOLTZMANN, spectra.H_PLANCK
    np.testing.assert_allclose(over_k, 12 * zeta(3) * kb ** 3 * t ** 2 / (v * h ** 2), rtol=0.02)
    _, power = spectra.transport(r, "power", t, subdivisions=16, incomplete=True)
    np.testing.assert_allclose(power, 4 * zeta(3) * kb ** 3 * t ** 3 / (v * h ** 2), rtol=0.02)


def test_conductance_plot_draws_over_its_temperature_range():
    """The plot computes the curve over the range it is given, rises with temperature, and picks nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solidocean.physics import of
    from solidocean.physics.solid.plots import conductance_plot
    from solidocean.results.modal import draw, temperatures_of

    p, r = mesh_results()
    fig, ax = plt.subplots()
    for kind in ("conductance", "power"):
        ax.clear()
        pl = conductance_plot(transport=kind)
        pl.t_min, pl.t_max, pl.t_points = "0.05[K]", "0.5[K]", 12
        assert draw(r, ax, p, pl).shape == (0, 5)
        x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
        assert len(x) == 12 and x[0] == pytest.approx(0.05) and x[-1] == pytest.approx(0.5)
        assert np.all(np.diff(y) > 0) and np.all(np.isfinite(y))
        assert len(temperatures_of(p, pl)) == 12
    assert ax.get_xscale() == "log" and "Temperature" in ax.get_xlabel()
    plt.close(fig)
    pl = conductance_plot()
    pl.t_min, pl.t_max = "1[K]", "0.1[K]"
    with pytest.raises(ValueError, match="positive and increasing"):
        temperatures_of(p, pl)
    assert of(p).plot_unavailable(p, conductance_plot()) == ""
    p.solid.problem.type = "eigenfrequency"
    assert "Conductance curves need" in of(p).plot_unavailable(p, conductance_plot())


def test_spectra_stop_where_the_bands_stop_being_complete():
    """A run computes a fixed number of bands, so the set is complete only below the lowest value the
    topmost band takes over the zone — not below the highest frequency it reaches somewhere."""
    from solidocean.physics.solid import spectra
    r = cone_results(n=12, region="full", branches=(1.0, 1.7), offsets=(0.0, 2e9))
    top = r.freqs[0, :, -1]
    assert spectra.complete_to(r) == pytest.approx(float(top.min()))
    assert spectra.highest(r) == pytest.approx(float(r.freqs.max()))
    assert spectra.complete_to(r) < spectra.highest(r)          # the topmost band is not flat

    levels, *_ = spectra.sampled(r, bins=40)
    assert levels[-1] == pytest.approx(spectra.complete_to(r))
    carried, *_ = spectra.sampled(r, bins=40, incomplete=True)
    assert carried[-1] == pytest.approx(spectra.highest(r))
    # the band range asked for moves the ceiling with it
    assert spectra.complete_to(r, bands=1) == pytest.approx(float(r.freqs[0, :, 0].min()))
    with pytest.raises(spectra.NoCompleteRange, match="falls to zero"):
        spectra.sampled(r, bins=10, bands=1)      # band 1 alone starts at zero: nothing is complete


def test_carrying_a_spectrum_past_the_ceiling_only_adds_a_sagging_tail():
    """Below the ceiling the two agree exactly; above it the carried curve is missing states, so it falls
    away, and the conductance picks up that unreliable tail."""
    from solidocean.physics.solid import spectra
    p, r = mesh_results()
    ceiling = spectra.complete_to(r)
    freq, dos = spectra.spectrum(r, "dos", bins=60)
    long_freq, long_dos = spectra.spectrum(r, "dos", bins=60, incomplete=True)
    shared = (freq > 0.2 * ceiling) & (freq < 0.9 * ceiling)
    np.testing.assert_allclose(np.interp(freq, long_freq, long_dos)[shared], dos[shared], rtol=0.05)
    assert long_dos[-1] < dos[-1]                                # the tail sags towards zero
    warm = np.array([0.4])
    complete = spectra.transport(r, "conductance", warm)[1][0]
    carried = spectra.transport(r, "conductance", warm, incomplete=True)[1][0]
    assert carried > complete                                    # the tail can only add


def test_the_spectrum_plot_marks_the_ceiling_when_carried_past_it():
    """The dashed line is drawn only when the curve runs beyond the complete range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solidocean.physics.solid import spectra
    from solidocean.physics.solid.plots import spectrum_plot
    from solidocean.results.modal import draw

    p, r = mesh_results()
    fig, ax = plt.subplots()
    draw(r, ax, p, spectrum_plot())
    assert not ax.lines[1:]                                      # the curve alone
    ax.clear()
    pl = spectrum_plot()
    pl.show_incomplete = True
    draw(r, ax, p, pl)
    marks = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(marks) == 1
    scale = 1e9 if p.frequency_unit == "GHz" else 1.0
    assert marks[0].get_xdata()[0] * scale == pytest.approx(spectra.complete_to(r))
    plt.close(fig)


def test_every_zone_plot_explains_its_sums():
    """Each plot over the zone carries a Formalism block, and a spectrum explains the one it draws rather
    than spectra in general; the field view has none."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from solidocean import templates
    from solidocean.gui.main_window import MainWindow
    from solidocean.gui.physics.solid import SolidUI
    from solidocean.physics.solid.plots import (band_map, band_surfaces, conductance_plot, field_plot,
                                                spectrum_plot)

    p = templates.create("zone_crystal")
    ui = SolidUI()
    for plot in (band_map(), band_surfaces(), conductance_plot()):
        assert ui.plot_formalism(p, plot), plot.kind
    assert ui.plot_formalism(p, field_plot()) == ""
    spectra_text = {kind: ui.plot_formalism(p, spectrum_plot(spectrum=kind))
                    for kind in ("dos", "velocity", "flux")}
    assert len(set(spectra_text.values())) == 3                  # one explanation each, not a shared one
    assert "Bose-Einstein" in spectra_text["flux"] and "Bose-Einstein" not in spectra_text["dos"]
    power = ui.plot_formalism(p, conductance_plot(transport="power"))
    assert power != ui.plot_formalism(p, conductance_plot(transport="conductance"))

    w = MainWindow()
    w.set_project(p)
    app.processEvents()
    index = next(n for n, pl in enumerate(p.plots) if pl.kind == "spectrum")
    w.select_node("plot", index)
    app.processEvents()
    assert w.settings.widget().formalism                          # the panel picked it up
    w.dirty = False
    w.close()
