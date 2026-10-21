"""GUI adapters of the physics: what the 3D view tints and which tree nodes each physics shows (no window
needed)."""
from solidocean import templates
from solidocean.gui import physics as gui_physics


def test_solid_highlights_periodicity_symmetry_and_conditions():
    p = templates.create("square")                       # periodic in x and y
    h = gui_physics.of(p).highlights(p)
    assert h["xmin"] == h["xmax"] == "periodic x" and h["ymin"] == "periodic y"
    p = templates.create("tensile")                      # symmetry planes + a prescribed displacement
    h = gui_physics.of(p).highlights(p)
    assert h["xmin"] == "symmetry" and h[("xmax", None, None)] == "displacement"
    p = templates.create("cantilever")                   # a fixed face and a load in a box
    h = gui_physics.of(p).highlights(p)
    assert h[("xmin", None, None)] == "fixed"
    (face, obj, box), = [k for k, v in h.items() if v == "load"]
    assert face == "zmax" and obj is None and box is not None


def test_heat_highlights_and_tree():
    p = templates.create("heat_pad")
    ui = gui_physics.of(p)
    h = ui.highlights(p)
    assert h[("xmin", None, None)] == "temperature"
    (face, obj, box), = [k for k, v in h.items() if v == "flux"]       # the top of the pad
    assert face == "zmax" and obj is not None and box is None
    kinds = [node[0] for _, node, _ in ui.physics_nodes(p)]
    assert kinds == ["heat_initial", "heat_symmetry", "heat_cond", "heat_cond"]
    q = templates.create("heat_fdtr")
    assert gui_physics.of(q).highlights(q)["ymin"] == "symmetry"


def test_invalid_variables_give_no_highlights_instead_of_an_error():
    p = templates.create("cantilever")
    p.variables["L"] = "1/("
    assert gui_physics.of(p).highlights(p) == {}


def test_bypassed_condition_is_dimmed_and_not_tinted():
    p = templates.create("heat_pad")
    p.heat.conditions[0].enabled = False
    ui = gui_physics.of(p)
    assert ("xmin", None, None) not in ui.highlights(p)
    label, node, dimmed = next(n for n in ui.physics_nodes(p) if n[1] == ("heat_cond", 0))
    assert dimmed and label.endswith("(bypassed)")


def test_flux_profile_tints_the_spot_and_refines_the_preview_around_it():
    import numpy as np
    import pyvista as pv
    from solidocean import meshing
    from solidocean.gui import viewer
    from solidocean.gui.physics.common import Tint
    p = templates.create("heat_fdtr")                    # a Gaussian pump at the origin, FWHM w = 2 um
    h = gui_physics.of(p).highlights(p)
    (key, tint), = [(k, v) for k, v in h.items() if v == "flux"]
    assert isinstance(tint, Tint) and tint.scale == 2e-6 and tint.focus == (0.0, 0.0)
    pre = meshing.preview_geometry(p, p.evaluator())
    pts, tris, dom = viewer._refine_tinted(pre.points, pre.triangles, pre.domain, h)
    assert len(tris) > len(pre.triangles) and len(dom) == len(tris)
    tri = pts[tris]
    on = viewer._on_condition(tri, key, pts.min(axis=0), pts.max(axis=0), 1e-6 * np.ptp(pts, axis=0).max())
    r = np.linalg.norm(tri[on].mean(axis=1)[:, :2], axis=1)
    assert r.min() < 0.5e-6                               # resolved: triangles well inside the FWHM
    poly = pv.PolyData(pts, np.hstack([np.full((len(tris), 1), 3), tris]).ravel())
    rgb = viewer._cell_colors(poly, dom, h)
    base = viewer._cell_colors(poly, dom, {})
    shift = np.linalg.norm(rgb - base, axis=1)[on]
    assert shift[r < 0.5e-6].min() > 2 * shift[r > 3e-6].max() > 0   # strong at the spot, a faint floor elsewhere
