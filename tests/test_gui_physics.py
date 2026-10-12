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
    fluxes = [k for k, v in h.items() if v == "flux"]                  # the tops of the pad and the ribbon
    assert len(fluxes) == 2 and all(face == "zmax" and obj is not None and box is None for face, obj, box in fluxes)
    kinds = [node[0] for _, node, _ in ui.physics_nodes(p)]
    assert kinds == ["heat_initial", "heat_cond", "heat_cond", "heat_cond"]


def test_invalid_variables_give_no_highlights_instead_of_an_error():
    p = templates.create("cantilever")
    p.variables["L"] = "1/("
    assert gui_physics.of(p).highlights(p) == {}
