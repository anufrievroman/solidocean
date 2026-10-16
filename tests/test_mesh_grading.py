"""mesh.grading: how fast the element size may grow away from a region finer than max_size. Without it a
region's size holds inside the region and nowhere else, so a small feature's surroundings are resolved only
as well as gmsh happens to resolve them — and are resolved differently whenever the feature moves."""
import numpy as np
import pytest

from solidocean import meshing
from solidocean.project import Feature, Material, MeshRegion, MeshSettings, Project

MATERIAL = {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
            "k": "100[W/(m*K)]", "cp": "500[J/(kg*K)]"}

# each shell doubles the distance from the feature
SHELLS = np.array([0.1e-6, 0.2e-6, 0.4e-6, 0.8e-6, 1.6e-6, 3.2e-6, 6.4e-6])


def slab_with_a_small_spot(grading=1.0, at_x="0"):
    """A 20 x 20 x 2 um slab meshed at 4 um with a 0.1 um cube in it meshed at 50 nm: a small feature whose
    surroundings, not the feature itself, are what the grading has to resolve."""
    return Project(
        materials=[Material("M", dict(MATERIAL))],
        objects=[Feature("block", "Slab", material="M",
                         params={"center": ["0", "0", "0"], "size": ["20[um]", "20[um]", "2[um]"]}),
                 Feature("block", "Spot", material="M",
                         params={"center": [at_x, "0", "0"], "size": ["0.1[um]", "0.1[um]", "0.1[um]"]})],
        mesh=MeshSettings(max_size="4[um]", grading=grading,
                          regions=[MeshRegion(on="Spot", max_size="0.05[um]")]))


def element_sizes(project):
    """Median element size of the slab in each shell around the spot, and the mesh."""
    ev = project.evaluator()
    mesh = meshing.generate(project, ev)
    corners = mesh.points[mesh.cells[:, :4]]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    h = np.mean([np.linalg.norm(corners[:, a] - corners[:, b], axis=1) for a, b in pairs], axis=0)
    spot = np.array([ev(project.objects[1].params["center"][0]), 0.0, 0.0])
    r = np.linalg.norm(corners.mean(axis=1) - spot, axis=1)
    slab = mesh.region("Slab")
    sizes = []
    for lo, hi in zip(SHELLS, SHELLS[1:]):
        sel = slab & (r >= lo) & (r < hi)
        sizes.append(float(np.median(h[sel])) if sel.any() else np.nan)
    return np.array(sizes), mesh


def test_grading_refines_the_surroundings_of_the_fine_region():
    """Without grading the size leaves the feature behind at once and saturates at max_size; with grading it
    is held down over the whole neighbourhood."""
    plain, _ = element_sizes(slab_with_a_small_spot(1.0))
    graded, _ = element_sizes(slab_with_a_small_spot(1.3))
    assert np.all(graded < plain), f"graded {graded * 1e9} is not finer everywhere than {plain * 1e9}"
    assert graded[1] < 0.5 * plain[1]            # 200-400 nm out: the constriction a feature sits in


def test_grading_bounds_how_fast_the_size_grows():
    """The point of the knob: a bounded ratio between neighbouring elements. Each shell here doubles the
    distance, so a graded size may about double per shell, while an ungraded one jumps further."""
    plain, _ = element_sizes(slab_with_a_small_spot(1.0))
    graded, _ = element_sizes(slab_with_a_small_spot(1.3))

    def worst(sizes):
        return max(b / a for a, b in zip(sizes, sizes[1:]) if np.isfinite(a) and np.isfinite(b))

    assert worst(graded) < worst(plain)
    assert worst(graded) < 2.5, f"the size still jumps between shells: {graded * 1e9}"


def test_a_smaller_growth_ratio_costs_more_elements():
    """Grading is a cost knob: the smoother the mesh, the more elements."""
    counts = [element_sizes(slab_with_a_small_spot(g))[1].info["n_elements"] for g in (1.0, 1.5, 1.2)]
    plain, loose, tight = counts
    assert plain < loose < tight, counts


def test_grading_is_off_by_default():
    assert MeshSettings().grading == 1.0
    plain, _ = element_sizes(slab_with_a_small_spot())
    same, _ = element_sizes(slab_with_a_small_spot(1.0))
    assert plain == pytest.approx(same, nan_ok=True)


def test_grading_round_trips_and_belongs_to_the_mesh_key(tmp_path):
    p = slab_with_a_small_spot(grading=1.25)
    assert p.validate_geometry() == []
    assert p.mesh.to_dict()["grading"] == 1.25
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert q.mesh.grading == 1.25
    ev = q.evaluator()
    assert q.mesh.key(ev) != slab_with_a_small_spot(1.0).mesh.key(ev)    # so a change rebuilds the mesh


def test_grading_outside_its_range_is_an_error():
    p = slab_with_a_small_spot(grading=0.5)
    assert any("grading" in e for e in p.validate_geometry())
    p.mesh.grading = 9.0
    assert any("grading" in e for e in p.validate_geometry())


def test_grading_steadies_the_mesh_around_a_feature_that_moves():
    """The jitter a position sweep suffers: without grading the feature's surroundings are re-resolved from
    scratch at each position, so the mesh there changes with where it happens to sit. With grading the size
    around it follows the distance to it, so it is the same wherever it is."""
    offsets = ("0", "0.137[um]", "0.291[um]", "0.428[um]")   # nothing lines up with the background mesh
    plain = np.array([element_sizes(slab_with_a_small_spot(1.0, at_x=x))[0][:3] for x in offsets])
    graded = np.array([element_sizes(slab_with_a_small_spot(1.3, at_x=x))[0][:3] for x in offsets])
    spread_plain = np.nanmean(np.nanstd(plain, axis=0) / np.nanmean(plain, axis=0))
    spread_graded = np.nanmean(np.nanstd(graded, axis=0) / np.nanmean(graded, axis=0))
    assert spread_graded < spread_plain, \
        f"grading did not steady the mesh: spread {spread_graded:.3f} vs {spread_plain:.3f}"
