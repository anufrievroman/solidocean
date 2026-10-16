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


# ---- per-region grading: the element budget ----------------------------------------------------------

def slab_with_two_spots(mesh_grading=1.0, fine_grading=None, coarse_grading=None):
    """A slab with a small spot on the left and a much bigger block on the right, each its own region: the
    case the per-region knob is for, where only the small one's surroundings are worth resolving."""
    p = Project(
        materials=[Material("M", dict(MATERIAL))],
        objects=[Feature("block", "Slab", material="M",
                         params={"center": ["0", "0", "0"], "size": ["20[um]", "20[um]", "2[um]"]}),
                 Feature("block", "Spot", material="M",
                         params={"center": ["-5[um]", "0", "0"], "size": ["0.1[um]", "0.1[um]", "0.1[um]"]}),
                 Feature("block", "Chunk", material="M",
                         params={"center": ["5[um]", "0", "0"], "size": ["3[um]", "3[um]", "1[um]"]})],
        mesh=MeshSettings(max_size="4[um]", grading=mesh_grading,
                          regions=[MeshRegion(on="Spot", max_size="0.05[um]", grading=fine_grading),
                                   MeshRegion(on="Chunk", max_size="1[um]", grading=coarse_grading)]))
    return p


def sizes_around(project, centre, radius=0.4e-6, inner=0.0):
    """Median slab element size in the shell `inner` < r < `radius` around a point (the shell matters: inside
    a region the size is its own whatever the grading, so only outside it shows), and the element count."""
    mesh = meshing.generate(project, project.evaluator())
    corners = mesh.points[mesh.cells[:, :4]]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    h = np.mean([np.linalg.norm(corners[:, a] - corners[:, b], axis=1) for a, b in pairs], axis=0)
    r = np.linalg.norm(corners.mean(axis=1) - np.asarray(centre), axis=1)
    sel = mesh.region("Slab") & (r < radius) & (r >= inner)
    return (float(np.median(h[sel])) if sel.any() else np.nan), mesh.info["n_elements"]


SPOT = (-5e-6, 0, 0)
CHUNK = (5e-6, 0, 0)


def test_a_region_without_a_grading_follows_the_mesh():
    p = slab_with_two_spots(mesh_grading=1.3)
    assert p.mesh.regions[0].grading is None
    assert p.mesh.grading_of(p.mesh.regions[0]) == 1.3
    assert p.mesh.grading_of(MeshRegion(on="Spot", grading=1.1)) == 1.1


def test_grading_one_region_leaves_the_other_alone():
    """The budget: grade around the small feature only, and the big region's surroundings stay coarse."""
    both = slab_with_two_spots(mesh_grading=1.3)                    # everything grades
    only_spot = slab_with_two_spots(mesh_grading=1.3, coarse_grading=1.0)   # the big one opts out
    h_spot_a, n_both = sizes_around(both, SPOT)
    h_spot_b, n_spot = sizes_around(only_spot, SPOT)
    # just outside the 3 um chunk, where its grading (not its own size) is what sets the slab
    h_chunk_a, _ = sizes_around(both, CHUNK, inner=2e-6, radius=3.5e-6)
    h_chunk_b, _ = sizes_around(only_spot, CHUNK, inner=2e-6, radius=3.5e-6)
    assert h_spot_b == pytest.approx(h_spot_a, rel=0.35)            # the spot is resolved either way
    assert h_chunk_b > 1.3 * h_chunk_a                              # but the chunk no longer pays
    assert n_spot < n_both                                          # which is the saving


def test_a_region_can_grade_tighter_than_the_mesh():
    loose, _ = sizes_around(slab_with_two_spots(mesh_grading=1.5), SPOT)
    tight, _ = sizes_around(slab_with_two_spots(mesh_grading=1.5, fine_grading=1.15), SPOT)
    assert tight < loose


def test_a_region_can_grade_while_the_mesh_does_not():
    """Grading only one region, with the mesh's own grading off: the cheapest useful configuration."""
    off, n_off = sizes_around(slab_with_two_spots(mesh_grading=1.0), SPOT)
    spot_only, n_spot = sizes_around(slab_with_two_spots(mesh_grading=1.0, fine_grading=1.2), SPOT)
    assert spot_only < 0.6 * off and n_spot > n_off
    everything, n_all = sizes_around(slab_with_two_spots(mesh_grading=1.2), SPOT)
    assert n_spot < n_all                                           # cheaper than grading the whole model
    assert spot_only == pytest.approx(everything, rel=0.35)         # with the same mesh where it matters


def test_per_region_grading_round_trips_and_is_part_of_the_mesh_key(tmp_path):
    p = slab_with_two_spots(mesh_grading=1.3, fine_grading=1.15, coarse_grading=1.0)
    assert p.validate_geometry() == []
    assert p.mesh.regions[0].to_dict()["grading"] == 1.15
    assert "grading" not in slab_with_two_spots().mesh.regions[0].to_dict()   # omitted when not set
    q = Project.load(p.save(tmp_path / "m.toml"))
    assert [r.grading for r in q.mesh.regions] == [1.15, 1.0]
    ev = q.evaluator()
    assert q.mesh.key(ev) != slab_with_two_spots(mesh_grading=1.3).mesh.key(ev)


def test_a_region_grading_outside_its_range_is_an_error():
    p = slab_with_two_spots(fine_grading=0.5)
    assert any("grading" in e and "Spot" in e for e in p.validate_geometry())
    p.mesh.regions[0].grading = 9.0
    assert any("grading" in e and "Spot" in e for e in p.validate_geometry())
