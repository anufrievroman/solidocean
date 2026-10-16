"""A mesh that cannot be solved is refused after meshing, naming the element — a degenerate tetrahedron
otherwise reaches the solver and comes back as a bare 'singular matrix' that says nothing about where it is."""
import numpy as np
import pytest

from solidocean import meshing, templates
from solidocean.fem import eigen
from solidocean.project import Feature, Material, MeshRegion, MeshSettings, Project

MATERIAL = {"model": "isotropic", "rho": "2000[kg/m^3]", "E": "100[GPa]", "nu": "0.3",
            "k": "100[W/(m*K)]", "cp": "500[J/(kg*K)]"}


def block(max_size="0.4[um]", regions=()):
    return Project(materials=[Material("M", dict(MATERIAL))],
                   objects=[Feature("block", "Slab", material="M",
                                    params={"center": ["0", "0", "0"],
                                            "size": ["4[um]", "2[um]", "1[um]"]})],
                   mesh=MeshSettings(max_size=max_size, regions=list(regions)))


def flatten_one(mesh, element=None):
    """Push one node of an element into the plane of the other three: a tet with no volume at all."""
    mesh.points = mesh.points.copy()
    e = int(np.argmin(mesh.quality())) if element is None else element
    nodes = mesh.cells[e, :4]
    corner = mesh.points[nodes]
    normal = np.cross(corner[1] - corner[0], corner[2] - corner[0])
    normal /= np.linalg.norm(normal)
    mesh.points[nodes[3]] -= normal * np.dot(mesh.points[nodes[3]] - corner[0], normal)
    return e


def test_an_ordinary_mesh_passes():
    p = block()
    mesh = meshing.generate(p, p.evaluator())        # generate() runs the check itself
    meshing.check_quality(mesh)
    assert mesh.quality().min() > meshing.DEGENERATE_QUALITY


def test_a_flat_element_is_not_rejected():
    """The check must not fire on elements that are merely flat: a region meshed much coarser than it is
    thin produces those by design, and they solve perfectly well. Only true degeneracy is an error."""
    p = Project(materials=[Material("M", dict(MATERIAL))],
                objects=[Feature("block", "Film", material="M",
                                 params={"center": ["0", "0", "0"],
                                         "size": ["40[um]", "40[um]", "0.2[um]"]})],
                mesh=MeshSettings(max_size="10[um]"))       # 50x coarser than the film is thick
    mesh = meshing.generate(p, p.evaluator())
    q = mesh.quality()
    assert (q < 0.05).any()                      # genuinely flat elements...
    assert q.min() > meshing.DEGENERATE_QUALITY  # ...but none of them degenerate
    meshing.check_quality(mesh)                  # so the mesh is accepted


def test_a_degenerate_element_is_refused_and_located():
    p = block()
    mesh = meshing.generate(p, p.evaluator())
    e = flatten_one(mesh)
    with pytest.raises(ValueError) as err:
        meshing.check_quality(mesh)
    msg = str(err.value)
    assert "unusable element" in msg
    assert f"element {e} " in msg                      # which one
    assert mesh.object_of(e) in msg                    # which object it is in
    assert "quality" in msg and "volume" in msg        # and how bad
    assert "Element quality by object" in msg          # with the context to judge it


def test_an_inverted_element_is_refused():
    """A tet wound inside out has negative volume; its quality alone would not give it away."""
    p = block()
    mesh = meshing.generate(p, p.evaluator())
    mesh.points = mesh.points.copy()
    e = 0
    nodes = mesh.cells[e, :4]
    corner = mesh.points[nodes]
    normal = np.cross(corner[1] - corner[0], corner[2] - corner[0])
    normal /= np.linalg.norm(normal)
    height = np.dot(mesh.points[nodes[3]] - corner[0], normal)
    mesh.points[nodes[3]] -= normal * 2.0 * height          # through the opposite face
    assert mesh.signed_volumes()[e] < 0          # wound inside out, not merely flat
    with pytest.raises(ValueError, match="inverted"):
        meshing.check_quality(mesh)


def test_describe_element_names_the_object_and_the_place():
    p = block()
    mesh = meshing.generate(p, p.evaluator())
    text = mesh.describe_element(0)
    assert "Slab" in text and "quality" in text and "edges" in text


def test_quality_by_object_separates_the_regions():
    """The split is what distinguishes a deliberately coarse region from a broken one."""
    p = block(max_size="1[um]", regions=[MeshRegion(on="Slab", max_size="0.3[um]")])
    mesh = meshing.generate(p, p.evaluator())
    report = mesh.quality_by_object()
    assert "Slab" in report and "elements" in report and "median" in report


def test_a_singular_matrix_is_one_error_type_whatever_the_backend():
    """So a caller that knows the mesh can add which element is to blame."""
    import scipy.sparse as sp
    singular = sp.csc_matrix(np.diag([1.0, 1.0, 0.0, 1.0]))     # a zero row: structurally singular
    for which in eigen.available_backends():
        with pytest.raises(eigen.SingularMatrix):
            eigen._factorize(singular, which)


def test_the_singular_matrix_help_reports_the_mesh():
    p = templates.create("heat_bar")
    mesh = meshing.generate(p, p.evaluator())
    text = meshing.singular_matrix_help(mesh)
    assert "flattest elements" in text and "Element quality by object" in text
    assert "physical reason" in text          # a sound mesh: it points away from the mesh

    flatten_one(mesh)
    assert "refine the object" in meshing.singular_matrix_help(mesh)   # a flat one: it points at the mesh
