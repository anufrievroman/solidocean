"""Mesh generation (gmsh) and extraction into numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from . import geometry
from .expressions import Evaluator
from .project import Project

# gmsh element type per order, nodes per element
TET_TYPES = {1: (4, 4), 2: (11, 10)}
TRI_TYPES = {1: 2, 2: 9}
# gmsh tet10 -> VTK quadratic tetra node order (gmsh edges 8/9 are 2-3/1-3, VTK has 1-3/2-3)
GMSH_TO_VTK_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]


@dataclass
class Mesh:
    points: np.ndarray            # (N, 3) SI coordinates
    cells: np.ndarray             # (E, 4|10) node indices, gmsh ordering
    cell_domain: np.ndarray       # (E,) index into material_names
    material_names: list[str]
    order: int
    length_scale: float = 1.0     # L0 used during geometry construction
    info: dict = field(default_factory=dict)
    cell_feature: np.ndarray | None = None   # (E,) index into feature_names (the object owning the cell)
    feature_names: list[str] = field(default_factory=list)

    def region(self, name: str) -> np.ndarray:
        """Boolean mask of the cells of an object or a material (objects take precedence)."""
        if name in self.feature_names and self.cell_feature is not None:
            return self.cell_feature == self.feature_names.index(name)
        if name in self.material_names:
            return self.cell_domain == self.material_names.index(name)
        known = ", ".join(self.feature_names + [m for m in self.material_names if m not in self.feature_names])
        raise ValueError(f"unknown object '{name}' (objects: {known})")

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_dofs(self) -> int:
        return 3 * len(self.points)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def quality(self) -> np.ndarray:
        """Normalised volume-to-edge ratio of the corner tets (1 = regular)."""
        p = self.points[self.cells[:, :4]]
        v = np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                             np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        l_rms = np.sqrt(np.mean([np.sum((p[:, a] - p[:, b]) ** 2, axis=1) for a, b in edges], axis=0))
        return 6 * np.sqrt(2) * v / l_rms**3

    def to_pyvista(self):
        import pyvista as pv
        if self.order == 2:
            cells, ctype = self.cells[:, GMSH_TO_VTK_TET10], pv.CellType.QUADRATIC_TETRA
        else:
            cells, ctype = self.cells, pv.CellType.TETRA
        n = cells.shape[1]
        flat = np.hstack([np.full((len(cells), 1), n), cells]).ravel()
        grid = pv.UnstructuredGrid(flat, np.full(len(cells), ctype, dtype=np.uint8), self.points)
        grid.cell_data["domain"] = self.cell_domain
        return grid


def _apply_mesh_options(project: Project, ev: Evaluator, L0: float) -> None:
    ms = project.mesh
    # one thread: multithreaded surface meshing gives a slightly different mesh on every run, so results
    # would not be reproducible (and sweeps would carry mesh noise); it is not slower for typical models
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", ev(ms.max_size) / L0)
    min_size = ev(ms.min_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_size / L0 if min_size > 0 else 0.0)  # reset between models
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.Optimize", 1 if ms.optimize else 0)


def _object_volumes(name: str) -> list[int]:
    """gmsh volumes of an object (what is left of it) or of all domains of a material."""
    for dim, tag in gmsh.model.getPhysicalGroups(3):
        if gmsh.model.getPhysicalName(dim, tag) == f"feature:{name}":
            return [int(v) for v in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)]
    for dim, tag in gmsh.model.getPhysicalGroups(3):
        if tag < geometry.FEATURE_TAG and gmsh.model.getPhysicalName(dim, tag) == name:
            return [int(v) for v in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)]
    return []


# sample points per direction on a surface the distance to a graded region is measured from; the distance
# is accurate to about the surface's extent over this, which only has to be well below the sizes it grades
GRADING_SAMPLING = 50


def _grading_fields(grading: float, by_size: dict[float, list[int]], h_max: float) -> list[int]:
    """Size fields that let the element size grow away from the regions finer than max_size, one per size:
    `h + (grading - 1) * distance to the region`, so that neighbouring elements differ by about `grading`.

    Without them a region's size holds inside it and nowhere else, so the size jumps back to max_size one
    element past its boundary. A small feature's surroundings are then never resolved, however fine the
    feature itself is — and they are resolved differently whenever it moves, so sweeping its position
    carries mesh noise into the results."""
    if grading <= 1:
        return []
    fields = []
    for h, vols in by_size.items():
        if h >= h_max:
            continue                    # the coarsest size: nothing finer to grade away from
        surfaces = {abs(int(s)) for _, s in gmsh.model.getBoundary([(3, v) for v in vols],
                                                                   oriented=False, combined=False)}
        if not surfaces:
            continue
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", sorted(surfaces))
        gmsh.model.mesh.field.setNumber(distance, "Sampling", GRADING_SAMPLING)
        ramp = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(ramp, "F", f"{h:.17g} + {grading - 1:.17g}*F{distance}")
        fields.append(ramp)
    return fields


def _apply_region_sizes(project: Project, ev: Evaluator, L0: float) -> None:
    """Element sizes per object (mesh regions): every volume gets the size of the last region containing it,
    else the mesh's max_size, through constant size fields restricted to the volumes (and their boundaries);
    the smallest applies on shared surfaces, so neighbours stay conforming. A region's size holds inside the
    region and nowhere else unless `grading` is above 1, which grows it outwards (see _grading_fields)."""
    regions = project.mesh.regions
    if not regions:
        return
    size = {int(v): ev(project.mesh.max_size) / L0 for _, v in gmsh.model.getEntities(3)}
    for r in regions:
        for v in _object_volumes(r.on):
            size[v] = ev(r.max_size) / L0
    by_size: dict[float, list[int]] = {}
    for v, h in size.items():
        by_size.setdefault(h, []).append(v)
    fields = []
    for h, vols in by_size.items():
        f = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.setNumbers(f, "VolumesList", sorted(vols))
        gmsh.model.mesh.field.setNumber(f, "VIn", h)
        gmsh.model.mesh.field.setNumber(f, "IncludeBoundary", 1)
        fields.append(f)
    fields += _grading_fields(project.mesh.grading, by_size, max(size.values()))
    smallest = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(smallest, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(smallest)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max(size.values()))


def generate(project: Project, ev: Evaluator, save_msh: str | Path | None = None) -> Mesh:
    """Build geometry and generate a volume mesh in one gmsh session."""
    with geometry.gmsh_session():
        L0, names = geometry.build(project, ev)
        _apply_mesh_options(project, ev, L0)
        _apply_region_sizes(project, ev, L0)
        try:
            gmsh.model.mesh.generate(3)
        except Exception as exc:
            raise ValueError(f"meshing failed: {exc}") from exc
        order = project.mesh.order
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        if save_msh:
            gmsh.write(str(save_msh))
        return _extract(names, order, L0)


def _extract(names: list[str], order: int, L0: float) -> Mesh:
    etype, nn = TET_TYPES[order]
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3) * L0
    lookup = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    lookup[node_tags] = np.arange(len(node_tags))

    feature_of, feature_names = {}, []
    for dim, tag in gmsh.model.getPhysicalGroups(3):
        if tag >= geometry.FEATURE_TAG:
            feature_names.append(gmsh.model.getPhysicalName(dim, tag).removeprefix("feature:"))
            for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
                feature_of[ent] = len(feature_names) - 1
    cells, domain, feature = [], [], []
    for i, _name in enumerate(names):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(3, i + 1):
            _, enodes = gmsh.model.mesh.getElementsByType(etype, ent)
            c = lookup[enodes.astype(np.int64)].reshape(-1, nn)
            cells.append(c)
            domain.append(np.full(len(c), i, dtype=np.int32))
            feature.append(np.full(len(c), feature_of.get(ent, -1), dtype=np.int32))
    cells = np.vstack(cells)
    domain = np.concatenate(domain)

    used = np.unique(cells)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(points=coords[used], cells=remap[cells], cell_domain=domain,
                material_names=names, order=order, length_scale=L0,
                info={"n_elements": len(cells), "n_nodes": len(used)},
                cell_feature=np.concatenate(feature), feature_names=feature_names)


@dataclass
class SurfacePreview:
    points: np.ndarray       # (N, 3) SI
    triangles: np.ndarray    # (T, 3)
    domain: np.ndarray       # (T,) material index per triangle
    material_names: list[str]


def preview_geometry(project: Project, ev: Evaluator) -> SurfacePreview:
    """Coarse surface triangulation of the geometry for display."""
    with geometry.gmsh_session():
        L0, names = geometry.build(project, ev)
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 25)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 16)
        gmsh.model.mesh.generate(2)

        vol_material = {}
        for i in range(len(names)):
            for v in gmsh.model.getEntitiesForPhysicalGroup(3, i + 1):
                vol_material[v] = i
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        lookup = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
        lookup[node_tags] = np.arange(len(node_tags))
        tris, dom = [], []
        for _, s in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, s)
            m = max((vol_material.get(int(v), -1) for v in up), default=-1)
            _, enodes = gmsh.model.mesh.getElementsByType(2, s)
            t = lookup[enodes.astype(np.int64)].reshape(-1, 3)
            tris.append(t)
            dom.append(np.full(len(t), m, dtype=np.int32))
        return SurfacePreview(coords.reshape(-1, 3) * L0, np.vstack(tris), np.concatenate(dom), names)
