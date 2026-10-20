"""Boundary selections of conditions (heat transfer, solid mechanics, any physics): the selection
fields of a condition (`on`, `faces`, `box`) and their resolution on a mesh.

A selection is the outer surface of the whole model or of one object (`on`: one of the objects, or all
domains of a material), restricted to faces of that object's bounding box (`faces`, or "all" for the whole
surface) and optionally to the faces lying inside a box (`box`). An object's outer
surface excludes its interfaces with other objects (e.g. the bottom of a pad bonded to a membrane).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .expressions import Evaluator
from .fem import surface


FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
ALL_FACES = "all"   # selection keyword: the whole outer surface of the model or of the object


def selection_to_dict(c, d: dict) -> dict:
    """Add a condition's selection (faces, object, box) to its table."""
    d["faces"] = list(c.faces)
    if c.on:
        d["on"] = c.on
    if c.box:
        d["box"] = {"min": list(c.box["min"]), "max": list(c.box["max"])}
    return d


def selection_text(c) -> str:
    """Short description of a condition's selection, e.g. 'Pad: zmax, in box'."""
    where = "all faces" if ALL_FACES in c.faces else ", ".join(c.faces) or "no faces"
    if c.on:
        where = f"{c.on}: {where}"
    if c.box:
        where += ", in box"
    return where


def check_selection(where: str, c, ev, objects) -> list[str]:
    """Problems with a condition's selection: object, faces, box."""
    errors = []
    if c.on and c.on not in objects:
        errors.append(f"{where}: unknown object '{c.on}' (an object or material name)")
    if not c.faces:
        errors.append(f"{where}: no faces selected (use bounding-box faces or \"all\")")
    for face in c.faces:
        if face not in FACES + (ALL_FACES,):
            errors.append(f"{where}: bad face '{face}' (use {', '.join(FACES)} or {ALL_FACES})")
    if c.box:
        try:
            lo, hi = ev.vector(c.box["min"]), ev.vector(c.box["max"])
            if len(lo) != 3 or len(hi) != 3:
                errors.append(f"{where}: box min and max need 3 coordinates")
        except KeyError:
            errors.append(f"{where}: box needs min and max")
        except Exception as exc:
            errors.append(f"{where}: box: {exc}")
    return errors


@dataclass
class Boundary:
    """Outer boundary triangles of a mesh and the element each belongs to."""
    tris: np.ndarray       # (F, 3|6) node indices
    owner: np.ndarray      # (F,) element index

    @classmethod
    def of(cls, mesh) -> "Boundary":
        tris, owner = surface.boundary_faces(mesh.cells, mesh.order, owners=True)
        return cls(tris, owner)


def tolerance(mesh) -> float:
    lo, hi = mesh.bounds()
    return 1e-6 * float(np.linalg.norm(hi - lo))


def select(mesh, boundary: Boundary, c, ev: Evaluator, exclude=()) -> np.ndarray:
    """Indices into boundary.tris of the triangles a condition (faces, on, box) selects. exclude: faces of the
    model's bounding box (symmetry planes) whose triangles no selection includes."""
    tol = tolerance(mesh)
    pts = mesh.points
    candidates = np.arange(len(boundary.tris))
    box = None
    if c.on:
        cells = mesh.region(c.on)
        candidates = candidates[cells[boundary.owner]]
        nodes = pts[np.unique(mesh.cells[cells])]
        box = (nodes.min(axis=0), nodes.max(axis=0))
    if not c.faces or len(candidates) == 0:
        return np.zeros(0, dtype=np.int64)
    within = (ev.vector(c.box["min"]), ev.vector(c.box["max"])) if c.box else None
    tris = boundary.tris[candidates]
    mask = surface.select(pts, tris, c.faces, tol, box, within)
    if exclude:
        mask &= ~surface.select(pts, tris, list(exclude), tol)
    return candidates[mask]


def key(c, ev: Evaluator):
    """What determines a selection (for caching): object, faces and the evaluated box."""
    box = tuple(np.round(np.r_[ev.vector(c.box["min"]), ev.vector(c.box["max"])], 15)) if c.box else ()
    return (c.on, tuple(c.faces), box)
