"""Thermal contact at interfaces (`[[heat.interfaces]]`): a boundary resistance R or a thin resistive layer.

The mesh is conforming across interfaces; their nodes are duplicated (fem.surface.split_nodes) so the
temperature may jump, and the two sides are coupled by the interface flux q = (T1 - T2) / R:
    K += int (T1 - T2)(v1 - v2) / R dA.
A thin layer of material (thickness d, not meshed) has R = d / k_n (k_n = n . k . n across the layer) and
adds its heat capacity rho cp d, half to each side. R, k and rho cp may depend on T, taken as the mean of
both sides at each quadrature point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ...fem import surface
from ...materials import REFERENCE_T, depends_on_T
from .properties import CHECK_T, Material, _Value


@dataclass
class Split:
    """A mesh with the interface nodes duplicated and the faces of every interface."""
    mesh: object
    tris: np.ndarray            # outer boundary triangles (renumbered)
    owner: np.ndarray           # element of each
    faces: list                 # per interface: (tri1, tri2), the same positions on both sides


def selection_key(project) -> tuple:
    return tuple(tuple(c.between) for c in project.heat.interfaces)


def interface_mask(mesh, c, faces) -> tuple[np.ndarray, np.ndarray]:
    """Interior faces of an interface (mask) and whether to swap sides so the first is `between[0]`."""
    _, e1, e2, _, _ = faces
    if not c.between:
        d = mesh.cell_domain
        return d[e1] != d[e2], np.zeros(len(e1), dtype=bool)
    a = mesh.region(c.between[0])
    b = mesh.region(c.between[1]) if len(c.between) == 2 else ~a
    fwd, back = a[e1] & b[e2], a[e2] & b[e1]
    return fwd | back, back


def split(mesh, interfaces, tris, owner) -> Split:
    """Duplicate the nodes of every interface (one copy per side); boundary triangles follow their cells."""
    from dataclasses import replace
    faces = surface.interior_faces(mesh.cells, mesh.order)
    masks, swaps = [], []
    used = np.zeros(len(faces[1]), dtype=bool)
    for c in interfaces:
        mask, swap = interface_mask(mesh, c, faces)
        name = c.name or (" | ".join(c.between) if c.between else "material interfaces")
        if not mask.any():
            raise ValueError(f"heat interface '{name}': the objects share no faces")
        if (mask & used).any():
            raise ValueError(f"heat interface '{name}': overlaps another interface")
        used |= mask
        masks.append(mask)
        swaps.append(swap)
    cells, source, (t1, t2) = surface.split_nodes(mesh.cells, mesh.order, used, faces)
    points = np.vstack([mesh.points, mesh.points[source]])
    new = replace(mesh, points=points, cells=cells, info={**mesh.info, "n_nodes": len(points)})
    index = np.flatnonzero(used)
    out = []
    for mask, swap in zip(masks, swaps):
        sel = np.searchsorted(index, np.flatnonzero(mask))
        a, b = t1[sel], t2[sel]
        s = swap[mask]
        out.append((np.where(s[:, None], b, a), np.where(s[:, None], a, b)))
    return Split(new, surface.renumber_faces(tris, owner, mesh.cells, cells), owner, out)


class _Contact:
    def __init__(self, c, project, ev, mesh, tri1, tri2):
        self.tri1, self.tri2, self.mesh = tri1, tri2, mesh
        where = f"heat interface '{c.name or ' | '.join(c.between) or 'material interfaces'}'"
        self.where = where
        if c.type == "layer":
            self.layer = Material(c.material, project.material(c.material).props, ev)
            self.d = ev(c.thickness)
            self.normals = surface.face_normals(mesh.points, tri1)
            self.k_varies, self.c_varies = self.layer.k_varies, self.layer.c_varies
        else:
            self.layer = None
            self.R = _Value(c.R, ev, f"{where}, R")
            self.k_varies, self.c_varies = self.R.varies, False

    def at_quadrature(self, T) -> np.ndarray:
        """Mean temperature of both sides at the surface quadrature points (F, Q)."""
        pts, o = self.mesh.points, self.mesh.order
        if T is None or np.ndim(T) == 0:
            return np.full(surface.surface_at_points(pts, self.tri1, o, np.zeros(len(pts))).shape,
                           REFERENCE_T if T is None else float(T))
        return 0.5 * (surface.surface_at_points(pts, self.tri1, o, T) + surface.surface_at_points(pts, self.tri2, o, T))

    def conductance(self, Tq) -> np.ndarray:
        """1 / R (W/(m^2 K)) at the quadrature points."""
        if self.layer is None:
            R = self.R(Tq)
            if np.any(R <= 0):
                raise ValueError(f"{self.where}: R not positive at T = {Tq[R <= 0].flat[0]:.6g} K")
            return 1 / R
        k = self.layer.conductivity(Tq)                                   # (F, Q, 3, 3)
        return np.einsum("fi,fqij,fj->fq", self.normals, k, self.normals) / self.d

    def matrices(self, T, capacity: bool):
        pts, o, n = self.mesh.points, self.mesh.order, self.mesh.n_nodes
        Tq = self.at_quadrature(T)
        K = surface.interface_terms(pts, self.tri1, self.tri2, o, self.conductance(Tq), n)
        C = None
        if capacity and self.layer is not None:
            half = 0.5 * self.d * self.layer.capacity(Tq)
            C = surface.surface_terms(pts, self.tri1, o, coeff=half, n_nodes=n)[0] + \
                surface.surface_terms(pts, self.tri2, o, coeff=half, n_nodes=n)[0]
        return K, C

    def key(self):
        T = np.array(CHECK_T)
        if self.layer is None:
            return ("R", np.round(self.R(T), 20).tobytes())
        m = self.layer
        return ("layer", self.d, [np.round(v(T), 12).tobytes() for v in (*m.k, m.rho, m.cp)],
                None if m.R is None else m.R.round(12).tobytes())


class Contacts:
    """The thermal contacts of a split mesh: conductance and layer capacity matrices."""

    def __init__(self, project, ev, split: Split | None):
        self.items = [] if split is None else [
            _Contact(c, project, ev, split.mesh, t1, t2) for c, (t1, t2) in zip(project.heat.interfaces, split.faces)]

    @property
    def k_varies(self) -> bool:
        return any(c.k_varies for c in self.items)

    @property
    def c_varies(self) -> bool:
        return any(c.c_varies for c in self.items)

    @property
    def varies(self) -> bool:
        return self.k_varies or self.c_varies

    def key(self) -> str:
        return repr([c.key() for c in self.items])

    def matrices(self, n: int, T=None, capacity: bool = True):
        """(K, C) of all contacts with the properties at temperatures T (None: REFERENCE_T)."""
        K, C = sp.csr_matrix((n, n)), sp.csr_matrix((n, n))
        for c in self.items:
            Kc, Cc = c.matrices(T, capacity)
            K = K + Kc
            if Cc is not None:
                C = C + Cc
        return K.tocsr(), (C.tocsr() if capacity else None)


def varies(project) -> bool:
    """True if a contact depends on T (R using T, or a layer of a temperature-dependent material)."""
    from ... import materials as mat
    for c in project.heat.interfaces:
        if c.type == "layer":
            try:
                _, p = mat.resolve_properties(project.material(c.material).props)
            except KeyError:
                continue
            if any(depends_on_T(p.get(key, "")) for key in (*mat.THERMAL_KEYS, "rho")):
                return True
        elif depends_on_T(c.R):
            return True
    return False

