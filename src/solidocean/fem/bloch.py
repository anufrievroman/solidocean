"""Bloch-Floquet periodicity and boundary constraints via a reduction matrix T.

With u_full = T u_red, the constrained problem is (T^H K T) u = w^2 (T^H M T) u.
For a node on a 'max' face, u(x + R) = exp(i k.R) u(x): its DOFs map onto the
DOFs of its image on the 'min' face (chained for edges and corners).

Lattice translations may point in any direction (hexagonal, oblique lattices):
a node is paired with the node at its position minus a lattice vector.

Boundary conditions on bounding-box faces remove individual DOFs:
fixed (all components), symmetry (normal component), antisymmetry
(tangential components).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

AXES = ("x", "y", "z")


@dataclass
class Constraints:
    root: np.ndarray        # (N,) independent node each node maps to
    shift: np.ndarray       # (N, n_vec) lattice-vector multiples between node and its root
    vectors: np.ndarray     # (n_vec, 3) lattice translation vectors (m)
    dof_index: np.ndarray   # (N, 3) reduced DOF index of each root DOF (-1 if constrained)
    n_free: int             # number of reduced DOFs

    def T(self, k: np.ndarray | None = None) -> sp.csr_matrix:
        """Reduction matrix (3N x n_free); complex unless all phases are 1."""
        k = np.zeros(3) if k is None else np.asarray(k, dtype=float)
        N = len(self.root)
        phase_arg = self.shift @ (self.vectors @ k) if len(self.vectors) else np.zeros(N)
        if np.allclose(np.mod(phase_arg + np.pi, 2 * np.pi) - np.pi, 0, atol=1e-12):
            phase = np.ones(N)
        else:
            phase = np.exp(1j * phase_arg)
        cols = self.dof_index[self.root]                      # (N, 3)
        keep = cols >= 0
        rows = 3 * np.arange(N)[:, None] + np.arange(3)
        vals = np.broadcast_to(phase[:, None], (N, 3))
        return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(3 * N, self.n_free))


def face_nodes(points: np.ndarray, face: str, tol: float) -> np.ndarray:
    """Boolean mask of nodes on a bounding-box face such as 'zmin'."""
    i = AXES.index(face[0])
    plane = points[:, i].min() if face.endswith("min") else points[:, i].max()
    return np.abs(points[:, i] - plane) < tol


def build_constraints(points: np.ndarray, periodic, fixed_faces=(),
                      symmetry_faces=(), antisymmetry_faces=(), rtol: float = 1e-6,
                      constrained_dofs: np.ndarray | None = None) -> Constraints:
    """periodic: {axis: period} (rectangular cell) or [(name, vector), ...] (any lattice);
    constrained_dofs: optional (N, 3) mask of further constrained components (fixed / prescribed)."""
    N = len(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    tol = rtol * float(np.max(hi - lo))
    box_cell = isinstance(periodic, dict)  # axis shorthand: the cell is the bounding box
    if box_cell:
        trans = []
        for ax, L in periodic.items():
            v = np.zeros(3)
            v[AXES.index(ax)] = L
            trans.append((ax, v))
    else:
        trans = [(name, np.asarray(v, dtype=float)) for name, v in periodic]
    vectors = np.array([v for _, v in trans]).reshape(-1, 3)
    partners = []
    tree = cKDTree(points)
    for n, (name, vec) in enumerate(trans):
        # nodes whose image under -vec is also a node: the 'far' side of the cell for this vector
        dist, idx = tree.query(points - vec)
        far = np.nonzero(dist < tol)[0]
        axis = [i for i in range(3) if abs(vec[i]) > 0]
        if box_cell and len(axis) == 1:  # box cell: every node on the far plane must have a partner
            i = axis[0]
            plane = np.nonzero(np.abs(points[:, i] - (lo[i] + abs(vec[i]) if vec[i] > 0 else lo[i])) < tol)[0]
            if vec[i] > 0 and (len(plane) == 0 or len(np.setdiff1d(plane, far))):
                raise ValueError(f"mesh is not periodic along {name} (period {abs(vec[i]):g} m)")
        if len(far) == 0:
            raise ValueError(f"mesh is not periodic along {name}")
        partner = np.full(N, -1, dtype=np.int64)
        partner[far] = idx[far]
        partners.append((n, partner))

    root = np.arange(N)
    shift = np.zeros((N, len(trans)), dtype=np.int64)
    for _ in range(8):
        changed = False
        for i, partner in partners:
            mask = partner[root] >= 0
            if mask.any():
                changed = True
                shift[mask, i] += 1
                root[mask] = partner[root[mask]]
        if not changed:
            break

    constrained = np.zeros((N, 3), dtype=bool) if constrained_dofs is None else constrained_dofs.copy()
    for face in fixed_faces:
        constrained[face_nodes(points, face, tol)] = True
    for face in symmetry_faces:
        constrained[face_nodes(points, face, tol), AXES.index(face[0])] = True
    for face in antisymmetry_faces:
        tangential = [c for c in range(3) if c != AXES.index(face[0])]
        constrained[np.ix_(face_nodes(points, face, tol), tangential)] = True
    # a constraint on any periodic image applies to the shared root DOF
    root_constrained = np.zeros((N, 3), dtype=bool)
    np.logical_or.at(root_constrained, root, constrained)

    is_root = np.zeros(N, dtype=bool)
    is_root[root] = True
    free = is_root[:, None] & ~root_constrained
    dof_index = np.full((N, 3), -1, dtype=np.int64)
    dof_index[free] = np.arange(int(free.sum()))
    return Constraints(root, shift, vectors, dof_index, int(free.sum()))
